"""Six hidden edge Linears: unchanged forward, explicit setup-tuned input VJP."""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_hidden_vjp import (
    PASS, CONTROL, LibraryContext, TunedInputVJP, HiddenVJPSetupError, HiddenVJPRuntimeError, validate_vjp,
)
from md_benchmark.opt4_registry import record


class _HiddenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tuner):
        ctx.save_for_backward(x, weight)
        ctx.has_bias, ctx.tuner = bias is not None, tuner
        return F.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            if isinstance(grad, FakeTensor):
                dx = x.new_empty(x.shape)
            else:
                tuner = ctx.tuner
                if not tuner.ready:
                    tuner.prepare(grad, weight)
                dx = tuner.execute(grad, weight)
                # Eager callers may retain gradients over subsequent calls.
                # Production Graph has one call/module and fixed private buffers.
                if not torch.cuda.is_current_stream_capturing():
                    dx = dx.clone()
        dw = grad.T @ x if ctx.needs_input_grad[1] else None
        db = grad.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


class LinearReference(nn.Module):
    def forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


class TunedLinear(nn.Module):
    def __init__(self, tuner):
        super().__init__()
        self.tuner = tuner

    def forward(self, x, weight, bias):
        if not isinstance(x, FakeTensor) and not x.is_cuda:
            raise HiddenVJPSetupError("hidden Linear VJP requires CUDA; no CPU fallback")
        return _HiddenLinear.apply(x, weight, bias, self.tuner)


class HiddenRegion(CheckedRegion):
    def __init__(self, detail, tuner):
        self.tuner = tuner
        @contextmanager
        def audit():
            tuner.audit = True
            try:
                yield
            finally:
                tuner.audit = False
        def derivative(actual, expected, args, index, probes):
            validate_vjp(actual, expected)
            return True
        super().__init__(LinearReference(), detail, candidate=TunedLinear(tuner),
            vjp_validator=derivative, validation_context=audit, validate_runtime_vjp=True)

    def forward(self, *args):
        if isinstance(args[0], FakeTensor):
            return self.compiled(*args)
        if not self.tuner.ready:
            if args[0].is_cuda and torch.cuda.is_current_stream_capturing():
                raise HiddenVJPRuntimeError("hidden VJP has not observed its real force adjoint before capture")
            # First full-checkpoint evaluation gathers actual adjoints. No
            # synthetic CheckedRegion probe is allowed to select the algorithm.
            return self.compiled(*args)
        return super().forward(*args)

    def _validate(self, args):
        super()._validate(args)
        # Keep a real adjoint only for the diagnostic/NCU stage. Production
        # never consumes this tensor; free it after full-boundary validation.
        if not self.detail.get("retain_diagnostic_probe"):
            self.tuner.real_probe = None

    def _benchmark(self, args):
        # CheckedRegion wraps all of _validate in the audit context, including
        # this benchmark. Its leaves share the real frozen parameters' storage;
        # use the prepared production pack here, not the setup-only audit path.
        previous = self.tuner.audit
        self.tuner.audit = False
        try:
            return super()._benchmark(args)
        finally:
            self.tuner.audit = previous


class HiddenEdgeLinear(nn.Module):
    def __init__(self, original, detail, context):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self._opt4_hidden_linear = HiddenRegion(detail, TunedInputVJP(context, detail))
        self.train(original.training)

    def forward(self, x):
        return self._opt4_hidden_linear(x, self.weight, self.bias)


def targets(model):
    stacks = [(path, module) for path, module in model.named_modules()
              if isinstance(getattr(module, "gnn_stacks", None), nn.ModuleList)]
    if len(stacks) != 1 or len(stacks[0][1].gnn_stacks) != 5:
        raise HiddenVJPSetupError("hidden VJP requires one five-block ORB processor")
    prefix, core = stacks[0]
    paths = ["_encoder._edge_fn.mlp"] + [f"gnn_stacks.{i}._edge_mlp.mlp" for i in range(5)]
    result = []
    for path in paths:
        try:
            parent = core.get_submodule(path)
            layer = parent.get_submodule("NN-1")
        except AttributeError as exc:
            raise HiddenVJPSetupError(f"hidden VJP missing {path}.NN-1") from exc
        if (not isinstance(parent, nn.Sequential) or type(layer) is not nn.Linear
                or layer.in_features != 1024 or layer.out_features != 1024):
            raise HiddenVJPSetupError(f"hidden VJP only supports native Linear(1024,1024): {path}.NN-1")
        full_path = (prefix + "." if prefix else "") + path + ".NN-1"
        result.append((full_path, parent, layer))
    return result


def install_hidden(model, report, options):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise HiddenVJPSetupError("hidden VJP requires an eval model with frozen parameters")
    matched = targets(model)  # Fail before mutating any part of an unsupported model.
    from .opt4_first_edge_vjp import install_first_edge
    retained = {"passes": {}, "benchmark_boundaries": False}
    install_first_edge(model, CONTROL, retained)
    context = LibraryContext(options.get("opt4_hidden_vjp_setup_dir"))
    modules = []
    for path, parent, layer in matched:
        detail = dict(module=path, boundary="hidden-linear-1024-input-vjp", calls_per_force=1,
            benchmark_requested=bool(report.get("benchmark_boundaries")), validated_shapes=0,
            retain_diagnostic_probe=bool(options.get("opt4_hidden_vjp_export")),
            reference=CONTROL, forward="unchanged-native-F.linear", new_fused_kernels=0)
        parent._modules["NN-1"] = HiddenEdgeLinear(layer, detail, context)
        modules.append({"module": path, "regions": [detail]})
    record(report, PASS, 6, "setup-tuned-native-or-cublaslt-input-vjp", modules=modules,
        retained_base=CONTROL, retained_base_details=retained["passes"][CONTROL],
        new_fused_kernels=0, forward="unchanged", replay_weight_pack=False,
        runtime_autotune=False, checkpoint_changed=False, precision="FP32/FP64-no-TF32",
        training_double_backward=False)


def release_hidden(model):
    """Called only after the old production Graph has been released."""
    for module in model.modules():
        region = getattr(module, "_opt4_hidden_linear", None)
        if isinstance(region, HiddenRegion):
            region.signatures.clear()
            region.tuner.reset()


def export_hidden_probes(model, output):
    """Setup-only operands/selection for targeted NCU, never a runtime cache."""
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    for module in model.modules():
        region = getattr(module, "_opt4_hidden_linear", None)
        if not isinstance(region, HiddenRegion):
            continue
        tuner = region.tuner
        if not tuner.ready or tuner.real_probe is None:
            raise RuntimeError("hidden VJP real adjoint export incomplete")
        torch.save(dict(schema=1, module=region.detail["module"],
            grad=tuner.real_probe.cpu(), weight=module.weight.detach().cpu(),
            grad_stride=list(tuner.real_probe.stride()), selection=tuner.selection,
            library=tuner.context.metadata, generation=tuner.generation), output / f"hidden-{count}.pt")
        tuner.real_probe = None
        count += 1
    if count != 6:
        raise RuntimeError(f"hidden VJP exported {count} modules, expected 6")
