"""First-block-only edge input VJP; two explicit layout ablations.

The preprojection-v1 forward and every later block stay unchanged. No GEMM
implementation, precision or edge-scatter order is changed. Node/parameter
derivatives remain available for exhaustive setup audits; production requires
the checkpoint's first-block nodes and parameters to be independent of geometry.
"""
from __future__ import annotations

import weakref
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from .opt4_edge_preproject import validate_inputs, linear_vjp, EdgeLinearPreproject

PASSES = {"orb_first_edge_vjp_native": "native", "orb_first_edge_vjp_packed": "packed"}


def capturing(weight):
    return weight.is_cuda and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class EdgeWeightPack(nn.Module):
    """Only C x H, not 3C x H or any subsequent Linear. Setup-only cache."""
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.register_buffer("edge", None, persistent=False)
        self._source = None
        self._version_seen = None
        self.builds = 0

    def _pack(self, weight):
        if isinstance(weight, FakeTensor):
            return weight.new_empty((self.width, weight.shape[0]))
        return weight.detach()[:, :self.width].T.contiguous()

    def forward(self, weight):
        if isinstance(weight, FakeTensor):
            return self._pack(weight)
        # Gradcheck perturbs data without necessarily incrementing _version.
        if weight.requires_grad:
            if capturing(weight):
                raise RuntimeError("first-edge parameter audits must finish before capture")
            return self._pack(weight)
        if (self.edge is None or self._source is None or self._source() is not weight
                or self._version_seen != weight._version
                or self.edge.device != weight.device or self.edge.dtype != weight.dtype):
            if capturing(weight):
                raise RuntimeError("first-edge weights not prepared before capture")
            self.edge = self._pack(weight)
            self._source, self._version_seen = weakref.ref(weight), weight._version
            self.builds += 1
        return self.edge

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self._source, self._version_seen = None, None
        return result


def first_edge_input_vjp(nodes, edges, senders, receivers, weight, dz, needs, packed=None):
    if needs[0]:
        # Exhaustive node/parameter audits preserve v1's native edge-GEMM ->
        # two independent index_put branches. Never detach a requested VJP.
        return linear_vjp(nodes, edges, senders, receivers, weight, dz, needs)
    de = None
    if needs[1]:
        de = dz @ weight[:, :nodes.shape[1]] if packed is None else F.linear(dz, packed)
    _, _, _, _, dw, db = linear_vjp(nodes, edges, senders, receivers, weight, dz,
                                   (False, False, False, False, needs[4], needs[5]))
    return None, de, None, None, dw, db


class _FirstEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, s, r, weight, bias, packed):
        validate_inputs(nodes, edges, s, r, weight, bias)
        if not nodes.is_cuda:
            raise RuntimeError("ORB first-edge candidate requires CUDA; no eager fallback")
        if isinstance(nodes, FakeTensor):
            out = edges.new_empty((edges.shape[0], weight.shape[0]))
            z = torch.empty_like(out)
        else:
            from .opt4_edge_preproject_kernels import gather_silu
            we, ws, wr = weight.split(nodes.shape[1], dim=1)
            pe, ps, pr = F.linear(edges, we), F.linear(nodes, ws), F.linear(nodes, wr)
            out, z = gather_silu(pe, ps, pr, s, r, bias)
        ctx.save_for_backward(nodes, edges, s, r, weight, z, packed)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        n, e, s, r, w, z, packed = ctx.saved_tensors
        dz = torch.ops.aten.silu_backward.default(upstream, z)
        return (*first_edge_input_vjp(n, e, s, r, w, dz, ctx.needs_input_grad[:6], packed), None)


class FirstEdgePreproject(nn.Module):
    def __init__(self, width, layout):
        super().__init__()
        if layout not in PASSES.values():
            raise ValueError("first-edge layout must be native or packed")
        self.layout = layout
        self.pack = EdgeWeightPack(width) if layout == "packed" else None

    def forward(self, n, e, s, r, w, b):
        validate_inputs(n, e, s, r, w, b)
        packed = self.pack(w) if self.pack is not None else None
        return _FirstEdge.apply(n, e, s, r, w, b, packed)


def install_first_edge(model, name, report):
    from md_benchmark.opt4_fx import CheckedRegion, assert_float32_vjp_reassociation_close
    from md_benchmark.opt4_registry import FusionSetupError, record
    from .opt4_fusion import _install_preproject

    # Match an actual ordered processor stack, not named_modules traversal order.
    stacks = [(path, module.gnn_stacks) for path, module in model.named_modules()
              if isinstance(getattr(module, "gnn_stacks", None), nn.ModuleList)]
    if len(stacks) != 1 or not stacks[0][1] or any(
            type(block).__name__ != "AttentionInteractionNetwork" for block in stacks[0][1]):
        raise FusionSetupError("first-edge VJP requires one ordered ORB gnn_stacks ModuleList")
    path, stack = stacks[0]
    all_blocks = [m for m in model.modules() if type(m).__name__ == "AttentionInteractionNetwork"]
    if len(all_blocks) != len(stack):
        raise FusionSetupError("first-edge VJP found interaction blocks outside the ordered stack")
    if any(p.requires_grad for p in model.parameters()):
        raise FusionSetupError("first-edge MD requires frozen model parameters")

    retained = {"passes": {}, "benchmark_boundaries": False}
    _install_preproject(model, retained)
    first = stack[0]
    base = first._opt4_edge_preproject
    width = first.latent_dim
    detail = {"module": (path + "." if path else "") + "gnn_stacks.0",
              "boundary": "preproject-v1-first-edge-only-vjp", "validated_shapes": 0,
              "layout": PASSES[name], "benchmark_requested": report.get("benchmark_boundaries", False),
              "boundary_gate_max_ratio": {"forward": 1.05, "forward_vjp": .85},
              "reference": "orb_edge_linear_preproject_vjp (full 3C input VJP)",
              "forward": "unchanged-preproject-v1", "runtime_input_vjp_columns": width,
              "eliminated_input_vjp_columns": 2 * width}

    def validate_vjp(actual, expected, args, index, probes):
        assert_float32_vjp_reassociation_close(actual, expected, label="ORB first-edge VJP")
        return True

    class FirstEdgeRegion(CheckedRegion):
        def forward(self, *args):
            if args[0].requires_grad or args[4].requires_grad or args[5].requires_grad:
                raise FusionSetupError("first-edge runtime requires geometry-independent nodes and frozen parameters")
            return super().forward(*args)

        def _validate(self, args):
            # Also retain the original native-cat vs preprojection numerical
            # audit for this block; its details are reported, not microbench-gated.
            base._validate(args)
            super()._validate(args)
            itemsize = args[1].element_size()
            self.detail.update(eliminated_intermediate_bytes=2 * args[1].shape[0] * width * itemsize,
                               packed_weight_bytes=width * args[4].shape[0] * itemsize
                               if PASSES[name] == "packed" else 0,
                               original_weight_stride=list(args[4].stride()),
                               vjp_output_shape=[args[1].shape[0], width])

    first._opt4_edge_preproject = FirstEdgeRegion(EdgeLinearPreproject(), detail,
        candidate=FirstEdgePreproject(width, PASSES[name]), vjp_validator=validate_vjp,
        validate_runtime_vjp=True)
    record(report, name, 1, "native-gemm-first-edge-only-vjp", modules=[{"regions": [detail]}],
           retained_base="orb_edge_linear_preproject_vjp", retained_base_modules=len(stack),
           retained_base_details=retained["passes"]["orb_edge_linear_preproject_vjp"],
           new_fused_kernels=0, input_vjp_native_gemms=1,
           optimization="omit inactive node columns; explicit original/packed layout ablation",
           unchanged=["preproject-v1-forward", "blocks-1-onward", "remaining-edge-Linears",
                      "attention", "RMSNorm", "node-MLP", "native-scatter"],
           replay_weight_pack=False, checkpoint_changed=False, precision="unchanged-fp32-or-fp64")
