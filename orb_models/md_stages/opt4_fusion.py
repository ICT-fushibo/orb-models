"""Independent ORB Opt4 experiments; never stack candidates implicitly."""
from __future__ import annotations

import torch

from md_benchmark.opt4_fx import CheckedRegion, assert_float32_vjp_reassociation_close
from md_benchmark.opt4_registry import FusionSetupError, record
from .opt4_rmsnorm_native import require_native_rmsnorm_ops


def _rms_eps(module) -> float:
    eps = module.eps
    if eps is None:
        eps = torch.finfo(module.weight.dtype).eps
    return float(eps)


def _vjp_validator(actual, expected, _args, _input_index, _output_probes):
    assert_float32_vjp_reassociation_close(
        actual,
        expected,
        label="ORB RMSNorm/residual VJP",
    )
    return True

def refresh(model, options) -> None:
    """Revalidate every Graph generation without reinstalling the boundary."""

    del options
    for module in model.modules():
        for name in (
            "_opt4_fasteq_edge_epilogue",
            "_opt4_fasteq_node_epilogue",
            "_opt4_edge_preproject",
        ):
            boundary = getattr(module, name, None)
            if isinstance(boundary, CheckedRegion):
                boundary.signatures.clear()


def install(model, passes, report, options):
    del options
    if "orb_edge_linear_preproject_vjp" in passes:
        if len(passes) != 1:
            raise FusionSetupError("ORB preprojection and RMSNorm are independent, mutually exclusive experiments")
        return _install_preproject(model, report)
    if "fasteq_orb_rmsnorm_residual_vjp" not in passes:
        return

    require_native_rmsnorm_ops()
    from .opt4_rmsnorm_residual import (
        FastEqRMSNormResidual,
        NativeRMSNormResidualReference,
    )

    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if module._node_cond != "none" or module._edge_cond != "none":
            raise FusionSetupError(
                "ORBv3 RMSNorm/residual VJP does not support conditioned blocks"
            )
        if not isinstance(module._edge_mlp.layer_norm, torch.nn.RMSNorm):
            raise FusionSetupError("ORBv3 edge MLP does not use native RMSNorm")
        if not isinstance(module._node_mlp.layer_norm, torch.nn.RMSNorm):
            raise FusionSetupError("ORBv3 node MLP does not use native RMSNorm")
        if hasattr(module, "_opt4_fasteq_edge_epilogue") or hasattr(
            module, "_opt4_fasteq_node_epilogue"
        ):
            raise FusionSetupError("ORBv3 RMSNorm/residual VJP installed twice")

        edge_detail = {
            "module": path,
            "boundary": "edge-rmsnorm-residual-native-forward-explicit-vjp",
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
            "boundary_gate_max_ratio": {"forward": 1.05, "forward_vjp": 0.85},
        }
        node_detail = {
            "module": path,
            "boundary": "node-rmsnorm-residual-native-forward-explicit-vjp",
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
            "boundary_gate_max_ratio": {"forward": 1.05, "forward_vjp": 0.85},
        }
        edge_eps = _rms_eps(module._edge_mlp.layer_norm)
        node_eps = _rms_eps(module._node_mlp.layer_norm)
        module._opt4_fasteq_edge_epilogue = CheckedRegion(
            NativeRMSNormResidualReference(edge_eps, True),
            edge_detail,
            candidate=FastEqRMSNormResidual(edge_eps, True),
            vjp_validator=_vjp_validator,
            validate_runtime_vjp=True,
        )
        module._opt4_fasteq_node_epilogue = CheckedRegion(
            NativeRMSNormResidualReference(node_eps, False),
            node_detail,
            candidate=FastEqRMSNormResidual(node_eps, False),
            vjp_validator=_vjp_validator,
            validate_runtime_vjp=True,
        )
        modules.append({"module": path, "regions": [edge_detail, node_detail]})

    record(
        report,
        "fasteq_orb_rmsnorm_residual_vjp",
        len(modules),
        "native-forward-triton-rmsnorm-residual-explicit-vjp",
        modules=modules,
        fused_boundaries=[
            "rmsnorm-backward-reduction",
            "rmsnorm-input-gradient",
            "residual-gradient-accumulation",
        ],
        forward="native-aten-rmsnorm-then-residual-add",
        gemm="unchanged-native-orb-linear",
        graph_reduction="unchanged-native-orb-segment-sum",
        backward="single-triton-kernel-per-edge-or-node-epilogue",
        saved_statistics="native-aten-inverse-rms",
        parameter_vjp="native-aten-weight-only-when-requested",
        runtime_vjp="frozen-weight-input-and-residual-only",
        replay_runtime_compile=False,
    )


class _PreprojectRegion(CheckedRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_validator=self._validate_output, **kwargs)

    def _validate_output(self, actual, expected, args, output_index):
        from .opt4_preproject_validation import validate_preproject_output
        audit = {"output_index": output_index}
        self.detail.setdefault("forward_validation_audits", []).append(audit)
        validate_preproject_output(actual, expected, args, report=audit)

    def _validate(self, args):
        nodes, edges, _s, _r, weight, _bias = args
        n, c = nodes.shape
        e, h = edges.shape[0], weight.shape[0]
        self.detail.update(
            eliminated_gather_and_concat_bytes=int(5 * e * c * nodes.element_size()),
            projection_temporary_bytes=int((e + 2 * n) * h * nodes.element_size()),
            saved_preactivation_bytes=int(e * h * nodes.element_size()),
            forward_projection_multiply_ratio=(e + 2 * n) / (3 * e) if e else None,
            node_vjp_reduction_width=int(c),
            vjp_edge_gradient_bytes=int(e * 3 * c * nodes.element_size()),
        )
        super()._validate(args)


def _preproject_vjp_validator(actual, expected, _args, _input_index, _output_probes):
    assert_float32_vjp_reassociation_close(actual, expected, label="ORB first-linear preprojection VJP")
    return True


def _install_preproject(model, report):
    from .opt4_edge_preproject import EdgeLinearPreproject, EdgeLinearReference

    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if module.training or module._node_cond != "none" or module._edge_cond != "none":
            raise FusionSetupError("ORB preprojection requires unconditioned eval-mode blocks")
        if any(hasattr(module, name) for name in ("_opt4_edge_preproject", "_opt4_fasteq_edge_epilogue", "_opt4_fasteq_node_epilogue")):
            raise FusionSetupError("ORB Opt4 boundary already installed; candidates cannot be stacked")
        mlp = module._edge_mlp.mlp
        if not isinstance(mlp, torch.nn.Sequential) or len(mlp) < 2:
            raise FusionSetupError("ORB preprojection requires a sequential edge MLP")
        first, activation = mlp[0], mlp[1]
        if (not isinstance(first, torch.nn.Linear) or first.bias is None
                or first.in_features != 3 * module.latent_dim
                or not isinstance(activation, torch.nn.SiLU) or activation.inplace):
            raise FusionSetupError("ORB preprojection only matches first Linear(3C,H,bias)+non-inplace SiLU")
        detail = {"module": path, "boundary": "edge-first-linear-gather-silu-preprojection",
                  "validated_shapes": 0, "benchmark_requested": report.get("benchmark_boundaries", False),
                  "boundary_gate_max_ratio": {"forward": 0.85, "forward_vjp": 0.85}}
        module._opt4_edge_preproject = _PreprojectRegion(
            EdgeLinearReference(), detail, candidate=EdgeLinearPreproject(),
            vjp_validator=_preproject_vjp_validator, validate_runtime_vjp=True)
        modules.append({"module": path, "regions": [detail]})
    record(report, "orb_edge_linear_preproject_vjp", len(modules),
           "native-gemm-triton-gather-silu-explicit-vjp", modules=modules,
           origin="ORB-specific-linear-distributivity; not a ported FastEq kernel",
           fused_boundaries=["edge-first-linear-node-preprojection", "gather-add-bias-silu"],
           forward_custom_kernels=1, vjp_custom_kernels=0, forward_native_gemms=3,
           input_vjp_native_gemms=1,
           activation_forward="libdevice-exp-ieee-fp32-div-rn",
           forward_validation="setup-only-fp64-oracle-and-bounded-reassociation",
           activation_vjp="native-aten-silu-backward-on-saved-preactivation",
           parameter_vjp="explicit-native-GEMM-only-when-requested",
           node_vjp="edge-gemm-then-two-native-index-put-branches-at-latent-width",
           weight_layout="checkpoint-strided-views-no-replay-pack",
           unchanged=["attention", "remaining-edge-MLP", "normalization", "node-MLP", "graph-reduction"],
           replay_runtime_compile=False)
