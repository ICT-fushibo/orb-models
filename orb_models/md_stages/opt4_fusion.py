"""ORBv3 native-forward, explicit-VJP Opt4 boundaries."""
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
        ):
            boundary = getattr(module, name, None)
            if isinstance(boundary, CheckedRegion):
                boundary.signatures.clear()


def install(model, passes, report, options):
    del options
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
