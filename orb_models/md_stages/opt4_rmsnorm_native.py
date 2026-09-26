"""Native RMSNorm statistics and parameter-only VJP for the Opt4 boundary.

PyTorch 2.11's public RMSNorm dispatches to this same ATen forward for our
2D, same-dtype FP32/FP64 inputs. Keep its saved inverse RMS and its native
weight-gradient reduction; never reconstruct either with an atomic sum.
"""
from __future__ import annotations

import torch

from md_benchmark.opt4_registry import FusionSetupError


def require_native_rmsnorm_ops() -> None:
    missing = [
        name for name in ("_fused_rms_norm", "_fused_rms_norm_backward")
        if not hasattr(torch.ops.aten, name)
    ]
    if missing:
        raise FusionSetupError(
            "ORB Opt4 RMSNorm VJP requires installed ATen operators "
            + ", ".join(missing)
            + "; environment is unchanged (no automatic install or fallback)"
        )


def native_rmsnorm_forward(value, weight, eps):
    return torch.ops.aten._fused_rms_norm.default(
        value, [value.shape[-1]], weight, eps
    )


def native_rmsnorm_weight_vjp(upstream, value, inverse_rms, weight):
    # The input VJP remains in Triton. This branch is requested by all-input
    # validation only when checkpoint weights are frozen for MD.
    return torch.ops.aten._fused_rms_norm_backward.default(
        upstream, value, [value.shape[-1]], inverse_rms, weight, [False, True]
    )[1]
