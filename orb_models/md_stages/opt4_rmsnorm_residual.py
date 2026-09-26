"""Native-forward RMSNorm/residual epilogues with an explicit Triton VJP.

The forward calls the same native ATen RMSNorm as ORB, retaining its inverse
RMS, then adds the residual in the original order. The MD input VJP is Triton;
an optional parameter VJP uses native ATen reduction, never cross-row atomics.
Dense GEMMs, graph reductions, checkpoint parameters and precision are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from .opt4_rmsnorm_native import (
    native_rmsnorm_forward,
    native_rmsnorm_weight_vjp,
)


@triton.jit
def _rmsnorm_residual_bwd(
    grad_norm,
    grad_sum,
    value,
    weight,
    inverse_rms_ptr,
    grad_value,
    grad_norm_stride_0: tl.constexpr,
    grad_norm_stride_1: tl.constexpr,
    grad_sum_stride_0: tl.constexpr,
    grad_sum_stride_1: tl.constexpr,
    value_stride_0: tl.constexpr,
    value_stride_1: tl.constexpr,
    grad_value_stride_0: tl.constexpr,
    grad_value_stride_1: tl.constexpr,
    weight_stride: tl.constexpr,
    width: tl.constexpr,
    HAS_NORM_GRAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < width

    x = tl.load(
        value + row * value_stride_0 + columns * value_stride_1,
        mask=valid,
        other=0.0,
    )
    gamma = tl.load(weight + columns * weight_stride, mask=valid, other=0.0)
    upstream = tl.load(
        grad_sum + row * grad_sum_stride_0 + columns * grad_sum_stride_1,
        mask=valid,
        other=0.0,
    )
    if HAS_NORM_GRAD:
        upstream += tl.load(
            grad_norm
            + row * grad_norm_stride_0
            + columns * grad_norm_stride_1,
            mask=valid,
            other=0.0,
        )

    # Use the exact statistic saved by native forward, including its FP32
    # reduction order. Recomputing it here perturbs cancellation-sensitive VJPs.
    inverse_rms = tl.load(inverse_rms_ptr + row)
    weighted = upstream * gamma
    projection = tl.sum(weighted * x, axis=0) / width
    dx = weighted * inverse_rms - x * projection * inverse_rms * inverse_rms * inverse_rms
    tl.store(
        grad_value + row * grad_value_stride_0 + columns * grad_value_stride_1,
        dx,
        mask=valid,
    )


def _validate(value, residual, weight) -> tuple[int, int]:
    if not value.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise RuntimeError("ORBv3 RMSNorm/residual VJP requires CUDA tensors")
    if value.device != residual.device or value.device != weight.device:
        raise ValueError("ORBv3 RMSNorm/residual inputs must share one CUDA device")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("ORBv3 RMSNorm/residual supports float32 and float64")
    if residual.dtype != value.dtype or weight.dtype != value.dtype:
        raise ValueError("ORBv3 RMSNorm/residual inputs must have one dtype")
    if value.ndim != 2 or residual.shape != value.shape:
        raise ValueError("ORBv3 RMSNorm/residual requires equal 2D tensors")
    if weight.ndim != 1 or weight.numel() != value.shape[1]:
        raise ValueError("ORBv3 RMSNorm weight does not match the feature width")
    rows, width = map(int, value.shape)
    if rows < 1 or width < 1:
        raise ValueError("ORBv3 RMSNorm/residual does not accept empty tensors")
    return rows, width


def _backward(ctx, grad_norm, grad_sum, *, has_norm_grad):
    value, weight, inverse_rms = ctx.saved_tensors
    rows, width = map(int, value.shape)
    grad_value = torch.empty_strided(
        value.shape,
        value.stride(),
        dtype=value.dtype,
        device=value.device,
    )
    need_weight_grad = bool(ctx.needs_input_grad[2])
    # ``grad_norm`` is unused by the node epilogue.  Passing grad_sum keeps the
    # launch ABI static while HAS_NORM_GRAD removes the load at compile time.
    if grad_norm is None:
        grad_norm = grad_sum
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise RuntimeError("ORBv3 RMSNorm feature width exceeds the Triton limit")
    _rmsnorm_residual_bwd[(rows,)](
        grad_norm,
        grad_sum,
        value,
        weight,
        inverse_rms,
        grad_value,
        grad_norm.stride(0),
        grad_norm.stride(1),
        grad_sum.stride(0),
        grad_sum.stride(1),
        value.stride(0),
        value.stride(1),
        grad_value.stride(0),
        grad_value.stride(1),
        weight.stride(0),
        width,
        HAS_NORM_GRAD=has_norm_grad,
        BLOCK=block,
    )
    grad_weight = None
    if need_weight_grad:
        upstream = grad_norm + grad_sum if has_norm_grad else grad_sum
        grad_weight = native_rmsnorm_weight_vjp(
            upstream, value, inverse_rms, weight
        )
    return grad_value, grad_sum, grad_weight, None


class _RMSNormResidualPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, residual, weight, eps):
        _validate(value, residual, weight)
        normalized, inverse_rms = native_rmsnorm_forward(value, weight, eps)
        result = residual + normalized
        ctx.save_for_backward(value, weight, inverse_rms)
        return normalized, result

    @staticmethod
    def backward(ctx, grad_norm, grad_sum):
        return _backward(ctx, grad_norm, grad_sum, has_norm_grad=True)


class _RMSNormResidualOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, residual, weight, eps):
        _validate(value, residual, weight)
        normalized, inverse_rms = native_rmsnorm_forward(value, weight, eps)
        result = residual + normalized
        ctx.save_for_backward(value, weight, inverse_rms)
        return result

    @staticmethod
    def backward(ctx, grad_sum):
        return _backward(ctx, None, grad_sum, has_norm_grad=False)


class NativeRMSNormResidualReference(nn.Module):
    """Validation oracle with the original PyTorch forward and backward."""

    def __init__(self, eps: float, return_normalized: bool) -> None:
        super().__init__()
        self.eps = float(eps)
        self.return_normalized = bool(return_normalized)

    def forward(self, value, residual, weight):
        normalized = F.rms_norm(value, (value.shape[-1],), weight, self.eps)
        result = residual + normalized
        if self.return_normalized:
            return normalized, result
        return result


class FastEqRMSNormResidual(nn.Module):
    """Native forward paired with the shape-specialized explicit VJP."""

    def __init__(self, eps: float, return_normalized: bool) -> None:
        super().__init__()
        self.eps = float(eps)
        self.return_normalized = bool(return_normalized)

    def forward(self, value, residual, weight):
        if self.return_normalized:
            return _RMSNormResidualPair.apply(value, residual, weight, self.eps)
        return _RMSNormResidualOnly.apply(value, residual, weight, self.eps)
