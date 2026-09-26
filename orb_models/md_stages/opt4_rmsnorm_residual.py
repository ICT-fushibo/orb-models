"""Native-forward RMSNorm/residual epilogues with an explicit Triton VJP.

The forward deliberately calls PyTorch's native RMSNorm and residual add in
the same order as ORB.  Only the first-order inference VJP is replaced.  Dense
GEMMs, graph reductions, checkpoint parameters, and model precision are left
unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rmsnorm_residual_bwd(
    grad_norm,
    grad_sum,
    value,
    weight,
    grad_value,
    grad_weight,
    grad_norm_stride_0: tl.constexpr,
    grad_norm_stride_1: tl.constexpr,
    grad_sum_stride_0: tl.constexpr,
    grad_sum_stride_1: tl.constexpr,
    value_stride_0: tl.constexpr,
    value_stride_1: tl.constexpr,
    grad_value_stride_0: tl.constexpr,
    grad_value_stride_1: tl.constexpr,
    weight_stride: tl.constexpr,
    eps: tl.constexpr,
    width: tl.constexpr,
    HAS_NORM_GRAD: tl.constexpr,
    NEED_WEIGHT_GRAD: tl.constexpr,
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

    mean_square = tl.sum(x * x, axis=0) / width
    inverse_rms = tl.rsqrt(mean_square + eps)
    weighted = upstream * gamma
    projection = tl.sum(weighted * x, axis=0) / width
    dx = weighted * inverse_rms - x * projection * inverse_rms * inverse_rms * inverse_rms
    tl.store(
        grad_value + row * grad_value_stride_0 + columns * grad_value_stride_1,
        dx,
        mask=valid,
    )

    if NEED_WEIGHT_GRAD:
        tl.atomic_add(
            grad_weight + columns * weight_stride,
            upstream * x * inverse_rms,
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
    value, weight = ctx.saved_tensors
    rows, width = map(int, value.shape)
    grad_value = torch.empty_strided(
        value.shape,
        value.stride(),
        dtype=value.dtype,
        device=value.device,
    )
    need_weight_grad = bool(ctx.needs_input_grad[2])
    grad_weight = torch.zeros_like(weight) if need_weight_grad else weight
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
        grad_value,
        grad_weight,
        grad_norm.stride(0),
        grad_norm.stride(1),
        grad_sum.stride(0),
        grad_sum.stride(1),
        value.stride(0),
        value.stride(1),
        grad_value.stride(0),
        grad_value.stride(1),
        weight.stride(0),
        ctx.eps,
        width,
        HAS_NORM_GRAD=has_norm_grad,
        NEED_WEIGHT_GRAD=need_weight_grad,
        BLOCK=block,
    )
    return grad_value, grad_sum, grad_weight if need_weight_grad else None, None


class _RMSNormResidualPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, residual, weight, eps):
        _validate(value, residual, weight)
        normalized = F.rms_norm(value, (value.shape[-1],), weight, eps)
        result = residual + normalized
        ctx.save_for_backward(value, weight)
        ctx.eps = float(eps)
        return normalized, result

    @staticmethod
    def backward(ctx, grad_norm, grad_sum):
        return _backward(ctx, grad_norm, grad_sum, has_norm_grad=True)


class _RMSNormResidualOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, residual, weight, eps):
        _validate(value, residual, weight)
        normalized = F.rms_norm(value, (value.shape[-1],), weight, eps)
        result = residual + normalized
        ctx.save_for_backward(value, weight)
        ctx.eps = float(eps)
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
