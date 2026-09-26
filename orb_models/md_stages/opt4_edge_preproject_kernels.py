"""Only gather/add/SiLU materialization; all dense products remain native."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gather_silu(PE, PS, PR, S, R, BIAS, Y, Z,
                 E: tl.constexpr, H: tl.constexpr,
                 PE0: tl.constexpr, PE1: tl.constexpr,
                 PS0: tl.constexpr, PS1: tl.constexpr,
                 PR0: tl.constexpr, PR1: tl.constexpr,
                 SS: tl.constexpr, RS: tl.constexpr, BS: tl.constexpr,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge, channel = offsets // H, offsets % H
    valid = edge < E
    sender = tl.load(S + edge * SS, valid, other=0).to(tl.int64)
    receiver = tl.load(R + edge * RS, valid, other=0).to(tl.int64)
    z = tl.load(PE + edge * PE0 + channel * PE1, valid, other=0)
    z = z + tl.load(PS + sender * PS0 + channel * PS1, valid, other=0)
    z = z + tl.load(PR + receiver * PR0 + channel * PR1, valid, other=0)
    z = z + tl.load(BIAS + channel * BS, valid, other=0)
    denominator = 1.0 + libdevice.exp(-z)
    if z.dtype == tl.float32:
        # Triton's '/' may lower to reciprocal-based approximate division.
        # Keep the native SiLU exp/div formula, with explicit IEEE FP32 divide.
        y = tl.div_rn(z, denominator)
    else:
        y = z / denominator
    tl.store(Z + offsets, z, valid)
    tl.store(Y + offsets, y, valid)


def gather_silu(pe, ps, pr, senders, receivers, bias):
    output, z = torch.empty_like(pe), torch.empty_like(pe)
    if pe.numel():
        _gather_silu[(triton.cdiv(pe.numel(), 256),)](
            pe, ps, pr, senders, receivers, bias, output, z, *pe.shape,
            *pe.stride(), *ps.stride(), *pr.stride(),
            senders.stride(0), receivers.stride(0), bias.stride(0),
            BLOCK=256, enable_fp_fusion=False)
    return output, z
