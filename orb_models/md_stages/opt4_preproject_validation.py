"""Setup-only audit of the split Linear, never used to compute MD outputs.

The ordinary elementwise comparison is ill-conditioned near cancellation.
Exceptional FP32 entries may pass ONLY if both implementations also agree with
an independent FP64 oracle within an input-dependent rounding bound, their
pairwise relative L2 error is <=2e-6, and exceptional differences are <=4e-5.
No sampling, inference fallback, changed VJP tolerance or model dtype conversion.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def validate_preproject_output(actual, expected, args, *, report=None, chunk_edges=256):
    if actual.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("preprojection FP64 audit must finish before CUDA Graph capture")
    if report is None:
        report = {}
    report.update(status="running", method="fp64-linear-silu-roundoff-audit-v1",
                  ordinary_rtol=1e-5, ordinary_atol=1e-6,
                  exceptional_absolute_ceiling=4e-5, relative_l2_limit=2e-6,
                  oracle_chunk_edges=chunk_edges, oracle_all_elements=True)
    try:
        _audit(actual, expected, args, report, chunk_edges)
    except Exception:
        report["status"] = "failed"
        raise
    report["status"] = "passed"
    return report


def _audit(actual, expected, args, report, chunk_edges):
    from .opt4_edge_preproject import validate_inputs

    validate_inputs(*args)
    nodes, edges, senders, receivers, weight, bias = (x.detach() for x in args)
    shape = (edges.shape[0], weight.shape[0])
    if chunk_edges < 1:
        raise ValueError("preprojection audit requires positive chunk_edges")
    for label, value in (("candidate", actual), ("native", expected)):
        if tuple(value.shape) != shape or value.dtype != nodes.dtype or value.device != nodes.device:
            raise AssertionError(f"preprojection {label} shape/dtype/device changed")
    for value in (*args, actual, expected):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise AssertionError("preprojection audit received non-finite values")
    if nodes.dtype == torch.float64:
        # Never apply an FP32 reassociation allowance to FP64.
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        report["method"] = "strict-fp64"
        return

    # Separate multiply/add worst-case path length, conservatively covering
    # both the single K-wide GEMM and the three C-wide products plus additions.
    # u=eps/2; gamma_n=n*u/(1-n*u), NOT a fitted tolerance from the failing log.
    k = weight.shape[1]
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    nu = (2 * k + 4) * unit_roundoff
    if nu >= 1:
        raise ValueError("preprojection rounding bound is not valid for this width")
    gamma = nu / (1 - nu)
    report.update(dot_terms=k, unit_roundoff=unit_roundoff, gamma=gamma)
    nd, wd, bd = nodes.double(), weight.double(), bias.double()
    wa, ba = wd.abs(), bd.abs()
    # Accumulate statistics on device. Host transfers occur once after the
    # complete, bounded-memory audit, before capture/benchmark warmup.
    stats = torch.zeros(9, dtype=torch.float64, device=nodes.device)
    for begin in range(0, shape[0], chunk_edges):
        end = min(begin + chunk_edges, shape[0])
        features = torch.cat((edges[begin:end].double(), nd[senders[begin:end]],
                              nd[receivers[begin:end]]), dim=1)
        z = F.linear(features, wd, bd)
        truth = F.silu(z)
        magnitude = F.linear(features.abs(), wa, ba)
        # |SiLU'| < 1.1: 2 is a conservative Lipschitz bound. The activation
        # budget covers FP32 exp/div rounding; the two magnitude gates below
        # prevent a poorly conditioned dot from accepting a large output error.
        allowed = 2 * gamma * magnitude + 8 * unit_roundoff * (1 + z.abs() + truth.abs())
        a, b = actual[begin:end].double(), expected[begin:end].double()
        difference = (a - b).abs()
        exceptional = difference > (1e-6 + 1e-5 * b.abs())
        ca, cb = (a - truth).abs(), (b - truth).abs()
        stats[0] += exceptional.sum()
        stats[1] = torch.maximum(stats[1], difference.max())
        stats[2] = torch.maximum(stats[2], torch.where(exceptional, difference, 0).max())
        stats[3] += difference.square().sum()
        stats[4] += b.square().sum()
        stats[5] = torch.maximum(stats[5], ca.max())
        stats[6] = torch.maximum(stats[6], cb.max())
        stats[7] = torch.maximum(stats[7], (ca - allowed).clamp_min(0).max())
        stats[8] = torch.maximum(stats[8], (cb - allowed).clamp_min(0).max())
    values = stats.tolist()
    count, max_abs, exceptional_abs, error2, norm2, ca, cb, ca_excess, cb_excess = values
    relative_l2 = math.sqrt(error2) / max(math.sqrt(norm2), max(1, actual.numel()) ** .5 * 1e-6)
    report.update(elements=actual.numel(), ordinary_mismatched=int(count),
                  max_abs=max_abs, exceptional_max_abs=exceptional_abs, relative_l2=relative_l2,
                  candidate_oracle_max_abs=ca, native_oracle_max_abs=cb,
                  candidate_oracle_bound_excess=ca_excess, native_oracle_bound_excess=cb_excess)
    failures = []
    if not all(math.isfinite(x) for x in values) or not math.isfinite(relative_l2):
        failures.append("non-finite oracle/statistics")
    if ca_excess > 0 or cb_excess > 0:
        failures.append("candidate/native exceeds input-dependent FP64 oracle bound")
    if exceptional_abs > 4e-5:
        failures.append("exceptional absolute error exceeds 4e-5")
    if relative_l2 > 2e-6:
        failures.append("relative L2 exceeds 2e-6")
    if failures:
        raise AssertionError("ORB preprojection forward audit: " + "; ".join(failures) + f"; metrics={report}")
