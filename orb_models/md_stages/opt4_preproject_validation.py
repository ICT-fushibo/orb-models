"""Setup-only audit of the split Linear, never used to compute MD outputs.

The ordinary elementwise comparison is ill-conditioned near cancellation.
Exceptional FP32 entries may pass ONLY if both implementations also agree with
an independent FP64 oracle within an input-dependent rounding bound, their
pairwise relative L2 error is <=2e-6, and exceptional differences are <=4e-5.
No sampling, inference fallback or model dtype conversion. Parameter gradients
have a separate, setup-only same-adjoint chain-rule audit below.
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


def _gradient_metrics(actual, expected):
    difference = (actual.detach().double() - expected.detach().double()).abs()
    norm = torch.linalg.vector_norm(expected.detach().double())
    error = torch.linalg.vector_norm(difference)
    return {"max_abs": float(difference.max()) if difference.numel() else 0.0,
            "relative_l2": float(error) / max(float(norm), max(1, expected.numel()) ** .5 * 1e-6)}


@torch.no_grad()
def parameter_chain_reference(args, probe, input_index, *, split_forward):
    """Independent ATen linear autograd at the appropriate local SiLU adjoint.

    Reconstruct the two preactivations with plain Torch, NOT the candidate's
    forward kernel or its backward/linear_vjp helper. Linear autograd then sees
    an identical, fixed dz. This separates parameter summation errors from
    the small, independently audited forward reassociation difference.
    Setup only: never called by _Preproject.backward or a captured replay.
    """
    if input_index not in (2, 3):
        raise ValueError("parameter chain reference expects weight or bias index")
    if args[0].is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("parameter chain reference must run before capture")
    # Match CheckedRegion's validation copies, including non-dense input views.
    nodes, edges, s, r, weight, bias = (
        v.detach().clone() if v.is_floating_point() else v for v in args
    )
    features = torch.cat((edges, nodes[s], nodes[r]), dim=1)
    if split_forward:
        width = nodes.shape[1]
        we, ws, wr = weight.split(width, dim=1)
        z = F.linear(edges, we) + F.linear(nodes, ws)[s] + F.linear(nodes, wr)[r] + bias
    else:
        z = F.linear(features, weight, bias)
    dz = torch.ops.aten.silu_backward.default(probe, z)
    del z
    # Do not differentiate the recomputed preactivation. dz is the fixed local
    # adjoint, and autograd checks the original Linear parameter layout/order.
    with torch.enable_grad():
        w = weight.detach().requires_grad_(input_index == 2)
        b = bias.detach().requires_grad_(input_index == 3)
        y = F.linear(features, w, b)
        return torch.autograd.grad(y, w if input_index == 2 else b, dz)[0]


@torch.no_grad()
def validate_preproject_parameter_vjp(actual, expected, args, input_index, output_probes, *, report=None):
    """Require both local chain rules AND unchanged candidate/native L2 bound.

    The old fixed 4e-5 cross-forward ceiling conflated different dz values with
    an incorrect weight/bias VJP after a sum over E edges. It is NOT replaced
    by a larger ceiling: the independent same-adjoint comparison is elementwise
    rtol=1e-5, atol=1e-6 for EACH implementation. Both parameters are still
    checked even when frozen in production. Node/edge VJP gates are untouched.
    """
    if report is None:
        report = {}
    report.update(status="running", method="parameter-same-adjoint-chain-v1",
                  input_index=input_index, same_adjoint_rtol=1e-5,
                  same_adjoint_atol=1e-6, cross_forward_relative_l2_limit=2e-6)
    try:
        if input_index not in (2, 3):
            raise ValueError("parameter audit is not valid for node/edge force gradients")
        if actual.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("parameter audit must finish before capture")
        if len(output_probes) != 1 or not isinstance(output_probes[0], torch.Tensor):
            raise ValueError("parameter audit requires the actual single-output VJP probe")
        parameter = args[4 if input_index == 2 else 5]
        probe = output_probes[0]
        if (probe.shape != (args[1].shape[0], args[4].shape[0])
                or probe.dtype != parameter.dtype or probe.device != parameter.device):
            raise ValueError("parameter audit probe shape/dtype/device changed")
        for value in (actual, expected):
            if value.shape != parameter.shape or value.dtype != parameter.dtype or value.device != parameter.device:
                raise AssertionError("parameter VJP shape/dtype/device changed")
        for value in (*args, actual, expected, probe):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise AssertionError("parameter VJP audit received non-finite values")
        if parameter.dtype != torch.float32:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            report["method"] = "unchanged-fp64-vjp"
        else:
            report["edge_terms"] = args[1].shape[0]
            report["cross_forward"] = _gradient_metrics(actual, expected)
            for label, value, split in (("candidate", actual, True), ("native", expected, False)):
                reference = parameter_chain_reference(args, probe, input_index, split_forward=split)
                report[label + "_same_adjoint"] = _gradient_metrics(value, reference)
                try:
                    torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
                except AssertionError as exc:
                    exc.add_note(f"{label} weight/bias VJP fails its independent same-adjoint chain rule")
                    raise
                del reference
            if report["cross_forward"]["relative_l2"] > 2e-6:
                raise AssertionError("parameter VJP cross-forward relative L2 exceeds unchanged 2e-6 bound")
    except Exception as exc:
        report["status"] = "failed"
        exc.add_note(f"ORB parameter VJP audit: {report}")
        raise
    report["status"] = "passed"
    return report
