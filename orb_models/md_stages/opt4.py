"""ORBv3 Opt4 route using the fixed-segment Opt3 graph.

ORBv3's GNS attention has separate send/receive segment reductions.  Until a
fixed-output fused implementation is proven for both paths, Opt4 keeps the
validated Opt3 segment metadata and records that the optional fusion is not
enabled.
"""

from __future__ import annotations

from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.opt4_policy import run_opt4_with_opt3

from . import opt3


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "orbv3" or request.stage != "opt4":
        raise ValueError(f"ORBv3 Opt4 route received {request.model}/{request.stage}")
    result, policy = run_opt4_with_opt3(request, opt3.run_md, model="orbv3")
    result.stage = "opt4"
    result.metadata.update(policy.metadata)
    result.metadata.update(
        {
            "opt4_model_strategy": "fixed-segment-attention-layout",
            "opt4_optimization_targets": [
                "send-segment-metadata",
                "receive-segment-metadata",
                "cutoff-value-preparation",
            ],
            "opt4_fixed_address_buffers": True,
            "opt4_custom_kernel": False,
            "opt4_reverse_edge_verified": False,
        }
    )
    validate_result(request, result)
    return result


__all__ = ["run_md"]
