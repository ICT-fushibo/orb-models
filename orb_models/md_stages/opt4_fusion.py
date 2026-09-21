"""FastEq-inspired edge gather/pack boundary for ORBv3 Opt4.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion, assert_associative_sum_close
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


class _EdgeGatherPackReference(nn.Module):
    """Native ORB edge-MLP input boundary used as the validation oracle."""

    def forward(self, edges, nodes, senders, receivers):
        return torch.cat(
            (edges, nodes.index_select(0, senders), nodes.index_select(0, receivers)),
            dim=-1,
        )

    def validate_vjp(self, actual, expected, args, input_index, output_probes):
        """Allow only legal fp32 reassociation in the node gather VJP."""

        if input_index != 1:
            return False
        edges, _nodes, senders, receivers = args
        probe = output_probes[0]
        if probe is None:
            raise RuntimeError("edge gather/pack validation requires an output probe")
        width = int(edges.shape[-1])
        values = torch.cat(
            (probe[:, width : 2 * width], probe[:, 2 * width :]), dim=0
        )
        rows = torch.cat((senders, receivers), dim=0)
        counts = torch.bincount(rows, minlength=actual.shape[0])
        max_terms = max(1, int(counts.max().item()))
        assert_associative_sum_close(
            actual,
            expected,
            values,
            rows,
            int(actual.shape[0]),
            max_terms,
        )
        return True


def refresh(model, options) -> None:
    """Invalidate setup signatures after a CAP promotion/Graph generation."""

    parameter = next(model.parameters())
    row_ptr, _edge_rows, max_row = fixed_csr_layout(options, parameter)
    edge_capacity = int(sum(options["neighbor_capacities"]))
    for module in model.modules():
        boundary = getattr(module, "_opt4_fasteq_edge_pack", None)
        if isinstance(boundary, CheckedRegion):
            module._opt4_edge_capacity = edge_capacity
            boundary.compiled.set_layout(row_ptr, max_row)
            boundary.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_edge_gather_pack_vjp" not in passes:
        return
    capacities = options.get("neighbor_capacities")
    if (
        not isinstance(capacities, (list, tuple))
        or not capacities
        or any(type(value) is not int or value < 1 for value in capacities)
    ):
        raise FusionSetupError(
            "ORBv3 edge gather/pack requires probe-derived neighbor capacities"
        )
    parameter = next(model.parameters())
    row_ptr, _edge_rows, max_row = fixed_csr_layout(options, parameter)
    edge_capacity = int(sum(capacities))

    # Importing this module imports Triton. Keep that dependency scoped to an
    # explicitly requested Opt4 pass so baseline/Opt1--Opt3 imports are intact.
    from .opt4_gather_pack import FastEqEdgeGatherPack

    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if module._node_cond != "none" or module._edge_cond != "none":
            raise FusionSetupError(
                "ORBv3 FastEq gather/pack does not support conditioned blocks"
            )
        detail = {
            "module": path,
            "boundary": "edge-node-gather-pack",
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        reference = _EdgeGatherPackReference()
        module._opt4_fasteq_edge_pack = CheckedRegion(
            reference,
            detail,
            candidate=FastEqEdgeGatherPack(row_ptr, max_row),
            vjp_validator=reference.validate_vjp,
        )
        module._opt4_edge_capacity = edge_capacity
        modules.append(detail)
    record(
        report,
        "fasteq_edge_gather_pack_vjp",
        len(modules),
        "triton-edge-gather-pack-explicit-vjp",
        modules=modules,
        fused_boundaries=[
            "sender-node-gather",
            "receiver-node-gather",
            "edge-mlp-input-pack",
            "dual-indexing-backward",
        ],
        edge_order="native-dynamic-directed",
        gemm="original-orb-mlp",
        backward="explicit-edge-copy-receiver-atomic-sender-csr-vjp",
        replay_runtime_compile=False,
    )
