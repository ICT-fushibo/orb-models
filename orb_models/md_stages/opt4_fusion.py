"""FastEq-inspired dual-attention aggregation boundary for ORBv3 Opt4.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion, assert_dot_reduction_close
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


class _DualAttention(nn.Module):
    """Compute both sigmoid gates and the cutoff with one compiled edge read."""

    def __init__(self, receive_attn: nn.Module, send_attn: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_receive_attn", receive_attn)
        object.__setattr__(self, "_send_attn", send_attn)

    def forward(self, edges, cutoff):
        receive = torch.sigmoid(self._receive_attn(edges)) * cutoff
        send = torch.sigmoid(self._send_attn(edges)) * cutoff
        return receive, send


class _DualAggregatePack(nn.Module):
    """Reduce receive/send messages and directly produce node-MLP input."""

    def __init__(self, edge_rows, rows: int) -> None:
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def set_layout(self, edge_rows, rows: int) -> None:
        self.edge_rows = edge_rows
        self.rows = int(rows)

    def forward(self, updated_edges, receive_attn, send_attn, nodes, senders):
        received = updated_edges.new_zeros((self.rows, updated_edges.shape[-1]))
        received.index_add_(0, self.edge_rows, updated_edges * receive_attn)
        sent = updated_edges.new_zeros((self.rows, updated_edges.shape[-1]))
        sent.index_add_(0, senders, updated_edges * send_attn)
        return torch.cat((nodes, received, sent), dim=-1)

    def validate_vjp(self, actual, expected, args, input_index, output_probes):
        """Account for legal fp32 dot-order changes in attention VJPs.

        The scalar attention gradients are width-sized dot reductions.  The
        eager and compiled paths may use different reduction trees, so compare
        both with the IEEE ``gamma_n`` bound instead of a global loose atol.
        """

        if input_index not in (1, 2):
            return False
        updated_edges, _receive_attn, _send_attn, _nodes, senders = args
        probe = output_probes[0]
        if probe is None:
            raise RuntimeError("dual aggregate validation requires an output probe")
        width = updated_edges.shape[-1]
        if input_index == 1:
            output_probe = probe[:, width : 2 * width].index_select(
                0, self.edge_rows
            )
        else:
            output_probe = probe[:, 2 * width :].index_select(0, senders)
        terms = updated_edges * output_probe
        assert_dot_reduction_close(actual, expected, terms)
        return True


def _layout(options, parameter):
    row_ptr, edge_rows, _max_row = fixed_csr_layout(
        options,
        parameter,
        extra_rows=int(options.get("cuda_graph_dummy_atoms", 32)),
    )
    return edge_rows, int(row_ptr.shape[0] - 1)


def refresh(model, options) -> None:
    edge_rows, rows = _layout(options, next(model.parameters()))
    for module in model.modules():
        aggregate = getattr(module, "_opt4_fasteq_aggregate", None)
        if isinstance(aggregate, CheckedRegion):
            module._opt4_edge_capacity = int(edge_rows.numel())
            aggregate.reference.set_layout(edge_rows, rows)
            aggregate.signatures.clear()
            attention = getattr(module, "_opt4_fasteq_attention", None)
            if isinstance(attention, CheckedRegion):
                attention.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_dual_attention_pack" not in passes:
        return
    edge_rows, rows = _layout(options, next(model.parameters()))
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if module._attention_gate != "sigmoid" or not module._distance_cutoff:
            raise FusionSetupError(
                "ORBv3 FastEq boundary requires sigmoid attention with distance cutoff"
            )
        if module._node_cond != "none" or module._edge_cond != "none":
            raise FusionSetupError(
                "ORBv3 FastEq boundary does not support conditioned interaction blocks"
            )
        detail = {
            "module": path,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        attention_detail = {
            "boundary": "dual-attention-cutoff",
            "validated_shapes": 0,
            "benchmark_requested": detail["benchmark_requested"],
        }
        aggregate_detail = {
            "boundary": "dual-reduction-node-pack",
            "validated_shapes": 0,
            "benchmark_requested": detail["benchmark_requested"],
        }
        detail["regions"] = [attention_detail, aggregate_detail]
        module._opt4_fasteq_attention = CheckedRegion(
            _DualAttention(module._receive_attn, module._send_attn),
            attention_detail,
        )
        aggregate = _DualAggregatePack(edge_rows, rows)
        module._opt4_fasteq_aggregate = CheckedRegion(
            aggregate,
            aggregate_detail,
            vjp_validator=aggregate.validate_vjp,
        )
        module._opt4_edge_capacity = int(edge_rows.numel())
        modules.append(detail)
    record(
        report,
        "fasteq_dual_attention_pack",
        len(modules),
        "inductor-triton-dual-attention-pack-aot-vjp",
        modules=modules,
        fused_boundaries=[
            "receive-send-linear-sigmoid-cutoff",
            "dual-weighted-reduction",
            "node-mlp-input-pack",
        ],
        receiver_layout="fixed-destination-slots",
        sender_layout="dynamic-directed-index",
        gemm="original-orb-mlp",
        backward="aot-compiled-complete-input-vjp",
        replay_runtime_compile=False,
    )
