"""FastEq-inspired, full ORB GNS processor-block AOT boundary.

The design follows the operator-coarsening principle used by FastEq commit
40ba40e72bee769d74a869bb4a4ba820ee1c55c0 (MIT): compile a sufficiently
large equivariant/message-passing region so that pointwise, layout and
reduction intermediates can be eliminated, while leaving dense GEMMs to the
vendor library. FastEq is a design source, not a runtime dependency.
"""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_registry import FusionSetupError, record
from orb_models.common.models import segment_ops


class _GNSProcessorBlockReference(nn.Module):
    """Exact inference graph for one native ``AttentionInteractionNetwork``.

    The referenced Linear/MLP modules are shared with the checkpoint. They are
    registered here only so Dynamo treats their parameters as module inputs;
    no parameter is copied or rewritten.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.receive_attn = block._receive_attn
        self.send_attn = block._send_attn
        self.edge_mlp = block._edge_mlp
        self.node_mlp = block._node_mlp
        self.distance_cutoff = bool(block._distance_cutoff)

    def forward(self, nodes, edges, senders, receivers, cutoff):
        receive_attn = torch.sigmoid(self.receive_attn(edges))
        send_attn = torch.sigmoid(self.send_attn(edges))
        if self.distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        edge_features = torch.cat(
            (
                edges,
                nodes.index_select(0, senders),
                nodes.index_select(0, receivers),
            ),
            dim=-1,
        )
        updated_edges = self.edge_mlp(edge_features)
        sent_attributes = segment_ops.segment_sum(
            updated_edges * send_attn, senders, nodes.shape[0]
        )
        received_attributes = segment_ops.segment_sum(
            updated_edges * receive_attn, receivers, nodes.shape[0]
        )
        updated_nodes = self.node_mlp(
            torch.cat((nodes, received_attributes, sent_attributes), dim=-1)
        )
        return nodes + updated_nodes, edges + updated_edges


def refresh(model, options) -> None:
    """Force validation/compilation of a promoted CAP shape before capture."""

    del options
    for module in model.modules():
        boundary = getattr(module, "_opt4_fasteq_processor_aot", None)
        if isinstance(boundary, CheckedRegion):
            boundary.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_gns_processor_aot_vjp" not in passes:
        return
    capacities = options.get("neighbor_capacities")
    if (
        not isinstance(capacities, (list, tuple))
        or not capacities
        or any(type(value) is not int or value < 1 for value in capacities)
    ):
        raise FusionSetupError(
            "ORBv3 processor AOT requires probe-derived neighbor capacities"
        )

    modules = []
    # Materialise the list before installing children that share checkpoint
    # submodules; this prevents traversal of the just-created AOT wrappers.
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if module._node_cond != "none" or module._edge_cond != "none":
            raise FusionSetupError(
                "ORBv3 processor AOT does not support conditioned blocks"
            )
        if module._attention_gate != "sigmoid":
            raise FusionSetupError(
                "ORBv3 processor AOT currently supports sigmoid attention only"
            )
        if hasattr(module, "_opt4_fasteq_processor_aot"):
            raise FusionSetupError("ORBv3 processor AOT was installed more than once")

        detail = {
            "module": path,
            "boundary": "complete-attention-interaction-network",
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        reference = _GNSProcessorBlockReference(module)
        module._opt4_fasteq_processor_aot = CheckedRegion(reference, detail)
        modules.append(detail)

    record(
        report,
        "fasteq_gns_processor_aot_vjp",
        len(modules),
        "torch-inductor-fullgraph-forward-vjp",
        modules=modules,
        fused_boundaries=[
            "dual-attention-sigmoid-cutoff",
            "sender-receiver-gather-pack",
            "edge-mlp-pointwise-and-normalization",
            "dual-weighted-segment-reduction",
            "node-mlp-input-pack-pointwise-and-normalization",
            "node-edge-residual",
        ],
        edge_order="native-dynamic-directed",
        gemm="native-orb-linear-external-calls",
        backward="inductor-aot-complete-first-order-vjp",
        internal_cuda_graph=False,
        replay_runtime_compile=False,
        shape_specialization="node-count-edge-capacity-dtype-stride",
    )
