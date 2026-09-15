"""ORBv3 Opt4: fixed receiver attention reduction."""
from __future__ import annotations

from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_ops import csr_weighted_segment_sum
from md_benchmark.opt4_registry import fixed_csr_layout, record


class _NativeWeightedScatter(nn.Module):
    def __init__(self, edge_rows, rows):
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def forward(self, values, weights, row_scale):
        out = values.new_zeros((self.rows, *values.shape[1:]))
        out.index_add_(0, self.edge_rows, values * weights.reshape(-1, 1))
        return out * row_scale.reshape(-1, 1)


class _FixedWeightedCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row, one):
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.register_buffer("row_scale", one, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row, one):
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.row_scale = one
        self.max_row = int(max_row)

    def forward(self, values, weights, row_scale):
        return csr_weighted_segment_sum(
            values.contiguous(),
            weights.reshape(-1).contiguous(),
            row_scale,
            self.row_ptr,
            self.edge_rows,
            self.max_row,
        )


def _layout(options, parameter):
    row_ptr, edge_rows, max_row = fixed_csr_layout(
        options,
        parameter,
        extra_rows=int(options.get("cuda_graph_dummy_atoms", 32)),
    )
    one = parameter.new_ones(row_ptr.shape[0] - 1)
    return row_ptr, edge_rows, max_row, one


def refresh(model, options):
    row_ptr, edge_rows, max_row, one = _layout(options, next(model.parameters()))
    for module in model.modules():
        region = getattr(module, "_opt4_receive_csr", None)
        if isinstance(region, CheckedRegion):
            region.reference.edge_rows = edge_rows
            region.reference.rows = row_ptr.shape[0] - 1
            region.compiled.set_layout(row_ptr, edge_rows, max_row, one)
            region.signatures.clear()


def install(model, passes, report, options):
    if "receive_attention_csr" not in passes:
        return
    row_ptr, edge_rows, max_row, one = _layout(
        options, next(model.parameters())
    )
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        module.register_buffer(
            "_opt4_receive_scale", one, persistent=False
        )
        module._opt4_receive_csr = CheckedRegion(
            _NativeWeightedScatter(edge_rows, row_ptr.shape[0] - 1),
            detail,
            _FixedWeightedCSR(row_ptr, edge_rows, max_row, one),
        )
        modules.append(detail)
    record(
        report,
        "receive_attention_csr",
        len(modules),
        "triton-fixed-csr-explicit-vjp",
        modules=modules,
        fused_boundaries=["attention-value-multiply", "receiver-segment-sum"],
        sender_path="native-dynamic-segment",
        gemm="unchanged",
        attention_normalization="unchanged",
        edge_layout="receiver-major-directed-reenumeration",
        reverse_edge=False,
        fusion_scope="forward-and-backward",
    )
