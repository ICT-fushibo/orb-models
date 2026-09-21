"""Shape-specialized Triton gather/pack with a complete first-order VJP."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _edge_gather_pack_fwd(
    edges,
    nodes,
    senders,
    receivers,
    output,
    edge_stride_0: tl.constexpr,
    edge_stride_1: tl.constexpr,
    node_stride_0: tl.constexpr,
    node_stride_1: tl.constexpr,
    edge_count: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_width: tl.constexpr = 3 * width
    total: tl.constexpr = edge_count * packed_width
    valid = offsets < total
    edge = offsets // packed_width
    column = offsets - edge * packed_width
    sender = tl.load(senders + edge, mask=valid, other=0)
    receiver = tl.load(receivers + edge, mask=valid, other=0)

    is_edge = column < width
    is_sender = (column >= width) & (column < 2 * width)
    value = tl.load(
        edges + edge * edge_stride_0 + column * edge_stride_1,
        mask=valid & is_edge,
        other=0.0,
    )
    value += tl.load(
        nodes + sender * node_stride_0 + (column - width) * node_stride_1,
        mask=valid & is_sender,
        other=0.0,
    )
    value += tl.load(
        nodes + receiver * node_stride_0 + (column - 2 * width) * node_stride_1,
        mask=valid & ~(is_edge | is_sender),
        other=0.0,
    )
    tl.store(output + offsets, value, mask=valid)


@triton.jit
def _edge_gather_pack_bwd(
    grad_output,
    senders,
    receivers,
    grad_edges,
    grad_nodes,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    edge_stride_0: tl.constexpr,
    edge_stride_1: tl.constexpr,
    node_stride_0: tl.constexpr,
    node_stride_1: tl.constexpr,
    real_nodes: tl.constexpr,
    edge_count: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total: tl.constexpr = edge_count * width
    valid = offsets < total
    edge = offsets // width
    column = offsets - edge * width
    sender = tl.load(senders + edge, mask=valid, other=0)
    receiver = tl.load(receivers + edge, mask=valid, other=0)
    base = edge * output_stride_0 + column * output_stride_1
    grad_edge = tl.load(grad_output + base, mask=valid, other=0.0)
    grad_sender = tl.load(
        grad_output + base + width * output_stride_1,
        mask=valid,
        other=0.0,
    )
    grad_receiver = tl.load(
        grad_output + base + 2 * width * output_stride_1,
        mask=valid,
        other=0.0,
    )
    tl.store(
        grad_edges + edge * edge_stride_0 + column * edge_stride_1,
        grad_edge,
        mask=valid,
    )
    tl.atomic_add(
        grad_nodes + receiver * node_stride_0 + column * node_stride_1,
        grad_receiver,
        mask=valid,
    )
    # Fixed capacity slots are sender-major for real nodes. Their sender VJP
    # is handled without atomics by the CSR kernel below. Padding sink edges
    # use dummy senders and remain on this dynamic atomic path.
    tl.atomic_add(
        grad_nodes + sender * node_stride_0 + column * node_stride_1,
        grad_sender,
        mask=valid & (sender >= real_nodes),
    )


@triton.jit
def _edge_gather_pack_sender_csr_bwd(
    grad_output,
    senders,
    row_ptr,
    grad_nodes,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    node_stride_0: tl.constexpr,
    node_stride_1: tl.constexpr,
    width: tl.constexpr,
    MAX_ROW: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid_column = columns < width
    begin = tl.load(row_ptr + row)
    end = tl.load(row_ptr + row + 1)
    if IS_FP64:
        accumulator = tl.zeros((BLOCK,), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for slot in tl.static_range(0, MAX_ROW):
        edge = begin + slot
        valid = valid_column & (edge < end)
        sender = tl.load(senders + edge, mask=edge < end, other=-1)
        value = tl.load(
            grad_output
            + edge * output_stride_0
            + (width + columns) * output_stride_1,
            mask=valid & (sender == row),
            other=0.0,
        )
        accumulator += value
    address = grad_nodes + row * node_stride_0 + columns * node_stride_1
    current = tl.load(address, mask=valid_column, other=0.0)
    tl.store(address, current + accumulator, mask=valid_column)


class _EdgeGatherPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, nodes, senders, receivers, row_ptr, max_row):
        if not edges.is_cuda or not nodes.is_cuda:
            raise RuntimeError("ORBv3 FastEq gather/pack requires CUDA tensors")
        if (
            nodes.device != edges.device
            or senders.device != edges.device
            or receivers.device != edges.device
        ):
            raise ValueError("edge gather/pack inputs must share one CUDA device")
        if (
            edges.dtype not in (torch.float32, torch.float64)
            or nodes.dtype != edges.dtype
        ):
            raise ValueError("edge gather/pack requires equal fp32/fp64 feature dtypes")
        if senders.dtype != torch.int64 or receivers.dtype != torch.int64:
            raise ValueError("edge gather/pack indices must use torch.int64")
        if edges.ndim != 2 or nodes.ndim != 2 or edges.shape[1] != nodes.shape[1]:
            raise ValueError("edge gather/pack requires equal-width 2D features")
        if senders.ndim != 1 or receivers.ndim != 1:
            raise ValueError("edge gather/pack requires one-dimensional edge indices")
        if senders.numel() != edges.shape[0] or receivers.numel() != edges.shape[0]:
            raise ValueError("edge gather/pack index length differs from edge count")
        edge_count, width = map(int, edges.shape)
        output = torch.empty(
            (edge_count, 3 * width), dtype=edges.dtype, device=edges.device
        )
        block = 256
        _edge_gather_pack_fwd[(triton.cdiv(output.numel(), block),)](
            edges,
            nodes,
            senders,
            receivers,
            output,
            edges.stride(0),
            edges.stride(1),
            nodes.stride(0),
            nodes.stride(1),
            edge_count,
            width,
            BLOCK=block,
        )
        ctx.save_for_backward(senders, receivers, row_ptr)
        ctx.edge_shape = tuple(edges.shape)
        ctx.edge_stride = tuple(edges.stride())
        ctx.node_shape = tuple(nodes.shape)
        ctx.node_stride = tuple(nodes.stride())
        ctx.max_row = int(max_row)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        senders, receivers, row_ptr = ctx.saved_tensors
        edge_count, width = ctx.edge_shape
        grad_edges = torch.empty_strided(
            ctx.edge_shape,
            ctx.edge_stride,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_nodes = torch.zeros(
            ctx.node_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        block = 256
        _edge_gather_pack_bwd[(triton.cdiv(edge_count * width, block),)](
            grad_output,
            senders,
            receivers,
            grad_edges,
            grad_nodes,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_edges.stride(0),
            grad_edges.stride(1),
            grad_nodes.stride(0),
            grad_nodes.stride(1),
            row_ptr.numel() - 1,
            edge_count,
            width,
            BLOCK=block,
        )
        sender_block = 64
        _edge_gather_pack_sender_csr_bwd[
            (row_ptr.numel() - 1, triton.cdiv(width, sender_block))
        ](
            grad_output,
            senders,
            row_ptr,
            grad_nodes,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_nodes.stride(0),
            grad_nodes.stride(1),
            width,
            ctx.max_row,
            grad_output.dtype == torch.float64,
            BLOCK=sender_block,
        )
        return grad_edges, grad_nodes, None, None, None, None


class FastEqEdgeGatherPack(nn.Module):
    """One forward kernel and one explicit-VJP kernel for ORB edge packing."""

    def __init__(self, row_ptr, max_row: int) -> None:
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, max_row: int) -> None:
        self.row_ptr = row_ptr
        self.max_row = int(max_row)

    def forward(self, edges, nodes, senders, receivers):
        return _EdgeGatherPackFunction.apply(
            edges, nodes, senders, receivers, self.row_ptr, self.max_row
        )
