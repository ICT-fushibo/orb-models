"""ORB inference candidate: hoist repeated node projections out of edge Linear.

No generic GEMM is replaced. Three native GEMMs feed one gather/add/SiLU
kernel. The explicit first-order VJP keeps native edge-GEMM-then-scatter order;
it does not assume receivers (or dummy senders) follow fixed CSR rows.
This is an ORB-specific algebraic experiment, not a ported FastEq kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def validate_inputs(nodes, edges, senders, receivers, weight, bias):
    if nodes.ndim != 2 or edges.ndim != 2 or edges.shape[1] != nodes.shape[1]:
        raise ValueError("ORB preprojection requires equal node/edge feature widths")
    if nodes.shape[0] < 1 or nodes.shape[1] < 1:
        raise ValueError("ORB preprojection requires nonempty nodes/features")
    if weight.ndim != 2 or weight.shape[1] != 3 * nodes.shape[1] or weight.shape[0] < 1:
        raise ValueError("ORB preprojection requires [hidden, 3 * channels] weight")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("ORB preprojection requires a hidden-width bias")
    tensors = (nodes, edges, senders, receivers, weight, bias)
    if len({t.device for t in tensors}) != 1:
        raise ValueError("ORB preprojection inputs must share one device")
    if nodes.dtype not in (torch.float32, torch.float64) or any(
        t.dtype != nodes.dtype for t in (edges, weight, bias)
    ):
        raise ValueError("ORB preprojection supports one float32/float64 dtype")
    for index in (senders, receivers):
        if index.ndim != 1 or index.numel() != edges.shape[0] or index.dtype not in (torch.int32, torch.int64):
            raise ValueError("ORB preprojection indices must have one integer entry per edge")


class EdgeLinearReference(nn.Module):
    def forward(self, nodes, edges, senders, receivers, weight, bias):
        return F.silu(F.linear(torch.cat((edges, nodes[senders], nodes[receivers]), 1), weight, bias))


def projected_preactivation(nodes, edges, senders, receivers, weight, bias):
    """Pure Torch algebra oracle for tests, not a production CPU/eager fallback."""
    width = nodes.shape[1]
    we, ws, wr = weight.split(width, dim=1)
    return F.linear(edges, we) + F.linear(nodes, ws)[senders] + F.linear(nodes, wr)[receivers] + bias


def linear_vjp(nodes, edges, senders, receivers, weight, grad_z, needs):
    """Native-order chain rule, not the reassociated (sum(dz) @ W) formula.

    Algebraically moving a GEMM past scatter changes FP32 rounding substantially
    for high-degree nodes. Keep ONE edge-wise full-width input-gradient GEMM,
    then the two independent advanced-index backward branches, as in native
    cat(edges, nodes[senders], nodes[receivers]) -> Linear.
    """
    width = nodes.shape[1]
    need_nodes, need_edges, _, _, need_weight, need_bias = needs
    dn = de = dw = db = None
    if need_nodes or need_edges:
        grad_features = grad_z @ weight
        if need_nodes:
            gs, gr = torch.zeros_like(nodes), torch.zeros_like(nodes)
            gs.index_put_((senders,), grad_features[:, width:2 * width], accumulate=True)
            gr.index_put_((receivers,), grad_features[:, 2 * width:], accumulate=True)
            # Do not merge both scatter branches into a single accumulation.
            dn = gs + gr
        if need_edges:
            de = grad_features[:, :width]
    if need_weight:
        # Only exhaustive setup/parameter tests use this path, not frozen MD.
        # Reconstruct linear inputs, not a full reference forward or backward.
        features = torch.cat((edges, nodes[senders], nodes[receivers]), dim=1)
        dw = grad_z.T @ features
    if need_bias:
        db = grad_z.sum(0)
    return dn, de, None, None, dw, db


class _Preproject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, senders, receivers, weight, bias):
        validate_inputs(nodes, edges, senders, receivers, weight, bias)
        if not nodes.is_cuda:
            raise RuntimeError("ORB preprojection candidate requires CUDA; no eager fallback")
        from .opt4_edge_preproject_kernels import gather_silu
        we, ws, wr = weight.split(nodes.shape[1], dim=1)
        # Weight views have fixed strides; ATen/cuBLAS handles their leading
        # dimensions. No per-step concatenation/packing/contiguous copy.
        pe, ps, pr = F.linear(edges, we), F.linear(nodes, ws), F.linear(nodes, wr)
        output, preactivation = gather_silu(pe, ps, pr, senders, receivers, bias)
        ctx.save_for_backward(nodes, edges, senders, receivers, weight, preactivation)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        nodes, edges, senders, receivers, weight, preactivation = ctx.saved_tensors
        # Native SiLU backward is already one CUDA kernel. Reuse it with our
        # saved preactivation; no forward recomputation or exception fallback.
        grad_z = torch.ops.aten.silu_backward.default(upstream, preactivation)
        return linear_vjp(nodes, edges, senders, receivers, weight, grad_z, ctx.needs_input_grad)


class EdgeLinearPreproject(nn.Module):
    def forward(self, nodes, edges, senders, receivers, weight, bias):
        return _Preproject.apply(nodes, edges, senders, receivers, weight, bias)
