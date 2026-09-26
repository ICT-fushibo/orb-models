"""ORB inference candidate: hoist repeated node projections out of edge Linear.

No generic GEMM is replaced. Three native GEMMs feed one gather/add/SiLU
kernel. The explicit first-order VJP uses native dynamic index_add reductions;
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
    """Exact chain rule; frozen parameter branches never execute during MD."""
    width = nodes.shape[1]
    we, ws, wr = weight.split(width, dim=1)
    need_nodes, need_edges, _, _, need_weight, need_bias = needs
    dn = de = dw = db = None
    if need_nodes or need_weight:
        # Same dynamic reduction semantics as native gather backward. In
        # particular padding sender indices point to sinks, NOT slot centres.
        shape = (nodes.shape[0], grad_z.shape[1])
        gs = grad_z.new_zeros(shape).index_add_(0, senders, grad_z)
        gr = grad_z.new_zeros(shape).index_add_(0, receivers, grad_z)
        if need_nodes:
            dn = gs @ ws + gr @ wr
        if need_weight:
            dw = torch.cat((grad_z.T @ edges, gs.T @ nodes, gr.T @ nodes), dim=1)
    if need_edges:
        de = grad_z @ we
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
        from .opt4_edge_preproject_kernels import silu_vjp
        nodes, edges, senders, receivers, weight, preactivation = ctx.saved_tensors
        grad_z = silu_vjp(preactivation, upstream)
        return linear_vjp(nodes, edges, senders, receivers, weight, grad_z, ctx.needs_input_grad)


class EdgeLinearPreproject(nn.Module):
    def forward(self, nodes, edges, senders, receivers, weight, bias):
        return _Preproject.apply(nodes, edges, senders, receivers, weight, bias)
