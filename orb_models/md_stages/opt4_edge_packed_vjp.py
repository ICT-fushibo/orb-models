"""Experimental native-GEMM layout specialization of ORB edge-MLP input VJPs.

The measured preprojection forward is unchanged. Frozen weights are transposed
and packed BEFORE capture; input VJPs use native F.linear instead of row-major
matmul. The first layer omits node-gradient columns when only edges need a VJP.
No GEMM kernel, precision mode, reduction order across edges or checkpoint is
changed. This is ORB-specific code, inspired by FastEq shape/layout specialization,
not a copy of a FastEq ORB implementation (there is no such implementation).
"""
from __future__ import annotations

import weakref
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from .opt4_edge_preproject import validate_inputs, linear_vjp


PASS = "orb_edge_mlp_packed_vjp"


def capturing(weight):
    return weight.is_cuda and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class FrozenWeightPack(nn.Module):
    """One current frozen-weight cache, never a graph/shape bucket cache.

Differentiable parameter audits/gradcheck always pack their supplied weight,
including finite-difference perturbations that do not increment _version.
The frozen MD path reuses one snapshot; mutation is only supported in setup.
"""
    def __init__(self, edge_columns=None):
        super().__init__()
        self.edge_columns = edge_columns
        self.register_buffer("full", None, persistent=False)
        self.register_buffer("edge", None, persistent=False)
        self._source = None
        self._version_seen = None
        self.builds = 0

    def _pack(self, weight):
        if isinstance(weight, FakeTensor):
            # Layout inference only: CPU PyTorch cannot contiguous() a fake
            # CUDA view. Do not cache fake storage or initialize CUDA here.
            return (weight.new_empty((weight.shape[1], weight.shape[0])),
                    weight.new_empty((self.edge_columns, weight.shape[0]))
                    if self.edge_columns is not None else None)
        full = weight.detach().T.contiguous()
        edge = (weight.detach()[:, :self.edge_columns].T.contiguous()
                if self.edge_columns is not None else None)
        return full, edge

    def forward(self, weight):
        if weight.ndim != 2 or weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("packed VJP requires a float32/float64 matrix")
        if isinstance(weight, FakeTensor):
            return self._pack(weight)
        if weight.requires_grad:
            if capturing(weight):
                raise RuntimeError("packed VJP parameter audits must finish before capture; MD weights must be frozen")
            return self._pack(weight)
        if (self._source is None or self._source() is not weight
                or self._version_seen != weight._version
                or self.full.device != weight.device or self.full.dtype != weight.dtype):
            if capturing(weight):
                raise RuntimeError("packed VJP weights not prepared before CUDA Graph capture")
            self.full, self.edge = self._pack(weight)
            self._source, self._version_seen = weakref.ref(weight), weight._version
            self.builds += 1
        return self.full, self.edge

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self._source, self._version_seen = None, None
        return result

    def workspace_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.full, self.edge) if t is not None)


def packed_input_vjp(nodes, edges, senders, receivers, weight, grad_z, needs, full, edge):
    need_nodes, need_edges, _, _, need_weight, need_bias = needs
    dn = de = None
    width = nodes.shape[1]
    if need_nodes:
        # Preserve edge-wise GEMM -> two independent native index_put -> add.
        features = F.linear(grad_z, full)
        gs, gr = torch.zeros_like(nodes), torch.zeros_like(nodes)
        gs.index_put_((senders,), features[:, width:2*width], accumulate=True)
        gr.index_put_((receivers,), features[:, 2*width:], accumulate=True)
        dn = gs + gr
        if need_edges: de = features[:, :width]
    elif need_edges:
        # Same K reduction and dtype; do not compute/discard 2C node columns.
        de = F.linear(grad_z, edge)
    _, _, _, _, dw, db = linear_vjp(
        nodes, edges, senders, receivers, weight, grad_z,
        (False, False, False, False, need_weight, need_bias))
    return dn, de, None, None, dw, db


class _PackedPreproject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, senders, receivers, weight, bias, full, edge):
        validate_inputs(nodes, edges, senders, receivers, weight, bias)
        if not nodes.is_cuda:
            raise RuntimeError("ORB packed preprojection requires CUDA; no eager fallback")
        if isinstance(nodes, FakeTensor):
            output = edges.new_empty((edges.shape[0], weight.shape[0]))
            preactivation = torch.empty_like(output)
            ctx.save_for_backward(nodes, edges, senders, receivers, weight, preactivation, full, edge)
            return output
        from .opt4_edge_preproject_kernels import gather_silu
        we, ws, wr = weight.split(nodes.shape[1], dim=1)
        pe, ps, pr = F.linear(edges, we), F.linear(nodes, ws), F.linear(nodes, wr)
        output, preactivation = gather_silu(pe, ps, pr, senders, receivers, bias)
        ctx.save_for_backward(nodes, edges, senders, receivers, weight, preactivation, full, edge)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        nodes, edges, s, r, w, z, full, edge = ctx.saved_tensors
        dz = torch.ops.aten.silu_backward.default(upstream, z)
        return (*packed_input_vjp(nodes, edges, s, r, w, dz, ctx.needs_input_grad[:6], full, edge), None, None)


class PackedPreproject(nn.Module):
    def __init__(self, width, detail=None):
        super().__init__()
        self.pack = FrozenWeightPack(width)
        self.detail = detail

    def forward(self, nodes, edges, s, r, weight, bias):
        validate_inputs(nodes, edges, s, r, weight, bias)
        full, edge = self.pack(weight)
        if self.detail is not None and not capturing(weight):
            self.detail.update(packed_weight_bytes=self.pack.workspace_bytes(),
                               input_vjp_columns=3*nodes.shape[1] if nodes.requires_grad else nodes.shape[1],
                               packed_weight_stride=list(full.stride()),
                               original_weight_stride=list(weight.stride()))
        return _PackedPreproject.apply(nodes, edges, s, r, weight, bias, full, edge)


class _PackedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, packed):
        ctx.save_for_backward(x, weight, packed)
        ctx.has_bias = bias is not None
        return F.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, packed = ctx.saved_tensors
        dx = F.linear(grad, packed) if ctx.needs_input_grad[0] else None
        dw = grad.T @ x if ctx.needs_input_grad[1] else None
        db = grad.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


class LinearReference(nn.Module):
    def forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


class PackedLinear(nn.Module):
    def __init__(self, detail=None):
        super().__init__()
        self.pack = FrozenWeightPack()
        self.detail = detail

    def forward(self, x, weight, bias):
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("packed Linear requires [rows,input] and [output,input]")
        if x.device != weight.device or x.dtype != weight.dtype:
            raise ValueError("packed Linear dtype/device mismatch")
        if not x.is_cuda:
            raise RuntimeError("packed Linear candidate requires CUDA; no eager fallback")
        full, _ = self.pack(weight)
        if self.detail is not None and not capturing(weight):
            self.detail.update(packed_weight_bytes=self.pack.workspace_bytes(),
                               packed_weight_stride=list(full.stride()),
                               original_weight_stride=list(weight.stride()))
        return _PackedLinear.apply(x, weight, bias, full)


def _validate_vjp(actual, expected, args, index, probes):
    from md_benchmark.opt4_fx import assert_float32_vjp_reassociation_close
    assert_float32_vjp_reassociation_close(actual, expected, label="ORB packed native GEMM input/parameter VJP")
    return True


class PackedEdgeLinear(nn.Module):
    def __init__(self, linear, detail):
        super().__init__()
        from md_benchmark.opt4_fx import CheckedRegion
        self.weight, self.bias = linear.weight, linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self._opt4_packed_linear = CheckedRegion(LinearReference(), detail,
            candidate=PackedLinear(detail), vjp_validator=_validate_vjp, validate_runtime_vjp=True)

    def forward(self, x):
        return self._opt4_packed_linear(x, self.weight, self.bias)


def install_packed(model, report):
    from md_benchmark.opt4_fx import CheckedRegion
    from md_benchmark.opt4_registry import FusionSetupError, record
    from .opt4_edge_preproject import EdgeLinearPreproject
    modules = []
    for path, block in list(model.named_modules()):
        if type(block).__name__ != "AttentionInteractionNetwork": continue
        mlp = block._edge_mlp.mlp
        if (block.training or block._node_cond != "none" or block._edge_cond != "none"
                or not isinstance(mlp, nn.Sequential) or len(mlp) < 3
                or not isinstance(mlp[0], nn.Linear) or mlp[0].bias is None
                or mlp[0].in_features != 3*block.latent_dim
                or not isinstance(mlp[1], nn.SiLU) or mlp[1].inplace):
            raise FusionSetupError("ORB packed VJP requires unconditioned eval edge Linear(3C,H)+SiLU")
        if any(hasattr(block, name) for name in ("_opt4_edge_preproject", "_opt4_fasteq_edge_epilogue", "_opt4_fasteq_node_epilogue")):
            raise FusionSetupError("ORB candidate already installed; do not stack alternatives")
        def detail(name, boundary):
            return {"module": name, "boundary": boundary, "validated_shapes": 0,
                    "benchmark_requested": report.get("benchmark_boundaries", False),
                    "boundary_gate_max_ratio": {"forward": 1.05, "forward_vjp": .85},
                    "reference": "preproject-v1/native-remaining-edge-MLP",
                    "forward": "unchanged", "input_vjp": "native-linear-with-setup-packed-transposed-weight"}
        first = detail(path, "preproject-v1-to-packed-input-vjp")
        block._opt4_edge_preproject = CheckedRegion(EdgeLinearPreproject(), first,
            candidate=PackedPreproject(block.latent_dim, first),
            vjp_validator=_validate_vjp, validate_runtime_vjp=True)
        regions = [first]
        for i, layer in list(enumerate(mlp)):
            if i >= 2 and isinstance(layer, nn.Linear):
                current = detail(f"{path}._edge_mlp.mlp.{i}", "native-edge-linear-packed-input-vjp")
                mlp[i] = PackedEdgeLinear(layer, current)
                regions.append(current)
        if len(regions) == 1:
            raise FusionSetupError("ORB packed VJP did not match remaining edge MLP Linears")
        modules.append({"module": path, "regions": regions})
    if not modules: raise FusionSetupError("ORB packed VJP matched no AttentionInteractionNetwork")
    record(report, PASS, len(modules), "native-gemm-packed-input-vjp", modules=modules,
           retained_forward="orb_edge_linear_preproject_vjp", new_fused_kernels=0,
           optimization="setup weight layout + runtime-gradient-mask specialization, not a new GEMM kernel",
           first_layer_inactive_node_columns="omitted only when node VJP is not required",
           node_vjp="edge-wise GEMM then unchanged two index_put branches",
           replay_weight_pack=False, precision="unchanged-fp32-or-fp64-native-gemm",
           checkpoint_changed=False, backward_order="no sum-before-GEMM reassociation")
