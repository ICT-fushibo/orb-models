"""ORB GNS candidates: keep Linear/hidden activation and attention intact."""
import torch
from md_benchmark.opt4_fx import checked_boundary
from md_benchmark.opt4_ops import gather_pack,norm_add
from md_benchmark.opt4_registry import FusionSetupError, record


def norm_reference(x,r,w,b,eps,rms):
    if rms:
        return torch.nn.functional.rms_norm(x,(x.shape[-1],),w,eps)+b+r
    return torch.nn.functional.layer_norm(x,(x.shape[-1],),w,b,eps)+r


def install(model, passes, report):
    matched = {p: [] for p in passes}
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "AttentionInteractionNetwork":
            continue
        if "gns_gather_pack" in passes:
            module._opt4_gather_pack = True
            detail={"module":path,"benchmark_requested":report.get("benchmark_boundaries",False)}
            module._opt4_gather_op=checked_boundary(lambda e,x,s,d:torch.cat((e,x[s],x[d]),dim=1),gather_pack,detail)
            matched["gns_gather_pack"].append(detail)
        if "gns_norm_residual" in passes:
            mlp = module._node_mlp
            norm = getattr(mlp, "layer_norm", None)
            if not hasattr(mlp, "mlp") or not isinstance(norm, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
                continue
            if len(norm.normalized_shape) != 1:
                continue
            if isinstance(norm, torch.nn.RMSNorm) and not hasattr(torch.ops.aten, "_fused_rms_norm_backward"):
                raise FusionSetupError("ORB Opt4 RMSNorm requires the installed native RMSNorm backward (PyTorch 2.11); no dependency changes performed")
            parameter = next(module.parameters())
            channels = norm.normalized_shape[0]
            # Non-affine normalization keeps persistent, setup-only constants.
            module.register_buffer("_opt4_norm_one", torch.ones(channels,device=parameter.device,dtype=parameter.dtype), persistent=False)
            module.register_buffer("_opt4_norm_zero", torch.zeros(channels,device=parameter.device,dtype=parameter.dtype), persistent=False)
            module._opt4_norm_residual = True
            detail={"module":path,"benchmark_requested":report.get("benchmark_boundaries",False)}
            detail.update(normalization=type(norm).__name__, channels=channels,
                          rms_forward="float4-accumulation-rsqrtf; unchanged dtype",
                          rms_backward="native-aten", fusion_scope="forward-only" if isinstance(norm, torch.nn.RMSNorm) else "forward-and-backward")
            module._opt4_norm_op=checked_boundary(norm_reference,lambda x,r,w,b,eps,rms:norm_add(x,r,w,b,eps,rms)[0],detail)
            matched["gns_norm_residual"].append(detail)
    for p, paths in matched.items():
        record(report,p,len(paths),"triton-explicit-autograd",modules=paths,
               gemm="unchanged",attention="unchanged",numerical_gate="pending CUDA operator and whole-model parity")
