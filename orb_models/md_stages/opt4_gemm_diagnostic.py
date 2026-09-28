"""Opt-in, setup-only inventory of the actual ORB energy/force GEMMs.

No hooks or dispatch modes survive this function. This is not an Opt4 pass.
Backward is identified by the autograd graph task, not by guessing kernel names.
Module provenance comes from frozen weight/buffer storage, including sliced W.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_spec(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), storage_offset=tensor.storage_offset())


def storage_key(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


class GemmInventory(TorchDispatchMode):
    """Retain one input example per op/layout/phase, bounded by storage bytes."""

    def __init__(self, model, *, max_bytes=2 * 1024**3, cuda=True):
        super().__init__()
        self.rows, self.inputs, self.owners = {}, {}, {}
        self.max_bytes, self.retained_bytes, self.cuda = max_bytes, 0, cuda
        if not hasattr(torch._C, "_current_graph_task_id"):
            raise RuntimeError("GEMM diagnostic requires autograd graph-task attribution")
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if value.numel():
                self.owners.setdefault(storage_key(value), []).append(name)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            return func(*args, **kwargs)
        phase = "force_vjp" if torch._C._current_graph_task_id() >= 0 else "forward"
        op = "mm" if func == torch.ops.aten.mm.default else "addmm"
        specs = [tensor_spec(t) for t in args]
        # Offset is provenance, not part of a GEMM's shape/stride signature.
        signature = dict(op=op, phase=phase, inputs=[{k: v for k, v in s.items()
                         if k != "storage_offset"} for s in specs], kwargs=kwargs)
        key = json.dumps(signature, sort_keys=True)
        if key not in self.rows:
            size = sum(t.untyped_storage().nbytes() for t in args)
            if self.retained_bytes + size > self.max_bytes:
                raise RuntimeError("GEMM diagnostic operand budget exceeded; increase max_bytes explicitly")
            # Preserve padded leading dimensions; copying is outside event timing.
            copies = []
            for t in args:
                clone = torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=t.dtype)
                clone.copy_(t.detach())
                copies.append(clone)
            self.retained_bytes += size
            self.inputs[key] = tuple(copies)
            a, b = args[-2:]
            self.rows[key] = dict(id=f"gemm-{len(self.rows):03d}", **signature,
                m=a.shape[0], n=b.shape[1], k=a.shape[1],
                flops=2*a.shape[0]*a.shape[1]*b.shape[1], calls_per_force=0,
                provenance=[], events=[])
        row = self.rows[key]
        row["calls_per_force"] += 1
        row["provenance"].append([
            dict(owners=self.owners.get(storage_key(t), []), **spec)
            for t, spec in zip(args, specs)])
        if not self.cuda:
            return func(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.nvtx.range(f"orb_setup_gemm::{row['id']}::{phase}"):
            start.record()
            result = func(*args, **kwargs)
            end.record()
        row["events"].append((start, end))
        return result


def collect_runner_gemms(runner, positions, output, *, passes=(), max_bytes=2 * 1024**3):
    from md_benchmark.opt4_gemm_benchmark import native_call, benchmark_native

    if positions.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("ORB GEMM inventory must run on CUDA before capture")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Complete fusion validation, lazy packs and Triton compilation before tracing.
    runner(positions)
    torch.cuda.synchronize(positions.device)
    inventory = GemmInventory(runner.force_only_model, max_bytes=max_bytes)
    with inventory:
        runner(positions)
    torch.cuda.synchronize(positions.device)
    if not inventory.rows:
        raise RuntimeError("No actual ORB GEMMs were observed; diagnostic is incomplete")
    rows = []
    for key, row in inventory.rows.items():
        row["setup_event_ms"] = [a.elapsed_time(b) for a, b in row.pop("events")]
        row["native_graph"] = benchmark_native(native_call(row, inventory.inputs[key]))
        row["estimated_ms_per_force"] = row["calls_per_force"] * row["native_graph"]["median_ms"]
        rows.append(row)
    rows.sort(key=lambda r: r["estimated_ms_per_force"], reverse=True)
    winner = rows[0]
    key = next(k for k, row in inventory.rows.items() if row["id"] == winner["id"])
    # CPU serialization is outside all timing; reload reconstructs original strides.
    torch.save(dict(schema=1, row=winner,
                    operands=[v.cpu().contiguous() for v in inventory.inputs[key]]),
               output / "hottest.pt")
    report = dict(schema=1, diagnostic_only=True, model="orbv3", rows=rows,
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(positions.device),
        candidate_passes=list(passes),
        edge_capacity=runner.edge_capacity,
        neighbor_capacities=list(runner.neighbor_capacities),
        retained_operand_budget_bytes=max_bytes, retained_operand_bytes=inventory.retained_bytes,
        hottest_id=winner["id"], exported="hottest.pt",
        limitations=["Isolated graph timings estimate a ranking; they are not additive production timings.",
            "Setup events include instrumentation effects; do not compare them as MD performance.",
            "Weight-storage ownership is provenance, not an exact module execution trace.",
            "Each backward row is one GEMM from the real force VJP, not a full boundary VJP benchmark.",
            "No algorithm is installed into the MD model by this diagnostic."])
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
