from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from md_benchmark.md_route import MDConfig, MDRunRequest
from torch import nn

from orb_models import md_route
from orb_models.common.models import segment_ops
from orb_models.md_stages.opt2 import (
    CUDAGraphCapacityError,
    ModelOnlyCUDAGraphEvaluator,
    _ORBForceOnlyModel,
    _RealAtomPairRepulsion,
    _maximum_neighbors_per_atom,
    _validate_request,
    edge_capacity_from_probe,
    staticize_graph_inputs_,
)


def _request(*, backend: str = "model-only-cuda-graph") -> MDRunRequest:
    return MDRunRequest(
        model="orbv3",
        stage="opt2",
        model_path="orb-v3-conservative-inf-mpa-20250404.ckpt",
        atoms=Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        config=MDConfig(
            device="cuda:0",
            dtype="float64",
            steps=1,
            observation_steps=(1,),
        ),
        backend=backend,
        options={
            "model_variant": "orb-v3-conservative-inf-mpa",
            "edge_method": "knn_alchemi",
            "half_supercell": False,
            "model_precision": "float32-highest",
        },
    )


def test_opt2_route_dispatches_without_loading_model(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_run_optimized_stage(request, *, module_prefix):
        seen["request"] = request
        seen["module_prefix"] = module_prefix
        return sentinel

    monkeypatch.setattr(md_route, "run_optimized_stage", fake_run_optimized_stage)
    request = _request()
    assert md_route.run_md(request) is sentinel
    assert seen == {
        "request": request,
        "module_prefix": "orb_models.md_stages",
    }


def test_opt2_rejects_non_graph_backend_before_loading_model():
    with pytest.raises(ValueError, match="model-only-cuda-graph"):
        _validate_request(_request(backend="gpu-resident"))


@pytest.mark.parametrize("option", ["compile", "torch_compile", "amp", "tf32"])
def test_opt2_rejects_extra_acceleration_options(option):
    request = _request()
    request.options[option] = True
    with pytest.raises(ValueError, match="forbids extra acceleration"):
        _validate_request(request)


def test_edge_capacity_has_headroom_and_alignment():
    assert edge_capacity_from_probe(128, margin=0.25, edge_step=128) == 256
    assert edge_capacity_from_probe(100, margin=0.0, edge_step=16) == 112


def test_probe_maximum_neighbors_uses_orb_sender_axis():
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 3]], dtype=torch.long
    )
    assert _maximum_neighbors_per_atom(edge_index, num_atoms=4) == 3


@pytest.mark.parametrize("reduction", ["sum", "mean", "max"])
def test_single_graph_aggregation_avoids_dynamic_segment_construction(
    monkeypatch, reduction
):
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *_args, **_kwargs: pytest.fail("singleton path called torch.arange"),
    )
    values = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    actual = segment_ops.aggregate_nodes(
        values, torch.tensor([2]), reduction=reduction
    )
    expected = getattr(values, "amax" if reduction == "max" else reduction)(
        dim=0, keepdim=True
    )
    torch.testing.assert_close(actual, expected)


def test_staticize_uses_fixed_buffers_and_dummy_padding():
    static_positions = torch.empty(3, 3)
    senders = torch.empty(5, dtype=torch.long)
    receivers = torch.empty(5, dtype=torch.long)
    shifts = torch.empty(5, 3)
    addresses = tuple(
        value.data_ptr() for value in (static_positions, senders, receivers, shifts)
    )
    real_positions = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    unit_shifts = torch.tensor([[0.0, 0.0, 0.0]] * 3)
    padding_shift = torch.tensor([2.0, 0.0, 0.0])

    assert staticize_graph_inputs_(
        static_positions,
        senders,
        receivers,
        shifts,
        real_positions,
        edge_index,
        unit_shifts,
        dummy_index=2,
        padding_unit_shift=padding_shift,
    ) == 3
    assert addresses == tuple(
        value.data_ptr() for value in (static_positions, senders, receivers, shifts)
    )
    torch.testing.assert_close(static_positions[:2], real_positions)
    torch.testing.assert_close(static_positions[2], torch.zeros(3))
    torch.testing.assert_close(senders, torch.tensor([0, 1, 0, 2, 2]))
    torch.testing.assert_close(receivers, torch.tensor([1, 0, 0, 2, 2]))
    torch.testing.assert_close(shifts[3:], padding_shift.expand(2, -1))


def test_staticize_overflow_fails_before_writing_or_truncating():
    static_positions = torch.full((3, 3), -7.0)
    senders = torch.full((2,), -7, dtype=torch.long)
    receivers = senders.clone()
    shifts = torch.full((2, 3), -7.0)
    before = tuple(value.clone() for value in (static_positions, senders, receivers, shifts))
    with pytest.raises(CUDAGraphCapacityError, match="required=3, capacity=2"):
        staticize_graph_inputs_(
            static_positions,
            senders,
            receivers,
            shifts,
            torch.zeros(2, 3),
            torch.tensor([[0, 1, 0], [1, 0, 0]]),
            torch.zeros(3, 3),
            dummy_index=2,
            padding_unit_shift=torch.tensor([2.0, 0.0, 0.0]),
        )
    for actual, expected in zip(
        (static_positions, senders, receivers, shifts), before, strict=True
    ):
        torch.testing.assert_close(actual, expected)


class _MeanPairRepulsion(nn.Module):
    node_aggregation = "mean"

    def forward(self, batch):
        return {"energy": torch.tensor([6.0])}


@pytest.mark.parametrize("reduction, expected", [("mean", 8.0), ("sum", 6.0)])
def test_pair_repulsion_wrapper_preserves_real_atom_reduction(reduction, expected):
    base = _MeanPairRepulsion()
    base.node_aggregation = reduction
    wrapped = _RealAtomPairRepulsion(base, n_real=3)
    assert wrapped(SimpleNamespace())["energy"].item() == expected


def test_force_only_wrapper_source_avoids_capture_unsafe_rotation():
    source = inspect.getsource(_ORBForceOnlyModel.forward)
    assert "matrix_exp" not in source
    assert "compute_differentiable_edge_vectors" not in source
    assert "torch.autograd.grad" in source
    assert "interaction_energy.sum()" in source


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def test_production_replay_never_recaptures_or_falls_back():
    evaluator = object.__new__(ModelOnlyCUDAGraphEvaluator)
    evaluator.captured = True
    evaluator.cuda_graph = _FakeGraph()
    evaluator.edge_capacity = 4
    evaluator.profiler = SimpleNamespace(phase=lambda _name: contextlib.nullcontext())
    evaluator.production_calls = 0
    evaluator.production_replays = 0
    evaluator.total_replays = 0
    evaluator.capacity_misses = 0
    evaluator.min_real_edges = None
    evaluator.max_real_edges = None
    evaluator.initial_max_neighbors_per_atom = None
    evaluator.max_neighbors_per_atom = None
    evaluator.track_neighbor_capacity = False
    evaluator.static_forces = torch.ones(2, 3)
    evaluator.static_energy = torch.tensor([2.0])
    evaluator._build_real_inputs = lambda positions: (
        positions.to(torch.float32),
        torch.tensor([[0, 1], [1, 0]]),
        torch.zeros(2, 3),
    )
    evaluator._staticize = lambda *_args: 2
    evaluator._input_addresses = lambda: (("positions", 11), ("senders", 12))
    evaluator._capture_input_addresses = (("positions", 11), ("senders", 12))

    forces, energy, stress = evaluator(torch.zeros(2, 3, dtype=torch.float64))
    assert evaluator.cuda_graph.replays == 1
    assert evaluator.production_calls == evaluator.production_replays == 1
    assert forces.dtype == torch.float64
    assert energy.dtype == torch.float64
    assert stress is None


def test_production_replay_asserts_fixed_input_addresses():
    source = inspect.getsource(ModelOnlyCUDAGraphEvaluator.__call__)
    capture_source = inspect.getsource(ModelOnlyCUDAGraphEvaluator.capture)

    assert "_input_addresses()" in source
    assert "_capture_input_addresses" in capture_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_buffers_keep_addresses_on_cuda():
    device = torch.device("cuda:0")
    static_positions = torch.empty(3, 3, device=device)
    senders = torch.empty(4, dtype=torch.long, device=device)
    receivers = torch.empty(4, dtype=torch.long, device=device)
    shifts = torch.empty(4, 3, device=device)
    addresses = tuple(
        value.data_ptr() for value in (static_positions, senders, receivers, shifts)
    )
    staticize_graph_inputs_(
        static_positions,
        senders,
        receivers,
        shifts,
        torch.zeros(2, 3, device=device),
        torch.tensor([[0, 1], [1, 0]], device=device),
        torch.zeros(2, 3, device=device),
        dummy_index=2,
        padding_unit_shift=torch.tensor([2.0, 0.0, 0.0], device=device),
    )
    assert addresses == tuple(
        value.data_ptr() for value in (static_positions, senders, receivers, shifts)
    )
