from __future__ import annotations

import inspect

import pytest
import torch
from ase import Atoms
from md_benchmark.md_route import MDConfig, MDRunRequest

from orb_models import md_route
from orb_models.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    NoseHooverChainIntegrator,
)
from orb_models.md_stages.opt3 import (
    WholeStepCUDAGraphRunner,
    _FixedShapeORBNeighborBuilder,
    _validate_request,
    sink_pad_neighbor_matrix,
    whole_step_in_place_,
)


def _request(*, backend: str = "whole-step-cuda-graph") -> MDRunRequest:
    return MDRunRequest(
        model="orbv3",
        stage="opt3",
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


def test_opt3_route_dispatches_without_loading_model(monkeypatch):
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


def test_opt3_rejects_non_whole_step_backend_before_loading_model():
    with pytest.raises(ValueError, match="whole-step-cuda-graph"):
        _validate_request(_request(backend="model-only-cuda-graph"))


def test_sink_padding_is_fixed_shape_far_shifted_and_distributed():
    neighbor_matrix = torch.tensor([[1, -1, -1], [0, -1, -1]])
    shifts = torch.zeros(2, 3, 3)
    far_shift = torch.tensor([3.0, 0.0, 0.0])

    senders, receivers, unit_shifts, valid = sink_pad_neighbor_matrix(
        neighbor_matrix,
        shifts,
        n_real=2,
        n_dummy=3,
        padding_unit_shift=far_shift,
    )

    torch.testing.assert_close(valid, torch.tensor([True, False, False, True, False, False]))
    torch.testing.assert_close(senders, torch.tensor([0, 3, 4, 1, 3, 4]))
    torch.testing.assert_close(receivers, torch.tensor([1, 3, 4, 0, 3, 4]))
    torch.testing.assert_close(unit_shifts[~valid], far_shift.expand(4, -1))
    assert senders.shape == receivers.shape == (6,)
    assert unit_shifts.shape == (6, 3)


class _HarmonicEvaluator:
    def __call__(self, positions):
        return -positions, 0.5 * positions.square().sum(), None


@pytest.mark.parametrize("integrator_name", ["berendsen", "nose_hoover_chain"])
def test_whole_step_in_place_matches_existing_gpu_integrator(integrator_name):
    masses = torch.tensor([1.0, 2.0], dtype=torch.float64)
    kwargs = {
        "timestep_fs": 0.25,
        "temperature_k": 300.0,
        "thermostat_time_fs": 25.0,
    }
    if integrator_name == "berendsen":
        reference_integrator = BerendsenIntegrator(
            masses, degrees_of_freedom=6, **kwargs
        )
        capture_integrator = BerendsenIntegrator(
            masses, degrees_of_freedom=6, **kwargs
        )
    else:
        reference_integrator = NoseHooverChainIntegrator(masses, **kwargs)
        capture_integrator = NoseHooverChainIntegrator(masses, **kwargs)

    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.8, -0.1, 0.5]], dtype=torch.float64
    )
    momenta = torch.tensor(
        [[0.01, -0.02, 0.03], [-0.03, 0.02, -0.01]], dtype=torch.float64
    )
    evaluator = _HarmonicEvaluator()
    forces, energy, _ = evaluator(positions)
    reference = GPUMDState(
        positions.clone(), momenta.clone(), forces.clone(), energy.clone()
    )
    captured = GPUMDState(
        positions.clone(), momenta.clone(), forces.clone(), energy.clone()
    )
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (
            captured.positions,
            captured.momenta,
            captured.forces,
            captured.potential_energy,
        )
    )

    for _ in range(3):
        reference_integrator.step(reference, evaluator)
        whole_step_in_place_(captured, capture_integrator, evaluator)

    torch.testing.assert_close(captured.positions, reference.positions)
    torch.testing.assert_close(captured.momenta, reference.momenta)
    torch.testing.assert_close(captured.forces, reference.forces)
    torch.testing.assert_close(captured.potential_energy, reference.potential_energy)
    assert addresses == tuple(
        tensor.data_ptr()
        for tensor in (
            captured.positions,
            captured.momenta,
            captured.forces,
            captured.potential_energy,
        )
    )
    if integrator_name == "nose_hoover_chain":
        torch.testing.assert_close(capture_integrator.eta, reference_integrator.eta)
        torch.testing.assert_close(capture_integrator.p_eta, reference_integrator.p_eta)


@pytest.mark.parametrize("integrator_name", ["berendsen", "nose_hoover_chain"])
def test_advance_zero_only_evaluates_initial_force(integrator_name):
    masses = torch.tensor([1.0, 2.0], dtype=torch.float64)
    kwargs = {
        "timestep_fs": 0.25,
        "temperature_k": 300.0,
        "thermostat_time_fs": 25.0,
    }
    if integrator_name == "berendsen":
        integrator = BerendsenIntegrator(masses, degrees_of_freedom=6, **kwargs)
    else:
        integrator = NoseHooverChainIntegrator(masses, **kwargs)
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.8, -0.1, 0.5]], dtype=torch.float64
    )
    momenta = torch.tensor(
        [[0.01, -0.02, 0.03], [-0.03, 0.02, -0.01]], dtype=torch.float64
    )
    state = GPUMDState(
        positions.clone(),
        momenta.clone(),
        torch.zeros_like(positions),
        torch.zeros((), dtype=torch.float64),
    )
    thermostat_before = (
        None
        if integrator_name == "berendsen"
        else (integrator.eta.clone(), integrator.p_eta.clone())
    )

    whole_step_in_place_(
        state,
        integrator,
        _HarmonicEvaluator(),
        torch.zeros((), dtype=torch.float64),
    )

    torch.testing.assert_close(state.positions, positions)
    torch.testing.assert_close(state.momenta, momenta)
    torch.testing.assert_close(state.forces, -positions)
    torch.testing.assert_close(state.potential_energy, 0.5 * positions.square().sum())
    if thermostat_before is not None:
        torch.testing.assert_close(integrator.eta, thermostat_before[0])
        torch.testing.assert_close(integrator.p_eta, thermostat_before[1])


def test_whole_step_source_contains_builder_model_and_state_update():
    builder = inspect.getsource(WholeStepCUDAGraphRunner._fixed_builder)
    fixed_builder = inspect.getsource(_FixedShapeORBNeighborBuilder.build)
    body = inspect.getsource(whole_step_in_place_)
    capture = inspect.getsource(WholeStepCUDAGraphRunner.capture)

    assert "self.fixed_neighbor_builder.build" in builder
    assert "nva_neighbor_list" not in builder
    assert "torch.topk" in fixed_builder
    assert "sink_pad_neighbor_matrix" in builder
    assert body.count("evaluator(evaluation_positions)") == 2
    assert "state.positions.copy_" in body
    assert "whole_step_in_place_" in capture
    assert "torch.cuda.graph" in capture


def test_per_centre_capacity_has_device_overflow_assertion():
    source = inspect.getsource(WholeStepCUDAGraphRunner._fixed_builder)
    assert "num_neighbors.max() <= self.slots_per_atom" in source
    assert "torch._assert_async" in source
    assert "self.capacity_misses.add_" in source
    assert "self.maximum_required_neighbors.copy_" in source


def test_fixed_candidate_builder_matches_simple_periodic_topology():
    builder = _FixedShapeORBNeighborBuilder(
        num_atoms=2,
        cell=torch.eye(3, dtype=torch.float32) * 8.0,
        pbc=torch.ones(3, dtype=torch.bool),
        cutoff=3.0,
        neighbors_per_atom=2,
    )
    matrix, counts, shifts = builder.build(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )

    assert counts.tolist() == [1, 1]
    assert matrix.tolist() == [[1, -1], [0, -1]]
    assert torch.equal(shifts, torch.zeros_like(shifts))


def test_skin_builder_matches_full_search_with_per_atom_cap():
    positions = torch.tensor([[0.1, 0.0, 0.0], [4.8, 0.0, 0.0]])
    options = dict(
        num_atoms=2,
        cell=torch.eye(3) * 5.0,
        pbc=torch.ones(3, dtype=torch.bool),
        cutoff=1.0,
        neighbors_per_atom=2,
        neighbor_capacities=[1, 2],
    )
    full = _FixedShapeORBNeighborBuilder(**options)
    skin = _FixedShapeORBNeighborBuilder(
        **options, verlet_skin=0.5, verlet_candidate_capacity=4
    )
    skin.initialize_skin(positions)

    full_output = full.build(positions)
    skin_output = skin.build(positions)
    for actual, expected in zip(skin_output, full_output):
        torch.testing.assert_close(actual, expected)
    assert skin.edge_capacity == 3

    moved_wrapped = positions.clone()
    moved_wrapped[0, 0] = 4.9
    image_offsets = torch.zeros_like(positions)
    image_offsets[0, 0] = -1.0
    for actual, expected in zip(
        skin.build(moved_wrapped, image_offsets), full.build(moved_wrapped)
    ):
        torch.testing.assert_close(actual, expected)


def test_total_capacity_uses_single_guard_and_initial_force_is_timed():
    init_source = inspect.getsource(WholeStepCUDAGraphRunner.__init__)
    run_source = inspect.getsource(__import__(
        "orb_models.md_stages.opt3", fromlist=["run_md"]
    ).run_md)
    assert "self.capacity_alignment = 8" in init_source
    assert "self.capacity_guard_slots = 0" in init_source
    assert "self.capacity_total_floor" in init_source
    assert "self.capacity_initial_safe_slots" in init_source
    assert 'profiler.phase("initial_force")' in run_source
    assert "expected_replays = config.steps + 1" in run_source


def test_capture_validates_full_state_and_nhc_state_against_eager_step():
    source = inspect.getsource(WholeStepCUDAGraphRunner.capture)
    assert "integrator.step(reference_state, self)" in source
    for field in (
        "whole_step_position_max_abs_error",
        "whole_step_momentum_max_abs_error",
        "whole_step_force_max_abs_error",
        "whole_step_energy_abs_error",
        "whole_step_eta_max_abs_error",
        "whole_step_p_eta_max_abs_error",
    ):
        assert field in source
    assert "non-finite state" in source
