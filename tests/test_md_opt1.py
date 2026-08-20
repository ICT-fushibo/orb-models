from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

from md_benchmark.md_route import MDConfig, MDRunRequest
from orb_models.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    NoseHooverChainIntegrator,
    _snapshot,
    _validate_request,
)


class _ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, forces: np.ndarray) -> None:
        super().__init__()
        self.forces = forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": self.forces.copy()}


class _ConstantEvaluator:
    def __init__(self, forces: torch.Tensor) -> None:
        self.forces = forces

    def __call__(self, positions):
        return self.forces.to(positions), positions.new_zeros(()), None


@pytest.mark.parametrize("integrator_name", ["berendsen", "nose_hoover_chain"])
def test_gpu_integrators_match_one_ase_step_on_cpu(integrator_name: str):
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]])
    momenta = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    forces = np.array([[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]])
    masses = np.array([12.0, 16.0])
    atoms = Atoms("CO", positions=positions, masses=masses)
    atoms.set_momenta(momenta)
    atoms.calc = _ConstantForceCalculator(forces)
    if integrator_name == "berendsen":
        ase_md = NVTBerendsen(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            taut=100.0 * units.fs,
            fixcm=True,
        )
        gpu_integrator = BerendsenIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
            degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
        )
    else:
        ase_md = NoseHooverChainNVT(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            tdamp=100.0 * units.fs,
        )
        gpu_integrator = NoseHooverChainIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
        )
    ase_md.run(1)
    state = GPUMDState(
        torch.tensor(positions, dtype=torch.float64),
        torch.tensor(momenta, dtype=torch.float64),
    )
    gpu_integrator.step(
        state, _ConstantEvaluator(torch.tensor(forces, dtype=torch.float64))
    )
    np.testing.assert_allclose(state.positions.numpy(), atoms.positions, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(state.momenta.numpy(), atoms.get_momenta(), rtol=1e-13, atol=1e-13)


def test_opt1_rejects_non_gpu_resident_backend_before_loading_model():
    request = MDRunRequest(
        model="orbv3",
        stage="opt1",
        model_path="orb-v3-conservative-inf-mpa-20250404.ckpt",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        config=MDConfig(device="cuda:0", dtype="float64", steps=1, observation_steps=(1,)),
        backend="eager",
        options={"model_variant": "orb-v3-conservative-inf-mpa"},
    )
    with pytest.raises(ValueError, match="gpu-resident"):
        _validate_request(request)


def test_opt1_rejects_cpu_neighbor_builder():
    request = MDRunRequest(
        model="orbv3",
        stage="opt1",
        model_path="orb-v3-conservative-inf-mpa-20250404.ckpt",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        config=MDConfig(device="cuda:0", dtype="float64", steps=1, observation_steps=(1,)),
        backend="gpu-resident",
        options={
            "model_variant": "orb-v3-conservative-inf-mpa",
            "edge_method": "knn_scipy",
        },
    )
    with pytest.raises(ValueError, match="knn_alchemi"):
        _validate_request(request)


def test_opt1_validates_periodic_cell_before_gpu_model_loading():
    request = MDRunRequest(
        model="orbv3",
        stage="opt1",
        model_path="orb-v3-conservative-inf-mpa-20250404.ckpt",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], pbc=True),
        config=MDConfig(device="cuda:0", dtype="float64", steps=1, observation_steps=(1,)),
        backend="gpu-resident",
        options={"model_variant": "orb-v3-conservative-inf-mpa"},
    )
    with pytest.raises(ValueError, match="cell.*zeros"):
        _validate_request(request)


def test_matbench_snapshot_contains_initial_step_and_stress():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True)
    state = GPUMDState(
        positions=torch.tensor(atoms.positions, dtype=torch.float64),
        momenta=torch.zeros(2, 3, dtype=torch.float64),
        forces=torch.ones(2, 3, dtype=torch.float64),
        potential_energy=torch.tensor(-1.25, dtype=torch.float64),
        stress=torch.eye(3, dtype=torch.float64),
    )
    frame = _snapshot(atoms, state, step=0, require_stress=True)
    assert frame.info["md_step"] == 0
    assert frame.get_potential_energy() == -1.25
    np.testing.assert_allclose(frame.get_forces(), np.ones((2, 3)))
    np.testing.assert_allclose(frame.get_stress(voigt=False), np.eye(3))
