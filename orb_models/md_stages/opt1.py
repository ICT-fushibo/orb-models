"""ORBv3 Opt1: eager GPU-resident molecular dynamics.

Only state placement and the MD engine change in this stage.  ORB remains an
uncompiled eager model, its existing Alchemi neighbor builder remains in use,
and neither CUDA Graph nor model-specific fused kernels are enabled.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import ase.io
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from md_benchmark.md_route import (
    MDObservation,
    MDRunRequest,
    MDRunResult,
    validate_result,
)

from orb_models.md_route import (
    _DEFAULT_MODEL_VARIANT,
    _MODEL_LOADERS,
    _normalise_variant,
    _option,
    _variant_in_filename,
)


_FOURTH_ORDER_COEFFS = (
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
    -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
)


class _Evaluator(Protocol):
    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor | None]: ...


@dataclass
class GPUMDState:
    """Mutable FP64 MD state; all tensors remain on one CUDA device."""

    positions: Tensor
    momenta: Tensor
    forces: Tensor | None = None
    potential_energy: Tensor | None = None
    stress: Tensor | None = None

    def clone(self) -> GPUMDState:
        return GPUMDState(
            positions=self.positions.clone(),
            momenta=self.momenta.clone(),
            forces=None if self.forces is None else self.forces.clone(),
            potential_energy=(
                None if self.potential_energy is None else self.potential_energy.clone()
            ),
            stress=None if self.stress is None else self.stress.clone(),
        )

    def restore_(self, other: GPUMDState) -> None:
        self.positions.copy_(other.positions)
        self.momenta.copy_(other.momenta)
        self.forces = None if other.forces is None else other.forces.clone()
        self.potential_energy = (
            None if other.potential_energy is None else other.potential_energy.clone()
        )
        self.stress = None if other.stress is None else other.stress.clone()


class OrbTorchSimEvaluator:
    """Evaluate eager ORB on persistent TorchSim data without leaving CUDA."""

    def __init__(
        self,
        atoms: Atoms,
        model_path: str,
        *,
        variant: str,
        device: torch.device,
        max_num_neighbors: int | None,
        compute_stress: bool,
    ) -> None:
        try:
            import torch_sim as ts
        except ImportError as exc:
            raise ImportError(
                "ORBv3 Opt1 requires torch-sim-atomistic>=0.6.0; install "
                "orb-models with `python -m pip install -e '.[torchsim]'`"
            ) from exc

        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

        loader = getattr(pretrained, _MODEL_LOADERS[variant])
        model, adapter = loader(
            weights_path=model_path,
            device=device,
            precision="float32-highest",
            compile=False,
        )
        self.model = OrbTorchSimModel(
            model,
            adapter,
            edge_method="knn_alchemi",
            max_num_neighbors=max_num_neighbors,
            device=device,
            dtype=torch.float32,
            graph_construction_dtype=torch.float32,
            static_alchemi_neighbor_list=True,
        )
        self.model.eval()
        self.sim_state = ts.io.atoms_to_state([atoms], device, dtype=torch.float64)
        self.device = device
        self.num_atoms = len(atoms)
        self.max_num_neighbors = max_num_neighbors or adapter.max_num_neighbors
        self.compute_stress = compute_stress
        # The one allowed calibration sync happens before warmup/timing. All
        # subsequent neighbor builds reuse the fixed Alchemi matrix capacity.
        self.model.prepare_neighbor_list(self.sim_state)

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        if positions.device != self.device or positions.dtype != torch.float64:
            raise ValueError("ORB Opt1 positions must be FP64 tensors on the selected CUDA device")
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"ORB Opt1 expected positions shape {(self.num_atoms, 3)}, "
                f"got {tuple(positions.shape)}"
            )
        # Persistent state object; the only precision conversion is CUDA FP64 MD
        # positions to FP32 AtomGraphs/model inputs inside OrbTorchSimModel.
        self.sim_state.positions = positions
        outputs = self.model(
            self.sim_state,
            compute_forces=True,
            compute_stress=self.compute_stress,
        )
        if "energy" not in outputs or "forces" not in outputs:
            raise RuntimeError(f"ORB model omitted energy/forces: {sorted(outputs)}")
        forces = outputs["forces"].reshape(self.num_atoms, 3).to(torch.float64)
        energy = outputs["energy"].reshape(-1)[0].to(torch.float64)
        stress = outputs.get("stress")
        if stress is not None:
            stress = stress.reshape(-1, 3, 3)[0].to(torch.float64)
        return forces.detach(), energy.detach(), None if stress is None else stress.detach()


class BerendsenIntegrator:
    """CUDA Velocity-Verlet/Berendsen equations matching unconstrained ASE."""

    name = "berendsen"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
        degrees_of_freedom: int,
    ) -> None:
        self.masses = masses.reshape(-1, 1)
        self.dt = float(timestep_fs) * units.fs
        self.target_temperature = float(temperature_k)
        self.taut = float(thermostat_time_fs) * units.fs
        self.degrees_of_freedom = int(degrees_of_freedom)
        if self.degrees_of_freedom <= 0:
            raise ValueError("degrees of freedom must be positive")

    def reset(self) -> None:
        """Berendsen has no persistent thermostat variables."""

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def step(self, state: GPUMDState, evaluator: _Evaluator) -> None:
        temperature = (
            2.0
            * self.kinetic_energy(state.momenta)
            / (self.degrees_of_freedom * units.kB)
        ).clamp_min(1.0e-12)
        scale = torch.sqrt(
            1.0
            + (self.target_temperature / temperature - 1.0) * (self.dt / self.taut)
        ).clamp(min=0.9, max=1.1)
        momenta = state.momenta * scale
        _ensure_evaluated(state, evaluator)
        assert state.forces is not None
        momenta = momenta + 0.5 * self.dt * state.forces
        # ASE NVTBerendsen defaults to fixcm=True.
        momenta = momenta - momenta.sum(dim=0, keepdim=True) / float(momenta.shape[0])
        positions = state.positions + self.dt * momenta / self.masses
        forces, energy, stress = evaluator(positions)
        momenta = momenta + 0.5 * self.dt * forces
        state.positions = positions
        state.momenta = momenta
        state.forces = forces
        state.potential_energy = energy
        state.stress = stress


class NoseHooverChainIntegrator:
    """FP64 CUDA port of ASE 3.29 NoseHooverChainNVT (tchain=3, tloop=1)."""

    name = "nose_hoover_chain"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
        chain_length: int = 3,
        chain_loops: int = 1,
    ) -> None:
        if chain_length < 1 or chain_loops < 1:
            raise ValueError("Nose-Hoover chain length/loops must be positive")
        self.masses = masses.reshape(-1, 1)
        self.num_atoms = int(self.masses.numel())
        self.dt = float(timestep_fs) * units.fs
        self.kT = float(temperature_k) * units.kB
        self.tdamp = float(thermostat_time_fs) * units.fs
        self.chain_length = int(chain_length)
        self.chain_loops = int(chain_loops)
        self.Q = self.masses.new_full((self.chain_length,), self.kT * self.tdamp**2)
        self.Q[0] *= 3.0 * self.num_atoms
        self.reset()

    def reset(self) -> None:
        self.eta = self.masses.new_zeros(self.chain_length)
        self.p_eta = self.masses.new_zeros(self.chain_length)

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def _integrate_p_eta_j(
        self, momenta: Tensor, j: int, delta2: float, delta4: float
    ) -> None:
        if j < self.chain_length - 1:
            self.p_eta[j] *= torch.exp(-delta4 * self.p_eta[j + 1] / self.Q[j + 1])
        if j == 0:
            g_j = (momenta.square() / self.masses).sum() - 3.0 * self.num_atoms * self.kT
        else:
            g_j = self.p_eta[j - 1].square() / self.Q[j - 1] - self.kT
        self.p_eta[j] += delta2 * g_j
        if j < self.chain_length - 1:
            self.p_eta[j] *= torch.exp(-delta4 * self.p_eta[j + 1] / self.Q[j + 1])

    def _integrate_loop(self, momenta: Tensor, delta: float) -> Tensor:
        delta2, delta4 = delta / 2.0, delta / 4.0
        for j in reversed(range(self.chain_length)):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        self.eta += delta * self.p_eta / self.Q
        momenta = momenta * torch.exp(-delta * self.p_eta[0] / self.Q[0])
        for j in range(self.chain_length):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        return momenta

    def _integrate_chain(self, momenta: Tensor, delta: float) -> Tensor:
        for _ in range(self.chain_loops):
            for coefficient in _FOURTH_ORDER_COEFFS:
                momenta = self._integrate_loop(
                    momenta, coefficient * delta / self.chain_loops
                )
        return momenta

    def step(self, state: GPUMDState, evaluator: _Evaluator) -> None:
        dt2 = self.dt / 2.0
        momenta = self._integrate_chain(state.momenta, dt2)
        _ensure_evaluated(state, evaluator)
        assert state.forces is not None
        momenta = momenta + dt2 * state.forces
        positions = state.positions + self.dt * momenta / self.masses
        forces, energy, stress = evaluator(positions)
        momenta = momenta + dt2 * forces
        momenta = self._integrate_chain(momenta, dt2)
        state.positions = positions
        state.momenta = momenta
        state.forces = forces
        state.potential_energy = energy
        state.stress = stress


def _ensure_evaluated(state: GPUMDState, evaluator: _Evaluator) -> None:
    if state.forces is None:
        state.forces, state.potential_energy, state.stress = evaluator(state.positions)


def _snapshot(
    template: Atoms,
    state: GPUMDState,
    *,
    step: int,
    require_stress: bool,
) -> Atoms:
    _require_evaluated_state(state)
    if require_stress and state.stress is None:
        raise RuntimeError("Matbench trajectory requires stress but ORB returned none")
    frame = template.copy()
    frame.set_positions(state.positions.detach().cpu().numpy())
    frame.set_momenta(state.momenta.detach().cpu().numpy())
    results: dict[str, Any] = {
        "energy": float(state.potential_energy.item()),
        "forces": state.forces.detach().cpu().numpy(),
    }
    if state.stress is not None:
        results["stress"] = state.stress.detach().cpu().numpy()
    frame.info["md_step"] = step
    frame.calc = SinglePointCalculator(frame, **results)
    return frame


def _require_evaluated_state(state: GPUMDState) -> None:
    if state.forces is None or state.potential_energy is None:
        raise RuntimeError("MD state has not been evaluated")


def _record_observation(state: GPUMDState, step: int, masses: Tensor) -> MDObservation:
    _require_evaluated_state(state)
    assert state.forces is not None and state.potential_energy is not None
    kinetic = (0.5 * state.momenta.square() / masses.reshape(-1, 1)).sum()
    return MDObservation(
        step=step,
        potential_energy_ev=float(state.potential_energy.item()),
        kinetic_energy_ev=float(kinetic.item()),
        forces_ev_per_a=state.forces.detach().cpu().numpy().copy(),
        positions_a=state.positions.detach().cpu().numpy().copy(),
    )


def _validate_final_state(state: GPUMDState) -> None:
    _require_evaluated_state(state)
    tensors = {
        "positions": state.positions,
        "momenta": state.momenta,
        "forces": state.forces,
        "potential_energy": state.potential_energy,
    }
    if state.stress is not None:
        tensors["stress"] = state.stress
    invalid = [
        name
        for name, value in tensors.items()
        if value is None or not bool(torch.isfinite(value).all().item())
    ]
    if invalid:
        raise FloatingPointError(f"ORBv3 Opt1 final state has non-finite {invalid}")


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _build_integrator(request: MDRunRequest, masses: Tensor):
    config = request.config
    if config.integrator == "berendsen":
        return BerendsenIntegrator(
            masses,
            timestep_fs=config.timestep_fs,
            temperature_k=config.temperature_k,
            thermostat_time_fs=config.thermostat_time_fs,
            degrees_of_freedom=request.atoms.get_number_of_degrees_of_freedom(),
        )
    if config.integrator == "nose_hoover_chain":
        return NoseHooverChainIntegrator(
            masses,
            timestep_fs=config.timestep_fs,
            temperature_k=config.temperature_k,
            thermostat_time_fs=config.thermostat_time_fs,
        )
    raise ValueError(f"ORB Opt1 does not support integrator {config.integrator!r}")


def _validate_request(request: MDRunRequest) -> tuple[str, int | None]:
    if request.backend != "gpu-resident":
        raise ValueError("ORBv3 opt1 backend must be 'gpu-resident'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("ORBv3 Opt1 is GPU-resident and requires a CUDA device")
    if request.config.dtype != "float64":
        raise ValueError("ORBv3 Opt1 requires --dtype float64 for the MD state")
    if request.atoms.constraints:
        raise NotImplementedError("ORBv3 Opt1 does not silently ignore ASE constraints")
    if np.any(request.atoms.pbc) and not np.any(np.asarray(request.atoms.cell)):
        raise ValueError("'pbc' is True, but 'cell' is all zeros")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")

    variant = _normalise_variant(
        _option(request.options, "model_variant", _DEFAULT_MODEL_VARIANT)
    )
    filename_variant = _variant_in_filename(request.model_path)
    if filename_variant is not None and filename_variant != variant:
        raise ValueError(
            f"model_variant={variant!r} does not match checkpoint filename "
            f"{Path(request.model_path).name!r} (looks like {filename_variant!r})"
        )
    precision = str(_option(request.options, "model_precision", "float32-highest"))
    if precision != "float32-highest":
        raise ValueError("ORBv3 Opt1 fixes model_precision='float32-highest' (TF32 disabled)")
    edge_method = str(_option(request.options, "edge_method", "knn_alchemi"))
    if edge_method != "knn_alchemi":
        raise ValueError("ORBv3 Opt1 requires the CUDA-resident knn_alchemi edge method")
    half_supercell = _option(request.options, "half_supercell", False)
    if half_supercell not in {False, None}:
        raise ValueError("ORBv3 Opt1 TorchSim adapter does not support half_supercell=true")
    max_num_neighbors = _option(request.options, "max_num_neighbors", None)
    if max_num_neighbors is not None:
        max_num_neighbors = int(max_num_neighbors)
        if max_num_neighbors < 1:
            raise ValueError("ORB max_num_neighbors must be positive")
    return variant, max_num_neighbors


def _configure_precision() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run eager GPU-resident ORB MD while preserving the shared route contract."""

    variant, max_num_neighbors = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; ORBv3 Opt1 never falls back to ASE/CPU")
    device = torch.device(request.config.device)
    if device.type != "cuda":
        raise ValueError(f"expected CUDA device, got {device}")
    _configure_precision()

    config = request.config
    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        rng=np.random.default_rng(config.seed),
    )
    positions = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float64, device=device
    ).clone()
    momenta = torch.as_tensor(
        np.asarray(atoms.get_momenta()), dtype=torch.float64, device=device
    ).clone()
    masses = torch.as_tensor(
        np.asarray(atoms.get_masses()), dtype=torch.float64, device=device
    ).clone()
    state = GPUMDState(positions=positions, momenta=momenta)
    initial_state = state.clone()
    evaluator = OrbTorchSimEvaluator(
        atoms,
        request.model_path,
        variant=variant,
        device=device,
        max_num_neighbors=max_num_neighbors,
        compute_stress=config.collect_trajectory,
    )
    integrator = _build_integrator(request, masses)

    if config.warmup_steps:
        for _ in range(config.warmup_steps):
            integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_(initial_state)
        integrator.reset()

    trajectory_path = Path(request.output_path) if request.output_path else None
    partial_path = (
        trajectory_path.with_name(f"{trajectory_path.stem}.part.extxyz")
        if trajectory_path is not None
        else None
    )
    trajectory: list[Atoms] | None = (
        [] if config.collect_trajectory and trajectory_path is None else None
    )
    if config.collect_trajectory and config.record_interval < 1:
        raise ValueError("collect_trajectory requires record_interval >= 1")
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if trajectory_path.exists() and not request.options.get("overwrite", False):
            raise FileExistsError(f"Refusing to overwrite {trajectory_path}")
        for stale in (trajectory_path, partial_path):
            if stale is not None and stale.exists():
                stale.unlink()

    def write_frame(step: int) -> None:
        frame = _snapshot(atoms, state, step=step, require_stress=True)
        if partial_path is not None:
            ase.io.write(partial_path, frame, append=True, format="extxyz")
        else:
            assert trajectory is not None
            trajectory.append(frame)

    observation_steps = set(config.observation_steps)
    observations: list[MDObservation] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    # Matbench requires the initial frame and every record_interval frame.
    if config.collect_trajectory:
        _ensure_evaluated(state, evaluator)
        write_frame(0)
    for step in range(1, config.steps + 1):
        integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_record_observation(state, step, masses))
        if config.collect_trajectory and step % config.record_interval == 0:
            write_frame(step)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    _validate_final_state(state)
    if trajectory_path is not None and partial_path is not None:
        os.replace(partial_path, trajectory_path)

    final_atoms = atoms.copy()
    final_atoms.set_positions(state.positions.detach().cpu().numpy())
    final_atoms.set_momenta(state.momenta.detach().cpu().numpy())
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory_gb,
        final_atoms=final_atoms,
        observations=observations,
        trajectory=trajectory,
        trajectory_path=str(trajectory_path) if trajectory_path is not None else None,
        metadata={
            "engine": "torch-sim-orb-gpu-resident-eager",
            "backend": request.backend,
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "model_path": str(Path(request.model_path).resolve()),
            "model_variant": variant,
            "compile": False,
            "cuda_graph": False,
            "model_specific_fusion": False,
            "md_state_device": str(device),
            "md_state_dtype": "float64",
            "model_dtype": "float32",
            "model_precision": "float32-highest",
            "tf32": False,
            "integrator": config.integrator,
            "integrator_implementation": "orb_models.md_stages.opt1",
            "torch_sim_role": "CUDA state container and ORB model interface",
            "positions_momenta_forces_cuda_resident": True,
            "hot_loop_per_step_cpu_or_numpy": False,
            "reporting_device_transfer_interval": (
                config.record_interval if config.collect_trajectory else None
            ),
            "edge_method": "knn_alchemi",
            "neighbor_builder": "existing ORB/Alchemi baseline implementation",
            "neighbor_builder_device": "cuda",
            "neighbor_builder_dtype": "float32",
            "max_num_neighbors": evaluator.max_num_neighbors,
            "half_supercell": False,
            "warmup_steps": config.warmup_steps,
            "trajectory_initial_frame": bool(config.collect_trajectory),
            "trajectory_stress": bool(config.collect_trajectory),
            "model_compute_stress": evaluator.compute_stress,
            "alchemi_matrix_capacity": (
                None
                if evaluator.model.alchemi_neighbor_state is None
                else evaluator.model.alchemi_neighbor_state.max_neighbors
            ),
        },
    )
    validate_result(request, result)
    return result
