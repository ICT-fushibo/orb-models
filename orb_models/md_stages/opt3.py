"""ORBv3 Opt3: one whole-step CUDA Graph for GPU-resident NVT MD.

The captured region closes the complete hot-loop over persistent CUDA tensors:
the thermostat/integrator update, fixed-candidate PBC neighbor construction,
sink-padded ORB forward/force VJP, and final state update.  Capacity is fixed
before capture and overflow is a hard error.  This stage deliberately does not
implement recapture, graph buckets, transaction rollback, compile, or fusion.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.performance import CudaPhaseProfiler, performance_profile_requested
from torch import nn

from orb_models.common.atoms import graph_featurization as graph_feat
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.models.conservative_regressor import (
    ConservativeForcefieldRegressor,
)
from orb_models.md_route import (
    _DEFAULT_MODEL_VARIANT,
    _normalise_variant,
    _option,
    _variant_in_filename,
)
from orb_models.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    NoseHooverChainIntegrator,
    OrbTorchSimEvaluator,
    _build_integrator,
    _configure_precision,
    _distribution_version,
    _record_observation,
    _validate_final_state,
)
from orb_models.md_stages.opt2 import (
    CUDAGraphCapacityError,
    CUDAGraphValidationError,
    _ORBForceOnlyModel,
    _RealAtomEnergyHead,
    edge_capacity_from_probe,
)


def sink_pad_neighbor_matrix(
    neighbor_matrix: torch.Tensor,
    unit_shift_matrix: torch.Tensor,
    *,
    n_real: int,
    n_dummy: int,
    padding_unit_shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten fixed per-atom slots and isolate invalid rows on dummy sinks.

    Invalid slots become far-shifted dummy self-edges.  Padding is distributed
    over a bank of sinks instead of contending on one node.  The function is
    shape-static and contains no host scalar extraction, so it is safe to call
    while a CUDA Graph is being captured.
    """

    if neighbor_matrix.ndim != 2:
        raise ValueError("neighbor_matrix must have shape [n_real, slots]")
    if neighbor_matrix.shape[0] != n_real:
        raise ValueError("neighbor_matrix first dimension must equal n_real")
    if unit_shift_matrix.shape != (*neighbor_matrix.shape, 3):
        raise ValueError("unit_shift_matrix must have shape [n_real, slots, 3]")
    if n_dummy < 1:
        raise ValueError("n_dummy must be positive")
    if padding_unit_shift.shape != (3,):
        raise ValueError("padding_unit_shift must have shape [3]")

    slots = neighbor_matrix.shape[1]
    valid = neighbor_matrix != -1
    real_senders = torch.arange(
        n_real, dtype=neighbor_matrix.dtype, device=neighbor_matrix.device
    ).view(-1, 1).expand(n_real, slots)
    sink_ids = n_real + (
        torch.arange(
            n_real * slots,
            dtype=neighbor_matrix.dtype,
            device=neighbor_matrix.device,
        ).reshape(n_real, slots)
        % n_dummy
    )
    senders = torch.where(valid, real_senders, sink_ids).reshape(-1)
    receivers = torch.where(valid, neighbor_matrix.clamp_min(0), sink_ids).reshape(-1)
    unit_shifts = torch.where(
        valid.unsqueeze(-1),
        unit_shift_matrix,
        padding_unit_shift.view(1, 1, 3),
    ).reshape(-1, 3)
    return senders, receivers, unit_shifts, valid.reshape(-1)


def _pad_node_tensor(value: torch.Tensor, n_real: int, n_dummy: int) -> torch.Tensor:
    if value.ndim and value.shape[0] == n_real:
        dummy = value[:1].expand(n_dummy, *value.shape[1:]).clone()
        return torch.cat((value, dummy), dim=0)
    return value.clone()


def _pbc_repetitions(
    cell: torch.Tensor, cutoff: float, pbc: torch.Tensor
) -> tuple[int, int, int]:
    """Resolve a complete periodic image range during setup only."""

    cell_cpu = cell.detach().to(device="cpu", dtype=torch.float64).reshape(3, 3)
    pbc_cpu = pbc.detach().to(device="cpu", dtype=torch.bool).reshape(3)
    cross_a2a3 = torch.cross(cell_cpu[1], cell_cpu[2], dim=0)
    volume = torch.dot(cell_cpu[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError("ORBv3 Opt3 cannot enumerate a singular periodic cell")
    reciprocal = (
        cross_a2a3,
        torch.cross(cell_cpu[2], cell_cpu[0], dim=0),
        torch.cross(cell_cpu[0], cell_cpu[1], dim=0),
    )
    repetitions = []
    for axis in range(3):
        if bool(pbc_cpu[axis]):
            inverse_plane_distance = torch.linalg.vector_norm(
                reciprocal[axis] / volume
            )
            repetitions.append(
                int(torch.ceil(cutoff * inverse_plane_distance).item())
            )
        else:
            repetitions.append(0)
    return tuple(repetitions)  # type: ignore[return-value]


class _FixedShapeORBNeighborBuilder:
    """Capture-safe fixed candidate form of ORB's single-system PBC graph.

    The PBC image universe and candidate receiver IDs are frozen at setup.
    Replay performs only fixed-shape distance, mask, and top-k tensor work;
    setup-only nvalchemi operations never enter stream capture.
    """

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
        neighbors_per_atom: int,
    ) -> None:
        if num_atoms < 2:
            raise ValueError("ORBv3 fixed builder requires at least two atoms")
        if cutoff <= 0.0:
            raise ValueError("ORBv3 fixed builder cutoff must be positive")
        if neighbors_per_atom < 1:
            raise ValueError("ORBv3 neighbors_per_atom must be positive")
        self.num_atoms = int(num_atoms)
        self.cutoff = float(cutoff)
        self.neighbors_per_atom = int(neighbors_per_atom)
        self.device = cell.device
        self.cell = cell.detach().reshape(3, 3).contiguous()
        self.repetitions = _pbc_repetitions(self.cell, self.cutoff, pbc)
        axes = [
            torch.arange(
                -repeat,
                repeat + 1,
                dtype=self.cell.dtype,
                device=self.device,
            )
            for repeat in self.repetitions
        ]
        self.unit_shifts = torch.cartesian_prod(*axes).reshape(-1, 3).contiguous()
        self.num_cells = int(self.unit_shifts.shape[0])
        self.candidates_per_atom = self.num_atoms * self.num_cells
        if self.neighbors_per_atom > self.candidates_per_atom:
            raise ValueError(
                "ORBv3 neighbor capacity exceeds the complete PBC universe"
            )
        self.candidate_receivers = torch.arange(
            self.num_atoms, dtype=torch.long, device=self.device
        ).repeat_interleave(self.num_cells)
        self.candidate_shifts = self.unit_shifts.repeat(self.num_atoms, 1)
        self.infinity = torch.tensor(
            float("inf"), dtype=self.cell.dtype, device=self.device
        )

    def build(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fixed receiver slots, exact counts, and lattice shifts."""

        if positions.shape != (self.num_atoms, 3):
            raise ValueError("ORBv3 fixed builder received the wrong atom shape")
        receiver_positions = positions.index_select(0, self.candidate_receivers)
        shift_vectors = torch.mm(
            self.candidate_shifts.to(dtype=positions.dtype),
            self.cell.to(dtype=positions.dtype),
        )
        vectors = (
            receiver_positions.unsqueeze(0)
            - positions.unsqueeze(1)
            + shift_vectors.unsqueeze(0)
        )
        distance_sqr = vectors.square().sum(dim=-1)
        valid = (distance_sqr <= self.cutoff * self.cutoff) & (
            distance_sqr > 1.0e-8
        )
        counts = valid.sum(dim=1)
        masked_distance = torch.where(valid, distance_sqr, self.infinity)
        selected_distance, selected = torch.topk(
            masked_distance,
            k=self.neighbors_per_atom,
            dim=1,
            largest=False,
            sorted=True,
        )
        selected_valid = torch.isfinite(selected_distance)
        flat = selected.reshape(-1)
        receivers = self.candidate_receivers.index_select(0, flat).reshape(
            self.num_atoms, self.neighbors_per_atom
        )
        shifts = self.candidate_shifts.index_select(0, flat).reshape(
            self.num_atoms, self.neighbors_per_atom, 3
        )
        receivers = torch.where(
            selected_valid, receivers, torch.full_like(receivers, -1)
        )
        shifts = torch.where(
            selected_valid.unsqueeze(-1), shifts, torch.zeros_like(shifts)
        )
        return receivers, counts, shifts


class _RealAtomPairRepulsion(nn.Module):
    """Restore the released ZBL reduction after adding multiple sink nodes."""

    def __init__(self, base: nn.Module, n_real: int, n_dummy: int) -> None:
        super().__init__()
        self.base = base
        reduction = getattr(base, "node_aggregation", None)
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"unsupported pair-repulsion aggregation {reduction!r}")
        self.scale = (n_real + n_dummy) / n_real if reduction == "mean" else 1.0

    def forward(self, batch: AtomGraphs) -> dict[str, torch.Tensor]:
        output = self.base(batch)
        return {**output, "energy": output["energy"] * self.scale}


def whole_step_in_place_(
    state: GPUMDState,
    integrator: BerendsenIntegrator | NoseHooverChainIntegrator,
    evaluator,
    advance: torch.Tensor | None = None,
) -> None:
    """Evaluate/advance NVT while retaining every persistent tensor address.

    ``advance=0`` performs only the initial force evaluation. ``advance=1``
    performs a complete MD step.  The branchless scalar is part of the captured
    graph so initial force and all production steps use the same graph.
    """

    if state.forces is None or state.potential_energy is None:
        raise RuntimeError("whole-step capture requires an evaluated initial state")

    if advance is None:
        advance = state.positions.new_ones(())
    old_positions = state.positions
    old_momenta = state.momenta

    if isinstance(integrator, BerendsenIntegrator):
        temperature = (
            2.0
            * integrator.kinetic_energy(state.momenta)
            / (integrator.degrees_of_freedom * units.kB)
        ).clamp_min(1.0e-12)
        scale = torch.sqrt(
            1.0
            + (integrator.target_temperature / temperature - 1.0)
            * (integrator.dt / integrator.taut)
        ).clamp(min=0.9, max=1.1)
        momenta = state.momenta * scale
        momenta = momenta + 0.5 * integrator.dt * state.forces
        momenta = momenta - momenta.sum(dim=0, keepdim=True) / float(
            momenta.shape[0]
        )
        proposed_positions = (
            state.positions + integrator.dt * momenta / integrator.masses
        )
        evaluation_positions = old_positions + advance * (
            proposed_positions - old_positions
        )
        forces, energy, _stress = evaluator(evaluation_positions)
        proposed_momenta = momenta + 0.5 * integrator.dt * forces
    elif isinstance(integrator, NoseHooverChainIntegrator):
        old_eta = integrator.eta.clone()
        old_p_eta = integrator.p_eta.clone()
        dt2 = integrator.dt / 2.0
        momenta = integrator._integrate_chain(state.momenta, dt2)
        momenta = momenta + dt2 * state.forces
        proposed_positions = (
            state.positions + integrator.dt * momenta / integrator.masses
        )
        evaluation_positions = old_positions + advance * (
            proposed_positions - old_positions
        )
        forces, energy, _stress = evaluator(evaluation_positions)
        momenta = momenta + dt2 * forces
        proposed_momenta = integrator._integrate_chain(momenta, dt2)
        integrator.eta.copy_(old_eta + advance * (integrator.eta - old_eta))
        integrator.p_eta.copy_(
            old_p_eta + advance * (integrator.p_eta - old_p_eta)
        )
    else:
        raise TypeError(f"unsupported whole-step integrator {type(integrator).__name__}")

    # Closing copies make the captured outputs the next replay's inputs.
    state.positions.copy_(
        old_positions + advance * (proposed_positions - old_positions)
    )
    state.momenta.copy_(old_momenta + advance * (proposed_momenta - old_momenta))
    state.forces.copy_(forces)
    state.potential_energy.copy_(energy)


class WholeStepCUDAGraphRunner(OrbTorchSimEvaluator):
    """Capture Alchemi builder + ORB VJP + NVT update in one CUDA Graph."""

    def __init__(
        self,
        atoms,
        model_path: str,
        *,
        variant: str,
        device: torch.device,
        max_num_neighbors: int | None,
        profiler: CudaPhaseProfiler,
        requested_edge_capacity: int | None,
        requested_neighbors_per_atom: int | None,
        edge_margin: float,
        edge_step: int,
        capture_warmup: int,
        n_dummy: int,
        energy_atol: float,
        force_atol: float,
    ) -> None:
        super().__init__(
            atoms,
            model_path,
            variant=variant,
            device=device,
            max_num_neighbors=max_num_neighbors,
            compute_stress=False,
            profiler=profiler,
        )
        if requested_edge_capacity is not None and requested_edge_capacity < 1:
            raise ValueError("cuda_graph_edge_capacity must be positive")
        if requested_neighbors_per_atom is not None and requested_neighbors_per_atom < 1:
            raise ValueError("whole_step_neighbors_per_atom must be positive")
        if edge_margin < 0 or edge_step < 1:
            raise ValueError("CUDA Graph edge margin/step is invalid")
        if capture_warmup < 0:
            raise ValueError("cuda_graph_capture_warmup cannot be negative")
        if n_dummy < 1:
            raise ValueError("cuda_graph_dummy_atoms must be positive")
        if not isinstance(self.model.model, ConservativeForcefieldRegressor):
            raise TypeError("ORBv3 Opt3 requires a conservative released checkpoint")
        if self.model.model.coulomb_module is not None:
            raise NotImplementedError("ORBv3 Opt3 sink padding does not support CoulombModule")
        if "latent_charges" in self.model.model.heads or "latent_spins" in self.model.model.heads:
            raise NotImplementedError(
                "ORBv3 Opt3 sink padding does not support latent charge/spin heads"
            )

        self.requested_edge_capacity = requested_edge_capacity
        self.requested_neighbors_per_atom = requested_neighbors_per_atom
        self.edge_margin = float(edge_margin)
        self.edge_step = int(edge_step)
        self.capture_warmup = int(capture_warmup)
        self.n_dummy = int(n_dummy)
        self.energy_atol = float(energy_atol)
        self.force_atol = float(force_atol)
        self.n_real = self.num_atoms

        self.sim_state.positions = self.sim_state.positions.contiguous()
        self.template = self.model._make_batch(self.sim_state)
        raw_capacity = self.model.alchemi_neighbor_state.max_neighbors
        if raw_capacity is None:
            raise RuntimeError("ORBv3 Opt3 Alchemi builder was not calibrated")
        self.raw_neighbor_capacity = int(raw_capacity)
        self.real_n_node = self.template.n_node
        self.model_cell = self.template.system_features["cell"].reshape(-1, 3, 3).contiguous()
        self.model_pbc = self.sim_state.pbc.view(-1, 3).contiguous()
        self.radius = float(self.template.radius)
        self.model_neighbor_limit = int(self.max_num_neighbors)

        # Probe the exact initial graph outside capture.  It defines a default
        # CAP and catches a too-small user capacity before any graph is built.
        model_positions, initial_edges, _unit_shifts = self._build_probe_inputs(
            self.sim_state.positions
        )
        initial_edge_count = int(initial_edges.shape[1])
        capacity = requested_edge_capacity or edge_capacity_from_probe(
            initial_edge_count, margin=edge_margin, edge_step=edge_step
        )
        initial_per_sender = torch.bincount(
            initial_edges[0], minlength=self.n_real
        )
        required_slots = int(initial_per_sender.max().item())
        self.capacity_average_derived = requested_neighbors_per_atom is None
        self.capacity_alignment = 8
        self.capacity_guard_slots = 0
        self.capacity_average_slots = math.ceil(capacity / self.n_real)
        self.capacity_total_floor = (
            math.ceil(self.capacity_average_slots / self.capacity_alignment)
            * self.capacity_alignment
        )
        guarded_initial = max(
            required_slots + 1,
            math.ceil(required_slots * (1.0 + edge_margin)),
        )
        self.capacity_initial_safe_slots = (
            math.ceil(guarded_initial / self.capacity_alignment)
            * self.capacity_alignment
        )
        if requested_neighbors_per_atom is not None:
            # The trajectory per-sender probe is authoritative for the
            # uniform CAP.  The separately aligned total-edge buffer can be
            # much larger for tiny systems because of its 256-edge alignment;
            # projecting that padding back onto every sender would overpad.
            slots = int(requested_neighbors_per_atom)
            self.capacity_source = "trajectory-total-and-per-atom-probe"
        else:
            slots = max(
                self.capacity_total_floor,
                self.capacity_initial_safe_slots,
            )
            self.capacity_source = "total-edge-plus-initial-per-atom"
        slots = min(slots, self.raw_neighbor_capacity)
        if self.model_neighbor_limit > 0:
            slots = min(slots, self.model_neighbor_limit)
        if required_slots > slots:
            raise CUDAGraphCapacityError(required_slots * self.n_real, slots * self.n_real)
        self.slots_per_atom = slots
        self.edge_capacity = self.n_real * self.slots_per_atom
        self.capacity_limit_enforced = (
            self.model_neighbor_limit < 0
            or self.slots_per_atom < self.model_neighbor_limit
        )

        # Preserve one released-path reference before replacing the energy/ZBL
        # reductions with real-atom-only variants for sink padding.
        (
            self.reference_initial_forces,
            self.reference_initial_energy,
            _reference_stress,
        ) = OrbTorchSimEvaluator.__call__(self, self.sim_state.positions)

        cell = self.model_cell[0].to(torch.float64)
        self.inverse_cell = torch.linalg.inv(cell).contiguous()
        self.periodic_mask = self.model_pbc[0].to(torch.bool).view(1, 3)
        self.fixed_neighbor_builder = _FixedShapeORBNeighborBuilder(
            num_atoms=self.n_real,
            cell=self.model_cell[0],
            pbc=self.model_pbc[0],
            cutoff=self.radius,
            neighbors_per_atom=self.slots_per_atom,
        )
        (
            _fixed_initial_matrix,
            fixed_initial_counts,
            _fixed_initial_shifts,
        ) = self.fixed_neighbor_builder.build(model_positions)
        fixed_required_slots = int(fixed_initial_counts.max().item())
        if fixed_required_slots != required_slots:
            raise RuntimeError(
                "ORBv3 Opt3 fixed candidate builder disagrees with the setup "
                f"nvalchemi probe: {fixed_required_slots} != {required_slots}"
            )
        cell_norms = torch.linalg.vector_norm(cell, dim=1)
        axis = int(cell_norms.argmax().item())
        axis_length = float(cell_norms[axis].item())
        if axis_length <= 0:
            raise ValueError("ORBv3 Opt3 requires a nonzero simulation cell")
        multiple = max(1, math.ceil((self.radius + 1.0) / axis_length))
        self.padding_unit_shift = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.padding_unit_shift[axis] = float(multiple)

        energy_head = self.model.model.heads["energy"]
        self.model.model.heads["energy"] = _RealAtomEnergyHead(
            energy_head, self.n_real, self.device
        )
        if self.model.model.pair_repulsion:
            self.model.model.pair_repulsion_fn = _RealAtomPairRepulsion(
                self.model.model.pair_repulsion_fn, self.n_real, self.n_dummy
            )
        self.force_only_model = _ORBForceOnlyModel(self.model.model).eval()
        self._initialize_batch(model_positions)

        self.advance = torch.ones((), dtype=torch.float64, device=self.device)
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.captured = False
        self.capture_count = 0
        self.capture_wall_time_s = 0.0
        self.total_replays = 0
        self.production_replays = 0
        self.capacity_misses = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.maximum_required_neighbors = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.maximum_capacity_excess = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.output_addresses_stable = False
        self.replay_stability_passed = False
        self.replay_energy_abs_error = 0.0
        self.replay_force_max_abs_error = 0.0
        self.validation_energy_abs_error = 0.0
        self.validation_force_max_abs_error = 0.0
        self.whole_step_position_max_abs_error = 0.0
        self.whole_step_momentum_max_abs_error = 0.0
        self.whole_step_force_max_abs_error = 0.0
        self.whole_step_energy_abs_error = 0.0
        self.whole_step_eta_max_abs_error = 0.0
        self.whole_step_p_eta_max_abs_error = 0.0
        self.whole_step_validation_within_tolerance = False
        self.min_real_edges = torch.full(
            (), self.edge_capacity, dtype=torch.long, device=self.device
        )
        self.max_real_edges = torch.zeros((), dtype=torch.long, device=self.device)

    def _build_probe_inputs(self, positions):
        # Reuse the released eager fixed-capacity builder only during setup.
        from orb_models.common.atoms import featurization as atom_feat

        wrapped = atom_feat.batch_map_to_pbc_cell(
            positions,
            self.sim_state.row_vector_cell.contiguous(),
            self.model_pbc,
            self.real_n_node,
        )
        edges, _vectors, shifts, _n_edges = graph_feat.batch_compute_pbc_radius_graph(
            positions=wrapped.contiguous(),
            cells=self.sim_state.row_vector_cell.contiguous(),
            pbcs=self.model_pbc,
            radius=self.radius,
            n_node=self.real_n_node,
            node_batch_index=self.template.node_batch_index,
            max_number_neighbors=self.template.max_num_neighbors,
            edge_method="knn_alchemi",
            device=self.device,
            float_dtype=torch.float32,
            static_max_number_neighbors=self.model_neighbor_limit,
            alchemi_neighbor_state=self.model.alchemi_neighbor_state,
            validate_pbc_cell=False,
        )
        return wrapped.to(torch.float32), edges, shifts.to(torch.float32)

    def _initialize_batch(self, initial_model_positions: torch.Tensor) -> None:
        total_nodes = self.n_real + self.n_dummy
        node_features = {
            name: _pad_node_tensor(value, self.n_real, self.n_dummy)
            for name, value in self.template.node_features.items()
        }
        node_features["positions"] = torch.cat(
            (
                initial_model_positions,
                initial_model_positions.new_zeros((self.n_dummy, 3)),
            ),
            dim=0,
        )
        self.static_batch = AtomGraphs(
            senders=torch.zeros(
                self.edge_capacity, dtype=torch.long, device=self.device
            ),
            receivers=torch.zeros(
                self.edge_capacity, dtype=torch.long, device=self.device
            ),
            n_node=torch.tensor([total_nodes], dtype=torch.long, device=self.device),
            n_edge=torch.tensor(
                [self.edge_capacity], dtype=torch.long, device=self.device
            ),
            node_features=node_features,
            edge_features={
                "vectors": torch.zeros(
                    (self.edge_capacity, 3), dtype=torch.float32, device=self.device
                ),
                "unit_shifts": torch.zeros(
                    (self.edge_capacity, 3), dtype=torch.float32, device=self.device
                ),
            },
            system_features={
                name: value.clone()
                for name, value in self.template.system_features.items()
            },
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self.radius,
            max_num_neighbors=self.template.max_num_neighbors.clone(),
        )

    def _wrap_positions(self, positions: torch.Tensor) -> torch.Tensor:
        fractional = positions @ self.inverse_cell
        wrapped_fractional = torch.remainder(fractional, 1.0)
        fractional = torch.where(
            self.periodic_mask, wrapped_fractional, fractional
        )
        return (fractional @ self.model_cell[0].to(torch.float64)).to(torch.float32)

    def _fixed_builder(self, positions: torch.Tensor):
        model_positions = self._wrap_positions(positions)
        neighbor_matrix, num_neighbors, shift_matrix = (
            self.fixed_neighbor_builder.build(model_positions)
        )
        required_neighbors = num_neighbors.max()
        capacity_excess = torch.clamp_min(
            required_neighbors - self.slots_per_atom, 0
        )
        self.capacity_misses.add_((capacity_excess > 0).to(torch.long))
        self.maximum_required_neighbors.copy_(
            torch.maximum(self.maximum_required_neighbors, required_neighbors)
        )
        self.maximum_capacity_excess.copy_(
            torch.maximum(self.maximum_capacity_excess, capacity_excess)
        )
        # Capacity overflow is a hard device-side failure.  There is no
        # truncation, recapture, eager fallback, or transaction rollback.
        torch._assert_async(
            num_neighbors.max() <= self.slots_per_atom,
            "ORBv3 Opt3 per-atom neighbor capacity exceeded; increase "
            "whole_step_neighbors_per_atom and restart",
        )
        senders, receivers, shifts, valid = sink_pad_neighbor_matrix(
            neighbor_matrix,
            shift_matrix.to(torch.float32),
            n_real=self.n_real,
            n_dummy=self.n_dummy,
            padding_unit_shift=self.padding_unit_shift,
        )
        real_edges = valid.sum()
        self.min_real_edges.copy_(torch.minimum(self.min_real_edges, real_edges))
        self.max_real_edges.copy_(torch.maximum(self.max_real_edges, real_edges))
        positions_with_sinks = torch.cat(
            (model_positions, model_positions.new_zeros((self.n_dummy, 3))), dim=0
        )
        return positions_with_sinks, senders, receivers, shifts

    def __call__(self, positions: torch.Tensor):
        model_positions, senders, receivers, shifts = self._fixed_builder(positions)
        self.static_batch.node_features["positions"] = model_positions
        self.static_batch.senders = senders
        self.static_batch.receivers = receivers
        self.static_batch.edge_features["unit_shifts"] = shifts
        with torch.enable_grad():
            forces, energy = self.force_only_model(self.static_batch)
        return (
            forces[: self.n_real].detach().to(torch.float64),
            energy.detach().reshape(-1)[0].to(torch.float64),
            None,
        )

    @staticmethod
    def _thermostat_snapshot(integrator):
        if isinstance(integrator, NoseHooverChainIntegrator):
            return integrator.eta.clone(), integrator.p_eta.clone()
        return None

    @staticmethod
    def _restore_thermostat(integrator, snapshot) -> None:
        if snapshot is not None:
            integrator.eta.copy_(snapshot[0])
            integrator.p_eta.copy_(snapshot[1])

    @staticmethod
    def _state_snapshot(state: GPUMDState):
        assert state.forces is not None and state.potential_energy is not None
        return (
            state.positions.clone(),
            state.momenta.clone(),
            state.forces.clone(),
            state.potential_energy.clone(),
        )

    @staticmethod
    def _restore_state(state: GPUMDState, snapshot) -> None:
        state.positions.copy_(snapshot[0])
        state.momenta.copy_(snapshot[1])
        assert state.forces is not None and state.potential_energy is not None
        state.forces.copy_(snapshot[2])
        state.potential_energy.copy_(snapshot[3])

    def capture(self, state: GPUMDState, integrator) -> None:
        if self.captured:
            raise RuntimeError("ORBv3 whole-step CUDA Graph was already captured")

        # Validate the sink-padded fixed builder against the released eager path
        # before replacing the state's initial force.
        eager_forces = self.reference_initial_forces
        eager_energy = self.reference_initial_energy
        fixed_forces, fixed_energy, _ = self(state.positions)
        self.validation_energy_abs_error = float(
            (fixed_energy - eager_energy).abs().item()
        )
        self.validation_force_max_abs_error = float(
            (fixed_forces - eager_forces).abs().max().item()
        )
        if not all(
            bool(value.isfinite().all().item())
            for value in (fixed_forces, fixed_energy, eager_forces, eager_energy)
        ):
            raise CUDAGraphValidationError(
                "ORBv3 fixed-builder validation produced non-finite output"
            )
        if (
            self.validation_energy_abs_error > self.energy_atol
            or self.validation_force_max_abs_error > self.force_atol
        ):
            warnings.warn(
                "ORBv3 sink-padded fixed builder differs from the released eager "
                "path beyond the reporting tolerance; performance testing continues "
                "under the benchmark warning-only numerical policy.",
                RuntimeWarning,
                stacklevel=2,
            )
        state.forces = fixed_forces.clone()
        state.potential_energy = fixed_energy.clone()

        state_snapshot = self._state_snapshot(state)
        thermostat_snapshot = self._thermostat_snapshot(integrator)

        # Independent eager fixed-builder + integrator reference from the same
        # evaluated state.  Run it before capture so the static batch retains
        # the captured tensors for the entire graph lifetime.
        reference_state = GPUMDState(
            positions=state_snapshot[0].clone(),
            momenta=state_snapshot[1].clone(),
            forces=state_snapshot[2].clone(),
            potential_energy=state_snapshot[3].clone(),
        )
        integrator.step(reference_state, self)
        reference_thermostat = self._thermostat_snapshot(integrator)
        self._restore_thermostat(integrator, thermostat_snapshot)

        self.advance.fill_(1.0)
        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(self.capture_warmup):
                whole_step_in_place_(state, integrator, self, self.advance)
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)
        self._restore_state(state, state_snapshot)
        self._restore_thermostat(integrator, thermostat_snapshot)
        self.reset_production_stats()

        side_stream.wait_stream(current_stream)
        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            whole_step_in_place_(state, integrator, self, self.advance)
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - started
        self.cuda_graph = graph
        self.capture_count = 1
        self.captured = True
        self._restore_state(state, state_snapshot)
        self._restore_thermostat(integrator, thermostat_snapshot)
        self.reset_production_stats()

        state_addresses = tuple(
            value.data_ptr()
            for value in (
                state.positions,
                state.momenta,
                state.forces,
                state.potential_energy,
            )
        )
        thermostat_addresses = (
            None
            if thermostat_snapshot is None
            else (integrator.eta.data_ptr(), integrator.p_eta.data_ptr())
        )

        self._restore_state(state, state_snapshot)
        self._restore_thermostat(integrator, thermostat_snapshot)
        self.advance.fill_(1.0)
        graph.replay()
        torch.cuda.synchronize(self.device)
        first_state = self._state_snapshot(state)
        first_thermostat = self._thermostat_snapshot(integrator)
        self._restore_state(state, state_snapshot)
        self._restore_thermostat(integrator, thermostat_snapshot)
        self.advance.fill_(1.0)
        graph.replay()
        torch.cuda.synchronize(self.device)
        second_state = self._state_snapshot(state)
        second_thermostat = self._thermostat_snapshot(integrator)
        self.total_replays += 2
        self.output_addresses_stable = state_addresses == tuple(
            value.data_ptr()
            for value in (
                state.positions,
                state.momenta,
                state.forces,
                state.potential_energy,
            )
        )
        if thermostat_snapshot is not None:
            self.output_addresses_stable = self.output_addresses_stable and (
                thermostat_addresses
                == (integrator.eta.data_ptr(), integrator.p_eta.data_ptr())
            )
        self.replay_energy_abs_error = float(
            (first_state[3] - second_state[3]).abs().item()
        )
        self.replay_force_max_abs_error = float(
            (first_state[2] - second_state[2]).abs().max().item()
        )
        self.replay_stability_passed = (
            self.replay_energy_abs_error <= self.energy_atol
            and self.replay_force_max_abs_error <= self.force_atol
        )
        if not self.output_addresses_stable:
            raise CUDAGraphValidationError(
                "ORBv3 whole-step CUDA Graph state addresses changed"
            )

        self.whole_step_position_max_abs_error = float(
            (first_state[0] - reference_state.positions).abs().max().item()
        )
        self.whole_step_momentum_max_abs_error = float(
            (first_state[1] - reference_state.momenta).abs().max().item()
        )
        self.whole_step_force_max_abs_error = float(
            (first_state[2] - reference_state.forces).abs().max().item()
        )
        self.whole_step_energy_abs_error = float(
            (first_state[3] - reference_state.potential_energy).abs().item()
        )
        if reference_thermostat is not None and first_thermostat is not None:
            self.whole_step_eta_max_abs_error = float(
                (first_thermostat[0] - reference_thermostat[0]).abs().max().item()
            )
            self.whole_step_p_eta_max_abs_error = float(
                (first_thermostat[1] - reference_thermostat[1]).abs().max().item()
            )
        state_atol = 1.0e-10
        self.whole_step_validation_within_tolerance = (
            self.whole_step_position_max_abs_error <= state_atol
            and self.whole_step_momentum_max_abs_error <= state_atol
            and self.whole_step_force_max_abs_error <= self.force_atol
            and self.whole_step_energy_abs_error <= self.energy_atol
            and self.whole_step_eta_max_abs_error <= state_atol
            and self.whole_step_p_eta_max_abs_error <= state_atol
        )

        finite_tensors = [
            *first_state,
            *second_state,
            reference_state.positions,
            reference_state.momenta,
            reference_state.forces,
            reference_state.potential_energy,
        ]
        if first_thermostat is not None:
            finite_tensors.extend(first_thermostat)
        if second_thermostat is not None:
            finite_tensors.extend(second_thermostat)
        if reference_thermostat is not None:
            finite_tensors.extend(reference_thermostat)
        if not all(bool(value.isfinite().all().item()) for value in finite_tensors):
            raise CUDAGraphValidationError(
                "ORBv3 whole-step validation produced non-finite state"
            )
        if not self.whole_step_validation_within_tolerance:
            warnings.warn(
                "ORBv3 whole-step CUDA Graph differs from eager fixed-builder "
                "integration beyond the reporting tolerance; performance testing "
                "continues under the benchmark warning-only numerical policy.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._restore_state(state, state_snapshot)
        self._restore_thermostat(integrator, thermostat_snapshot)
        self.advance.zero_()
        self.reset_production_stats()

    def replay(self) -> None:
        if not self.captured or self.cuda_graph is None:
            raise RuntimeError("ORBv3 whole-step CUDA Graph is not captured")
        self.cuda_graph.replay()
        self.total_replays += 1
        self.production_replays += 1

    def reset_production_stats(self) -> None:
        self.production_replays = 0
        self.capacity_misses.zero_()
        self.maximum_required_neighbors.zero_()
        self.maximum_capacity_excess.zero_()
        self.min_real_edges.fill_(self.edge_capacity)
        self.max_real_edges.zero_()

    def stats(self) -> dict[str, Any]:
        minimum = int(self.min_real_edges.item()) if self.production_replays else None
        maximum = int(self.max_real_edges.item()) if self.production_replays else None
        return {
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_production_capture_count": 0,
            "cuda_graph_total_replays": self.total_replays,
            "cuda_graph_production_calls": self.production_replays,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_capacity_misses": int(self.capacity_misses.item()),
            "cuda_graph_max_required_neighbors_per_atom": int(
                self.maximum_required_neighbors.item()
            ),
            "cuda_graph_max_capacity_excess_per_atom": int(
                self.maximum_capacity_excess.item()
            ),
            "cuda_graph_hit_rate": (
                1.0
                if self.production_replays and int(self.capacity_misses.item()) == 0
                else 0.0
            ),
            "cuda_graph_edge_capacity_requested": self.requested_edge_capacity,
            "cuda_graph_neighbors_per_atom_requested": self.requested_neighbors_per_atom,
            "cuda_graph_edge_capacity": self.edge_capacity,
            "cuda_graph_neighbors_per_atom": self.slots_per_atom,
            "cuda_graph_capacity_average_derived": self.capacity_average_derived,
            "cuda_graph_capacity_average_slots": self.capacity_average_slots,
            "cuda_graph_capacity_total_floor": self.capacity_total_floor,
            "cuda_graph_capacity_initial_safe_slots": (
                self.capacity_initial_safe_slots
            ),
            "cuda_graph_capacity_source": self.capacity_source,
            "cuda_graph_capacity_alignment": self.capacity_alignment,
            "cuda_graph_capacity_guard_slots": self.capacity_guard_slots,
            "cuda_graph_capacity_limit_enforced": self.capacity_limit_enforced,
            "cuda_graph_raw_neighbor_capacity": self.raw_neighbor_capacity,
            "cuda_graph_neighbor_backend": "orb-fixed-candidate-cap",
            "cuda_graph_capture_safe_setup_hoisted": True,
            "cuda_graph_candidate_universe_size": (
                self.n_real * self.fixed_neighbor_builder.candidates_per_atom
            ),
            "cuda_graph_candidates_per_atom": (
                self.fixed_neighbor_builder.candidates_per_atom
            ),
            "cuda_graph_num_pbc_cells": self.fixed_neighbor_builder.num_cells,
            "cuda_graph_pbc_repetitions": list(
                self.fixed_neighbor_builder.repetitions
            ),
            "cuda_graph_min_real_edges": minimum,
            "cuda_graph_max_real_edges": maximum,
            "cuda_graph_dummy_atoms": self.n_dummy,
            "cuda_graph_capture_warmup": self.capture_warmup,
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_replay_output_addresses_stable": self.output_addresses_stable,
            "cuda_graph_replay_stability_pass": self.replay_stability_passed,
            "cuda_graph_replay_stability_energy_abs_error_eV": self.replay_energy_abs_error,
            "cuda_graph_replay_stability_force_max_abs_error_eV_per_A": self.replay_force_max_abs_error,
            "cuda_graph_validation_energy_abs_error_eV": self.validation_energy_abs_error,
            "cuda_graph_validation_force_max_abs_error_eV_per_A": self.validation_force_max_abs_error,
            "cuda_graph_whole_step_position_max_abs_error_A": self.whole_step_position_max_abs_error,
            "cuda_graph_whole_step_momentum_max_abs_error": self.whole_step_momentum_max_abs_error,
            "cuda_graph_whole_step_force_max_abs_error_eV_per_A": self.whole_step_force_max_abs_error,
            "cuda_graph_whole_step_energy_abs_error_eV": self.whole_step_energy_abs_error,
            "cuda_graph_whole_step_eta_max_abs_error": self.whole_step_eta_max_abs_error,
            "cuda_graph_whole_step_p_eta_max_abs_error": self.whole_step_p_eta_max_abs_error,
            "cuda_graph_whole_step_validation_within_tolerance": self.whole_step_validation_within_tolerance,
            "cuda_graph_validation_energy_atol_eV": self.energy_atol,
            "cuda_graph_validation_force_atol_eV_per_A": self.force_atol,
            "cuda_graph_numerical_validation_failure_policy": "report_only",
            "cuda_graph_numerical_validation_within_tolerance": (
                self.replay_stability_passed
                and self.validation_energy_abs_error <= self.energy_atol
                and self.validation_force_max_abs_error <= self.force_atol
            ),
        }


def _validate_request(request: MDRunRequest) -> tuple[str, int | None]:
    if request.model != "orbv3" or request.stage != "opt3":
        raise ValueError(
            f"orb_models.md_stages.opt3 owns orbv3/opt3, got "
            f"{request.model}/{request.stage}"
        )
    if request.backend != "whole-step-cuda-graph":
        raise ValueError("ORBv3 opt3 backend must be 'whole-step-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("ORBv3 Opt3 is CUDA-only; CPU fallback is forbidden")
    if request.config.dtype != "float64":
        raise ValueError("ORBv3 Opt3 requires FP64 MD state")
    if request.atoms.constraints:
        raise NotImplementedError("ORBv3 Opt3 does not support ASE constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError("ORBv3 Opt3 currently captures force-only MD, not stress")
    if bool(_option(request.options, "compute_stress", False)):
        raise NotImplementedError("ORBv3 Opt3 does not capture stress")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("ORBv3 Opt3 forbids TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1")
    forbidden_options = [
        name
        for name in ("compile", "torch_compile", "amp", "tf32")
        if bool(_option(request.options, name, False))
    ]
    if forbidden_options:
        raise ValueError(
            f"ORBv3 Opt3 forbids extra acceleration options: {forbidden_options}"
        )
    if np.any(request.atoms.pbc) and not np.any(np.asarray(request.atoms.cell)):
        raise ValueError("'pbc' is True, but 'cell' is all zeros")

    variant = _normalise_variant(
        _option(request.options, "model_variant", _DEFAULT_MODEL_VARIANT)
    )
    if "conservative" not in variant:
        raise ValueError("ORBv3 Opt3 requires a conservative model_variant")
    filename_variant = _variant_in_filename(request.model_path)
    if filename_variant is not None and filename_variant != variant:
        raise ValueError("model_variant does not match the checkpoint filename")
    if str(_option(request.options, "model_precision", "float32-highest")) != "float32-highest":
        raise ValueError("ORBv3 Opt3 fixes model_precision='float32-highest'")
    if str(_option(request.options, "edge_method", "knn_alchemi")) != "knn_alchemi":
        raise ValueError("ORBv3 Opt3 requires knn_alchemi")
    if _option(request.options, "half_supercell", False) not in {False, None}:
        raise ValueError("ORBv3 Opt3 requires half_supercell=false")
    max_num_neighbors = _option(request.options, "max_num_neighbors", None)
    if max_num_neighbors is not None:
        max_num_neighbors = int(max_num_neighbors)
        if max_num_neighbors < 1:
            raise ValueError("max_num_neighbors must be positive")
    return variant, max_num_neighbors


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run ORBv3 NVT MD with exactly one whole-step CUDA Graph."""

    variant, max_num_neighbors = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("ORBv3 Opt3 requested CUDA, but CUDA is unavailable")
    _configure_precision()
    device = torch.device(request.config.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    config = request.config
    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        rng=np.random.default_rng(config.seed),
    )
    state = GPUMDState(
        positions=torch.as_tensor(
            np.asarray(atoms.positions), dtype=torch.float64, device=device
        ).clone(),
        momenta=torch.as_tensor(
            np.asarray(atoms.get_momenta()), dtype=torch.float64, device=device
        ).clone(),
    )
    masses = torch.as_tensor(
        np.asarray(atoms.get_masses()), dtype=torch.float64, device=device
    ).clone()
    integrator = _build_integrator(request, masses)
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options), device=device
    )
    requested_capacity = request.options.get(
        "cuda_graph_edge_capacity",
        request.options.get("graph_edge_capacity"),
    )
    runner = WholeStepCUDAGraphRunner(
        atoms,
        request.model_path,
        variant=variant,
        device=device,
        max_num_neighbors=max_num_neighbors,
        profiler=profiler,
        requested_edge_capacity=(
            int(requested_capacity) if requested_capacity is not None else None
        ),
        requested_neighbors_per_atom=(
            int(request.options["whole_step_neighbors_per_atom"])
            if "whole_step_neighbors_per_atom" in request.options
            else None
        ),
        edge_margin=float(request.options.get("cuda_graph_edge_margin", 0.10)),
        edge_step=int(request.options.get("cuda_graph_edge_step", 256)),
        capture_warmup=int(request.options.get("cuda_graph_capture_warmup", 3)),
        n_dummy=int(request.options.get("cuda_graph_dummy_atoms", 32)),
        energy_atol=float(request.options.get("cuda_graph_energy_atol_ev", 2e-4)),
        force_atol=float(
            request.options.get("cuda_graph_force_atol_ev_per_a", 2e-4)
        ),
    )
    runner.capture(state, integrator)
    initial_state = runner._state_snapshot(state)
    initial_thermostat = runner._thermostat_snapshot(integrator)

    if config.warmup_steps:
        runner.advance.fill_(1.0)
        for _ in range(config.warmup_steps):
            runner.replay()
        torch.cuda.synchronize(device)
        runner._restore_state(state, initial_state)
        runner._restore_thermostat(integrator, initial_thermostat)

    # Initial force follows the same timed convention as baseline/Opt1/Opt2.
    # The first replay uses advance=0 and evaluates without changing x/p/NHC.
    assert state.forces is not None and state.potential_energy is not None
    state.forces.zero_()
    state.potential_energy.zero_()
    runner.advance.zero_()

    observation_steps = set(config.observation_steps)
    observations = []
    runner.reset_production_stats()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase("initial_force"):
        runner.replay()
    runner.advance.fill_(1.0)
    if config.collect_statistics and 0 in observation_steps:
        observations.append(_record_observation(state, 0, masses))
    for step in range(1, config.steps + 1):
        with profiler.phase("whole_step_replay"):
            runner.replay()
        if config.collect_statistics and step in observation_steps:
            observations.append(_record_observation(state, step, masses))
    torch.cuda.synchronize(device)
    profiler.stop()
    elapsed = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    _validate_final_state(state)
    expected_replays = config.steps + 1
    if runner.production_replays != expected_replays:
        raise RuntimeError(
            "ORBv3 Opt3 production replay mismatch: "
            f"expected={expected_replays}, actual={runner.production_replays}"
        )

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
        metadata={
            "engine": "torch-sim-orb-gpu-resident-whole-step-cuda-graph",
            "backend": "whole-step-cuda-graph",
            "model_path": str(Path(request.model_path).resolve()),
            "model_variant": variant,
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "gpu_resident": True,
            "md_state_dtype": "float64",
            "model_dtype": "float32",
            "model_precision": "float32-highest",
            "compile": False,
            "tf32": False,
            "amp": False,
            "cuda_graph": True,
            "cuda_graph_scope": "whole_step",
            "cuda_graph_neighbor_build_inside": True,
            "cuda_graph_md_update_inside": True,
            "cuda_graph_thermostat_inside": True,
            "cuda_graph_initial_force_inside": True,
            "timed_force_evaluations": expected_replays,
            "fixed_edge_capacity": True,
            "capacity_policy": "single-cap-uniform-per-atom",
            "capacity_overflow_policy": "device-assert-error-no-recapture-no-fallback",
            "transaction_rollback": False,
            "graph_buckets": False,
            "dummy_padding": True,
            "sink_padding": True,
            "compute_stress": False,
            "edge_method": "knn_alchemi",
            "max_num_neighbors": runner.max_num_neighbors,
            "integrator": config.integrator,
            "warmup_steps": config.warmup_steps,
            "model_specific_fusion": False,
            "performance_profile": profiler.summary(synchronize=False),
            **runner.stats(),
        },
    )
    validate_result(request, result)
    return result
