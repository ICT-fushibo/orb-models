"""ORBv3 Opt2: model-only CUDA Graph inside GPU-resident MD.

Neighbor construction, fixed-capacity input packing, thermostat integration,
and all reporting stay outside the graph.  The captured region contains only
the released eager ORB conservative model and the autograd path required for
forces.  No compile, AMP, TF32, whole-step graph, or model-specific fusion is
enabled at this stage.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.performance import CudaPhaseProfiler, performance_profile_requested
from torch import nn

from orb_models.common.atoms import featurization as atom_feat
from orb_models.common.atoms import graph_featurization as graph_feat
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import segment_ops
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
    GPUMDState,
    OrbTorchSimEvaluator,
    _build_integrator,
    _configure_precision,
    _distribution_version,
    _ensure_evaluated,
    _record_observation,
    _validate_final_state,
)


class CUDAGraphCapacityError(RuntimeError):
    """Raised instead of recapturing or truncating an oversized edge graph."""

    def __init__(self, required_edges: int, edge_capacity: int) -> None:
        self.required_edges = int(required_edges)
        self.edge_capacity = int(edge_capacity)
        super().__init__(
            "ORBv3 CUDA Graph edge capacity exceeded: "
            f"required={required_edges}, capacity={edge_capacity}"
        )


class CUDAGraphValidationError(RuntimeError):
    """Raised when replay is unstable or disagrees with eager ORB."""


def edge_capacity_from_probe(
    maximum_edges: int,
    *,
    margin: float = 0.25,
    edge_step: int = 128,
) -> int:
    """Add headroom and round a probed ragged edge count to a fixed size."""

    if maximum_edges < 1:
        raise ValueError("maximum_edges must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if edge_step < 1:
        raise ValueError("edge_step must be positive")
    required = max(maximum_edges + 1, math.ceil(maximum_edges * (1.0 + margin)))
    return int(math.ceil(required / edge_step) * edge_step)


def _maximum_neighbors_per_atom(
    edge_index: torch.Tensor,
    *,
    num_atoms: int,
) -> int:
    """Return the largest ORB sender degree during a capacity probe."""
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[0], minlength=num_atoms)[:num_atoms]
    return int(counts.max().item())


@torch.no_grad()
def staticize_graph_inputs_(
    static_positions: torch.Tensor,
    static_senders: torch.Tensor,
    static_receivers: torch.Tensor,
    static_unit_shifts: torch.Tensor,
    model_positions: torch.Tensor,
    real_edge_index: torch.Tensor,
    real_unit_shifts: torch.Tensor,
    *,
    dummy_index: int,
    padding_unit_shift: torch.Tensor,
) -> int:
    """Copy one ragged ORB graph into persistent fixed-address buffers."""

    n_real = int(model_positions.shape[0])
    capacity = int(static_senders.shape[0])
    num_edges = int(real_edge_index.shape[1])
    if static_positions.shape != (n_real + 1, 3):
        raise ValueError("static_positions must contain real atoms plus one dummy")
    if static_receivers.shape != (capacity,):
        raise ValueError("static sender/receiver buffers must have the same shape")
    if static_unit_shifts.shape != (capacity, 3):
        raise ValueError("static_unit_shifts must have shape [capacity, 3]")
    if real_edge_index.shape != (2, num_edges):
        raise ValueError("real_edge_index must have shape [2, num_edges]")
    if real_unit_shifts.shape != (num_edges, 3):
        raise ValueError("real_unit_shifts must have shape [num_edges, 3]")
    if padding_unit_shift.shape != (3,):
        raise ValueError("padding_unit_shift must have shape [3]")
    if num_edges > capacity:
        raise CUDAGraphCapacityError(num_edges, capacity)

    static_positions[:n_real].copy_(model_positions)
    static_positions[n_real].zero_()
    if num_edges:
        static_senders[:num_edges].copy_(real_edge_index[0])
        static_receivers[:num_edges].copy_(real_edge_index[1])
        static_unit_shifts[:num_edges].copy_(real_unit_shifts)
    if num_edges < capacity:
        static_senders[num_edges:].fill_(dummy_index)
        static_receivers[num_edges:].fill_(dummy_index)
        static_unit_shifts[num_edges:].copy_(padding_unit_shift)
    return num_edges


class _RealAtomEnergyHead(nn.Module):
    """Run the released energy head while excluding the isolated dummy node."""

    def __init__(self, base: nn.Module, n_real: int, device: torch.device) -> None:
        super().__init__()
        self.base = base
        self.n_real = int(n_real)
        self.register_buffer(
            "real_n_node",
            torch.tensor([self.n_real], dtype=torch.long, device=device),
            persistent=False,
        )

    def forward(self, node_features: torch.Tensor, batch: AtomGraphs) -> torch.Tensor:
        real_features = node_features[: self.n_real]
        aggregated = segment_ops.aggregate_nodes(
            real_features,
            self.real_n_node,
            reduction=self.base.node_aggregation,
        )
        result = self.base.mlp(aggregated).squeeze(-1)
        result = self.base.normalizer.inverse(result)
        if self.base.atom_avg:
            result = result * self.real_n_node
        return result

    def absolute_energy(
        self,
        interaction_energy: torch.Tensor,
        batch: AtomGraphs,
        fp64: bool = True,
    ) -> torch.Tensor:
        reference = self.base.reference(
            batch.atomic_numbers[: self.n_real], self.real_n_node
        )
        if fp64:
            return interaction_energy.double() + reference.double()
        return interaction_energy + reference.to(interaction_energy.dtype)


class _RealAtomPairRepulsion(nn.Module):
    """Preserve the released ZBL reduction when one dummy node is added.

    The April 2025 ORBv3 checkpoints intentionally restore the legacy
    ``mean`` ZBL aggregation.  Padding changes ``batch.n_node`` from ``N`` to
    ``N + 1``; the isolated dummy has zero ZBL energy, but would still change
    that denominator.  Rescaling the wrapped result restores the exact
    real-atom reduction while retaining the released ZBL implementation.
    """

    def __init__(self, base: nn.Module, n_real: int) -> None:
        super().__init__()
        self.base = base
        reduction = getattr(base, "node_aggregation", None)
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"unsupported pair-repulsion aggregation {reduction!r}")
        self.scale = (n_real + 1) / n_real if reduction == "mean" else 1.0

    def forward(self, batch: AtomGraphs) -> dict[str, torch.Tensor]:
        output = self.base(batch)
        return {**output, "energy": output["energy"] * self.scale}


class _ORBForceOnlyModel(nn.Module):
    """Exact fixed-cell force path without ORB's stress/equigrad auxiliaries.

    The released conservative regressor always constructs a differentiable
    strain and rotation, even when ``compute_stress=False``.  The latter calls
    ``torch.matrix_exp``, whose implementation performs a capture-unsafe host
    scalar transfer.  At zero strain/rotation both transformations are the
    identity, so fixed-cell force-only inference can build the same edge
    vectors directly and differentiate energy only with respect to positions.
    Capture-time validation still compares this path with the official eager
    regressor.
    """

    def __init__(self, regressor: ConservativeForcefieldRegressor) -> None:
        super().__init__()
        self.regressor = regressor

    def forward(self, batch: AtomGraphs) -> tuple[torch.Tensor, torch.Tensor]:
        positions = batch.node_features["positions"]
        positions.requires_grad_(True)
        cell = batch.system_features["cell"].reshape(-1, 3, 3)
        if cell.shape[0] != 1:
            raise RuntimeError("ORBv3 Opt2 currently captures one MD system")
        unit_shifts = batch.edge_features["unit_shifts"]
        shifts = torch.matmul(unit_shifts, cell[0])
        vectors = (
            positions[batch.receivers]
            - positions[batch.senders]
            + shifts
        )
        batch.edge_features["vectors"] = vectors
        batch.node_features["strained_positions"] = positions
        batch.system_features["strained_cell"] = cell

        out = self.regressor.model(batch)
        node_features = out["node_features"]
        energy_head = self.regressor.heads["energy"]
        interaction_energy = energy_head(node_features, batch)
        if self.regressor.pair_repulsion:
            interaction_energy = interaction_energy + (
                self.regressor.pair_repulsion_fn(batch)["energy"]
            )
        energy = energy_head.absolute_energy(
            interaction_energy, batch, fp64=True
        )
        gradient = torch.autograd.grad(
            interaction_energy.sum(),
            positions,
            create_graph=False,
            retain_graph=False,
        )[0]
        if gradient is None:
            raise RuntimeError("ORBv3 Opt2 force-only VJP returned no gradient")
        return gradient.neg(), energy


def _pad_node_tensor(value: torch.Tensor, n_real: int) -> torch.Tensor:
    if value.ndim and value.shape[0] == n_real:
        return torch.cat((value, value[:1]), dim=0)
    return value.clone()


class ModelOnlyCUDAGraphEvaluator(OrbTorchSimEvaluator):
    """ORB evaluator with eager Alchemi edges and one model CUDA Graph."""

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
        edge_margin: float,
        edge_step: int,
        track_neighbor_capacity: bool,
        capture_warmup: int,
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
        if edge_margin < 0 or edge_step < 1:
            raise ValueError("CUDA Graph edge margin/step is invalid")
        if capture_warmup < 0:
            raise ValueError("cuda_graph_capture_warmup cannot be negative")
        if energy_atol < 0 or force_atol < 0:
            raise ValueError("CUDA Graph validation tolerances cannot be negative")
        if not isinstance(self.model.model, ConservativeForcefieldRegressor):
            raise TypeError("ORBv3 Opt2 requires a conservative released checkpoint")
        if self.model.model.coulomb_module is not None:
            raise NotImplementedError("ORBv3 Opt2 dummy padding does not support CoulombModule")
        if "latent_charges" in self.model.model.heads or "latent_spins" in self.model.model.heads:
            raise NotImplementedError(
                "ORBv3 Opt2 dummy padding does not support latent charge/spin heads"
            )

        self.requested_edge_capacity = requested_edge_capacity
        self.edge_margin = float(edge_margin)
        self.edge_step = int(edge_step)
        self.track_neighbor_capacity = bool(track_neighbor_capacity)
        self.capture_warmup = int(capture_warmup)
        self.energy_atol = float(energy_atol)
        self.force_atol = float(force_atol)
        self.n_real = self.num_atoms
        self.dummy_index = self.n_real

        # Build one setup-only eager graph to cache immutable model features.
        self.sim_state.positions = self.sim_state.positions.contiguous()
        template = self.model._make_batch(self.sim_state)
        self.template = template
        self.real_n_node = template.n_node
        self.node_batch_index = template.node_batch_index
        self.cell_md = self.sim_state.row_vector_cell.contiguous()
        self.pbc_md = torch.repeat_interleave(
            self.sim_state.pbc.view(-1, 3), self.real_n_node.shape[0], dim=0
        )
        self.radius = float(template.radius)
        self.model_max_neighbors = int(self.max_num_neighbors)

        self.edge_capacity = 0
        self.static_positions: torch.Tensor | None = None
        self.static_senders: torch.Tensor | None = None
        self.static_receivers: torch.Tensor | None = None
        self.static_unit_shifts: torch.Tensor | None = None
        self.padding_unit_shift: torch.Tensor | None = None
        self.static_batch: AtomGraphs | None = None
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.static_energy: torch.Tensor | None = None
        self.static_forces: torch.Tensor | None = None
        self.force_only_model: _ORBForceOnlyModel | None = None
        self.captured = False

        self.capture_count = 0
        self.capture_wall_time_s = 0.0
        self.total_replays = 0
        self.production_calls = 0
        self.production_replays = 0
        self.capacity_misses = 0
        self.min_real_edges: int | None = None
        self.max_real_edges: int | None = None
        self.initial_max_neighbors_per_atom: int | None = None
        self.max_neighbors_per_atom: int | None = None
        self.output_addresses_stable = False
        self.input_addresses_stable = False
        self._capture_input_addresses: tuple[tuple[str, int], ...] | None = None
        self.replay_stability_passed = False
        self.replay_energy_abs_error = 0.0
        self.replay_force_max_abs_error = 0.0
        self.validation_energy_abs_error = 0.0
        self.validation_force_max_abs_error = 0.0

    def _build_real_inputs(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.profiler.phase("neighbor_list"):
            wrapped_positions = atom_feat.batch_map_to_pbc_cell(
                positions,
                self.cell_md,
                self.pbc_md,
                self.real_n_node,
            )
            edge_index, _edge_vectors, unit_shifts, _n_edges = (
                graph_feat.batch_compute_pbc_radius_graph(
                    positions=wrapped_positions.contiguous(),
                    cells=self.cell_md,
                    pbcs=self.pbc_md,
                    radius=self.radius,
                    n_node=self.real_n_node,
                    node_batch_index=self.node_batch_index,
                    max_number_neighbors=self.template.max_num_neighbors,
                    edge_method="knn_alchemi",
                    device=self.device,
                    float_dtype=torch.float32,
                    static_max_number_neighbors=self.model_max_neighbors,
                    alchemi_neighbor_state=self.model.alchemi_neighbor_state,
                    validate_pbc_cell=False,
                )
            )
        return wrapped_positions.to(torch.float32), edge_index, unit_shifts.to(torch.float32)

    def _initialize_static_batch(self, capacity: int) -> None:
        self.edge_capacity = int(capacity)
        self.static_positions = torch.zeros(
            (self.n_real + 1, 3), dtype=torch.float32, device=self.device
        )
        self.static_senders = torch.empty(capacity, dtype=torch.long, device=self.device)
        self.static_receivers = torch.empty(capacity, dtype=torch.long, device=self.device)
        self.static_unit_shifts = torch.zeros(
            (capacity, 3), dtype=torch.float32, device=self.device
        )

        model_cell = self.template.system_features["cell"]
        cell_norms = torch.linalg.vector_norm(model_cell.reshape(-1, 3), dim=1)
        axis = int(cell_norms.argmax().item())
        axis_length = float(cell_norms[axis].item())
        if axis_length <= 0:
            raise ValueError("ORBv3 Opt2 requires a nonzero simulation cell")
        multiple = max(1, math.ceil((self.radius + 1.0) / axis_length))
        self.padding_unit_shift = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.padding_unit_shift[axis] = float(multiple)

        node_features = {
            name: (
                self.static_positions
                if name == "positions"
                else _pad_node_tensor(value, self.n_real)
            )
            for name, value in self.template.node_features.items()
        }
        system_features = {
            name: value.clone() for name, value in self.template.system_features.items()
        }
        self.static_batch = AtomGraphs(
            senders=self.static_senders,
            receivers=self.static_receivers,
            n_node=torch.tensor(
                [self.n_real + 1], dtype=torch.long, device=self.device
            ),
            n_edge=torch.tensor([capacity], dtype=torch.long, device=self.device),
            node_features=node_features,
            edge_features={
                "vectors": torch.zeros(
                    (capacity, 3), dtype=torch.float32, device=self.device
                ),
                "unit_shifts": self.static_unit_shifts,
            },
            system_features=system_features,
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self.radius,
            max_num_neighbors=self.template.max_num_neighbors.clone(),
        )

    def _staticize(
        self,
        model_positions: torch.Tensor,
        edge_index: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> int:
        assert self.static_positions is not None
        assert self.static_senders is not None
        assert self.static_receivers is not None
        assert self.static_unit_shifts is not None
        assert self.padding_unit_shift is not None
        return staticize_graph_inputs_(
            self.static_positions,
            self.static_senders,
            self.static_receivers,
            self.static_unit_shifts,
            model_positions,
            edge_index,
            unit_shifts,
            dummy_index=self.dummy_index,
            padding_unit_shift=self.padding_unit_shift,
        )

    def _static_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.static_batch is not None
        assert self.static_positions is not None
        assert self.static_senders is not None
        assert self.static_receivers is not None
        assert self.static_unit_shifts is not None
        self.static_batch.node_features["positions"] = self.static_positions
        self.static_batch.senders = self.static_senders
        self.static_batch.receivers = self.static_receivers
        self.static_batch.edge_features["unit_shifts"] = self.static_unit_shifts
        if self.force_only_model is None:
            raise RuntimeError("ORBv3 Opt2 force-only model is not initialized")
        with torch.enable_grad():
            forces, energy = self.force_only_model(self.static_batch)
        return forces[: self.n_real].detach(), energy.detach()

    def _input_addresses(self) -> tuple[tuple[str, int], ...]:
        assert self.static_batch is not None
        tensors: dict[str, torch.Tensor] = {
            "senders": self.static_batch.senders,
            "receivers": self.static_batch.receivers,
            "n_node": self.static_batch.n_node,
            "n_edge": self.static_batch.n_edge,
        }
        for namespace in ("node_features", "edge_features", "system_features"):
            values = getattr(self.static_batch, namespace)
            tensors.update(
                {
                    f"{namespace}.{name}": value
                    for name, value in values.items()
                    if isinstance(value, torch.Tensor)
                }
            )
        return tuple(
            sorted((name, tensor.data_ptr()) for name, tensor in tensors.items())
        )

    def capture(self, positions: torch.Tensor) -> None:
        """Capture once and validate fixed-buffer replay against eager Opt1."""

        if self.captured:
            raise RuntimeError("ORBv3 CUDA Graph has already been captured")
        eager_forces, eager_energy, _stress = OrbTorchSimEvaluator.__call__(self, positions)
        model_positions, edge_index, unit_shifts = self._build_real_inputs(positions)
        probed_edges = int(edge_index.shape[1])
        if self.track_neighbor_capacity:
            self.initial_max_neighbors_per_atom = _maximum_neighbors_per_atom(
                edge_index,
                num_atoms=self.n_real,
            )
            self.max_neighbors_per_atom = self.initial_max_neighbors_per_atom
        capacity = self.requested_edge_capacity or edge_capacity_from_probe(
            probed_edges,
            margin=self.edge_margin,
            edge_step=self.edge_step,
        )
        if capacity < probed_edges:
            raise CUDAGraphCapacityError(probed_edges, capacity)

        energy_head = self.model.model.heads["energy"]
        self.model.model.heads["energy"] = _RealAtomEnergyHead(
            energy_head, self.n_real, self.device
        )
        if self.model.model.pair_repulsion:
            self.model.model.pair_repulsion_fn = _RealAtomPairRepulsion(
                self.model.model.pair_repulsion_fn, self.n_real
            )
        self.force_only_model = _ORBForceOnlyModel(self.model.model).eval()
        self._initialize_static_batch(capacity)
        self._staticize(model_positions, edge_index, unit_shifts)

        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(self.capture_warmup):
                self._static_forward()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            static_forces, static_energy = self._static_forward()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - started
        self.cuda_graph = graph
        self.static_forces = static_forces
        self.static_energy = static_energy
        self._capture_input_addresses = self._input_addresses()
        self.capture_count = 1
        self.captured = True

        force_address = static_forces.data_ptr()
        energy_address = static_energy.data_ptr()
        graph.replay()
        replay_forces = static_forces.clone()
        replay_energy = static_energy.clone()
        graph.replay()
        second_forces = static_forces.clone()
        second_energy = static_energy.clone()
        torch.cuda.synchronize(self.device)
        self.total_replays += 2
        self.output_addresses_stable = (
            static_forces.data_ptr() == force_address
            and static_energy.data_ptr() == energy_address
        )
        self.input_addresses_stable = (
            self._input_addresses() == self._capture_input_addresses
        )
        self.replay_energy_abs_error = float(
            (replay_energy - second_energy).abs().max().item()
        )
        self.replay_force_max_abs_error = float(
            (replay_forces - second_forces).abs().max().item()
        )
        self.validation_energy_abs_error = float(
            (replay_energy.reshape(-1)[0].to(torch.float64) - eager_energy).abs().item()
        )
        self.validation_force_max_abs_error = float(
            (replay_forces.to(torch.float64) - eager_forces).abs().max().item()
        )
        self.replay_stability_passed = (
            self.replay_energy_abs_error <= self.energy_atol
            and self.replay_force_max_abs_error <= self.force_atol
        )
        if not self.output_addresses_stable:
            raise CUDAGraphValidationError("ORBv3 CUDA Graph output addresses changed")
        if not self.input_addresses_stable:
            raise CUDAGraphValidationError("ORBv3 CUDA Graph input addresses changed")
        for name, value in (
            ("replay energy", replay_energy),
            ("replay forces", replay_forces),
            ("second replay energy", second_energy),
            ("second replay forces", second_forces),
        ):
            if not bool(torch.isfinite(value).all()):
                raise CUDAGraphValidationError(
                    f"ORBv3 CUDA Graph produced non-finite {name}"
                )
        self.numerical_validation_within_tolerance = (
            self.replay_stability_passed
            and self.validation_energy_abs_error <= self.energy_atol
            and self.validation_force_max_abs_error <= self.force_atol
        )
    def reset_production_stats(self) -> None:
        self.production_calls = 0
        self.production_replays = 0
        self.capacity_misses = 0
        self.min_real_edges = None
        self.max_real_edges = None
        self.max_neighbors_per_atom = self.initial_max_neighbors_per_atom

    def __call__(self, positions: torch.Tensor):
        if not self.captured or self.cuda_graph is None:
            raise RuntimeError("ORBv3 CUDA Graph must be captured before replay")
        model_positions, edge_index, unit_shifts = self._build_real_inputs(positions)
        num_edges = int(edge_index.shape[1])
        if self.track_neighbor_capacity:
            maximum = _maximum_neighbors_per_atom(
                edge_index,
                num_atoms=self.n_real,
            )
            self.max_neighbors_per_atom = (
                maximum
                if self.max_neighbors_per_atom is None
                else max(self.max_neighbors_per_atom, maximum)
            )
        self.production_calls += 1
        if num_edges > self.edge_capacity:
            self.capacity_misses += 1
            raise CUDAGraphCapacityError(num_edges, self.edge_capacity)
        with self.profiler.phase("model_input"):
            self._staticize(model_positions, edge_index, unit_shifts)
        if self._input_addresses() != self._capture_input_addresses:
            raise CUDAGraphValidationError(
                "ORBv3 CUDA Graph static input address changed"
            )
        with self.profiler.phase("calculator_force"):
            self.cuda_graph.replay()
        self.total_replays += 1
        self.production_replays += 1
        self.min_real_edges = (
            num_edges if self.min_real_edges is None else min(self.min_real_edges, num_edges)
        )
        self.max_real_edges = (
            num_edges if self.max_real_edges is None else max(self.max_real_edges, num_edges)
        )
        assert self.static_forces is not None and self.static_energy is not None
        return (
            self.static_forces.to(torch.float64),
            self.static_energy.reshape(-1)[0].to(torch.float64),
            None,
        )

    def stats(self) -> dict[str, Any]:
        hit_rate = (
            self.production_replays / self.production_calls
            if self.production_calls
            else 0.0
        )
        return {
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_production_capture_count": 0,
            "cuda_graph_total_replays": self.total_replays,
            "cuda_graph_production_calls": self.production_calls,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_capacity_misses": self.capacity_misses,
            "cuda_graph_hit_rate": hit_rate,
            "cuda_graph_edge_capacity": self.edge_capacity,
            "cuda_graph_min_real_edges": self.min_real_edges,
            "cuda_graph_max_real_edges": self.max_real_edges,
            "cuda_graph_max_neighbors_per_atom": self.max_neighbors_per_atom,
            "capacity_probe_collect_per_atom": self.track_neighbor_capacity,
            "cuda_graph_dummy_atoms": 1,
            "cuda_graph_capture_warmup": self.capture_warmup,
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_replay_output_addresses_stable": self.output_addresses_stable,
            "cuda_graph_input_addresses_stable": self.input_addresses_stable,
            "cuda_graph_replay_stability_pass": self.replay_stability_passed,
            "cuda_graph_replay_stability_energy_abs_error_eV": self.replay_energy_abs_error,
            "cuda_graph_replay_stability_force_max_abs_error_eV_per_A": (
                self.replay_force_max_abs_error
            ),
            "cuda_graph_validation_energy_abs_error_eV": self.validation_energy_abs_error,
            "cuda_graph_validation_force_max_abs_error_eV_per_A": (
                self.validation_force_max_abs_error
            ),
            "cuda_graph_validation_energy_atol_eV": self.energy_atol,
            "cuda_graph_validation_force_atol_eV_per_A": self.force_atol,
            "cuda_graph_force_path": "fixed-cell-position-only-vjp",
            "cuda_graph_stress_rotation_auxiliaries": False,
            "cuda_graph_numerical_validation_failure_policy": "report_only",
            "cuda_graph_numerical_validation_within_tolerance": (
                self.numerical_validation_within_tolerance
            ),
        }


def _validate_request(request: MDRunRequest) -> tuple[str, int | None]:
    if request.model != "orbv3" or request.stage != "opt2":
        raise ValueError(
            f"orb_models.md_stages.opt2 owns orbv3/opt2, got "
            f"{request.model}/{request.stage}"
        )
    if request.backend != "model-only-cuda-graph":
        raise ValueError("ORBv3 opt2 backend must be 'model-only-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("ORBv3 Opt2 is CUDA-only; CPU fallback is forbidden")
    if request.config.dtype != "float64":
        raise ValueError("ORBv3 Opt2 requires FP64 MD state")
    if request.atoms.constraints:
        raise NotImplementedError("ORBv3 Opt2 does not support ASE constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError("ORBv3 Opt2 currently captures force-only MD, not stress")
    if bool(_option(request.options, "compute_stress", False)):
        raise NotImplementedError("ORBv3 Opt2 does not capture stress")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("ORBv3 Opt2 forbids TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1")
    forbidden_options = [
        name
        for name in ("compile", "torch_compile", "amp", "tf32")
        if bool(_option(request.options, name, False))
    ]
    if forbidden_options:
        raise ValueError(
            f"ORBv3 Opt2 forbids extra acceleration options: {forbidden_options}"
        )
    if np.any(request.atoms.pbc) and not np.any(np.asarray(request.atoms.cell)):
        raise ValueError("'pbc' is True, but 'cell' is all zeros")

    variant = _normalise_variant(
        _option(request.options, "model_variant", _DEFAULT_MODEL_VARIANT)
    )
    if "conservative" not in variant:
        raise ValueError("ORBv3 Opt2 requires a conservative model_variant")
    filename_variant = _variant_in_filename(request.model_path)
    if filename_variant is not None and filename_variant != variant:
        raise ValueError("model_variant does not match the checkpoint filename")
    precision = str(_option(request.options, "model_precision", "float32-highest"))
    if precision != "float32-highest":
        raise ValueError("ORBv3 Opt2 fixes model_precision='float32-highest'")
    edge_method = str(_option(request.options, "edge_method", "knn_alchemi"))
    if edge_method != "knn_alchemi":
        raise ValueError("ORBv3 Opt2 requires knn_alchemi")
    if _option(request.options, "half_supercell", False) not in {False, None}:
        raise ValueError("ORBv3 Opt2 requires half_supercell=false")
    max_num_neighbors = _option(request.options, "max_num_neighbors", None)
    if max_num_neighbors is not None:
        max_num_neighbors = int(max_num_neighbors)
        if max_num_neighbors < 1:
            raise ValueError("max_num_neighbors must be positive")
    return variant, max_num_neighbors


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run ORBv3 GPU-resident MD with a strict model-only CUDA Graph."""

    variant, max_num_neighbors = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("ORBv3 Opt2 requested CUDA, but CUDA is unavailable")
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
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options), device=device
    )
    requested_capacity = request.options.get("cuda_graph_edge_capacity")
    evaluator = ModelOnlyCUDAGraphEvaluator(
        atoms,
        request.model_path,
        variant=variant,
        device=device,
        max_num_neighbors=max_num_neighbors,
        profiler=profiler,
        requested_edge_capacity=(
            int(requested_capacity) if requested_capacity is not None else None
        ),
        edge_margin=float(request.options.get("cuda_graph_edge_margin", 0.25)),
        edge_step=int(request.options.get("cuda_graph_edge_step", 128)),
        track_neighbor_capacity=bool(
            request.options.get("capacity_probe_collect_per_atom", False)
        ),
        capture_warmup=int(request.options.get("cuda_graph_capture_warmup", 3)),
        energy_atol=float(request.options.get("cuda_graph_energy_atol_ev", 2e-4)),
        force_atol=float(
            request.options.get("cuda_graph_force_atol_ev_per_a", 2e-4)
        ),
    )
    evaluator.capture(positions)
    integrator = _build_integrator(request, masses)

    if config.warmup_steps:
        for _ in range(config.warmup_steps):
            integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_(initial_state)
        integrator.reset()

    observation_steps = set(config.observation_steps)
    observations = []
    evaluator.reset_production_stats()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase("initial_force"):
        _ensure_evaluated(state, evaluator)
    if config.collect_statistics and 0 in observation_steps:
        observations.append(_record_observation(state, 0, masses))
    for step in range(1, config.steps + 1):
        with profiler.phase("md_step"):
            integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_record_observation(state, step, masses))
    torch.cuda.synchronize(device)
    profiler.stop()
    elapsed = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    _validate_final_state(state)

    expected_replays = config.steps + 1
    if evaluator.production_replays != expected_replays:
        raise RuntimeError(
            "ORBv3 Opt2 production replay mismatch: "
            f"expected={expected_replays}, actual={evaluator.production_replays}"
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
            "engine": "torch-sim-orb-gpu-resident-model-cuda-graph",
            "backend": "model-only-cuda-graph",
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
            "cuda_graph_scope": "model_only",
            "cuda_graph_neighbor_build_outside": True,
            "cuda_graph_md_update_outside": True,
            "fixed_edge_capacity": True,
            "capacity_overflow_policy": "error_no_recapture_no_fallback",
            "dummy_padding": True,
            "compute_stress": False,
            "edge_method": "knn_alchemi",
            "max_num_neighbors": evaluator.max_num_neighbors,
            "integrator": config.integrator,
            "warmup_steps": config.warmup_steps,
            "model_specific_fusion": False,
            "performance_profile": profiler.summary(synchronize=False),
            **evaluator.stats(),
        },
    )
    validate_result(request, result)
    return result
