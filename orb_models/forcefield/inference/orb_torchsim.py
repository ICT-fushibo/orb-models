try:
    import torch_sim as ts
    from torch_sim.elastic import voigt_6_to_full_3x3_stress
    from torch_sim.models.interface import ModelInterface
except ImportError as e:
    raise ImportError(
        "torch_sim is required for the OrbTorchSimModel interface. "
        "Install it with: pip install torch-sim-atomistic"
    ) from e

import torch

from orb_models.common.atoms.graph_featurization import (
    AlchemiNeighborListState,
    EdgeCreationMethod,
)
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.d3_model import D3SumModel
from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor


class OrbTorchSimModel(ModelInterface):
    """A TorchSim model interface to Orb, analogous to an ASE calculator."""

    def __init__(
        self,
        model: DirectForcefieldRegressor | ConservativeForcefieldRegressor | D3SumModel,
        atoms_adapter: ForcefieldAtomsAdapter,
        *,
        edge_method: EdgeCreationMethod | None = None,
        max_num_neighbors: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        graph_construction_dtype: torch.dtype | None = None,
        static_alchemi_neighbor_list: bool = False,
    ) -> None:
        """Initialize OrbTorchSimModel interface to Orb models.

        Args:
            model (DirectForcefieldRegressor | ConservativeForcefieldRegressor | D3SumModel): The Orb model to use for predictions.
            atoms_adapter (ForcefieldAtomsAdapter): The adapter to convert between TorchSim SimState and AtomGraphs.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction. Defaults to knn_alchemi.
            max_num_neighbors: Optional override for the adapter's neighbor cap.
            device (torch.device or str, optional): Device to run the model on
            dtype (torch.dtype, optional): Data type for computation
            graph_construction_dtype: Floating point dtype used while constructing
                the neighbor graph. Defaults to TorchSim's state/default dtype.
        """
        super().__init__()
        device = torch.device(device) if device is not None else None
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype

        self.atoms_adapter = atoms_adapter
        self.edge_method = edge_method
        self.max_num_neighbors = max_num_neighbors
        self.graph_construction_dtype = graph_construction_dtype
        self.alchemi_neighbor_state = (
            AlchemiNeighborListState()
            if static_alchemi_neighbor_list and edge_method == "knn_alchemi"
            else None
        )

        self.model = model.to(self._device)  # type: ignore
        self.model = self.model.eval()
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        self.implemented_properties = self.model.properties

        self._compute_stress = "stress" in self.implemented_properties
        self._compute_forces = "forces" in self.implemented_properties

    def _make_batch(self, state: ts.SimState):
        """Create the ORB graph, using a calibrated capacity when configured."""

        return self.atoms_adapter.from_torchsim_state(
            state,
            max_num_neighbors=self.max_num_neighbors,
            edge_method=self.edge_method,
            device=self.device,
            output_dtype=self._dtype,
            graph_construction_dtype=self.graph_construction_dtype,
            alchemi_neighbor_state=self.alchemi_neighbor_state,
            validate_pbc_cell=(
                self.alchemi_neighbor_state is None
                or self.alchemi_neighbor_state.max_neighbors is None
            ),
        )

    def prepare_neighbor_list(self, state: ts.SimState) -> None:
        """Calibrate a static Alchemi capacity before a timed MD loop."""

        if self.alchemi_neighbor_state is not None:
            self._make_batch(state)

    def forward(self, state: ts.SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and other properties.

        Args:
            state (SimState): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3],
                    if compute_stress is True
        """
        if state.device != self._device:
            state = state.to(self._device)

        compute_forces = bool(kwargs.pop("compute_forces", self._compute_forces))
        compute_stress = bool(kwargs.pop("compute_stress", self._compute_stress))
        batch = self._make_batch(state)
        predictions = self.model.predict(
            batch,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
        )
        results = self._get_results(predictions)

        if not compute_forces:
            results.pop("forces", None)
        if "stress" in results and not compute_stress:
            results.pop("stress", None)

        return {k: v.detach() if hasattr(v, "detach") else v for k, v in results.items()}

    def _get_results(self, out: dict[str, torch.Tensor]):
        """Parses the results into a final output dictionary."""
        results = {}
        for prop in self.implemented_properties:
            out_key = "energy" if prop == "free_energy" else prop
            if out_key not in out:
                continue
            results[prop] = torch.atleast_1d(out[out_key])

        # Ensure stress has shape [-1, 3, 3]
        if "stress" in results and results["stress"].shape[-1] == 6:
            results["stress"] = voigt_6_to_full_3x3_stress(torch.atleast_2d(results["stress"]))

        return results
