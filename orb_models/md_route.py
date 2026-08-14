"""Stable ORBv3 MD route for the shared acceleration benchmark.

The permanent baseline is deliberately the uncompiled ASE calculator.  ORB's
pretrained helpers compile on CUDA when ``compile`` is left as ``None``, so this
route always passes the flag explicitly and records every graph/model choice in
the result metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    run_ase_baseline,
    run_optimized_stage,
)


_DEFAULT_MODEL_VARIANT = "orb-v3-conservative-inf-mpa"
_MODEL_LOADERS = {
    "orb-v3-conservative-inf-mpa": "orb_v3_conservative_inf_mpa",
    "orb-v3-conservative-inf-omat": "orb_v3_conservative_inf_omat",
    "orb-v3-conservative-20-mpa": "orb_v3_conservative_20_mpa",
    "orb-v3-conservative-20-omat": "orb_v3_conservative_20_omat",
    "orb-v3-direct-inf-mpa": "orb_v3_direct_inf_mpa",
    "orb-v3-direct-inf-omat": "orb_v3_direct_inf_omat",
    "orb-v3-direct-20-mpa": "orb_v3_direct_20_mpa",
    "orb-v3-direct-20-omat": "orb_v3_direct_20_omat",
}
_EDGE_METHODS = {
    "knn_alchemi",
    "knn_brute_force",
    "knn_scipy",
    "knn_cuml_brute",
    "knn_cuml_rbc",
}


def _option(options: dict[str, Any], name: str, default: Any) -> Any:
    """Read an ORB-specific option while accepting a namespaced spelling."""
    return options.get(f"orb_{name}", options.get(name, default))


def _normalise_variant(value: object) -> str:
    variant = str(value).lower().replace("_", "-")
    if variant not in _MODEL_LOADERS:
        raise ValueError(
            f"unsupported ORBv3 model_variant {value!r}; choose one of "
            f"{sorted(_MODEL_LOADERS)}"
        )
    return variant


def _variant_in_filename(model_path: str) -> str | None:
    """Identify official checkpoint names and catch accidental MPA/OMat swaps."""
    filename = Path(model_path).name.lower().replace("_", "-")
    return next((variant for variant in _MODEL_LOADERS if variant in filename), None)


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "orbv3":
        raise ValueError(f"orb_models.md_route does not own model {request.model!r}")
    if request.stage != "baseline":
        return run_optimized_stage(request, module_prefix="orb_models.md_stages")

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    if request.backend not in {"eager", "compile"}:
        raise ValueError("ORBv3 baseline backend must be eager or compile")

    variant = _normalise_variant(
        _option(request.options, "model_variant", _DEFAULT_MODEL_VARIANT)
    )
    filename_variant = _variant_in_filename(request.model_path)
    if filename_variant is not None and filename_variant != variant:
        raise ValueError(
            f"model_variant={variant!r} does not match checkpoint filename "
            f"{Path(request.model_path).name!r} (looks like {filename_variant!r})"
        )

    edge_method = str(_option(request.options, "edge_method", "knn_alchemi"))
    if edge_method not in _EDGE_METHODS:
        raise ValueError(
            f"unsupported ORB edge_method {edge_method!r}; choose one of "
            f"{sorted(_EDGE_METHODS)}"
        )
    if edge_method == "knn_scipy" and request.config.device != "cpu":
        raise ValueError("ORB knn_scipy edge construction requires --device cpu")

    half_supercell = _option(request.options, "half_supercell", False)
    if half_supercell is not None and not isinstance(half_supercell, bool):
        raise TypeError("ORB half_supercell must be true, false, or null")
    max_num_neighbors = _option(request.options, "max_num_neighbors", None)
    if max_num_neighbors is not None:
        max_num_neighbors = int(max_num_neighbors)
        if max_num_neighbors < 1:
            raise ValueError("ORB max_num_neighbors must be positive")

    configure_torch_baseline()
    # Matbench's global ``dtype=float64`` controls the MD runner but is not
    # forwarded by its ORB calculator factory.  Keep the released ORB model in
    # FP32 by default and require an explicit route option to change the model
    # arithmetic.
    precision = str(_option(request.options, "model_precision", "float32-highest"))
    if precision not in {"float32-highest", "float32-high", "float64"}:
        raise ValueError(
            "ORB model_precision must be float32-highest, float32-high, or float64"
        )
    loader = getattr(pretrained, _MODEL_LOADERS[variant])
    model, adapter = loader(
        weights_path=request.model_path,
        device=request.config.device,
        precision=precision,
        compile=request.backend == "compile",
    )
    calculator = ORBCalculator(
        model,
        atoms_adapter=adapter,
        device=request.config.device,
        edge_method=edge_method,
        max_num_neighbors=max_num_neighbors,
        half_supercell=half_supercell,
    )
    return run_ase_baseline(
        request,
        calculator,
        metadata={
            "model_variant": variant,
            "compile": request.backend == "compile",
            "precision": precision,
            "md_config_dtype": request.config.dtype,
            "edge_method": edge_method,
            "max_num_neighbors": max_num_neighbors or adapter.max_num_neighbors,
            "half_supercell": half_supercell,
        },
    )
