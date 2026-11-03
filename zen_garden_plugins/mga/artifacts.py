"""Result files of the polytope modes: polytope.npz and diagnostics.csv.

Both are written to the summary folder <model_name>_<mode>_summary/<subfolder>,
a sibling of the per-solve Postprocess folders. The npz schema is owned by
``polytope_io``.
"""

import importlib.metadata
import logging
from pathlib import Path

from . import normalisation as norm
from .axes import axis_physical_unit
from .polytope_io import (
    NODE_CAPACITY_RATIO,
    NODE_OUTPUT_RATIO,
    TOTAL_COST,
    Polytope,
    save_polytope,
)

logger = logging.getLogger(__name__)

RESULT_PACKAGES = ("zen-garden", "pyoNearOpt", "linopy")


def package_versions(packages=RESULT_PACKAGES) -> dict:
    """Installed versions of the packages that determine a run's results."""
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def summary_dir(mga, mode: str) -> Path:
    """Create and return the summary folder of a polytope mode."""
    out = (
        Path(mga.optimization_step.config.analysis.folder_output)
        / f"{mga.postprocess_ctx['model_name']}_{mode}_summary"
        / mga.postprocess_ctx["subfolder"]
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def polytope_metadata(mga) -> dict:
    """Per-axis metadata and the normalisation description of a run."""
    units = mga.optimization_model.variables.units
    ureg = mga.optimization_step.unit_converter.ureg
    share = mga.normalisation == "share"
    per_axes = mga.normalisation == "per_axes"
    denominators = mga.evaluator.ratio_denominator
    return {
        "axes": [
            {
                "name": axis.name,
                "kind": axis.kind,
                "members": list(axis.members),
                "capacity_type": axis.capacity_type,
                "period": list(axis.period) if axis.period is not None else None,
                "technologies": (
                    list(axis.technologies) if axis.technologies is not None else None
                ),
                "denominator_technologies": (
                    list(axis.denominator_technologies)
                    if axis.denominator_technologies is not None
                    else None
                ),
                "capacity_ratio_denominator_baseline": (
                    float(denominators[axis.name])
                    if axis.kind == NODE_CAPACITY_RATIO
                    else None
                ),
                "output_ratio_denominator_baseline": (
                    float(denominators[axis.name])
                    if axis.kind == NODE_OUTPUT_RATIO
                    else None
                ),
                "unit": axis_physical_unit(axis, units, ureg),
                "share_reference_total": (
                    float(mga.share_reference[i])
                    if share and axis.kind != TOTAL_COST
                    else None
                ),
                "share_reference_total_raw": (
                    float(mga.share_reference_raw[i])
                    if share and axis.kind != TOTAL_COST
                    else None
                ),
                "per_axes_reference": (
                    float(mga.per_axes_reference[axis.name])
                    if per_axes and axis.kind != TOTAL_COST
                    else None
                ),
            }
            for i, axis in enumerate(mga.axes)
        ],
        "normalisation": norm.describe(mga.normalisation),
    }


def save_artifacts(
    mga,
    out: Path,
    poly,
    point_origin,
    convergence_threshold: float,
    converged: bool,
    final_gap: float,
    run_info: dict,
    diagnostics=None,
) -> None:
    """Write polytope.npz and, if there are diagnostics, diagnostics.csv.

    Args:
        mga: The ``MGA`` of the run.
        out: The summary folder.
        poly: The final pyoNearOpt approximation.
        point_origin: Labels of the initial inner points.
        convergence_threshold: The run's target for its convergence metric.
        converged: Whether the run converged.
        final_gap: Final gap between the outer and inner approximation.
        run_info: How the run was configured and ended (stored as run_json).
        diagnostics: Per-iteration DataFrame, or None if no iteration finished.
    """
    meta = polytope_metadata(mga)
    # X grows by one certified point per iteration after the initial set.
    origins = list(point_origin)
    origins += ["iterate"] * (poly.X.shape[0] - len(origins))
    save_polytope(
        out / "polytope.npz",
        Polytope(
            A=poly.A,
            b=poly.b,
            X=poly.X,
            names=list(mga.z_names),
            kinds=[axis.kind for axis in mga.axes],
            units=[axis_meta["unit"] or "" for axis_meta in meta["axes"]],
            scale=mga.scale,
            offset=mga.offset,
            bounds_phys=mga.bounds_phys,
            z_star_phys=mga.z_star_phys,
            c_star=float(mga.c_star),
            epsilon=float(mga.epsilon),
            convergence_threshold=float(convergence_threshold),
            converged=bool(converged),
            final_gap=float(final_gap),
            n_initial_rows=int(mga.n_initial_rows),
            point_origin=origins,
            meta=meta,
            run=run_info,
        ),
    )
    mode = run_info.get("mode", "oracle")
    if diagnostics is not None:
        diagnostics.to_csv(out / "diagnostics.csv", index=False)
        log = logger.info if converged else logger.warning
        log(
            f"MGA {mode}: {'CONVERGED' if converged else 'did NOT converge'} "
            f"after {len(diagnostics)} iterations, final gap = {final_gap:.4g} "
            f"(threshold = {convergence_threshold:.4g})."
        )
    else:
        logger.warning(
            f"MGA {mode}: no diagnostics to save (the refinement loop raised "
            f"before any iteration completed); see traceback above."
        )
    logger.info(f"MGA {mode}: artifacts saved to {out}")
