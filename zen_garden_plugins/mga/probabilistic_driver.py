"""Probabilistic-mode driver for the MGA plugin.

Wires MGA.support_function into pyoNearOpt's probabilistic support-function
exploration (`prob_direction_explore`). Unlike ORACLE, this family never
solves a bilevel/MILP model: each iteration samples candidate directions,
picks the one with the largest observed gap between the inner and outer
polytope approximations, and probes it with a single support-function LP
(`MGA.support_function`) on the ZEN-garden model. pyoNearOpt drives the loop
itself (`explorer.explore(n_iter)`) and owns the polytope bookkeeping
(`poly_approx.add_point`/`add_cut`).

No Gurobi dependency: the only ZEN-garden-model solve is the LP inside
`support_function`, using whatever solver the run is already configured
with; pyoNearOpt's own CI-sampling LPs (`support_fn_inner`/`support_fn_outer`)
default to scipy's HiGHS. There is therefore no certificate solve here, unlike
oracle_driver.py's `_run_final_certificate`.

pyoNearOpt is imported lazily so that "weights" mode works without it
installed, exactly as oracle_driver.py does.
"""

import importlib.metadata
import logging
from pathlib import Path

import pandas as pd

from .polytope_io import Polytope, save_polytope


def run_probabilistic_mode(mga, cfg):
    """Run the probabilistic support-function pipeline on a set-up MGA instance.

    Returns the summary directory holding the polytope npz and diagnostics.
    """
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.exploration_methods.prob_ORACLE import prob_direction_explore
    from pyoNearOpt.polytope_approximation.approximation_class import approximation

    if not mga.design_axes:
        raise ValueError(
            "MGA probabilistic: no exploration axes configured; set "
            "plugins.mga.axes.technologies and/or axes.carrier_imports."
        )
    tolerance_prob = float(cfg["tolerance_prob"])  # required, deliberately no default
    max_iterations = int(cfg.get("max_iterations", 200))
    tolerance_explore = float(cfg.get("tolerance_explore", 0.1))
    n_samples = int(cfg.get("n_samples", 1000))
    initial_bounds = cfg.get("initial_bounds", "vmm")

    # Step 1: bounds (and, with VMM, the extreme designs) must exist before
    # the outer approximation is built, because they define the coordinates.
    # Unlike oracle mode, no projection model is needed: support_function
    # reuses the model as-is, swapping only the objective per call.
    supplied = None if initial_bounds == "vmm" else initial_bounds
    mga.solve_axis_bounds(supplied)
    initial_points, point_origin = mga.initial_inner_points()

    A0, b0 = mga.build_initial_outer_approximation()
    poly = approximation(A=A0, X=initial_points, b=b0, name_list=list(mga.z_names))

    explorer = prob_direction_explore(
        support_function=mga.support_function,
        poly_approx=poly,
        tolerance_prob=tolerance_prob,
        tolerance_explore=tolerance_explore,
        n_samples=n_samples,
        alpha=float(cfg.get("alpha", 0.05)),
        method=cfg.get("method", "jeffreys"),
        use_bounding_box=bool(cfg.get("use_bounding_box", False)),
        seed_rng=cfg.get("seed_rng"),
        history=True,
        print_lv=1,
    )
    logging.info(
        f"MGA probabilistic: tolerance_prob = {tolerance_prob:.3g}, "
        f"max_iterations = {max_iterations}, initial bounds = "
        f"{'VMM' if supplied is None else 'supplied'}"
    )

    # The summary folder is a sibling of the per-iteration Postprocess
    # folders; the subfolder separates scenarios (empty for plain runs).
    out = (
        Path(mga.optimization_setup.analysis.folder_output)
        / f"{mga.postprocess_ctx['model_name']}_probabilistic_summary"
        / mga.postprocess_ctx["subfolder"]
    )
    out.mkdir(parents=True, exist_ok=True)
    iteration_history = None
    converged = False
    final_gap = float("nan")
    try:
        _, iteration_history = explorer.explore(max_iterations)
        # explorer.explore() itself decides convergence from a fresh CI
        # estimate on the final approximation; ask it the same question
        # again rather than inferring the answer from its internal
        # bookkeeping (e.g. history-list lengths), which would silently
        # start reporting the wrong thing if pyoNearOpt's internals change.
        _, _, ci_lower, _, progress_metrics = explorer.estimate_fraction_of_directions()
        converged = bool(ci_lower > tolerance_prob)
        final_gap = float(progress_metrics["max_gap"])
    finally:
        # Persist artifacts even if the loop raised mid-way.
        run_info = {
            "tolerance_explore": tolerance_explore,
            "n_samples": n_samples,
            "max_iterations": max_iterations,
            "iterations_done": 0 if iteration_history is None else len(iteration_history),
            "initial_bounds": "vmm" if supplied is None else "supplied",
            "versions": _package_versions(),
        }
        _save_artifacts(
            mga, poly, iteration_history, tolerance_prob, converged, final_gap,
            out, point_origin, run_info,
        )
    return out


def _save_artifacts(
    mga, poly, iteration_history, tolerance_prob, converged, final_gap,
    out, point_origin, run_info,
):
    """Persist the polytope npz + diagnostics csv and log the outcome.

    Runs in run_probabilistic_mode's `finally`, so completed iterations
    survive a mid-loop exception (iteration_history is None in that case).
    """
    meta = mga.polytope_metadata()
    # X grows by one certified point per iteration; the rows beyond the
    # initial set are those iterates.
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
            convergence_threshold=float(tolerance_prob),
            converged=bool(converged),
            final_gap=float(final_gap),
            n_initial_rows=int(mga.n_initial_rows),
            point_origin=origins,
            meta=meta,
            run=run_info,
        ),
    )
    if iteration_history:
        pd.DataFrame(iteration_history).to_csv(out / "diagnostics.csv", index=False)
        log = logging.info if converged else logging.warning
        log(
            f"MGA probabilistic: {'CONVERGED' if converged else 'did NOT converge'} "
            f"after {len(iteration_history)} iterations, final gap = "
            f"{final_gap:.4g} (tolerance_prob = {tolerance_prob:.4g})."
        )
    else:
        logging.warning(
            "MGA probabilistic: no diagnostics to save (the refinement "
            "loop raised before any iteration completed); see traceback above."
        )
    logging.info(f"MGA probabilistic: artifacts saved to {out}")


def _package_versions() -> dict:
    """Versions of the packages that determine a run's results."""
    versions = {}
    for package in ("zen-garden", "pyoNearOpt", "linopy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions
