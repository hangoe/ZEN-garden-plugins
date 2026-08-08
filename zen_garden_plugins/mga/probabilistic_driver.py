"""Probabilistic-mode driver for the MGA plugin.

Wires MGA.support_function into pyoNearOpt's probabilistic support-function
exploration. Unlike ORACLE, this family never solves a bilevel/MILP model:
each iteration samples candidate directions, picks the one with the largest
observed gap between the inner and outer polytope approximations, and probes
it with a single support-function LP (`MGA.support_function`) on the
ZEN-garden model.

As of near_optimal_tools' package_version branch (post commit 73d5359,
"rename heurestic to bbo, prob to sampling", and 7dd8e6f, "refactor to a
single exploration loop for all sup fun methods"), the old monolithic
`prob_ORACLE.prob_direction_explore` no longer exists. It is replaced by
three composable pieces:
  - `supf_explore`: the generic exploration harness that drives the loop
    (`explorer.explore(n_iter)`) and owns the polytope bookkeeping
    (`poly_approx.add_point`/`add_cut`), same as before.
  - `sampling_exploration`: the strategy that samples candidate directions
    and picks the one with the largest gap (what `prob_direction_explore`
    used to do internally).
  - `ci_convergence_metric`: decides convergence from the same confidence
    interval on the fraction of well-approximated directions that
    `prob_direction_explore.estimate_fraction_of_directions` used to compute.
The harness checks `metric` before every iteration's query (as
`prob_direction_explore.explore` used to), so a final, fresh convergence
check after the loop is still needed -- see the comment in
`run_probabilistic_mode` below.

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
    from pyoNearOpt.exploration_methods.sampling_ORACLE import sampling_exploration
    from pyoNearOpt.exploration_methods.supf_explore import supf_explore
    from pyoNearOpt.metrics import ci_convergence_metric
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

    alpha = float(cfg.get("alpha", 0.05))
    method = cfg.get("method", "jeffreys")
    seed_rng = cfg.get("seed_rng")

    exploration = sampling_exploration(
        n_samples=n_samples,
        tolerance_explore=tolerance_explore,
        alpha=alpha,
        method=method,
        use_bounding_box=bool(cfg.get("use_bounding_box", False)),
        seed_rng=seed_rng,
        print_lv=1,
    )
    # Same tolerance_prob/tolerance_explore/n_samples/alpha/method as
    # `exploration` above, so the convergence check and the direction search
    # agree on what "well-approximated" means; seeded independently (its own
    # sampling_exploration-style RNG), which only changes which random
    # directions each draws, not the semantics.
    metric = ci_convergence_metric(
        tolerance_prob=tolerance_prob,
        tolerance_explore=tolerance_explore,
        n_samples=n_samples,
        alpha=alpha,
        method=method,
        seed_rng=seed_rng,
        print_lv=1,
    )
    explorer = supf_explore(
        support_function=mga.support_function,
        poly_approx=poly,
        exploration=exploration,
        metric=metric,
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
        # supf_explore checks `metric` at the *start* of every iteration,
        # before that iteration's point/cut are added to `poly`; so its last
        # recorded check does not reflect the final query when the loop runs
        # out of max_iterations without converging. Ask the same question
        # again on the now-final `poly`, rather than inferring the answer
        # from internal bookkeeping (e.g. history-list lengths), which would
        # silently start reporting the wrong thing if pyoNearOpt's internals
        # change.
        converged, final_metrics = metric.evaluate(poly, iteration=len(iteration_history))
        final_gap = float(final_metrics["max_gap"])
        # Merge each iteration's convergence check (ci_lower, ci_upper,
        # mean_gap, max_gap) into the same row of the saved diagnostics, as
        # the pre-refactor prob_direction_explore.explore() used to.
        for entry, metric_fields in zip(iteration_history, explorer.convergence_history):
            entry.update(metric_fields)
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
