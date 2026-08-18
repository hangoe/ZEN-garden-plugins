"""Batch support-function-mode driver for the MGA plugin.

Wires `ForkedBatchSupportFunction` (parallel_solve.py) into pyoNearOpt's
`batch_oracle`. Like `sampling`/`bbo`, each iteration probes candidate
directions and adds the resulting points/cuts to the shared inner/outer
polytope approximation -- but as a *batch* of up to `batch_size` directions,
solved concurrently by a worker pool instead of one at a time.

`batch_oracle` owns its own convergence bookkeeping internally (no external
`ci_convergence_metric` like `supf_explore` uses), so this driver mirrors
`_run_supf_mode`'s shape without constructing one.

pyoNearOpt is imported lazily, as in oracle_driver.py/supf_driver.py. Batch
mode always needs pyoNearOpt's optional `bbo` extra (pulls in `pypop7`),
even with `strategy_mode="sampling"`: `batch_ORACLE` imports `SHADE`
unconditionally at module level.

Each saved diagnostics.csv row also carries its own "wall_time_seconds":
the real elapsed time of that outer iteration's whole batch_size-worker
round, timed directly around the concurrent submit/gather in
ForkedBatchSupportFunction.__call__ (see parallel_solve.py) -- the same
idea as near_optimal_tools' own docs/examples/method_comparison.ipynb
wrapping its (sequential) support_function in a TimedCallback, adapted here
for a *concurrent* callback. This is deliberately NOT derived from
ZEN-garden's own per-solve benchmarking.json: that only records each
worker's own LP/MILP time (missing model-construction/I/O overhead, and
per-worker rather than per-batch-round, so batch_size of them can't be
combined into a real elapsed time without guessing how much they actually
overlapped), and -- found the hard way, analysing an earlier batch run --
a range of a run's own per-point benchmarking.json files can go missing
independently of whether the underlying solve succeeded, if whatever
copies the run's outputs off Euler gets interrupted (see
plot_mga_results.py's REAL_ELAPSED_SECONDS/_calibrate_cum_seconds for the
external-SLURM-history workaround that earlier run needed). Living in
diagnostics.csv instead means it's saved once, atomically, with the rest
of the run's own artifacts.
"""

import logging
import os
from pathlib import Path

from .parallel_solve import ForkedBatchSupportFunction
from .supf_driver import _package_versions, _save_artifacts, _setup_bounds_and_approximation


def run_batch_mode(mga, cfg):
    """Run the batch support-function pipeline on a set-up MGA instance.

    Returns the summary directory holding the polytope npz and diagnostics.
    """
    from pyoNearOpt.exploration_methods.batch_ORACLE import batch_oracle

    tolerance_prob = float(cfg["tolerance_prob"])  # required, deliberately no default
    max_iterations = int(cfg.get("max_iterations", 200))
    tolerance_explore = float(cfg.get("tolerance_explore", 0.1))
    n_samples = int(cfg.get("n_samples", 1000))
    alpha = float(cfg.get("alpha", 0.05))
    method = cfg.get("method", "jeffreys")
    seed_rng = cfg.get("seed_rng")
    initial_bounds = cfg.get("initial_bounds", "vmm")
    use_bounding_box = bool(cfg.get("use_bounding_box", True))
    batch_size = int(cfg.get("batch_size", 4))
    strategy_mode = cfg.get("strategy_mode", "sampling")
    convergence_mode = cfg.get("convergence_mode", "ci")
    bbo_enabled = bool(cfg.get("bbo_enabled", True))
    max_function_evaluations = int(cfg.get("max_function_evaluations", 2000))
    n_restarts = int(cfg.get("n_restarts", 1))
    n_workers = int(cfg.get("n_workers", os.cpu_count() or 1))

    poly, point_origin, supplied = _setup_bounds_and_approximation(mga, "batch", initial_bounds)

    logging.info(
        f"MGA batch: tolerance_prob = {tolerance_prob:.3g}, "
        f"max_iterations = {max_iterations}, batch_size = {batch_size}, "
        f"n_workers = {n_workers}, strategy_mode = {strategy_mode!r}, "
        f"initial bounds = {'VMM' if supplied is None else 'supplied'}"
    )

    # Sibling of the per-query iter<batch>_<job> Postprocess folders.
    out = (
        Path(mga.optimization_setup.analysis.folder_output)
        / f"{mga.postprocess_ctx['model_name']}_batch_summary"
        / mga.postprocess_ctx["subfolder"]
    )
    out.mkdir(parents=True, exist_ok=True)

    batch_support_function = ForkedBatchSupportFunction(mga, n_workers)
    explorer = batch_oracle(
        batch_support_function=batch_support_function,
        poly_approx=poly,
        strategy_mode=strategy_mode,
        batch_size=batch_size,
        tolerance_explore=tolerance_explore,
        tolerance_prob=tolerance_prob,
        n_samples=n_samples,
        alpha=alpha,
        method=method,
        convergence_mode=convergence_mode,
        use_bounding_box=use_bounding_box,
        seed_rng=seed_rng,
        # Pinned to avoid nesting pyoNearOpt's own candidate-sampling pool
        # inside our worker pool.
        n_jobs_fraction=1,
        bbo_enabled=bbo_enabled,
        max_function_evaluations=max_function_evaluations,
        n_restarts=n_restarts,
        history=True,
        print_lv=1,
    )

    iteration_history = None
    converged = False
    final_gap = float("nan")
    try:
        _, iteration_history = explorer.explore(max_iterations)
        # batch_oracle.explore calls batch_support_function exactly once per
        # iteration_history entry it appends, in the same order (both happen
        # in the same loop body, uninterrupted by any other call) -- so
        # ForkedBatchSupportFunction.times zips onto iteration_history 1:1.
        # A length mismatch would mean that invariant broke upstream; fail
        # loud here rather than silently truncate/misalign timings.
        assert len(iteration_history) == len(batch_support_function.times), (
            f"MGA batch: {len(iteration_history)} iteration_history entries "
            f"but {len(batch_support_function.times)} recorded batch call "
            f"times; these should always match 1:1."
        )
        for entry, elapsed in zip(iteration_history, batch_support_function.times):
            entry["wall_time_seconds"] = elapsed
        # batch_oracle has no public re-check; read its own last recorded
        # history entry instead (same staleness caveat as sampling/bbo's
        # pre-refactor check).
        if convergence_mode == "ci":
            converged = bool(
                explorer.ci_lower_history and explorer.ci_lower_history[-1] > tolerance_prob
            )
        else:  # "gap" or "both"
            converged = bool(explorer.gap_history and explorer.gap_history[-1] < tolerance_explore)
        if explorer.max_gap_history:
            final_gap = float(explorer.max_gap_history[-1])
    finally:
        # Persist artifacts even if the loop raised mid-way.
        batch_support_function.close()
        run_info = {
            "mode": "batch",
            "normalisation": mga.normalisation,
            "tolerance_explore": tolerance_explore,
            "n_samples": n_samples,
            "max_iterations": max_iterations,
            "iterations_done": 0 if iteration_history is None else len(iteration_history),
            "initial_bounds": "vmm" if supplied is None else "supplied",
            "batch_size": batch_size,
            "strategy_mode": strategy_mode,
            "convergence_mode": convergence_mode,
            "n_workers": n_workers,
            "versions": _package_versions(),
        }
        _save_artifacts(
            mga, "batch", poly, iteration_history, tolerance_prob, converged, final_gap,
            out, point_origin, run_info,
        )
    return out
