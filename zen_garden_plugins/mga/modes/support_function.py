"""Support-function modes: sampling, bbo and batch.

Each query maximises d . z for a direction d with one LP on the ZEN-garden
model (``solve_direction``) and adds the point and its cut to pyoNearOpt's
inner/outer approximation. The modes differ only in how directions are
chosen: random sampling, a black-box optimiser (SHADE, needs pyoNearOpt's
optional "bbo" extra), or batches of directions solved in parallel by a
worker pool. Convergence is a confidence interval on the fraction of
well-approximated directions. See the docs, section "Support-function modes".
"""

import logging
import os
import time

import numpy as np
import pandas as pd

from ..artifacts import package_versions, save_artifacts, summary_dir
from ..initialisation import initialise

logger = logging.getLogger(__name__)


def solve_direction(mga, direction: np.ndarray, label: str):
    """Maximise ``direction`` (normalised), solve and post-process to ``label``.

    Returns:
        (z_feas, support_value): the solution in normalised coordinates and
        direction @ z_feas.
    """
    if direction.shape != (mga.n_z,):
        raise ValueError(f"direction shape {direction.shape}, expected ({mga.n_z},)")
    objective = sum(
        (float(direction[i]) / mga.scale[i]) * mga.axis_expression(axis)
        for i, axis in enumerate(mga.axes)
    )
    mga.model.add_objective(objective, sense="max", overwrite=True)
    logger.info(
        f"MGA solve_direction {label!r}: "
        f"||direction||_2 = {np.linalg.norm(direction):.4g}"
    )
    mga.solve_and_postprocess(label)
    z_feas = mga.current_point_norm()
    support_value = float(direction @ z_feas)
    logger.info(f"MGA solve_direction {label!r}: support_value = {support_value:.4g}")
    return z_feas, support_value


class SupportFunction:
    """Sequential pyoNearOpt callback: one direction per call, timed.

    Labels its solves supf_iter_<n> and records each call's wall-clock time.
    """

    def __init__(self, mga):
        self.mga = mga
        # 1-based, matching pyoNearOpt's iteration numbers.
        self.iteration = 1
        self.times: list[float] = []

    def __call__(self, direction: np.ndarray):
        start = time.perf_counter()
        logger.info(f"MGA support_function: starting iteration {self.iteration}")
        result = solve_direction(self.mga, direction, f"supf_iter_{self.iteration}")
        logger.info(
            f"MGA support_function iter {self.iteration}: "
            f"support_value = {result[1]:.4g}"
        )
        self.iteration += 1
        self.times.append(time.perf_counter() - start)
        return result


def run_sampling_mode(mga, cfg):
    """Run the sampling method; returns the summary folder."""
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.exploration_methods.sampling_ORACLE import sampling_exploration

    exploration = sampling_exploration(
        n_samples=cfg.n_samples,
        tolerance_explore=cfg.tolerance_explore,
        alpha=cfg.alpha,
        method=cfg.method,
        use_bounding_box=cfg.use_bounding_box,
        seed_rng=cfg.seed_rng,
        print_lv=1,
    )
    return _run_sequential(mga, cfg, "sampling", exploration)


def run_bbo_mode(mga, cfg):
    """Run the black-box-optimiser method; returns the summary folder."""
    from pyoNearOpt.exploration_methods.bbo_ORACLE import bbo_exploration

    exploration = bbo_exploration(
        max_function_evaluations=cfg.max_function_evaluations,
        optimizer_options=cfg.optimizer_options,
        n_restarts=cfg.n_restarts,
        seed_rng=cfg.seed_rng,
        use_bounding_box=cfg.use_bounding_box,
        print_lv=1,
    )
    return _run_sequential(mga, cfg, "bbo", exploration)


def _run_sequential(mga, cfg, mode, exploration):
    """The supf_explore loop shared by sampling and bbo."""
    from pyoNearOpt.exploration_methods.supf_explore import supf_explore
    from pyoNearOpt.metrics import ci_convergence_metric

    poly, _, point_origin, supplied = initialise(mga, mode, cfg.initial_bounds)
    # Same settings as the exploration, so both agree on "well-approximated".
    metric = ci_convergence_metric(
        tolerance_prob=cfg.tolerance_prob,
        tolerance_explore=cfg.tolerance_explore,
        n_samples=cfg.n_samples,
        alpha=cfg.alpha,
        method=cfg.method,
        seed_rng=cfg.seed_rng,
        track_implied_threshold=cfg.track_implied_threshold,
        print_lv=1,
    )
    support_function = SupportFunction(mga)
    explorer = supf_explore(
        support_function=support_function,
        poly_approx=poly,
        exploration=exploration,
        metric=metric,
        history=True,
        print_lv=1,
    )
    _log_start(mode, cfg, supplied)

    out = summary_dir(mga, mode)
    history = None
    converged = False
    final_gap = float("nan")
    try:
        _, history = explorer.explore(cfg.max_iterations)
        _attach(history, "wall_time_seconds", support_function.times)
        # supf_explore checks convergence before each query, so re-check on
        # the final approximation to include the last one.
        converged, final_metrics = metric.evaluate(poly, iteration=len(history))
        final_gap = float(final_metrics["max_gap"])
        for entry, metric_fields in zip(
            history, explorer.convergence_history, strict=False
        ):
            entry.update(metric_fields)
    finally:
        run_info = _run_info(mode, mga, cfg, history, supplied)
        run_info["track_implied_threshold"] = cfg.track_implied_threshold
        run_info["versions"] = package_versions()
        save_artifacts(
            mga,
            out,
            poly,
            point_origin,
            cfg.tolerance_prob,
            converged,
            final_gap,
            run_info,
            diagnostics=pd.DataFrame(history) if history else None,
        )
    return out


def run_batch_mode(mga, cfg):
    """Run pyoNearOpt's batch_oracle with a worker pool; returns the summary folder.

    Needs pyoNearOpt's "bbo" extra even with strategy_mode="sampling".
    """
    from pyoNearOpt.exploration_methods.batch_ORACLE import batch_oracle

    from .worker_pool import WorkerPoolSupportFunction

    n_workers = cfg.n_workers or os.cpu_count() or 1
    poly, _, point_origin, supplied = initialise(mga, "batch", cfg.initial_bounds)
    logger.info(
        f"MGA batch: tolerance_prob = {cfg.tolerance_prob:.3g}, "
        f"max_iterations = {cfg.max_iterations}, batch_size = {cfg.batch_size}, "
        f"n_workers = {n_workers}, n_jobs_sampling = {cfg.n_jobs_sampling}, "
        f"strategy_mode = {cfg.strategy_mode!r}, "
        f"initial bounds = {'VMM' if supplied is None else 'supplied'}"
    )
    out = summary_dir(mga, "batch")

    support_function = WorkerPoolSupportFunction(mga, n_workers)
    explorer = batch_oracle(
        batch_support_function=support_function,
        poly_approx=poly,
        strategy_mode=cfg.strategy_mode,
        batch_size=cfg.batch_size,
        tolerance_explore=cfg.tolerance_explore,
        tolerance_prob=cfg.tolerance_prob,
        n_samples=cfg.n_samples,
        alpha=cfg.alpha,
        method=cfg.method,
        convergence_mode=cfg.convergence_mode,
        use_bounding_box=cfg.use_bounding_box,
        seed_rng=cfg.seed_rng,
        # Gap sampling runs in this (parent) process between worker rounds,
        # in its own short-lived spawn pool; it never nests inside a worker.
        n_jobs_fraction=cfg.n_jobs_sampling,
        bbo_enabled=cfg.bbo_enabled,
        max_function_evaluations=cfg.max_function_evaluations,
        n_restarts=cfg.n_restarts,
        history=True,
        track_implied_threshold=cfg.track_implied_threshold,
        print_lv=1,
    )

    history = None
    converged = False
    final_gap = float("nan")
    try:
        _, history = explorer.explore(cfg.max_iterations)
        _attach(history, "wall_time_seconds", support_function.times)
        if cfg.track_implied_threshold:
            # Recorded before the convergence check, so the converging
            # iteration may add one trailing entry, which zip drops.
            thresholds = explorer.implied_threshold_history
            assert len(thresholds) in (len(history), len(history) + 1), (
                f"MGA batch: {len(history)} iterations but {len(thresholds)} "
                f"implied thresholds; expected the same count or one more."
            )
            for entry, threshold in zip(history, thresholds, strict=False):
                entry["implied_threshold_for_tolerance"] = threshold
        # batch_oracle has no public re-check; read its last history entry.
        if cfg.convergence_mode == "ci":
            converged = bool(
                explorer.ci_lower_history
                and explorer.ci_lower_history[-1] > cfg.tolerance_prob
            )
        else:  # "gap" or "both"
            converged = bool(
                explorer.gap_history
                and explorer.gap_history[-1] < cfg.tolerance_explore
            )
        if explorer.max_gap_history:
            final_gap = float(explorer.max_gap_history[-1])
    finally:
        support_function.close()
        run_info = _run_info("batch", mga, cfg, history, supplied)
        run_info.update(
            batch_size=cfg.batch_size,
            strategy_mode=cfg.strategy_mode,
            convergence_mode=cfg.convergence_mode,
            n_workers=n_workers,
            n_jobs_sampling=cfg.n_jobs_sampling,
            track_implied_threshold=cfg.track_implied_threshold,
            versions=package_versions(),
        )
        save_artifacts(
            mga,
            out,
            poly,
            point_origin,
            cfg.tolerance_prob,
            converged,
            final_gap,
            run_info,
            diagnostics=pd.DataFrame(history) if history else None,
        )
    return out


def _log_start(mode, cfg, supplied):
    logger.info(
        f"MGA {mode}: tolerance_prob = {cfg.tolerance_prob:.3g}, "
        f"max_iterations = {cfg.max_iterations}, initial bounds = "
        f"{'VMM' if supplied is None else 'supplied'}"
    )


def _attach(history, key, values):
    """Add one value per iteration; the callback runs exactly once per entry."""
    assert len(history) == len(values), (
        f"MGA: {len(history)} iterations but {len(values)} recorded {key} values."
    )
    for entry, value in zip(history, values, strict=True):
        entry[key] = value


def _run_info(mode, mga, cfg, history, supplied) -> dict:
    """The run_info fields shared by the support-function modes, in order."""
    return {
        "mode": mode,
        "normalisation": mga.normalisation,
        "tolerance_explore": cfg.tolerance_explore,
        "n_samples": cfg.n_samples,
        "max_iterations": cfg.max_iterations,
        "iterations_done": 0 if history is None else len(history),
        "initial_bounds": "vmm" if supplied is None else "supplied",
    }
