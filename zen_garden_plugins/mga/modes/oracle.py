"""Oracle mode: pyoNearOpt's ORACLE on the ZEN-garden model.

ORACLE (Turan, Moret, Bardow 2026) alternates a max-min MILP over the inner
and outer approximation (inside pyoNearOpt, solved by Gurobi) with an
L-infinity projection of the resulting trial point onto the near-optimal
space (``OracleProjection.find_nearest_point``, one ZEN-garden solve). The
projection duals give the next cutting plane. Details and the pyoNearOpt
compatibility notes are in the docs, section "Oracle mode".
"""

import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr

from ..artifacts import RESULT_PACKAGES, package_versions, save_artifacts, summary_dir
from ..initialisation import initialise

logger = logging.getLogger(__name__)

# Cuts violating a known inner point by more than this are relaxed outward
# (inexact duals); a false trigger from rounding is harmless.
CUT_GUARD_TRIGGER = 1e-4

# Postprocess cannot save dimensionless variables, so t gets a trivial dim.
Z_DIM = "mga_z_axis"
_SCALAR_DIM = "mga_oracle_scalar_dim"


def _scalar_da(value):
    """A one-element DataArray on the trivial scalar dimension."""
    return xr.DataArray(np.array([value]), dims=_SCALAR_DIM, coords={_SCALAR_DIM: [0]})


class OracleProjection:
    """The L-infinity projection model and ORACLE's nearest-point callback."""

    def __init__(self, mga, initial_points=None):
        """Add the projection model to the linopy model (once per run).

        Variables: delta (one entry per axis) and a scalar t. Constraints:
        axis_i - delta_i == trial_i (right-hand side set per call) and
        |delta_i| / scale_i <= t, so min t is the normalised L-infinity
        distance to the trial point.

        Args:
            mga: The ``MGA``, with bounds solved.
            initial_points: Normalised inner points seeding the cut guard;
                defaults to z* alone and should match the X given to ORACLE.
        """
        self.mga = mga
        model = mga.model
        z_coord = xr.DataArray(
            np.array(mga.z_names), dims=Z_DIM, coords={Z_DIM: mga.z_names}
        )
        self.delta = model.add_variables(
            coords=[z_coord], name="mga_oracle_delta", lower=-np.inf, upper=np.inf
        )
        self.t_var = model.add_variables(
            coords=[_scalar_da(0)], name="mga_oracle_t", lower=0.0
        )
        # Postprocess looks up every solution variable in ZEN-garden's registry.
        variables = mga.optimization_model.variables
        for name, doc in (
            ("mga_oracle_delta", "MGA oracle: projection offset per axis"),
            ("mga_oracle_t", "MGA oracle: normalised L-infinity distance"),
        ):
            variables.units[name] = None
            variables.docs[name] = doc

        for i, axis in enumerate(mga.axes):
            model.add_constraints(
                mga.axis_expression(axis) - self.delta.sel({Z_DIM: axis.name}) == 0.0,
                name=f"mga_oracle_proj_eq_axis{i}",
            )
        d_scale = xr.DataArray(1.0 / mga.scale, dims=Z_DIM, coords={Z_DIM: mga.z_names})
        model.add_constraints(
            d_scale * self.delta - self.t_var <= 0, name="mga_oracle_t_pos"
        )
        model.add_constraints(
            -(d_scale * self.delta) - self.t_var <= 0, name="mga_oracle_t_neg"
        )

        if initial_points is None:
            initial_points = [mga.z_star_norm]
        self.inner_points = [np.asarray(p, dtype=float) for p in initial_points]
        # 1-based, matching pyoNearOpt's iteration numbers.
        self.iteration = 1
        logger.info(
            f"MGA oracle: projection model added (n_z = {mga.n_z}, "
            f"{len(self.inner_points)} initial inner point(s))"
        )

    def find_nearest_point(self, trial_point: np.ndarray):
        """Project a normalised trial point onto the near-optimal space.

        Returns:
            (z_feas, dist, mu_cut, b_cut, 0) in normalised coordinates: dist
            is the L-infinity distance t*, mu_cut the projection duals times
            scale, b_cut = mu_cut @ z_feas (possibly relaxed by the cut guard).
        """
        mga = self.mga
        model = mga.model
        assert trial_point.shape == (mga.n_z,), (
            f"trial point shape {trial_point.shape}, expected ({mga.n_z},)"
        )

        trial_phys = mga.to_phys(trial_point)
        for i in range(mga.n_z):
            model.constraints[f"mga_oracle_proj_eq_axis{i}"].rhs = float(trial_phys[i])
        model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self.iteration}"
        logger.info(
            f"MGA oracle: starting iteration {self.iteration}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        try:
            mga.solve_and_postprocess(label)
        except RuntimeError:
            # Retry once (threaded barrier solves are not deterministic); a
            # second failure returns a zero-distance copy of the last inner
            # point, which stops ORACLE gracefully.
            logger.exception(
                f"MGA oracle iter {self.iteration}: projection solve failed; "
                f"retrying once."
            )
            try:
                mga.solve_and_postprocess(label)
            except RuntimeError:
                logger.error(
                    "MGA oracle: projection failed twice; stopping ORACLE gracefully."
                )
                z_prev = np.asarray(self.inner_points[-1], dtype=float).copy()
                self.iteration += 1
                return z_prev, 0.0, None, None, 0

        z_feas = mga.current_point_norm()
        dist = float(self.t_var.solution.values[0])
        mu_phys = np.array(
            [
                float(model.constraints[f"mga_oracle_proj_eq_axis{i}"].dual.values)
                for i in range(mga.n_z)
            ],
            dtype=float,
        )
        mu_cut = mu_phys * mga.scale
        b_cut = float(mu_cut @ z_feas)

        # A valid cut keeps every known near-optimal point; relax it otherwise.
        points = np.vstack(self.inner_points)
        worst = float((points @ mu_cut - b_cut).max())
        if worst > CUT_GUARD_TRIGGER:
            logger.warning(
                f"MGA oracle iter {self.iteration}: cut would remove "
                f"known near-optimal point(s) by up to {worst:.4g}; "
                f"relaxing b_cut {b_cut:.6g} -> {b_cut + worst:.6g}"
            )
            b_cut = float(b_cut + worst + 1e-9)
            if float(mu_cut @ trial_point) <= b_cut:
                logger.error(
                    "MGA oracle: relaxed cut no longer excludes the trial "
                    "point (unreliable projection duals this iteration)."
                )
        self.inner_points.append(np.asarray(z_feas, dtype=float))

        logger.info(
            f"MGA oracle iter {self.iteration}: dist = {dist:.4g} "
            f"(normalised L-inf), |mu|_max = {np.max(np.abs(mu_cut)):.4g}, "
            f"b_cut = {b_cut:.4g}"
        )
        self.iteration += 1
        return z_feas, dist, mu_cut, b_cut, 0


def run_oracle_mode(mga, cfg):
    """Run ORACLE on a set-up ``MGA`` and write its summary folder.

    Args:
        mga: The set-up ``MGA``.
        cfg: The validated ``plugins.mga.oracle`` block.

    Returns:
        The summary folder.
    """
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm

    max_min = cfg.max_min
    # With base pyoNearOpt, a numeric Md selects big-M and None selects SOS1.
    big_m = max_min.big_M if max_min.use_bigM else None
    with _coordinate_warnings_suppressed():
        poly, initial_points, point_origin, supplied = initialise(
            mga, "oracle", cfg.initial_bounds, big_m=big_m
        )
        if max_min.t_max is not None:
            poly.t_max = max_min.t_max
    projection = OracleProjection(mga, initial_points=initial_points)

    solver_options = dict(max_min.solver_options)
    # Keep Gurobi's console output only, not its duplicate logger copy.
    logging.getLogger("gurobipy").propagate = False
    algo = ORACLEAlgorithm(
        poly_approx=poly,
        find_nearest_point=projection.find_nearest_point,
        max_iterations=cfg.max_iterations,
        tol=cfg.tolerance,
        pyomo_solver=_gurobi_solver(solver_options),
        print_lv=1,
    )
    logger.info(
        f"MGA oracle: max-min formulation = {max_min.formulation!r}, tol = "
        f"{cfg.tolerance:.3g}, initial bounds = "
        f"{'VMM' if supplied is None else 'supplied'}"
    )

    out = summary_dir(mga, "oracle")
    df = None
    metric_source = "loop"
    iterations_done = 0
    try:
        with _coordinate_warnings_suppressed():
            df = algo.refine_approximations()
        iterations_done = 0 if df is None else int(len(df))
        df, metric_source = _run_final_certificate(
            df, poly, cfg.tolerance, max_min, solver_options
        )
    finally:
        # Persist artifacts even if the loop raised mid-way (df stays None).
        if df is not None:
            final_distance = float(df["max_min_distance"].iloc[-1])
            converged = bool(final_distance <= cfg.tolerance)
        else:
            final_distance = float("nan")
            converged = False
        run_info = {
            "normalisation": mga.normalisation,
            "formulation": max_min.formulation,
            "use_bigM": max_min.use_bigM,
            "max_iterations": cfg.max_iterations,
            "iterations_done": iterations_done,
            "metric_source": metric_source,
            "initial_bounds": "vmm" if supplied is None else "supplied",
            "versions": package_versions((*RESULT_PACKAGES, "gurobipy")),
        }
        save_artifacts(
            mga,
            out,
            poly,
            point_origin,
            cfg.tolerance,
            converged,
            final_distance,
            run_info,
            diagnostics=df,
        )
    return out


def _run_final_certificate(df, poly, tolerance, max_min, solver_options):
    """One long max-min solve on the final geometry after a non-converged loop.

    Controlled by max_min.certificate_time_limit (seconds, 0 = off). The
    certified value is appended as an extra diagnostics row.

    Returns:
        (diagnostics, metric_source) with metric_source "loop" or "certificate".
    """
    time_limit = float(max_min.certificate_time_limit or 0)
    if df is None or len(df) == 0 or time_limit <= 0:
        return df, "loop"
    last = float(df["max_min_distance"].iloc[-1])
    if last <= tolerance:
        return df, "loop"

    logger.info(
        f"MGA oracle: final certificate solve "
        f"(TimeLimit = {time_limit:.0f}s, cap = {last:.4g})."
    )
    try:
        poly.t_max = last
        poly.inner_outer_model()
        solver = _gurobi_solver(dict(solver_options, TimeLimit=time_limit))
        # Only the dual bound is needed; a solution-less timeout must not raise.
        solver.solve(poly.out_inner, load_solutions=False, tee=True)
        certified = min(last, -float(solver._solver_model.ObjBound))
    except Exception:
        logger.exception(
            "MGA oracle: final certificate failed; keeping the loop's metric."
        )
        return df, "loop"

    logger.info(
        f"MGA oracle: final certificate {last:.4g} -> {certified:.4g} "
        f"(tol = {tolerance:.4g})."
    )
    row = {c: None for c in df.columns}
    row["max_min_distance"] = certified
    if "iteration" in df.columns:
        row["iteration"] = int(df["iteration"].iloc[-1]) + 1
    if "max_min_solve_time" in df.columns:
        row["max_min_solve_time"] = time_limit
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True), "certificate"


def _gurobi_solver(options):
    """Pyomo Gurobi solver on the Python API; the env is released with it."""
    import pyomo.environ as pyo

    solver = pyo.SolverFactory("gurobi", solver_io="python", manage_env=True)
    solver.set_options(" ".join(f"{k}={v}" for k, v in options.items()))
    return solver


@contextmanager
def _coordinate_warnings_suppressed():
    """Silence linopy's intended coordinate-mismatch warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Coordinates across variables not equal.*",
            category=UserWarning,
        )
        yield
