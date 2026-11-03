"""Shared first step of every polytope mode (oracle, sampling, bbo, batch).

Bounds per design axis (VMM min/max LPs or user-supplied) define the
normalisation; the baseline z* plus the VMM extreme designs form the initial
inner points, and the bounds plus the monotonicity rows the initial outer
box. ``initialise`` returns all of it as a pyoNearOpt ``approximation``.
"""

import logging
import time

import numpy as np

from . import normalisation as norm
from .axes import MAY_BE_NEGATIVE_KINDS

logger = logging.getLogger(__name__)


def initialise(mga, mode, initial_bounds, big_m=None):
    """Bounds, normalisation and the initial pyoNearOpt approximation.

    Call after ``mga.setup()``.

    Args:
        mga: The set-up ``MGA``.
        mode: Mode name, for messages.
        initial_bounds: "vmm" or {axis: [lower, upper]} for every design axis.
        big_m: Big-M of ORACLE's max-min MILP (None selects SOS1).

    Returns:
        (poly, initial_points, point_origin, supplied): the approximation, its
        seed points (normalised) with their provenance labels, and the
        supplied bounds dict (None for VMM).

    Raises:
        ValueError: If no design axis is configured or the bounds are invalid.
    """
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.polytope_approximation.approximation_class import approximation

    if not mga.design_axes:
        raise ValueError(
            f"MGA {mode}: no exploration axes "
            f"configured; set at least one design axis under plugins.mga.axes."
        )
    supplied = None if initial_bounds == "vmm" else initial_bounds
    solve_axis_bounds(mga, supplied)
    initial_points, point_origin = initial_inner_points(mga)
    A0, b0 = build_initial_outer_approximation(mga)
    poly = approximation(
        A=A0, X=initial_points, b=b0, name_list=list(mga.z_names), Md=big_m
    )
    return poly, initial_points, point_origin, supplied


def solve_axis_bounds(mga, supplied_bounds=None) -> None:
    """Set ``mga.bounds_phys``, ``mga.scale`` and ``mga.offset``.

    Without ``supplied_bounds`` this runs VMM: two LPs per design axis drive
    it to its near-optimal minimum and maximum, which yields certified bounds
    and the extreme designs that seed the inner approximation. The cost
    axis's bounds are [C*, (1 + epsilon) C*] by construction.
    """
    if supplied_bounds is None:
        upper = solve_extreme_lps(mga, "max")
        lower = solve_extreme_lps(mga, "min")
        # Axis values are sums of non-negative variables, so a negative
        # minimum is solver round-off; emissions can be truly negative.
        non_negative = np.array(
            [axis.kind not in MAY_BE_NEGATIVE_KINDS for axis in mga.design_axes]
        )
        lower = np.where(non_negative, np.maximum(0.0, lower), lower)
    else:
        lower, upper = read_supplied_bounds(mga, supplied_bounds)

    mga.bounds_phys, mga.scale, mga.offset = norm.scale_and_offset(
        mga.axes,
        lower,
        upper,
        mga.c_star,
        mga.epsilon,
        mga.normalisation,
        share_reference=mga.share_reference,
        per_axes_reference=mga.per_axes_reference,
    )
    logger.info(
        f"MGA: bounds of {mga.n_z} axes ready "
        f"({'supplied' if supplied_bounds else 'VMM LPs'}); normalised "
        f"lower bounds "
        f"{np.round((mga.bounds_phys[:, 0] - mga.offset) / mga.scale, 4)}"
    )


def read_supplied_bounds(mga, supplied_bounds):
    """Check a {axis: [lower, upper]} dict; return (lower, upper) arrays."""
    design_names = [axis.name for axis in mga.design_axes]
    missing = [name for name in design_names if name not in supplied_bounds]
    unknown = sorted(set(supplied_bounds) - set(design_names))
    if missing or unknown:
        raise ValueError(
            f"MGA initial_bounds must cover every design axis exactly; "
            f"missing {missing}, unknown {unknown}."
        )
    lower, upper = [], []
    for name in design_names:
        pair = supplied_bounds[name]
        if len(pair) != 2:
            raise ValueError(
                f"MGA initial_bounds[{name!r}] must be [lower, upper], got {pair!r}."
            )
        lower.append(float(pair[0]))
        upper.append(float(pair[1]))
    return np.array(lower), np.array(upper)


def solve_extreme_lps(mga, sense: str) -> np.ndarray:
    """Drive every design axis to one near-optimal extreme ("max" or "min").

    Each LP must be proven optimal, since an unconverged solve gives no valid
    bound. Every solution is written as <model_name>_vmm_<sense>_<axis> and
    recorded in ``mga.extreme_designs``.

    Returns:
        The extreme values, in design-axis order.
    """
    values = []
    for axis in mga.design_axes:
        mga.model.add_objective(mga.axis_expression(axis), sense=sense, overwrite=True)
        start = time.time()
        mga.optimization_step.solve()
        if not mga.optimization_step.optimality:
            raise RuntimeError(
                f"MGA VMM {sense} LP for axis {axis.name!r} ended with "
                f"{mga.model.termination_condition!r}."
            )
        mga.postprocess(f"vmm_{sense}_{axis.name}")
        values.append(mga.axis_value(axis))
        mga.extreme_designs.append(
            (
                f"{sense}:{axis.name}",
                np.array([mga.axis_value(a) for a in mga.axes], dtype=float),
            )
        )
        logger.info(
            f"MGA: VMM {sense} {axis.name} = {values[-1]:.6g} "
            f"(LP took {time.time() - start:.1f} s)"
        )
    return np.array(values, dtype=float)


def initial_inner_points(mga) -> tuple[np.ndarray, list[str]]:
    """z* followed by every extreme design, normalised, with origin labels."""
    points = [mga.z_star_norm]
    origins = ["z_star"]
    for label, point_phys in mga.extreme_designs:
        points.append(mga.to_norm(point_phys))
        origins.append(label)
    return np.vstack(points), origins


def build_initial_outer_approximation(mga) -> tuple[np.ndarray, np.ndarray]:
    """(A0, b0) of the initial outer polytope in normalised coordinates.

    Two rows per axis (lower and upper bound), plus z(earlier) <= z(later)
    for consecutive axes of each cumulative-capex chain (docs:
    "Monotonicity"). Rows are mapped to normalised coordinates and scaled to
    unit length. Sets ``mga.n_initial_rows``.

    Raises:
        RuntimeError: If z* violates the result (inconsistent bounds).
    """
    n_z = mga.n_z
    A0 = np.vstack([-np.eye(n_z), np.eye(n_z)])
    b0 = np.concatenate([-mga.bounds_phys[:, 0], mga.bounds_phys[:, 1]])

    if mga.monotone_chains:
        index = {name: i for i, name in enumerate(mga.z_names)}
        mono_rows = []
        for chain in mga.monotone_chains:
            for earlier, later in zip(chain, chain[1:], strict=False):
                row = np.zeros(n_z)
                row[index[earlier]] = 1.0
                row[index[later]] = -1.0
                mono_rows.append(row)
        if mono_rows:
            A0 = np.vstack([A0, np.array(mono_rows)])
            b0 = np.concatenate([b0, np.zeros(len(mono_rows))])

    # a^T z_phys <= b with z_phys = offset + diag(scale) z_norm.
    b0 = b0 - A0 @ mga.offset  # must precede the column scaling below
    A0 = A0 @ np.diag(mga.scale)
    A0, b0 = normalise_rows(A0, b0)

    violation = A0 @ mga.z_star_norm - b0
    if (violation > 1e-6 * (np.abs(b0) + 1.0)).any():
        raise RuntimeError(
            f"MGA: initial outer approximation excludes z* "
            f"(max violation {violation.max():.3g})."
        )
    mga.n_initial_rows = A0.shape[0]
    logger.info(
        f"MGA: outer approximation of {A0.shape[0]} unit-norm rows over "
        f"{n_z} normalised axes."
    )
    return A0, b0


def normalise_rows(A, b):
    """Scale every row of ``A z <= b`` to unit Euclidean length.

    This keeps each half-space and matches pyoNearOpt's convention for the
    cuts it adds.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    return A / norms[:, None], b / norms
