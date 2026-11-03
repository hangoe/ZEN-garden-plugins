"""Normalisation conventions: per-axis scale and offset.

Every axis maps as physical = normalised * scale + offset. The design axes
follow ``plugins.mga.normalisation`` ("relative", "minmax", "share" or
"per_axes"); the cost axis always uses (epsilon * C*, C*), so 0 is the cost
optimum and 1 the budget. The conventions are described in the docs, section
"Normalisation".
"""

import logging

import numpy as np

from .axes import RATIO_KINDS
from .polytope_io import (
    NODE_CAPEX_CUMULATIVE,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    TOTAL_COST,
)

logger = logging.getLogger(__name__)

# Stored verbatim in each polytope's axis_meta_json; keep these strings stable.
_DESIGN_AXES_DESCRIPTION = {
    "minmax": "(upper bound - lower bound, lower bound) -- "
    "near-optimal [min, max] mapped to [0, 1]",
    "share": "(reference total, 0) -- share of a fixed "
    "baseline (z*) reference total; one shared total per "
    "run for node_capex/node_capex_cumulative (all nodes, "
    "full horizon), a separate per-technology-group total "
    "for node_capex_tech/node_capex_cumulative_tech",
    "per_axes": "(per-axis-kind reference, 0) -- "
    "node_capex_cumulative divides by a fixed configurable "
    "constant, node_capacity_ratio is already a fraction "
    "(reference 1), node_carbon_emissions_cumulative divides "
    "by the model's carbon_emissions_budget",
    "relative": "(upper bound, 0)",
}


def describe(normalisation: str) -> str:
    """One-line description of a convention, as stored in the polytope."""
    return (
        "physical = normalised * scale + offset, per axis; design axes use "
        + _DESIGN_AXES_DESCRIPTION[normalisation]
        + ", the cost axis always (epsilon * c_star, c_star)"
    )


def round_to_one_significant_figure(value: float) -> float:
    """Round a positive value to one significant figure (12.3e6 -> 1e7).

    Non-positive and non-finite values (e.g. the cost axis's NaN) are
    returned unchanged.
    """
    if value <= 0 or not np.isfinite(value):
        return value
    exponent = int(np.floor(np.log10(value)))
    mantissa = round(value / 10.0**exponent)
    if mantissa >= 10:
        mantissa = 1
        exponent += 1
    return float(mantissa * 10.0**exponent)


def share_references(axes, reference_value) -> tuple[np.ndarray, np.ndarray]:
    """Raw and rounded baseline reference totals for normalisation="share".

    Args:
        axes: The axes, in column order.
        reference_value: Callable giving an axis's baseline reference total.

    Returns:
        (raw, rounded), NaN for the cost axis.
    """
    raw = np.array(
        [np.nan if axis.kind == TOTAL_COST else reference_value(axis) for axis in axes],
        dtype=float,
    )
    rounded = np.array([round_to_one_significant_figure(v) for v in raw], dtype=float)
    logged = set()
    for axis, value in zip(axes, raw, strict=True):
        if axis.kind == TOTAL_COST or value in logged:
            continue
        logged.add(value)
        logger.info(
            f"MGA: normalisation='share' reference total for "
            f"{axis.name!r}'s group: {value:.6g} rounded to "
            f"{round_to_one_significant_figure(value):.6g}"
        )
    return raw, rounded


def per_axes_references(axes, capex_reference, carbon_budget) -> dict[str, float]:
    """Per-axis reference for normalisation="per_axes", by axis kind."""
    references: dict[str, float] = {}
    logged_kinds = set()
    for axis in axes:
        if axis.kind == NODE_CAPEX_CUMULATIVE:
            ref = float(capex_reference)
        elif axis.kind in RATIO_KINDS:
            ref = 1.0
        elif axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE:
            ref = float(carbon_budget)
        else:
            continue
        references[axis.name] = ref
        if axis.kind not in logged_kinds:
            logged_kinds.add(axis.kind)
            logger.info(
                f"MGA: normalisation='per_axes' reference for kind "
                f"{axis.kind!r}: {ref:.6g}"
            )
    return references


def scale_and_offset(
    axes,
    lower,
    upper,
    c_star,
    epsilon,
    normalisation,
    share_reference=None,
    per_axes_reference=None,
):
    """Bounds, scale and offset per axis.

    Args:
        axes: The axes, in column order.
        lower: Lower bounds of the design axes, in design-axis order.
        upper: Upper bounds of the design axes, in design-axis order.
        c_star: Baseline cost C*.
        epsilon: Near-optimality slack.
        normalisation: The convention for the design axes.
        share_reference: Rounded references per axis (normalisation="share").
        per_axes_reference: {axis name: reference} (normalisation="per_axes").

    Returns:
        (bounds_phys, scale, offset) as float arrays.

    Raises:
        RuntimeError: For bounds or references the convention cannot use.
    """
    bounds, scale, offset = [], [], []
    design_index = 0
    for i, axis in enumerate(axes):
        if axis.kind == TOTAL_COST:
            lo, hi = c_star, (1.0 + epsilon) * c_star
            bounds.append((lo, hi))
            scale.append(hi - lo)
            offset.append(lo)
            continue
        lo, hi = float(lower[design_index]), float(upper[design_index])
        design_index += 1
        # Only "relative" divides by hi; "minmax" just needs a positive span.
        if not np.isfinite(hi) or (hi <= 0 and normalisation != "minmax"):
            raise RuntimeError(
                f"MGA: axis {axis.name!r} has non-positive near-optimal "
                f"maximum {hi:.6g}; remove the axis or supply a positive "
                f"bound."
            )
        if lo > hi:
            raise RuntimeError(
                f"MGA: axis {axis.name!r} has lower bound {lo:.6g} above "
                f"upper bound {hi:.6g}."
            )
        bounds.append((lo, hi))
        if normalisation == "minmax":
            span = hi - lo
            if span <= 0:
                raise RuntimeError(
                    f"MGA: axis {axis.name!r} has zero near-optimal "
                    f"range (lower={lo:.6g} == upper={hi:.6g}); "
                    f"normalisation='minmax' requires a positive span. "
                    f"Remove the axis or use 'relative'."
                )
            scale.append(span)
            offset.append(lo)
        elif normalisation in ("share", "per_axes"):
            ref = (
                float(share_reference[i])
                if normalisation == "share"
                else float(per_axes_reference[axis.name])
            )
            if not np.isfinite(ref) or ref <= 0:
                raise RuntimeError(
                    f"MGA: axis {axis.name!r} has non-positive {normalisation} "
                    f"reference {ref:.6g}; normalisation={normalisation!r} "
                    f"requires a positive reference. Remove the axis or use "
                    f"'relative'/'minmax'."
                )
            scale.append(ref)
            offset.append(0.0)
        else:
            scale.append(hi)
            offset.append(0.0)
    return (
        np.array(bounds, dtype=float),
        np.array(scale, dtype=float),
        np.array(offset, dtype=float),
    )
