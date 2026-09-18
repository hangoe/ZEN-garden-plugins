"""
MGA (Modelling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` handler that adds a near-optimality cost budget
(cost <= (1 + epsilon) * C*) to the solved baseline model and re-solves it
under one of two modes:

* "weights": one solve per user-provided weight dict, each minimising
  sum_i w_i * capacity_addition_i on each technology's selected capacity
  type (energy for storage technologies, power otherwise).
* "oracle": the ORACLE algorithm (Turan, Moret, Bardow 2026) iteratively
  refines inner and outer polytope approximations of the near-optimal space
  via L-infinity projections. The refinement loop lives in pyoNearOpt; the
  projection solves run on the ZEN-garden model (driver in oracle_driver.py).
* "sampling": pyoNearOpt's sampling support-function method iteratively
  refines the same inner/outer approximations by sampling directions and
  probing each with a single support-function LP (much cheaper than
  ORACLE's projection). The refinement loop lives in pyoNearOpt; the
  support-function solves run on the ZEN-garden model (driver in
  supf_driver.py).
* "bbo": the same support-function pipeline as "sampling", but each
  direction is instead chosen by a black-box optimiser (SHADE) searching
  for the largest gap between the inner and outer approximations. Needs
  pyoNearOpt's optional "bbo" extra (pypop7). Driver in supf_driver.py.
* "batch": the same support-function pipeline as "sampling"/"bbo", but each
  iteration probes a *batch* of directions concurrently via a persistent
  worker pool, instead of one direction at a time. Writes one Postprocess
  folder per query, like "sampling"/"bbo". Always needs pyoNearOpt's
  optional "bbo" extra (pypop7), even with strategy_mode="sampling".
  Driver in batch_driver.py, worker pool in parallel_solve.py.

Rolling-horizon runs are rejected: after_solve fires after the horizon loop,
so the plugin would only see the final step's model. Scaled runs
(solver.use_scaling) and non-cost objectives are rejected too: MGA's added
expressions assume an unscaled model and a total-cost C*.

Glossary
    axis        one exploratory variable, i.e. one coordinate of the explored
                space: a technology-capacity group, a carrier-import group, a
                per-node (or per-node-lump) capex group, a per-node capex
                group restricted to an explicit calendar-year span, a
                per-node capex group restricted to a named technology group,
                a per-node capex group restricted to both a calendar-year
                span and a named technology group at once, a per-node ratio
                of two named technology groups' installed capacity at a
                snapshot calendar year, a per-node cumulative
                carbon-emissions group restricted to an explicit
                calendar-year span, or the total cost (see axes.py)
    n_z         number of axes, cost included -- the dimension of the
                explored space
    C* (c_star) the baseline (cost-optimal) net present cost
    z*          the baseline design in axis coordinates
    phys        physical units (GW, GWh, MEUR, ...)
    norm        normalised coordinates: phys = norm * scale + offset, per
                axis. Under normalisation="relative" (default), design axes
                use scale = upper bound and offset = 0, so they reach 1 at
                their near-optimal maximum; under "minmax", scale = upper -
                lower bound and offset = lower bound, so each axis spans
                exactly [0, 1] between its own near-optimal min and max;
                under "units", scale = 1 and offset = 0, so design axes are
                reported in their own physical units; under "share" (capex
                axes only), scale = a fixed reference total evaluated once
                on the baseline design z*, rounded to one significant figure
                (not recomputed per explored point) and offset = 0 -- every
                NODE_CAPEX/_CUMULATIVE axis in the run shares one identical
                total (all nodes, the full model horizon), while a
                NODE_CAPEX_TECH or NODE_CAPEX_CUMULATIVE_TECH axis instead
                divides by its own technology group's total (all nodes, full
                horizon); both the rounded
                total and its raw, unrounded value are recorded per axis in
                polytope_metadata(). Since that reference total is shared
                across a group's axes rather than being each axis's own
                near-optimal range (unlike "minmax"), sampling/bbo/batch's
                CI-based convergence check (a fixed tolerance_explore) is
                satisfied sooner for an axis whose own range is a small
                share of the group total, and later for one whose range is
                a large share -- deliberately spending more exploration on
                larger/more decision-relevant axes and less on smaller
                ones. This is intended behaviour of "share" normalisation,
                not a defect; use "minmax" instead when every axis should
                be explored to the same relative depth regardless of size.
                Under "per_axes" (NODE_CAPEX_CUMULATIVE,
                NODE_CAPACITY_RATIO and NODE_CARBON_EMISSIONS_CUMULATIVE
                axes only), scale = a reference specific to each axis's own
                kind, not one shared group total: NODE_CAPEX_CUMULATIVE
                divides by a fixed configurable constant
                (per_axes_capex_reference), NODE_CAPACITY_RATIO by 1 (it is
                already a fraction of its own baseline-frozen denominator),
                and NODE_CARBON_EMISSIONS_CUMULATIVE by the model's own
                carbon_emissions_budget parameter; offset = 0. The reference
                actually used is recorded per axis in polytope_metadata().
                Either way the cost axis uses
                scale = epsilon * C*
                and offset = C*, so 0 is the cost optimum and 1 the budget.

Every solve is written to disk as a sibling sub-solution of the baseline via
Postprocess.

Configured via the "plugins.mga" block in config.json; unknown keys are
rejected. Full config reference: the plugin's docs page in
docs/files/available_plugins/mga.

pyoNearOpt compatibility is documented in oracle_driver.py.
"""

import dataclasses
import logging
import time

import numpy as np
import xarray as xr
from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess

from .axes import COST_VARIABLE, Axis, axis_physical_unit, build_axis_groups
from .batch_driver import run_batch_mode
from .oracle_driver import run_oracle_mode
from .polytope_io import (
    CARRIER_IMPORT,
    NODE_CAPACITY_RATIO,
    NODE_CAPEX,
    NODE_CAPEX_CUMULATIVE,
    NODE_CAPEX_CUMULATIVE_TECH,
    NODE_CAPEX_TECH,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    TECH_CAPACITY,
    TOTAL_COST,
)
from .supf_driver import run_bbo_mode, run_sampling_mode

# Module-level config, updated by the plugin loader with the user's
# "plugins.mga" block. The merge is a SHALLOW dict.update: nested dicts like
# "axes" and "oracle" are replaced wholesale, so their defaults must be
# applied at access time via .get(), never stored here.
config = {
    "epsilon": 0.1,
    "mode": "weights",
    "normalisation": "relative",
    "per_axes_capex_reference": 15e12,
    "iterations": [],
    "axes": {},
    "oracle": {},
    "sampling": {},
    "bbo": {},
    "batch": {},
}

# Recognised config keys per block; anything else is rejected rather than
# silently ignored.
_KNOWN_KEYS = {
    "plugins.mga": {
        "epsilon",
        "mode",
        "normalisation",
        "per_axes_capex_reference",
        "iterations",
        "axes",
        "oracle",
        "sampling",
        "bbo",
        "batch",
    },
    "plugins.mga.axes": {
        "technologies",
        "carrier_imports",
        "include_cost",
        "node_capex",
        "node_capex_cumulative",
        "node_capex_by_technology",
        "node_capex_cumulative_tech",
        "node_capacity_ratio",
        "node_carbon_emissions_cumulative",
    },
    "plugins.mga.axes.node_capex_cumulative": {"nodes", "until_years"},
    "plugins.mga.axes.node_capex_by_technology": {"nodes", "technology_groups"},
    "plugins.mga.axes.node_capex_cumulative_tech": {
        "nodes",
        "until_years",
        "technology_groups",
    },
    "plugins.mga.axes.node_capacity_ratio": {"nodes", "years", "ratio_groups"},
    "plugins.mga.axes.node_carbon_emissions_cumulative": {"nodes", "until_years"},
    "plugins.mga.oracle": {"tolerance", "max_iterations", "initial_bounds", "max_min"},
    "plugins.mga.oracle.max_min": {
        "formulation",
        "use_bigM",
        "big_M",
        "t_max",
        "solver_options",
        "certificate_time_limit",
    },
    "plugins.mga.sampling": {
        "tolerance_prob",
        "max_iterations",
        "tolerance_explore",
        "n_samples",
        "alpha",
        "method",
        "initial_bounds",
        "use_bounding_box",
        "seed_rng",
        "track_implied_threshold",
    },
    "plugins.mga.bbo": {
        "tolerance_prob",
        "max_iterations",
        "tolerance_explore",
        "n_samples",
        "alpha",
        "method",
        "initial_bounds",
        "use_bounding_box",
        "seed_rng",
        "max_function_evaluations",
        "n_restarts",
        "optimizer_options",
        "track_implied_threshold",
    },
    "plugins.mga.batch": {
        "tolerance_prob",
        "max_iterations",
        "tolerance_explore",
        "n_samples",
        "alpha",
        "method",
        "initial_bounds",
        "use_bounding_box",
        "seed_rng",
        "batch_size",
        "strategy_mode",
        "convergence_mode",
        "max_function_evaluations",
        "n_restarts",
        "bbo_enabled",
        "n_workers",
        "track_implied_threshold",
    },
}

# A cut returned by find_nearest_point must keep every known near-optimal
# point inside the outer approximation; inexact projection duals (e.g.
# barrier without crossover) can violate that. Violations above this trigger
# are repaired by relaxing the cut offset out to the farthest known inner
# point. Relaxing outward is always valid, so a rare false trigger from
# coordinate rounding is harmless; real dual failures sit orders of
# magnitude above the trigger.
CUT_GUARD_TRIGGER = 1e-4

# Dimension names of the projection-model variables. Postprocess cannot save
# dimensionless variables, so the scalar t gets a trivial one-element
# dimension.
Z_DIM = "mga_z_axis"
_SCALAR_DIM = "mga_oracle_scalar_dim"


def _scalar_da(value):
    """A one-element DataArray on the trivial scalar dimension."""
    return xr.DataArray(np.array([value]), dims=_SCALAR_DIM, coords={_SCALAR_DIM: [0]})


def _year_indices_in_period(period, year_indices, real_years):
    """`set_time_steps_yearly` indices whose calendar year falls in `period`.

    `period` is (start_year, end_year), inclusive on both ends; start may be
    None, meaning no lower bound (every model year up to and including end).
    `year_indices` and `real_years` are parallel sequences (year_indices[i]'s
    calendar year is real_years[i]), as read off cost_capex_yearly's
    set_time_steps_yearly coordinate and energy_system.set_time_steps_years
    respectively. Returns an empty list if the period covers no model year.
    """
    start, end = period
    return [
        year_id
        for year_id, real_year in zip(year_indices, real_years, strict=True)
        if (start is None or start <= real_year) and real_year <= end
    ]


def _year_interval_expansion_factors(
    year_indices, discount_rate: float, interval_between_years: int, last_year_index: int
) -> "xr.DataArray":
    """Per-sampled-year discount/interval-expansion factor, indexed by
    set_time_steps_yearly.

    With discount_rate > 0, reproduces ZEN-garden's own
    constraint_net_present_cost formula (energy_system.py) exactly -- so a
    node-capex axis can be discounted the same way net_present_cost itself
    is:

    factor[y] = sum_{i=0}^{n-1} (1/(1+discount_rate)) ** (interval_between_years*(y-y0) + i)

    where n = 1 for the final sampled year (no extrapolation past the
    horizon, matching constraint_net_present_cost's own last-year special
    case), else n = interval_between_years; y0 = year_indices[0]. With
    discount_rate = 0.0, every term collapses to 1, so factor[y] = n --
    exactly the plain interval multiplicity ZEN-garden's own
    constraint_carbon_emissions_cumulative uses (each non-final sampled
    year counts interval_between_years times, undiscounted), used for the
    node-carbon-emissions-cumulative axis, which is interval-expanded but
    not discounted.
    """
    y0 = year_indices[0]
    factors = []
    for y in year_indices:
        n = 1 if y == last_year_index else interval_between_years
        factors.append(sum(
            (1.0 / (1.0 + discount_rate)) ** (interval_between_years * (y - y0) + i)
            for i in range(n)
        ))
    return xr.DataArray(
        factors, dims=["set_time_steps_yearly"], coords={"set_time_steps_yearly": year_indices}
    )


def _round_to_one_significant_figure(value: float) -> float:
    """Round a positive value to one significant figure, nearest -- e.g.
    12.3e6 -> 1e7 (10e6), 18e12 -> 2e13 (20e12). Leaves non-positive or
    non-finite values (e.g. the TOTAL_COST axis's NaN placeholder)
    unchanged. Used to give normalisation="share" a round, easily
    communicated reference total instead of an arbitrary baseline value.
    """
    if value <= 0 or not np.isfinite(value):
        return value
    exponent = int(np.floor(np.log10(value)))
    mantissa = round(value / 10.0**exponent)
    if mantissa >= 10:
        mantissa = 1
        exponent += 1
    return float(mantissa * 10.0**exponent)


def normalise_rows(A, b):
    """Scale every row of ``A z <= b`` to unit Euclidean length.

    Each row is one half-space, and multiplying a row by a positive constant
    leaves that half-space unchanged, so this alters the representation and
    not the geometry. It is pyoNearOpt's convention for the cutting planes it
    adds; applying it to the initial rows too gives the whole system one
    scale, instead of rows carrying the physical magnitude of their axis.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    return A / norms[:, None], b / norms


def validate_config(cfg) -> None:
    """Reject unknown keys and invalid values in the plugin config.

    Every setting is read with a default, so an unrecognised key would
    otherwise be ignored in silence and the run would proceed on defaults.
    """
    for label, block in (
        ("plugins.mga", cfg),
        ("plugins.mga.axes", cfg.get("axes", {})),
        (
            "plugins.mga.axes.node_capex_cumulative",
            cfg.get("axes", {}).get("node_capex_cumulative", {}),
        ),
        (
            "plugins.mga.axes.node_capex_by_technology",
            cfg.get("axes", {}).get("node_capex_by_technology", {}),
        ),
        (
            "plugins.mga.axes.node_capex_cumulative_tech",
            cfg.get("axes", {}).get("node_capex_cumulative_tech", {}),
        ),
        (
            "plugins.mga.axes.node_capacity_ratio",
            cfg.get("axes", {}).get("node_capacity_ratio", {}),
        ),
        (
            "plugins.mga.axes.node_carbon_emissions_cumulative",
            cfg.get("axes", {}).get("node_carbon_emissions_cumulative", {}),
        ),
        ("plugins.mga.oracle", cfg.get("oracle", {})),
        ("plugins.mga.oracle.max_min", cfg.get("oracle", {}).get("max_min", {})),
        ("plugins.mga.sampling", cfg.get("sampling", {})),
        ("plugins.mga.bbo", cfg.get("bbo", {})),
        ("plugins.mga.batch", cfg.get("batch", {})),
    ):
        unknown = sorted(set(block) - _KNOWN_KEYS[label])
        if unknown:
            raise ValueError(
                f"Unknown MGA config key(s) in {label}: {unknown}. "
                f"Known keys: {sorted(_KNOWN_KEYS[label])}."
            )

    normalisation = cfg.get("normalisation", "relative")
    if normalisation not in ("relative", "minmax", "units", "share", "per_axes"):
        raise ValueError(
            f"Unknown MGA normalisation: {normalisation!r}. Expected "
            f"'relative', 'minmax', 'units', 'share' or 'per_axes'."
        )
    axes_cfg = cfg.get("axes", {})
    if normalisation == "share" and (
        axes_cfg.get("technologies") or axes_cfg.get("carrier_imports")
    ):
        raise ValueError(
            "MGA: normalisation='share' only supports capex axes "
            "(node_capex / node_capex_cumulative / node_capex_by_technology / "
            "node_capex_cumulative_tech); remove axes.technologies/"
            "axes.carrier_imports or use a different normalisation."
        )
    if normalisation == "per_axes" and (
        axes_cfg.get("technologies")
        or axes_cfg.get("carrier_imports")
        or axes_cfg.get("node_capex")
        or axes_cfg.get("node_capex_by_technology")
        or axes_cfg.get("node_capex_cumulative_tech")
    ):
        raise ValueError(
            "MGA: normalisation='per_axes' only supports node_capex_cumulative, "
            "node_capacity_ratio and node_carbon_emissions_cumulative axes (plus "
            "the cost axis); remove axes.technologies/axes.carrier_imports/"
            "axes.node_capex/axes.node_capex_by_technology/"
            "axes.node_capex_cumulative_tech or use a different normalisation."
        )
    if cfg.get("mode") == "oracle" and normalisation in (
        "units",
        "minmax",
        "share",
        "per_axes",
    ):
        raise ValueError(
            f"MGA: normalisation={normalisation!r} is not supported in "
            "oracle mode -- oracle's max-min MILP relies on big_M/t_max "
            "dominating axis magnitudes and a cut-validity guard sized for "
            "O(1) normalised coordinates anchored at offset=0, both of "
            "which assume 'relative' normalisation. Use sampling or bbo "
            "mode for 'units'/'minmax'/'share'/'per_axes'."
        )
    per_axes_capex_reference = cfg.get("per_axes_capex_reference", 15e12)
    if (
        isinstance(per_axes_capex_reference, bool)
        or not isinstance(per_axes_capex_reference, (int, float))
        or not np.isfinite(per_axes_capex_reference)
        or per_axes_capex_reference <= 0
    ):
        raise ValueError(
            "MGA: per_axes_capex_reference must be a positive finite number, "
            f"got {per_axes_capex_reference!r}."
        )


class MGA:
    """Near-optimal exploration on a solved ZEN-garden model.

    Holds the linopy model and the helpers shared by both modes. The split
    between `setup()` and the per-iteration methods exists because
    `add_constraints` is not idempotent: the near-optimality constraint is
    added exactly once, while the objective is swapped per iteration via
    `add_objective(..., overwrite=True)`. Oracle mode additionally adds an
    L-infinity projection model once (`setup_projection_model`).
    """

    # Dims aggregated away per axis kind, besides the member dim itself.
    _TECH_AGG = ["set_capacity_types", "set_location", "set_time_steps_yearly"]
    _CARRIER_AGG = ["set_nodes", "set_time_steps_operation"]
    _NODE_AGG_CAPEX = [
        "set_technologies",
        "set_capacity_types",
        "set_time_steps_yearly",
    ]

    def __init__(
        self,
        optimization_setup,
        epsilon,
        postprocess_ctx,
        technologies=None,
        carrier_imports=None,
        include_cost=False,
        node_capex=None,
        node_capex_cumulative=None,
        node_capex_by_technology=None,
        node_capex_cumulative_tech=None,
        node_capacity_ratio=None,
        node_carbon_emissions_cumulative=None,
        normalisation="relative",
        per_axes_capex_reference=15e12,
    ):
        """
        Args:
            optimization_setup: OptimizationSetup holding the solved baseline
                (model.objective.value is C*).
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Dict of Postprocess arguments for the iteration
                outputs (scenarios, subfolder, model_name, scenario_name,
                param_map).
            technologies: Technology axes (names or {group: [members]} lumps).
            carrier_imports: Carrier-import axes, same format.
            include_cost: Add the total-cost axis (oracle mode only).
            node_capex: Per-node capex axes, summed over all model years
                (names or {group: [members]} lumps of node names).
            node_capex_cumulative: Per-node capex axes restricted to all
                model years up to a target calendar year: {"nodes": [...same
                format as node_capex...], "until_years": [2030, 2040, ...]}.
                Produces one axis per (node/lump, until_year) combination,
                named f"{name}_until_{until_year}". Axes sharing a node/lump
                group are constrained to be monotonically non-decreasing
                (see build_initial_outer_approximation()).
            node_capex_by_technology: Per-node capex axes restricted to a
                named technology group: {"nodes": [...same format as
                node_capex...], "technology_groups": [...same format,
                technology names/lumps...]}. Produces one axis per
                (node/lump, technology-group) combination.
            node_capex_cumulative_tech: Per-node capex axes restricted to
                both a target calendar year and a named technology group at
                once: {"nodes": [...same format as node_capex...],
                "until_years": [2030, 2040, ...], "technology_groups":
                [...same format, technology names/lumps...]}. Produces one
                axis per (node/lump, technology-group, until_year)
                combination, named f"{node_name}_{tech_name}_until_
                {until_year}". Axes sharing both a node/lump group and a
                technology group are constrained to be monotonically
                non-decreasing across ascending until_year (see
                build_initial_outer_approximation()); axes are never
                compared across different technology groups.
            node_capacity_ratio: Per-node axes of one named technology
                group's installed capacity as a fraction of another named
                technology group's: {"nodes": [...same format as
                node_capex...], "years": [2030, 2040, ...], "ratio_groups":
                [{group_name: {"numerator": [...technology names/lumps...],
                "denominator": [...technology names/lumps...]}}, ...]}.
                Produces one axis per (node/lump, ratio-group, year)
                combination, named f"{node_name}_{ratio_name}_{year}". The
                denominator is evaluated once on the baseline design z* and
                held fixed (see axes.py's Axis docstring) since a live
                ratio of two decision-variable sums cannot serve as an LP
                objective/projection term; axes are not constrained to be
                monotonic across years.
            node_carbon_emissions_cumulative: Per-node cumulative
                carbon-emissions axes restricted to all model years up to a
                target calendar year: {"nodes": [...same format as
                node_capex...], "until_years": [2030, 2040, ...]}. Produces
                one axis per (node/lump, until_year) combination, named
                f"{name}_until_{until_year}"; interval-expanded but not
                discounted (unlike the capex axes). Axes sharing a node/lump
                group are constrained to be monotonically non-decreasing
                (see build_initial_outer_approximation()).
            normalisation: "relative" (default) scales design axes by their
                near-optimal maximum; "minmax" maps each axis's own
                near-optimal [min, max] onto [0, 1]; "units" reports them in
                raw physical units; "share" (capex axes only) reports each
                axis as a fraction of a fixed reference total at the
                baseline design z*, rounded to one significant figure --
                one shared total across all nodes and the full model
                horizon for every node_capex/node_capex_cumulative axis,
                and a separate per-technology-group total (also all nodes,
                full horizon) for node_capex_by_technology and
                node_capex_cumulative_tech axes; both the rounded total and
                the raw value it came from are recorded per axis in
                polytope_metadata(). "per_axes" (node_capex_cumulative,
                node_capacity_ratio and node_carbon_emissions_cumulative
                axes only) reports each axis against its own kind-specific
                reference instead of one shared run-wide total:
                node_capex_cumulative divides by `per_axes_capex_reference`;
                node_capacity_ratio is already a fraction (reference 1);
                node_carbon_emissions_cumulative divides by the model's own
                carbon_emissions_budget parameter. The cost axis is always
                budget-relative.
                See solve_axis_bounds().
            per_axes_capex_reference: The fixed reference total (in the
                model's capex unit) that node_capex_cumulative axes divide
                by under normalisation="per_axes". Unused otherwise.
        """
        if epsilon <= 0:
            raise ValueError(f"MGA epsilon must be positive, got {epsilon!r}")
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.epsilon = epsilon
        self.postprocess_ctx = postprocess_ctx
        # Baseline objective C*, captured before MGA touches the model.
        self.c_star = self.model.objective.value

        self.capacity_addition = self.model.variables["capacity_addition"]
        # Selects the capacity type each axis aggregates per technology.
        self._capacity_mask = self._build_capacity_type_mask()

        all_technologies = list(
            self.capacity_addition.coords["set_technologies"].values
        )
        if "flow_import" in self.model.variables:
            all_carriers = list(
                self.model.variables["flow_import"].coords["set_carriers"].values
            )
        else:
            all_carriers = []
        self.all_nodes = list(optimization_setup.energy_system.set_nodes)
        (
            tech_groups,
            carrier_groups,
            node_capex_groups,
            node_capex_cumulative_axes,
            node_capex_tech_axes,
            node_capex_cumulative_chains,
            node_capex_cumulative_tech_axes,
            node_capex_cumulative_tech_chains,
            node_capacity_ratio_axes,
            node_carbon_emissions_cumulative_axes,
            node_carbon_emissions_cumulative_chains,
        ) = build_axis_groups(
            technologies,
            carrier_imports,
            all_technologies,
            all_carriers,
            node_capex=node_capex,
            node_capex_cumulative=node_capex_cumulative,
            node_capex_by_technology=node_capex_by_technology,
            node_capex_cumulative_tech=node_capex_cumulative_tech,
            node_capacity_ratio=node_capacity_ratio,
            node_carbon_emissions_cumulative=node_carbon_emissions_cumulative,
            all_nodes=self.all_nodes,
        )
        # Chains of cumulative axis names (same node/lump group, and for
        # node_capex_cumulative_tech also the same technology group,
        # ascending until_year/year), used by build_initial_outer_approximation
        # to add monotonicity rows; empty when there are no such axes.
        # node_capacity_ratio has no chains -- a ratio is not expected to be
        # monotonic across years.
        self._monotone_capex_chains = (
            node_capex_cumulative_chains
            + node_capex_cumulative_tech_chains
            + node_carbon_emissions_cumulative_chains
        )

        # The single source of truth for axis order everywhere downstream
        # (coordinates, cut normals, polytope columns): technology axes,
        # carrier axes, node-capex axes, node-capex-cumulative axes,
        # node-capex-technology axes, node-capex-cumulative-technology axes,
        # then the cost axis.
        self.axes: list[Axis] = (
            [
                Axis(
                    name,
                    TECH_CAPACITY,
                    tuple(members),
                    self._selected_capacity_type(name, members),
                )
                for name, members in tech_groups
            ]
            + [
                Axis(name, CARRIER_IMPORT, tuple(members), None)
                for name, members in carrier_groups
            ]
            + [
                Axis(name, NODE_CAPEX, tuple(members), None)
                for name, members in node_capex_groups
            ]
            + [
                Axis(
                    name,
                    NODE_CAPEX_CUMULATIVE,
                    tuple(members),
                    None,
                    period=(None, until_year),
                )
                for name, members, until_year in node_capex_cumulative_axes
            ]
            + [
                Axis(
                    name,
                    NODE_CAPEX_TECH,
                    tuple(node_members),
                    None,
                    technologies=tuple(tech_members),
                )
                for name, node_members, tech_members in node_capex_tech_axes
            ]
            + [
                Axis(
                    name,
                    NODE_CAPEX_CUMULATIVE_TECH,
                    tuple(node_members),
                    None,
                    period=(None, until_year),
                    technologies=tuple(tech_members),
                )
                for name, node_members, tech_members, until_year in (
                    node_capex_cumulative_tech_axes
                )
            ]
            + [
                Axis(
                    name,
                    NODE_CAPACITY_RATIO,
                    tuple(node_members),
                    self._matching_ratio_capacity_type(
                        name, numerator_members, denominator_members
                    ),
                    period=(year, year),
                    technologies=tuple(numerator_members),
                    denominator_technologies=tuple(denominator_members),
                )
                for name, node_members, numerator_members, denominator_members, year in (
                    node_capacity_ratio_axes
                )
            ]
            + [
                Axis(
                    name,
                    NODE_CARBON_EMISSIONS_CUMULATIVE,
                    tuple(members),
                    None,
                    period=(None, until_year),
                )
                for name, members, until_year in node_carbon_emissions_cumulative_axes
            ]
        )
        if include_cost:
            self.axes.append(Axis(COST_VARIABLE, TOTAL_COST, (), None))
        self.z_names = [axis.name for axis in self.axes]
        self.n_z = len(self.axes)

        # Model handles needed by carrier-import axes.
        if any(axis.kind == CARRIER_IMPORT for axis in self.axes):
            self.flow_import = self.model.variables["flow_import"]
            self._ts_duration = (
                optimization_setup.parameters.time_steps_operation_duration
            )
        else:
            self.flow_import = None
            self._ts_duration = None

        # Model handles needed by capacity-ratio axes: the *stock* capacity
        # variable (distinct from capacity_addition, which tech axes use).
        if any(axis.kind == NODE_CAPACITY_RATIO for axis in self.axes):
            self.capacity = self.model.variables["capacity"]
        else:
            self.capacity = None

        # Model handle needed by node-capex axes. cost_capex_yearly is
        # indexed by set_location, a per-technology-family dimension whose
        # coordinate values are node names for conversion/storage/
        # retrofitting technologies and edge names for transport
        # technologies; .sel(set_location=<node names>) therefore naturally
        # excludes transport-technology capex (invalid entries), the same
        # mechanism the tech-capacity axis already relies on.
        _node_capex_kinds = (
            NODE_CAPEX,
            NODE_CAPEX_CUMULATIVE,
            NODE_CAPEX_TECH,
            NODE_CAPEX_CUMULATIVE_TECH,
        )
        if any(axis.kind in _node_capex_kinds for axis in self.axes):
            self.cost_capex_yearly = self.model.variables["cost_capex_yearly"]
        else:
            self.cost_capex_yearly = None

        # Model handle needed by node-carbon-emissions-cumulative axes:
        # carbon_emissions_technology is an instantaneous per-operation-step
        # rate (like flow_import), annualised once here the same way
        # ZEN-garden's own constraint_carbon_emissions_technology_total
        # annualises it (energy_system.py), so it can be sliced/summed by
        # location/year in _design_axis_terms without re-annualising on
        # every call.
        if any(axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE for axis in self.axes):
            carbon_emissions_technology = self.model.variables[
                "carbon_emissions_technology"
            ]
            year_op_duration = (
                optimization_setup.energy_system.rules.get_year_time_step_duration_array()
            )
            self.carbon_emissions_technology_yearly = (
                carbon_emissions_technology * year_op_duration
            ).sum("set_time_steps_operation")
        else:
            self.carbon_emissions_technology_yearly = None

        # period->year-index lookup needed by every axis kind restricted to
        # a calendar-year window (cumulative or single-year snapshot).
        # Sourced directly from the model's own yearly time-step set/
        # calendar-year mapping, independent of whether any capex axis
        # exists (unlike the capex discount factor below, which capex axes
        # alone need).
        _period_kinds = (
            NODE_CAPEX_CUMULATIVE,
            NODE_CAPEX_CUMULATIVE_TECH,
            NODE_CAPACITY_RATIO,
            NODE_CARBON_EMISSIONS_CUMULATIVE,
        )
        self._axis_year_indices: dict[str, list] = {}
        if any(axis.kind in _period_kinds for axis in self.axes):
            year_indices = list(optimization_setup.energy_system.set_time_steps_yearly)
            real_years = list(optimization_setup.energy_system.set_time_steps_years)
            last_year_index = int(
                optimization_setup.energy_system.set_time_steps_yearly_entire_horizon[-1]
            )
            interval_between_years = int(optimization_setup.system.interval_between_years)
            for axis in self.axes:
                if axis.kind not in _period_kinds:
                    continue
                ids = _year_indices_in_period(axis.period, year_indices, real_years)
                if not ids:
                    target_year = axis.period[1]
                    raise ValueError(
                        f"MGA: axis {axis.name!r} year(s) up to {target_year} "
                        f"cover no model year (model years: "
                        f"{real_years[0]}-{real_years[-1]})."
                    )
                self._axis_year_indices[axis.name] = ids
        else:
            year_indices = None
            interval_between_years = None
            last_year_index = None

        # Discount + interval-expand each sampled year's capex the same way
        # net_present_cost itself is (constraint_net_present_cost in
        # energy_system.py) -- otherwise this axis sums each sampled year's
        # capex once, undiscounted, while net_present_cost counts it
        # interval_between_years times (once per real calendar year it
        # represents) and discounts each occurrence, making the two
        # non-comparable. Applied in _design_axis_terms.
        self._capex_discount_factor = (
            _year_interval_expansion_factors(
                year_indices,
                float(optimization_setup.parameters.discount_rate),
                interval_between_years,
                last_year_index,
            )
            if self.cost_capex_yearly is not None
            else None
        )

        # Carbon emissions aren't discounted, only interval-expanded (the
        # r=0 case of the same formula reduces to the plain interval
        # multiplicity ZEN-garden's own constraint_carbon_emissions_cumulative
        # uses -- see _year_interval_expansion_factors' docstring).
        self._emissions_interval_factor = (
            _year_interval_expansion_factors(
                year_indices, 0.0, interval_between_years, last_year_index
            )
            if self.carbon_emissions_technology_yearly is not None
            else None
        )

        # Fixed baseline denominators for node_capacity_ratio axes, read now
        # (see z_star_phys below) rather than recomputed per explored
        # point -- both technology groups are decision-variable sums, so a
        # live ratio would be a nonlinear term unusable as an LP objective.
        # Scoped to each axis's own node(s)/year (not widened to all nodes
        # like the "share" reference total, since this is per-axis, not a
        # shared group total).
        self._ratio_denominator_baseline: dict[str, float] = {
            axis.name: self._capacity_ratio_denominator_baseline(axis)
            for axis in self.axes
            if axis.kind == NODE_CAPACITY_RATIO
        }

        # Normalisation convention used by solve_axis_bounds(); the
        # scale/offset it computes below are set once and derived from this.
        self.normalisation = normalisation
        self.bounds_phys = None
        self.scale = None
        self.offset = None
        self.n_initial_rows = None
        # (origin label, physical point) per extreme design found by the
        # bound LPs; they seed the initial inner approximation.
        self._extreme_designs = []
        # Known near-optimal points in normalised coordinates; backs the
        # cut-validity guard in find_nearest_point.
        self._inner_points = None

        # Baseline point z*, read now while the baseline solution is still
        # loaded (the bound LPs overwrite it).
        self.z_star_phys = np.array(
            [self.axis_value(axis) for axis in self.axes], dtype=float
        )

        # Fixed baseline reference totals for normalisation="share", read now
        # alongside z_star_phys for the same reason -- not recomputed per
        # explored point (see _design_axis_reference_total). validate_config
        # guarantees every axis but TOTAL_COST is a capex-kind axis here.
        # The raw value is kept (and reported, see polytope_metadata()) for
        # traceability of how much _round_to_one_significant_figure changed
        # it; solve_axis_bounds() uses only the rounded scale.
        self._share_reference_phys_raw = (
            np.array(
                [
                    np.nan
                    if axis.kind == TOTAL_COST
                    else self._axis_reference_total_value(axis)
                    for axis in self.axes
                ],
                dtype=float,
            )
            if self.normalisation == "share"
            else None
        )
        self._share_reference_phys = (
            np.array(
                [
                    _round_to_one_significant_figure(v)
                    for v in self._share_reference_phys_raw
                ],
                dtype=float,
            )
            if self.normalisation == "share"
            else None
        )
        if self.normalisation == "share":
            # Log each distinct reference total once -- every node_capex/
            # node_capex_cumulative axis shares one, and each node_capex_tech
            # or node_capex_cumulative_tech technology group contributes its
            # own.
            logged_raw = set()
            for axis, raw in zip(self.axes, self._share_reference_phys_raw):
                if axis.kind == TOTAL_COST or raw in logged_raw:
                    continue
                logged_raw.add(raw)
                logging.info(
                    f"MGA: normalisation='share' reference total for "
                    f"{axis.name!r}'s group: {raw:.6g} rounded to "
                    f"{_round_to_one_significant_figure(raw):.6g}"
                )

        # Per-axis-kind reference for normalisation="per_axes": unlike
        # "share" (one shared baseline total per run), each axis kind
        # supplies its own physically-appropriate reference -- validate_config
        # guarantees every axis here is node_capex_cumulative,
        # node_capacity_ratio, node_carbon_emissions_cumulative or
        # TOTAL_COST.
        self._per_axes_reference: dict[str, float] = {}
        if self.normalisation == "per_axes":
            logged_kinds = set()
            for axis in self.axes:
                if axis.kind == NODE_CAPEX_CUMULATIVE:
                    ref = float(per_axes_capex_reference)
                elif axis.kind == NODE_CAPACITY_RATIO:
                    ref = 1.0
                elif axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE:
                    ref = float(optimization_setup.parameters.carbon_emissions_budget)
                else:
                    continue
                self._per_axes_reference[axis.name] = ref
                if axis.kind not in logged_kinds:
                    logged_kinds.add(axis.kind)
                    logging.info(
                        f"MGA: normalisation='per_axes' reference for kind "
                        f"{axis.kind!r}: {ref:.6g}"
                    )

        # 1-based, matching pyoNearOpt's iteration numbers in diagnostics.csv.
        self._iter_count = 1

    @property
    def design_axes(self) -> list[Axis]:
        """The axes whose bounds come from the model (everything but cost)."""
        return [axis for axis in self.axes if axis.kind != TOTAL_COST]

    @property
    def include_cost(self) -> bool:
        """Whether the exploration carries the total-cost axis."""
        return any(axis.kind == TOTAL_COST for axis in self.axes)

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        self.model.add_constraints(
            self._total_cost_expression() <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality",
        )
        logging.info(
            f"MGA: near-optimality constraint added, cost <= "
            f"{(1 + self.epsilon) * self.c_star} (C* = {self.c_star}, "
            f"epsilon = {self.epsilon})"
        )

    def _total_cost_expression(self):
        """The model's original total-cost objective as a linopy expression.

        This is exactly COST_VARIABLE summed over set_years, which has no
        other dimension -- so the projection equality built from this
        expression and the value read in axis_value are the same quantity.
        """
        return self.optimization_setup.energy_system.rules.objective_total_cost(
            self.model
        )

    # ------------------------------------------------------------------
    # weights mode
    # ------------------------------------------------------------------

    def run_iteration(self, weights: dict, iter_id: int):
        """Solve one weights-mode iteration: min sum_i w_i * axis_expression(axis_i).

        Weight keys are axis names from the axes config block (self.z_names)
        -- technology names or lumped tech groups, carrier-import axes,
        node-capex axes (optionally period- or technology-restricted), or
        the total-cost axis. The capacity-type mask applies to tech axes as
        in oracle mode, so storage technologies are weighted on their energy
        capacity only (summing power and energy would mix units).
        """
        obj = self._build_weighted_objective(weights)
        self.model.add_objective(obj, sense="min", overwrite=True)
        logging.info(f"MGA: iteration {iter_id} weights = {weights}")
        self._solve_and_postprocess(f"mga_iter_{iter_id}")

    def _build_weighted_objective(self, weights: dict):
        """Weighted sum of axis expressions: sum_i w_i * axis_expression(axis_i).

        Weight keys must be axis names (self.z_names, built from the axes
        config block); unknown keys raise. Axes without an explicit weight
        contribute nothing.
        """
        unknown = sorted(set(weights) - set(self.z_names))
        if unknown:
            raise ValueError(
                f"MGA weights mode: unknown axis name(s) {unknown}; "
                f"expected one of {self.z_names} (see the axes config block)."
            )
        terms = [
            float(weights[axis.name]) * self.axis_expression(axis)
            for axis in self.axes
            if axis.name in weights
        ]
        if not terms:
            raise ValueError("MGA weights mode: iteration has no (known) weights.")
        return sum(terms)

    # ------------------------------------------------------------------
    # oracle mode: axes on the model
    # ------------------------------------------------------------------

    def _build_capacity_type_mask(self):
        """0/1 mask over (set_technologies, set_capacity_types): the capacity
        type each axis aggregates per technology.

        Storage technologies (more than one active capacity type) keep only
        their energy type; every other technology keeps its single power
        type. Active entries are detected from the live variable (labels !=
        -1), so no technology list is hardcoded; "power" is
        system.set_capacity_types[0] by ZEN-garden convention.
        """
        type_dim = "set_capacity_types"
        other_dims = [
            d
            for d in self.capacity_addition.dims
            if d not in ("set_technologies", type_dim)
        ]
        active = (self.capacity_addition.labels != -1).any(other_dims)
        power_type = str(self.optimization_setup.system.set_capacity_types[0])
        known_types = [str(c) for c in self.capacity_addition.coords[type_dim].values]
        if power_type not in known_types:
            raise RuntimeError(
                f"MGA: power capacity type {power_type!r} not found in "
                f"capacity_addition."
            )
        is_storage = active.sum(type_dim) > 1
        is_power = active[type_dim] == power_type
        keep = active & ~(is_storage & is_power)
        storage_techs = [
            str(t)
            for t in active["set_technologies"].values
            if bool(is_storage.sel(set_technologies=t))
        ]
        logging.info(
            f"MGA: storage tech(s) {storage_techs} use energy capacity, all "
            f"other technologies their power capacity."
        )
        return keep.astype(float)

    def _selected_capacity_type(self, name: str, members: list) -> str:
        """The "+"-joined capacity type(s) the mask selects for one tech axis.

        Every member must map to the same selected type: lumping storage
        (energy, e.g. GWh) with non-storage (power, e.g. GW) members would
        mix incommensurable units.
        """
        selected = {
            member: tuple(
                str(c)
                for c in self._capacity_mask.coords["set_capacity_types"].values
                if float(
                    self._capacity_mask.sel(
                        set_technologies=member, set_capacity_types=c
                    )
                )
                > 0.5
            )
            for member in members
        }
        distinct = set(selected.values())
        if len(distinct) > 1:
            raise ValueError(
                f"MGA tech axis {name!r} mixes capacity types {selected}; "
                f"split storage and non-storage members into separate axes."
            )
        types = next(iter(distinct))
        if not types:
            raise RuntimeError(
                f"MGA tech axis {name!r}: no active capacity type for {members}."
            )
        return "+".join(types)

    def _matching_ratio_capacity_type(
        self, name: str, numerator_members: list, denominator_members: list
    ) -> str:
        """The capacity type shared by a node_capacity_ratio axis's
        numerator and denominator technology groups.

        Each group's type is resolved independently via
        _selected_capacity_type; the two must agree, since a ratio of power
        capacity to energy capacity (or vice versa) is not physically
        meaningful.
        """
        numerator_type = self._selected_capacity_type(
            f"{name} (numerator)", numerator_members
        )
        denominator_type = self._selected_capacity_type(
            f"{name} (denominator)", denominator_members
        )
        if numerator_type != denominator_type:
            raise ValueError(
                f"MGA node_capacity_ratio axis {name!r}: numerator capacity "
                f"type {numerator_type!r} != denominator capacity type "
                f"{denominator_type!r}."
            )
        return numerator_type

    def _capacity_ratio_denominator_baseline(self, axis: Axis) -> float:
        """A node_capacity_ratio axis's frozen denominator: its denominator
        technology group's total installed capacity, at the axis's own
        member node(s) and snapshot year, on the baseline (cost-optimal)
        solution. Computed once, in __init__, before any bound LP touches
        the model -- see the Axis class docstring for why the denominator
        is frozen rather than recomputed per explored design.

        Unlike normalisation="share"'s reference total, this is NOT widened
        to all nodes: it stays scoped to this axis's own region/year, since
        every (region, year) combination gets its own baseline, not one
        shared total across a run.
        """
        year_ids = self._axis_year_indices[axis.name]
        value = float(
            (self._capacity_mask * self.capacity.solution)
            .sel(
                set_technologies=list(axis.denominator_technologies),
                set_location=list(axis.members),
                set_time_steps_yearly=year_ids,
            )
            .sum()
        )
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"MGA: node_capacity_ratio axis {axis.name!r} has a "
                f"non-positive baseline denominator {value:.6g} (no member "
                f"of {axis.denominator_technologies} has capacity at "
                f"{axis.members}/{axis.period[1]} in the baseline design); "
                f"remove the axis or check its denominator technology group."
            )
        return value

    def _design_axis_terms(
        self,
        axis: Axis,
        capacity_addition,
        flow,
        capex=None,
        capacity_stock=None,
        emissions=None,
    ):
        """One design axis over `capacity_addition`/`flow`/`capex`/
        `capacity_stock`/`emissions` data.

        The arguments are either the linopy variables/precomputed
        expressions (yielding the axis LinearExpression) or their
        `.solution` arrays (yielding the axis value): technology axes sum
        the capacity addition of the member technologies, restricted by the
        capacity-type mask; carrier axes sum the duration-weighted annual
        import of the member carriers, sum_{m,n,t} tau_t * flow[m, n, t];
        node-capex axes sum the annualised capex of the member nodes over
        all capacity types, over all model years (NODE_CAPEX) or every
        model year up to the axis's until_year (NODE_CAPEX_CUMULATIVE), and
        over either all technologies or just the axis's technology group
        (NODE_CAPEX_TECH), or both a year cutoff and a technology group at
        once (NODE_CAPEX_CUMULATIVE_TECH); every node-capex kind weights
        each sampled year's capex by `_capex_discount_factor` before
        summing, reproducing ZEN-garden's own net_present_cost
        discounting/interval-expansion (see
        _year_interval_expansion_factors) so this axis is on the same
        accounting basis as the TOTAL_COST axis. NODE_CARBON_EMISSIONS_CUMULATIVE
        axes sum the annualised carbon emissions of the member nodes over
        every model year up to the axis's until_year, weighted by
        `_emissions_interval_factor` (interval-expansion only, no
        discounting -- emissions aren't discounted). NODE_CAPACITY_RATIO
        axes are numerator_capacity / denominator_baseline: the live
        installed capacity of the axis's technology (numerator) group at
        its member node(s) and snapshot year, divided by the precomputed,
        baseline-frozen `_ratio_denominator_baseline` scalar for that axis
        (see the Axis class docstring for why the denominator is frozen).
        """
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            return (
                (self._capacity_mask * capacity_addition)
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        if axis.kind == CARRIER_IMPORT:
            return (self._ts_duration * flow.sel(set_carriers=members)).sum(
                self._CARRIER_AGG + ["set_carriers"]
            )
        if axis.kind == NODE_CAPACITY_RATIO:
            year_ids = self._axis_year_indices[axis.name]
            numerator = (
                (self._capacity_mask * capacity_stock)
                .sel(
                    set_technologies=list(axis.technologies),
                    set_location=members,
                    set_time_steps_yearly=year_ids,
                )
                .sum(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"])
            )
            return numerator / self._ratio_denominator_baseline[axis.name]
        if axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE:
            # Transport technologies have no valid entry at a node-named
            # location, so they drop out of this selection for free (same
            # mechanism as the node-capex kinds below).
            term = emissions.sel(set_location=members)
            year_ids = self._axis_year_indices[axis.name]
            term = term.sel(set_time_steps_yearly=year_ids)
            term = term * self._emissions_interval_factor
            return term.sum(["set_technologies", "set_location", "set_time_steps_yearly"])
        # NODE_CAPEX / NODE_CAPEX_CUMULATIVE / NODE_CAPEX_TECH /
        # NODE_CAPEX_CUMULATIVE_TECH: transport technologies have no valid
        # entry at a node-named location, so they drop out of this selection
        # for free (same mechanism as _TECH_AGG's unrestricted set_location
        # sum for tech-capacity axes).
        term = capex.sel(set_location=members)
        if axis.technologies is not None:
            term = term.sel(set_technologies=list(axis.technologies))
        year_ids = self._axis_year_indices.get(axis.name)
        if year_ids is not None:
            term = term.sel(set_time_steps_yearly=year_ids)
        # Discount/interval-expand each sampled year's capex the same way
        # net_present_cost is (see __init__'s _capex_discount_factor); xarray
        # aligns term's (possibly year-restricted) set_time_steps_yearly
        # coordinate against the factor automatically.
        term = term * self._capex_discount_factor
        return term.sum(self._NODE_AGG_CAPEX + ["set_location"])

    def axis_expression(self, axis: Axis):
        """Linopy expression of one axis (bound LP objective, projection)."""
        if axis.kind == TOTAL_COST:
            return self._total_cost_expression()
        return self._design_axis_terms(
            axis,
            self.capacity_addition,
            self.flow_import,
            capex=self.cost_capex_yearly,
            capacity_stock=self.capacity,
            emissions=self.carbon_emissions_technology_yearly,
        )

    def axis_value(self, axis: Axis) -> float:
        """Value of one axis on the currently loaded solution."""
        if axis.kind == TOTAL_COST:
            return float(self.model.variables[COST_VARIABLE].solution.sum())
        flow = None if self.flow_import is None else self.flow_import.solution
        capex = (
            None if self.cost_capex_yearly is None else self.cost_capex_yearly.solution
        )
        capacity_stock = None if self.capacity is None else self.capacity.solution
        emissions = (
            None
            if self.carbon_emissions_technology_yearly is None
            else self.carbon_emissions_technology_yearly.solution
        )
        return float(
            self._design_axis_terms(
                axis,
                self.capacity_addition.solution,
                flow,
                capex=capex,
                capacity_stock=capacity_stock,
                emissions=emissions,
            )
        )

    def _design_axis_reference_total(self, axis: Axis, capacity, flow, capex=None):
        """Reference total for normalisation="share": the same
        `_design_axis_terms` computation as `axis`, with its node membership
        widened to every node in the model and any year-window restriction
        removed -- the sentinel `name` is never a key in
        `_axis_year_indices`, so a NODE_CAPEX_CUMULATIVE axis's own
        until_year no longer applies, and every NODE_CAPEX/_CUMULATIVE axis
        in a run shares one identical full-horizon, all-node total.
        `axis.technologies` is left untouched, so NODE_CAPEX_TECH and
        NODE_CAPEX_CUMULATIVE_TECH axes still divide by their own technology
        group's full-horizon, all-node total, not the grand total. Only
        called for capex-kind axes
        (NODE_CAPEX/_CUMULATIVE/_TECH/_CUMULATIVE_TECH) --
        normalisation="share" is rejected at config-validation time for any
        other axis kind, including TOTAL_COST.
        """
        widened = dataclasses.replace(
            axis, name="__share_reference_total__", members=tuple(self.all_nodes)
        )
        return self._design_axis_terms(widened, capacity, flow, capex=capex)

    def _axis_reference_total_value(self, axis: Axis) -> float:
        """Value of one axis's share reference total on the baseline
        solution (called once, in __init__, alongside z_star_phys)."""
        flow = None if self.flow_import is None else self.flow_import.solution
        capex = (
            None if self.cost_capex_yearly is None else self.cost_capex_yearly.solution
        )
        return float(
            self._design_axis_reference_total(
                axis, self.capacity_addition.solution, flow, capex=capex
            )
        )

    # ------------------------------------------------------------------
    # oracle mode: coordinates
    # ------------------------------------------------------------------

    def to_norm(self, point_phys) -> np.ndarray:
        """Physical point -> normalised coordinates."""
        assert self.scale is not None, "solve_axis_bounds() must run first"
        return (np.asarray(point_phys, dtype=float) - self.offset) / self.scale

    def to_phys(self, point_norm) -> np.ndarray:
        """Normalised point -> physical coordinates."""
        assert self.scale is not None, "solve_axis_bounds() must run first"
        return np.asarray(point_norm, dtype=float) * self.scale + self.offset

    def current_point_norm(self) -> np.ndarray:
        """The most recent solve as a normalised point, in axis order."""
        return self.to_norm([self.axis_value(axis) for axis in self.axes])

    @property
    def z_star_norm(self) -> np.ndarray:
        """The baseline point z* in normalised coordinates."""
        return self.to_norm(self.z_star_phys)

    def polytope_metadata(self) -> dict:
        """Self-describing metadata for the saved polytope: per-axis kind,
        members, capacity type, period, technologies, physical unit and (for
        normalisation="share") the rounded reference total actually used as
        scale plus the raw baseline value it was rounded from, or (for
        normalisation="per_axes") the per-axis-kind reference actually used
        as scale, alongside the normalisation convention (schema owned by
        polytope_io). node_capacity_ratio axes additionally report their
        denominator technology group and the baseline value it was frozen
        at, regardless of normalisation mode, since that freezing is a
        property of the axis itself, not of "per_axes"."""
        units = self.optimization_setup.variables.units
        ureg = self.optimization_setup.energy_system.unit_handling.ureg
        return {
            "axes": [
                {
                    "name": axis.name,
                    "kind": axis.kind,
                    "members": list(axis.members),
                    "capacity_type": axis.capacity_type,
                    "period": list(axis.period) if axis.period is not None else None,
                    "technologies": (
                        list(axis.technologies)
                        if axis.technologies is not None
                        else None
                    ),
                    "denominator_technologies": (
                        list(axis.denominator_technologies)
                        if axis.denominator_technologies is not None
                        else None
                    ),
                    "capacity_ratio_denominator_baseline": (
                        float(self._ratio_denominator_baseline[axis.name])
                        if axis.kind == NODE_CAPACITY_RATIO
                        else None
                    ),
                    "unit": axis_physical_unit(axis, units, ureg),
                    "share_reference_total": (
                        float(self._share_reference_phys[i])
                        if self.normalisation == "share" and axis.kind != TOTAL_COST
                        else None
                    ),
                    "share_reference_total_raw": (
                        float(self._share_reference_phys_raw[i])
                        if self.normalisation == "share" and axis.kind != TOTAL_COST
                        else None
                    ),
                    "per_axes_reference": (
                        float(self._per_axes_reference[axis.name])
                        if self.normalisation == "per_axes" and axis.kind != TOTAL_COST
                        else None
                    ),
                }
                for i, axis in enumerate(self.axes)
            ],
            "normalisation": (
                "physical = normalised * scale + offset, per axis; design "
                "axes use "
                + (
                    "(1, 0) -- raw physical units"
                    if self.normalisation == "units"
                    else "(upper bound - lower bound, lower bound) -- "
                    "near-optimal [min, max] mapped to [0, 1]"
                    if self.normalisation == "minmax"
                    else "(reference total, 0) -- share of a fixed "
                    "baseline (z*) reference total; one shared total per "
                    "run for node_capex/node_capex_cumulative (all nodes, "
                    "full horizon), a separate per-technology-group total "
                    "for node_capex_tech/node_capex_cumulative_tech"
                    if self.normalisation == "share"
                    else "(per-axis-kind reference, 0) -- "
                    "node_capex_cumulative divides by a fixed configurable "
                    "constant, node_capacity_ratio is already a fraction "
                    "(reference 1), node_carbon_emissions_cumulative divides "
                    "by the model's carbon_emissions_budget"
                    if self.normalisation == "per_axes"
                    else "(upper bound, 0)"
                )
                + ", the cost axis always (epsilon * c_star, c_star)"
            ),
        }

    # ------------------------------------------------------------------
    # oracle mode: bounds, projection model, ORACLE callback
    # ------------------------------------------------------------------

    def solve_axis_bounds(self, supplied_bounds=None) -> None:
        """Determine the per-axis bounds and the normalisation they define.

        With `supplied_bounds` None this is a full VMM (variable min/max)
        pass: two LPs per design axis drive it to its near-optimal minimum
        and maximum, which yields certified bounds and, as a by-product, the
        2 * n_design extreme designs that seed the inner approximation. A
        supplied dict {axis: [lower, upper]} must cover every design axis and
        replaces those LPs entirely, so the bounds are as good as the caller's
        claim and no extreme designs are available.

        The cost axis is never solved for: the near-optimality constraint
        defines its bounds as [C*, (1 + epsilon) * C*] exactly.

        Sets bounds_phys, scale and offset; call after setup().
        """
        if supplied_bounds is None:
            upper = self.solve_extreme_lps("max")
            # Axis values are sums of non-negative variables (capacity_addition,
            # flow_import and cost_capex_yearly are all bounds=(0, inf) in
            # ZEN-garden), so a negative minimum can only be solver round-off.
            lower = np.maximum(0.0, self.solve_extreme_lps("min"))
        else:
            lower, upper = self._read_supplied_bounds(supplied_bounds)

        bounds, scale, offset = [], [], []
        design_index = 0
        for i, axis in enumerate(self.axes):
            if axis.kind == TOTAL_COST:
                lo, hi = self.c_star, (1.0 + self.epsilon) * self.c_star
                # 0 is the cost optimum, 1 the near-optimality budget.
                bounds.append((lo, hi))
                scale.append(hi - lo)
                offset.append(lo)
                continue
            lo, hi = float(lower[design_index]), float(upper[design_index])
            design_index += 1
            if not np.isfinite(hi) or hi <= 0:
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
            if self.normalisation == "units":
                # The axis is reported in its own physical unit.
                scale.append(1.0)
                offset.append(0.0)
            elif self.normalisation == "minmax":
                # The axis spans exactly [0, 1] between its own near-optimal
                # min and max, so a fixed exploration tolerance means the
                # same fraction of *this axis's* range for every axis --
                # unlike "relative", which wastes [0, lo/hi) whenever an
                # axis's near-optimal minimum sits well above 0.
                span = hi - lo
                if span <= 0:
                    raise RuntimeError(
                        f"MGA: axis {axis.name!r} has zero near-optimal "
                        f"range (lower={lo:.6g} == upper={hi:.6g}); "
                        f"normalisation='minmax' requires a positive span. "
                        f"Remove the axis or use 'relative'/'units'."
                    )
                scale.append(span)
                offset.append(lo)
            elif self.normalisation == "share":
                # The axis reaches 1 if it accounts for its whole group's
                # baseline (z*) total -- a fixed reference, not recomputed
                # per explored point (see _design_axis_reference_total).
                ref = float(self._share_reference_phys[i])
                if not np.isfinite(ref) or ref <= 0:
                    raise RuntimeError(
                        f"MGA: axis {axis.name!r} has non-positive baseline "
                        f"reference total {ref:.6g}; normalisation='share' "
                        f"requires a positive reference. Remove the axis or "
                        f"use 'relative'/'minmax'/'units'."
                    )
                scale.append(ref)
                offset.append(0.0)
            elif self.normalisation == "per_axes":
                # The axis reaches 1 at its own kind's physically-appropriate
                # reference -- a fixed value per axis kind, not one shared
                # run-wide total (see __init__'s _per_axes_reference).
                ref = float(self._per_axes_reference[axis.name])
                if not np.isfinite(ref) or ref <= 0:
                    raise RuntimeError(
                        f"MGA: axis {axis.name!r} has non-positive "
                        f"per_axes reference {ref:.6g}; normalisation="
                        f"'per_axes' requires a positive reference. Remove "
                        f"the axis or use 'relative'/'minmax'/'units'."
                    )
                scale.append(ref)
                offset.append(0.0)
            else:
                # The axis reaches 1 at its near-optimal maximum.
                scale.append(hi)
                offset.append(0.0)

        self.bounds_phys = np.array(bounds, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.offset = np.array(offset, dtype=float)
        if self.normalisation == "units":
            units = self.optimization_setup.variables.units
            ureg = self.optimization_setup.energy_system.unit_handling.ureg
            seen = {axis_physical_unit(axis, units, ureg) for axis in self.design_axes}
            if len(seen) > 1:
                logging.warning(
                    "MGA: normalisation='units' but design axes span "
                    "multiple physical units (%s); a single tolerance will "
                    "not be comparable across them.",
                    sorted(unit for unit in seen if unit),
                )
        logging.info(
            f"MGA: bounds of {self.n_z} axes ready "
            f"({'supplied' if supplied_bounds else 'VMM LPs'}); normalised "
            f"lower bounds "
            f"{np.round((self.bounds_phys[:, 0] - self.offset) / self.scale, 4)}"
        )

    def _read_supplied_bounds(self, supplied_bounds):
        """Validate a user-supplied bounds dict and return (lower, upper)."""
        if not isinstance(supplied_bounds, dict):
            raise ValueError(
                f"MGA initial_bounds must be 'vmm' or a dict of "
                f"{{axis: [lower, upper]}}, got {type(supplied_bounds).__name__}."
            )
        design_names = [axis.name for axis in self.design_axes]
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
                    f"MGA initial_bounds[{name!r}] must be [lower, upper], "
                    f"got {pair!r}."
                )
            lower.append(float(pair[0]))
            upper.append(float(pair[1]))
        return np.array(lower), np.array(upper)

    def solve_extreme_lps(self, sense: str) -> np.ndarray:
        """Drive every design axis to one near-optimal extreme.

        One LP per design axis with the axis value as objective, `sense`
        being "max" or "min" -- the two halves of VMM. Each LP must reach a
        proven optimum: an outer bound derived from an unconverged solve
        would not be valid ("unbounded" on a max means the axis has no finite
        near-optimal maximum). Every solution is written via Postprocess as
        <model_name>_vmm_<sense>_<axis> and recorded as an extreme design.
        Returns the extreme values in design-axis order.
        """
        values = []
        for axis in self.design_axes:
            self.model.add_objective(
                self.axis_expression(axis), sense=sense, overwrite=True
            )
            start = time.time()
            self.optimization_setup.solve()
            if not self.optimization_setup.optimality:
                raise RuntimeError(
                    f"MGA VMM {sense} LP for axis {axis.name!r} ended with "
                    f"{self.model.termination_condition!r}."
                )
            self._postprocess(f"vmm_{sense}_{axis.name}")
            values.append(self.axis_value(axis))
            self._extreme_designs.append(
                (
                    f"{sense}:{axis.name}",
                    np.array([self.axis_value(a) for a in self.axes], dtype=float),
                )
            )
            logging.info(
                f"MGA: VMM {sense} {axis.name} = {values[-1]:.6g} "
                f"(LP took {time.time() - start:.1f} s)"
            )
        return np.array(values, dtype=float)

    def initial_inner_points(self) -> tuple[np.ndarray, list[str]]:
        """The certified points seeding the inner approximation.

        Returns (X0, origins): the baseline z* followed by every extreme
        design found by the bound LPs, in normalised coordinates, with a
        provenance label per row.
        """
        points = [self.z_star_norm]
        origins = ["z_star"]
        for label, point_phys in self._extreme_designs:
            points.append(self.to_norm(point_phys))
            origins.append(label)
        return np.vstack(points), origins

    def build_initial_outer_approximation(self) -> tuple[np.ndarray, np.ndarray]:
        """(A0, b0) of the initial outer polytope in normalised coordinates.

        Two rows per axis, lower and upper, uniformly for design and cost
        axes: -z_i <= -lower_i and z_i <= upper_i in physical units. Plus,
        for every consecutive pair (earlier, later) in each chain of
        `self._monotone_capex_chains`, one row z_phys(earlier) <=
        z_phys(later): cost_capex_yearly is non-negative and each
        cumulative-capex axis's year window nests inside the next, so this
        holds for every feasible model point and is a free tightening of the
        outer approximation, not an extra restriction on the model. Each raw
        row a^T z_phys <= b maps to normalised coordinates via
        z_phys = offset + diag(scale) z_norm, i.e. it becomes
        (a o scale)^T z_norm <= b - a^T offset, and the rows are then scaled
        to unit length (see normalise_rows). The result is exactly the box
        lower_i/upper_i <= z_norm,i <= 1 for design axes and 0 <= z_norm <= 1
        for the cost axis, plus the monotonicity half-spaces.
        """
        n_z = self.n_z
        A0 = np.vstack([-np.eye(n_z), np.eye(n_z)])
        b0 = np.concatenate([-self.bounds_phys[:, 0], self.bounds_phys[:, 1]])

        if self._monotone_capex_chains:
            index = {name: i for i, name in enumerate(self.z_names)}
            mono_rows = []
            for chain in self._monotone_capex_chains:
                for earlier, later in zip(chain, chain[1:], strict=False):
                    row = np.zeros(n_z)
                    row[index[earlier]] = 1.0
                    row[index[later]] = -1.0
                    mono_rows.append(row)
            if mono_rows:
                A0 = np.vstack([A0, np.array(mono_rows)])
                b0 = np.concatenate([b0, np.zeros(len(mono_rows))])

        b0 = b0 - A0 @ self.offset  # must precede the column scaling below
        A0 = A0 @ np.diag(self.scale)
        A0, b0 = normalise_rows(A0, b0)

        # Sanity check: z* must satisfy the initial outer approximation.
        violation = A0 @ self.z_star_norm - b0
        if (violation > 1e-6 * (np.abs(b0) + 1.0)).any():
            raise RuntimeError(
                f"MGA: initial outer approximation excludes z* "
                f"(max violation {violation.max():.3g})."
            )
        self.n_initial_rows = A0.shape[0]
        logging.info(
            f"MGA: outer approximation of {A0.shape[0]} unit-norm rows over "
            f"{n_z} normalised axes."
        )
        return A0, b0

    def _postprocess(self, label: str) -> None:
        """Write a Postprocess folder `<model_name>_<label>` for the
        currently loaded solution."""
        ctx = self.postprocess_ctx
        Postprocess(
            self.optimization_setup,
            scenarios=ctx["scenarios"],
            subfolder=ctx["subfolder"],
            model_name=f"{ctx['model_name']}_{label}",
            scenario_name=ctx["scenario_name"],
            param_map=ctx["param_map"],
        )

    def _solve_and_postprocess(self, label: str) -> None:
        """Solve the current model state and write its Postprocess folder."""
        self.optimization_setup.solve()
        if not self.optimization_setup.optimality:
            raise RuntimeError(
                f"MGA solve {label!r} failed: termination = "
                f"{self.model.termination_condition}"
            )
        self._postprocess(label)

    def setup_projection_model(self, initial_points=None) -> None:
        """Add the L-infinity projection model to the linopy model; call once.

        Variables: delta (one entry per axis on Z_DIM) and a scalar t.
        Constraints: one projection equality per axis,
        axis_expr_i - delta_i == trial_i (the right-hand side is updated per
        iteration in find_nearest_point), and the scaled t-bounds
        |delta_i| / scale_i <= t, so that min t is the normalised
        L-infinity distance to the trial point.

        `initial_points` (rows in normalised coordinates) seeds the
        cut-validity guard; it defaults to z* alone and should match the X
        handed to ORACLE.
        """
        z_coord = xr.DataArray(
            np.array(self.z_names), dims=Z_DIM, coords={Z_DIM: self.z_names}
        )
        self.delta = self.model.add_variables(
            coords=[z_coord],
            name="mga_oracle_delta",
            lower=-np.inf,
            upper=np.inf,
        )
        self.t_var = self.model.add_variables(
            coords=[_scalar_da(0)],
            name="mga_oracle_t",
            lower=0.0,
        )

        for i, axis in enumerate(self.axes):
            self.model.add_constraints(
                self.axis_expression(axis) - self.delta.sel({Z_DIM: axis.name}) == 0.0,
                name=f"mga_oracle_proj_eq_axis{i}",
            )

        d_scale = xr.DataArray(
            1.0 / self.scale, dims=Z_DIM, coords={Z_DIM: self.z_names}
        )
        self.model.add_constraints(
            d_scale * self.delta - self.t_var <= 0, name="mga_oracle_t_pos"
        )
        self.model.add_constraints(
            -(d_scale * self.delta) - self.t_var <= 0, name="mga_oracle_t_neg"
        )

        # Seed the cut-validity guard with the initial inner points.
        if initial_points is None:
            initial_points = [self.z_star_norm]
        self._inner_points = [np.asarray(p, dtype=float) for p in initial_points]
        logging.info(
            f"MGA oracle: projection model added (n_z = {self.n_z}, "
            f"{len(self._inner_points)} initial inner point(s))"
        )

    def find_nearest_point(self, trial_point: np.ndarray):
        """pyoNearOpt callback: project one trial point onto the near-optimal
        space and return it with its supporting cut.

        `trial_point` arrives in normalised coordinates. Returns
        (z_feas, dist, mu_cut, b_cut, 0) in the same coordinates: dist = t*
        is the normalised L-infinity distance, mu_cut is the dual of the
        projection equalities rescaled into normalised coordinates
        (mu_phys o scale; the affine offset cancels), and
        b_cut = mu_cut @ z_feas -- possibly relaxed by the cut-validity guard
        (CUT_GUARD_TRIGGER). The plane's scaling is left to pyoNearOpt, which
        normalises every cut it accepts.
        """
        assert trial_point.shape == (
            self.n_z,
        ), f"trial point shape {trial_point.shape}, expected ({self.n_z},)"

        # Projection-equality RHS <- the trial point in physical coordinates.
        trial_phys = self.to_phys(trial_point)
        for i in range(self.n_z):
            self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].rhs = float(
                trial_phys[i]
            )

        # min t; .sum() collapses the trivial scalar dim for the objective.
        self.model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self._iter_count}"
        logging.info(
            f"MGA oracle: starting iteration {self._iter_count}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        try:
            self._solve_and_postprocess(label)
        except RuntimeError:
            # One numerically distressed projection must not kill a long run:
            # retry once (threaded barrier solves are not deterministic); on a
            # second failure return a zero-distance copy of the previous inner
            # point, which stops ORACLE gracefully via its identical-point
            # check, so artifacts and the final certificate still happen.
            logging.exception(
                f"MGA oracle iter {self._iter_count}: projection solve failed; "
                f"retrying once."
            )
            try:
                self._solve_and_postprocess(label)
            except RuntimeError:
                logging.error(
                    "MGA oracle: projection failed twice; stopping ORACLE "
                    "gracefully."
                )
                z_prev = np.asarray(self._inner_points[-1], dtype=float).copy()
                self._iter_count += 1
                return z_prev, 0.0, None, None, 0

        z_feas = self.current_point_norm()
        dist = float(self.t_var.solution.values[0])

        # Cut normal: duals of the projection equalities, rescaled into
        # normalised coordinates.
        mu_phys = np.array(
            [
                float(self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].dual.values)
                for i in range(self.n_z)
            ],
            dtype=float,
        )
        mu_cut = mu_phys * self.scale
        b_cut = float(mu_cut @ z_feas)

        # Cut-validity guard: a valid supporting hyperplane keeps every known
        # near-optimal point inside the outer approximation. If inexact duals
        # violate that, keep the direction but relax the offset out to the
        # farthest known inner point.
        if self._inner_points is not None:
            points = np.vstack(self._inner_points)
            worst = float((points @ mu_cut - b_cut).max())
            if worst > CUT_GUARD_TRIGGER:
                logging.warning(
                    f"MGA oracle iter {self._iter_count}: cut would remove "
                    f"known near-optimal point(s) by up to {worst:.4g}; "
                    f"relaxing b_cut {b_cut:.6g} -> {b_cut + worst:.6g}"
                )
                b_cut = float(b_cut + worst + 1e-9)
                if float(mu_cut @ trial_point) <= b_cut:
                    # The relaxed plane no longer separates the trial point;
                    # ORACLE may stall on a repeating trial point -- a loud
                    # stop, preferred over corrupting the outer approximation.
                    logging.error(
                        "MGA oracle: relaxed cut no longer excludes the trial "
                        "point (unreliable projection duals this iteration)."
                    )
            self._inner_points.append(np.asarray(z_feas, dtype=float))

        logging.info(
            f"MGA oracle iter {self._iter_count}: dist = {dist:.4g} "
            f"(normalised L-inf), |mu|_max = {np.max(np.abs(mu_cut)):.4g}, "
            f"b_cut = {b_cut:.4g}"
        )
        self._iter_count += 1
        return z_feas, dist, mu_cut, b_cut, 0

    # ------------------------------------------------------------------
    # support-function modes (sampling, bbo): support-function callback
    # ------------------------------------------------------------------

    def solve_direction(self, direction: np.ndarray, label: str):
        """Set the objective to `direction`, solve, postprocess to `label`.

        Shared unit of work behind `support_function` (sequential modes)
        and batch mode's workers, which call this directly with their own
        label instead of `_iter_count`.

        Returns (z_feas, support_value): z_feas is the solution in normalised
        coordinates, support_value = direction @ z_feas.
        """
        if direction.shape != (self.n_z,):
            raise ValueError(f"direction shape {direction.shape}, expected ({self.n_z},)")

        obj = sum(
            (float(direction[i]) / self.scale[i]) * self.axis_expression(axis)
            for i, axis in enumerate(self.axes)
        )
        self.model.add_objective(obj, sense="max", overwrite=True)

        logging.info(f"MGA solve_direction {label!r}: ||direction||_2 = {np.linalg.norm(direction):.4g}")
        self._solve_and_postprocess(label)

        z_feas = self.current_point_norm()
        support_value = float(direction @ z_feas)
        logging.info(f"MGA solve_direction {label!r}: support_value = {support_value:.4g}")
        return z_feas, support_value

    def support_function(self, direction: np.ndarray):
        """pyoNearOpt callback for support-function methods (sampling, bbo):
        the point maximising `direction` and its value.

        Thin wrapper around `solve_direction` that owns the `supf_iter_<n>`
        iteration-labeling/counting sequential modes use.

        Returns (z_feas, support_value): z_feas is the solution in normalised
        coordinates, support_value = direction @ z_feas.
        """
        label = f"supf_iter_{self._iter_count}"
        logging.info(f"MGA support_function: starting iteration {self._iter_count}")
        z_feas, support_value = self.solve_direction(direction, label)
        logging.info(f"MGA support_function iter {self._iter_count}: support_value = {support_value:.4g}")
        self._iter_count += 1
        return z_feas, support_value


# ----------------------------------------------------------------------
# Event handler and mode dispatch
# ----------------------------------------------------------------------


@EventPublisher.register(Event.after_solve)
def run_mga(
    *, optimization_setup, scenarios, subfolder, model_name, scenario_name, param_map
):
    """Run MGA after the baseline solve.

    Returns the oracle summary directory (oracle mode) or None.
    """
    validate_config(config)
    mode = config["mode"]
    if mode not in ("weights", "oracle", "sampling", "bbo", "batch"):
        raise ValueError(
            f"Unknown MGA mode: {mode!r}. Expected 'weights', 'oracle', "
            f"'sampling', 'bbo' or 'batch'."
        )
    if optimization_setup.system.use_rolling_horizon:
        raise ValueError(
            "MGA does not support rolling-horizon runs: after_solve fires "
            "after the horizon loop, so MGA would explore only the final "
            "step's model."
        )
    if optimization_setup.solver.use_scaling:
        raise ValueError(
            "MGA does not support solver.use_scaling: re-scaling restores "
            "only the solution values, so MGA would add physical-unit "
            "expressions to a still-scaled model."
        )
    if optimization_setup.analysis.objective != "total_cost":
        raise ValueError(
            f"MGA defines near-optimality on the total cost, but the "
            f"baseline objective is {optimization_setup.analysis.objective!r}."
        )
    if mode == "weights" and not config["iterations"]:
        logging.warning("MGA: weights mode without iterations; skipping.")
        return None
    axes_cfg = config["axes"]

    logging.info(f"MGA: mode = {mode!r}, epsilon = {config['epsilon']}")
    mga = MGA(
        optimization_setup,
        epsilon=config["epsilon"],
        postprocess_ctx={
            "scenarios": scenarios,
            "subfolder": subfolder,
            "model_name": model_name,
            "scenario_name": scenario_name,
            "param_map": param_map,
        },
        technologies=axes_cfg.get("technologies"),
        carrier_imports=axes_cfg.get("carrier_imports"),
        include_cost=bool(axes_cfg.get("include_cost", False)),
        node_capex=axes_cfg.get("node_capex"),
        node_capex_cumulative=axes_cfg.get("node_capex_cumulative"),
        node_capex_by_technology=axes_cfg.get("node_capex_by_technology"),
        node_capex_cumulative_tech=axes_cfg.get("node_capex_cumulative_tech"),
        node_capacity_ratio=axes_cfg.get("node_capacity_ratio"),
        node_carbon_emissions_cumulative=axes_cfg.get(
            "node_carbon_emissions_cumulative"
        ),
        normalisation=config.get("normalisation", "relative"),
        per_axes_capex_reference=config.get("per_axes_capex_reference", 15e12),
    )
    mga.setup()

    result = None
    if mode == "weights":
        for i, iteration in enumerate(config["iterations"]):
            mga.run_iteration(iteration["weights"], i)
    elif mode == "oracle":
        result = run_oracle_mode(mga, config["oracle"])
    elif mode == "sampling":
        result = run_sampling_mode(mga, config["sampling"])
    elif mode == "bbo":
        result = run_bbo_mode(mga, config["bbo"])
    else:
        result = run_batch_mode(mga, config["batch"])
    logging.info("MGA: complete.")
    return result
