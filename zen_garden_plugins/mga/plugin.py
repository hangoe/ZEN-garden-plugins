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
                or the total cost (see axes.py)
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
                reported in their own physical units. Either way the cost
                axis uses scale = epsilon * C* and offset = C*, so 0 is the
                cost optimum and 1 the budget.

Every solve is written to disk as a sibling sub-solution of the baseline via
Postprocess.

Configured via the "plugins.mga" block in config.json; unknown keys are
rejected. Full config reference: the plugin's docs page in
docs/files/available_plugins/mga.

pyoNearOpt compatibility is documented in oracle_driver.py.
"""

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
    NODE_CAPEX,
    NODE_CAPEX_CUMULATIVE,
    NODE_CAPEX_TECH,
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
    },
    "plugins.mga.axes.node_capex_cumulative": {"nodes", "until_years"},
    "plugins.mga.axes.node_capex_by_technology": {"nodes", "technology_groups"},
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
    if normalisation not in ("relative", "minmax", "units"):
        raise ValueError(
            f"Unknown MGA normalisation: {normalisation!r}. Expected "
            f"'relative', 'minmax' or 'units'."
        )
    if cfg.get("mode") == "oracle" and normalisation in ("units", "minmax"):
        raise ValueError(
            f"MGA: normalisation={normalisation!r} is not supported in "
            "oracle mode -- oracle's max-min MILP relies on big_M/t_max "
            "dominating axis magnitudes and a cut-validity guard sized for "
            "O(1) normalised coordinates anchored at offset=0, both of "
            "which assume 'relative' normalisation. Use sampling or bbo "
            "mode for 'units'/'minmax'."
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
        normalisation="relative",
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
            normalisation: "relative" (default) scales design axes by their
                near-optimal maximum; "minmax" maps each axis's own
                near-optimal [min, max] onto [0, 1]; "units" reports them in
                raw physical units. The cost axis is always budget-relative.
                See solve_axis_bounds().
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
        all_nodes = list(optimization_setup.energy_system.set_nodes)
        (
            tech_groups,
            carrier_groups,
            node_capex_groups,
            node_capex_cumulative_axes,
            node_capex_tech_axes,
            node_capex_cumulative_chains,
        ) = build_axis_groups(
            technologies,
            carrier_imports,
            all_technologies,
            all_carriers,
            node_capex=node_capex,
            node_capex_cumulative=node_capex_cumulative,
            node_capex_by_technology=node_capex_by_technology,
            all_nodes=all_nodes,
        )
        # Chains of cumulative-capex axis names (same node/lump group,
        # ascending until_year), used by build_initial_outer_approximation
        # to add monotonicity rows; empty when there are no such axes.
        self._monotone_capex_chains = node_capex_cumulative_chains

        # The single source of truth for axis order everywhere downstream
        # (coordinates, cut normals, polytope columns): technology axes,
        # carrier axes, node-capex axes, node-capex-cumulative axes,
        # node-capex-technology axes, then the cost axis.
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

        # Model handle and period->year-index lookup needed by node-capex
        # axes. cost_capex_yearly is indexed by set_location, a per-
        # technology-family dimension whose coordinate values are node names
        # for conversion/storage/retrofitting technologies and edge names for
        # transport technologies; .sel(set_location=<node names>) therefore
        # naturally excludes transport-technology capex (invalid entries),
        # the same mechanism the tech-capacity axis already relies on.
        _node_kinds = (NODE_CAPEX, NODE_CAPEX_CUMULATIVE, NODE_CAPEX_TECH)
        self._axis_year_indices: dict[str, list] = {}
        if any(axis.kind in _node_kinds for axis in self.axes):
            self.cost_capex_yearly = self.model.variables["cost_capex_yearly"]
            year_indices = list(
                self.cost_capex_yearly.coords["set_time_steps_yearly"].values
            )
            real_years = list(optimization_setup.energy_system.set_time_steps_years)
            for axis in self.axes:
                if axis.kind != NODE_CAPEX_CUMULATIVE:
                    continue
                ids = _year_indices_in_period(axis.period, year_indices, real_years)
                if not ids:
                    until_year = axis.period[1]
                    raise ValueError(
                        f"MGA: axis {axis.name!r} until_year {until_year} "
                        f"covers no model year (model years: "
                        f"{real_years[0]}-{real_years[-1]})."
                    )
                self._axis_year_indices[axis.name] = ids
        else:
            self.cost_capex_yearly = None

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

    def _design_axis_terms(self, axis: Axis, capacity, flow, capex=None):
        """One design axis over `capacity`/`flow`/`capex` data.

        The arguments are either the linopy variables (yielding the axis
        LinearExpression) or their `.solution` arrays (yielding the axis
        value): technology axes sum the capacity addition of the member
        technologies, restricted by the capacity-type mask; carrier axes sum
        the duration-weighted annual import of the member carriers,
        sum_{m,n,t} tau_t * flow[m, n, t]; node-capex axes sum the annualised
        capex of the member nodes over all capacity types, over all model
        years (NODE_CAPEX) or every model year up to the axis's until_year
        (NODE_CAPEX_CUMULATIVE), and over either all technologies or just
        the axis's technology group (NODE_CAPEX_TECH).
        """
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            return (
                (self._capacity_mask * capacity)
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        if axis.kind == CARRIER_IMPORT:
            return (self._ts_duration * flow.sel(set_carriers=members)).sum(
                self._CARRIER_AGG + ["set_carriers"]
            )
        # NODE_CAPEX / NODE_CAPEX_CUMULATIVE / NODE_CAPEX_TECH: transport
        # technologies have no valid entry at a node-named location, so they
        # drop out of this selection for free (same mechanism as
        # _TECH_AGG's unrestricted set_location sum for tech-capacity axes).
        term = capex.sel(set_location=members)
        if axis.technologies is not None:
            term = term.sel(set_technologies=list(axis.technologies))
        year_ids = self._axis_year_indices.get(axis.name)
        if year_ids is not None:
            term = term.sel(set_time_steps_yearly=year_ids)
        return term.sum(self._NODE_AGG_CAPEX + ["set_location"])

    def axis_expression(self, axis: Axis):
        """Linopy expression of one axis (bound LP objective, projection)."""
        if axis.kind == TOTAL_COST:
            return self._total_cost_expression()
        return self._design_axis_terms(
            axis, self.capacity_addition, self.flow_import, capex=self.cost_capex_yearly
        )

    def axis_value(self, axis: Axis) -> float:
        """Value of one axis on the currently loaded solution."""
        if axis.kind == TOTAL_COST:
            return float(self.model.variables[COST_VARIABLE].solution.sum())
        flow = None if self.flow_import is None else self.flow_import.solution
        capex = (
            None if self.cost_capex_yearly is None else self.cost_capex_yearly.solution
        )
        return float(
            self._design_axis_terms(
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
        members, capacity type, period, technologies and physical unit, plus
        the normalisation convention (schema owned by polytope_io)."""
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
                    "unit": axis_physical_unit(axis, units, ureg),
                }
                for axis in self.axes
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
        for axis in self.axes:
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
        normalisation=config.get("normalisation", "relative"),
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
