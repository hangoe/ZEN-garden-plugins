"""Model-coupled axis maths: linopy expressions and solution values.

``AxisEvaluator`` turns each ``Axis`` into a linopy expression (objective,
projection) or a float on the currently loaded solution. Both go through one
term function per axis kind, applied either to the model variables or to
their ``.solution`` arrays. Accounting conventions (discounting, interval
expansion, frozen ratio denominators) are described in the docs, section
"Axes".
"""

import logging

import numpy as np
import xarray as xr
from zen_garden.model.component_types.constraint import GenericConstraint

from .axes import COST_VARIABLE, Axis
from .polytope_io import (
    CARRIER_IMPORT,
    NODE_CAPACITY_RATIO,
    NODE_CAPEX_CUMULATIVE,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    NODE_LIMIT_UTILISATION,
    NODE_OUTPUT_RATIO,
    TECH_CAPACITY,
    TOTAL_COST,
)

logger = logging.getLogger(__name__)

# Dims summed away per axis kind.
_TECH_DIMS = ["set_capacity_types", "set_location", "set_years", "set_technologies"]
_CARRIER_DIMS = ["set_nodes", "set_time_steps_operation", "set_carriers"]
_CAPACITY_RATIO_DIMS = [
    "set_technologies",
    "set_capacity_types",
    "set_location",
    "set_years",
]
_OUTPUT_DIMS = [
    "set_conversion_technologies",
    "set_output_carriers",
    "set_nodes",
    "set_years",
]
_EMISSIONS_DIMS = ["set_technologies", "set_location", "set_years"]
_EMISSIONS_CARRIER_DIMS = ["set_carriers", "set_nodes", "set_years"]
_CAPEX_DIMS = ["set_technologies", "set_capacity_types", "set_years", "set_location"]


def capacity_type_mask(capacity_addition, power_type: str):
    """0/1 mask over (set_technologies, set_capacity_types).

    Storage technologies (more than one active capacity type) keep only their
    energy type, every other technology its single power type. Active entries
    are read from the live variable (labels != -1).
    """
    type_dim = "set_capacity_types"
    other_dims = [
        d for d in capacity_addition.dims if d not in ("set_technologies", type_dim)
    ]
    active = (capacity_addition.labels != -1).any(other_dims)
    known_types = [str(c) for c in capacity_addition.coords[type_dim].values]
    if power_type not in known_types:
        raise RuntimeError(
            f"MGA: power capacity type {power_type!r} not found in capacity_addition."
        )
    is_storage = active.sum(type_dim) > 1
    is_power = active[type_dim] == power_type
    keep = active & ~(is_storage & is_power)
    storage_techs = [
        str(t)
        for t in active["set_technologies"].values
        if bool(is_storage.sel(set_technologies=t))
    ]
    logger.info(
        f"MGA: storage tech(s) {storage_techs} use energy capacity, all "
        f"other technologies their power capacity."
    )
    return keep.astype(float)


def selected_capacity_type(mask, name: str, members: list) -> str:
    """The "+"-joined capacity type(s) the mask selects for a technology group.

    Raises:
        ValueError: If the members map to different types (e.g. storage and
            non-storage technologies lumped together).
    """
    selected = {
        member: tuple(
            str(c)
            for c in mask.coords["set_capacity_types"].values
            if float(mask.sel(set_technologies=member, set_capacity_types=c)) > 0.5
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


def year_indices_in_period(period, year_indices, real_years):
    """``set_years`` indices whose calendar year lies in ``period``.

    Args:
        period: (start, end), inclusive; start None means no lower bound.
        year_indices: The model's set_years.
        real_years: Calendar year of each entry of year_indices.
    """
    start, end = period
    return [
        year_id
        for year_id, real_year in zip(year_indices, real_years, strict=True)
        if (start is None or start <= real_year) and real_year <= end
    ]


def year_interval_expansion_factors(
    year_indices,
    discount_rate: float,
    interval_between_years: int,
    last_year_index: int,
) -> xr.DataArray:
    """Per-year discount and interval-expansion factor, indexed by set_years.

    factor[y] = sum_{i=0}^{n-1} (1/(1+r)) ** (interval*(y-y0) + i), with n = 1
    for the last model year and n = interval otherwise. This reproduces
    ZEN-garden's constraint_net_present_cost; with r = 0 it is the plain
    interval multiplicity of constraint_carbon_emissions_cumulative.
    """
    y0 = year_indices[0]
    factors = []
    for y in year_indices:
        n = 1 if y == last_year_index else interval_between_years
        factors.append(
            sum(
                (1.0 / (1.0 + discount_rate)) ** (interval_between_years * (y - y0) + i)
                for i in range(n)
            )
        )
    return xr.DataArray(factors, dims=["set_years"], coords={"set_years": year_indices})


def annual_capacity_limit(capacity_limit, existing_capacities, hours_per_year):
    """Annual flow allowed by max(limit, existing): existing capacity stays usable."""
    limit = capacity_limit.sel(set_capacity_types="power", drop=True)
    existing = existing_capacities.sel(set_capacity_types="power", drop=True)
    return limit.where(limit >= existing, existing) * hours_per_year


def _limit_reference_carriers(axes, sets) -> dict[str, str]:
    """The input reference carrier shared by each limit axis's technologies."""
    carriers = {}
    for axis in axes:
        references = {
            tech: sets["set_reference_carriers"][tech][0]
            for tech in axis.technologies
            if tech in sets["set_conversion_technologies"]
        }
        if (
            len(references) < len(axis.technologies)
            or len(set(references.values())) > 1
            or any(
                c not in sets["set_input_carriers"][t] for t, c in references.items()
            )
        ):
            raise ValueError(
                f"MGA axis {axis.name!r}: {axis.technologies} must be conversion "
                f"technologies sharing one input reference carrier."
            )
        carriers[axis.name] = next(iter(references.values()))
    return carriers


class AxisEvaluator:
    """Expressions and values of the MGA axes on one solved model.

    Construct it while the baseline solution is loaded: the frozen ratio
    denominators are read from it.
    """

    def __init__(self, optimization_step, axes: list[Axis], capacity_mask, all_nodes):
        """Prepare the model handles the configured axis kinds need.

        Args:
            optimization_step: The solved ZEN-garden OptimizationStep.
            axes: The axes, in column order.
            capacity_mask: See ``capacity_type_mask``.
            all_nodes: The model's node names.
        """
        om = optimization_step.optimization_model
        self.model = om.lp_model
        self.expressions = om.expressions
        self.capacity_mask = capacity_mask
        self.all_nodes = list(all_nodes)
        model_schema = optimization_step.model_schema
        parameters = om.parameters
        kinds = {axis.kind for axis in axes}
        variables = self.model.variables

        self.capacity_addition = variables["capacity_addition"]
        self.flow_import = None
        self.ts_duration = None
        if CARRIER_IMPORT in kinds:
            self.flow_import = variables["flow_import"]
            self.ts_duration = parameters.time_steps_operation_duration
            # Per-year durations, for import axes restricted to a year.
            if any(a.kind == CARRIER_IMPORT and a.period for a in axes):
                self.ts_duration_yearly = (
                    GenericConstraint.get_year_time_step_duration_array(
                        optimization_step
                    )
                )
        # Stock capacity, distinct from the capacity_addition of tech axes.
        self.capacity = variables["capacity"] if NODE_CAPACITY_RATIO in kinds else None
        # Indexed by set_location: selecting node names drops transport
        # technologies (located on edges) automatically.
        self.cost_capex_yearly = (
            variables["cost_capex_yearly"] if NODE_CAPEX_CUMULATIVE in kinds else None
        )
        # Instantaneous rates, annualised once like ZEN-garden does. Both
        # parts of carbon_emissions_annual (what the budget limits) are
        # needed: fuel CO2 is usually booked on the carriers, at import.
        self.carbon_emissions_yearly = None
        self.carbon_emissions_carrier_yearly = None
        if NODE_CARBON_EMISSIONS_CUMULATIVE in kinds:
            year_op_duration = GenericConstraint.get_year_time_step_duration_array(
                optimization_step
            )
            self.carbon_emissions_yearly = (
                variables["carbon_emissions_technology"] * year_op_duration
            ).sum("set_time_steps_operation")
            self.carbon_emissions_carrier_yearly = (
                variables["carbon_emissions_carrier"] * year_op_duration
            ).sum("set_time_steps_operation")
        self.flow_conversion_output_yearly = None
        if NODE_OUTPUT_RATIO in kinds:
            self._validate_output_ratio_axes(axes, om.variables.units)
            year_op_duration = GenericConstraint.get_year_time_step_duration_array(
                optimization_step
            )
            # Only the technologies an output-ratio axis uses.
            ratio_techs = sorted(
                {
                    tech
                    for axis in axes
                    if axis.kind == NODE_OUTPUT_RATIO
                    for tech in axis.technologies + axis.denominator_technologies
                }
            )
            self.flow_conversion_output_yearly = (
                variables["flow_conversion_output"].sel(
                    set_conversion_technologies=ratio_techs
                )
                * year_op_duration
            ).sum("set_time_steps_operation")
        self.flow_conversion_input_yearly = None
        self.annual_capacity_limit = None
        self.limit_reference_carrier: dict[str, str] = {}
        if NODE_LIMIT_UTILISATION in kinds:
            limit_axes = [a for a in axes if a.kind == NODE_LIMIT_UTILISATION]
            self.limit_reference_carrier = _limit_reference_carriers(
                limit_axes, om.sets
            )
            year_op_duration = GenericConstraint.get_year_time_step_duration_array(
                optimization_step
            )
            limit_techs = sorted({t for a in limit_axes for t in a.technologies})
            self.flow_conversion_input_yearly = (
                variables["flow_conversion_input"].sel(
                    set_conversion_technologies=limit_techs
                )
                * year_op_duration
            ).sum("set_time_steps_operation")
            self.annual_capacity_limit = annual_capacity_limit(
                parameters.capacity_limit.sel(set_technologies=limit_techs),
                parameters.existing_capacities.sel(set_technologies=limit_techs),
                year_op_duration.sum("set_time_steps_operation"),
            )

        self.axis_year_indices: dict[str, list] = {}
        year_indices = interval_between_years = last_year_index = None
        if any(axis.period is not None for axis in axes):
            year_indices = list(model_schema.set_years)
            real_years = list(model_schema.set_time_steps_years)
            last_year_index = int(model_schema.set_years_entire_horizon[-1])
            interval_between_years = int(
                optimization_step.config.system.interval_between_years
            )
            for axis in axes:
                if axis.period is None:
                    continue
                ids = year_indices_in_period(axis.period, year_indices, real_years)
                if not ids:
                    raise ValueError(
                        f"MGA: axis {axis.name!r} year(s) up to {axis.period[1]} "
                        f"cover no model year (model years: "
                        f"{real_years[0]}-{real_years[-1]})."
                    )
                self.axis_year_indices[axis.name] = ids
        # Capex is discounted and interval-expanded like net_present_cost;
        # emissions are interval-expanded only.
        self.capex_discount_factor = (
            year_interval_expansion_factors(
                year_indices,
                float(parameters.discount_rate),
                interval_between_years,
                last_year_index,
            )
            if self.cost_capex_yearly is not None
            else None
        )
        self.emissions_interval_factor = (
            year_interval_expansion_factors(
                year_indices, 0.0, interval_between_years, last_year_index
            )
            if self.carbon_emissions_yearly is not None
            else None
        )

        # Frozen at the baseline so the ratio stays linear (docs: "Axes").
        self.ratio_denominator: dict[str, float] = {
            axis.name: self._capacity_ratio_denominator(axis)
            for axis in axes
            if axis.kind == NODE_CAPACITY_RATIO
        }
        self.ratio_denominator.update(
            {
                axis.name: self._output_ratio_denominator(axis)
                for axis in axes
                if axis.kind == NODE_OUTPUT_RATIO
            }
        )
        self.ratio_denominator.update(
            {
                axis.name: self._limit_denominator(axis)
                for axis in axes
                if axis.kind == NODE_LIMIT_UTILISATION
            }
        )

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def expression(self, axis: Axis):
        """Linopy expression of one axis."""
        if axis.kind == TOTAL_COST:
            return self.expressions["total_cost"]
        return self._term(axis, self._sources(solution=False), self._years(axis))

    def value(self, axis: Axis) -> float:
        """Value of one axis on the currently loaded solution."""
        if axis.kind == TOTAL_COST:
            return float(self.model.variables[COST_VARIABLE].solution.sum())
        return float(self._term(axis, self._sources(solution=True), self._years(axis)))

    def share_reference_value(self, axis: Axis) -> float:
        """Baseline value of a capex axis widened to all nodes and all years."""
        widened = Axis(axis.name, axis.kind, tuple(self.all_nodes), None)
        return float(self._term(widened, self._sources(solution=True), None))

    # ------------------------------------------------------------------
    # term functions, one per kind
    # ------------------------------------------------------------------

    def _sources(self, solution: bool) -> dict:
        """The model variables, or their solution arrays, by role."""
        handles = {
            "capacity_addition": self.capacity_addition,
            "flow_import": self.flow_import,
            "capacity": self.capacity,
            "capex": self.cost_capex_yearly,
            "emissions": self.carbon_emissions_yearly,
            "emissions_carrier": self.carbon_emissions_carrier_yearly,
            "output": self.flow_conversion_output_yearly,
            "input": self.flow_conversion_input_yearly,
        }
        if not solution:
            return handles
        return {k: None if v is None else v.solution for k, v in handles.items()}

    def _years(self, axis: Axis):
        return self.axis_year_indices.get(axis.name)

    def _term(self, axis: Axis, src: dict, year_ids):
        return _TERMS[axis.kind](self, axis, src, year_ids)

    def _tech_capacity(self, axis, src, year_ids):
        return (
            (self.capacity_mask * src["capacity_addition"])
            .sel(set_technologies=list(axis.members))
            .sum(_TECH_DIMS)
        )

    def _carrier_import(self, axis, src, year_ids):
        if year_ids is not None:
            # Annual import in the axis year(s) only.
            return (
                self.ts_duration_yearly.sel(set_years=year_ids)
                * src["flow_import"].sel(set_carriers=list(axis.members))
            ).sum([*_CARRIER_DIMS, "set_years"])
        return (
            self.ts_duration * src["flow_import"].sel(set_carriers=list(axis.members))
        ).sum(_CARRIER_DIMS)

    def _capex_cumulative(self, axis, src, year_ids):
        term = src["capex"].sel(set_location=list(axis.members))
        if year_ids is not None:
            term = term.sel(set_years=year_ids)
        # xarray aligns the (restricted) set_years against the factor.
        term = term * self.capex_discount_factor
        return term.sum(_CAPEX_DIMS)

    def _capacity_ratio(self, axis, src, year_ids):
        numerator = (
            (self.capacity_mask * src["capacity"])
            .sel(
                set_technologies=list(axis.technologies),
                set_location=list(axis.members),
                set_years=year_ids,
            )
            .sum(_CAPACITY_RATIO_DIMS)
        )
        return numerator / self.ratio_denominator[axis.name]

    def _output_ratio(self, axis, src, year_ids):
        numerator = (
            src["output"]
            .sel(
                set_conversion_technologies=list(axis.technologies),
                set_nodes=list(axis.members),
                set_years=year_ids,
            )
            .sum(_OUTPUT_DIMS)
        )
        return numerator / self.ratio_denominator[axis.name]

    def _limit_utilisation(self, axis, src, year_ids):
        numerator = src["input"].sel(
            set_conversion_technologies=list(axis.technologies),
            set_input_carriers=[self.limit_reference_carrier[axis.name]],
            set_nodes=list(axis.members),
            set_years=year_ids,
        )
        return numerator.sum() / self.ratio_denominator[axis.name]

    def _emissions_cumulative(self, axis, src, year_ids):
        """Technology plus carrier emissions of the member nodes.

        Carrier emissions count at the node where the carrier is imported or
        exported; technology emissions on edges (transport) are dropped.
        """
        technology = (
            src["emissions"].sel(set_location=list(axis.members), set_years=year_ids)
            * self.emissions_interval_factor
        ).sum(_EMISSIONS_DIMS)
        carrier = (
            src["emissions_carrier"].sel(set_nodes=list(axis.members), set_years=year_ids)
            * self.emissions_interval_factor
        ).sum(_EMISSIONS_CARRIER_DIMS)
        return technology + carrier

    # ------------------------------------------------------------------
    # frozen ratio denominators
    # ------------------------------------------------------------------

    def _capacity_ratio_denominator(self, axis: Axis) -> float:
        """Baseline capacity of the denominator group at the axis's node/year."""
        value = float(
            (self.capacity_mask * self.capacity.solution)
            .sel(
                set_technologies=list(axis.denominator_technologies),
                set_location=list(axis.members),
                set_years=self.axis_year_indices[axis.name],
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

    def _output_ratio_denominator(self, axis: Axis) -> float:
        """Baseline annual output of the denominator group at the axis's node/year."""
        value = float(
            self.flow_conversion_output_yearly.solution.sel(
                set_conversion_technologies=list(axis.denominator_technologies),
                set_nodes=list(axis.members),
                set_years=self.axis_year_indices[axis.name],
            ).sum()
        )
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"MGA: node_output_ratio axis {axis.name!r} has a "
                f"non-positive baseline denominator {value:.6g} (no member "
                f"of {axis.denominator_technologies} produces output at "
                f"{axis.members}/{axis.period[1]} in the baseline design); "
                f"remove the axis or check its denominator technology group."
            )
        return value

    def _limit_denominator(self, axis: Axis) -> float:
        value = float(
            self.annual_capacity_limit.sel(
                set_technologies=list(axis.technologies),
                set_location=list(axis.members),
                set_years=self.axis_year_indices[axis.name],
            ).sum()
        )
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"MGA: axis {axis.name!r} needs a finite, positive capacity limit, "
                f"got {value:.6g}."
            )
        return value

    def _validate_output_ratio_axes(self, axes, variable_units):
        """Members must be conversion technologies sharing one output unit.

        The unit check is skipped when ZEN-garden's unit tracking is off.
        """
        if "flow_conversion_output" not in self.model.variables:
            raise ValueError(
                "MGA node_output_ratio: the model has no flow_conversion_output "
                "variable (no conversion technologies)."
            )
        conversion_techs = set(
            self.model.variables["flow_conversion_output"]
            .coords["set_conversion_technologies"]
            .values
        )
        units = variable_units.get("flow_conversion_output")
        if units is None:
            logger.warning(
                "MGA node_output_ratio: variable units unavailable "
                "(unit tracking off); output-unit consistency is not checked."
            )
        for axis in axes:
            if axis.kind != NODE_OUTPUT_RATIO:
                continue
            members = set(axis.technologies) | set(axis.denominator_technologies)
            not_conversion = sorted(members - conversion_techs)
            if not_conversion:
                raise ValueError(
                    f"MGA node_output_ratio axis {axis.name!r}: "
                    f"{not_conversion} are not conversion technologies; only "
                    f"conversion technologies have flow_conversion_output."
                )
            if units is None:
                continue
            mask = units.index.get_level_values("set_conversion_technologies").isin(
                members
            )
            found = sorted({str(u) for u in units[mask].to_numpy()})
            if len(found) > 1:
                raise ValueError(
                    f"MGA node_output_ratio axis {axis.name!r}: its technologies "
                    f"produce outputs in different units {found}; a ratio group "
                    f"must share one output unit -- split it into separate "
                    f"groups (e.g. passenger vs. freight transport)."
                )


_TERMS = {
    TECH_CAPACITY: AxisEvaluator._tech_capacity,
    CARRIER_IMPORT: AxisEvaluator._carrier_import,
    NODE_CAPEX_CUMULATIVE: AxisEvaluator._capex_cumulative,
    NODE_CAPACITY_RATIO: AxisEvaluator._capacity_ratio,
    NODE_OUTPUT_RATIO: AxisEvaluator._output_ratio,
    NODE_LIMIT_UTILISATION: AxisEvaluator._limit_utilisation,
    NODE_CARBON_EMISSIONS_CUMULATIVE: AxisEvaluator._emissions_cumulative,
}
