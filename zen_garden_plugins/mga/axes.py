"""Exploration axes of the MGA plugin (no optimization-model access).

An axis is one coordinate of the explored near-optimal space. This module
owns the ``Axis`` type, the axis-kind groupings, the parsing and validation
of the ``axes`` config block into an ordered list of axes, and the
physical-unit lookup for the polytope metadata. The model-coupled maths
(expressions and values) lives in ``axis_evaluation.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config_schema import AxesConfig
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

# The model variable behind the total-cost axis.
COST_VARIABLE = "net_present_cost"

# Kinds restricted to a calendar-year window (cumulative or snapshot).
PERIOD_KINDS = frozenset(
    {
        NODE_CAPEX_CUMULATIVE,
        NODE_CAPACITY_RATIO,
        NODE_OUTPUT_RATIO,
        NODE_LIMIT_UTILISATION,
        NODE_CARBON_EMISSIONS_CUMULATIVE,
    }
)
# Dimensionless kinds, divided by a constant denominator.
RATIO_KINDS = frozenset(
    {NODE_CAPACITY_RATIO, NODE_OUTPUT_RATIO, NODE_LIMIT_UTILISATION}
)
# Kinds whose value can be negative (negative-emission technologies); they
# are neither clipped at 0 nor constrained to be monotone (docs: "Monotonicity").
MAY_BE_NEGATIVE_KINDS = frozenset({NODE_CARBON_EMISSIONS_CUMULATIVE})


@dataclass(frozen=True)
class Axis:
    """One exploration axis of the MGA polytope (kinds: see the docs, "Axes").

    Attributes:
        name: Axis name, unique across all axes.
        kind: One of the axis-kind constants of ``polytope_io``.
        members: Technologies (tech axes), carriers (import axes) or nodes.
        capacity_type: "+"-joined capacity type(s) for tech and capacity-ratio
            axes, None otherwise.
        period: (None, until_year) for cumulative kinds, (year, year) for
            ratio kinds, None otherwise.
        technologies: Numerator group of ratio axes, group of limit axes.
        denominator_technologies: Denominator technology group of ratio axes.
    """

    name: str
    kind: str
    members: tuple[str, ...]
    capacity_type: str | None
    period: tuple[int | None, int] | None = None
    technologies: tuple[str, ...] | None = None
    denominator_technologies: tuple[str, ...] | None = None


# Resolves the capacity type of a technology group: (axis label, members) -> type.
CapacityTypeOf = Callable[[str, list], str]


def build_axes(
    axes_cfg: AxesConfig,
    all_technologies,
    all_carriers,
    all_nodes,
    capacity_type_of: CapacityTypeOf,
) -> tuple[list[Axis], list[list[str]]]:
    """Turn the ``axes`` config block into the ordered list of axes.

    The order is the single source of truth for every column downstream:
    technology, carrier-import, cumulative-capex, capacity-ratio,
    output-ratio, limit-utilisation and cumulative-emissions axes (each in
    config order), then the cost axis.

    Args:
        axes_cfg: The validated ``plugins.mga.axes`` block.
        all_technologies: Technology names of the model.
        all_carriers: Carrier names of the model.
        all_nodes: Node names of the model.
        capacity_type_of: Resolver of a technology group's capacity type.

    Returns:
        (axes, monotone_chains): the axes, and per node group the
        cumulative-capex axis names in ascending until_year order.

    Raises:
        ValueError: For unknown members, malformed entries or name clashes.
    """
    tech_set, carrier_set, node_set = (
        set(all_technologies),
        set(all_carriers),
        set(all_nodes),
    )
    # Axis names share one namespace with the model names in the polytope
    # file, so group names must not reuse a technology, carrier or node name.
    all_names = tech_set | carrier_set | node_set
    tech_groups = _parse_axis_list(
        axes_cfg.technologies, tech_set, tech_set, "axes.technologies"
    )
    carrier_groups = _parse_axis_list(
        axes_cfg.carrier_imports, carrier_set, all_names, "axes.carrier_imports"
    )
    capex_axes, capex_chains = _parse_cumulative_block(
        axes_cfg.node_capex_cumulative,
        node_set,
        all_names,
        "axes.node_capex_cumulative",
    )
    capacity_ratio_axes = _parse_ratio_block(
        axes_cfg.node_capacity_ratio,
        tech_set,
        node_set,
        all_names,
        "axes.node_capacity_ratio",
    )
    output_ratio_axes = _parse_ratio_block(
        axes_cfg.node_output_ratio,
        tech_set,
        node_set,
        all_names,
        "axes.node_output_ratio",
    )
    limit_axes = _parse_limit_block(
        axes_cfg.node_limit_utilisation, tech_set, node_set, all_names
    )
    emissions_axes, _ = _parse_cumulative_block(
        axes_cfg.node_carbon_emissions_cumulative,
        node_set,
        all_names,
        "axes.node_carbon_emissions_cumulative",
    )
    generated_names = [
        name
        for group in (
            tech_groups,
            carrier_groups,
            capex_axes,
            capacity_ratio_axes,
            output_ratio_axes,
            limit_axes,
            emissions_axes,
        )
        for name, *_ in group
    ]
    duplicates = {n for n in generated_names if generated_names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"MGA: axis name(s) {sorted(duplicates)} used in more than one "
            f"axes.* block."
        )

    axes = [
        Axis(name, TECH_CAPACITY, tuple(members), capacity_type_of(name, members))
        for name, members in tech_groups
    ]
    axes += [
        Axis(name, CARRIER_IMPORT, tuple(members), None)
        for name, members in carrier_groups
    ]
    axes += [
        Axis(name, NODE_CAPEX_CUMULATIVE, tuple(members), None, (None, until_year))
        for name, members, until_year in capex_axes
    ]
    axes += [
        Axis(
            name,
            NODE_CAPACITY_RATIO,
            tuple(nodes),
            _ratio_capacity_type(capacity_type_of, name, numerator, denominator),
            period=(year, year),
            technologies=tuple(numerator),
            denominator_technologies=tuple(denominator),
        )
        for name, nodes, numerator, denominator, year in capacity_ratio_axes
    ]
    axes += [
        Axis(
            name,
            NODE_OUTPUT_RATIO,
            tuple(nodes),
            None,
            period=(year, year),
            technologies=tuple(numerator),
            denominator_technologies=tuple(denominator),
        )
        for name, nodes, numerator, denominator, year in output_ratio_axes
    ]
    axes += [
        Axis(
            name, NODE_LIMIT_UTILISATION, tuple(nodes), None, (year, year), tuple(techs)
        )
        for name, nodes, techs, year in limit_axes
    ]
    axes += [
        Axis(
            name,
            NODE_CARBON_EMISSIONS_CUMULATIVE,
            tuple(members),
            None,
            (None, until_year),
        )
        for name, members, until_year in emissions_axes
    ]
    if axes_cfg.include_cost:
        axes.append(Axis(COST_VARIABLE, TOTAL_COST, (), None))
    return axes, capex_chains


def _ratio_capacity_type(capacity_type_of, name, numerator, denominator) -> str:
    """Capacity type shared by a capacity-ratio axis's two technology groups."""
    numerator_type = capacity_type_of(f"{name} (numerator)", numerator)
    denominator_type = capacity_type_of(f"{name} (denominator)", denominator)
    if numerator_type != denominator_type:
        raise ValueError(
            f"MGA node_capacity_ratio axis {name!r}: numerator capacity "
            f"type {numerator_type!r} != denominator capacity type "
            f"{denominator_type!r}."
        )
    return numerator_type


def _parse_cumulative_block(block, node_set, reserved_names, label):
    """Parse a {nodes, until_years} block into axes and their chains.

    Returns:
        (axes, chains): (name, [nodes], until_year) tuples named
        f"{node_name}_until_{until_year}", nodes-major/until-years-minor in
        config order; and per node group its axis names by ascending year.
    """
    nodes = block.nodes if block is not None else None
    until_years = block.until_years if block is not None else None
    node_groups = _parse_axis_list(nodes, node_set, reserved_names, f"{label}.nodes")
    if node_groups and not until_years:
        raise ValueError(f"MGA {label}: 'nodes' given without 'until_years'.")
    if until_years and not node_groups:
        raise ValueError(f"MGA {label}: 'until_years' given without 'nodes'.")
    years = _parse_year_list(until_years, f"{label}.until_years") if node_groups else []
    axes = [
        (f"{name}_until_{year}", members, year)
        for name, members in node_groups
        for year in years
    ]
    chains = [
        [f"{name}_until_{year}" for year in sorted(years)] for name, _ in node_groups
    ]
    return axes, chains


def _parse_ratio_block(block, tech_set, node_set, reserved_names, label):
    """Parse a {nodes, years, ratio_groups} block (all three or none).

    Returns:
        (name, [nodes], [numerator], [denominator], year) tuples named
        f"{node_name}_{ratio_name}_{year}", nodes-major/ratio-groups-middle/
        years-minor in config order.
    """
    nodes = block.nodes if block is not None else None
    years = block.years if block is not None else None
    ratio_groups = block.ratio_groups if block is not None else None
    node_groups = _parse_axis_list(nodes, node_set, reserved_names, f"{label}.nodes")
    have_nodes, have_years, have_groups = (
        bool(node_groups),
        bool(years),
        bool(ratio_groups),
    )
    if have_nodes and not (have_years and have_groups):
        raise ValueError(
            f"MGA {label}: 'nodes' given without 'years' and/or 'ratio_groups'."
        )
    if have_years and not (have_nodes and have_groups):
        raise ValueError(
            f"MGA {label}: 'years' given without 'nodes' and/or 'ratio_groups'."
        )
    if have_groups and not (have_nodes and have_years):
        raise ValueError(
            f"MGA {label}: 'ratio_groups' given without 'nodes' and/or 'years'."
        )
    if not have_nodes:
        return []
    parsed_years = _parse_year_list(years, f"{label}.years")
    groups = _parse_ratio_groups(
        ratio_groups, tech_set, reserved_names, f"{label}.ratio_groups"
    )
    return [
        (f"{node_name}_{ratio_name}_{year}", node_members, numerator, denominator, year)
        for node_name, node_members in node_groups
        for ratio_name, numerator, denominator in groups
        for year in parsed_years
    ]


def _parse_limit_block(block, tech_set, node_set, reserved_names):
    """(f"{nodes}_{group}_{year}", [nodes], [techs], year) tuples, in config order."""
    label = "axes.node_limit_utilisation"
    if block is None:
        return []
    nodes, years, groups = block.nodes, block.years, block.technology_groups
    if not (nodes or years or groups):
        return []
    if not (nodes and years and groups):
        raise ValueError(
            f"MGA {label}: give 'nodes', 'years' and 'technology_groups' together."
        )
    node_groups = _parse_axis_list(nodes, node_set, reserved_names, f"{label}.nodes")
    years = _parse_year_list(years, f"{label}.years")
    groups = _parse_axis_list(groups, tech_set, reserved_names, f"{label}.groups")
    return [
        (f"{node}_{group}_{year}", node_members, techs, year)
        for node, node_members in node_groups
        for group, techs in groups
        for year in years
    ]


def _parse_year_list(years, label):
    """Validate a non-empty list of distinct int years; returns it in order."""
    if not years:
        raise ValueError(f"MGA {label}: must be a non-empty list of years.")
    duplicates = {y for y in years if years.count(y) > 1}
    if duplicates:
        raise ValueError(f"MGA {label}: duplicate year(s) {sorted(duplicates)}.")
    return list(years)


def _parse_axis_list(entries, valid_members, reserved_names, label):
    """Parse one axis config list into ordered (name, members) tuples.

    Each entry is an axis name (singleton axis) or a single-key dict
    ``{group_name: [member, ...]}`` (lumped axis). Axis names must be unique,
    group names must not shadow a model name, and each member may appear in
    at most one axis (it would otherwise be counted twice).
    """
    groups = []
    seen_names = set()
    axis_of_member = {}
    unknown = []
    for entry in entries or []:
        if isinstance(entry, str):
            name, members = entry, [entry]
        elif len(entry) == 1:
            name, members = next(iter(entry.items()))
            if name in reserved_names:
                raise ValueError(
                    f"MGA {label}: group name {name!r} shadows an "
                    f"existing technology or carrier name."
                )
        else:
            raise ValueError(
                f"MGA {label}: invalid entry {entry!r}, expected "
                f"a name or a single {{group: [members]}} dict."
            )
        if not (name and members and all(members)):
            raise ValueError(f"MGA {label}: invalid entry {entry!r}.")
        if name in seen_names:
            raise ValueError(f"MGA {label}: duplicate axis name {name!r}.")
        seen_names.add(name)
        for member in members:
            if member not in valid_members:
                unknown.append(member)
            elif member in axis_of_member:
                raise ValueError(
                    f"MGA {label}: {member!r} appears in both axis "
                    f"{axis_of_member[member]!r} and {name!r}."
                )
            else:
                axis_of_member[member] = name
        groups.append((name, list(members)))
    if unknown:
        raise ValueError(f"MGA {label}: unknown names {sorted(set(unknown))}")
    return groups


def _parse_ratio_groups(entries, valid_members, reserved_names, label):
    """Parse a ratio_groups list into (name, [numerator], [denominator]).

    Membership is not exclusive: a numerator technology may (and usually
    does) also appear in its own denominator.
    """
    groups = []
    seen_names = set()
    unknown = []
    for entry in entries or []:
        if len(entry) != 1:
            raise ValueError(
                f"MGA {label}: invalid entry {entry!r}, expected a single "
                f"{{group: {{'numerator': [...], 'denominator': [...]}}}} dict."
            )
        name, spec = next(iter(entry.items()))
        if name in reserved_names:
            raise ValueError(
                f"MGA {label}: group name {name!r} shadows an existing "
                f"technology, carrier or node name."
            )
        numerator, denominator = spec.numerator, spec.denominator
        if not (name and numerator and denominator and all(numerator + denominator)):
            raise ValueError(
                f"MGA {label}: invalid entry {entry!r}, expected non-empty "
                f"numerator and denominator lists."
            )
        if name in seen_names:
            raise ValueError(f"MGA {label}: duplicate ratio-group name {name!r}.")
        seen_names.add(name)
        unknown += [
            m for m in set(numerator) | set(denominator) if m not in valid_members
        ]
        groups.append((name, list(numerator), list(denominator)))
    if unknown:
        raise ValueError(f"MGA {label}: unknown names {sorted(set(unknown))}")
    return groups


def axis_physical_unit(axis, units, ureg):
    """Physical unit string of one axis value, or None if unavailable.

    Heterogeneous lumps yield a " + "-joined string. ``units`` is the model's
    variable-unit mapping (empty when unit tracking is off), indexed by the
    variables' set names.
    """
    if axis.kind == TOTAL_COST:
        series = units.get(COST_VARIABLE)
        if series is None:
            return None
        found = sorted({str(u) for u in np.atleast_1d(np.asarray(series))})
        return " + ".join(found) if found else None

    if axis.kind in RATIO_KINDS:
        return "dimensionless"

    if axis.kind == TECH_CAPACITY:
        series = units.get("capacity_addition")
        if series is None:
            return None
        mask = series.index.get_level_values("set_technologies").isin(
            axis.members
        ) & series.index.get_level_values("set_capacity_types").isin(
            axis.capacity_type.split("+")
        )
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    if axis.kind == NODE_CAPEX_CUMULATIVE:
        series = units.get("cost_capex_yearly")
        if series is None:
            return None
        mask = series.index.get_level_values("set_location").isin(axis.members)
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    # Carrier imports and emissions are instantaneous rates summed into an
    # annual value, so their unit gains an hour.
    if axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE:
        series = units.get("carbon_emissions_technology")
        level = "set_location"
    else:
        series = units.get("flow_import")
        level = "set_carriers"
    if series is None:
        return None
    mask = series.index.get_level_values(level).isin(axis.members)
    return _annualised_unit(series, mask, ureg)


def _annualised_unit(series, mask, ureg):
    """Unit string of an instantaneous-rate series, multiplied by an hour."""
    annual = set()
    for unit in {str(u) for u in series[mask].to_numpy()}:
        try:
            annual.add(str(ureg(f"({unit}) * hour").units))
        except Exception:
            annual.add(f"({unit}) * hour")
    return " + ".join(sorted(annual)) if annual else None
