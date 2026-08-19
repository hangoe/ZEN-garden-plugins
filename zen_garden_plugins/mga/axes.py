"""Exploration-axis definitions for the MGA plugin.

An axis is one coordinate of the explored near-optimal space: the summed
capacity addition of a group of technologies, the duration-weighted annual
import of a group of carriers, the summed annualised capex of a group of
nodes (optionally restricted to an explicit calendar-year span), or the
total system cost. This module owns everything about axes that does not
need the optimization model: the Axis type, the parsing and validation of
the axis config lists, and the physical-unit lookup for the polytope
metadata. Model-coupled axis logic lives in plugin.MGA.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from .polytope_io import (
    NODE_CAPEX,
    NODE_CAPEX_PERIOD,
    NODE_CAPEX_TECH,
    TECH_CAPACITY,
    TOTAL_COST,
)

# The model variable behind the total-cost axis.
COST_VARIABLE = "net_present_cost"


@dataclass(frozen=True)
class Axis:
    """One exploration axis of the MGA polytope.

    TECH_CAPACITY axes sum capacity_addition over the member technologies,
    restricted to the selected capacity type; CARRIER_IMPORT axes sum the
    duration-weighted annual flow_import over the member carriers;
    NODE_CAPEX axes sum cost_capex_yearly over the member nodes and all
    model years; NODE_CAPEX_PERIOD axes do the same but restricted to the
    years falling in `period`; NODE_CAPEX_TECH axes do the same but
    restricted to the member technologies in `technologies`; the single
    TOTAL_COST axis is the model's net present cost and has no members.
    capacity_type is the "+"-joined selected type(s) for tech axes and None
    otherwise. period is (start_year, end_year) for NODE_CAPEX_PERIOD axes
    and None otherwise. technologies is the selected technology group for
    NODE_CAPEX_TECH axes and None otherwise.
    """

    name: str
    kind: str
    members: tuple[str, ...]
    capacity_type: str | None
    period: tuple[int, int] | None = None
    technologies: tuple[str, ...] | None = None


def build_axis_groups(
    technologies,
    carrier_imports,
    all_technologies,
    all_carriers,
    node_capex=None,
    node_capex_periods=None,
    node_capex_by_technology=None,
    all_nodes=(),
):
    """Turn the axis config lists into ordered (name, members) groups.

    Returns (tech_groups, carrier_groups, node_capex_groups,
    node_capex_period_axes, node_capex_tech_axes): the first three are
    (name, [members]) tuples in the user's order; node_capex_period_axes is
    a list of (name, [members], (start_year, end_year)) tuples, one per
    (node/node-lump, period) combination named f"{name}_{start}_{end}",
    iterated nodes-major/periods-minor; node_capex_tech_axes is a list of
    (name, [node members], [technology members]) tuples, one per
    (node/node-lump, technology-group) combination named
    f"{name}_{tech_group_name}", iterated nodes-major/tech-groups-minor.
    """
    tech_set, carrier_set, node_set = (
        set(all_technologies),
        set(all_carriers),
        set(all_nodes),
    )
    # Axis names share one namespace with the model names in the polytope
    # file, so every group's name must not reuse a technology, carrier or
    # node name.
    all_names = tech_set | carrier_set | node_set
    tech_groups = _parse_axis_list(
        technologies, tech_set, tech_set, "axes.technologies"
    )
    carrier_groups = _parse_axis_list(
        carrier_imports, carrier_set, all_names, "axes.carrier_imports"
    )
    node_capex_groups = _parse_axis_list(
        node_capex, node_set, all_names, "axes.node_capex"
    )

    node_capex_periods = node_capex_periods or {}
    period_node_groups = _parse_axis_list(
        node_capex_periods.get("nodes"),
        node_set,
        all_names,
        "axes.node_capex_periods.nodes",
    )
    if period_node_groups and not node_capex_periods.get("periods"):
        raise ValueError(
            "MGA axes.node_capex_periods: 'nodes' given without 'periods'."
        )
    if node_capex_periods.get("periods") and not period_node_groups:
        raise ValueError(
            "MGA axes.node_capex_periods: 'periods' given without 'nodes'."
        )
    periods = (
        _parse_period_list(
            node_capex_periods["periods"], "axes.node_capex_periods.periods"
        )
        if period_node_groups
        else []
    )
    node_capex_period_axes = [
        (f"{name}_{start}_{end}", members, (start, end))
        for name, members in period_node_groups
        for start, end in periods
    ]

    node_capex_by_technology = node_capex_by_technology or {}
    tech_filter_node_groups = _parse_axis_list(
        node_capex_by_technology.get("nodes"),
        node_set,
        all_names,
        "axes.node_capex_by_technology.nodes",
    )
    if tech_filter_node_groups and not node_capex_by_technology.get(
        "technology_groups"
    ):
        raise ValueError(
            "MGA axes.node_capex_by_technology: 'nodes' given without "
            "'technology_groups'."
        )
    if (
        node_capex_by_technology.get("technology_groups")
        and not tech_filter_node_groups
    ):
        raise ValueError(
            "MGA axes.node_capex_by_technology: 'technology_groups' given "
            "without 'nodes'."
        )
    technology_groups = (
        _parse_axis_list(
            node_capex_by_technology["technology_groups"],
            tech_set,
            all_names,
            "axes.node_capex_by_technology.technology_groups",
        )
        if tech_filter_node_groups
        else []
    )
    node_capex_tech_axes = [
        (f"{node_name}_{tech_name}", node_members, tech_members)
        for node_name, node_members in tech_filter_node_groups
        for tech_name, tech_members in technology_groups
    ]

    generated_names = (
        [n for n, _ in tech_groups]
        + [n for n, _ in carrier_groups]
        + [n for n, _ in node_capex_groups]
        + [n for n, _, _ in node_capex_period_axes]
        + [n for n, _, _ in node_capex_tech_axes]
    )
    duplicates = {n for n in generated_names if generated_names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"MGA: axis name(s) {sorted(duplicates)} used in more than one "
            f"axes.* block."
        )
    return (
        tech_groups,
        carrier_groups,
        node_capex_groups,
        node_capex_period_axes,
        node_capex_tech_axes,
    )


def _parse_period_list(periods, label):
    """Parse and validate a list of [start_year, end_year] pairs.

    Rejects a missing/empty list, malformed entries, start > end, and
    pairwise-overlapping periods (inclusive bounds). Returns a list of
    (start, end) int tuples, in the given order.
    """
    if not periods:
        raise ValueError(
            f"MGA {label}: must be a non-empty list of [start_year, end_year] pairs."
        )
    parsed = []
    for period in periods:
        well_formed = (
            isinstance(period, (list, tuple))
            and len(period) == 2
            and all(isinstance(y, int) and not isinstance(y, bool) for y in period)
        )
        if not well_formed:
            raise ValueError(
                f"MGA {label}: invalid period {period!r}, expected "
                f"[start_year, end_year]."
            )
        start, end = period
        if start > end:
            raise ValueError(f"MGA {label}: period [{start}, {end}] has start > end.")
        parsed.append((start, end))
    for (s1, e1), (s2, e2) in itertools.combinations(parsed, 2):
        if s1 <= e2 and s2 <= e1:
            raise ValueError(
                f"MGA {label}: periods [{s1}, {e1}] and [{s2}, {e2}] overlap."
            )
    return parsed


def _parse_axis_list(entries, valid_members, reserved_names, label):
    """Parse one axis config list into ordered (name, members) tuples.

    Each entry is an axis name (singleton axis) or a single-key dict
    ``{group_name: [member, ...]}`` (lumped axis). Axis names must be unique,
    group names must not shadow an existing model name, and each member may
    appear in at most one axis (it would otherwise be counted twice).
    """
    groups = []
    seen_names = set()
    axis_of_member = {}
    unknown = []
    for entry in entries or []:
        if isinstance(entry, str):
            name, members = entry, [entry]
        elif isinstance(entry, dict) and len(entry) == 1:
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
        well_formed = (
            isinstance(name, str)
            and name
            and isinstance(members, list)
            and members
            and all(isinstance(m, str) and m for m in members)
        )
        if not well_formed:
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


def axis_physical_unit(axis, units, ureg):
    """Physical unit string of one axis value, or None if unavailable.

    Tech axes read the capacity_addition unit at the selected capacity type;
    carrier axes annualise the instantaneous flow_import unit (x hour); node
    capex axes (period- or technology-restricted or not) read
    cost_capex_yearly's unit at the member nodes, additionally masked to the
    member technologies when set; the cost axis reads COST_VARIABLE's unit.
    Heterogeneous lumps yield a ' + '-joined string. `units` is the model's
    variable-unit mapping, which is empty when unit tracking is switched off.

    The unit series are indexed by ZEN-garden's documentation names for the
    dimensions ("technology", "capacity_type", "carrier", "location"), which
    differ from the set names the variables themselves are indexed by
    ("set_technologies" and so on).
    """
    if axis.kind == TOTAL_COST:
        series = units.get(COST_VARIABLE)
        if series is None:
            return None
        found = sorted({str(u) for u in np.atleast_1d(np.asarray(series))})
        return " + ".join(found) if found else None

    if axis.kind == TECH_CAPACITY:
        series = units.get("capacity_addition")
        if series is None:
            return None
        mask = series.index.get_level_values("technology").isin(
            axis.members
        ) & series.index.get_level_values("capacity_type").isin(
            axis.capacity_type.split("+")
        )
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    if axis.kind in (NODE_CAPEX, NODE_CAPEX_PERIOD, NODE_CAPEX_TECH):
        series = units.get("cost_capex_yearly")
        if series is None:
            return None
        mask = series.index.get_level_values("location").isin(axis.members)
        if axis.technologies is not None:
            mask &= series.index.get_level_values("technology").isin(axis.technologies)
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    # CARRIER_IMPORT: flow_import is an instantaneous rate, while the axis is
    # the duration-weighted annual import, so the unit gains an hour.
    series = units.get("flow_import")
    if series is None:
        return None
    mask = series.index.get_level_values("carrier").isin(axis.members)
    annual = set()
    for unit in {str(u) for u in series[mask].to_numpy()}:
        try:
            annual.add(str(ureg(f"({unit}) * hour").units))
        except Exception:
            annual.add(f"({unit}) * hour")
    return " + ".join(sorted(annual)) if annual else None
