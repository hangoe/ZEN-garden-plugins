"""Exploration-axis definitions for the MGA plugin.

An axis is one coordinate of the explored near-optimal space: the summed
capacity addition of a group of technologies, the duration-weighted annual
import of a group of carriers, the summed annualised capex of a group of
nodes (optionally restricted to all years up to a target calendar year), or
the total system cost. This module owns everything about axes that does not
need the optimization model: the Axis type, the parsing and validation of
the axis config lists, and the physical-unit lookup for the polytope
metadata. Model-coupled axis logic lives in plugin.MGA.
"""

from dataclasses import dataclass

import numpy as np

from .polytope_io import (
    NODE_CAPACITY_RATIO,
    NODE_CAPEX,
    NODE_CAPEX_CUMULATIVE,
    NODE_CAPEX_CUMULATIVE_TECH,
    NODE_CAPEX_TECH,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
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
    model years; NODE_CAPEX_CUMULATIVE axes do the same but restricted to
    every model year up to and including `period[1]` (the axis's
    until_year); NODE_CAPEX_TECH axes do the same but restricted to the
    member technologies in `technologies`; NODE_CAPEX_CUMULATIVE_TECH axes
    combine both restrictions at once (`period` and `technologies` both
    set); every node-capex kind weights each sampled year's capex by the
    same discount/interval-expansion factor ZEN-garden's own
    net_present_cost uses (see plugin.py's _year_interval_expansion_factors),
    so it is on the same accounting basis as the single TOTAL_COST axis,
    which is the model's net present cost and has no members.
    NODE_CARBON_EMISSIONS_CUMULATIVE axes sum carbon_emissions_technology
    (annualised) over the member nodes, restricted to every model year up
    to and including `period[1]`, exactly like NODE_CAPEX_CUMULATIVE but
    interval-expanded only (no discounting -- emissions aren't discounted).
    NODE_CAPACITY_RATIO axes are numerator_capacity / denominator_capacity
    at one member node and one snapshot calendar year: the numerator is the
    live installed `capacity` of the `technologies` group at that node/year;
    the denominator is the `denominator_technologies` group's installed
    capacity at that same node/year, but FROZEN at its value on the
    cost-optimal baseline design z* (computed once, per axis, in
    plugin.py's MGA.__init__) rather than recomputed for each explored
    design -- both technology groups are decision-variable sums, so a live
    ratio of the two would be a genuine nonlinear term and could not serve
    as an LP objective/projection/support-function term the way every other
    axis does. `period` is reused here as `(year, year)` to select the one
    relevant `set_time_steps_yearly` index via the existing
    `_year_indices_in_period` helper, rather than adding a separate
    single-year field.
    capacity_type is the "+"-joined selected type(s) for tech
    axes and None otherwise; for NODE_CAPACITY_RATIO axes it is the type
    shared by both the numerator and denominator technology groups (the two
    are required to match). period is (None, until_year) for
    NODE_CAPEX_CUMULATIVE, NODE_CAPEX_CUMULATIVE_TECH and
    NODE_CARBON_EMISSIONS_CUMULATIVE axes -- the leading None means "no
    lower bound, from the first model year" -- (year, year) for
    NODE_CAPACITY_RATIO axes, and None otherwise. technologies is the
    selected technology group for NODE_CAPEX_TECH and
    NODE_CAPEX_CUMULATIVE_TECH axes, the numerator technology group for
    NODE_CAPACITY_RATIO axes, and None otherwise. denominator_technologies
    is the denominator technology group for NODE_CAPACITY_RATIO axes and
    None for every other kind.
    """

    name: str
    kind: str
    members: tuple[str, ...]
    capacity_type: str | None
    period: tuple[int | None, int] | None = None
    technologies: tuple[str, ...] | None = None
    denominator_technologies: tuple[str, ...] | None = None


def build_axis_groups(
    technologies,
    carrier_imports,
    all_technologies,
    all_carriers,
    node_capex=None,
    node_capex_cumulative=None,
    node_capex_by_technology=None,
    node_capex_cumulative_tech=None,
    node_capacity_ratio=None,
    node_carbon_emissions_cumulative=None,
    all_nodes=(),
):
    """Turn the axis config lists into ordered (name, members) groups.

    Returns (tech_groups, carrier_groups, node_capex_groups,
    node_capex_cumulative_axes, node_capex_tech_axes,
    node_capex_cumulative_chains, node_capex_cumulative_tech_axes,
    node_capex_cumulative_tech_chains, node_capacity_ratio_axes,
    node_carbon_emissions_cumulative_axes,
    node_carbon_emissions_cumulative_chains): the first three are (name,
    [members]) tuples in the user's order; node_capex_cumulative_axes is a
    list of (name, [members], until_year) tuples, one per (node/node-lump,
    until_year) combination named f"{name}_until_{until_year}", iterated
    nodes-major/until-years-minor (config order); node_capex_tech_axes is a
    list of (name, [node members], [technology members]) tuples, one per
    (node/node-lump, technology-group) combination named
    f"{name}_{tech_group_name}", iterated nodes-major/tech-groups-minor;
    node_capex_cumulative_chains is a list of [name, ...] lists, one per
    node/node-lump group, holding that group's generated axis names sorted
    ascending by until_year (independent of config order) -- the chains used
    to constrain cumulative-capex axes to be monotonically non-decreasing.
    node_capex_cumulative_tech_axes is a list of (name, [node members],
    [technology members], until_year) tuples, one per (node/node-lump,
    technology-group, until_year) combination named
    f"{node_name}_{tech_name}_until_{until_year}", iterated
    nodes-major/tech-groups-middle/until-years-minor (config order);
    node_capex_cumulative_tech_chains is a list of [name, ...] lists, one per
    (node/node-lump, technology-group) pair, holding that pair's generated
    axis names sorted ascending by until_year -- the chains used to
    constrain each (node, technology-group) pair's cumulative capex to be
    monotonically non-decreasing, without comparing across technology
    groups. node_capacity_ratio_axes is a list of (name, [node members],
    [numerator technology members], [denominator technology members], year)
    tuples, one per (node/node-lump, ratio-group, year) combination named
    f"{node_name}_{ratio_name}_{year}", iterated
    nodes-major/ratio-groups-middle/years-minor (config order); unlike every
    cumulative kind, this axis has no associated chains list -- a ratio is
    not expected to be monotone across years. node_carbon_emissions_cumulative_axes
    and node_carbon_emissions_cumulative_chains mirror
    node_capex_cumulative_axes/node_capex_cumulative_chains in shape and
    order only (same nodes-major/until-years-minor iteration), but the
    emissions chains are NOT used to build monotonicity rows in plugin.py's
    build_initial_outer_approximation(): carbon_emissions_technology is
    unbounded below in ZEN-garden, unlike cost_capex_yearly, so cumulative
    emissions are not guaranteed non-decreasing across until_year and the
    monotonicity reasoning that holds for capex does not carry over.
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

    node_capex_cumulative = node_capex_cumulative or {}
    cumulative_node_groups = _parse_axis_list(
        node_capex_cumulative.get("nodes"),
        node_set,
        all_names,
        "axes.node_capex_cumulative.nodes",
    )
    if cumulative_node_groups and not node_capex_cumulative.get("until_years"):
        raise ValueError(
            "MGA axes.node_capex_cumulative: 'nodes' given without 'until_years'."
        )
    if node_capex_cumulative.get("until_years") and not cumulative_node_groups:
        raise ValueError(
            "MGA axes.node_capex_cumulative: 'until_years' given without 'nodes'."
        )
    until_years = (
        _parse_until_years_list(
            node_capex_cumulative["until_years"],
            "axes.node_capex_cumulative.until_years",
        )
        if cumulative_node_groups
        else []
    )
    node_capex_cumulative_axes = [
        (f"{name}_until_{until_year}", members, until_year)
        for name, members in cumulative_node_groups
        for until_year in until_years
    ]
    node_capex_cumulative_chains = [
        [f"{name}_until_{until_year}" for until_year in sorted(until_years)]
        for name, _ in cumulative_node_groups
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

    node_capex_cumulative_tech = node_capex_cumulative_tech or {}
    cct_node_groups = _parse_axis_list(
        node_capex_cumulative_tech.get("nodes"),
        node_set,
        all_names,
        "axes.node_capex_cumulative_tech.nodes",
    )
    have_nodes = bool(cct_node_groups)
    have_until_years = bool(node_capex_cumulative_tech.get("until_years"))
    have_tech_groups = bool(node_capex_cumulative_tech.get("technology_groups"))
    if have_nodes and not (have_until_years and have_tech_groups):
        raise ValueError(
            "MGA axes.node_capex_cumulative_tech: 'nodes' given without "
            "'until_years' and/or 'technology_groups'."
        )
    if have_until_years and not (have_nodes and have_tech_groups):
        raise ValueError(
            "MGA axes.node_capex_cumulative_tech: 'until_years' given "
            "without 'nodes' and/or 'technology_groups'."
        )
    if have_tech_groups and not (have_nodes and have_until_years):
        raise ValueError(
            "MGA axes.node_capex_cumulative_tech: 'technology_groups' given "
            "without 'nodes' and/or 'until_years'."
        )
    cct_until_years = (
        _parse_until_years_list(
            node_capex_cumulative_tech["until_years"],
            "axes.node_capex_cumulative_tech.until_years",
        )
        if have_nodes
        else []
    )
    cct_tech_groups = (
        _parse_axis_list(
            node_capex_cumulative_tech["technology_groups"],
            tech_set,
            all_names,
            "axes.node_capex_cumulative_tech.technology_groups",
        )
        if have_nodes
        else []
    )
    node_capex_cumulative_tech_axes = [
        (f"{node_name}_{tech_name}_until_{until_year}", node_members, tech_members, until_year)
        for node_name, node_members in cct_node_groups
        for tech_name, tech_members in cct_tech_groups
        for until_year in cct_until_years
    ]
    node_capex_cumulative_tech_chains = [
        [
            f"{node_name}_{tech_name}_until_{until_year}"
            for until_year in sorted(cct_until_years)
        ]
        for node_name, _ in cct_node_groups
        for tech_name, _ in cct_tech_groups
    ]

    node_capacity_ratio = node_capacity_ratio or {}
    ratio_node_groups = _parse_axis_list(
        node_capacity_ratio.get("nodes"),
        node_set,
        all_names,
        "axes.node_capacity_ratio.nodes",
    )
    have_ratio_nodes = bool(ratio_node_groups)
    have_ratio_years = bool(node_capacity_ratio.get("years"))
    have_ratio_groups = bool(node_capacity_ratio.get("ratio_groups"))
    if have_ratio_nodes and not (have_ratio_years and have_ratio_groups):
        raise ValueError(
            "MGA axes.node_capacity_ratio: 'nodes' given without 'years' "
            "and/or 'ratio_groups'."
        )
    if have_ratio_years and not (have_ratio_nodes and have_ratio_groups):
        raise ValueError(
            "MGA axes.node_capacity_ratio: 'years' given without 'nodes' "
            "and/or 'ratio_groups'."
        )
    if have_ratio_groups and not (have_ratio_nodes and have_ratio_years):
        raise ValueError(
            "MGA axes.node_capacity_ratio: 'ratio_groups' given without "
            "'nodes' and/or 'years'."
        )
    ratio_years = (
        _parse_until_years_list(
            node_capacity_ratio["years"], "axes.node_capacity_ratio.years"
        )
        if have_ratio_nodes
        else []
    )
    ratio_groups = (
        _parse_ratio_groups(
            node_capacity_ratio["ratio_groups"],
            tech_set,
            all_names,
            "axes.node_capacity_ratio.ratio_groups",
        )
        if have_ratio_nodes
        else []
    )
    node_capacity_ratio_axes = [
        (
            f"{node_name}_{ratio_name}_{year}",
            node_members,
            numerator_members,
            denominator_members,
            year,
        )
        for node_name, node_members in ratio_node_groups
        for ratio_name, numerator_members, denominator_members in ratio_groups
        for year in ratio_years
    ]

    node_carbon_emissions_cumulative = node_carbon_emissions_cumulative or {}
    emissions_node_groups = _parse_axis_list(
        node_carbon_emissions_cumulative.get("nodes"),
        node_set,
        all_names,
        "axes.node_carbon_emissions_cumulative.nodes",
    )
    if emissions_node_groups and not node_carbon_emissions_cumulative.get(
        "until_years"
    ):
        raise ValueError(
            "MGA axes.node_carbon_emissions_cumulative: 'nodes' given "
            "without 'until_years'."
        )
    if (
        node_carbon_emissions_cumulative.get("until_years")
        and not emissions_node_groups
    ):
        raise ValueError(
            "MGA axes.node_carbon_emissions_cumulative: 'until_years' given "
            "without 'nodes'."
        )
    emissions_until_years = (
        _parse_until_years_list(
            node_carbon_emissions_cumulative["until_years"],
            "axes.node_carbon_emissions_cumulative.until_years",
        )
        if emissions_node_groups
        else []
    )
    node_carbon_emissions_cumulative_axes = [
        (f"{name}_until_{until_year}", members, until_year)
        for name, members in emissions_node_groups
        for until_year in emissions_until_years
    ]
    node_carbon_emissions_cumulative_chains = [
        [f"{name}_until_{until_year}" for until_year in sorted(emissions_until_years)]
        for name, _ in emissions_node_groups
    ]

    generated_names = (
        [n for n, _ in tech_groups]
        + [n for n, _ in carrier_groups]
        + [n for n, _ in node_capex_groups]
        + [n for n, _, _ in node_capex_cumulative_axes]
        + [n for n, _, _ in node_capex_tech_axes]
        + [n for n, _, _, _ in node_capex_cumulative_tech_axes]
        + [n for n, _, _, _, _ in node_capacity_ratio_axes]
        + [n for n, _, _ in node_carbon_emissions_cumulative_axes]
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
        node_capex_cumulative_axes,
        node_capex_tech_axes,
        node_capex_cumulative_chains,
        node_capex_cumulative_tech_axes,
        node_capex_cumulative_tech_chains,
        node_capacity_ratio_axes,
        node_carbon_emissions_cumulative_axes,
        node_carbon_emissions_cumulative_chains,
    )


def _parse_until_years_list(until_years, label):
    """Parse and validate a list of cumulative-capex target years.

    Rejects a missing/empty list, non-int (or bool) entries, and duplicates.
    Unlike periods, cumulative windows are meant to overlap (each nests
    inside the next), so there is no overlap check. Returns the list of int
    years, in the given order.
    """
    if not until_years:
        raise ValueError(f"MGA {label}: must be a non-empty list of years.")
    parsed = []
    for year in until_years:
        if not isinstance(year, int) or isinstance(year, bool):
            raise ValueError(f"MGA {label}: invalid year {year!r}, expected an int.")
        parsed.append(year)
    duplicates = {y for y in parsed if parsed.count(y) > 1}
    if duplicates:
        raise ValueError(f"MGA {label}: duplicate year(s) {sorted(duplicates)}.")
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


def _parse_ratio_groups(entries, valid_members, reserved_names, label):
    """Parse one node_capacity_ratio.ratio_groups config list.

    Each entry is a single-key dict ``{group_name: {"numerator": [...],
    "denominator": [...]}}``. Unlike `_parse_axis_list`, membership is NOT
    exclusive across entries or between a group's own numerator and
    denominator: a numerator technology appearing in its own denominator
    (e.g. "BEV" in both) is the expected, normal case for a capacity-share
    axis, not a modelling error. Returns a list of (name, [numerator
    members], [denominator members]) tuples, in the given order.
    """
    groups = []
    seen_names = set()
    unknown = []
    for entry in entries or []:
        if not (isinstance(entry, dict) and len(entry) == 1):
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
        well_formed_spec = isinstance(spec, dict) and set(spec) == {
            "numerator",
            "denominator",
        }
        if well_formed_spec:
            numerator, denominator = spec["numerator"], spec["denominator"]
            well_formed_spec = (
                isinstance(numerator, list)
                and numerator
                and all(isinstance(m, str) and m for m in numerator)
                and isinstance(denominator, list)
                and denominator
                and all(isinstance(m, str) and m for m in denominator)
            )
        if not (isinstance(name, str) and name and well_formed_spec):
            raise ValueError(
                f"MGA {label}: invalid entry {entry!r}, expected a single "
                f"{{group: {{'numerator': [...], 'denominator': [...]}}}} dict "
                f"with non-empty member lists."
            )
        if name in seen_names:
            raise ValueError(f"MGA {label}: duplicate ratio-group name {name!r}.")
        seen_names.add(name)
        for member in set(numerator) | set(denominator):
            if member not in valid_members:
                unknown.append(member)
        groups.append((name, list(numerator), list(denominator)))
    if unknown:
        raise ValueError(f"MGA {label}: unknown names {sorted(set(unknown))}")
    return groups


def axis_physical_unit(axis, units, ureg):
    """Physical unit string of one axis value, or None if unavailable.

    Tech axes read the capacity_addition unit at the selected capacity type;
    carrier axes annualise the instantaneous flow_import unit (x hour); node
    capex axes (cumulative- or technology-restricted or not) read
    cost_capex_yearly's unit at the member nodes, additionally masked to the
    member technologies when set; node carbon-emissions-cumulative axes
    annualise the instantaneous carbon_emissions_technology unit (x hour),
    same mechanism as carrier axes; node capacity-ratio axes are a ratio of
    two capacity sums and so report "dimensionless"; the cost axis reads
    COST_VARIABLE's unit. Heterogeneous lumps yield a ' + '-joined string.
    `units` is the model's variable-unit mapping, which is empty when unit
    tracking is switched off.

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

    if axis.kind == NODE_CAPACITY_RATIO:
        return "dimensionless"

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

    if axis.kind in (
        NODE_CAPEX,
        NODE_CAPEX_CUMULATIVE,
        NODE_CAPEX_TECH,
        NODE_CAPEX_CUMULATIVE_TECH,
    ):
        series = units.get("cost_capex_yearly")
        if series is None:
            return None
        mask = series.index.get_level_values("location").isin(axis.members)
        if axis.technologies is not None:
            mask &= series.index.get_level_values("technology").isin(axis.technologies)
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    if axis.kind == NODE_CARBON_EMISSIONS_CUMULATIVE:
        series = units.get("carbon_emissions_technology")
        if series is None:
            return None
        mask = series.index.get_level_values("location").isin(axis.members)
        return _annualised_unit(series, mask, ureg)

    # CARRIER_IMPORT: flow_import is an instantaneous rate, while the axis is
    # the duration-weighted annual import, so the unit gains an hour.
    series = units.get("flow_import")
    if series is None:
        return None
    mask = series.index.get_level_values("carrier").isin(axis.members)
    return _annualised_unit(series, mask, ureg)


def _annualised_unit(series, mask, ureg):
    """Duration-weighted-annual unit string of an instantaneous-rate series.

    Shared by CARRIER_IMPORT (flow_import) and
    NODE_CARBON_EMISSIONS_CUMULATIVE (carbon_emissions_technology): both
    variables are instantaneous rates, while their axis is a
    duration-weighted annual sum, so the unit gains an hour.
    """
    annual = set()
    for unit in {str(u) for u in series[mask].to_numpy()}:
        try:
            annual.add(str(ureg(f"({unit}) * hour").units))
        except Exception:
            annual.add(f"({unit}) * hour")
    return " + ".join(sorted(annual)) if annual else None
