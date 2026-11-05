"""Tests of axes.py: config parsing into ordered axes, and axis units."""

import pandas as pd
import pint
import pytest
from mga_helpers import PV_SHARE, axes_of, units_series

from zen_garden_plugins.mga.axes import Axis, axis_physical_unit
from zen_garden_plugins.mga.polytope_io import (
    CARRIER_IMPORT,
    NODE_CAPACITY_RATIO,
    NODE_CAPEX_CUMULATIVE,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    NODE_LIMIT_UTILISATION,
    NODE_OUTPUT_RATIO,
    TECH_CAPACITY,
    TOTAL_COST,
)

BENELUX = {"benelux": ["BE", "NL", "LU"]}


def _tuples(axes):
    return [(a.name, a.kind, a.members, a.period) for a in axes]


class TestTechnologyAndCarrierAxes:
    def test_singleton_and_lumped_axes_keep_user_order(self):
        axes, chains = axes_of(
            technologies=["nuclear", {"hydro": ["hydro_a", "hydro_b"]}],
            carrier_imports=["biomass"],
        )
        assert _tuples(axes) == [
            ("nuclear", TECH_CAPACITY, ("nuclear",), None),
            ("hydro", TECH_CAPACITY, ("hydro_a", "hydro_b"), None),
            ("biomass", CARRIER_IMPORT, ("biomass",), None),
        ]
        assert [a.capacity_type for a in axes] == ["power", "power", None]
        assert chains == []

    def test_empty_config_yields_no_axes(self):
        assert axes_of() == ([], [])

    def test_include_cost_appends_the_cost_axis_last(self):
        axes, _ = axes_of(technologies=["pv"], include_cost=True)
        assert [(a.name, a.kind) for a in axes] == [
            ("pv", TECH_CAPACITY),
            ("net_present_cost", TOTAL_COST),
        ]

    @pytest.mark.parametrize(
        "block",
        [
            {"technologies": ["typo"]},  # unknown technology
            {"carrier_imports": ["typo"]},  # unknown carrier
            {"technologies": ["nuclear", "nuclear"]},  # duplicate axis name
            {"technologies": [{"nuclear": ["pv"]}]},  # group shadows a tech
            {"technologies": [{"g": ["nuclear"]}, {"h": ["nuclear"]}]},  # twice
            {"technologies": [{"g": ["nuclear"], "h": ["pv"]}]},  # two-key dict
            {"technologies": [{"g": []}]},  # empty member list
            {"technologies": [42]},  # not a name or dict
        ],
    )
    def test_invalid_axis_configs_are_rejected(self, block):
        with pytest.raises(ValueError):
            axes_of(**block)

    def test_axis_name_cannot_be_used_twice_across_blocks(self):
        with pytest.raises(ValueError):
            axes_of(
                technologies=[{"shared": ["nuclear"]}],
                carrier_imports=[{"shared": ["biomass"]}],
            )


class TestCumulativeAxes:
    @pytest.mark.parametrize(
        "key, kind",
        [
            ("node_capex_cumulative", NODE_CAPEX_CUMULATIVE),
            ("node_carbon_emissions_cumulative", NODE_CARBON_EMISSIONS_CUMULATIVE),
        ],
    )
    def test_nodes_major_until_years_minor(self, key, kind):
        axes, _ = axes_of(
            **{key: {"nodes": ["DE", BENELUX], "until_years": [2030, 2040]}}
        )
        assert _tuples(axes) == [
            ("DE_until_2030", kind, ("DE",), (None, 2030)),
            ("DE_until_2040", kind, ("DE",), (None, 2040)),
            ("benelux_until_2030", kind, ("BE", "NL", "LU"), (None, 2030)),
            ("benelux_until_2040", kind, ("BE", "NL", "LU"), (None, 2040)),
        ]

    def test_capex_chains_are_sorted_ascending_by_until_year(self):
        _, chains = axes_of(
            node_capex_cumulative={
                "nodes": ["DE", BENELUX],
                "until_years": [2050, 2030, 2040],
            }
        )
        assert chains == [
            ["DE_until_2030", "DE_until_2040", "DE_until_2050"],
            ["benelux_until_2030", "benelux_until_2040", "benelux_until_2050"],
        ]

    def test_emissions_axes_have_no_monotone_chains(self):
        _, chains = axes_of(
            node_carbon_emissions_cumulative={
                "nodes": ["DE"],
                "until_years": [2030, 2040],
            }
        )
        assert chains == []

    @pytest.mark.parametrize(
        "key", ["node_capex_cumulative", "node_carbon_emissions_cumulative"]
    )
    @pytest.mark.parametrize(
        "block",
        [
            {"nodes": ["DE"]},  # nodes without until_years
            {"until_years": [2030]},  # until_years without nodes
            {"nodes": ["DE"], "until_years": []},  # empty until_years list
            {"nodes": ["DE"], "until_years": [2030.5]},  # non-int year
            {"nodes": ["DE"], "until_years": [2030, 2030]},  # duplicate year
            {"nodes": ["typo"], "until_years": [2030]},  # unknown node
        ],
    )
    def test_invalid_configs_are_rejected(self, key, block):
        with pytest.raises(ValueError):
            axes_of(**{key: block})

    def test_names_cannot_collide_across_blocks(self):
        with pytest.raises(ValueError, match="more than one"):
            axes_of(
                node_capex_cumulative={"nodes": ["DE"], "until_years": [2030]},
                node_carbon_emissions_cumulative={
                    "nodes": ["DE"],
                    "until_years": [2030],
                },
            )


class TestLimitUtilisationAxes:
    def test_nodes_major_groups_middle_years_minor(self):
        axes, chains = axes_of(
            node_limit_utilisation={
                "nodes": ["DE", BENELUX],
                "years": [2030, 2050],
                "technology_groups": [{"store": ["pv", "nuclear"]}],
            }
        )
        assert _tuples(axes) == [
            ("DE_store_2030", NODE_LIMIT_UTILISATION, ("DE",), (2030, 2030)),
            ("DE_store_2050", NODE_LIMIT_UTILISATION, ("DE",), (2050, 2050)),
            (
                "benelux_store_2030",
                NODE_LIMIT_UTILISATION,
                ("BE", "NL", "LU"),
                (2030, 2030),
            ),
            (
                "benelux_store_2050",
                NODE_LIMIT_UTILISATION,
                ("BE", "NL", "LU"),
                (2050, 2050),
            ),
        ]
        assert axes[0].technologies == ("pv", "nuclear")
        assert axes[0].denominator_technologies is None
        assert chains == []

    def test_unit_is_dimensionless(self):
        axis = Axis("a", NODE_LIMIT_UTILISATION, ("DE",), None, (2030, 2030))
        assert axis_physical_unit(axis, {}, pint.UnitRegistry()) == "dimensionless"

    @pytest.mark.parametrize(
        "block",
        [
            {"nodes": ["DE"]},
            {"years": [2030]},
            {"technology_groups": ["pv"]},
            {"nodes": ["DE"], "years": [2030]},
            {"nodes": ["DE"], "technology_groups": ["pv"]},
            {"years": [2030], "technology_groups": ["pv"]},
            {"nodes": ["typo"], "years": [2030], "technology_groups": ["pv"]},
            {"nodes": ["DE"], "years": [2030], "technology_groups": ["typo"]},
            {"nodes": ["DE"], "years": [2030, 2030], "technology_groups": ["pv"]},
            # group name shadows a technology
            {"nodes": ["DE"], "years": [2030], "technology_groups": [{"pv": ["pv"]}]},
        ],
    )
    def test_invalid_configs_are_rejected(self, block):
        with pytest.raises(ValueError):
            axes_of(node_limit_utilisation=block)

    def test_names_cannot_collide_with_ratio_axes(self):
        with pytest.raises(ValueError, match="more than one"):
            axes_of(
                node_output_ratio={
                    "nodes": ["DE"],
                    "years": [2030],
                    "ratio_groups": [PV_SHARE],
                },
                node_limit_utilisation={
                    "nodes": ["DE"],
                    "years": [2030],
                    "technology_groups": [{"g": ["pv"]}],
                },
            )


class TestRatioAxes:
    @pytest.mark.parametrize(
        "key, kind, capacity_type",
        [
            ("node_capacity_ratio", NODE_CAPACITY_RATIO, "power"),
            ("node_output_ratio", NODE_OUTPUT_RATIO, None),
        ],
    )
    def test_nodes_major_ratio_groups_middle_years_minor(
        self, key, kind, capacity_type
    ):
        renewables = {
            "numerator": ["pv", "hydro_a"],
            "denominator": ["pv", "hydro_a", "nuclear"],
        }
        hydro = {
            "numerator": ["hydro_a", "hydro_b"],
            "denominator": ["hydro_a", "hydro_b"],
        }
        axes, chains = axes_of(
            **{
                key: {
                    "nodes": ["DE", BENELUX],
                    "years": [2030, 2040],
                    "ratio_groups": [{"renewables": renewables}, {"hydro": hydro}],
                }
            }
        )
        assert [a.name for a in axes] == [
            "DE_renewables_2030",
            "DE_renewables_2040",
            "DE_hydro_2030",
            "DE_hydro_2040",
            "benelux_renewables_2030",
            "benelux_renewables_2040",
            "benelux_hydro_2030",
            "benelux_hydro_2040",
        ]
        first = axes[0]
        assert (first.kind, first.members, first.period) == (
            kind,
            ("DE",),
            (2030, 2030),
        )
        assert first.technologies == ("pv", "hydro_a")
        assert first.denominator_technologies == ("pv", "hydro_a", "nuclear")
        assert first.capacity_type == capacity_type
        assert chains == []

    def test_numerator_may_overlap_its_denominator(self):
        axes, _ = axes_of(
            node_capacity_ratio={
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [PV_SHARE],
            }
        )
        assert [(a.name, a.technologies, a.denominator_technologies) for a in axes] == [
            ("DE_g_2030", ("pv",), ("pv", "nuclear"))
        ]

    def test_absent_block_yields_no_axes(self):
        assert axes_of(node_output_ratio=None) == ([], [])

    @pytest.mark.parametrize("key", ["node_capacity_ratio", "node_output_ratio"])
    @pytest.mark.parametrize(
        "block",
        [
            {"nodes": ["DE"]},  # nodes without years/ratio_groups
            {"years": [2030]},  # years without nodes/ratio_groups
            {"ratio_groups": [PV_SHARE]},
            {"nodes": ["DE"], "years": [2030]},  # missing ratio_groups
            {"nodes": ["DE"], "ratio_groups": [PV_SHARE]},  # missing years
            {"years": [2030], "ratio_groups": [PV_SHARE]},  # missing nodes
            {"nodes": ["typo"], "years": [2030], "ratio_groups": [PV_SHARE]},
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [{"g": {"numerator": ["typo"], "denominator": ["pv"]}}],
            },  # unknown numerator technology
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [{"g": {"numerator": ["pv"], "denominator": ["typo"]}}],
            },  # unknown denominator technology
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [{"g": {"numerator": ["pv"]}}],
            },  # missing denominator key
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [{"g": {"numerator": [], "denominator": ["pv"]}}],
            },  # empty numerator list
            {"nodes": ["DE"], "years": [2030, 2030], "ratio_groups": [PV_SHARE]},
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [
                    PV_SHARE,
                    {"g": {"numerator": ["nuclear"], "denominator": ["nuclear"]}},
                ],
            },  # duplicate ratio-group name
            {
                "nodes": ["DE"],
                "years": [2030],
                "ratio_groups": [{"pv": {"numerator": ["pv"], "denominator": ["pv"]}}],
            },  # group name shadows a technology
        ],
    )
    def test_invalid_configs_are_rejected(self, key, block):
        with pytest.raises(ValueError):
            axes_of(**{key: block})

    def test_names_cannot_collide_across_ratio_blocks(self):
        block = {"nodes": ["DE"], "years": [2030], "ratio_groups": [PV_SHARE]}
        with pytest.raises(ValueError, match="more than one"):
            axes_of(node_capacity_ratio=block, node_output_ratio=block)

    def test_capacity_ratio_rejects_mismatched_capacity_types(self):
        def capacity_type_of(name, members):
            return "energy" if "battery" in members else "power"

        with pytest.raises(ValueError, match="numerator capacity type"):
            axes_of(
                capacity_type_of,
                node_capacity_ratio={
                    "nodes": ["DE"],
                    "years": [2030],
                    "ratio_groups": [
                        {"g": {"numerator": ["pv"], "denominator": ["battery"]}}
                    ],
                },
            )


class TestPhysicalUnits:
    CAPACITY_UNITS = units_series(
        ["set_technologies", "set_capacity_types", "set_location", "set_years"],
        [
            ("nuclear", "power", "DE", 2050),
            ("battery", "power", "DE", 2050),
            ("battery", "energy", "DE", 2050),
        ],
        ["gigawatt", "gigawatt", "gigawatt_hour"],
    )
    IMPORT_UNITS = units_series(
        ["set_carriers", "set_nodes", "set_time_steps_operation"],
        [("biomass", "DE", 0), ("hydrogen", "DE", 0)],
        ["gigawatt", "gigawatt"],
    )
    CAPEX_UNITS = units_series(
        ["set_technologies", "set_capacity_types", "set_location", "set_years"],
        [("nuclear", "power", "DE", 2050), ("battery", "power", "CH", 2050)],
        ["megaEuro", "megaEuro"],
    )
    EMISSIONS_UNITS = units_series(
        ["set_technologies", "set_location", "set_time_steps_operation"],
        [("nuclear", "DE", 0), ("nuclear", "CH", 0)],
        ["kilogram", "kilogram"],
    )

    def _unit(self, axis, units):
        return axis_physical_unit(axis, units, pint.UnitRegistry())

    def test_tech_axis_unit_follows_the_selected_capacity_type(self):
        units = {"capacity_addition": self.CAPACITY_UNITS}
        power = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
        energy = Axis("battery", TECH_CAPACITY, ("battery",), "energy")
        assert self._unit(power, units) == "gigawatt"
        assert self._unit(energy, units) == "gigawatt_hour"

    def test_carrier_axis_unit_is_annualised(self):
        axis = Axis("biomass", CARRIER_IMPORT, ("biomass",), None)
        assert self._unit(axis, {"flow_import": self.IMPORT_UNITS}) == "gigawatt * hour"

    def test_capex_cumulative_axis_unit_reads_the_capex_variable_unit(self):
        axis = Axis("DE_until_2030", NODE_CAPEX_CUMULATIVE, ("DE",), None, (None, 2030))
        assert self._unit(axis, {"cost_capex_yearly": self.CAPEX_UNITS}) == "megaEuro"

    @pytest.mark.parametrize("kind", [NODE_CAPACITY_RATIO, NODE_OUTPUT_RATIO])
    def test_ratio_axis_unit_is_dimensionless(self, kind):
        axis = Axis("DE_g_2030", kind, ("DE",), "power", (2030, 2030), ("pv",), ("pv",))
        assert self._unit(axis, {}) == "dimensionless"
        assert self._unit(axis, {"capacity_addition": self.CAPACITY_UNITS}) == (
            "dimensionless"
        )

    def test_emissions_axis_unit_is_annualised(self):
        axis = Axis(
            "DE_until_2030",
            NODE_CARBON_EMISSIONS_CUMULATIVE,
            ("DE",),
            None,
            (None, 2030),
        )
        units = {"carbon_emissions_technology": self.EMISSIONS_UNITS}
        assert self._unit(axis, units) == "hour * kilogram"
        assert self._unit(axis, {}) is None

    def test_cost_axis_reads_the_cost_variable_unit(self):
        axis = Axis("net_present_cost", TOTAL_COST, (), None)
        units = {"net_present_cost": pd.Series(["megaEuro"], dtype=str)}
        assert self._unit(axis, units) == "megaEuro"

    def test_unit_is_none_when_the_model_tracks_no_units(self):
        axis = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
        assert self._unit(axis, {}) is None
