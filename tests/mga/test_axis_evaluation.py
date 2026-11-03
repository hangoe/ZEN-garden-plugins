"""Tests of axis_evaluation.py: per-kind axis values and frozen denominators."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mga_helpers import FakeSolved, evaluator_stub

from zen_garden_plugins.mga.axes import Axis
from zen_garden_plugins.mga.axis_evaluation import (
    selected_capacity_type,
    year_indices_in_period,
    year_interval_expansion_factors,
)
from zen_garden_plugins.mga.polytope_io import (
    NODE_CAPACITY_RATIO,
    NODE_CAPEX_CUMULATIVE,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    NODE_OUTPUT_RATIO,
)

YEARS = [0, 1, 2]


def _factor(values=(1.0, 1.0, 1.0)):
    return xr.DataArray(list(values), dims=["set_years"], coords={"set_years": YEARS})


def _capex():
    """cost_capex_yearly stand-in: nuclear at DE/CH, transport_ab at edge AB."""
    data = np.full((2, 1, 3, 3), np.nan)
    data[0, 0, 0, :] = [10.0, 20.0, 30.0]  # nuclear @ DE
    data[0, 0, 1, :] = [5.0, 5.0, 5.0]  # nuclear @ CH
    data[1, 0, 2, :] = [100.0, 100.0, 100.0]  # transport_ab @ edge AB
    return xr.DataArray(
        data,
        dims=["set_technologies", "set_capacity_types", "set_location", "set_years"],
        coords={
            "set_technologies": ["nuclear", "transport_ab"],
            "set_capacity_types": ["power"],
            "set_location": ["DE", "CH", "AB"],
            "set_years": YEARS,
        },
    )


def _emissions():
    """Annual carbon_emissions_technology stand-in, same values as _capex."""
    return _capex().isel(set_capacity_types=0, drop=True)


def _capacity():
    """Stock capacity stand-in; wind has zero capacity everywhere."""
    data = np.zeros((3, 1, 2, 2))
    data[0, 0, 0, :] = [30.0, 40.0]  # pv @ DE
    data[0, 0, 1, :] = [3.0, 4.0]  # pv @ CH
    data[1, 0, 0, :] = [70.0, 60.0]  # nuclear @ DE
    data[1, 0, 1, :] = [7.0, 6.0]  # nuclear @ CH
    return xr.DataArray(
        data,
        dims=["set_technologies", "set_capacity_types", "set_location", "set_years"],
        coords={
            "set_technologies": ["pv", "nuclear", "wind"],
            "set_capacity_types": ["power"],
            "set_location": ["DE", "CH"],
            "set_years": [0, 1],
        },
    )


def _output():
    """Annual flow_conversion_output stand-in, same values as _capacity."""
    return (
        _capacity()
        .rename(
            set_technologies="set_conversion_technologies",
            set_capacity_types="set_output_carriers",
            set_location="set_nodes",
        )
        .assign_coords(set_output_carriers=["electricity"])
    )


def _ones_mask():
    return xr.ones_like(_capacity().isel(set_location=0, set_years=0, drop=True))


def _ratio_axis(kind, denominator=("pv", "nuclear"), year=1):
    return Axis(
        "DE_g",
        kind,
        ("DE",),
        "power" if kind == NODE_CAPACITY_RATIO else None,
        period=(year, year),
        technologies=("pv",),
        denominator_technologies=denominator,
    )


class TestCapexCumulative:
    def _evaluator(self, years, factor=None):
        return evaluator_stub(
            cost_capex_yearly=FakeSolved(_capex()),
            axis_year_indices={"a": years},
            capex_discount_factor=factor if factor is not None else _factor(),
        )

    def test_restricts_to_years_up_to_until_year_and_drops_transport(self):
        axis = Axis("a", NODE_CAPEX_CUMULATIVE, ("DE",), None, (None, 1))
        assert self._evaluator([0, 1]).value(axis) == pytest.approx(30.0)  # 10+20

    def test_lumps_multiple_nodes(self):
        axis = Axis("a", NODE_CAPEX_CUMULATIVE, ("DE", "CH"), None, (None, 2))
        assert self._evaluator(YEARS).value(axis) == pytest.approx(75.0)

    def test_applies_the_discount_factor_within_its_year_window(self):
        axis = Axis("a", NODE_CAPEX_CUMULATIVE, ("DE",), None, (None, 1))
        evaluator = self._evaluator([0, 1], _factor((1.0, 2.0, 4.0)))
        assert evaluator.value(axis) == pytest.approx(50.0)  # 10*1 + 20*2

    def test_share_reference_widens_to_all_nodes_and_years(self):
        axis = Axis("a", NODE_CAPEX_CUMULATIVE, ("DE",), None, (None, 1))
        evaluator = self._evaluator([0, 1])
        evaluator.all_nodes = ["DE", "CH"]
        assert evaluator.share_reference_value(axis) == pytest.approx(75.0)


class TestEmissionsCumulative:
    def _evaluator(self, years, factor=None):
        return evaluator_stub(
            carbon_emissions_yearly=FakeSolved(_emissions()),
            axis_year_indices={"a": years},
            emissions_interval_factor=factor if factor is not None else _factor(),
        )

    def test_restricts_to_years_and_drops_transport(self):
        axis = Axis("a", NODE_CARBON_EMISSIONS_CUMULATIVE, ("DE",), None, (None, 1))
        assert self._evaluator([0, 1]).value(axis) == pytest.approx(30.0)

    def test_lumps_multiple_nodes(self):
        axis = Axis(
            "a", NODE_CARBON_EMISSIONS_CUMULATIVE, ("DE", "CH"), None, (None, 2)
        )
        assert self._evaluator(YEARS).value(axis) == pytest.approx(75.0)

    def test_applies_the_interval_expansion_factor(self):
        axis = Axis("a", NODE_CARBON_EMISSIONS_CUMULATIVE, ("DE",), None, (None, 2))
        evaluator = self._evaluator(YEARS, _factor((1.0, 2.0, 4.0)))
        assert evaluator.value(axis) == pytest.approx(170.0)  # 10 + 40 + 120


class TestRatioAxes:
    def _evaluator(self, year):
        return evaluator_stub(
            capacity=FakeSolved(_capacity()),
            flow_conversion_output_yearly=FakeSolved(_output()),
            capacity_mask=_ones_mask(),
            axis_year_indices={"DE_g": [year]},
            ratio_denominator={"DE_g": 100.0},
        )

    @pytest.mark.parametrize("kind", [NODE_CAPACITY_RATIO, NODE_OUTPUT_RATIO])
    @pytest.mark.parametrize("year, expected", [(1, 0.4), (0, 0.3)])
    def test_divides_own_year_numerator_by_frozen_denominator(
        self, kind, year, expected
    ):
        axis = _ratio_axis(kind, year=year)
        assert self._evaluator(year).value(axis) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "kind, method",
        [
            (NODE_CAPACITY_RATIO, "_capacity_ratio_denominator"),
            (NODE_OUTPUT_RATIO, "_output_ratio_denominator"),
        ],
    )
    def test_denominator_stays_scoped_to_own_node_and_year(self, kind, method):
        evaluator = self._evaluator(1)
        value = getattr(evaluator, method)(_ratio_axis(kind))
        assert value == pytest.approx(100.0)  # pv@DE 40 + nuclear@DE 60, not CH

    @pytest.mark.parametrize(
        "kind, method",
        [
            (NODE_CAPACITY_RATIO, "_capacity_ratio_denominator"),
            (NODE_OUTPUT_RATIO, "_output_ratio_denominator"),
        ],
    )
    def test_non_positive_denominator_is_rejected(self, kind, method):
        evaluator = self._evaluator(1)
        with pytest.raises(RuntimeError, match="non-positive baseline denominator"):
            getattr(evaluator, method)(_ratio_axis(kind, denominator=("wind",)))


class TestOutputRatioValidation:
    def _validate(self, unit_by_tech):
        techs = list(unit_by_tech)
        variables = {
            "flow_conversion_output": SimpleNamespace(
                coords={
                    "set_conversion_technologies": SimpleNamespace(
                        values=np.array(techs)
                    )
                }
            )
        }
        units = pd.Series(
            list(unit_by_tech.values()),
            index=pd.MultiIndex.from_tuples(
                [(t, "c", "DE") for t in techs],
                names=[
                    "set_conversion_technologies",
                    "set_output_carriers",
                    "set_nodes",
                ],
            ),
        )
        evaluator = evaluator_stub(model=SimpleNamespace(variables=variables))
        evaluator._validate_output_ratio_axes(
            [_ratio_axis(NODE_OUTPUT_RATIO)], {"flow_conversion_output": units}
        )

    def test_accepts_a_single_output_unit(self):
        self._validate({"pv": "gigawatt", "nuclear": "gigawatt"})

    def test_rejects_mixed_output_units(self):
        with pytest.raises(ValueError, match="different units"):
            self._validate({"pv": "megavkm / hour", "nuclear": "megatkm / hour"})

    def test_rejects_non_conversion_technologies(self):
        with pytest.raises(ValueError, match="not conversion technologies"):
            self._validate({"pv": "gigawatt"})


class TestCapacityType:
    MASK = xr.DataArray(
        [[1.0, 0.0], [0.0, 1.0]],
        dims=["set_technologies", "set_capacity_types"],
        coords={
            "set_technologies": ["pv", "battery"],
            "set_capacity_types": ["power", "energy"],
        },
    )

    def test_selected_type_per_group(self):
        assert selected_capacity_type(self.MASK, "a", ["pv"]) == "power"
        assert selected_capacity_type(self.MASK, "b", ["battery"]) == "energy"

    def test_mixing_storage_and_non_storage_is_rejected(self):
        with pytest.raises(ValueError, match="mixes capacity types"):
            selected_capacity_type(self.MASK, "a", ["pv", "battery"])


class TestYears:
    def test_year_indices_in_period_is_boundary_inclusive(self):
        assert year_indices_in_period(
            (2030, 2040), [0, 1, 2, 3], [2025, 2030, 2040, 2045]
        ) == [1, 2]

    def test_year_indices_in_period_is_empty_outside_the_horizon(self):
        assert year_indices_in_period((2050, 2060), [0, 1], [2030, 2040]) == []

    def test_no_lower_bound_covers_from_the_first_year(self):
        assert year_indices_in_period((None, 2040), [0, 1, 2], [2030, 2040, 2050]) == [
            0,
            1,
        ]

    def test_expansion_factors_match_constraint_net_present_cost(self):
        factors = year_interval_expansion_factors(
            YEARS, discount_rate=0.1, interval_between_years=2, last_year_index=2
        )
        r = 1.0 / 1.1
        assert list(factors.values) == pytest.approx([1 + r, r**2 + r**3, r**4])

    def test_zero_discount_rate_gives_the_plain_interval_count(self):
        factors = year_interval_expansion_factors(
            YEARS, discount_rate=0.0, interval_between_years=2, last_year_index=2
        )
        assert list(factors.values) == pytest.approx([2.0, 2.0, 1.0])
