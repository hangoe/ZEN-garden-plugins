"""Tests of normalisation.py: scale/offset per convention and references."""

import numpy as np
import pytest

from zen_garden_plugins.mga.axes import Axis
from zen_garden_plugins.mga.normalisation import (
    describe,
    per_axes_references,
    round_to_one_significant_figure,
    scale_and_offset,
    share_references,
)
from zen_garden_plugins.mga.polytope_io import (
    NODE_CAPACITY_RATIO,
    NODE_CAPEX_CUMULATIVE,
    NODE_CARBON_EMISSIONS_CUMULATIVE,
    TOTAL_COST,
)

CAPEX = Axis("DE_until_2030", NODE_CAPEX_CUMULATIVE, ("DE",), None, (None, 2030))
EMISSIONS = Axis(
    "DE_co2", NODE_CARBON_EMISSIONS_CUMULATIVE, ("DE",), None, (None, 2030)
)
RATIO = Axis(
    "DE_g_2030", NODE_CAPACITY_RATIO, ("DE",), "power", (2030, 2030), ("pv",), ("pv",)
)
COST = Axis("net_present_cost", TOTAL_COST, (), None)


def _scale(axes, lower, upper, normalisation, **refs):
    return scale_and_offset(axes, lower, upper, 1000.0, 0.1, normalisation, **refs)


class TestScaleAndOffset:
    def test_relative_scales_by_the_upper_bound(self):
        bounds, scale, offset = _scale([CAPEX, COST], [10.0], [50.0], "relative")
        assert scale.tolist() == [50.0, pytest.approx(100.0)]
        assert offset.tolist() == [0.0, 1000.0]
        assert bounds.tolist() == [[10.0, 50.0], [1000.0, pytest.approx(1100.0)]]

    def test_minmax_maps_the_range_to_the_unit_interval(self):
        _, scale, offset = _scale([CAPEX], [50.0], [150.0], "minmax")
        assert (scale.tolist(), offset.tolist()) == ([100.0], [50.0])

    def test_minmax_accepts_a_non_positive_emissions_maximum(self):
        bounds, scale, offset = _scale([EMISSIONS], [-500.0], [-20.0], "minmax")
        assert (scale.tolist(), offset.tolist()) == ([480.0], [-500.0])
        assert ((bounds - offset[:, None]) / scale[:, None]).tolist() == [[0.0, 1.0]]

    def test_relative_rejects_a_non_positive_maximum(self):
        with pytest.raises(RuntimeError, match="non-positive near-optimal maximum"):
            _scale([EMISSIONS], [-500.0], [-20.0], "relative")

    def test_minmax_rejects_a_zero_span(self):
        with pytest.raises(RuntimeError, match="zero near-optimal range"):
            _scale([EMISSIONS], [-20.0], [-20.0], "minmax")

    def test_lower_above_upper_is_rejected(self):
        with pytest.raises(RuntimeError, match="above"):
            _scale([CAPEX], [60.0], [50.0], "relative")

    def test_share_uses_the_reference_with_zero_offset(self):
        _, scale, offset = _scale(
            [CAPEX, COST],
            [0.0],
            [50.0],
            "share",
            share_reference=np.array([200.0, np.nan]),
        )
        assert scale[0] == 200.0 and offset[0] == 0.0

    def test_share_rejects_a_non_positive_reference(self):
        with pytest.raises(RuntimeError, match="non-positive share reference"):
            _scale([CAPEX], [10.0], [50.0], "share", share_reference=np.array([0.0]))

    def test_per_axes_uses_the_reference_of_the_axis(self):
        _, scale, _ = _scale(
            [CAPEX, RATIO],
            [0.0, 0.0],
            [50.0, 0.5],
            "per_axes",
            per_axes_reference={CAPEX.name: 15e12, RATIO.name: 1.0},
        )
        assert scale.tolist() == [15e12, 1.0]

    def test_cost_axis_maps_optimum_to_zero_and_budget_to_one(self):
        bounds, scale, offset = _scale([COST], [], [], "minmax")
        assert ((bounds - offset) / scale).tolist() == [[0.0, pytest.approx(1.0)]]


class TestReferences:
    def test_share_references_are_rounded_and_nan_for_cost(self):
        raw, rounded = share_references([CAPEX, COST], lambda axis: 12.3e6)
        assert raw[0] == 12.3e6 and np.isnan(raw[1])
        assert rounded[0] == 1e7 and np.isnan(rounded[1])

    def test_per_axes_references_by_kind(self):
        refs = per_axes_references([CAPEX, RATIO, EMISSIONS, COST], 15e12, 500.0)
        assert refs == {CAPEX.name: 15e12, RATIO.name: 1.0, EMISSIONS.name: 500.0}

    @pytest.mark.parametrize(
        "value, expected",
        [
            (12.3e6, 1e7),  # mantissa rounds down
            (18e12, 2e13),  # mantissa rounds up
            (1e5, 1e5),  # power of ten unchanged
            (9.7e3, 1e4),  # mantissa rounds to ten and carries
        ],
    )
    def test_round_to_one_significant_figure(self, value, expected):
        assert round_to_one_significant_figure(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [0.0, -5.0])
    def test_round_leaves_non_positive_values_unchanged(self, value):
        assert round_to_one_significant_figure(value) == value

    def test_round_leaves_non_finite_values_unchanged(self):
        assert np.isnan(round_to_one_significant_figure(np.nan))
        assert round_to_one_significant_figure(np.inf) == np.inf


class TestDescription:
    """The description is stored in every polytope; it must not change."""

    def test_relative(self):
        assert describe("relative") == (
            "physical = normalised * scale + offset, per axis; design axes use "
            "(upper bound, 0), the cost axis always (epsilon * c_star, c_star)"
        )

    def test_minmax(self):
        assert describe("minmax") == (
            "physical = normalised * scale + offset, per axis; design axes use "
            "(upper bound - lower bound, lower bound) -- near-optimal [min, max] "
            "mapped to [0, 1], the cost axis always (epsilon * c_star, c_star)"
        )

    def test_share_and_per_axes_keep_their_stored_text(self):
        assert "a separate per-technology-group total" in describe("share")
        assert "divides by the model's carbon_emissions_budget" in describe("per_axes")
