"""Fast unit tests for the MGA plugin's model-independent parts.

These run in milliseconds and need neither a solver nor a dataset: axis
config parsing, the config-key validation, row normalisation, and the
polytope npz round-trip.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr

from zen_garden_plugins.mga.axes import Axis, axis_physical_unit, build_axis_groups
from zen_garden_plugins.mga.batch_driver import run_batch_mode
from zen_garden_plugins.mga.oracle_driver import run_oracle_mode
from zen_garden_plugins.mga.plugin import (
    MGA,
    _year_indices_in_period,
    normalise_rows,
    validate_config,
)
from zen_garden_plugins.mga.polytope_io import (
    CARRIER_IMPORT,
    NODE_CAPEX,
    NODE_CAPEX_PERIOD,
    NODE_CAPEX_TECH,
    TECH_CAPACITY,
    TOTAL_COST,
    Polytope,
    load_polytope,
    norm_to_phys,
    phys_to_norm,
    save_polytope,
)
from zen_garden_plugins.mga.supf_driver import run_bbo_mode, run_sampling_mode

TECHS = ["nuclear", "pv", "battery", "hydro_a", "hydro_b"]
CARRIERS = ["biomass", "hydrogen"]
NODES = ["DE", "CH", "FR", "BE", "NL", "LU"]


# ---------------------------------------------------------------- axis config


def test_singleton_and_lumped_axes_keep_user_order():
    (
        tech_groups,
        carrier_groups,
        node_capex_groups,
        node_capex_period_axes,
        node_capex_tech_axes,
    ) = build_axis_groups(
        ["nuclear", {"hydro": ["hydro_a", "hydro_b"]}],
        ["biomass"],
        TECHS,
        CARRIERS,
    )
    assert tech_groups == [("nuclear", ["nuclear"]), ("hydro", ["hydro_a", "hydro_b"])]
    assert carrier_groups == [("biomass", ["biomass"])]
    assert node_capex_groups == []
    assert node_capex_period_axes == []
    assert node_capex_tech_axes == []


def test_empty_config_yields_no_axes():
    assert build_axis_groups(None, None, TECHS, CARRIERS) == ([], [], [], [], [])


@pytest.mark.parametrize(
    "technologies, carriers",
    [
        (["typo"], None),  # unknown technology
        (None, ["typo"]),  # unknown carrier
        (["nuclear", "nuclear"], None),  # duplicate axis name
        ([{"nuclear": ["pv"]}], None),  # group shadows a tech
        ([{"g": ["nuclear"]}, {"h": ["nuclear"]}], None),  # member twice
        ([{"g": ["nuclear"], "h": ["pv"]}], None),  # two-key dict
        ([{"g": []}], None),  # empty member list
        ([42], None),  # not a name or dict
    ],
)
def test_invalid_axis_configs_are_rejected(technologies, carriers):
    with pytest.raises(ValueError):
        build_axis_groups(technologies, carriers, TECHS, CARRIERS)


def test_axis_name_cannot_be_used_twice_across_blocks():
    with pytest.raises(ValueError):
        build_axis_groups(
            [{"shared": ["nuclear"]}], [{"shared": ["biomass"]}], TECHS, CARRIERS
        )


# --------------------------------------------------------- node capex axes


def test_singleton_and_lumped_node_capex_axes_keep_user_order():
    _, _, node_capex_groups, _, _ = build_axis_groups(
        None,
        None,
        TECHS,
        CARRIERS,
        node_capex=["DE", {"benelux": ["BE", "NL", "LU"]}],
        all_nodes=NODES,
    )
    assert node_capex_groups == [
        ("DE", ["DE"]),
        ("benelux", ["BE", "NL", "LU"]),
    ]


@pytest.mark.parametrize(
    "node_capex",
    [
        ["typo"],  # unknown node
        ["DE", "DE"],  # duplicate axis name
        [{"DE": ["CH"]}],  # group shadows a node
        [{"g": ["DE"]}, {"h": ["DE"]}],  # member twice
        [{"g": []}],  # empty member list
    ],
)
def test_invalid_node_capex_configs_are_rejected(node_capex):
    with pytest.raises(ValueError):
        build_axis_groups(
            None, None, TECHS, CARRIERS, node_capex=node_capex, all_nodes=NODES
        )


def test_node_capex_axis_name_cannot_collide_with_tech_or_carrier_axis():
    with pytest.raises(ValueError):
        build_axis_groups(
            [{"shared": ["nuclear"]}],
            None,
            TECHS,
            CARRIERS,
            node_capex=[{"shared": ["DE"]}],
            all_nodes=NODES,
        )


def test_node_capex_periods_produce_nodes_major_periods_minor_axes():
    _, _, _, node_capex_period_axes, _ = build_axis_groups(
        None,
        None,
        TECHS,
        CARRIERS,
        node_capex_periods={
            "nodes": ["DE", {"benelux": ["BE", "NL", "LU"]}],
            "periods": [[2021, 2030], [2031, 2040]],
        },
        all_nodes=NODES,
    )
    assert node_capex_period_axes == [
        ("DE_2021_2030", ["DE"], (2021, 2030)),
        ("DE_2031_2040", ["DE"], (2031, 2040)),
        ("benelux_2021_2030", ["BE", "NL", "LU"], (2021, 2030)),
        ("benelux_2031_2040", ["BE", "NL", "LU"], (2031, 2040)),
    ]


@pytest.mark.parametrize(
    "node_capex_periods",
    [
        {"nodes": ["DE"]},  # nodes without periods
        {"periods": [[2021, 2030]]},  # periods without nodes
        {"nodes": ["DE"], "periods": []},  # empty periods list
        {"nodes": ["DE"], "periods": [[2030, 2021]]},  # start > end
        {"nodes": ["DE"], "periods": [[2021, 2030.5]]},  # non-int bound
        {"nodes": ["DE"], "periods": [[2021, 2030], [2025, 2035]]},  # overlap
    ],
)
def test_invalid_node_capex_periods_are_rejected(node_capex_periods):
    with pytest.raises(ValueError):
        build_axis_groups(
            None,
            None,
            TECHS,
            CARRIERS,
            node_capex_periods=node_capex_periods,
            all_nodes=NODES,
        )


# ------------------------------------------------ node capex by technology


def test_node_capex_by_technology_produces_nodes_major_tech_groups_minor_axes():
    _, _, _, _, node_capex_tech_axes = build_axis_groups(
        None,
        None,
        TECHS,
        CARRIERS,
        node_capex_by_technology={
            "nodes": ["DE", {"benelux": ["BE", "NL", "LU"]}],
            "technology_groups": [
                {"renewables": ["pv", "hydro_a", "hydro_b"]},
                "nuclear",
            ],
        },
        all_nodes=NODES,
    )
    assert node_capex_tech_axes == [
        ("DE_renewables", ["DE"], ["pv", "hydro_a", "hydro_b"]),
        ("DE_nuclear", ["DE"], ["nuclear"]),
        ("benelux_renewables", ["BE", "NL", "LU"], ["pv", "hydro_a", "hydro_b"]),
        ("benelux_nuclear", ["BE", "NL", "LU"], ["nuclear"]),
    ]


@pytest.mark.parametrize(
    "node_capex_by_technology",
    [
        {"nodes": ["DE"]},  # nodes without technology_groups
        {"technology_groups": [{"renewables": ["pv"]}]},  # groups without nodes
        {"nodes": ["DE"], "technology_groups": [{"g": ["typo"]}]},  # unknown tech
        {"nodes": ["typo"], "technology_groups": [{"g": ["pv"]}]},  # unknown node
        {"nodes": ["DE"], "technology_groups": [{"DE": ["pv"]}]},  # shadows a node
        {"nodes": ["DE"], "technology_groups": [{"g": []}]},  # empty member list
    ],
)
def test_invalid_node_capex_by_technology_configs_are_rejected(
    node_capex_by_technology,
):
    with pytest.raises(ValueError):
        build_axis_groups(
            None,
            None,
            TECHS,
            CARRIERS,
            node_capex_by_technology=node_capex_by_technology,
            all_nodes=NODES,
        )


def test_node_capex_by_technology_axis_name_cannot_collide_with_node_capex_axis():
    with pytest.raises(ValueError):
        build_axis_groups(
            None,
            None,
            TECHS,
            CARRIERS,
            node_capex=[{"DE_renewables": ["DE"]}],
            node_capex_by_technology={
                "nodes": ["DE"],
                "technology_groups": [{"renewables": ["pv"]}],
            },
            all_nodes=NODES,
        )


# ------------------------------------------------------- node capex aggregation


def _capex_data_array():
    """A tiny cost_capex_yearly-shaped DataArray: `nuclear` valid at nodes
    DE/CH (3 years each), `transport_ab` valid only at edge "AB" -- mirroring
    ZEN-garden's mixed node/edge set_location coordinate, with NaN at every
    invalid (technology, location) combination."""
    coords = {
        "set_technologies": ["nuclear", "transport_ab"],
        "set_capacity_types": ["power"],
        "set_location": ["DE", "CH", "AB"],
        "set_time_steps_yearly": [0, 1, 2],
    }
    data = np.full((2, 1, 3, 3), np.nan)
    data[0, 0, 0, :] = [10.0, 20.0, 30.0]  # nuclear @ DE, years 0/1/2
    data[0, 0, 1, :] = [5.0, 5.0, 5.0]  # nuclear @ CH, years 0/1/2
    data[1, 0, 2, :] = [100.0, 100.0, 100.0]  # transport_ab @ edge AB
    return xr.DataArray(
        data,
        dims=[
            "set_technologies",
            "set_capacity_types",
            "set_location",
            "set_time_steps_yearly",
        ],
        coords=coords,
    )


def _node_capex_stub(axis_year_indices=None):
    return SimpleNamespace(
        _axis_year_indices=axis_year_indices or {},
        _NODE_AGG_CAPEX=MGA._NODE_AGG_CAPEX,
    )


def test_node_capex_axis_sums_capex_over_years_and_transport_tech_drops_out():
    axis = Axis("DE", NODE_CAPEX, ("DE",), None)
    value = MGA._design_axis_terms(
        _node_capex_stub(), axis, None, None, capex=_capex_data_array()
    )
    assert float(value.sum(skipna=True)) == pytest.approx(60.0)  # 10+20+30


def test_node_capex_axis_lumps_multiple_nodes():
    axis = Axis("DE_CH", NODE_CAPEX, ("DE", "CH"), None)
    value = MGA._design_axis_terms(
        _node_capex_stub(), axis, None, None, capex=_capex_data_array()
    )
    assert float(value.sum(skipna=True)) == pytest.approx(75.0)  # 60 + 15


def test_node_capex_period_axis_restricts_to_the_axis_year_indices():
    axis = Axis("DE_early", NODE_CAPEX_PERIOD, ("DE",), None, period=(0, 1))
    stub = _node_capex_stub({"DE_early": [0, 1]})
    value = MGA._design_axis_terms(stub, axis, None, None, capex=_capex_data_array())
    assert float(value.sum(skipna=True)) == pytest.approx(30.0)  # 10+20


def _capex_data_array_multi_tech():
    """A cost_capex_yearly-shaped DataArray with two real technologies
    ("nuclear", "pv") both valid at the same node "DE", for testing that a
    NODE_CAPEX_TECH axis's technology restriction actually excludes the
    other technology's capex at that same node (unlike NODE_CAPEX, which
    sums every technology)."""
    coords = {
        "set_technologies": ["nuclear", "pv"],
        "set_capacity_types": ["power"],
        "set_location": ["DE"],
        "set_time_steps_yearly": [0, 1, 2],
    }
    data = np.empty((2, 1, 1, 3))
    data[0, 0, 0, :] = [10.0, 20.0, 30.0]  # nuclear @ DE, years 0/1/2
    data[1, 0, 0, :] = [100.0, 100.0, 100.0]  # pv @ DE, years 0/1/2
    return xr.DataArray(
        data,
        dims=[
            "set_technologies",
            "set_capacity_types",
            "set_location",
            "set_time_steps_yearly",
        ],
        coords=coords,
    )


def test_node_capex_tech_axis_restricts_to_the_axis_technologies():
    axis = Axis("DE_nuclear", NODE_CAPEX_TECH, ("DE",), None, technologies=("nuclear",))
    value = MGA._design_axis_terms(
        _node_capex_stub(), axis, None, None, capex=_capex_data_array_multi_tech()
    )
    assert float(value.sum(skipna=True)) == pytest.approx(60.0)  # 10+20+30, pv excluded


def test_node_capex_axis_without_technology_restriction_sums_every_technology():
    axis = Axis("DE", NODE_CAPEX, ("DE",), None)
    value = MGA._design_axis_terms(
        _node_capex_stub(), axis, None, None, capex=_capex_data_array_multi_tech()
    )
    assert float(value.sum(skipna=True)) == pytest.approx(360.0)  # 60 nuclear + 300 pv


def test_year_indices_in_period_is_boundary_inclusive():
    year_indices = [0, 1, 2, 3]
    real_years = [2021, 2025, 2030, 2035]
    assert _year_indices_in_period((2021, 2030), year_indices, real_years) == [0, 1, 2]
    assert _year_indices_in_period((2025, 2025), year_indices, real_years) == [1]


def test_year_indices_in_period_is_empty_when_period_covers_no_model_year():
    year_indices = [0, 1]
    real_years = [2024, 2025]
    assert _year_indices_in_period((2030, 2035), year_indices, real_years) == []


# ------------------------------------------------------------ config validation


def test_valid_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "oracle",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            "oracle": {"tolerance": 0.1, "max_min": {"use_bigM": True}},
        }
    )


def test_valid_node_capex_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "oracle",
            "axes": {
                "node_capex": ["DE", {"benelux": ["BE", "NL", "LU"]}],
                "node_capex_periods": {
                    "nodes": ["DE"],
                    "periods": [[2021, 2030], [2031, 2040]],
                },
                "node_capex_by_technology": {
                    "nodes": ["DE"],
                    "technology_groups": [{"renewables": ["pv", "wind"]}],
                },
            },
        }
    )


def test_valid_sampling_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "sampling",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            "sampling": {"tolerance_prob": 0.9, "n_samples": 500},
        }
    )


def test_valid_bbo_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "bbo",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            "bbo": {"tolerance_prob": 0.9, "max_function_evaluations": 500},
        }
    )


def test_valid_batch_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "batch",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            "batch": {
                "tolerance_prob": 0.9,
                "batch_size": 8,
                "strategy_mode": "sampling",
                "convergence_mode": "ci",
                "n_workers": 4,
            },
        }
    )


@pytest.mark.parametrize("mode", ["sampling", "bbo"])
def test_units_normalisation_accepted_in_sampling_and_bbo_modes(mode):
    validate_config(
        {
            "epsilon": 0.1,
            "mode": mode,
            "normalisation": "units",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            mode: {"tolerance_prob": 0.9},
        }
    )


def test_units_normalisation_rejected_in_oracle_mode():
    with pytest.raises(ValueError, match="normalisation='units'"):
        validate_config(
            {
                "epsilon": 0.1,
                "mode": "oracle",
                "normalisation": "units",
                "axes": {"technologies": ["nuclear"], "include_cost": True},
                "oracle": {"tolerance": 0.1},
            }
        )


@pytest.mark.parametrize("mode", ["sampling", "bbo", "batch"])
def test_minmax_normalisation_accepted_outside_oracle_mode(mode):
    validate_config(
        {
            "epsilon": 0.1,
            "mode": mode,
            "normalisation": "minmax",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            mode: {"tolerance_prob": 0.9},
        }
    )


def test_minmax_normalisation_rejected_in_oracle_mode():
    with pytest.raises(ValueError, match="normalisation='minmax'"):
        validate_config(
            {
                "epsilon": 0.1,
                "mode": "oracle",
                "normalisation": "minmax",
                "axes": {"technologies": ["nuclear"], "include_cost": True},
                "oracle": {"tolerance": 0.1},
            }
        )


def test_unknown_normalisation_value_is_rejected():
    with pytest.raises(ValueError, match="Unknown MGA normalisation"):
        validate_config({"mode": "sampling", "normalisation": "absolute"})


@pytest.mark.parametrize(
    "cfg",
    [
        {"epsilonn": 0.1},  # top-level typo
        {"axes": {"technolgies": []}},  # axes typo
        {"oracle": {"tolerance": 0.1, "max_iter": 10}},  # oracle typo
        {"oracle": {"max_min": {"milp_options": {}}}},  # renamed key
        {"sampling": {"tolerance_probb": 0.9}},  # sampling typo
        {"bbo": {"tolerance_probb": 0.9}},  # bbo typo
        {"batch": {"batch_sizee": 8}},  # batch typo
        {"axes": {"node_capex_periods": {"preiods": []}}},  # nested typo
        {"axes": {"node_capex_by_technology": {"technolgy_groups": []}}},  # nested typo
    ],
)
def test_unknown_config_keys_are_rejected(cfg):
    with pytest.raises(ValueError, match="Unknown MGA config key"):
        validate_config(cfg)


# ------------------------------------------------------------------ oracle mode


def test_oracle_mode_rejects_non_kkt_milp_formulation():
    pytest.importorskip("pyoNearOpt")

    class _StubMGA:
        design_axes = ["dummy_axis"]  # only thing checked before 

    with pytest.raises(ValueError, match="dual_bilinear"):
        run_oracle_mode(
            _StubMGA(), {"tolerance": 0.1, "max_min": {"formulation": "dual_bilinear"}}
        )


# ----------------------------------------------------------- support-function modes


def test_sampling_mode_requires_exploration_axes():
    pytest.importorskip("pyoNearOpt")

    class _StubMGA:
        design_axes = []  # no axes configured

    with pytest.raises(ValueError, match="no exploration axes configured"):
        run_sampling_mode(_StubMGA(), {"tolerance_prob": 0.9})


def test_bbo_mode_requires_exploration_axes():
    pytest.importorskip("pyoNearOpt")
    pytest.importorskip("pypop7")

    class _StubMGA:
        design_axes = []  # no axes configured

    with pytest.raises(ValueError, match="no exploration axes configured"):
        run_bbo_mode(_StubMGA(), {"tolerance_prob": 0.9})


def test_batch_mode_requires_exploration_axes():
    pytest.importorskip("pyoNearOpt")
    # batch_ORACLE imports pypop7 unconditionally at module level.
    pytest.importorskip("pypop7")

    class _StubMGA:
        design_axes = []  # no axes configured

    with pytest.raises(ValueError, match="no exploration axes configured"):
        run_batch_mode(_StubMGA(), {"tolerance_prob": 0.9})


# ------------------------------------------------------- solve_direction / support_function


def test_solve_direction_builds_objective_solves_and_reports_support_value():
    calls = {"objective_args": None, "postprocess_label": None}

    class _StubModel:
        def add_objective(self, obj, sense, overwrite):
            calls["objective_args"] = (obj, sense, overwrite)

    class _StubMGA:
        n_z = 2
        scale = np.array([2.0, 4.0])
        axes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
        model = _StubModel()

        def axis_expression(self, axis):
            return {"a": 10.0, "b": 20.0}[axis.name]

        def _solve_and_postprocess(self, label):
            calls["postprocess_label"] = label

        def current_point_norm(self):
            return np.array([0.5, 0.25])

    direction = np.array([1.0, -1.0])
    z_feas, support_value = MGA.solve_direction(_StubMGA(), direction, "iter1_0")

    assert calls["postprocess_label"] == "iter1_0"
    obj, sense, overwrite = calls["objective_args"]
    assert sense == "max" and overwrite is True
    assert obj == pytest.approx(1.0 / 2.0 * 10.0 + (-1.0) / 4.0 * 20.0)
    assert z_feas.tolist() == [0.5, 0.25]
    assert support_value == pytest.approx(direction @ z_feas)


def test_solve_direction_rejects_wrong_direction_shape():
    class _StubMGA:
        n_z = 2

    with pytest.raises(ValueError, match=r"direction shape"):
        MGA.solve_direction(_StubMGA(), np.array([1.0]), "label")


def test_support_function_wraps_solve_direction_with_iter_label():
    calls = {"labels": []}

    class _StubMGA:
        _iter_count = 3

        def solve_direction(self, direction, label):
            calls["labels"].append(label)
            return np.array([0.1]), 0.42

    stub = _StubMGA()
    z_feas, support_value = MGA.support_function(stub, np.array([1.0]))

    assert calls["labels"] == ["supf_iter_3"]
    assert stub._iter_count == 4
    assert support_value == 0.42


# ------------------------------------------------------------- weights mode


def test_build_weighted_objective_sums_weighted_axis_expressions():
    class _StubMGA:
        z_names = ["nuclear", "DE_2021_2030"]
        axes = [
            SimpleNamespace(name="nuclear"),
            SimpleNamespace(name="DE_2021_2030"),
        ]

        def axis_expression(self, axis):
            return {"nuclear": 10.0, "DE_2021_2030": 5.0}[axis.name]

    obj = MGA._build_weighted_objective(
        _StubMGA(), {"nuclear": 2.0, "DE_2021_2030": -1.0}
    )

    assert obj == pytest.approx(2.0 * 10.0 + (-1.0) * 5.0)


def test_build_weighted_objective_skips_axes_without_an_explicit_weight():
    class _StubMGA:
        z_names = ["nuclear", "pv"]
        axes = [SimpleNamespace(name="nuclear"), SimpleNamespace(name="pv")]

        def axis_expression(self, axis):
            return {"nuclear": 10.0, "pv": 99.0}[axis.name]

    obj = MGA._build_weighted_objective(_StubMGA(), {"nuclear": 3.0})

    assert obj == pytest.approx(30.0)


def test_build_weighted_objective_rejects_unknown_axis_name():
    class _StubMGA:
        z_names = ["nuclear", "pv"]
        axes = [SimpleNamespace(name="nuclear"), SimpleNamespace(name="pv")]

    with pytest.raises(ValueError, match=r"unknown axis name.*wind"):
        MGA._build_weighted_objective(_StubMGA(), {"wind": 1.0})


def test_build_weighted_objective_rejects_empty_weights():
    class _StubMGA:
        z_names = ["nuclear"]
        axes = [SimpleNamespace(name="nuclear")]

    with pytest.raises(ValueError, match=r"no \(known\) weights"):
        MGA._build_weighted_objective(_StubMGA(), {})


def test_run_iteration_solves_and_postprocesses_with_weighted_objective():
    calls = {"objective_args": None, "postprocess_label": None}

    class _StubModel:
        def add_objective(self, obj, sense, overwrite):
            calls["objective_args"] = (obj, sense, overwrite)

    class _StubMGA:
        model = _StubModel()

        def _build_weighted_objective(self, weights):
            assert weights == {"nuclear": 1.0}
            return "the-objective"

        def _solve_and_postprocess(self, label):
            calls["postprocess_label"] = label

    MGA.run_iteration(_StubMGA(), {"nuclear": 1.0}, 3)

    assert calls["objective_args"] == ("the-objective", "min", True)
    assert calls["postprocess_label"] == "mga_iter_3"


# -------------------------------------------------------------- supplied bounds


def _mga_with_design_axes(*names):
    """A bare MGA stand-in with only design_axes, enough for bounds parsing."""
    axes = [Axis(n, TECH_CAPACITY, (n,), "power") for n in names]
    return SimpleNamespace(design_axes=axes)


def test_supplied_bounds_are_read_in_axis_order():
    mga = _mga_with_design_axes("nuclear", "pv")
    lower, upper = MGA._read_supplied_bounds(
        mga, {"pv": [1.0, 9000.0], "nuclear": [0.0, 110.0]}
    )
    assert lower.tolist() == [0.0, 1.0]
    assert upper.tolist() == [110.0, 9000.0]


@pytest.mark.parametrize(
    "bounds",
    [
        {},  # missing every axis
        {"nuclear": [0.0, 110.0]},  # missing pv
        {"nuclear": [0.0, 110.0], "pv": [0.0, 1.0], "typo": [0.0, 1.0]},
        "vmm-typo",  # not a dict
    ],
)
def test_supplied_bounds_must_cover_design_axes_exactly(bounds):
    mga = _mga_with_design_axes("nuclear", "pv")
    with pytest.raises(ValueError):
        MGA._read_supplied_bounds(mga, bounds)


# ----------------------------------------------------------- row normalisation


def test_normalise_rows_gives_unit_rows_and_keeps_the_half_spaces():
    A = np.array([[3.0, 0.0], [0.0, 743303.0], [-3.0, 4.0]])
    b = np.array([3.0, 743303.0, 5.0])
    point = np.array([0.5, 0.25])
    An, bn = normalise_rows(A, b)

    assert np.allclose(np.linalg.norm(An, axis=1), 1.0)
    # membership is unchanged: same sign of the residual, row by row
    assert np.array_equal(A @ point <= b, An @ point <= bn)
    assert np.allclose(bn[:2], 1.0)  # z <= 1 in both rows


# ------------------------------------------------------------- coordinate maps


def test_norm_phys_round_trip():
    scale = np.array([110.0, 9834.0, 3.0e8])
    offset = np.array([0.0, 0.0, 1.25e9])
    norm = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]])
    assert np.allclose(
        phys_to_norm(norm_to_phys(norm, scale, offset), scale, offset), norm
    )


def test_units_normalisation_scale_offset_is_identity():
    # normalisation="units" sets scale=1, offset=0 for design axes, so
    # normalised and physical coordinates coincide.
    scale = np.ones(3)
    offset = np.zeros(3)
    phys = np.array([[0.0, 12.5, 110.0], [3.0, 0.0, 42.0]])
    assert np.array_equal(norm_to_phys(phys, scale, offset), phys)
    assert np.array_equal(phys_to_norm(phys, scale, offset), phys)


def test_minmax_normalisation_maps_near_optimal_range_to_unit_interval():
    # normalisation="minmax" sets scale=upper-lower, offset=lower for design
    # axes, so the near-optimal minimum maps to 0 and the maximum to 1 --
    # unlike "relative", this holds even when the minimum is well above 0.
    lower = np.array([50.0, 0.0])
    upper = np.array([150.0, 20.0])
    scale, offset = upper - lower, lower
    assert np.allclose(phys_to_norm(np.array([lower, upper]), scale, offset), [[0.0, 0.0], [1.0, 1.0]])
    midpoint = lower + 0.5 * (upper - lower)
    assert np.allclose(phys_to_norm(midpoint, scale, offset), [0.5, 0.5])


def test_cost_axis_maps_optimum_to_zero_and_budget_to_one():
    c_star, epsilon = 1000.0, 0.1
    scale, offset = np.array([epsilon * c_star]), np.array([c_star])
    assert norm_to_phys(np.array([0.0]), scale, offset)[0] == c_star
    assert norm_to_phys(np.array([1.0]), scale, offset)[0] == 1100.0


def test_coordinate_map_rejects_wrong_axis_count():
    with pytest.raises(ValueError):
        norm_to_phys(np.zeros(2), np.ones(3), np.zeros(3))


# ------------------------------------------------------------------ npz schema


def _polytope() -> Polytope:
    return Polytope(
        A=np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        b=np.array([0.0, 1.0, 1.0, 1.0]),
        X=np.array([[0.1, 0.2, 0.0], [1.0, 0.5, 0.4]]),
        names=["nuclear", "biomass", "net_present_cost"],
        kinds=[TECH_CAPACITY, CARRIER_IMPORT, TOTAL_COST],
        units=["gigawatt", "gigawatt_hour", "megaEuro"],
        scale=np.array([110.0, 777923.0, 1.25e8]),
        offset=np.array([0.0, 0.0, 1.25e9]),
        bounds_phys=np.array([[0.0, 110.0], [0.0, 777923.0], [1.25e9, 1.3750e9]]),
        z_star_phys=np.array([11.0, 155584.6, 1.25e9]),
        c_star=1.25e9,
        epsilon=0.1,
        convergence_threshold=0.1,
        converged=False,
        final_gap=0.26,
        n_initial_rows=4,
        point_origin=["z_star", "max:nuclear"],
        meta={"axes": [{"name": "nuclear"}], "normalisation": "affine"},
        run={"formulation": "dual_bilinear", "iterations_done": 7},
    )


def test_polytope_npz_round_trip(tmp_path):
    original = _polytope()
    path = tmp_path / "polytope_test.npz"
    save_polytope(path, original)
    loaded = load_polytope(path)

    for field in ("A", "b", "X", "scale", "offset", "bounds_phys", "z_star_phys"):
        assert np.allclose(getattr(loaded, field), getattr(original, field))
    assert loaded.names == original.names
    assert loaded.kinds == original.kinds
    assert loaded.units == original.units
    assert loaded.point_origin == original.point_origin
    assert loaded.meta == original.meta
    assert loaded.run == original.run
    assert loaded.n_initial_rows == 4
    assert loaded.converged is False
    assert loaded.c_star == original.c_star


def test_loaded_polytope_exposes_axis_roles(tmp_path):
    path = tmp_path / "polytope_test.npz"
    save_polytope(path, _polytope())
    loaded = load_polytope(path)

    assert loaded.n_axes == 3
    assert loaded.cost_col == 2
    assert loaded.design_cols == [0, 1]
    assert loaded.design_names == ["nuclear", "biomass"]
    assert np.allclose(loaded.to_norm(loaded.to_phys(loaded.X)), loaded.X)


def test_loader_reports_every_missing_schema_key(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, A=np.eye(2), b=np.ones(2))
    with pytest.raises(KeyError, match="missing schema keys"):
        load_polytope(path)


# ------------------------------------------------------------- physical units


def _units_series(index_names, tuples, unit_strings):
    index = pd.MultiIndex.from_tuples(tuples, names=index_names)
    return pd.Series(unit_strings, index=index, dtype=str)


# Level names as ZEN-garden builds them for the unit series: the dimensions'
# documentation names, not the set names the variables are indexed by.
CAPACITY_UNITS = _units_series(
    ["technology", "capacity_type", "location", "year"],
    [
        ("nuclear", "power", "DE", 2050),
        ("battery", "power", "DE", 2050),
        ("battery", "energy", "DE", 2050),
    ],
    ["gigawatt", "gigawatt", "gigawatt_hour"],
)
IMPORT_UNITS = _units_series(
    ["carrier", "node", "time_operation"],
    [("biomass", "DE", 0), ("hydrogen", "DE", 0)],
    ["gigawatt", "gigawatt"],
)
CAPEX_UNITS = _units_series(
    ["technology", "capacity_type", "location", "year"],
    [
        ("nuclear", "power", "DE", 2050),
        ("battery", "power", "CH", 2050),
    ],
    ["megaEuro", "megaEuro"],
)
CAPEX_UNITS_MULTI_TECH = _units_series(
    ["technology", "capacity_type", "location", "year"],
    [
        ("nuclear", "power", "DE", 2050),
        ("pv", "power", "DE", 2050),
    ],
    ["megaEuro", "kiloEuro"],
)


def test_tech_axis_unit_follows_the_selected_capacity_type():
    units = {"capacity_addition": CAPACITY_UNITS}
    power = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
    energy = Axis("battery", TECH_CAPACITY, ("battery",), "energy")

    assert axis_physical_unit(power, units, pint.UnitRegistry()) == "gigawatt"
    assert axis_physical_unit(energy, units, pint.UnitRegistry()) == "gigawatt_hour"


def test_carrier_axis_unit_is_annualised():
    axis = Axis("biomass", CARRIER_IMPORT, ("biomass",), None)
    unit = axis_physical_unit(axis, {"flow_import": IMPORT_UNITS}, pint.UnitRegistry())
    assert unit == "gigawatt * hour"


def test_node_capex_axis_unit_reads_the_capex_variable_unit():
    axis = Axis("DE", NODE_CAPEX, ("DE",), None)
    units = {"cost_capex_yearly": CAPEX_UNITS}
    assert axis_physical_unit(axis, units, pint.UnitRegistry()) == "megaEuro"


def test_node_capex_period_axis_unit_reads_the_capex_variable_unit():
    axis = Axis("DE_2021_2030", NODE_CAPEX_PERIOD, ("DE",), None, period=(2021, 2030))
    units = {"cost_capex_yearly": CAPEX_UNITS}
    assert axis_physical_unit(axis, units, pint.UnitRegistry()) == "megaEuro"


def test_node_capex_tech_axis_unit_is_masked_to_the_selected_technologies():
    units = {"cost_capex_yearly": CAPEX_UNITS_MULTI_TECH}
    unfiltered = Axis("DE", NODE_CAPEX, ("DE",), None)
    nuclear_only = Axis(
        "DE_nuclear", NODE_CAPEX_TECH, ("DE",), None, technologies=("nuclear",)
    )

    assert axis_physical_unit(unfiltered, units, pint.UnitRegistry()) == (
        "kiloEuro + megaEuro"
    )
    assert axis_physical_unit(nuclear_only, units, pint.UnitRegistry()) == "megaEuro"


def test_cost_axis_reads_the_cost_variable_unit():
    axis = Axis("net_present_cost", TOTAL_COST, (), None)
    units = {"net_present_cost": pd.Series(["megaEuro"], dtype=str)}
    assert axis_physical_unit(axis, units, pint.UnitRegistry()) == "megaEuro"


def test_unit_is_none_when_the_model_tracks_no_units():
    axis = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
    assert axis_physical_unit(axis, {}, pint.UnitRegistry()) is None
