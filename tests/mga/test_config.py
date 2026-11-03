"""Tests of config_schema.py: defaults, validation and error messages."""

import pytest

from zen_garden_plugins.mga.config_schema import validate_config

TECH_AXES = {"technologies": ["nuclear"], "include_cost": True}
CAPEX_AXES = {"node_capex_cumulative": {"nodes": ["DE"], "until_years": [2030]}}
RATIO = {
    "nodes": ["DE"],
    "years": [2030],
    "ratio_groups": [{"g": {"numerator": ["pv"], "denominator": ["pv"]}}],
}
EMISSIONS = {"nodes": ["DE"], "until_years": [2030]}


class TestValidConfigs:
    @pytest.mark.parametrize(
        "cfg",
        [
            {
                "mode": "oracle",
                "normalisation": "relative",
                "axes": TECH_AXES,
                "oracle": {"tolerance": 0.1, "max_min": {"use_bigM": True}},
            },
            {
                "mode": "sampling",
                "axes": TECH_AXES,
                "sampling": {"tolerance_prob": 0.9, "n_samples": 500},
            },
            {"mode": "bbo", "axes": TECH_AXES, "bbo": {"tolerance_prob": 0.9}},
            {
                "mode": "batch",
                "axes": TECH_AXES,
                "batch": {
                    "tolerance_prob": 0.9,
                    "batch_size": 8,
                    "strategy_mode": "sampling",
                    "n_workers": 4,
                },
            },
            {
                "mode": "weights",
                "axes": {
                    **CAPEX_AXES,
                    "node_capacity_ratio": RATIO,
                    "node_output_ratio": RATIO,
                },
                "iterations": [{"weights": {"DE_until_2030": 1.0}}],
            },
        ],
    )
    def test_valid_config_passes(self, cfg):
        validate_config(cfg)

    @pytest.mark.parametrize("normalisation", ["relative", "minmax"])
    @pytest.mark.parametrize("mode", ["sampling", "bbo", "batch"])
    def test_relative_and_minmax_work_with_every_axis_kind(self, mode, normalisation):
        validate_config(
            {
                "mode": mode,
                "normalisation": normalisation,
                "axes": {**TECH_AXES, **CAPEX_AXES, "node_capacity_ratio": RATIO},
                mode: {"tolerance_prob": 0.9},
            }
        )

    def test_share_accepts_capex_cumulative_axes(self):
        validate_config(
            {
                "mode": "sampling",
                "normalisation": "share",
                "axes": {**CAPEX_AXES, "include_cost": True},
                "sampling": {"tolerance_prob": 0.9},
            }
        )

    def test_per_axes_accepts_capex_ratio_and_emissions_axes(self):
        validate_config(
            {
                "mode": "sampling",
                "normalisation": "per_axes",
                "axes": {
                    **CAPEX_AXES,
                    "node_capacity_ratio": RATIO,
                    "node_output_ratio": RATIO,
                    "node_carbon_emissions_cumulative": EMISSIONS,
                },
                "sampling": {"tolerance_prob": 0.9},
            }
        )


class TestDefaults:
    def test_schema_defaults(self):
        """The loader stores model_dump(), so every default is materialised."""
        cfg = validate_config({}).model_dump()
        assert cfg["epsilon"] == 0.1
        assert cfg["mode"] == "weights"
        assert cfg["normalisation"] == "minmax"
        assert cfg["per_axes_capex_reference"] == 15e12
        assert cfg["iterations"] == []
        assert cfg["axes"]["include_cost"] is False
        assert cfg["oracle"]["tolerance"] is None
        assert cfg["oracle"]["max_iterations"] == 200
        assert cfg["oracle"]["initial_bounds"] == "vmm"
        assert cfg["oracle"]["max_min"] == {
            "formulation": "kkt_milp",
            "use_bigM": True,
            "big_M": 1e8,
            "t_max": None,
            "solver_options": {},
            "certificate_time_limit": 0.0,
        }
        for block in ("sampling", "bbo", "batch"):
            assert cfg[block]["tolerance_prob"] is None
            assert cfg[block]["max_iterations"] == 200
            assert cfg[block]["tolerance_explore"] == 0.1
            assert cfg[block]["n_samples"] == 1000
            assert cfg[block]["alpha"] == 0.05
            assert cfg[block]["method"] == "jeffreys"
            assert cfg[block]["initial_bounds"] == "vmm"
            assert cfg[block]["seed_rng"] is None
            assert cfg[block]["track_implied_threshold"] is False
        assert cfg["sampling"]["use_bounding_box"] is False
        assert cfg["bbo"]["use_bounding_box"] is False
        assert cfg["batch"]["use_bounding_box"] is True
        assert cfg["bbo"]["max_function_evaluations"] == 2000
        assert cfg["bbo"]["n_restarts"] == 1
        assert cfg["bbo"]["optimizer_options"] is None
        assert cfg["batch"]["batch_size"] == 4
        assert cfg["batch"]["strategy_mode"] == "sampling"
        assert cfg["batch"]["convergence_mode"] == "ci"
        assert cfg["batch"]["bbo_enabled"] is True
        assert cfg["batch"]["n_workers"] is None

    def test_integers_become_floats_for_float_fields(self):
        cfg = validate_config({"epsilon": 1, "sampling": {"tolerance_explore": 1}})
        assert isinstance(cfg.epsilon, float)
        assert isinstance(cfg.sampling.tolerance_explore, float)


class TestRejectedConfigs:
    @pytest.mark.parametrize("mode", ["sampling", "bbo", "batch"])
    @pytest.mark.parametrize("normalisation", ["share", "per_axes"])
    @pytest.mark.parametrize(
        "axes", [{"technologies": ["nuclear"]}, {"carrier_imports": ["biomass"]}]
    )
    def test_share_and_per_axes_reject_tech_and_carrier_axes(
        self, mode, normalisation, axes
    ):
        with pytest.raises(ValueError, match=normalisation):
            validate_config(
                {
                    "mode": mode,
                    "normalisation": normalisation,
                    "axes": axes,
                    mode: {"tolerance_prob": 0.9},
                }
            )

    @pytest.mark.parametrize(
        "axes",
        [
            {"node_capacity_ratio": RATIO},
            {"node_output_ratio": RATIO},
            {"node_carbon_emissions_cumulative": EMISSIONS},
        ],
    )
    def test_share_rejects_non_capex_axes(self, axes):
        with pytest.raises(ValueError, match="share"):
            validate_config(
                {
                    "mode": "sampling",
                    "normalisation": "share",
                    "axes": axes,
                    "sampling": {"tolerance_prob": 0.9},
                }
            )

    @pytest.mark.parametrize("normalisation", ["minmax", "share", "per_axes"])
    def test_oracle_requires_relative(self, normalisation):
        with pytest.raises(ValueError, match="oracle mode requires normalisation"):
            validate_config(
                {
                    "mode": "oracle",
                    "normalisation": normalisation,
                    "axes": CAPEX_AXES,
                    "oracle": {"tolerance": 0.1},
                }
            )

    def test_oracle_with_the_default_normalisation_is_rejected(self):
        with pytest.raises(ValueError, match="set it explicitly"):
            validate_config({"mode": "oracle", "oracle": {"tolerance": 0.1}})

    @pytest.mark.parametrize("normalisation", ["units", "absolute"])
    def test_unknown_normalisation_is_rejected(self, normalisation):
        with pytest.raises(ValueError, match="Unknown MGA normalisation"):
            validate_config({"normalisation": normalisation})

    @pytest.mark.parametrize(
        "cfg, message",
        [
            ({"mode": "oracle", "normalisation": "relative"}, "oracle.tolerance"),
            ({"mode": "sampling"}, "sampling.tolerance_prob"),
            ({"mode": "bbo"}, "bbo.tolerance_prob"),
            ({"mode": "batch"}, "batch.tolerance_prob"),
        ],
    )
    def test_required_fields_of_the_mode(self, cfg, message):
        with pytest.raises(ValueError, match=message):
            validate_config(cfg)

    def test_only_kkt_milp_formulation_is_accepted(self):
        with pytest.raises(ValueError, match="dual_bilinear"):
            validate_config({"oracle": {"max_min": {"formulation": "dual_bilinear"}}})

    @pytest.mark.parametrize("value", [0.0, -1.0, "15e12", True, float("nan")])
    def test_invalid_per_axes_capex_reference_is_rejected(self, value):
        with pytest.raises(ValueError, match="per_axes_capex_reference"):
            validate_config({"per_axes_capex_reference": value})

    @pytest.mark.parametrize(
        "cfg",
        [
            {"epsilonn": 0.1},  # top-level typo
            {"axes": {"technolgies": []}},  # axes typo
            {"oracle": {"tolerance": 0.1, "max_iter": 10}},  # oracle typo
            {"oracle": {"max_min": {"milp_options": {}}}},  # renamed key
            {"sampling": {"tolerance_probb": 0.9}},
            {"bbo": {"tolerance_probb": 0.9}},
            {"batch": {"batch_sizee": 8}},
            {"axes": {"node_capex_cumulative": {"unti_years": []}}},
            {"axes": {"node_capacity_ratio": {"raito_groups": []}}},
            {"axes": {"node_carbon_emissions_cumulative": {"unti_years": []}}},
            {"axes": {"node_capex": ["DE"]}},  # removed axis kinds
            {"axes": {"node_capex_by_technology": {}}},
            {"axes": {"node_capex_cumulative_tech": {}}},
        ],
    )
    def test_unknown_config_keys_are_rejected(self, cfg):
        with pytest.raises(ValueError, match="Unknown MGA config key"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "cfg",
        [
            {"epsilon": "0.1"},  # strict: no string -> float coercion
            {"epsilon": 0.0},
            {"mode": "random"},
            {"axes": {"node_capex_cumulative": {"until_years": [2030.0]}}},
            {"batch": {"batch_size": 4.5}},
        ],
    )
    def test_invalid_config_values_are_rejected(self, cfg):
        with pytest.raises(ValueError):
            validate_config(cfg)
