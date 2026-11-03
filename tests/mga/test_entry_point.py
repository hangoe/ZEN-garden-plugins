"""Tests of the plugin entry point and its ZEN-garden integration."""

import pickle
from types import SimpleNamespace

import pytest


def _step_stub(plugins_mga, use_rolling_horizon=False, use_scaling=False):
    return SimpleNamespace(
        config=SimpleNamespace(
            plugins={"mga": plugins_mga},
            system=SimpleNamespace(use_rolling_horizon=use_rolling_horizon),
            solver=SimpleNamespace(use_scaling=use_scaling),
            analysis=SimpleNamespace(objective="total_cost"),
        )
    )


def _run_mga(step):
    from zen_garden_plugins.mga.plugin import run_mga

    return run_mga(
        optimization_step=step,
        scenarios={"": {}},
        subfolder="",
        model_name="model",
        scenario_name="",
        param_map=None,
    )


class TestPluginInterface:
    def test_public_api_is_importable(self):
        from zen_garden_plugins.mga.plugin import MGA, Config, run_mga, validate_config
        from zen_garden_plugins.mga.polytope_io import (  # noqa: F401
            Polytope,
            load_polytope,
            norm_to_phys,
            phys_to_norm,
            save_polytope,
        )

        assert callable(run_mga) and callable(validate_config)
        assert MGA.__name__ == "MGA" and Config.__name__ == "Config"

    def test_plugin_exposes_a_config_base_schema(self):
        from zen_garden.config import ConfigBase

        from zen_garden_plugins.mga import plugin

        assert issubclass(plugin.Config, ConfigBase)

    def test_zen_garden_loader_validates_and_fills_defaults(self):
        from zen_garden.config import Config as ZenConfig
        from zen_garden.plugin_system.loader import register_plugins

        run_config = ZenConfig(
            plugins={"mga": {"mode": "batch", "batch": {"tolerance_prob": 0.9}}}
        )
        register_plugins(run_config, source_package="zen_garden_plugins")
        stored = run_config.plugins["mga"]
        assert stored["mode"] == "batch"
        assert stored["batch"]["tolerance_prob"] == 0.9
        assert stored["epsilon"] == 0.1  # default materialised by model_dump()

    def test_zen_garden_loader_rejects_unknown_keys(self):
        from zen_garden.config import Config as ZenConfig
        from zen_garden.plugin_system.loader import register_plugins

        run_config = ZenConfig(plugins={"mga": {"epsilonn": 0.1}})
        with pytest.raises(ValueError, match="epsilonn"):
            register_plugins(run_config, source_package="zen_garden_plugins")


class TestRunMgaGuards:
    def test_rejects_rolling_horizon(self):
        with pytest.raises(ValueError, match="rolling-horizon"):
            _run_mga(_step_stub({"mode": "weights"}, use_rolling_horizon=True))

    def test_rejects_scaled_models(self):
        with pytest.raises(ValueError, match="use_scaling"):
            _run_mga(_step_stub({"mode": "weights"}, use_scaling=True))

    def test_skips_weights_mode_without_iterations(self):
        assert _run_mga(_step_stub({"mode": "weights"})) is None


class TestWorkerPoolPickling:
    def test_parameter_registry_survives_pickling(self):
        """ParameterRegistry.__getattr__ recursed while unpickling."""
        from zen_garden.model.registries.parameter import ParameterRegistry

        import zen_garden_plugins.mga.modes.worker_pool  # noqa: F401 (applies patches)

        registry = object.__new__(ParameterRegistry)
        registry.__dict__.update(dict_parameters={"a": 1}, docs={})
        restored = pickle.loads(pickle.dumps(registry))
        assert restored.dict_parameters == {"a": 1}
