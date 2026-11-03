"""Tests of the exploration modes: weights, oracle projection, support function."""

from types import SimpleNamespace

import numpy as np
import pytest

from zen_garden_plugins.mga.modes.support_function import (
    SupportFunction,
    solve_direction,
)
from zen_garden_plugins.mga.modes.weights import run_weights_mode, weighted_objective


class _StubModel:
    def __init__(self):
        self.objective = None

    def add_objective(self, obj, sense, overwrite):
        self.objective = (obj, sense, overwrite)


def _stub_mga(expressions, scale=None, point=None):
    labels = []
    names = list(expressions)
    return SimpleNamespace(
        z_names=names,
        axes=[SimpleNamespace(name=n) for n in names],
        n_z=len(names),
        scale=np.asarray(scale if scale is not None else np.ones(len(names))),
        model=_StubModel(),
        axis_expression=lambda axis: expressions[axis.name],
        solve_and_postprocess=labels.append,
        current_point_norm=lambda: np.asarray(point),
        labels=labels,
    )


class TestWeightsMode:
    def test_objective_sums_weighted_axis_expressions(self):
        mga = _stub_mga({"nuclear": 10.0, "DE_until_2030": 5.0})
        obj = weighted_objective(mga, {"nuclear": 2.0, "DE_until_2030": -1.0})
        assert obj == pytest.approx(2.0 * 10.0 - 5.0)

    def test_axes_without_a_weight_are_skipped(self):
        mga = _stub_mga({"nuclear": 10.0, "pv": 99.0})
        assert weighted_objective(mga, {"nuclear": 3.0}) == pytest.approx(30.0)

    def test_unknown_axis_name_is_rejected(self):
        with pytest.raises(ValueError, match=r"unknown axis name.*wind"):
            weighted_objective(_stub_mga({"nuclear": 1.0}), {"wind": 1.0})

    def test_empty_weights_are_rejected(self):
        with pytest.raises(ValueError, match=r"no \(known\) weights"):
            weighted_objective(_stub_mga({"nuclear": 1.0}), {})

    def test_each_iteration_is_minimised_and_postprocessed(self):
        mga = _stub_mga({"nuclear": 10.0})
        iterations = [SimpleNamespace(weights={"nuclear": 1.0})] * 2
        run_weights_mode(mga, iterations)
        assert mga.model.objective == (pytest.approx(10.0), "min", True)
        assert mga.labels == ["mga_iter_0", "mga_iter_1"]


class TestSupportFunction:
    def test_solve_direction_maximises_the_scaled_direction(self):
        mga = _stub_mga({"a": 10.0, "b": 20.0}, scale=[2.0, 4.0], point=[0.5, 0.25])
        direction = np.array([1.0, -1.0])
        z_feas, support_value = solve_direction(mga, direction, "iter1_0")
        obj, sense, overwrite = mga.model.objective
        assert (sense, overwrite) == ("max", True)
        assert obj == pytest.approx(10.0 / 2.0 - 20.0 / 4.0)
        assert mga.labels == ["iter1_0"]
        assert z_feas.tolist() == [0.5, 0.25]
        assert support_value == pytest.approx(direction @ z_feas)

    def test_solve_direction_rejects_a_wrong_shape(self):
        with pytest.raises(ValueError, match="direction shape"):
            solve_direction(_stub_mga({"a": 1.0, "b": 1.0}), np.array([1.0]), "x")

    def test_sequential_callback_labels_counts_and_times_its_calls(self):
        mga = _stub_mga({"a": 1.0}, point=[0.1])
        callback = SupportFunction(mga)
        callback(np.array([1.0]))
        _, support_value = callback(np.array([1.0]))
        assert mga.labels == ["supf_iter_1", "supf_iter_2"]
        assert callback.iteration == 3
        assert len(callback.times) == 2
        assert support_value == pytest.approx(0.1)


class TestOracleProjection:
    def test_registers_its_variables_for_postprocess(self):
        """Postprocess looks every solution variable up in the registry."""
        import linopy

        from zen_garden_plugins.mga.modes.oracle import OracleProjection

        lp = linopy.Model()
        x = lp.add_variables(lower=0, upper=1, name="x")
        registry = SimpleNamespace(units={}, docs={})
        mga = SimpleNamespace(
            z_names=["a"],
            axes=[SimpleNamespace(name="a")],
            n_z=1,
            scale=np.array([1.0]),
            model=lp,
            optimization_model=SimpleNamespace(variables=registry),
            axis_expression=lambda axis: 1.0 * x,
            z_star_norm=np.array([0.5]),
        )
        projection = OracleProjection(mga)
        assert registry.units == {"mga_oracle_delta": None, "mga_oracle_t": None}
        assert set(registry.docs) == {"mga_oracle_delta", "mga_oracle_t"}
        assert projection.iteration == 1
        assert len(projection.inner_points) == 1
