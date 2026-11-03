"""Tests of polytope_io.py: npz round trip and coordinate maps."""

import numpy as np
import pytest

from zen_garden_plugins.mga.polytope_io import (
    CARRIER_IMPORT,
    TECH_CAPACITY,
    TOTAL_COST,
    Polytope,
    load_polytope,
    norm_to_phys,
    phys_to_norm,
    save_polytope,
)


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
        run={"mode": "sampling", "iterations_done": 7},
    )


class TestNpz:
    def test_round_trip(self, tmp_path):
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

    def test_loaded_polytope_exposes_axis_roles(self, tmp_path):
        path = tmp_path / "polytope_test.npz"
        save_polytope(path, _polytope())
        loaded = load_polytope(path)
        assert loaded.n_axes == 3
        assert loaded.cost_col == 2
        assert loaded.design_cols == [0, 1]
        assert loaded.design_names == ["nuclear", "biomass"]
        assert np.allclose(loaded.to_norm(loaded.to_phys(loaded.X)), loaded.X)

    def test_loader_reports_every_missing_schema_key(self, tmp_path):
        path = tmp_path / "incomplete.npz"
        np.savez(path, A=np.eye(2), b=np.ones(2))
        with pytest.raises(KeyError, match="missing schema keys"):
            load_polytope(path)


class TestCoordinateMaps:
    def test_round_trip(self):
        scale = np.array([110.0, 9834.0, 3.0e8])
        offset = np.array([0.0, 0.0, 1.25e9])
        norm = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]])
        assert np.allclose(
            phys_to_norm(norm_to_phys(norm, scale, offset), scale, offset), norm
        )

    def test_minmax_scale_maps_the_range_to_the_unit_interval(self):
        lower, upper = np.array([50.0, 0.0]), np.array([150.0, 20.0])
        result = phys_to_norm(np.array([lower, upper]), upper - lower, lower)
        assert np.allclose(result, [[0.0, 0.0], [1.0, 1.0]])

    def test_rejects_a_wrong_axis_count(self):
        with pytest.raises(ValueError):
            norm_to_phys(np.zeros(2), np.ones(3), np.zeros(3))
