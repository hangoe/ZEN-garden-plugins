"""Tests of initialisation.py: supplied bounds and the initial outer box."""

from types import SimpleNamespace

import numpy as np
import pytest

from zen_garden_plugins.mga.axes import Axis
from zen_garden_plugins.mga.initialisation import (
    build_initial_outer_approximation,
    initialise,
    normalise_rows,
    read_supplied_bounds,
)
from zen_garden_plugins.mga.polytope_io import TECH_CAPACITY


def _mga_with_design_axes(*names):
    return SimpleNamespace(
        design_axes=[Axis(n, TECH_CAPACITY, (n,), "power") for n in names]
    )


class TestSuppliedBounds:
    def test_are_read_in_axis_order(self):
        lower, upper = read_supplied_bounds(
            _mga_with_design_axes("nuclear", "pv"),
            {"pv": [1.0, 9000.0], "nuclear": [0.0, 110.0]},
        )
        assert lower.tolist() == [0.0, 1.0]
        assert upper.tolist() == [110.0, 9000.0]

    @pytest.mark.parametrize(
        "bounds",
        [
            {},  # missing every axis
            {"nuclear": [0.0, 110.0]},  # missing pv
            {"nuclear": [0.0, 110.0], "pv": [0.0, 1.0], "typo": [0.0, 1.0]},
            {"nuclear": [0.0], "pv": [0.0, 1.0]},  # not a pair
        ],
    )
    def test_must_cover_the_design_axes_exactly(self, bounds):
        with pytest.raises(ValueError):
            read_supplied_bounds(_mga_with_design_axes("nuclear", "pv"), bounds)


class TestOuterApproximation:
    def _stub(self, z_names, bounds_phys, z_star_norm, chains):
        bounds_phys = np.array(bounds_phys)
        return SimpleNamespace(
            n_z=len(z_names),
            z_names=z_names,
            bounds_phys=bounds_phys,
            scale=bounds_phys[:, 1],  # "relative": scale = upper, offset = 0
            offset=np.zeros(len(z_names)),
            z_star_norm=np.array(z_star_norm),
            monotone_chains=chains,
        )

    def test_adds_a_monotonicity_row_per_consecutive_pair(self):
        names = ["DE_until_2030", "DE_until_2040"]
        stub = self._stub(names, [[0.0, 100.0], [0.0, 200.0]], [0.5, 0.5], [names])
        A0, b0 = build_initial_outer_approximation(stub)
        assert A0.shape[0] == 2 * stub.n_z + 1 == stub.n_initial_rows
        assert A0[-1][0] > 0 and A0[-1][1] < 0
        assert b0[-1] == pytest.approx(0.0)

    def test_adds_no_rows_without_chains(self):
        names = ["DE_until_2030", "DE_until_2040"]
        stub = self._stub(names, [[0.0, 100.0], [0.0, 200.0]], [0.5, 0.5], [])
        A0, _ = build_initial_outer_approximation(stub)
        assert A0.shape[0] == 2 * stub.n_z

    def test_falling_emissions_at_z_star_are_allowed(self):
        """Emissions axes have no chain, so DAC-style falling emissions pass."""
        names = ["DE_until_2030", "DE_until_2040", "DE_co2_2030", "DE_co2_2040"]
        stub = self._stub(
            names,
            [[0.0, 100.0], [0.0, 200.0], [-50.0, 100.0], [-50.0, 100.0]],
            [0.5, 0.5, 0.2, -0.1],
            [names[:2]],
        )
        A0, _ = build_initial_outer_approximation(stub)
        assert A0.shape[0] == 2 * stub.n_z + 1
        assert A0[-1][2] == 0 and A0[-1][3] == 0

    def test_rejects_a_z_star_outside_the_box(self):
        stub = self._stub(["a"], [[0.0, 100.0]], [1.5], [])
        with pytest.raises(RuntimeError, match="excludes z"):
            build_initial_outer_approximation(stub)

    def test_normalise_rows_gives_unit_rows_and_keeps_the_half_spaces(self):
        A = np.array([[3.0, 0.0], [0.0, 743303.0], [-3.0, 4.0]])
        b = np.array([3.0, 743303.0, 5.0])
        point = np.array([0.5, 0.25])
        An, bn = normalise_rows(A, b)
        assert np.allclose(np.linalg.norm(An, axis=1), 1.0)
        assert np.array_equal(A @ point <= b, An @ point <= bn)
        assert np.allclose(bn[:2], 1.0)


def test_initialise_requires_a_design_axis():
    pytest.importorskip("pyoNearOpt")
    with pytest.raises(ValueError, match="no exploration axes configured"):
        initialise(SimpleNamespace(design_axes=[]), "sampling", "vmm")
