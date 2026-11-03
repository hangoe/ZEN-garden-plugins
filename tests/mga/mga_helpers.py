"""Shared helpers and small model stand-ins for the MGA unit tests.

The tests need neither a solver nor a dataset: model variables are replaced
by small xarray objects carrying the same dimension names.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden_plugins.mga.axes import build_axes
from zen_garden_plugins.mga.axis_evaluation import AxisEvaluator
from zen_garden_plugins.mga.config_schema import AxesConfig

TECHS = ["nuclear", "pv", "battery", "hydro_a", "hydro_b"]
CARRIERS = ["biomass", "hydrogen"]
NODES = ["DE", "CH", "FR", "BE", "NL", "LU"]

# A ratio group in config form, reused by several tests.
PV_SHARE = {"g": {"numerator": ["pv"], "denominator": ["pv", "nuclear"]}}


def power_type(name, members):
    """Capacity-type resolver that maps every group to "power"."""
    return "power"


def axes_of(capacity_type_of=power_type, **block):
    """build_axes on an axes config block, with the test model's names."""
    return build_axes(
        AxesConfig.model_validate(block), TECHS, CARRIERS, NODES, capacity_type_of
    )


def units_series(index_names, tuples, unit_strings):
    """A unit series as ZEN-garden builds it (MultiIndex of set names)."""
    index = pd.MultiIndex.from_tuples(tuples, names=index_names)
    return pd.Series(unit_strings, index=index, dtype=str)


def data_array(values, **coords):
    """DataArray with the given dims in keyword order."""
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=list(coords), coords=coords
    )


class FakeSolved:
    """Stand-in for a linopy variable or expression: ``.solution`` is itself."""

    def __init__(self, data):
        self.solution = data


def evaluator_stub(**attrs):
    """An AxisEvaluator without __init__, for testing its term functions."""
    evaluator = AxisEvaluator.__new__(AxisEvaluator)
    defaults = dict(
        capacity_addition=None,
        flow_import=None,
        ts_duration=None,
        capacity=None,
        cost_capex_yearly=None,
        carbon_emissions_yearly=None,
        carbon_emissions_carrier_yearly=None,
        flow_conversion_output_yearly=None,
        axis_year_indices={},
        capex_discount_factor=None,
        emissions_interval_factor=None,
        ratio_denominator={},
        capacity_mask=None,
        all_nodes=list(NODES),
        model=SimpleNamespace(variables={}),
        expressions={},
    )
    defaults.update(attrs)
    evaluator.__dict__.update(defaults)
    return evaluator
