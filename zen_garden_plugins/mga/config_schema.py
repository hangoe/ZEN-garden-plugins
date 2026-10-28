"""Pydantic schema of the MGA plugin's "plugins.mga" config block.

ZEN-garden v3's plugin loader validates the user's block against the
module-level `Config` class of plugin.py (re-exported from here) and stores
`Config.model_validate(block).model_dump()` back into the run config, so
every default below is materialised explicitly. Each default therefore
mirrors the `.get(key, default)` its driver applied under v2, so a block that
omits a key still behaves exactly as before.

`ConfigBase` is strict and forbids extra keys: unknown keys and wrong types
(e.g. "0.1" for a float) are rejected. The schema checks structure and types
only; whether axis members exist in the model is checked later, against the
built model, by axes.build_axis_groups.
"""

import math
from typing import Any, Literal

from pydantic import ValidationError, field_validator, model_validator
from zen_garden.config import ConfigBase

# An axis-list entry: a member name (singleton axis) or a single-key
# {group_name: [member, ...]} dict (lumped axis); see axes._parse_axis_list.
AxisEntry = str | dict[str, list[str]]


class NodeCapexCumulative(ConfigBase):
    """axes.node_capex_cumulative: per-node capex up to each until_year."""

    nodes: list[AxisEntry] | None = None
    until_years: list[int] | None = None


class NodeCapexByTechnology(ConfigBase):
    """axes.node_capex_by_technology: per-node capex of a technology group."""

    nodes: list[AxisEntry] | None = None
    technology_groups: list[AxisEntry] | None = None


class NodeCapexCumulativeTech(ConfigBase):
    """axes.node_capex_cumulative_tech: both restrictions at once."""

    nodes: list[AxisEntry] | None = None
    until_years: list[int] | None = None
    technology_groups: list[AxisEntry] | None = None


class RatioGroup(ConfigBase):
    """One node_capacity_ratio group: numerator/denominator technologies."""

    numerator: list[str]
    denominator: list[str]


class NodeCapacityRatio(ConfigBase):
    """axes.node_capacity_ratio: capacity share at snapshot years."""

    nodes: list[AxisEntry] | None = None
    years: list[int] | None = None
    ratio_groups: list[dict[str, RatioGroup]] | None = None


class NodeCarbonEmissionsCumulative(ConfigBase):
    """axes.node_carbon_emissions_cumulative: emissions up to each until_year."""

    nodes: list[AxisEntry] | None = None
    until_years: list[int] | None = None


class AxesConfig(ConfigBase):
    """plugins.mga.axes: the exploration axes (see axes.py)."""

    technologies: list[AxisEntry] | None = None
    carrier_imports: list[AxisEntry] | None = None
    include_cost: bool = False
    node_capex: list[AxisEntry] | None = None
    node_capex_cumulative: NodeCapexCumulative | None = None
    node_capex_by_technology: NodeCapexByTechnology | None = None
    node_capex_cumulative_tech: NodeCapexCumulativeTech | None = None
    node_capacity_ratio: NodeCapacityRatio | None = None
    node_carbon_emissions_cumulative: NodeCarbonEmissionsCumulative | None = None


class MaxMinConfig(ConfigBase):
    """plugins.mga.oracle.max_min: ORACLE's max-min MILP settings."""

    formulation: str = "kkt_milp"
    use_bigM: bool = True
    big_M: float = 1e8
    t_max: float | None = None
    solver_options: dict[str, Any] = {}
    certificate_time_limit: float = 0.0


# Initial bounds: "vmm" (solve min/max LPs) or {axis: [lower, upper]}.
InitialBounds = Literal["vmm"] | dict[str, list[float]]


class OracleConfig(ConfigBase):
    """plugins.mga.oracle (mode "oracle")."""

    # Required in oracle mode; checked by oracle_driver.run_oracle_mode.
    tolerance: float | None = None
    max_iterations: int = 200
    initial_bounds: InitialBounds = "vmm"
    max_min: MaxMinConfig = MaxMinConfig()


class _SupportFunctionConfig(ConfigBase):
    """Settings shared by the sampling, bbo and batch blocks."""

    # Required in these modes; checked by the drivers.
    tolerance_prob: float | None = None
    max_iterations: int = 200
    tolerance_explore: float = 0.1
    n_samples: int = 1000
    alpha: float = 0.05
    method: str = "jeffreys"
    initial_bounds: InitialBounds = "vmm"
    use_bounding_box: bool = False
    seed_rng: int | None = None
    track_implied_threshold: bool = False


class SamplingConfig(_SupportFunctionConfig):
    """plugins.mga.sampling (mode "sampling")."""


class BboConfig(_SupportFunctionConfig):
    """plugins.mga.bbo (mode "bbo")."""

    max_function_evaluations: int = 2000
    n_restarts: int = 1
    optimizer_options: dict[str, Any] | None = None


class BatchConfig(_SupportFunctionConfig):
    """plugins.mga.batch (mode "batch")."""

    use_bounding_box: bool = True
    batch_size: int = 4
    strategy_mode: str = "sampling"
    convergence_mode: str = "ci"
    max_function_evaluations: int = 2000
    n_restarts: int = 1
    bbo_enabled: bool = True
    # None -> os.cpu_count(), resolved in batch_driver.run_batch_mode.
    n_workers: int | None = None


class WeightsIteration(ConfigBase):
    """One weights-mode iteration: {axis name: weight}."""

    weights: dict[str, float]


class Config(ConfigBase):
    """plugins.mga: the full MGA plugin config."""

    epsilon: float = 0.1
    mode: Literal["weights", "oracle", "sampling", "bbo", "batch"] = "weights"
    normalisation: Literal["relative", "minmax", "units", "share", "per_axes"] = (
        "relative"
    )
    per_axes_capex_reference: float = 15e12
    iterations: list[WeightsIteration] = []
    axes: AxesConfig = AxesConfig()
    oracle: OracleConfig = OracleConfig()
    sampling: SamplingConfig = SamplingConfig()
    bbo: BboConfig = BboConfig()
    batch: BatchConfig = BatchConfig()

    @field_validator("epsilon")
    @classmethod
    def _epsilon_positive(cls, value: float) -> float:
        if not value > 0:
            raise ValueError(f"MGA epsilon must be positive, got {value!r}")
        return value

    @field_validator("per_axes_capex_reference")
    @classmethod
    def _reference_positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "MGA: per_axes_capex_reference must be a positive finite "
                f"number, got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _normalisation_fits_axes_and_mode(self) -> "Config":
        axes = self.axes
        has_tech_or_carrier_axes = bool(axes.technologies or axes.carrier_imports)
        if self.normalisation == "share" and has_tech_or_carrier_axes:
            raise ValueError(
                "MGA: normalisation='share' only supports capex axes "
                "(node_capex / node_capex_cumulative / node_capex_by_technology / "
                "node_capex_cumulative_tech); remove axes.technologies/"
                "axes.carrier_imports or use a different normalisation."
            )
        if self.normalisation == "per_axes" and (
            axes.technologies
            or axes.carrier_imports
            or axes.node_capex
            or axes.node_capex_by_technology
            or axes.node_capex_cumulative_tech
        ):
            raise ValueError(
                "MGA: normalisation='per_axes' only supports node_capex_cumulative, "
                "node_capacity_ratio and node_carbon_emissions_cumulative axes (plus "
                "the cost axis); remove axes.technologies/axes.carrier_imports/"
                "axes.node_capex/axes.node_capex_by_technology/"
                "axes.node_capex_cumulative_tech or use a different normalisation."
            )
        if self.mode == "oracle" and self.normalisation != "relative":
            raise ValueError(
                f"MGA: normalisation={self.normalisation!r} is not supported in "
                "oracle mode -- oracle's max-min MILP relies on big_M/t_max "
                "dominating axis magnitudes and a cut-validity guard sized for "
                "O(1) normalised coordinates anchored at offset=0, both of "
                "which assume 'relative' normalisation. Use sampling or bbo "
                "mode for 'units'/'minmax'/'share'/'per_axes'."
            )
        return self


def _format_validation_error(exc: ValidationError) -> str:
    """Readable one-message summary of a pydantic ValidationError.

    Unknown keys are grouped per block ("Unknown MGA config key(s) in
    plugins.mga.axes: [...]"), an unknown normalisation names the accepted
    values, and everything else is reported with its dotted config path.
    """
    unknown: dict[str, list[str]] = {}
    messages = []
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        if error["type"] == "extra_forbidden":
            block = ".".join(["plugins.mga", *loc[:-1]])
            unknown.setdefault(block, []).append(loc[-1])
        elif loc == ["normalisation"] and error["type"] == "literal_error":
            messages.append(
                f"Unknown MGA normalisation: {error['input']!r}. Expected "
                f"'relative', 'minmax', 'units', 'share' or 'per_axes'."
            )
        else:
            msg = error["msg"].removeprefix("Value error, ")
            path = ".".join(["plugins.mga", *loc])
            messages.append(f"{path}: {msg} (got {error['input']!r})" if loc else msg)
    for block, keys in unknown.items():
        messages.insert(0, f"Unknown MGA config key(s) in {block}: {sorted(keys)}.")
    return "\n".join(messages)


def validate_config(cfg: dict) -> Config:
    """Validate a plugins.mga block; return it as a typed `Config`.

    Raises ValueError (not pydantic's ValidationError) with the readable
    summary of `_format_validation_error`, so callers such as ZEN-models'
    run_model.py can fail fast before the baseline solve.
    """
    try:
        return Config.model_validate(cfg)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from None
