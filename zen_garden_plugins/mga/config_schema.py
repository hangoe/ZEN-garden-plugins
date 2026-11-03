"""Pydantic schema of the ``plugins.mga`` config block.

ZEN-garden's plugin loader validates the user's block against the
module-level ``Config`` of ``plugin.py`` (re-exported from here) and stores
``Config.model_validate(block).model_dump()`` back into the run config. This
module is the only place that defines defaults; the drivers read the typed
fields directly.

``ConfigBase`` is strict and forbids extra keys. The schema checks structure,
types and cross-field rules; whether axis members exist in the model is
checked later, against the built model, by ``axes.build_axes``.
"""

import math
from typing import Any, Literal

from pydantic import ValidationError, field_validator, model_validator
from zen_garden.config import ConfigBase

# An axis-list entry: a member name (singleton axis) or a single-key
# {group_name: [member, ...]} dict (lumped axis).
AxisEntry = str | dict[str, list[str]]

# Initial bounds: "vmm" (solve min/max LPs) or {axis: [lower, upper]}.
InitialBounds = Literal["vmm"] | dict[str, list[float]]

Normalisation = Literal["relative", "minmax", "share", "per_axes"]


class NodeCapexCumulative(ConfigBase):
    """axes.node_capex_cumulative: per-node capex up to each until_year."""

    nodes: list[AxisEntry] | None = None
    until_years: list[int] | None = None


class RatioGroup(ConfigBase):
    """One ratio group: numerator and denominator technologies."""

    numerator: list[str]
    denominator: list[str]


class NodeCapacityRatio(ConfigBase):
    """axes.node_capacity_ratio: capacity share at snapshot years."""

    nodes: list[AxisEntry] | None = None
    years: list[int] | None = None
    ratio_groups: list[dict[str, RatioGroup]] | None = None


class NodeOutputRatio(ConfigBase):
    """axes.node_output_ratio: delivered-output share at snapshot years."""

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
    node_capex_cumulative: NodeCapexCumulative | None = None
    node_capacity_ratio: NodeCapacityRatio | None = None
    node_output_ratio: NodeOutputRatio | None = None
    node_carbon_emissions_cumulative: NodeCarbonEmissionsCumulative | None = None


class MaxMinConfig(ConfigBase):
    """plugins.mga.oracle.max_min: ORACLE's max-min MILP settings."""

    formulation: str = "kkt_milp"
    use_bigM: bool = True
    big_M: float = 1e8
    t_max: float | None = None
    solver_options: dict[str, Any] = {}
    certificate_time_limit: float = 0.0

    @field_validator("formulation")
    @classmethod
    def _only_kkt_milp(cls, value: str) -> str:
        value = value.lower()
        if value != "kkt_milp":
            raise ValueError(
                f"MGA oracle: formulation {value!r} not implemented; only "
                "'kkt_milp' is supported."
            )
        return value


class OracleConfig(ConfigBase):
    """plugins.mga.oracle (mode "oracle"); tolerance is required there."""

    tolerance: float | None = None
    max_iterations: int = 200
    initial_bounds: InitialBounds = "vmm"
    max_min: MaxMinConfig = MaxMinConfig()


class SupportFunctionConfig(ConfigBase):
    """Settings shared by the sampling, bbo and batch blocks.

    tolerance_prob is required in these modes.
    """

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


class SamplingConfig(SupportFunctionConfig):
    """plugins.mga.sampling (mode "sampling")."""


class BboConfig(SupportFunctionConfig):
    """plugins.mga.bbo (mode "bbo")."""

    max_function_evaluations: int = 2000
    n_restarts: int = 1
    optimizer_options: dict[str, Any] | None = None


class BatchConfig(SupportFunctionConfig):
    """plugins.mga.batch (mode "batch"); n_workers None means one per CPU.

    n_jobs_sampling: processes for pyoNearOpt's gap sampling (convergence
    check and candidate directions), which runs in the main process between
    worker rounds while the workers are idle; 1 keeps it serial.
    """

    use_bounding_box: bool = True
    batch_size: int = 4
    strategy_mode: str = "sampling"
    convergence_mode: str = "ci"
    max_function_evaluations: int = 2000
    n_restarts: int = 1
    bbo_enabled: bool = True
    n_workers: int | None = None
    n_jobs_sampling: int = 1

    @field_validator("n_jobs_sampling")
    @classmethod
    def _n_jobs_sampling_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"n_jobs_sampling must be >= 1, got {value!r}")
        return value


class WeightsIteration(ConfigBase):
    """One weights-mode iteration: {axis name: weight}."""

    weights: dict[str, float]


class Config(ConfigBase):
    """plugins.mga: the full MGA plugin config."""

    epsilon: float = 0.1
    mode: Literal["weights", "oracle", "sampling", "bbo", "batch"] = "weights"
    normalisation: Normalisation = "minmax"
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
        if self.normalisation in ("share", "per_axes") and (
            axes.technologies or axes.carrier_imports
        ):
            raise ValueError(
                f"MGA: normalisation={self.normalisation!r} does not support "
                "axes.technologies or axes.carrier_imports; remove them or use "
                "'relative'/'minmax'."
            )
        if self.normalisation == "share" and (
            axes.node_capacity_ratio
            or axes.node_output_ratio
            or axes.node_carbon_emissions_cumulative
        ):
            raise ValueError(
                "MGA: normalisation='share' only supports node_capex_cumulative "
                "axes (plus the cost axis)."
            )
        if self.mode == "oracle" and self.normalisation != "relative":
            raise ValueError(
                f"MGA: oracle mode requires normalisation: relative (got "
                f"{self.normalisation!r}; the default is 'minmax', so set it "
                "explicitly). See the docs, section Normalisation."
            )
        return self

    @model_validator(mode="after")
    def _required_fields_of_mode(self) -> "Config":
        if self.mode == "oracle" and self.oracle.tolerance is None:
            raise ValueError("MGA oracle: plugins.mga.oracle.tolerance must be set.")
        if self.mode in ("sampling", "bbo", "batch"):
            if getattr(self, self.mode).tolerance_prob is None:
                raise ValueError(
                    f"MGA {self.mode}: plugins.mga.{self.mode}.tolerance_prob "
                    "must be set."
                )
        return self


def _format_validation_error(exc: ValidationError) -> str:
    """Readable one-message summary of a pydantic ValidationError.

    Unknown keys are grouped per block, an unknown normalisation names the
    accepted values, and everything else is reported with its config path.
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
                "'relative', 'minmax', 'share' or 'per_axes'."
            )
        else:
            msg = error["msg"].removeprefix("Value error, ")
            path = ".".join(["plugins.mga", *loc])
            messages.append(f"{path}: {msg} (got {error['input']!r})" if loc else msg)
    for block, keys in unknown.items():
        messages.insert(0, f"Unknown MGA config key(s) in {block}: {sorted(keys)}.")
    return "\n".join(messages)


def validate_config(cfg: dict) -> Config:
    """Validate a plugins.mga block and return it as a typed ``Config``.

    Raises:
        ValueError: With the readable summary of ``_format_validation_error``,
            so callers such as ZEN-models' run_model.py can fail fast before
            the baseline solve.
    """
    try:
        return Config.model_validate(cfg)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from None
