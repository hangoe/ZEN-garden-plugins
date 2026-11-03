"""The near-optimal model: a solved ZEN-garden model plus a cost budget.

``MGA`` holds the solved baseline, the exploration axes and their
coordinates, adds the near-optimality constraint once (``setup``) and solves
and post-processes the model for every objective a mode sets. The
constraint is added exactly once because ``add_constraints`` is not
idempotent; the objective is swapped with ``add_objective(..., overwrite=True)``.
"""

import logging

import numpy as np
from zen_garden.postprocess.postprocess import Postprocess

from . import normalisation as norm
from .axes import Axis, build_axes
from .axis_evaluation import AxisEvaluator, capacity_type_mask, selected_capacity_type
from .config_schema import AxesConfig
from .polytope_io import TOTAL_COST, norm_to_phys, phys_to_norm

logger = logging.getLogger(__name__)


class MGA:
    """Near-optimal exploration on a solved ZEN-garden model.

    Attributes set by ``initialisation.solve_axis_bounds``: ``bounds_phys``,
    ``scale``, ``offset`` and ``extreme_designs``; by
    ``initialisation.build_initial_outer_approximation``: ``n_initial_rows``.
    """

    def __init__(
        self,
        optimization_step,
        epsilon,
        postprocess_ctx,
        axes_cfg: AxesConfig | None = None,
        normalisation="minmax",
        per_axes_capex_reference=15e12,
    ):
        """Read C*, build the axes and record the baseline point z*.

        Args:
            optimization_step: The solved ZEN-garden OptimizationStep.
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Postprocess arguments for every MGA solve
                (scenarios, subfolder, model_name, scenario_name, param_map).
            axes_cfg: The validated ``plugins.mga.axes`` block.
            normalisation: Convention of the design axes (see normalisation.py).
            per_axes_capex_reference: Capex reference of normalisation="per_axes".
        """
        self.optimization_step = optimization_step
        self.optimization_model = optimization_step.optimization_model
        self.model = self.optimization_model.lp_model
        self.epsilon = epsilon
        self.postprocess_ctx = postprocess_ctx
        self.normalisation = normalisation
        # Captured before MGA touches the model.
        self.c_star = self.model.objective.value

        system = optimization_step.config.system
        capacity_addition = self.model.variables["capacity_addition"]
        # "power" is system.set_capacity_types[0] by ZEN-garden convention.
        capacity_mask = capacity_type_mask(
            capacity_addition, str(system.set_capacity_types[0])
        )
        all_carriers = (
            list(self.model.variables["flow_import"].coords["set_carriers"].values)
            if "flow_import" in self.model.variables
            else []
        )
        self.all_nodes = list(system.set_nodes)
        self.axes, self.monotone_chains = build_axes(
            axes_cfg if axes_cfg is not None else AxesConfig(),
            list(capacity_addition.coords["set_technologies"].values),
            all_carriers,
            self.all_nodes,
            lambda name, members: selected_capacity_type(capacity_mask, name, members),
        )
        self.z_names = [axis.name for axis in self.axes]
        self.n_z = len(self.axes)
        self.evaluator = AxisEvaluator(
            optimization_step, self.axes, capacity_mask, self.all_nodes
        )

        self.bounds_phys = None
        self.scale = None
        self.offset = None
        self.n_initial_rows = None
        # (origin label, physical point) per extreme design of the bound LPs.
        self.extreme_designs = []
        # Read now: the bound LPs overwrite the baseline solution.
        self.z_star_phys = np.array(
            [self.axis_value(axis) for axis in self.axes], dtype=float
        )

        self.share_reference_raw = self.share_reference = None
        if normalisation == "share":
            self.share_reference_raw, self.share_reference = norm.share_references(
                self.axes, self.evaluator.share_reference_value
            )
        self.per_axes_reference: dict[str, float] = {}
        if normalisation == "per_axes":
            self.per_axes_reference = norm.per_axes_references(
                self.axes,
                per_axes_capex_reference,
                self.optimization_model.parameters.carbon_emissions_budget,
            )

    @property
    def design_axes(self) -> list[Axis]:
        """The axes whose bounds come from the model (everything but cost)."""
        return [axis for axis in self.axes if axis.kind != TOTAL_COST]

    @property
    def include_cost(self) -> bool:
        """Whether the exploration carries the total-cost axis."""
        return any(axis.kind == TOTAL_COST for axis in self.axes)

    def axis_expression(self, axis: Axis):
        """Linopy expression of one axis."""
        return self.evaluator.expression(axis)

    def axis_value(self, axis: Axis) -> float:
        """Value of one axis on the currently loaded solution."""
        return self.evaluator.value(axis)

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        self.model.add_constraints(
            self.optimization_model.expressions["total_cost"]
            <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality",
        )
        logger.info(
            f"MGA: near-optimality constraint added, cost <= "
            f"{(1 + self.epsilon) * self.c_star} (C* = {self.c_star}, "
            f"epsilon = {self.epsilon})"
        )

    # ------------------------------------------------------------------
    # coordinates
    # ------------------------------------------------------------------

    def to_norm(self, point_phys) -> np.ndarray:
        """Physical point -> normalised coordinates."""
        assert self.scale is not None, "solve_axis_bounds() must run first"
        return phys_to_norm(point_phys, self.scale, self.offset)

    def to_phys(self, point_norm) -> np.ndarray:
        """Normalised point -> physical coordinates."""
        assert self.scale is not None, "solve_axis_bounds() must run first"
        return norm_to_phys(point_norm, self.scale, self.offset)

    def current_point_norm(self) -> np.ndarray:
        """The most recent solve as a normalised point, in axis order."""
        return self.to_norm([self.axis_value(axis) for axis in self.axes])

    @property
    def z_star_norm(self) -> np.ndarray:
        """The baseline point z* in normalised coordinates."""
        return self.to_norm(self.z_star_phys)

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------

    def solve(self, label: str) -> None:
        """Solve the current model state and fail unless it is optimal."""
        self.optimization_step.solve()
        if not self.optimization_step.optimality:
            raise RuntimeError(
                f"MGA solve {label!r} failed: termination = "
                f"{self.model.termination_condition}"
            )

    def postprocess(self, label: str) -> None:
        """Write the Postprocess folder <model_name>_<label> of the solution."""
        ctx = self.postprocess_ctx
        step = self.optimization_step
        Postprocess(
            step.model_schema,
            step.unit_converter,
            step.optimization_model,
            step.scaling,
            step.time_steps,
            optimized_time_steps=step.optimized_time_steps,
            scenarios=ctx["scenarios"],
            model_name=f"{ctx['model_name']}_{label}",
            subfolder=ctx["subfolder"],
            param_map=ctx["param_map"],
        ).save_results(ctx["scenario_name"])

    def solve_and_postprocess(self, label: str) -> None:
        """Solve the current model state and write its Postprocess folder."""
        self.solve(label)
        self.postprocess(label)
