"""MGA (Modelling to Generate Alternatives) plugin for ZEN-garden.

Registers an ``after_solve`` handler that adds the near-optimality budget
cost <= (1 + epsilon) * C* to the solved baseline and explores the
near-optimal space in one of five modes: weights, oracle, sampling, bbo or
batch. Configured by the ``plugins.mga`` block (``Config``); see the docs
page ``docs/files/available_plugins/mga/mga.rst``.
"""

import logging

from zen_garden.plugin_system.events import Event, EventPublisher

from .config_schema import Config, validate_config
from .modes.oracle import run_oracle_mode
from .modes.support_function import run_batch_mode, run_bbo_mode, run_sampling_mode
from .modes.weights import run_weights_mode
from .near_optimal_model import MGA

__all__ = ["MGA", "Config", "run_mga", "validate_config"]

logger = logging.getLogger(__name__)


@EventPublisher.register(Event.after_solve)
def run_mga(
    *, optimization_step, scenarios, subfolder, model_name, scenario_name, param_map
):
    """Run MGA after the baseline solve.

    The arguments are what the OptimizationStep passed to its own
    Postprocess; they are reused for every MGA solve. Must not call
    ``optimization_step.write_results()``, which fires after_solve again.

    Returns:
        The summary folder (polytope modes) or None (weights mode).

    Raises:
        ValueError: For rolling-horizon or scaled runs and non-cost objectives.
    """
    # The loader stored the validated model_dump(); re-validating gives typed
    # access and also serves direct callers.
    config = validate_config(optimization_step.config.plugins.get("mga", {}))
    run_config = optimization_step.config
    if run_config.system.use_rolling_horizon:
        raise ValueError(
            "MGA does not support rolling-horizon runs: after_solve fires "
            "once per horizon step, so MGA would explore each step's myopic "
            "model separately."
        )
    if run_config.solver.use_scaling:
        raise ValueError(
            "MGA does not support solver.use_scaling: re-scaling restores "
            "only the solution values, so MGA would add physical-unit "
            "expressions to a still-scaled model."
        )
    if run_config.analysis.objective != "total_cost":
        raise ValueError(
            f"MGA defines near-optimality on the total cost, but the "
            f"baseline objective is {run_config.analysis.objective!r}."
        )
    if config.mode == "weights" and not config.iterations:
        logger.warning("MGA: weights mode without iterations; skipping.")
        return None

    logger.info(f"MGA: mode = {config.mode!r}, epsilon = {config.epsilon}")
    mga = MGA(
        optimization_step,
        epsilon=config.epsilon,
        postprocess_ctx={
            "scenarios": scenarios,
            "subfolder": subfolder,
            "model_name": model_name,
            "scenario_name": scenario_name,
            "param_map": param_map,
        },
        axes_cfg=config.axes,
        normalisation=config.normalisation,
        per_axes_capex_reference=config.per_axes_capex_reference,
    )
    mga.setup()

    result = None
    if config.mode == "weights":
        run_weights_mode(mga, config.iterations)
    elif config.mode == "oracle":
        result = run_oracle_mode(mga, config.oracle)
    elif config.mode == "sampling":
        result = run_sampling_mode(mga, config.sampling)
    elif config.mode == "bbo":
        result = run_bbo_mode(mga, config.bbo)
    else:
        result = run_batch_mode(mga, config.batch)
    logger.info("MGA: complete.")
    return result
