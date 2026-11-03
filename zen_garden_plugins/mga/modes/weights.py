"""Weights mode: one solve per weight dict, minimising sum_i w_i * axis_i."""

import logging

logger = logging.getLogger(__name__)


def run_weights_mode(mga, iterations) -> None:
    """Solve and post-process one iteration per entry, as mga_iter_<i>.

    Args:
        mga: The set-up ``MGA``.
        iterations: The ``plugins.mga.iterations`` entries.
    """
    for iter_id, iteration in enumerate(iterations):
        weights = iteration.weights
        mga.model.add_objective(
            weighted_objective(mga, weights), sense="min", overwrite=True
        )
        logger.info(f"MGA: iteration {iter_id} weights = {weights}")
        mga.solve_and_postprocess(f"mga_iter_{iter_id}")


def weighted_objective(mga, weights: dict):
    """sum_i w_i * axis_i over the axes named in ``weights``.

    Raises:
        ValueError: For unknown axis names or if no weight is given.
    """
    unknown = sorted(set(weights) - set(mga.z_names))
    if unknown:
        raise ValueError(
            f"MGA weights mode: unknown axis name(s) {unknown}; "
            f"expected one of {mga.z_names} (see the axes config block)."
        )
    terms = [
        float(weights[axis.name]) * mga.axis_expression(axis)
        for axis in mga.axes
        if axis.name in weights
    ]
    if not terms:
        raise ValueError("MGA weights mode: iteration has no (known) weights.")
    return sum(terms)
