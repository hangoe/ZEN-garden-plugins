"""Persistent worker pool behind batch mode.

Each worker holds its own copy of the set-up ``MGA`` and solves the
directions it receives with ``solve_direction``. Workers are forked, or
spawned with Gurobi (gurobipy is not fork-safe). Spawning pickles ``MGA``
once per worker, which needs the pickling patches below; they are applied
when this module is imported, which a spawned worker does before unpickling
its arguments. See the docs, section "Batch mode".
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from .support_function import solve_direction

if TYPE_CHECKING:
    from ..near_optimal_model import MGA

logger = logging.getLogger(__name__)

# Set once per worker process by _worker_init; never touched in the parent.
_MGA: MGA | None = None


def _patch_unit_converter_for_pickling() -> None:
    """Drop the unpicklable pint registry and rebuild it after unpickling.

    ``UnitConverter._get_base_units()`` rebuilds ``ureg`` from
    ``folder_path`` alone.
    """
    import pint
    from zen_garden.input.unit_converter import UnitConverter

    if getattr(UnitConverter, "_mga_batch_pickle_patched", False):
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ureg"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ureg = pint.UnitRegistry()
        self._get_base_units()

    UnitConverter.__getstate__ = __getstate__
    UnitConverter.__setstate__ = __setstate__
    UnitConverter._mga_batch_pickle_patched = True


def _patch_linopy_model_for_pickling() -> None:
    """Drop the live solver handle; each worker's first solve creates its own."""
    from linopy.model import Model

    if getattr(Model, "_mga_batch_pickle_patched", False):
        return

    def __getstate__(self):
        state = {
            slot: getattr(self, slot) for slot in Model.__slots__ if hasattr(self, slot)
        }
        state["solver_model"] = None
        return state

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)

    Model.__getstate__ = __getstate__
    Model.__setstate__ = __setstate__
    Model._mga_batch_pickle_patched = True


def _patch_parameter_registry_for_pickling() -> None:
    """Give ParameterRegistry a class-level __setstate__.

    Without it, pickle's lookup of __setstate__ on the empty instance falls
    through to ``__getattr__`` and recurses.
    """
    from zen_garden.model.registries.parameter import ParameterRegistry

    if getattr(ParameterRegistry, "_mga_batch_pickle_patched", False):
        return

    def __setstate__(self, state):
        self.__dict__.update(state)

    ParameterRegistry.__setstate__ = __setstate__
    ParameterRegistry._mga_batch_pickle_patched = True


def _install_application_registry(folder_path, rounding_decimal_points_units) -> None:
    """Install ZEN-garden's unit registry as pint's application registry.

    Unpickled ``pint.Quantity`` objects attach to the application registry,
    which must know the custom units (e.g. "kilotCO2eq").
    """
    import pint
    from zen_garden.input.unit_converter import UnitConverter

    pint.set_application_registry(
        UnitConverter(folder_path, rounding_decimal_points_units).ureg
    )


class _RegistryBootstrap:
    """Pickle sentinel placed first in ``initargs``.

    Unpickling it calls ``_install_application_registry`` before any
    ``pint.Quantity`` inside ``mga`` is reached.
    """

    def __init__(self, folder_path, rounding_decimal_points_units):
        self._args = (folder_path, rounding_decimal_points_units)

    def __reduce__(self):
        return (_install_application_registry, self._args)


_patch_unit_converter_for_pickling()
_patch_parameter_registry_for_pickling()
_patch_linopy_model_for_pickling()


def _worker_init(_registry_bootstrap: None, mga: MGA) -> None:
    """Pool initializer: keep this worker's MGA copy."""
    global _MGA
    _MGA = mga


def _warmup_task() -> None:
    """No-op that forces every worker to start before the first timed round."""
    return None


def _solve_one(batch_idx: int, job_idx: int, direction: np.ndarray):
    """Pool task: solve one direction as iter<batch>_<job>."""
    if _MGA is None:
        raise RuntimeError("MGA batch worker: _MGA is unset; _worker_init did not run.")
    z_feas, support_value = solve_direction(
        _MGA, direction, f"iter{batch_idx}_{job_idx}"
    )
    return job_idx, z_feas, support_value


class WorkerPoolSupportFunction:
    """pyoNearOpt batch callback backed by a persistent worker pool.

    Construct it after the bounds are solved, so workers inherit a fully
    set-up ``MGA``. ``times`` holds the wall-clock time of each batch.
    """

    def __init__(self, mga: MGA, n_workers: int):
        solver_name = mga.optimization_step.config.solver.name
        start_method = "spawn" if solver_name == "gurobi" else "fork"
        logger.info(
            f"MGA batch mode: solver is {solver_name!r}, using {start_method!r} "
            f"worker start method, {n_workers} workers."
        )
        unit_converter = mga.optimization_step.unit_converter
        registry_bootstrap = _RegistryBootstrap(
            unit_converter.folder_path, unit_converter.rounding_decimal_points_units
        )
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context(start_method),
            initializer=_worker_init,
            initargs=(registry_bootstrap, mga),
        )
        # Workers start lazily; start them now so the first timed round
        # does not include the start-up cost.
        warmup_start = time.perf_counter()
        for future in [self._executor.submit(_warmup_task) for _ in range(n_workers)]:
            future.result()
        logger.info(
            f"MGA batch mode: worker pool warmed up in "
            f"{time.perf_counter() - warmup_start:.1f}s."
        )
        self._batch_idx = 0
        self.times: list[float] = []

    def __call__(self, directions: np.ndarray):
        """Solve a (k, n_z) batch of directions; returns (points, values)."""
        start = time.perf_counter()
        self._batch_idx += 1
        futures = [
            self._executor.submit(_solve_one, self._batch_idx, job_idx, direction)
            for job_idx, direction in enumerate(directions)
        ]
        # A worker's exception re-raises here and aborts the whole batch.
        results = [future.result() for future in futures]
        self.times.append(time.perf_counter() - start)
        results.sort(key=lambda r: r[0])
        points = np.vstack([z_feas for _, z_feas, _ in results])
        support_values = np.array([value for _, _, value in results], dtype=float)
        return points, support_values

    def close(self) -> None:
        """Shut the pool down."""
        self._executor.shutdown(wait=True)
