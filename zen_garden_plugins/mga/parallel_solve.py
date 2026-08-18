"""Persistent forked/spawned worker pool for MGA's `batch` mode.

`MGA.solve_direction(direction, label)` is the complete one-direction unit
of work (mutate objective, solve, write a Postprocess folder, read back
the point). `multiprocessing`'s `"fork"` start method hands each worker a
free, already-solved private copy of `MGA` -- no pickling, since the
worker just inherits the parent's memory via `os.fork()`. Workers start
once, at pool construction, and are reused for the whole run.

Exception: with `solver.name == "gurobi"`, workers use `"spawn"` instead,
since gurobipy has documented fork-safety caveats. That means `mga` has to
survive a real `pickle` round trip once per spawned worker -- cheaper than
rebuilding the model from scratch, but `pint.UnitRegistry`, a solved
`linopy.Model`'s live gurobipy handle, and any stray `pint.Quantity` aren't
picklable across processes as-is. `_patch_unit_handling_for_pickling` and
`_patch_linopy_model_for_pickling` (below) teach the two classes involved to
drop/rebuild the offending state around the pickle boundary; `_RegistryBootstrap`
handles the `Quantity` case. All three run unconditionally at import time
(not just when the parent constructs the pool) -- a spawned worker
re-imports this module fresh, as its own process, before unpickling
`initargs`, so anything that must be in place *before* unpickling `mga` has
to be applied here at module scope, not from inside `__init__`.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .plugin import MGA

# Set once per worker process by _worker_init; never touched in the parent.
_MGA: "MGA | None" = None


def _patch_unit_handling_for_pickling() -> None:
    """Make `mga` (and thus `initargs`) survive `spawn`'s pickling.

    `optimization_setup.energy_system.unit_handling.ureg` is a plain
    `pint.UnitRegistry()`. pint's built-in conversion contexts (e.g.
    "spectroscopy", "boltzmann") each store a `Relation.transformation`
    closure -- a locally-defined lambda -- in `Context.funcs`, so *any*
    `UnitRegistry()` instance is unpicklable out of the box (reproduces with
    a bare `pickle.dumps(pint.UnitRegistry())`). `ureg` is reachable from
    `mga` via `optimization_setup`, so it breaks pickling `initargs=(mga,)`
    for spawned workers.

    `UnitHandling.get_base_units()` deterministically rebuilds `ureg` (plus
    `base_units`/`dim_matrix`) from `self.folder_path` alone, so it's safe to
    drop `ureg` before pickling and rebuild it after unpickling in each
    worker, rather than trying to transmit the pint registry itself.
    """
    from zen_garden.preprocess.unit_handling import UnitHandling

    if getattr(UnitHandling, "_mga_batch_pickle_patched", False):
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ureg"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.get_base_units()

    UnitHandling.__getstate__ = __getstate__
    UnitHandling.__setstate__ = __setstate__
    UnitHandling._mga_batch_pickle_patched = True


def _patch_linopy_model_for_pickling() -> None:
    """Drop the live solver handle before pickling `mga` for a spawned worker.

    After `optimization_setup.solve()`, `linopy.Model.solver_model` holds the
    live gurobipy `Model` -- a thin Python wrapper around a C pointer
    (`PyCapsule`), which cannot be serialized (or meaningfully handed to a
    different process; a Gurobi model/env is tied to the process that
    created it). `Model` is a `__slots__` class with no `__dict__`, so this
    needs its own `__getstate__`/`__setstate__` rather than a dict patch.

    Each worker's first `solve_direction()` call runs `optimization_setup
    .solve()`, which resets `solver_model = None` before building its own
    fresh solver connection -- so dropping it here (rather than rebuilding
    it, as with `ureg`) is enough; nothing reads it before that first solve.
    """
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


def _install_application_registry(folder_path, rounding_decimal_points_units) -> None:
    """Rebuild ZEN-garden's custom-unit registry and install it as pint's
    process-global default, so `pint.Quantity.__reduce__` can resolve
    custom units (e.g. "kilotCO2eq", from `unit_definitions.txt`) when a
    worker unpickles one.

    `pint.Quantity.__reduce__` never pickles the registry a quantity was
    built with -- pint's own docstring says so: "Since UnitRegistries are
    not pickled, upon unpickling the new object is always attached to the
    application registry." A worker's application registry starts out as
    pint's stock, definitions-free default, so any custom unit blows up
    with `UndefinedUnitError` the moment a `Quantity` using it is unpickled.
    """
    import pint
    from zen_garden.preprocess.unit_handling import UnitHandling

    pint.set_application_registry(
        UnitHandling(folder_path, rounding_decimal_points_units).ureg
    )


class _RegistryBootstrap:
    """First element of `initargs`: a pickle sentinel, not real payload.

    Tuple elements unpickle strictly in order, so putting this first in
    `initargs=(_RegistryBootstrap(...), mga)` guarantees
    `_install_application_registry` runs -- as a side effect of
    reconstructing *this* element -- before pickle ever reaches a
    `pint.Quantity` nested inside `mga`. `__reduce__` is pickle's hook for
    "call this function with these args to rebuild me"; here the "rebuild"
    is pure side effect, and the sentinel itself is discarded.
    """

    def __init__(self, folder_path, rounding_decimal_points_units):
        self._args = (folder_path, rounding_decimal_points_units)

    def __reduce__(self):
        return (_install_application_registry, self._args)


# Applied at import time, not lazily inside ForkedBatchSupportFunction.__init__:
# a spawned worker re-imports this module (to resolve _worker_init/_solve_one)
# *before* unpickling initargs, so these must already be in effect by then.
_patch_unit_handling_for_pickling()
_patch_linopy_model_for_pickling()


def _worker_init(_registry_bootstrap: None, mga: "MGA") -> None:
    """ProcessPoolExecutor initializer: stash this worker's MGA copy.

    `_registry_bootstrap` unpickles to plain `None` -- `_RegistryBootstrap.
    __reduce__` reconstructs this argument by *calling*
    `_install_application_registry`, whose return value (implicitly `None`)
    is what actually lands here. Its job (see `_RegistryBootstrap`) was the
    side effect during unpickling, not this value.
    """
    global _MGA
    _MGA = mga


def _warmup_task() -> None:
    """No-op submitted once per worker at pool construction, purely to force
    `ProcessPoolExecutor` to spawn (and, under `"spawn"`, unpickle `initargs`
    into) every worker up front -- see `ForkedBatchSupportFunction.__init__`.
    """
    return None


def _solve_one(batch_idx: int, job_idx: int, direction: np.ndarray):
    """Task submitted to the pool: solve one direction, tag it with its
    position in the batch so results can be re-sorted after gathering."""
    if _MGA is None:
        raise RuntimeError(
            "MGA batch worker: _MGA is unset; _worker_init did not run "
            "(ProcessPoolExecutor misconfigured?)."
        )
    label = f"iter{batch_idx}_{job_idx}"
    z_feas, support_value = _MGA.solve_direction(direction, label)
    return job_idx, z_feas, support_value


class ForkedBatchSupportFunction:
    """pyoNearOpt `BatchSupportFunction` callback backed by a persistent
    pool of worker processes, each holding its own live copy of `mga`.
    Construct after `mga`'s bounds and extreme-point LPs are solved, so
    workers inherit a fully set-up instance.
    """

    def __init__(self, mga: "MGA", n_workers: int):
        solver_name = mga.optimization_setup.solver.name
        start_method = "spawn" if solver_name == "gurobi" else "fork"
        mp_context = multiprocessing.get_context(start_method)
        logging.info(
            f"MGA batch mode: solver is {solver_name!r}, using {start_method!r} "
            f"worker start method, {n_workers} workers."
        )

        unit_handling = mga.optimization_setup.energy_system.unit_handling
        registry_bootstrap = _RegistryBootstrap(
            unit_handling.folder_path, unit_handling.rounding_decimal_points_units
        )
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
            initializer=_worker_init,
            initargs=(registry_bootstrap, mga),
        )
        # ProcessPoolExecutor spawns workers lazily, from submit() -- not at
        # construction -- so without this, the FIRST __call__ below would pay
        # for spawning every worker and (under "spawn") unpickling `mga` into
        # each one, silently inflating that round's recorded wall-clock time
        # relative to every later round. Forcing that one-time cost here,
        # before any timed call, keeps self.times comparable across rounds.
        warmup_start = time.perf_counter()
        warmup_futures = [self._executor.submit(_warmup_task) for _ in range(n_workers)]
        for future in warmup_futures:
            future.result()
        logging.info(
            f"MGA batch mode: worker pool warmed up in "
            f"{time.perf_counter() - warmup_start:.1f}s."
        )
        self._batch_idx = 0
        # Real elapsed wall-clock time per __call__ (i.e. per outer batch
        # iteration), measured directly around the concurrent submit/gather
        # below -- unlike ZEN-garden's own per-solve benchmarking.json
        # (each worker's OWN LP/MILP time only, missing model-construction/
        # I/O overhead, and per-worker rather than per-batch-round so it
        # can't be summed into a real elapsed time without guessing how much
        # the batch_size workers actually overlapped), this is the genuine
        # duration of "how long did this whole round of batch_size
        # concurrent solves take", the same quantity near_optimal_tools' own
        # docs/examples/method_comparison.ipynb gets from wrapping its
        # (sequential) support_function in a TimedCallback. batch_driver.py
        # zips this 1:1 onto iteration_history (batch_oracle.explore calls
        # this callback exactly once per iteration_history entry, in the
        # same order) as a "wall_time_seconds" diagnostics.csv column, so
        # it's saved once, atomically, with the rest of the run's own
        # artifacts -- not scattered across batch_size x n_iterations
        # separate benchmarking.json files whose write/download can fail
        # independently of whether the underlying solve succeeded (see
        # plot_mga_results.py's REAL_ELAPSED_SECONDS/_calibrate_cum_seconds
        # for the workaround this was needed for on runs before this fix).
        self.times: list[float] = []

    def __call__(self, directions: np.ndarray):
        """The `batch_support_function` callback `batch_oracle` calls once
        per iteration with `directions` of shape (k, n_explore)."""
        start = time.perf_counter()
        self._batch_idx += 1
        futures = [
            self._executor.submit(_solve_one, self._batch_idx, job_idx, direction)
            for job_idx, direction in enumerate(directions)
        ]
        # future.result() re-raises a worker's exception here, aborting the
        # whole batch -- matches batch_oracle's all-or-nothing contract.
        results = [future.result() for future in futures]
        self.times.append(time.perf_counter() - start)
        results.sort(key=lambda r: r[0])
        points = np.vstack([z_feas for _, z_feas, _ in results])
        support_values = np.array([value for _, _, value in results], dtype=float)
        return points, support_values

    def close(self) -> None:
        self._executor.shutdown(wait=True)
