"""Persistent forked/spawned worker pool for MGA's `batch` mode.

`MGA.solve_direction(direction, label)` is the complete one-direction unit
of work (mutate objective, solve, write a Postprocess folder, read back
the point). `multiprocessing`'s `"fork"` start method hands each worker a
free, already-solved private copy of `MGA` -- no pickling, since the
worker just inherits the parent's memory via `os.fork()`. Workers start
once, at pool construction, and are reused for the whole run.

Exception: with `solver.name == "gurobi"`, workers use `"spawn"` instead,
since gurobipy has documented fork-safety caveats. `linopy.Model` is plain
default-picklable, so a spawned worker's one-time pickle at startup is
still far cheaper than rebuilding the model from scratch.
"""

from __future__ import annotations

import logging
import multiprocessing
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


def _worker_init(mga: "MGA") -> None:
    """ProcessPoolExecutor initializer: stash this worker's MGA copy."""
    global _MGA
    _MGA = mga


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
        if start_method == "spawn":
            _patch_unit_handling_for_pickling()
        mp_context = multiprocessing.get_context(start_method)
        logging.info(
            f"MGA batch mode: solver is {solver_name!r}, using {start_method!r} "
            f"worker start method, {n_workers} workers."
        )

        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
            initializer=_worker_init,
            initargs=(mga,),
        )
        self._batch_idx = 0

    def __call__(self, directions: np.ndarray):
        """The `batch_support_function` callback `batch_oracle` calls once
        per iteration with `directions` of shape (k, n_explore)."""
        self._batch_idx += 1
        futures = [
            self._executor.submit(_solve_one, self._batch_idx, job_idx, direction)
            for job_idx, direction in enumerate(directions)
        ]
        # future.result() re-raises a worker's exception here, aborting the
        # whole batch -- matches batch_oracle's all-or-nothing contract.
        results = [future.result() for future in futures]
        results.sort(key=lambda r: r[0])
        points = np.vstack([z_feas for _, z_feas, _ in results])
        support_values = np.array([value for _, _, value in results], dtype=float)
        return points, support_values

    def close(self) -> None:
        self._executor.shutdown(wait=True)
