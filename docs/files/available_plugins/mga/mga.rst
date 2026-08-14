.. _available_plugins.mga:

MGA (Modelling to Generate Alternatives)
========================================

The ``mga`` plugin explores the near-optimal space of a ZEN-garden model:
the set of designs whose total cost stays within ``(1 + epsilon)`` times the
cost optimum. It subscribes to the ``after_solve`` event and re-solves the
baseline model under a near-optimality budget in one of five modes:

* ``weights`` -- one re-solve per user-provided weight dict, minimising
  ``sum_i w_i * capacity_addition_i`` (the classical MGA recipe).
* ``oracle`` -- the ORACLE algorithm (Turan, Moret, Bardow 2026,
  `doi:10.1016/j.compchemeng.2026.109630
  <https://doi.org/10.1016/j.compchemeng.2026.109630>`_), which iteratively
  refines inner and outer polytope approximations of the near-optimal space
  until their maximal distance falls below a tolerance. Each iteration
  solves a bilevel (nearest-point) problem, which is the most informative
  choice per iteration but the most expensive.
* ``sampling`` -- pyoNearOpt's sampling support-function method. It refines
  the same inner/outer approximations, but instead of the bilevel solve it
  samples directions and probes each with a single, cheap support-function
  LP (``max direction . design``), stopping once a confidence interval on
  the fraction of well-approximated directions clears a target. Cheaper per
  iteration than oracle mode and scales better to many exploration axes, at
  the cost of a probabilistic (rather than certified) coverage guarantee.
* ``bbo`` -- the same support-function pipeline as ``sampling``, but each
  direction is instead chosen by a black-box optimiser (SHADE) searching
  for the largest gap between the inner and outer approximations, rather
  than by sampling. Can be more effective per query in high dimensions, at
  the cost of no convergence guarantee on the direction search itself
  (overall convergence is still decided by the same confidence-interval
  metric as ``sampling``).
* ``batch`` -- the same support-function pipeline as ``sampling``/``bbo``,
  but each iteration probes a *batch* of up to ``batch_size`` directions
  concurrently, via a persistent pool of worker processes (one process per
  direction, up to ``n_workers``), instead of one direction at a time. The
  LP solve, not ZEN-garden's one-time input-read/model-construction step,
  is the bottleneck a batch is meant to parallelize, so this can give a
  real, close-to-``n_workers`` speedup on multi-core hardware (e.g. an
  Euler compute node) with no change to the modelled problem.

Oracle mode and all three support-function modes (``sampling``, ``bbo``,
``batch``) produce the same kind of output: a polytope file plus one solved
design per iteration (per query, in batch mode's case).

The code lives in ``zen_garden_plugins/mga``: ``plugin.py`` (event handler
and model interface), ``axes.py`` (exploration-axis parsing),
``oracle_driver.py`` (pyoNearOpt ORACLE wiring), ``supf_driver.py``
(pyoNearOpt support-function wiring for ``sampling`` and ``bbo``,
including the bounds/initial-approximation setup batch mode also uses),
``batch_driver.py`` (pyoNearOpt ``batch_oracle`` wiring),
``parallel_solve.py`` (the worker pool batch mode solves directions
through), and ``polytope_io.py`` (the polytope npz schema with its reader
and writer).

Requirements
------------

* A ZEN-garden that provides the ``after_solve`` event (not yet in a
  released version).
* Oracle, sampling, bbo and batch modes all need **pyoNearOpt** (E. Turan,
  ETH Zurich) -- not public yet; request access and install it manually,
  e.g. ``pip install -e path/to/pyoNearOpt``. Weights mode works without
  pyoNearOpt.
* Oracle mode additionally needs **Gurobi** and a license -- its max-min
  solver is hardcoded to Gurobi. See the compatibility note in
  ``oracle_driver.py``.
* Bbo mode additionally needs pyoNearOpt's optional ``bbo`` extra
  (``pip install "pyoNearOpt[bbo]"``, which pulls in ``pypop7``) for its
  black-box optimiser. **Batch mode always needs this extra too**, even
  with ``strategy_mode: "sampling"``: pyoNearOpt's ``batch_ORACLE`` module
  imports ``pypop7``'s ``SHADE`` unconditionally at module level.
* Sampling and bbo modes need no dedicated solver license: the only
  ZEN-garden-model solve per iteration is a plain LP, using whatever solver
  the run is already configured with. Batch mode is the same, but solves
  ``n_workers`` of them concurrently, so a solver with a per-seat/floating
  license (e.g. Gurobi) needs enough concurrent tokens available -- ETH's
  Euler cluster grants a *floating* academic Gurobi license (a shared pool
  of tokens across all HPC users), not a personal single-use license, so
  ``n_workers > 1`` is fine there but may not be on a laptop with a
  personal license.
* Batch mode parallelizes solves via OS-level ``multiprocessing``: the
  default ``"fork"`` start method (used for every solver except Gurobi)
  gives each worker a private copy of the already-built model at zero
  serialization cost; only available on Linux/macOS (not Windows), so
  Euler is the native case. With ``solver.name: "gurobi"``, batch mode
  automatically switches to ``"spawn"`` instead, to sidestep gurobipy's
  fork-safety caveats.

Configuration
-------------

Activate the plugin by adding an ``mga`` entry to the ``plugins`` block of
your ``config.json``. Unknown keys anywhere in the block are rejected.

``epsilon`` (float, default 0.1)
    Near-optimality slack, e.g. 0.1 for a 10 % cost budget.

``mode`` (str, default ``"weights"``)
    ``"weights"``, ``"oracle"``, ``"sampling"``, ``"bbo"`` or ``"batch"``.

``normalisation`` (str, default ``"relative"``, oracle/sampling/bbo/batch modes)
    ``"relative"`` scales each design axis by its near-optimal maximum, so it
    reaches 1 there; this is what makes a single scalar tolerance meaningful
    across axes with different physical units. ``"units"`` instead reports
    design axes in their own physical units (scale = 1, offset = 0) -- only
    sensible when the selected axes already share comparable units and
    magnitudes, which is the user's judgement to make, not the plugin's. The
    cost axis is always normalised relative to the near-optimality budget
    regardless of this setting. Oracle mode requires the default
    ``"relative"``: its max-min MILP relies on ``big_M``/``t_max`` dominating
    axis magnitudes and a cut-validity guard sized for O(1) normalised
    coordinates, both of which assume relative normalisation; ``"units"``
    raises a config error there. This key has no effect in ``"weights"``
    mode, which never normalises axes.

``iterations`` (list of dicts, weights mode)
    One ``{"weights": {technology: weight}}`` dict per iteration.

``axes`` (dict, oracle, sampling, bbo and batch modes)
    * ``technologies``: technology axes; entries are technology names or
      single-key dicts ``{group_name: [members]}`` for lumped axes.
    * ``carrier_imports``: carrier-import axes, same entry format.
    * ``include_cost`` (bool, default false): add the total-cost axis.

``oracle`` (dict, oracle mode)
    * ``tolerance`` (float, required): convergence tolerance in normalised
      coordinates.
    * ``max_iterations`` (int, default 200).
    * ``initial_bounds``: ``"vmm"`` (default; two LPs per design axis yield
      certified bounds plus extreme designs) or a dict
      ``{axis: [lower, upper]}`` covering every design axis.
    * ``max_min`` (dict) -- settings of the max-min distance problem that
      picks each trial point and reports the metric: ``formulation``
      (``"kkt_milp"`` only -- ``"dual_bilinear"`` is not implemented and
      raises error), ``use_bigM`` (default true), ``big_M`` (default 1e8),
      ``t_max``, ``solver_options`` (Gurobi options for the max-min
      solves), and ``certificate_time_limit`` (seconds, 0 = off; one long
      max-min solve after a non-converged loop to tighten the stored
      metric).

``sampling`` (dict, sampling mode)
    * ``tolerance_prob`` (float, required): the exploration stops once the
      lower confidence bound on the fraction of well-approximated
      directions exceeds this value.
    * ``max_iterations`` (int, default 200).
    * ``initial_bounds``: same as oracle mode's ``initial_bounds``.
    * ``tolerance_explore`` (float, default 0.1): the support-function gap
      below which a direction counts as well-approximated. Its meaning
      depends on ``normalisation``: a fraction of each axis's near-optimal
      range under ``"relative"``, a raw physical-unit gap under ``"units"``.
    * ``n_samples`` (int, default 1000): directions sampled per iteration to
      estimate the confidence interval.
    * ``alpha`` (float, default 0.05): significance level of the interval.
    * ``method`` (str, default ``"jeffreys"``): confidence-interval method,
      passed to ``statsmodels.stats.proportion.proportion_confint``.
    * ``use_bounding_box`` (bool, default false): probe the axis-aligned
      directions first, before falling back to sampled directions.
    * ``seed_rng`` (int, default unset): seed for reproducible sampling.

``bbo`` (dict, bbo mode)
    * ``tolerance_prob``, ``max_iterations``, ``initial_bounds``,
      ``tolerance_explore``, ``n_samples``, ``alpha``, ``method``,
      ``seed_rng``: same as sampling mode's keys of the same name -- they
      configure the shared confidence-interval convergence check, not the
      direction search itself. ``tolerance_explore``'s meaning likewise
      depends on the top-level ``normalisation`` setting.
    * ``use_bounding_box`` (bool, default false): probe the axis-aligned
      directions first, before falling back to the black-box search.
    * ``max_function_evaluations`` (int, default 2000): evaluation budget
      of the black-box optimiser (SHADE) per restart per iteration.
    * ``n_restarts`` (int, default 1): independent optimiser restarts per
      iteration; the restart with the largest gap is kept.
    * ``optimizer_options`` (dict, default unset): extra options merged
      into the optimiser's own options dict (e.g. population size).

``batch`` (dict, batch mode)
    * ``tolerance_prob``, ``max_iterations``, ``initial_bounds``,
      ``tolerance_explore``, ``n_samples``, ``alpha``, ``method``,
      ``seed_rng``: same as sampling mode's keys of the same name.
    * ``batch_size`` (int, default 4): maximum number of directions probed
      concurrently per iteration; the actual batch may be smaller if too
      few well-separated far directions are available.
    * ``strategy_mode`` (str, default ``"sampling"``): ``"sampling"`` or
      ``"bbo"`` -- which method chooses each batch's one "main" direction
      (the rest of the batch is filled by bounding-box directions, then by
      distance-ranked random fill). Regardless of this setting, pyoNearOpt's
      optional ``bbo`` extra must be installed (see Requirements above).
    * ``convergence_mode`` (str, default ``"ci"``): ``"ci"`` stops on the
      same confidence-interval criterion as ``sampling``/``bbo``; ``"gap"``
      stops once the main direction's gap falls below
      ``tolerance_explore``; ``"both"`` tracks the CI for monitoring but
      stops only on the gap criterion.
    * ``use_bounding_box`` (bool, default true, note the different default
      from sampling/bbo): probe the axis-aligned directions first, cycling
      through them across iterations, before filling the rest of each
      batch by the configured ``strategy_mode``/random fill.
    * ``max_function_evaluations``, ``n_restarts``: same as bbo mode's keys
      of the same name; only used when ``strategy_mode: "bbo"``.
    * ``bbo_enabled`` (bool, default true): if false, disables the BBO
      main-direction search even when ``strategy_mode: "bbo"``, falling
      back to the largest-gap sampled direction instead.
    * ``n_workers`` (int, default the machine's CPU count): number of
      persistent worker processes solving directions concurrently.
      Independent of ``batch_size`` -- if ``batch_size > n_workers``,
      workers just pick up the next queued direction as they free up.

Example (oracle mode):

.. code-block:: json

    {
        "plugins": {
            "mga": {
                "mode": "oracle",
                "epsilon": 0.1,
                "axes": {
                    "technologies": [
                        "nuclear",
                        {"hydro_lump": ["reservoir_hydro", "run-of-river_hydro"]}
                    ],
                    "carrier_imports": ["biomass"],
                    "include_cost": true
                },
                "oracle": {
                    "tolerance": 0.1,
                    "max_min": {"solver_options": {"TimeLimit": 120}}
                }
            }
        }
    }

Example (sampling mode):

.. code-block:: json

    {
        "plugins": {
            "mga": {
                "mode": "sampling",
                "epsilon": 0.1,
                "axes": {
                    "technologies": [
                        "nuclear",
                        {"hydro_lump": ["reservoir_hydro", "run-of-river_hydro"]}
                    ],
                    "carrier_imports": ["biomass"],
                    "include_cost": true
                },
                "sampling": {
                    "tolerance_prob": 0.9,
                    "max_iterations": 200
                }
            }
        }
    }

Example (bbo mode):

.. code-block:: json

    {
        "plugins": {
            "mga": {
                "mode": "bbo",
                "epsilon": 0.1,
                "axes": {
                    "technologies": [
                        "nuclear",
                        {"hydro_lump": ["reservoir_hydro", "run-of-river_hydro"]}
                    ],
                    "carrier_imports": ["biomass"],
                    "include_cost": true
                },
                "bbo": {
                    "tolerance_prob": 0.9,
                    "max_iterations": 200,
                    "max_function_evaluations": 2000
                }
            }
        }
    }

Example (batch mode):

.. code-block:: json

    {
        "plugins": {
            "mga": {
                "mode": "batch",
                "epsilon": 0.1,
                "axes": {
                    "technologies": [
                        "nuclear",
                        {"hydro_lump": ["reservoir_hydro", "run-of-river_hydro"]}
                    ],
                    "carrier_imports": ["biomass"],
                    "include_cost": true
                },
                "batch": {
                    "tolerance_prob": 0.9,
                    "max_iterations": 200,
                    "batch_size": 8,
                    "strategy_mode": "sampling",
                    "n_workers": 8
                }
            }
        }
    }

Outputs
-------

Every solve is written as an ordinary ZEN-garden results folder next to the
baseline (``<model>`` is the dataset name):

.. code-block:: text

    <model>/                            baseline (written by ZEN-garden itself)
    <model>_vmm_max_<axis>/            VMM maximum LP, one per design axis
    <model>_vmm_min_<axis>/            VMM minimum LP, one per design axis
    <model>_mga_iter_<i>/              weights mode: one folder per iteration
    <model>_oracle_iter_<n>/           oracle mode: one folder per projection
                                       solve (numbering matches diagnostics.csv)
    <model>_oracle_summary/             polytope.npz + diagnostics.csv
    <model>_supf_iter_<n>/             sampling/bbo mode: one folder per
                                       support-function solve (numbering
                                       matches diagnostics.csv). The
                                       callback is shared by both modes, so
                                       the folder name is not mode-prefixed.
    <model>_iter<b>_<j>/               batch mode: one folder per queried
                                       direction (batch b, job j within that
                                       batch), written by whichever worker
                                       process solved it.
    <model>_sampling_summary/           polytope.npz + diagnostics.csv (sampling mode)
    <model>_bbo_summary/                polytope.npz + diagnostics.csv (bbo mode)
    <model>_batch_summary/              polytope.npz + diagnostics.csv (batch mode)

``polytope.npz`` holds the outer approximation, the certified inner points,
the normalisation, and per-axis metadata; the schema is documented in and
read back by ``polytope_io.py`` (``load_polytope``). Its ``convergence_threshold``
and ``final_gap`` fields are shared across modes: ``convergence_threshold`` is
the configured target (``oracle.tolerance``, a max-min distance bound, or
``sampling``/``bbo``/``batch``'s ``tolerance_prob``, a confidence-interval
target), and ``final_gap`` is the worst-case gap between the outer and inner
approximation at the end of the run (a certified max-min distance for
oracle mode, the largest observed support-function gap for
sampling/bbo/batch mode). ``diagnostics.csv`` is the exploration method's
per-iteration record (pyoNearOpt's for oracle mode; sampling/bbo mode's
records each iteration's queried point, cut and confidence-interval
progress; batch mode's records each iteration's *whole batch* of queried
directions, points, cuts and progress in one row, since one call to the
worker pool answers a full batch at once). In scenario runs, each scenario
writes its own subfolder inside the summary folder.

Batch mode's convergence/gap fields carry one caveat beyond the others:
``batch_oracle`` (unlike ``sampling``/``bbo``'s harness) has no public
re-check method, so ``converged``/``final_gap`` are read from its last
recorded history entry rather than a fresh check against the final
approximation -- accepted, documented staleness, same as the pre-refactor
check the other two modes used to have.

Limitations
-----------

* Rolling-horizon runs, scaled runs (``solver.use_scaling``), and non-cost
  objectives are rejected; see the module docstring of ``plugin.py``.
* Config errors are only reported after the baseline solve (``after_solve``
  is the only event the plugin can use).
