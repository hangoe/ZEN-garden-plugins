.. _available_plugins.mga:

MGA (Modelling to Generate Alternatives)
========================================

The ``mga`` plugin explores the near-optimal space of a ZEN-garden model:
the set of designs whose total cost stays within ``(1 + epsilon)`` times the
cost optimum. It subscribes to the ``after_solve`` event and re-solves the
baseline model under a near-optimality budget in one of four modes:

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

Oracle mode and both support-function modes (``sampling``, ``bbo``) produce
the same kind of output: a polytope file plus one solved design per
iteration.

The code lives in ``zen_garden_plugins/mga``: ``plugin.py`` (event handler
and model interface), ``axes.py`` (exploration-axis parsing),
``oracle_driver.py`` (pyoNearOpt ORACLE wiring), ``supf_driver.py``
(pyoNearOpt support-function wiring for ``sampling`` and ``bbo``), and
``polytope_io.py`` (the polytope npz schema with its reader and writer).

Requirements
------------

* A ZEN-garden that provides the ``after_solve`` event (not yet in a
  released version).
* Oracle, sampling and bbo modes all need **pyoNearOpt** (E. Turan, ETH
  Zurich) -- not public yet; request access and install it manually, e.g.
  ``pip install -e path/to/pyoNearOpt``. Weights mode works without
  pyoNearOpt.
* Oracle mode additionally needs **Gurobi** and a license -- its max-min
  solver is hardcoded to Gurobi. See the compatibility note in
  ``oracle_driver.py``.
* Bbo mode additionally needs pyoNearOpt's optional ``bbo`` extra
  (``pip install "pyoNearOpt[bbo]"``, which pulls in ``pypop7``) for its
  black-box optimiser.
* Sampling and bbo modes need no dedicated solver license: the only
  ZEN-garden-model solve per iteration is a plain LP, using whatever solver
  the run is already configured with.

Configuration
-------------

Activate the plugin by adding an ``mga`` entry to the ``plugins`` block of
your ``config.json``. Unknown keys anywhere in the block are rejected.

``epsilon`` (float, default 0.1)
    Near-optimality slack, e.g. 0.1 for a 10 % cost budget.

``mode`` (str, default ``"weights"``)
    ``"weights"``, ``"oracle"``, ``"sampling"`` or ``"bbo"``.

``iterations`` (list of dicts, weights mode)
    One ``{"weights": {technology: weight}}`` dict per iteration.

``axes`` (dict, oracle, sampling and bbo modes)
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
      below which a direction counts as well-approximated.
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
      direction search itself.
    * ``use_bounding_box`` (bool, default false): probe the axis-aligned
      directions first, before falling back to the black-box search.
    * ``max_function_evaluations`` (int, default 2000): evaluation budget
      of the black-box optimiser (SHADE) per restart per iteration.
    * ``n_restarts`` (int, default 1): independent optimiser restarts per
      iteration; the restart with the largest gap is kept.
    * ``optimizer_options`` (dict, default unset): extra options merged
      into the optimiser's own options dict (e.g. population size).

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
    <model>_sampling_summary/           polytope.npz + diagnostics.csv (sampling mode)
    <model>_bbo_summary/                polytope.npz + diagnostics.csv (bbo mode)

``polytope.npz`` holds the outer approximation, the certified inner points,
the normalisation, and per-axis metadata; the schema is documented in and
read back by ``polytope_io.py`` (``load_polytope``). Its ``convergence_threshold``
and ``final_gap`` fields are shared across modes: ``convergence_threshold`` is
the configured target (``oracle.tolerance``, a max-min distance bound, or
``sampling``/``bbo``'s ``tolerance_prob``, a confidence-interval target), and
``final_gap`` is the worst-case gap between the outer and inner approximation
at the end of the run (a certified max-min distance for oracle mode, the
largest observed support-function gap for sampling/bbo mode). ``diagnostics.csv``
is the exploration method's per-iteration record (pyoNearOpt's for oracle
mode; sampling/bbo mode's records each iteration's queried point, cut and
confidence-interval progress). In scenario runs, each scenario writes its
own subfolder inside the summary folder.

Limitations
-----------

* Rolling-horizon runs, scaled runs (``solver.use_scaling``), and non-cost
  objectives are rejected; see the module docstring of ``plugin.py``.
* Config errors are only reported after the baseline solve (``after_solve``
  is the only event the plugin can use).
