.. _available_plugins.mga:

MGA (Modelling to Generate Alternatives)
========================================

The ``mga`` plugin explores the near-optimal space of a ZEN-garden model:
the set of designs whose total cost stays within ``(1 + epsilon)`` times the
cost optimum. It subscribes to the ``after_solve`` event and re-solves the
baseline model under a near-optimality budget in one of two modes:

* ``weights`` -- one re-solve per user-provided weight dict, minimising
  ``sum_i w_i * capacity_addition_i`` (the classical MGA recipe).
* ``oracle`` -- the ORACLE algorithm (Turan, Moret, Bardow 2026,
  `doi:10.1016/j.compchemeng.2026.109630
  <https://doi.org/10.1016/j.compchemeng.2026.109630>`_), which iteratively
  refines inner and outer polytope approximations of the near-optimal space
  until their maximal distance falls below a tolerance. The result is a
  certified map of the entire space (a polytope file plus one solved design
  per iteration).

The code lives in ``zen_garden_plugins/mga``: ``plugin.py`` (event handler
and model interface), ``axes.py`` (exploration-axis parsing),
``oracle_driver.py`` (pyoNearOpt wiring), and ``polytope_io.py`` (the
polytope npz schema with its reader and writer).

Requirements
------------

* A ZEN-garden that provides the ``after_solve`` event (not yet in a
  released version).
* Oracle mode additionally needs:

  * **Gurobi** and a license -- the max-min solver is hardcoded to Gurobi.
  * **pyoNearOpt** (E. Turan, ETH Zurich) -- not public yet; request access
    and install it manually, e.g. ``pip install -e path/to/pyoNearOpt``.
    The plugin runs with the base (published) package; see the compatibility
    note in ``oracle_driver.py``. Weights mode works without pyoNearOpt.

Configuration
-------------

Activate the plugin by adding an ``mga`` entry to the ``plugins`` block of
your ``config.json``. Unknown keys anywhere in the block are rejected.

``epsilon`` (float, default 0.1)
    Near-optimality slack, e.g. 0.1 for a 10 % cost budget.

``mode`` (str, default ``"weights"``)
    ``"weights"`` or ``"oracle"``.

``iterations`` (list of dicts, weights mode)
    One ``{"weights": {technology: weight}}`` dict per iteration.

``axes`` (dict, oracle mode)
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

Outputs
-------

Every solve is written as an ordinary ZEN-garden results folder next to the
baseline (``<model>`` is the dataset name):

.. code-block:: text

    <model>/                    baseline (written by ZEN-garden itself)
    <model>_vmm_max_<axis>/    VMM maximum LP, one per design axis
    <model>_vmm_min_<axis>/    VMM minimum LP, one per design axis
    <model>_mga_iter_<i>/      weights mode: one folder per iteration
    <model>_oracle_iter_<n>/   oracle mode: one folder per projection solve
                               (numbering matches diagnostics.csv)
    <model>_oracle_summary/     polytope.npz + diagnostics.csv

``polytope.npz`` holds the outer approximation, the certified inner points,
the normalisation, and per-axis metadata; the schema is documented in and
read back by ``polytope_io.py`` (``load_polytope``). ``diagnostics.csv`` is
pyoNearOpt's per-iteration record. In scenario runs, each scenario writes
its own subfolder inside the summary folder.

Limitations
-----------

* Rolling-horizon runs, scaled runs (``solver.use_scaling``), and non-cost
  objectives are rejected; see the module docstring of ``plugin.py``.
* Config errors are only reported after the baseline solve (``after_solve``
  is the only event the plugin can use).
