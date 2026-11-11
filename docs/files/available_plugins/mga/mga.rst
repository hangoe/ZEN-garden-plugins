.. _available_plugins.mga:

MGA (Modelling to Generate Alternatives)
========================================

The ``mga`` plugin explores the near-optimal space of a ZEN-garden model:
the set of designs whose total cost stays within ``(1 + epsilon)`` times the
cost optimum. It subscribes to the ``after_solve`` event, adds the
constraint ``total_cost <= (1 + epsilon) * C*`` to the solved baseline and
re-solves the model under different objectives in one of five modes:

* ``weights`` -- one re-solve per user-provided weight dict, minimising
  ``sum_i w_i * axis_i`` over the axes named in the ``axes`` block (the
  classical MGA recipe, for any axis kind).
* ``oracle`` -- the ORACLE algorithm (Turan, Moret, Bardow 2026,
  `doi:10.1016/j.compchemeng.2026.109630
  <https://doi.org/10.1016/j.compchemeng.2026.109630>`_). It refines an inner
  and an outer polytope approximation of the near-optimal space until their
  maximal distance falls below a tolerance. Each iteration solves a max-min
  MILP and a projection; it is the most informative choice per iteration but
  the most expensive. See `Oracle mode`_.
* ``sampling`` -- pyoNearOpt's sampling support-function method: it samples
  directions and probes each with one cheap LP (``max direction . z``), and
  stops once a confidence interval on the fraction of well-approximated
  directions clears a target. Cheaper per iteration than oracle and scales
  better to many axes, with a probabilistic instead of a certified coverage
  guarantee. See `Support-function modes`_.
* ``bbo`` -- the same pipeline as ``sampling``, but each direction is chosen
  by a black-box optimiser (SHADE) searching for the largest gap between the
  approximations.
* ``batch`` -- the same pipeline, but each iteration probes a batch of up to
  ``batch_size`` directions concurrently in a persistent worker pool. See
  `Batch mode`_.

Oracle and the three support-function modes ("polytope modes") write a
polytope file plus one solved design per query.

Code layout
-----------

The code lives in ``zen_garden_plugins/mga``:

.. code-block:: text

    plugin.py              entry point: Config re-export, run_mga (guards, dispatch)
    config_schema.py       pydantic Config: the only place with defaults; all validation
    axes.py                Axis type, config -> ordered list of axes, axis units
    axis_evaluation.py     model-coupled axis maths (expressions, values, references)
    near_optimal_model.py  MGA: baseline, axes, cost budget, solve + Postprocess
    normalisation.py       scale/offset per normalisation convention
    initialisation.py      shared first step of the polytope modes: bounds (VMM),
                           initial inner points, initial outer box
    artifacts.py           polytope.npz / diagnostics.csv of the polytope modes
    modes/weights.py       weights mode
    modes/oracle.py        oracle mode: projection model, ORACLE loop, certificate
    modes/support_function.py  sampling, bbo and batch modes
    modes/worker_pool.py   batch-mode worker pool and its pickling patches
    polytope_io.py         polytope npz schema, reader and writer (numpy only)

Dependencies run one way: ``plugin`` builds the ``MGA``
(``near_optimal_model`` -> ``axis_evaluation`` -> ``axes``) and hands it to
a mode; the modes use ``initialisation`` and ``artifacts``.
``polytope_io`` imports nothing from the plugin, so analysis scripts can use
it without ZEN-garden.

Requirements
------------

* ZEN-garden v3 (3.1.1 or later) with the ``after_solve`` event, which is
  not yet in a released version (use the ``feature/mga`` branch). The event
  fires at the end of ``OptimizationStep.write_results`` and hands the
  plugin the solved ``OptimizationStep``.
* The polytope modes need **pyoNearOpt** (E. Turan, ETH Zurich), which is not
  public yet; install it manually, e.g. ``pip install -e path/to/pyoNearOpt``.
  Weights mode works without it.
* Oracle mode additionally needs **Gurobi** and a license (see `Oracle mode`_).
* Bbo and batch mode need pyoNearOpt's optional ``bbo`` extra
  (``pip install "pyoNearOpt[bbo]"``, which pulls in ``pypop7``). Batch mode
  needs it even with ``strategy_mode: "sampling"``, because pyoNearOpt's
  ``batch_ORACLE`` imports ``SHADE`` at module level.
* Sampling, bbo and batch mode solve plain LPs with the run's own solver.
  Batch mode solves ``n_workers`` of them at once, so a solver with
  per-seat licenses needs enough concurrent tokens (Euler's floating
  academic Gurobi license has them; a personal single-use license may not).

Configuration
-------------

Activate the plugin with an ``mga`` entry in the ``plugins`` block of
``config.yaml``. ZEN-garden validates the block against the pydantic
``Config`` when it loads the plugin: unknown keys are rejected, values must
have the documented type (strict validation, e.g. ``"0.1"`` is not a float),
and all defaults are filled in.

``epsilon`` (float, default 0.1)
    Near-optimality slack, e.g. 0.1 for a 10 % cost budget. Must be positive.

``mode`` (str, default ``"weights"``)
    ``"weights"``, ``"oracle"``, ``"sampling"``, ``"bbo"`` or ``"batch"``.

``normalisation`` (str, default ``"minmax"``)
    ``"relative"``, ``"minmax"``, ``"share"`` or ``"per_axes"``; see
    `Normalisation`_. Oracle mode requires ``"relative"``, which must be set
    explicitly because it is not the default. No effect in weights mode.

``per_axes_capex_reference`` (float, default ``15e12``)
    The reference (in the model's capex unit) that ``node_capex_cumulative``
    axes divide by under ``normalisation: per_axes``. A deliberate
    simplification: pick a round number that puts the capex axes in a
    sensible range.

``iterations`` (list, weights mode)
    One ``{"weights": {axis_name: weight}}`` entry per iteration. Keys must
    be axis names from the ``axes`` block; axes without a weight contribute
    nothing, and an unknown name raises an error listing the valid names.
    Weights apply to the axes in their own physical units.

``axes`` (dict, all modes)
    The exploration axes, see `Axes`_.

``oracle`` (dict, oracle mode)
    * ``tolerance`` (float, required): convergence tolerance (max-min
      distance in normalised coordinates).
    * ``max_iterations`` (int, default 200).
    * ``initial_bounds``: ``"vmm"`` (default; two LPs per design axis give
      certified bounds plus extreme designs) or ``{axis: [lower, upper]}``
      covering every design axis.
    * ``max_min`` (dict), the max-min problem that picks each trial point:
      ``formulation`` (only ``"kkt_milp"``), ``use_bigM`` (default true),
      ``big_M`` (default 1e8), ``t_max`` (default unset),
      ``solver_options`` (Gurobi options), ``certificate_time_limit``
      (seconds, default 0 = off).

``sampling`` (dict, sampling mode)
    * ``tolerance_prob`` (float, required): stop once the lower confidence
      bound on the fraction of well-approximated directions exceeds it.
    * ``max_iterations`` (int, default 200).
    * ``initial_bounds``: as for oracle mode.
    * ``tolerance_explore`` (float, default 0.1): the support-function gap,
      in normalised coordinates, below which a direction counts as
      well-approximated. Its meaning depends on ``normalisation``.
    * ``n_samples`` (int, default 1000): directions sampled per check.
    * ``alpha`` (float, default 0.05): significance level of the interval.
    * ``method`` (str, default ``"jeffreys"``): interval method of
      ``statsmodels.stats.proportion.proportion_confint``.
    * ``use_bounding_box`` (bool, default false): probe the axis-aligned
      directions first.
    * ``seed_rng`` (int, default unset): seed for reproducible sampling.
    * ``track_implied_threshold`` (bool, default false): each check also
      reports the smallest ``tolerance_explore`` that would already satisfy
      ``tolerance_prob``, as column ``implied_threshold_for_tolerance`` of
      ``diagnostics.csv`` (NaN where no sampled gap reaches it).

``bbo`` (dict, bbo mode)
    * All keys of ``sampling``; they configure the convergence check.
    * ``max_function_evaluations`` (int, default 2000): SHADE's evaluation
      budget per restart and iteration.
    * ``n_restarts`` (int, default 1): optimiser restarts per iteration; the
      largest gap is kept.
    * ``optimizer_options`` (dict, default unset): extra SHADE options.

``batch`` (dict, batch mode)
    * All keys of ``sampling``, but ``use_bounding_box`` defaults to true
      (the axis-aligned directions are cycled through across iterations).
    * ``batch_size`` (int, default 4): maximum directions per iteration.
    * ``strategy_mode`` (str, default ``"sampling"``): ``"sampling"`` or
      ``"bbo"``, the method choosing each batch's main direction; the rest
      is filled by bounding-box and distance-ranked random directions.
    * ``convergence_mode`` (str, default ``"ci"``): ``"ci"`` stops on the
      confidence interval, ``"gap"`` once the main direction's gap falls
      below ``tolerance_explore``, ``"both"`` tracks the interval but stops
      on the gap.
    * ``max_function_evaluations``, ``n_restarts``: as for bbo mode, used
      with ``strategy_mode: "bbo"``.
    * ``bbo_enabled`` (bool, default true): false falls back to the
      largest-gap sampled direction even with ``strategy_mode: "bbo"``.
    * ``n_workers`` (int, default the CPU count): worker processes. If
      ``batch_size > n_workers``, workers pick up queued directions.
    * ``n_jobs_sampling`` (int, default 1): processes for pyoNearOpt's gap
      sampling (the ``n_samples`` convergence-check directions), which runs
      in the main process between worker rounds while the workers wait.
      With many axes and points this bookkeeping, not the model solves, can
      dominate the run time; values > 1 spread it over the otherwise idle
      cores (each process needs a few hundred MB). The BBO search for the
      main direction stays serial.

Axes
----

An axis is one coordinate of the explored space. Axis lists take member
names (one axis each) or single-key dicts ``{group_name: [members]}`` (one
axis for the summed group). Group names must not reuse a technology, carrier
or node name, a member may appear in only one axis of a list, and axis names
must be unique across all blocks. The column order is: technology, carrier,
cumulative-capex, capacity-ratio, output-ratio and cumulative-emissions
axes, each in config order, then the cost axis.

``technologies``
    Summed ``capacity_addition`` of the members over all locations and
    years. Storage technologies (more than one active capacity type) count
    their energy capacity, all others their power capacity
    (``system.set_capacity_types[0]``); lumping storage with non-storage
    technologies is rejected because it would mix units.

``carrier_imports``
    Duration-weighted annual import ``sum tau_t * flow_import`` of the
    member carriers. With ``carrier_imports_years: [y, ...]`` it becomes one
    axis per (carrier group, year), named ``<group>_<year>``, holding the
    import of that model year only; otherwise it sums over all model years.

``node_capex_cumulative`` (``nodes``, ``until_years``)
    Annualised ``cost_capex_yearly`` of the member nodes over every model
    year up to and including ``until_year``; one axis per (node group,
    until_year), named ``<group>_until_<year>``. Each year's capex is
    discounted and interval-expanded exactly like ZEN-garden's
    ``constraint_net_present_cost`` (factor
    ``sum_{i<n} (1/(1+r))^(interval*(y-y0)+i)``, n = 1 in the last year), so
    the axis is on the same basis as the cost axis. Capex only (no opex or
    carbon cost); transport technologies drop out because their capex sits on
    edges, not nodes. These axes get `Monotonicity`_ rows.

``node_capacity_ratio`` (``nodes``, ``years``, ``ratio_groups``)
    Installed ``capacity`` of a numerator technology group divided by a
    denominator group's, at the member nodes and one snapshot model year;
    one axis per (node group, ratio group, year), named
    ``<group>_<ratio_group>_<year>``. ``ratio_groups`` entries are
    ``{name: {"numerator": [...], "denominator": [...]}}``; a numerator
    technology may also be in its own denominator. Both groups must share
    one capacity type. The year must be a model year.

    **The denominator is frozen** at its value on the cost-optimal baseline
    and not recomputed per explored design: a ratio of two decision-variable
    sums is nonlinear and could not be an LP objective or projection term.
    This is a good proxy when the denominator group's total is
    demand-driven (e.g. all road-transport capacity). Because the numerator
    is installed capacity, the cost slack can buy idle capacity, so the
    near-optimal maxima can be far above what demand needs; use
    ``node_output_ratio`` for the share of delivered energy.

``node_output_ratio`` (same keys)
    As ``node_capacity_ratio``, but over the annual ``flow_conversion_output``
    of the groups, so idle capacity does not move the axis. Members must be
    conversion technologies, and one ratio group must have one output unit
    (e.g. passenger and freight transport go into separate groups; checked
    when ZEN-garden's unit tracking is on).

``node_limit_utilisation`` (``nodes``, ``years``, ``technology_groups``)
    Annual ``flow_conversion_input`` of the reference carrier divided by
    ``max(capacity_limit, existing_capacities) * hours``. Members must be
    conversion technologies sharing one input reference carrier. For
    ``carbon_storage`` this is the use of the injection rate, not of a reservoir.

``node_carbon_emissions_cumulative`` (``nodes``, ``until_years``)
    Annualised ``carbon_emissions_technology`` plus
    ``carbon_emissions_carrier`` of the member nodes up to ``until_year``,
    i.e. the node's share of ``carbon_emissions_annual``, which the carbon
    budget limits. Interval-expanded like the capex axes but not
    discounted. Carrier emissions count at the node where the carrier is
    imported or exported (e.g. natural gas at its import node, not where it
    is burned); technology emissions of transport technologies (on edges)
    are not counted. Values can be negative (negative-emission technologies
    such as DAC, or carbon storage), so these axes are neither clipped at 0
    nor monotone. Use
    distinct group names from ``node_capex_cumulative`` (e.g. ``DE`` and
    ``DE_co2``), since both name their axes ``<group>_until_<year>``.

``include_cost`` (bool, default false)
    Add the total-cost axis (the model's net present cost).

The frozen ratio denominators and each axis's physical unit are recorded in
the polytope's ``axis_meta_json``.

Normalisation
-------------

The polytope modes work in normalised coordinates,
``physical = normalised * scale + offset`` per axis, so a single tolerance is
meaningful across axes with different units. The cost axis always uses
``scale = epsilon * C*``, ``offset = C*``: 0 is the cost optimum and 1 the
budget. For the design axes, with near-optimal bounds ``[lower, upper]``
from VMM or ``initial_bounds``:

``relative``
    ``scale = upper``, ``offset = 0``: each axis reaches 1 at its
    near-optimal maximum, and its lower bound sits at ``lower/upper``. Needs
    a positive maximum. The only convention oracle mode accepts, because its
    max-min MILP (``big_M``, ``t_max``) and cut guard assume O(1) coordinates
    with offset 0.

``minmax`` (default)
    ``scale = upper - lower``, ``offset = lower``: each axis spans exactly
    ``[0, 1]``, so ``tolerance_explore`` is the same fraction of every axis's
    own near-optimal range. Needs a positive span, but allows negative
    values (e.g. net-negative emissions).

``share`` (``node_capex_cumulative`` axes and cost only)
    ``scale`` = a fixed reference total, ``offset = 0``: the baseline
    capex summed over all nodes and the full horizon (ignoring each axis's
    own ``until_year``), rounded to one significant figure (``12.3e6`` ->
    ``1e7``). All capex axes of a run share this total. Because the
    reference is shared instead of each axis's own range, a fixed
    ``tolerance_explore`` is met sooner for axes whose range is a small part
    of the total: exploration deliberately concentrates on the larger
    regions. Use ``minmax`` to explore every axis to the same relative
    depth. The rounded and the raw total are stored per axis as
    ``share_reference_total`` and ``share_reference_total_raw``.

``per_axes`` (capex-cumulative, ratio, emissions and cost axes)
    ``scale`` = a reference per axis kind, ``offset = 0``:
    ``per_axes_capex_reference`` for ``node_capex_cumulative``, 1 for the
    ratio axes (already fractions), and the model's
    ``carbon_emissions_budget`` for ``node_carbon_emissions_cumulative``.
    Stored per axis as ``per_axes_reference``.

``share`` and ``per_axes`` reject ``technologies`` and ``carrier_imports``
axes, which have no defined reference.

Monotonicity
------------

``cost_capex_yearly`` is non-negative and each ``node_capex_cumulative``
window nests inside the next, so ``z(group, until_year_1) <=
z(group, until_year_2)`` holds for every feasible design when
``until_year_1 < until_year_2``. The initial outer approximation therefore
gets one such row per consecutive pair of a node group's axes: a free
tightening that never excludes a near-optimal design. Emissions axes get no
such rows, since negative-emission technologies can make cumulative
emissions fall as later years are added.

Initialisation
--------------

All polytope modes start the same way (``initialisation.py``):

1. **Bounds.** With ``initial_bounds: vmm``, two LPs per design axis drive
   it to its near-optimal minimum and maximum; each must be proven optimal.
   Minima of non-negative axes are clipped at 0 (solver round-off).
2. **Normalisation** from the bounds (see `Normalisation`_).
3. **Initial inner points:** the baseline z* and every VMM extreme design.
4. **Initial outer box:** the bounds plus the monotonicity rows, mapped to
   normalised coordinates and scaled to unit row length (pyoNearOpt's
   convention for cuts). z* must lie inside.

Oracle mode
-----------

``modes/oracle.py`` connects pyoNearOpt's ORACLE to the ZEN-garden model:

1. Initialise (above).
2. pyoNearOpt solves the max-min problem over the two approximations for the
   next trial point (Gurobi).
3. ``OracleProjection.find_nearest_point`` minimises the normalised
   L-infinity distance ``t`` from the trial point to the near-optimal space
   (variables ``mga_oracle_delta`` and ``mga_oracle_t``, one ZEN-garden
   solve). The duals of the projection equalities, times ``scale``, give the
   cut normal.
4. pyoNearOpt adds the point to the inner hull and the cut to the outer
   approximation.
5. After the loop, optionally one long max-min solve on the final geometry
   (``certificate_time_limit``) tightens the stored metric; it is appended
   as an extra diagnostics row.

Robustness: inexact duals (e.g. barrier without crossover) can give a cut
that would remove a known near-optimal point. Violations above ``1e-4``
relax the cut offset outward to the farthest known inner point, which is
always valid. A failed projection solve is retried once; a second failure
returns the previous inner point with distance 0, which stops ORACLE so the
artifacts are still written.

pyoNearOpt compatibility: the published package provides only the
``kkt_milp`` formulation. ``use_bigM`` defaults to true because only the
big-M encoding is sound there: with SOS1 the reported metric is the max-min
incumbent, not a valid upper bound. The solver is Gurobi because the max-min
model needs SOS1 constraints or global nonconvex solving and the certificate
reads Gurobi's dual bound.

Support-function modes
----------------------

``modes/support_function.py`` runs sampling, bbo and batch mode. Each query
maximises ``d . z`` for a normalised direction ``d`` (``solve_direction``)
and returns the point and its support value; pyoNearOpt turns them into a
new inner point and cut. pyoNearOpt checks convergence *before* each query,
so for sampling and bbo the check is repeated once on the final
approximation to include the last query. ``batch_oracle`` has no such
re-check, so batch mode reads its last recorded history entry.

Each ``diagnostics.csv`` row carries ``wall_time_seconds``: the elapsed time
of that iteration's support-function call (sequential modes) or of the whole
batch (batch mode), measured around the call itself. It is stored with the
run's own artifacts rather than derived from ZEN-garden's per-solve
``benchmarking.json``, which only times each solver call and is written
per solve folder.

Batch mode
----------

``modes/worker_pool.py`` starts ``n_workers`` processes once, each holding a
copy of the fully set-up ``MGA``; each direction of a batch is solved by one
worker and post-processed to ``<model>_iter<b>_<j>``. A worker's exception
aborts the whole batch.

Workers are forked (no copying cost; Linux/macOS only). With Gurobi they are
spawned instead, because gurobipy is not fork-safe, so ``MGA`` is pickled
once per worker. Three patches, applied when the module is imported, make
that possible without changing ZEN-garden: ``UnitConverter`` drops its pint
registry and rebuilds it from its folder after unpickling; ``linopy.Model``
drops the live solver handle (each worker's first solve creates its own);
``ParameterRegistry`` gets a class-level ``__setstate__`` so unpickling does
not recurse through its ``__getattr__``. A sentinel at the front of the
worker arguments installs ZEN-garden's unit registry as pint's application
registry before any ``pint.Quantity`` is unpickled. A warm-up task starts
all workers before the first timed batch.

Examples
--------

Weights mode:

.. code-block:: yaml

    plugins:
      mga:
        mode: weights
        epsilon: 0.1
        axes:
          technologies:
          - nuclear
          - hydro_lump: [reservoir_hydro, run-of-river_hydro]
          node_capex_cumulative:
            nodes: [DE, CH]
            until_years: [2030, 2040, 2050]
        iterations:
        - weights:
            nuclear: 1.0
        - weights:
            hydro_lump: 1.0
            DE_until_2030: 0.5

Oracle mode:

.. code-block:: yaml

    plugins:
      mga:
        mode: oracle
        epsilon: 0.1
        normalisation: relative
        axes:
          technologies: [nuclear]
          carrier_imports: [biomass]
          node_capex_cumulative:
            nodes: [DE, CH, {benelux: [BE, NL, LU]}]
            until_years: [2030, 2040, 2050]
          include_cost: true
        oracle:
          tolerance: 0.1
          max_min:
            solver_options:
              TimeLimit: 120

Batch mode with ``per_axes`` normalisation:

.. code-block:: yaml

    plugins:
      mga:
        mode: batch
        epsilon: 0.1
        normalisation: per_axes
        per_axes_capex_reference: 15000000000000.0
        axes:
          node_capex_cumulative:
            nodes: [north: [DK, 'NO', SE], south: [ES, IT, PT]]
            until_years: [2030, 2040, 2050]
          node_output_ratio:
            nodes: [north: [DK, 'NO', SE], south: [ES, IT, PT]]
            years: [2030, 2038, 2050]
            ratio_groups:
            - heating_electrification:
                numerator: [electrode_boiler, heat_pump]
                denominator: [electrode_boiler, heat_pump, natural_gas_boiler, oil_boiler]
          node_carbon_emissions_cumulative:
            nodes: [north_co2: [DK, 'NO', SE], south_co2: [ES, IT, PT]]
            until_years: [2030, 2040, 2050]
          include_cost: true
        batch:
          tolerance_prob: 0.9
          batch_size: 8
          strategy_mode: bbo
          n_workers: 8

Sampling and bbo mode take the same ``axes`` block with a ``sampling`` or
``bbo`` block holding at least ``tolerance_prob``.

Outputs
-------

Every solve is written as an ordinary ZEN-garden results folder next to the
baseline (``<model>`` is the dataset name):

.. code-block:: text

    <model>/                     baseline (written by ZEN-garden itself)
    <model>_vmm_max_<axis>/      VMM maximum LP, one per design axis
    <model>_vmm_min_<axis>/      VMM minimum LP, one per design axis
    <model>_mga_iter_<i>/        weights mode, one per iteration
    <model>_oracle_iter_<n>/     oracle mode, one per projection
    <model>_supf_iter_<n>/       sampling/bbo mode, one per query
    <model>_iter<b>_<j>/         batch mode, job j of batch b
    <model>_<mode>_summary/      polytope.npz + diagnostics.csv (polytope modes)

``polytope.npz`` holds the outer approximation, the certified inner points,
the normalisation and per-axis metadata; its schema is documented in
``polytope_io.py`` and read with ``load_polytope``. ``convergence_threshold``
is the configured target (``oracle.tolerance`` or ``tolerance_prob``) and
``final_gap`` the worst-case gap between the approximations at the end (a
certified max-min distance for oracle, the largest observed support-function
gap otherwise). ``diagnostics.csv`` is the per-iteration record of the
exploration method (one row per whole batch in batch mode). In scenario
runs, each scenario writes its own subfolder in the summary folder.

Limitations
-----------

* Rolling-horizon runs (``after_solve`` would fire once per horizon step),
  scaled runs (``solver.use_scaling``) and non-cost objectives are rejected.
* Config values are validated when ZEN-garden loads the plugin; axis names
  are checked against the model only after the baseline solve.
