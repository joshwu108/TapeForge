=========
TapeForge
=========

TapeForge is an offline research platform for optimal trade execution. It
turns historical market events into features, replays execution policies in
a deterministic CPU simulator, and records comparable experiments. A Triton
backend accelerates feature aggregation; a Gymnasium adapter exposes the
reference simulator to reinforcement-learning libraries.

The research objective is to measure whether GPU feature computation and,
eventually, batched GPU simulation reduce training time while preserving
numerical and temporal semantics. No learned policy or GPU execution
simulator is implemented yet.

Architecture
============

.. code-block:: text

   Historical aggregate trades / synthetic events
       -> validated integer-scaled data
       -> packed window offsets
       -> NumPy reference / Triton feature aggregation
       -> bucket tapes and chronological splits
       -> sampled execution episodes
       -> baseline policy or Gymnasium adapter
       -> CPU simulator: fills, fees, inventory, rewards
       -> trajectories, paired reports, reproducible run records

The CPU references define numerical behavior. The GPU feature engine uses
one-pass reductions for short buckets and two-pass integer partial sums for
large buckets. A separate streaming CPU aggregator checks the same feature
semantics across ordered input chunks.

The execution simulator currently uses a synthetic, volume-capped bucket
VWAP fill model and, in its primary configuration, a synthetic terminal
settlement of whatever is unfilled at the horizon. It is not an exchange
simulator or a live trading system. There is no order routing, passive-order
queue model, market impact, or market response to the policy's actions, and
neither the fills nor the settlement represent real execution.

Status
======

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Component
     - Current state
   * - Data and features
     - Aggregate-trade ingestion, checksum-verified acquisition, synthetic
       generation, fixed-width offsets, independent CPU references, and
       ordered-chunk streaming aggregation implemented and CPU-tested.
   * - GPU feature engine
     - One-pass and two-pass kernels, dispatch, and prepared execution
       implemented. The 82-test GPU suite passed on a Tesla T4 at commit
       ``c5d0f80`` (PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0).
       This verification is specific to that revision's GPU-facing code.
   * - CPU execution research
     - Reference simulator (version 2: terminal settlement as the primary
       rule, the legacy residual rule named explicitly), day tapes,
       chronological splits, episode sampling, train-only normalization with
       versioned statistics files, three baselines plus a zero-action
       diagnostic, experiment CLI, reporting, replay and cross-run
       sensitivity tables implemented and CPU-tested.
   * - RL interface
     - Optional Gymnasium CPU adapter tested against direct simulator calls
       and the baseline policies; environment checker command; fixed
       evaluation manifests; external-policy evaluation writing
       baseline-comparable records; checkpoint provenance schema. No PPO or
       SAC learner is included.
   * - Performance evaluation
     - Kernel, stage, and pageable/pinned transfer benchmarks implemented.
       Recorded tuning sweeps and pinned-stream overlap remain pending.
   * - Planned research
     - Learned execution policies, GPU-vectorized environments, quote-aware
       fills, and robustness/fidelity experiments remain unimplemented.
       ``model.py`` and ``benchmarks/train.py`` are stubs.

CI checks CPU behavior; it has no CUDA runner. Hardware verification must be
repeated after GPU-facing changes. The quick T4 benchmark checked correctness
but was not recorded in ``benchmarks/results.json``. **No speedup figures are
published.** Local three-day futures smoke experiments demonstrated the CPU
pipeline under the terminal-settlement and legacy rules; they do not
establish policy superiority, out-of-sample skill, or a research value of
the terminal surcharge.

Install
=======

Requires Python 3.11 or 3.12. CPU research runs on macOS or Linux. The Triton
backend requires Linux, an NVIDIA CUDA GPU, and a compatible PyTorch/Triton
installation.

.. code-block:: console

   $ python3.12 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install -e '.[dev,torch,rl]'   # CPU development and Gymnasium
   $ python -m pip install -e '.[dev,gpu,rl]'     # Linux CUDA development

NumPy is the only required runtime dependency. CPU references, streaming,
execution simulation, and baseline experiments do not require PyTorch.
``ops.aggregate`` requires the ``torch`` extra; the environment adapter
requires ``rl``. The ``rl`` extra installs Gymnasium, not an RL trainer.

Run a baseline experiment
=========================

Start with synthetic data; no downloads or GPU are needed:

.. code-block:: console

   $ python -m tapeforge.execution.experiment run experiments/configs/smoke-synthetic.toml

The command creates ``runs/<name>-<utc>/``. Use the printed directory in
subsequent commands:

.. code-block:: console

   $ python -m tapeforge.execution.report runs/<run-dir>
   $ python -m tapeforge.execution.replay runs/<run-dir> \
       --episode validation/s1/i0 --policy twap --verify
   $ python -m tapeforge.execution.experiment compare runs/<run-a> runs/<run-b>
   $ python -m tapeforge.execution.env_check experiments/configs/smoke-synthetic.toml

The report writes JSON, CSV, and reStructuredText under ``<run-dir>/report/``.
Replay prints the visible market bucket, action, submitted quantity, fill
cap, fill price, fees, settled quantity and price, remaining inventory,
reward components, and terminal flags; ``--verify`` checks the replay
against its recorded outcome. The environment checker runs Gymnasium's
``check_env``, drives random actions through the adapter and compares every
step with a direct simulator call; it needs the ``rl`` extra.

For historical data, the futures smoke configurations use BTCUSDT aggregate
trades from 2023-12-01 through 2023-12-03: two training days, one validation
day, and no test period. They allow at most three downloads totaling 500 MiB.
They are demonstration configurations, not a research evaluation protocol.
One configuration per terminal-rule setting keeps the sampled episodes
identical across settings so the sensitivity tool can line them up:

.. code-block:: console

   $ python -m tapeforge.execution.datasets fetch --market futures-um \
       --symbol BTCUSDT --date 2023-12-01 --date 2023-12-02 --date 2023-12-03
   $ for s in 0 10 25 50; do python -m tapeforge.execution.experiment run \
       experiments/configs/smoke-futures-btcusdt-2023-12-settlement-${s}bps.toml; done
   $ python -m tapeforge.execution.experiment run \
       experiments/configs/smoke-futures-btcusdt-2023-12-legacy-perold.toml
   $ python -m tapeforge.execution.sensitivity runs/<run-0bps> runs/<run-10bps> \
       runs/<run-25bps> runs/<run-50bps> runs/<run-legacy> --out runs/<sensitivity-dir>

``smoke-futures-btcusdt-2023-12.toml`` is the 25 bps settlement configuration
with downloads enabled. The surcharge values are experimental settings for a
sensitivity comparison, not calibrated market-impact estimates; 25 bps is the
initial smoke setting, not a chosen research coefficient.

Archives live under ``data/<market>/aggTrades/`` and are checked against the
source's published SHA-256 checksum. Verified cached archives can be reused
without network access. Spot and futures are separate markets; a run names
one market and symbol. Check source licensing before redistributing data.

Experiment contract
===================

The TOML configuration (schema 2) specifies data source, dates, split roles,
seeds, episodes per day, target fractions, policies, and simulator settings.
Unknown keys are rejected, and ``template.terminal_rule`` must be named:
``"terminal_settlement"`` with an explicit ``terminal_surcharge_bps``, or
the legacy ``"residual_perold"``. Schema 1 configurations, run records and
manifests, which predate the terminal rule, are refused with a migration
hint rather than reinterpreted. Research configurations need explicit dates;
copying the smoke configuration does not create a held-out experiment.

* Each policy receives the same seeded episode draws with fresh state.
* Splits are chronological, with embargo and episode-window purge. History,
  fills, and terminal marks must remain within the selected split segment.
* Targets are sized from volume before the episode, not future volume.
* Observation normalization is fitted on training episodes only and frozen
  for evaluation. The adapter never fits statistics automatically.
* Test-role evaluation requires ``--include-test``. Keep that period out of
  debugging and parameter selection.

A run directory contains ``manifest.json`` and
``records/<policy>-<role>-seed<n>.json``. The manifest records resolved
configuration, data checksums, revision and dirty state, simulator version
and assumptions, split metadata, episode IDs, timings, and peak memory.
Result files are written atomically; repeated runs get distinct directories.
Externally evaluated policies add records and manifest entries to the same
directory (see "Evaluating a policy").

Reports separate four outcomes: completed (ordinary fills reached the
target), settled (the target was reached only through the terminal
settlement), incomplete (legacy rule, residual left) and truncated (the tape
ran out). They include per-policy and per-day shortfall split into the
fills-only part and the terminal term, ordinary and settlement fees, the
risk charge, completion by fills next to final completion, the settled
fraction with a ``none`` / ``light`` / ``heavy`` settlement label, tail
costs, and paired differences on identical episode IDs. Truncated episodes
have fills-only shortfall and are not pooled with terminated episodes.
Day-cluster bootstrap intervals require at least five days; smaller samples
report insufficient data for uncertainty estimation. The zero-action policy
is labelled a diagnostic.

**Completion must be reported alongside cost.** Under terminal settlement
every terminated episode reaches the target by construction, so a 100 %
final completion rate says nothing by itself; read it next to completion by
fills, the settled fraction and the settlement cost. Under the legacy rule
with ``residual_mu = 0``, unfilled quantity incurs only the terminal
price-movement charge and avoids execution fees, so a policy that leaves
more unfilled can have lower reported shortfall; that is not evidence of
better execution. Surcharge and risk settings must be chosen before
optimizing a learned policy.

Execution model
===============

An episode attempts to buy or sell a target quantity within a fixed horizon.
``EpisodeSpec`` defines the rules, ``MarketSlice`` supplies bucket data,
``State`` tracks progress, and ``StepResult`` records one transition.
``run_episode`` coordinates a policy callable and the simulator;
``run_policy`` samples and evaluates multiple episodes without training.

The supplied smoke configurations use:

* 300 one-second decisions; target fractions of 2%, 5%, or 10% of preceding
  300-bucket volume, with the configured target cap and a one-tick floor.
* Actions in ``[0, 1]``: fraction of remaining quantity to submit, floored
  to quantity ticks.
* Fills at tick-rounded execution-bucket VWAP, capped at 10% of that bucket's
  quantity. Empty or zero-quantity buckets produce no fills.
* A 10-basis-point fee, rounded up in cash ticks; zero slippage and latency.
* Observations at decision ``t`` use buckets through ``t - 1 - data_delay``.
* An arrival price from the last valid visible bucket in the lookback.
* Terminal settlement of the quantity still unfilled at the horizon, at the
  terminal mark moved 25 basis points against the side (the smoke setting),
  with the same fee; an optional inventory-risk term.

Quantities use 1e-5 units and prices use 0.01 units. Cash ticks are the
product of price and quantity ticks. Accounting uses integers; bucket VWAP
is rounded half-up to price ticks using its integer numerator. These scales
must represent the chosen instrument exactly; ingestion rejects excess
precision instead of truncating it.

For a fill, execution cost is
``side * (notional - arrival_price * filled_quantity) + fee``.
Reward is the negative sum of execution cost, inventory risk, and the
terminal term. Completion or horizon exhaustion terminates an episode.
Missing data truncates it; the simulator does not invent a terminal mark,
a settlement or a residual charge for a truncated episode.

Terminal rules
--------------

The terminal mark is the VWAP of the first valid bucket at or after bucket
``horizon + latency``, searched over ``arrival_lookback`` buckets. Two rules
decide what happens to quantity still unfilled when the horizon ends; a
specification names one explicitly.

``terminal_settlement`` (primary research configuration). The residual is
settled in one synthetic fill at the terminal mark moved against the side
by ``terminal_surcharge_bps``: a buy pays the mark times
``(10000 + surcharge) / 10000`` rounded up to a price tick, a sell receives
the mark times ``(10000 - surcharge) / 10000`` rounded down. The settlement
pays the configured fee with the usual ceiling and updates inventory, cash,
the settled quantity, the settlement notional and the settlement fee, which
are kept apart from the ordinary fills. No opportunity-cost term is charged
in addition. The task therefore always ends with the target reached unless
the tape runs out. The settlement ignores the participation cap of the
terminal bucket and the record flags when it exceeds it; it is a boundary
condition of the task, not executable depth, impact or guaranteed liquidity.
``residual_mu`` must be zero under this rule, and ordinary
``slippage_ticks`` applies to fills only. Early completion settles nothing.

``residual_perold`` (legacy, explicit). The residual stays unfilled and is
charged ``side * (terminal_mark - arrival) * residual + residual_mu *
residual``. With ``residual_mu = 0`` the residual pays no fee and a
favourable price move rewards inaction. Simulator version 1 used this rule
for every episode; naming it reproduces those transitions.

Worked example on the reference fixture (arrival 10,000 ticks, fee 10 bps,
terminal mark 10,010, 120,000 of 300,000 ticks left after fills whose own
shortfall is 2,200,400 cash ticks):

.. list-table::
   :header-rows: 1
   :widths: 10 14 16 14 18 18

   * - Side
     - Surcharge
     - Settlement price
     - Settlement fee
     - Settlement cost
     - Total shortfall
   * - buy
     - 0 bps
     - 10,010
     - 1,201,200
     - 2,401,200
     - 4,601,600
   * - buy
     - 25 bps
     - 10,036
     - 1,204,320
     - 5,524,320
     - 7,724,720
   * - sell
     - 25 bps
     - 9,984
     - 1,198,080
     - 3,118,080
     - 4,518,480

Settlement cost is ``side * (price * 120,000 - 10,000 * 120,000) + fee``;
the sell row adds the sell-side fill shortfall of 1,400,400. Under the legacy
rule with ``residual_mu = 0`` the same buy residual costs 1,200,000 and no
fee. A zero-action buy on a terminal mark of 9,990 settles at 0 bps for
minus 3,000 cash ticks: favourable market movement can make delaying
execution beneficial, and the rule does not pretend otherwise; what it
removes is the fee avoidance. These cases are pinned by the oracle tests.

Under both rules the risk term charges ``risk_lambda`` per tick unfilled
after each step's ordinary fill, before any settlement, and the per-step
rewards of a terminated episode sum to minus the implementation shortfall
minus the risk charge.

The baselines are:

* ``twap``: divide remaining quantity by remaining decisions.
* ``immediate``: request all remaining quantity at every decision; fills
  remain capped.
* ``cap_aware_twap``: increase TWAP's submission if recent liquidity suggests
  the remaining decisions may not absorb the outstanding quantity. This is
  a heuristic, not a forecast.
* ``zero_action``: never trade. A diagnostic, not a strategy: it isolates
  the cost of the terminal rule alone and may win an episode when the
  market moves favourably.

Gymnasium interface
===================

``ExecutionEnv`` wraps the CPU simulator without duplicating accounting.
``SplitPool`` supplies reproducible sampled episodes; ``ListPool`` accepts
explicit episode entries.

* Observation: float32 vector of shape ``(10,)``; remaining/time fractions,
  side, causal mark information, and recent bucket features.
* Action: ``Box(0, 1, shape=(1,), dtype=float32)``. Invalid or non-finite
  actions raise rather than being silently clipped.
* Reward: basis points of arrival notional by default. ``info`` retains the
  exact integer reward and its components; raw-tick rewards are optional.
  The horizon step's reward includes the settlement cost, and its
  ``step_result`` carries the settled quantity, price, notional, fee and
  cap-override flag. The settlement price is read from the bucket after the
  horizon and never enters an observation the agent acted on.
* ``terminated`` and ``truncated`` follow the reference simulator. No
  wrapper time limit changes the horizon.
* Frozen training statistics can be supplied explicitly or loaded from a
  versioned file. Sequential pools support fixed evaluation sets; random
  selection supports training.

A CPU smoke example:

.. code-block:: python

   from tapeforge.execution import EpisodeSpec, make_episode, twap, run_episode
   from tapeforge import generate, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=40, profile="uniform")
   offsets = windows.build_offsets(trades["timestamp"], 0, 40, DEFAULT_BUCKET_WIDTH_US)
   template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=10, arrival_lookback=5)
   spec, market = make_episode(trades, offsets, 15, side=1, rho_bps=500, template=template)
   trajectory = run_episode(spec, market, twap)
   assert trajectory.last_result.terminated
   assert not trajectory.last_result.truncated

To use the same episode with Gymnasium after installing ``.[rl]``:

.. code-block:: text

   from tapeforge.execution.gym_env import ExecutionEnv, ListPool

   env = ExecutionEnv(ListPool([("example", spec, market)]))
   observation, info = env.reset(seed=7)
   observation, reward, terminated, truncated, info = env.step([0.1])

The environment is ready for a trainer adapter. PPO/SAC training, policy
checkpoint selection, and GPU-vectorized environments are not included.

Evaluating a policy
===================

``tapeforge.execution.evaluation`` holds what a trainer needs around the
environment, and nothing of the trainer itself.

* ``load_training_config`` reads a strict TOML description
  (``tapeforge.execution.training-config/1``): the experiment configuration,
  the training role (always ``train``) and seeds, the reference baseline and
  seed for the frozen statistics, and the evaluation roles and seeds. The
  test role is refused unless ``include_test`` is set.
* ``training_statistics`` fits the statistics on training draws only;
  ``save_statistics`` and ``load_statistics`` freeze them to a JSON file
  tagged with its schema and the simulator version, and refuse files from
  another version.
* ``evaluation_manifest`` lists the fixed evaluation episodes of a run
  directory (the identifiers every baseline replayed) with the split record,
  data digests and a content digest; ``load_evaluation_manifest`` rejects an
  edited file.
* ``vector_policy`` adapts a callable over the learner's float32 observation
  vector to the runner's policy interface. Its output is validated, never
  clipped.
* ``evaluate_policy`` replays any policy on a run's episodes and writes
  records with the baseline schema into the run directory, so the report
  pairs it against the baselines on identical episode IDs.
* ``checkpoint_provenance`` returns the model-agnostic record a checkpoint
  must carry (configuration and its hash, split and data hashes, statistics,
  seed, action and reward conventions, simulator version, evaluation
  digest); ``validate_checkpoint_provenance`` checks it before an
  evaluation.

.. code-block:: python

   import tempfile

   import numpy as np

   from tapeforge.execution import evaluate_policy, run_experiment, vector_policy
   from tapeforge.execution.experiment import load_config, rebuild_splits

   config = load_config("experiments/configs/smoke-synthetic.toml")
   run_dir = run_experiment(config, output_dir=tempfile.mkdtemp())
   splits = rebuild_splits(config)
   policy = vector_policy(lambda vector: np.float32(0.1))  # a stand-in for a learned policy
   written = evaluate_policy(run_dir, policy, "constant-0.1", splits=splits)
   assert len(written) == 2  # one record per evaluated role and seed

Feature engine
==============

Each bucket is a half-open slice ``[offsets[b], offsets[b + 1])`` into sorted
records. Repeated offsets represent empty buckets. The feature outputs are:

* ``event_count``: aggregate-trade record count, int64.
* ``total_quantity``: quantity sum in integer ticks, int64.
* ``signed_quantity``: buy-initiated minus sell-initiated quantity, int64.
  ``buyer_is_maker=True`` denotes a sell-initiated trade.
* ``vwap``: quantity-weighted price in ordinary price units, float64.
* ``valid``: whether the bucket contains records, bool.

VWAP is zero when total quantity is zero. A bucket containing zero-quantity
records still has ``valid=True``. Integer results are exact within the
validated overflow bound; float VWAP comparisons use the contract tolerance.
These are trade features, not order-book imbalance features.

.. code-block:: python

   from tapeforge import generate, ops, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=100, profile="burst")
   offsets = windows.build_offsets(trades["timestamp"], 0, 100, DEFAULT_BUCKET_WIDTH_US)
   features = ops.aggregate(trades, offsets)
   assert features["event_count"].sum().item() == len(trades)

``ops.aggregate`` accepts a structured NumPy trade array with NumPy int64
offsets, or a mapping of contiguous column tensors with offsets on the same
device. CPU inputs use the reference; CUDA inputs use Triton. There is no
implicit cross-device fallback. ``reference.aggregate_reference`` is the
NumPy-only API.

The one-pass capacity is 4,096 records per bucket. Automatic dispatch uses
two-pass beyond that; an explicitly requested one-pass block size remains a
hard capacity limit. Prices and quantities are int64 and maker flags bool.
Validation checks shape, dtype, device, contiguity, offsets, non-negative
values, and conservative overflow bounds.

For unchanged CUDA inputs, ``prepared.prepare(columns, offsets)`` validates,
plans, and allocates once. ``execute()`` reuses output/scratch buffers;
``execute_owned()`` returns independent copies. Inputs are borrowed and must
not be mutated. One prepared object cannot be executed concurrently, and
cross-stream consumers must synchronize. There is no global cache.

``StreamingAggregator.push`` accepts timestamp-ordered chunks and emits
closed buckets; ``finalize`` emits remaining buckets. It preserves batch
feature semantics, rejects decreasing timestamps, and does not deduplicate
IDs or buffer late arrivals. It is an offline replay/reference component;
no live collector is provided.

Checks and GPU verification
===========================

.. code-block:: console

   $ python -m ruff check src tests benchmarks tools
   $ python -m pytest -q
   $ python -m pytest -q --cov=tapeforge --cov-report=term-missing
   $ python -m pytest -q tests/test_execution/test_gym_env.py tests/test_execution/test_end_to_end.py
   $ python -m mypy

The strict type check currently reports two known errors in the ``model.py``
stub. CUDA tests skip with a capability reason on CPU machines. Tests using
optional downloaded archives name missing files when skipped. The default
suite includes slow tests. README Python examples execute in the docs tests.

To verify the current working tree on a CUDA machine:

.. code-block:: console

   $ python tools/verify_snapshot.py pack --out /tmp/tapeforge-snapshot.tgz

Transfer the archive, then run on that machine:

.. code-block:: console

   $ tar xzf tapeforge-snapshot.tgz
   $ cd TapeForge
   $ python -m pip install -e '.[dev,rl]'   # use the runtime's CUDA torch and Triton
   $ python tools/verify_snapshot.py run --artifacts verification --record --zip

The snapshot includes tracked and eligible untracked files, excluding ignored
files, and records a content digest. The runner saves step logs, environment,
JUnit XML, status manifest, and a results copy. Download ``verification.zip``
and recorded benchmark results before disposing of the remote environment.

Benchmarks
==========

.. code-block:: console

   $ python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record
   $ python benchmarks/kernel.py --timing cuda_events --no-record
   $ python benchmarks/kernel.py
   $ python benchmarks/pipeline.py

``kernel.py`` compares the checked API, prepared execution, individual passes,
forced two-pass execution, eager/compiled PyTorch, and the CPU reference.
Stage rows separate validation, metadata, allocation, and preparation; these
overlap with end-to-end rows and must not be added to them.
``pipeline.py`` measures host/device transfers and compute using pageable or
pinned memory. Pinned-stream overlap and ``train.py`` remain unimplemented.

Correctness is checked before timing. Warm-up excludes compilation. Results
record wall-clock or CUDA-event timing, workload distribution, setup included
in each region, hardware, software, revision, dirty state, and snapshot hash
when available. Preparation costs must be reported separately when comparing
repeated prepared execution with a checked API. Record tuning sweeps in
``benchmarks/results.json`` before publishing performance claims.

Repository layout
=================

.. code-block:: text

   src/tapeforge/
     contract.py, validation.py       units, bounds, input checks
     ingest.py, generate.py           historical and synthetic events
     windows.py                      packed bucket offsets
     reference.py, streaming.py      batch and incremental CPU features
     kernels.py, prepared.py, ops.py  GPU kernels and public feature API
     execution/
       spec.py, state.py             execution rules and data structures
       simulator.py, episodes.py     transitions and episode construction
       days.py, splits.py            day tapes and chronological partitions
       sampling.py, normalization.py seeded tasks and training statistics
       baselines.py, runner.py       policies, replay loop, summaries
       datasets.py, experiment.py    acquisition and configured experiments
       report.py, replay.py          aggregate reports and episode inspection
       sensitivity.py                cross-run terminal-rule comparison
       evaluation.py                 evaluation sets, external policies, provenance
       gym_env.py, env_check.py      optional Gymnasium interface and checker
     model.py                        classifier stub
   experiments/configs/              synthetic and futures smoke configs per terminal setting
   benchmarks/                       compute/transfer timings and result records
   tests/                            CPU/GPU parity, accounting, leakage, tooling
   tools/                            parity summary and snapshot verification
   data/, runs/                      local datasets and outputs; Git-ignored

Development guidelines
======================

Write comments, docstrings, errors, and public documentation for developers
maintaining the code. Explain behavior, units, invariants, algorithm choices,
and non-obvious constraints. Do not label source comments or public docs by
curriculum weeks, lesson milestones, or classroom instructions. Describe
unfinished work by the missing capability and acceptance criteria. Keep
personal learning schedules and exercises under ignored ``planning/``.
These conventions apply to human and AI-generated changes alike.

Distinguish intended behavior from implemented and verified behavior. Report
CPU tests, interpreter checks, and actual CUDA execution separately. Support
performance claims with reproducible measurements and environment metadata.
Preserve exact reductions within the data contract and verify correctness
against an independent reference before benchmarking.

Documentation policy
====================

``README.rst`` is the public documentation. Markdown, private design notes,
and learning plans remain local and ignored. Shared results use JSON, CSV,
or reStructuredText. Downloaded market data and run artifacts are not committed.

License
=======

MIT.
