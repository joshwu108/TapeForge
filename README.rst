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
VWAP fill model. It is not an exchange simulator or a live trading system.
There is no order routing, passive-order queue model, market impact, or
market response to the policy's actions.

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
     - Reference simulator, day tapes, chronological splits, episode sampling,
       train-only normalization, three baselines, experiment CLI, reporting,
       and replay implemented and CPU-tested.
   * - RL interface
     - Optional Gymnasium CPU adapter tested against direct simulator calls
       and the baseline policies. No PPO or SAC learner is included.
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
published.** A local three-day futures smoke experiment demonstrated the CPU
pipeline; it does not establish policy superiority or out-of-sample skill.

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

The report writes JSON, CSV, and reStructuredText under ``<run-dir>/report/``.
Replay prints the visible market bucket, action, submitted quantity, fill
cap, fill price, fees, remaining inventory, reward components, and terminal
flags. ``--verify`` checks the replay against its recorded outcome.

For historical data, the futures smoke configuration uses BTCUSDT aggregate
trades from 2023-12-01 through 2023-12-03: two training days, one validation
day, and no test period. It allows at most three downloads totaling 500 MiB.
It is a demonstration configuration, not a research evaluation protocol.

.. code-block:: console

   $ python -m tapeforge.execution.datasets fetch --market futures-um \
       --symbol BTCUSDT --date 2023-12-01 --date 2023-12-02 --date 2023-12-03
   $ python -m tapeforge.execution.experiment run experiments/configs/smoke-futures-btcusdt-2023-12.toml

Archives live under ``data/<market>/aggTrades/`` and are checked against the
source's published SHA-256 checksum. Verified cached archives can be reused
without network access. Spot and futures are separate markets; a run names
one market and symbol. Check source licensing before redistributing data.

Experiment contract
===================

The TOML configuration specifies data source, dates, split roles, seeds,
episodes per day, target fractions, policies, and simulator settings.
Unknown keys are rejected. Research configurations need explicit dates;
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

Reports separate completed, incomplete, and truncated episodes. They include
per-policy and per-day shortfall, fees, completion, residuals, and tail costs,
plus paired differences on identical episode IDs. Truncated episodes have
fills-only shortfall and are not pooled with terminated episodes. Day-cluster
bootstrap intervals require at least five days; smaller samples report
insufficient data for uncertainty estimation.

**Completion must be reported alongside cost.** With ``residual_mu = 0``,
unfilled quantity incurs only the terminal price-movement charge and avoids
execution fees. A policy that leaves more unfilled can therefore have lower
reported shortfall. This is not evidence of better execution. Residual and
risk settings must be settled before optimizing a learned policy.

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
* A terminal residual charge based on price movement from arrival to the
  terminal mark, with optional additional residual and inventory-risk terms.

Quantities use 1e-5 units and prices use 0.01 units. Cash ticks are the
product of price and quantity ticks. Accounting uses integers; bucket VWAP
is rounded half-up to price ticks using its integer numerator. These scales
must represent the chosen instrument exactly; ingestion rejects excess
precision instead of truncating it.

For a fill, execution cost is
``side * (notional - arrival_price * filled_quantity) + fee``.
Reward is the negative sum of execution cost, inventory risk, and any
terminal residual component. Completion or horizon exhaustion terminates
an episode. Missing data truncates it; the simulator does not invent a
terminal mark or residual charge for a truncated episode.

The three baselines are:

* ``twap``: divide remaining quantity by remaining decisions.
* ``immediate``: request all remaining quantity at every decision; fills
  remain capped.
* ``cap_aware_twap``: increase TWAP's submission if recent liquidity suggests
  the remaining decisions may not absorb the outstanding quantity. This is
  a heuristic, not a forecast.

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
* ``terminated`` and ``truncated`` follow the reference simulator. No
  wrapper time limit changes the horizon.
* Frozen training statistics can be supplied explicitly. Sequential pools
  support fixed evaluation sets; random selection supports training.

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
   $ python -m pytest -q tests/test_execution/test_gym_env.py
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
       gym_env.py                    optional Gymnasium interface
     model.py                        classifier stub
   experiments/configs/              synthetic and futures smoke configs
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
