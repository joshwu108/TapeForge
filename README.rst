=========
TapeForge
=========

TapeForge builds a market-data feature engine that converts timestamped
aggregate-trade records into fixed-width feature tensors for machine learning.
It combines a NumPy correctness reference, Triton GPU reductions, a
PyTorch-facing API, and benchmarks for compute and data-transfer costs.

Overview
========

Market activity is uneven: a time bucket may contain no records or thousands.
Padding every bucket to the busiest bucket wastes memory. TapeForge keeps
records packed and represents each bucket with two boundary offsets. A fused
GPU kernel reads each slice and computes counts, quantity sums, signed flow,
and volume-weighted average price (VWAP).

.. code-block:: text

   Public aggTrades archive or seeded synthetic records
       -> normalized timestamps and integer-scaled prices/quantities
       -> fixed-width bucket offsets
       -> CPU reference or Triton GPU aggregation
       -> per-bucket PyTorch feature tensors

The project investigates how workload skew, memory layout, validation, and
host/device transfers affect feature-generation throughput. A streaming CPU
aggregator computes the same features one chunk at a time, bitwise equal to
the batch result, as the live-path counterpart of the batch engine. Buckets
longer than one GPU tile are reduced by a two-pass kernel pair that carries
integer partial sums and divides once. On top of the features, a torch-free
reference execution simulator replays an optimal-execution episode against
the bucket tape with exact integer accounting (see "Execution simulator"),
and signal-free baselines replay over chronological day splits with embargo
and purge (see "Episodes, splits and baselines"). Planned extensions include
pinned-memory stream overlap in the transfer benchmark and an end-to-end
training throughput experiment. The classifier and training benchmark remain
stubs; no policy is learned. The input contains trades rather than order-book
data.

Implementation status
=====================

Three things are kept apart below. *Implemented* means the code and its
tests exist in this tree. *CPU-verified* means the test suite asserts the
behaviour on a CPU; every such check runs in CI on each push. *GPU-verified*
means the CUDA code path was executed on hardware against a clean snapshot of
a named commit; CI has no GPU, so that column only ever reflects the last
manual run and must be refreshed whenever the GPU-facing modules change.

.. list-table::
   :header-rows: 1
   :widths: 34 12 24 30

   * - component
     - implemented
     - CPU-verified
     - GPU-verified
   * - data contract, ingest, synthetic generation, window index
     - yes
     - yes
     - no GPU code
   * - NumPy reference features and the plain-loop oracle
     - yes
     - yes (each against the other)
     - no GPU code
   * - streaming CPU aggregator
     - yes
     - yes, bitwise equal to the batch reference
     - runs on the CPU only, by design
   * - ``ops.aggregate`` CPU path, tensor input validation
     - yes
     - yes
     - CUDA dispatch: Tesla T4, commit ``c5d0f80``
   * - one-pass Triton kernel, dispatch plan, launcher
     - yes
     - plan and tile selection only
     - Tesla T4, commit ``c5d0f80``
   * - two-pass Triton kernels (chunk partials, combine)
     - yes
     - chunk schedule only
     - Tesla T4, commit ``c5d0f80``
   * - ``tapeforge.prepared`` validate-once execution
     - yes
     - error paths and immutability only
     - Tesla T4, commit ``c5d0f80``
   * - execution simulator (``tapeforge.execution``: spec, state, transitions, episodes)
     - yes
     - yes, against hand-calculated tables and Hypothesis properties
     - no GPU counterpart exists
   * - day tapes, chronological splits (embargo, purge), seeded sampling, train-only statistics, replay runner, run records, three baselines
     - yes
     - yes, on synthetic days; on the downloaded futures day when present
     - no GPU counterpart exists
   * - kernel and pipeline benchmarks
     - yes
     - CPU smoke run of the CPU-capable rows
     - executed on a T4, not recorded (``results.json`` is empty)
   * - ``pinned_streams`` transfer variant
     - no
     - --
     - --
   * - training throughput benchmark, ``model.py``
     - stubs that raise
     - --
     - --

The one-pass Triton kernel assigns one program per bucket and covers buckets
of up to 4,096 records with a single tile. Longer buckets dispatch
automatically to the two-pass path: pass 1 reduces 1,024-record chunks to
int64 partial sums, pass 2 combines each bucket's contiguous partials in
128-wide tiles and performs the only division. No bucket length is rejected
or truncated; an explicit one-pass tile size remains a hard capacity check.
``tapeforge.prepared`` validates and plans an input once for repeated
execution into reused buffers. No policy has been trained.

Hardware verification so far: the GPU parity suite of commit ``c5d0f80`` (82
tests: one-pass and two-pass paths, prepared execution, multi-tile pass-2
cases, forced two-pass, stress burst workloads) passed on a Tesla T4 (torch
2.14.1+cu130, Triton 3.8.0, CUDA 13.0) against a clean snapshot of the
committed tree, and the quick kernel benchmark reported correct results for
every row on sparse, uniform and burst workloads. The kernel, launcher,
validation and prepared-execution modules and the GPU test suites are
unchanged since that commit; later commits added the CPU-only
``tapeforge.execution`` package, tests and documentation. The run's manifest,
JUnit report, logs and environment capture are retained by the owner outside
the repository. **No speedup figures are published.** ``benchmarks/results.json``
is empty until a recorded sweep lands, and every performance claim must cite
an entry there with its hardware and software provenance.

Preliminary dataset observations
================================

Earlier measurements on BTCUSDT 2025-01-02 (1,299,165 aggregate-trade records)
predate the benchmark scripts and are not yet reproduced in
``benchmarks/results.json``:

============================================  =============================
measurement                                   result
============================================  =============================
window index, 1.3M records to 86,400 buckets    3.2 ms
CPU reference features, same input             7 ms (~185M records/s)
padded ``[n_buckets, max_len]`` equivalent      2.78 GB vs 31 MB packed (89x)
busiest second                                1,343 records
============================================  =============================

The memory comparison motivates packed storage. GPU benchmarks must establish
whether it improves execution time with and without transfer costs.

Features
========

For every fixed-width time bucket (1 second by default):

===================  =============================================================
feature              plain-English definition
===================  =============================================================
``event_count``      number of aggregate-trade records in the bucket (exact)
``total_quantity``   sum of traded quantity (exact, quantity ticks)
``signed_quantity``  buy-initiated minus sell-initiated quantity (exact)
``vwap``             quantity-weighted average price (float, documented tolerance)
``valid``            false when the bucket contained no records
===================  =============================================================

A row of the input is an *aggregate* trade record, not an individual fill.
Record counts in this project always mean aggregate-trade records. The input
carries no order book, so none of these are order-book imbalance features.

Units, the taker-side sign convention derived from ``buyer_is_maker``, the
price/quantity integer scaling, the overflow bound and the VWAP tolerance are
all defined in one place: ``src/tapeforge/contract.py``.

Price and quantity are scaled integers at 1e2 and 1e5, chosen by measuring the
canonical day rather than by guessing: BTCUSDT uses exactly 2 and 5
significant decimals, and those scales leave the VWAP numerator
``sum(price_ticks * quantity_ticks)`` 96,869x inside int64 on the busiest
second. Counts, total quantity and signed quantity are therefore exact, and
VWAP's only floating-point operation is its final division.

These scales are exact for BTCUSDT-like instruments and cannot represent a
low-priced symbol such as SHIBUSDT, whose prices carry 8 decimals. Ingest
raises in that case rather than truncating, so the limitation fails loudly;
per-symbol scales from the exchange tick and step size are the documented next
step.

Install
=======

Requires Python 3.11-3.12. CPU-only is enough for the reference
implementation and its tests; the GPU engine needs Linux with an NVIDIA CUDA
device and a working Triton build.

.. code-block:: console

   $ python -m venv .venv && source .venv/bin/activate
   $ pip install -e '.[torch,dev]'          # CPU development
   $ pip install -e '.[gpu,ml,dev]'         # Linux + NVIDIA CUDA GPU

Verify a checkout
=================

.. code-block:: console

   $ python -m ruff check src tests benchmarks tools
   $ python -m mypy                         # strict; the model.py stub has two known errors
   $ python -m pytest -q                    # all tests; GPU tests skip with a reason
   $ python -m pytest -q --cov=tapeforge --cov-report=term-missing   # what CI runs
   $ python -m pytest -q -m gpu             # GPU parity suite, needs CUDA + Triton
   $ python -m pytest -q -m slow            # stress suite (GPU cases skip without CUDA)
   $ python tools/verify_snapshot.py run --artifacts verification   # all of the above, with artifacts

The last command runs lint, the test suites, the parity script and the quick
kernel benchmark as subprocesses and writes a manifest, logs, JUnit reports
and the captured environment under ``verification/``. Steps that need CUDA
are recorded as skipped, with the reason, on a machine without a GPU. The
default ``pytest`` run includes the ``slow`` marker; tests that need a
downloaded archive skip, naming the missing file, when ``data/`` is empty.

Usage
=====

.. code-block:: python

   from tapeforge import generate, ops, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=1000, profile="burst")
   offsets = windows.build_offsets(trades["timestamp"], 0, 1000, DEFAULT_BUCKET_WIDTH_US)
   features = ops.aggregate(trades, offsets)   # dict of (n_buckets,) CPU tensors
   assert features["event_count"].sum() == len(trades)

``ops.aggregate`` needs torch and accepts two input formats; ``offsets`` must
be the same kind as ``trades``:

==========================================  =====================  ==================
``trades``                                  ``offsets``            computed by
==========================================  =====================  ==================
NumPy structured array of ``TRADE_DTYPE``   NumPy int64 array      CPU reference
mapping of column name to CPU tensors       CPU int64 tensor       CPU reference
mapping of column name to CUDA tensors      CUDA int64 tensor      Triton kernel
==========================================  =====================  ==================

Tensor columns are 1-D and contiguous: ``price`` and ``quantity`` int64
ticks (non-negative), ``buyer_is_maker`` bool. Other columns are ignored.
It returns ``event_count``, ``total_quantity`` and ``signed_quantity``
(int64), ``vwap`` (float64, 0.0 where a bucket has no quantity) and ``valid``
(bool), on the input device. Invalid input raises rather than being
corrected. That covers wrong dtypes, strided tensors, mixed or unsupported
devices, offsets outside the record range, and sums that could overflow
int64. Nothing is silently copied between devices or truncated.
``tapeforge.reference.aggregate_reference`` is the torch-free NumPy API.

On CUDA the launcher reads the longest bucket during validation and picks
the kernel path from it (``kernels.plan_dispatch``): one-pass up to
``kernels.MAX_ONE_PASS_BUCKET`` records, two-pass beyond. Both paths produce
identical outputs by contract and are held to the same parity rule against
the plain-loop oracle.

Prepared execution
==================

``ops.aggregate`` validates, plans and allocates on every call. For repeated
aggregation of fixed, resident tensors, ``tapeforge.prepared`` does that work
once:

.. code-block:: text

   ready = prepared.prepare(columns, offsets)   # validation, dispatch plan,
                                                # chunk schedule, buffers
   features = ready.execute()                   # kernel launches only
   features = ready.execute()                   # same borrowed buffers again
   kept = ready.execute_owned()                 # independent copies

The prepared object borrows the input tensors (they must stay unchanged; new
data means a new ``prepare``), returns the same output tensors from every
``execute`` (overwritten by the next call), launches on the current stream of
the input device, owns one set of scratch buffers (so it is not shared
between concurrent callers), and caches nothing: every ``prepare`` validates
from scratch. ``run_pass_1`` and ``run_pass_2`` expose the two-pass stages
for instrumentation, and ``path="two_pass"`` forces the two-pass path on
inputs that would fit one tile so both paths can be measured on the same
workload.

Command line:

.. code-block:: console

   $ python -m tapeforge.generate --seed 7 --seconds 10 --profile burst
   $ python -m tapeforge.ingest --symbol BTCUSDT --date 2025-01-02

Downloaded archives land in ``data/`` and are never committed.

Streaming
=========

``tapeforge.streaming`` computes the same features from timestamp-ordered
chunks, the way a live system receives them, without torch or CUDA. A chunk
may be one record or a whole day. Buckets are emitted as they close, and
``finalize`` returns the rest at end of stream:

.. code-block:: python

   import numpy as np
   from tapeforge import generate, reference, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US
   from tapeforge.streaming import StreamingAggregator, concat_features

   trades = generate.generate_trades(seed=7, n_buckets=100, profile="burst")
   live = StreamingAggregator(start_us=0, n_buckets=100, width_us=DEFAULT_BUCKET_WIDTH_US)
   parts = [live.push(chunk) for chunk in np.array_split(trades, 13)]
   parts.append(live.finalize())
   streamed = concat_features(parts)

   offsets = windows.build_offsets(trades["timestamp"], 0, 100, DEFAULT_BUCKET_WIDTH_US)
   batch = reference.aggregate_reference(trades, offsets)
   assert np.array_equal(streamed.vwap, batch.vwap)   # bitwise, not approximately

Semantics match the batch path exactly: half-open buckets, records outside
the window ignored, empty buckets invalid and zeroed, int64 partial sums
carried across chunk boundaries and divided once per bucket. Chunks must
arrive in timestamp order; an out-of-order chunk raises ``OutOfOrderError``
and leaves the aggregator untouched rather than being re-sorted. The int64
accumulation bound is checked before every chunk is absorbed, so partial sums
never wrap. Property-based tests split generated days and arbitrary ordered
records at random cut points and assert bitwise equality with the batch
reference; the slow suite runs 1,000 such chunkings.

Execution simulator (reference)
===============================

``tapeforge.execution`` is a deterministic CPU simulator of one
optimal-execution episode: trade a target quantity Q* within a horizon of H
one-second buckets, one decision per bucket, paying fees and an explicit
residual charge for whatever is left at the end. It is torch-free, built from
pure functions over frozen dataclasses, and every monetary and quantity field
is a Python integer, so a replayed episode is byte-identical and a future
batched implementation can be compared with it element by element.

Accepted first configuration (owner decisions of 2026-10-23; the coefficients
marked *swept* have no default and are zero, i.e. off, unless set):

=========================  =========================================================
parameter                  value
=========================  =========================================================
horizon                    H = 300 buckets of 1 s, one decision at each boundary
side                       buy from zero to +Q*, or liquidate +Q* to zero
target                     Q* = round_half_up(rho x V_ref), V_ref = total quantity
                           over the 300 buckets before the start; rho in
                           {2 %, 5 %, 10 %}; at least one tick; at most the
                           participation cap of V_ref
action                     fraction of the *remaining* quantity in [0, 1],
                           floored to quantity ticks (the only float, converted once)
fill rule                  ``bucket_vwap_capped`` (synthetic, see below)
participation cap          10 % of the execution bucket's total quantity
slippage                   0 price ticks (parameter)
fee                        taker, 10 bps of notional, rounded *up* to the cash tick
latency ``l``              0 (decision at boundary t fills in bucket t); 1 as ablation
data delay ``d``           0 (a decision at t reads buckets <= t - 1); 1 as ablation
arrival price P_0          VWAP of the last valid bucket before the first decision,
                           searched over the lookback L
residual at the horizon    Perold opportunity cost ``side x (P_T - P_0) x residual``
                           plus ``mu x residual``; P_T = first valid bucket after
                           the last fill bucket (``mu`` swept)
per-step risk              ``lambda x remaining_after`` (``lambda`` swept)
=========================  =========================================================

Units follow the data contract: quantity ticks (1e-5 BTC), price ticks (0.01
USDT) and *cash ticks* = price tick x quantity tick (1e-7 USDT). A notional is
``price_ticks x quantity_ticks`` with no rounding; the fee ceiling is the only
rounding anywhere in the accounting. Rates are integer basis points so caps,
fees and targets never pass through a float. The bucket VWAP used as a fill
price is the engine's exact integer numerator divided by the quantity and
rounded half-up to a price tick, not the float ``vwap`` feature.

**The fill assumption is synthetic.** The submitted quantity fills at the
execution bucket's tick-rounded VWAP, capped at 10 % of that bucket's traded
quantity; nothing fills in a bucket with no records or no quantity, and the
decision is consumed. A real taker order would not get the bucket VWAP. The
simulator models no market impact, no reaction of the market to the agent,
no passive or limit orders, no queue position and no spread, because the
input is aggregate trades without a book. The rule is a named string so a
best-bid/ask fill-at-touch rule can be added beside it once quote data is
ingested; nothing else changes.

Reward and shortfall. Each ``StepResult`` carries the reward split into
``cost_component`` = ``side x (notional - P_0 x fill) + fee`` (positive is a
cost for both sides), ``risk_component`` = ``lambda x remaining_after`` and
``terminal_component`` = the residual term at the horizon, with
``reward_ticks = -(cost + risk + terminal)``. Summed over a terminated episode
the rewards equal ``-IS - lambda x sum(remaining_after)`` exactly, where the
implementation shortfall is ``IS = side x (exec_notional - P_0 x filled) +
fees + residual term``. The accounting identity ``cash + inventory x P_0``
(relative to the initial portfolio) ``== -IS`` of the fills so far holds after
every step and is asserted in the tests. ``shortfall_bps`` divides by
``P_0 x Q*`` for reporting only.

Termination and truncation. ``remaining == 0`` or ``step == H`` sets
``terminated`` (the horizon is part of the task, since time remaining is
observed). A tape that ends before the fill bucket of a decision, or that has
no valid bucket after the horizon when residual inventory needs a terminal
mark, sets ``truncated`` instead; the two flags are never both set. Stepping a
finished episode raises ``EpisodeOver``.

Observations (``observe``) are a fixed float64 vector (``OBSERVATION_FIELDS``):
remaining and time-remaining fractions, side, the return of the mark price
(last valid visible VWAP) against P_0 and its age, and the newest visible
bucket's validity, record count, quantity relative to Q*, taker imbalance and
return against P_0. The visible tape is cut at bucket ``t - 1 - d`` before
anything is read, so later buckets cannot leak in by construction; the tests
perturb every later bucket and assert the observation is unchanged.

.. code-block:: python

   from tapeforge import generate, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US
   from tapeforge.execution import (
       EpisodeSpec, action_for_quantity, implementation_shortfall, is_done,
       make_episode, remaining_qty_ticks, reset, step,
   )

   trades = generate.generate_trades(seed=7, n_buckets=40, profile="uniform")
   offsets = windows.build_offsets(trades["timestamp"], 0, 40, DEFAULT_BUCKET_WIDTH_US)
   template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=10, arrival_lookback=5)
   spec, market = make_episode(trades, offsets, start_bucket=15, side=1, rho_bps=500,
                               template=template)   # Q* from the 10 buckets before 15

   state, rewards = reset(spec, market), []
   while not is_done(state, spec):                   # TWAP: remaining / time remaining
       remaining = remaining_qty_ticks(state, spec)
       action = action_for_quantity(remaining // (spec.horizon_buckets - state.step), remaining)
       state, result = step(state, action, market, spec)
       rewards.append(result.reward_ticks)
       if result.terminated or result.truncated:
           break
   shortfall = implementation_shortfall(state, spec, result.terminal_price_ticks or None)
   assert sum(rewards) == -shortfall                 # lambda = 0: rewards are minus the shortfall

``make_episode`` cuts the slice (``max(H, L + d)`` buckets before the start,
``H + l + L`` after) and sizes Q*; ``reset`` fixes P_0 and runs the int64 bound
check (``check_episode_bounds``: ``H x cap x max_price`` and ``cap x
max_price x fee_bps`` must fit, like the feature engine's accumulation bound).
``action_for_quantity`` is the inverse of the floor conversion for
quantity-based baselines. The simulator is verified on the CPU only, against
the hand-calculated worksheet tables (buy, sell, partial completion with the
three residual rules, invalid bucket, the 5,002.5 -> 5,003 fee case) and
Hypothesis properties over random tapes and actions. No learned policy
exists; the next section covers how episodes are drawn and replayed.

Episodes, splits and baselines
==============================

The pieces needed to run signal-free baselines reproducibly on real or
synthetic tapes, with the leakage controls the owner accepted. Nothing in
them is trained or fitted, apart from the observation statistics described
below.

**Day tapes.** A ``DayTape`` is one UTC calendar day of bucket features (an
``origin == 0`` ``MarketSlice`` plus its date, start timestamp and bucket
width). ``synthetic_day`` builds one from the seeded generator;
``day_from_archive`` builds the 86,400 one-second buckets of a downloaded
aggTrades day through the trade ingest and refuses records outside the day.
Contiguous days join into one tape, so an episode may run across midnight
inside a split.

**Splits.** Days are assigned to train, validation and test explicitly, by
date, with no defaults (the test role may be empty), and the assignment
must be chronological. Contiguous same-role days form a segment. An episode
may start at ``t_0`` only if the whole slice it reads (``max(H, L + d)``
buckets before, ``H + l + L`` after) lies inside its segment, if the window
``[t_0 - L, t_0 + H + l + 1]`` lies inside it (purge), and if the slice
avoids the first ``max(H + l, L)`` buckets of a segment that follows a
different role (embargo). The admissible starts of each segment are one
half-open range, listed in a JSON-serialisable ``SplitRecord`` together
with every day, the segment structure and the parameters that produced the
ranges; the tests check the ranges on hand-built boundaries and with
Hypothesis over random day structures.

**Sampling.** ``sample_episodes`` draws episodes inside one role. Draw
``index`` of ``seed`` is a pure function of ``(seed, role, index)``: the
start is uniform over the admissible starts, the side is buy or sell with
equal probability, rho is uniform over {2 %, 5 %, 10 %}, and Q* is sized
from the volume before the start when the episode is cut. A run record that
lists ``(seed, index)`` per episode is enough to rebuild it exactly.

**Normalisation.** ``observation_statistics`` computes per-field mean and
standard deviation of the observations a reference policy sees on training
draws only; it refuses draws from any other role, and a test changes the
validation and test days and asserts the statistics are unchanged. The
baselines do not use them.

**Replay and baselines.** ``run_episode`` steps a policy callable through an
episode and returns the full trajectory (states, observations, actions, step
results with the reward split); ``summarize_episode`` reduces it to exact
integer accounting plus the shortfall in basis points, and ``summarize_set``
aggregates a set with completion and truncation rates and nearest-rank
tail percentiles. A policy receives an ``Observation``: the float vector of
``observe`` next to the exact integers behind it, so the baselines never
round through a float. Three baselines are plain functions: ``twap`` submits
``remaining // time_remaining`` and completes exactly whenever the fill rule
allows; ``immediate`` submits everything and is spread out by the
participation cap alone; ``cap_aware_twap`` raises the TWAP share to
whatever the newest visible bucket's cap estimate says the remaining
decisions cannot absorb (a proposed heuristic, not an accepted
specification).

**Run records.** ``run_policy`` samples a role, replays one policy and
returns a record with the configuration, seeds, split record, code revision,
per-episode draws and summaries, and the set summary; ``write_run_record``
writes it as JSON under ``runs/``, which Git ignores. Two runs with the same
inputs produce identical records except for the creation time and the code
revision fields, and the tests assert exactly that.

.. code-block:: python

   from tapeforge.execution import (
       BASELINES, EpisodeSpec, SplitDates, build_splits, run_policy, synthetic_day,
   )

   SECOND = 1_000_000
   days = [
       synthetic_day(seed=1, day="2024-01-01", n_buckets=80, start_us=0),
       synthetic_day(seed=2, day="2024-01-02", n_buckets=80, start_us=80 * SECOND),
       synthetic_day(seed=3, day="2024-01-03", n_buckets=60, start_us=160 * SECOND),
   ]
   template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=3)
   splits = build_splits(days, SplitDates(("2024-01-01", "2024-01-02"), ("2024-01-03",)),
                         template)
   assert splits.record.segments[1].embargo_before == 5   # max(H + l, L) after a role change

   record, trajectories = run_policy(
       splits, "validation", BASELINES["twap"], "twap", template, n_episodes=10, seed=1
   )
   assert record["summary"]["n_episodes"] == 10
   assert all(t.last_result.terminated for t in trajectories)

The example runs on CPU without data (a real day replaces ``synthetic_day``
with ``day_from_archive``). The real futures day, when downloaded under
``data/``, is covered by tests that skip precisely when it is absent; no
result from it is published here.

Reproducible GPU verification
=============================

The GPU work is verified on a separate CUDA machine (a Colab GPU runtime, for
example) against the exact working tree, uncommitted edits included, so that
the tested code and the reported code are the same thing:

.. code-block:: console

   $ python tools/verify_snapshot.py pack --out /tmp/tapeforge-snapshot.tgz

The archive holds every file Git tracks or would track and a
``snapshot.json`` with the commit, a dirty flag and a digest over the packed
files; ignored material never leaves the machine. On the GPU machine, after
uploading the archive:

.. code-block:: console

   $ tar xzf tapeforge-snapshot.tgz && cd TapeForge
   $ pip install -e '.[dev]'                       # keep the runtime's torch and triton
   $ python tools/verify_snapshot.py run --artifacts verification --record --zip

That runs, in order: lint, the full test suite, the GPU parity suite (both
kernel paths and prepared execution), the stress suite,
``tools/check_gpu_parity.py``, the toy program, the quick kernel check, and
with ``--record`` the kernel sweep and the pipeline benchmark, which append
to ``benchmarks/results.json``. Every step is a
subprocess with its own log, pytest steps also write JUnit XML, and
``manifest.json`` lists each step's status, exit code and duration next to
``environment.json`` (GPU, driver, CUDA, torch, Triton, NumPy, commit, dirty
flag, snapshot digest). Download ``verification.zip`` and the updated
``results.json``; a claim in this file must cite an entry recorded that way.

Benchmarks
==========

Three scopes are timed separately and never collapsed into one headline
number:

1. ``benchmarks/kernel.py`` -- inputs already resident on the device. Its
   feature rows are ``ops.aggregate`` on CUDA as a user calls it (validation,
   planning, allocation, launches), the prepared path (launches into reused
   buffers, preparation untimed), the bare unchecked launchers, pass 1 and
   pass 2 alone on two-pass workloads, the two-pass path forced onto
   one-pass workloads so both paths see identical data, an eager PyTorch
   baseline (``searchsorted`` plus ``index_add_``), the same baseline under
   ``torch.compile``, and the NumPy reference on the host. Its stage rows
   time validation, chunk-metadata construction, allocation and
   ``prepare`` on their own; they overlap with the end-to-end rows and are
   never summed with them. Every row records which host-side work its timed
   region includes, the dispatch path, the chunk count and the bucket-length
   distribution. Without CUDA it runs the CPU-capable implementations as a
   smoke test and records nothing.
2. ``benchmarks/pipeline.py`` -- host-to-device copy, compute and
   device-to-host copy, timed separately and end to end, from pageable and
   from pinned host memory, against the NumPy reference. It needs CUDA. The
   ``pinned_streams`` variant is not implemented yet.
3. ``benchmarks/train.py`` -- the whole training loop. Not implemented yet.

Every implementation must match the CPU reference before it is timed:
integers exactly, VWAP within the contract tolerance. Each timed region is a
complete Python call, so every number includes Python dispatch and launch
overhead; compilation happens in untimed warm-up. Two timing methods are
recorded by name: ``wall`` (``time.perf_counter`` with
``torch.cuda.synchronize()`` before and after every call on CUDA, the
default) and ``cuda_events`` (CUDA event pairs on the current stream, which
exclude the host-side synchronize but still charge launch latency when the
stream is idle). Configurations are versioned TOML under
``benchmarks/configs/``. Results are appended to ``benchmarks/results.json``
with the workload, the timing method, and hardware and software provenance
(GPU, CUDA, torch, Triton, NumPy, git commit, whether the tree had
uncommitted changes, and the snapshot digest when run from a packed
snapshot). Negative results are published too. Dispatch thresholds and
tuning choices are made from recorded sweeps, never from a single run.

.. code-block:: console

   $ python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record   # quick check
   $ python benchmarks/kernel.py --timing cuda_events --no-record        # device-stream view
   $ python benchmarks/kernel.py                                         # full sweep, recorded
   $ python benchmarks/pipeline.py

Layout
======

.. code-block:: text

   src/tapeforge/
     contract.py    units, sign convention, scaling, tolerances
     generate.py    seeded synthetic trades (sparse / uniform / burst)
     ingest.py      public aggTrades loading, normalization, provenance
     windows.py     bucket boundary offsets -- the no-padding window index
     reference.py   NumPy reference features: vectorized, plus a naive
                    plain-loop oracle that validates it
     streaming.py   chunk-at-a-time CPU aggregator, bitwise equal to the batch path
     kernels.py     Triton kernels: one-pass bucket reduction, two-pass
                    chunk partials + combine, dispatch plan and launchers
     prepared.py    validate-once, launch-many execution on fixed inputs
     validation.py  tensor input checks shared by ops and the kernel launcher
     ops.py         PyTorch-facing API: CUDA -> Triton, CPU -> reference
     model.py       small volatility-regime classifier
     execution/     reference execution simulator (CPU, torch-free)
       spec.py        EpisodeSpec, units, fee ceiling, participation cap, int64 bounds
       state.py       MarketSlice (bucket tape, integer VWAP ticks), State, StepResult
       simulator.py   reset / observe / step, shortfall, residual rules
       episodes.py    tape window, reference volume, target Q*, make_episode
       days.py        DayTape: one UTC day of features, synthetic / archive adapters
       splits.py      chronological day splits, embargo, purge, SplitRecord
       sampling.py    seeded episode draws, reproducible from (seed, role, index)
       runner.py      Observation, run_episode, summaries, JSON run records
       baselines.py   twap, immediate, cap_aware_twap
       normalization.py  train-only observation statistics
   tests/           parity, edge-case, streaming, two-pass, prepared, stress,
                    docs and tooling suites; tests/test_execution/ holds the
                    simulator, splits, sampling, runner, baseline, leakage and
                    real-day suites with their tape and policy helpers;
                    tests/fixtures/ holds five-row aggTrades excerpts
   test_program.py  toy CUDA example, run by the snapshot verifier
   benchmarks/      harness, the three timed scopes, baselines, workloads,
                    versioned configs/, results.json (empty until recorded)
   tools/           check_gpu_parity.py (one-screen GPU parity summary),
                    verify_snapshot.py (snapshot packer + verification runner)
   .github/workflows/ci.yml   lint and CPU test matrix (Python 3.11, 3.12)
   runs/            baseline run records (ignored, never committed)
   data/            downloaded archives (ignored, never committed)

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

This file is the public documentation. The repository tracks no Markdown:
planning documents, design notes and PRDs stay local under ``planning/``.
Shared results are ``.json``, ``.csv`` or ``.rst``.

License
=======

MIT.
