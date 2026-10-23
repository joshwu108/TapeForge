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
integer partial sums and divides once. Planned extensions include
pinned-memory stream overlap in the transfer benchmark and an end-to-end
training throughput experiment. The classifier and training benchmark remain
stubs. The input contains trades rather than order-book data.

Implementation status
=====================

Ingestion, deterministic generation, window indexing, CPU reference features,
the streaming CPU aggregator, CPU/CUDA API dispatch, and both GPU kernel
paths are implemented. The one-pass Triton kernel assigns one program per
bucket and covers buckets of up to 4,096 records with a single tile. Longer
buckets dispatch automatically to the two-pass path: pass 1 reduces
1,024-record chunks to int64 partial sums, pass 2 combines each bucket's
contiguous partials in 128-wide tiles and performs the only division. No
bucket length is rejected or truncated; an explicit one-pass tile size
remains a hard capacity check. ``tapeforge.prepared`` validates and plans an
input once for repeated execution into reused buffers.

Hardware verification so far: the GPU parity suite of this revision (82 tests:
one-pass and two-pass paths, prepared execution, multi-tile pass-2 cases,
forced two-pass, stress burst workloads) passed on a Tesla T4 (torch
2.14.1+cu130, Triton 3.8.0, CUDA 13.0) against a clean snapshot of the
committed tree, and the quick kernel benchmark reported correct results for
every row on sparse, uniform and burst workloads. The run's manifest, JUnit
report, logs and environment capture are retained by the owner outside the
repository. CI has no GPU, so each revision must be re-verified that way.
**No speedup figures are published.** ``benchmarks/results.json`` is empty
until a recorded sweep lands, and every performance claim must cite an
entry there with its hardware and software provenance. The streaming
aggregator is verified on the CPU only, which is the only place it runs.

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

   $ python -m pytest -q                    # all tests; GPU tests skip with a reason
   $ python -m pytest -q -m gpu             # GPU parity suite, needs CUDA + Triton
   $ python -m pytest -q -m slow            # stress suite (GPU cases skip without CUDA)
   $ python tools/verify_snapshot.py run --artifacts verification   # all of the above, with artifacts

The last command runs lint, the test suites, the parity script and the quick
kernel benchmark as subprocesses and writes a manifest, logs, JUnit reports
and the captured environment under ``verification/``. Steps that need CUDA
are recorded as skipped, with the reason, on a machine without a GPU.

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
   tests/           parity, edge-case, streaming, two-pass, prepared and stress suites
   benchmarks/      harness, three timed scopes, versioned configs, results
   tools/           commit skill bootstrap, safe local commit helper,
                    GPU parity script, snapshot packer + verification runner

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
