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
host/device transfers affect feature-generation throughput. Planned extensions
include a two-pass reduction for large buckets and an end-to-end training
throughput experiment. The classifier and training benchmark remain stubs.
The input contains trades rather than order-book data.

Implementation status
=====================

Ingestion, deterministic generation, window indexing, CPU reference features,
and CPU/CUDA API dispatch are implemented. The one-pass Triton kernel assigns
one program per bucket and supports up to 4,096 records per bucket. Larger
buckets raise ``BucketTooLargeError`` until the two-pass reduction exists.

The GPU parity suite exercises the real code path but has **not yet been run
on a CUDA GPU**, and CI has no GPU. GPU correctness and performance remain
unverified. **No speedup figures are published.** Results belong in
``benchmarks/results.json`` with hardware and software provenance.

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
   $ python tools/bootstrap_commit_skill.py --check

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
devices, offsets outside the record range, sums that could overflow int64,
and buckets beyond the one-pass limit on CUDA. Nothing is silently copied
between devices or truncated. ``tapeforge.reference.aggregate_reference`` is
the torch-free NumPy API.

Command line:

.. code-block:: console

   $ python -m tapeforge.generate --seed 7 --seconds 10 --profile burst
   $ python -m tapeforge.ingest --symbol BTCUSDT --date 2025-01-02

Downloaded archives land in ``data/`` and are never committed.

Benchmarks
==========

Three scopes are timed separately and never collapsed into one headline
number:

1. ``benchmarks/kernel.py`` -- inputs already resident on the device. It
   compares ``ops.aggregate`` on CUDA (validation plus the one-pass Triton
   kernel), the bare kernel launch with validation done outside the timed
   region, an eager PyTorch baseline (``searchsorted`` plus ``index_add_``),
   the same baseline under ``torch.compile``, and the NumPy reference on the
   host. Workloads with buckets beyond the one-pass limit are recorded as
   unsupported for the Triton path. Without CUDA it runs the CPU-capable
   implementations as a smoke test and records nothing.
2. ``benchmarks/pipeline.py`` -- host-to-device copy, compute and
   device-to-host copy, timed separately and end to end, from pageable and
   from pinned host memory, against the NumPy reference. It needs CUDA. The
   ``pinned_streams`` variant is not implemented yet.
3. ``benchmarks/train.py`` -- the whole training loop. Not implemented yet.

Every implementation must match the CPU reference before it is timed:
integers exactly, VWAP within the contract tolerance. Timing is wall clock
around each call, after untimed warm-up calls, with
``torch.cuda.synchronize()`` before and after every timed call on CUDA.
Configurations are versioned TOML under ``benchmarks/configs/``. Results are
appended to ``benchmarks/results.json`` with the workload, the timing method,
and hardware and software provenance (GPU, CUDA, torch, Triton, NumPy, git
commit and whether the tree had uncommitted changes). Negative results are
published too.

.. code-block:: console

   $ python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record   # quick check
   $ python benchmarks/kernel.py                                         # full sweep
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
     kernels.py     Triton kernels: one-pass bucket reduction and its launcher
     validation.py  tensor input checks shared by ops and the kernel launcher
     ops.py         PyTorch-facing API: CUDA -> Triton, CPU -> reference
     model.py       small volatility-regime classifier
   tests/           parity, edge-case and stress suites
   benchmarks/      harness, three timed scopes, versioned configs, results
   tools/           commit skill bootstrap and safe local commit helper

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
