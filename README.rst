=========
TapeForge
=========

GPU-native market-data feature engine: segmented reductions over unevenly
sized time windows, without padding every window to the longest one.

TapeForge turns millions of timestamped aggregated-trade records into
per-bucket feature tensors for machine learning. The engineering idea is a
fused Triton kernel that walks a window-boundary index (``offsets``) instead
of a padded ``[n_buckets, max_len]`` tensor, so sparse seconds cost nothing
and 10,000-record bursts stay on one device pass.

**What this is not:** a trading strategy, an exchange connector, a promise of
predictive returns, or a production risk system. It is an ML-systems
throughput project, measured as such.

Status
======

Weeks 0-2 complete: project scaffold and data contract, seeded trade
generator, public ``aggTrades`` ingest, the no-padding window index, and the
CPU reference features with their test suite. The GPU engine is not written
yet, so **no speedup figures are published**. Any number that appears in
``benchmarks/results.json`` was produced by the committed benchmark scripts on
the hardware recorded in the same entry.

Measured so far, on BTCUSDT 2025-01-02 (1,299,165 aggregate-trade records):

============================================  =============================
measurement                                   result
============================================  =============================
window index, 1.3M records to 86,400 buckets  3.2 ms
CPU reference features, same input            7 ms (~185M records/s)
padded ``[n_buckets, max_len]`` equivalent    2.78 GB vs 31 MB packed (89x)
busiest second                                1,343 records
============================================  =============================

The 89x figure is the design argument for the offsets index, and it holds on
real data with only a mild tail. The 7 ms figure is the honest baseline a GPU
kernel has to beat, before host-to-device transfer is even counted.

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

   $ python -m pytest -q                    # fast suite; GPU tests skip with a reason
   $ python -m pytest -q -m gpu             # parity suite, needs CUDA + Triton
   $ python -m pytest -q -m slow            # stress suite
   $ python tools/bootstrap_commit_skill.py --check

Usage
=====

.. code-block:: python

   from tapeforge import generate, ops, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=1000, profile="burst")
   offsets = windows.build_offsets(trades["timestamp"], 0, 1000, DEFAULT_BUCKET_WIDTH_US)
   features = ops.aggregate(trades, offsets)   # dict of (n_buckets,) tensors

Command line:

.. code-block:: console

   $ python -m tapeforge.generate --seed 7 --seconds 10 --profile burst
   $ python -m tapeforge.ingest --symbol BTCUSDT --date 2025-01-02

Downloaded archives land in ``data/`` and are never committed.

Benchmarks
==========

Three scopes are timed separately and never collapsed into one headline
number:

1. ``benchmarks/kernel.py`` -- kernel only, inputs already resident on the
   device, against eager PyTorch and ``torch.compile``.
2. ``benchmarks/pipeline.py`` -- host-to-device transfer plus the feature
   pipeline, reporting the copy-versus-compute split.
3. ``benchmarks/train.py`` -- the whole training loop: examples/s, GPU wait,
   peak VRAM.

Configurations are versioned TOML under ``benchmarks/configs/``; results,
with hardware and environment provenance, are appended to
``benchmarks/results.json``. Negative results are published too.

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
     kernels.py     Triton kernels
     ops.py         PyTorch-facing API
     model.py       small volatility-regime classifier
   tests/           parity, edge-case and stress suites
   benchmarks/      harness, three timed scopes, versioned configs, results
   tools/           commit skill bootstrap and safe local commit helper

Documentation policy
====================

This file is the public documentation. The repository tracks no Markdown:
planning documents, design notes and PRDs stay local under ``planning/``.
Shared results are ``.json``, ``.csv`` or ``.rst``.

License
=======

MIT.
