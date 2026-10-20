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

Week 0 complete: project scaffold, data contract, test and benchmark
harnesses, commit tooling. Feature and kernel implementation is in progress;
no performance numbers are published yet. Any number that appears in
``benchmarks/results.json`` was produced by the committed benchmark scripts
on the hardware recorded in the same entry.

Features
========

For every fixed-width time bucket (1 second by default):

==================  ==========================================================
``event_count``     number of aggregate-trade records in the bucket (exact)
``total_quantity``  sum of traded quantity (exact, scaled integer)
``signed_quantity`` buy-initiated minus sell-initiated quantity (exact)
``vwap``            quantity-weighted average price (float, documented tolerance)
``valid``           false when the bucket contained no records
==================  ==========================================================

A row of the input is an *aggregate* trade record, not an individual fill.
Record counts in this project always mean aggregate-trade records. The input
carries no order book, so none of these are order-book imbalance features.

Units, the taker-side sign convention derived from ``buyer_is_maker``, the
price/quantity integer scaling, the overflow bound and the VWAP tolerance are
all defined in one place: ``src/tapeforge/contract.py``.

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
     reference.py   readable NumPy reference features (the oracle)
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
