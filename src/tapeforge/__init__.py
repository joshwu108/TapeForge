"""TapeForge: GPU-native market-data feature engine.

The package computes exact per-bucket trade features (event count, total and
taker-signed quantity, VWAP and validity) from timestamp-sorted aggregate-trade
records. ``tapeforge.contract`` fixes the dtypes, units and tolerances;
``tapeforge.reference`` is the NumPy oracle; ``tapeforge.ops`` and
``tapeforge.kernels`` form the torch/Triton path; ``tapeforge.streaming`` is
the chunked CPU path; ``tapeforge.ingest`` and ``tapeforge.generate`` produce
input tapes.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
