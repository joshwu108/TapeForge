"""TapeForge: GPU-native market-data feature engine.

Turns timestamped aggregated-trade records into fixed-width time-bucket
feature tensors using segmented reductions over uneven windows, without
padding every bucket to the maximum length.

Layer map (see also README.rst and the architecture diagram in the PRD):

    generate.py   seeded synthetic trade generation (deterministic fixtures)
    ingest.py     public aggTrades loading, normalization, validation
    contract.py   data contract: dtypes, sign convention, scaling, tolerances
    windows.py    bucket boundary offsets (the no-padding window index)
    reference.py  readable NumPy/CPU reference features (the oracle)
    kernels.py    Triton GPU kernels  -- STUDENT-OWNED, see module docstring
    ops.py        PyTorch-facing API returning typed tensors
    model.py      small volatility-regime classifier over the features

Nothing in this package is a trading strategy, an exchange connector, or a
claim about predictive returns. It is an ML-systems throughput project.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
