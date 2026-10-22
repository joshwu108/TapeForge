"""Triton GPU kernels"""

from __future__ import annotations

TRITON_AVAILABLE: bool
try:  # pragma: no cover - import probe
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only machine (weeks 1-2)
    TRITON_AVAILABLE = False

# TODO(student, week 3): implement after running the official Triton
# vector-add tutorial. Leave this NotImplementedError in place until then --
# the parity tests and benchmarks detect it and skip.
