"""Shared test fixtures and capability probes."""

from __future__ import annotations

import pytest


def _cuda_triton_ready() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed (CPU-only phase)"
    if not torch.cuda.is_available():
        return False, "no CUDA device available"
    from tapeforge import kernels

    if not kernels.TRITON_AVAILABLE:
        return False, "triton is not installed"
    if not hasattr(kernels, "agg_kernel_one_pass"):
        return False, "student's Triton kernel is not implemented yet (week 3)"
    return True, ""


@pytest.fixture(scope="session")
def gpu_ready() -> None:
    """Skip a GPU test with a precise reason instead of failing on a CPU box."""
    ready, reason = _cuda_triton_ready()
    if not ready:
        pytest.skip(reason)


# The PRD section 3 toy example, kept in one place so CPU and GPU tests
# assert against identical hand-calculated numbers.
#   bucket at 10:00:01: buy-initiated price 100 qty 2, sell-initiated price 102 qty 1
#   -> event_count 2, total_quantity 3, signed_quantity +1, vwap 100.666...
TOY_EXPECTED = {
    "event_count": 2,
    "total_quantity": 3.0,
    "signed_quantity": 1.0,
    "vwap": (100 * 2 + 102 * 1) / 3,
    "valid": True,
}
