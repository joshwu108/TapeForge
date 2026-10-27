"""Shared pytest fixtures, capability probes and GPU parity helpers.

Owns the two-record reference example (``TOY_EXPECTED``) and the parity
rule used by every CUDA test, so CPU and GPU suites assert against the same
numbers and the same oracle.
"""

from __future__ import annotations

import pytest


def _cuda_triton_ready() -> tuple[bool, str]:
    """Report whether CUDA and Triton are both usable, with a skip reason when not."""
    try:
        import torch
    except ImportError:
        return False, "torch is not installed (CPU-only phase)"
    if not torch.cuda.is_available():
        return False, "no CUDA device available"
    from tapeforge import kernels

    if not kernels.TRITON_AVAILABLE:
        return False, "CUDA is available but triton is not installed (pip install -e '.[gpu]')"
    return True, ""


@pytest.fixture(scope="session")
def gpu_ready() -> None:
    """Skip a GPU test with a precise reason instead of failing on a CPU-only host."""
    ready, reason = _cuda_triton_ready()
    if not ready:
        pytest.skip(reason)


# The two-record reference example (also exercised by test_program.py), kept
# in one place so CPU and GPU tests assert against identical numbers.
#   bucket at 10:00:01: buy-initiated price 100 qty 2, sell-initiated price 102 qty 1
#   -> event_count 2, total_quantity 3, signed_quantity +1, vwap 100.666...
TOY_EXPECTED = {
    "event_count": 2,
    "total_quantity": 3.0,
    "signed_quantity": 1.0,
    "vwap": (100 * 2 + 102 * 1) / 3,
    "valid": True,
}


# --- GPU parity helpers ----------------------------------------------------
#
# Shared by test_gpu_parity.py (one-pass path), test_two_pass.py (buckets
# beyond the one-pass tile) and test_prepared.py. All compare the real CUDA
# code path against reference.aggregate_reference_naive, the plain-loop
# oracle, so the GPU is never checked against code that shares its
# vectorization tricks.
#
# Parity rule: counts and quantity sums exactly; VWAP within
# contract.VWAP_RTOL / VWAP_ATOL; identical dtypes; outputs on the GPU.

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from tapeforge import contract, reference  # noqa: E402


def cuda_inputs(trades: np.ndarray, offsets: np.ndarray) -> tuple[dict[str, Any], Any]:
    """Return ``trades`` as column tensors and ``offsets`` as a tensor on the CUDA device."""
    import torch

    columns = {name: torch.as_tensor(trades[name].copy()).cuda() for name in trades.dtype.names}
    return columns, torch.as_tensor(offsets).cuda()


def assert_gpu_parity(got: dict[str, Any], trades: np.ndarray, offsets: np.ndarray) -> None:
    """Assert the parity rule above between CUDA outputs and the plain-loop oracle.

    Args:
        got: Feature tensors returned by the CUDA path, keyed by feature name.
        trades: Host-side trade records in ``contract.TRADE_DTYPE``.
        offsets: Host-side bucket offsets matching ``trades``.
    """
    expected = reference.aggregate_reference_naive(trades, offsets)
    assert tuple(got) == contract.FEATURE_NAMES
    for name in contract.FEATURE_NAMES:
        actual = got[name]
        assert actual.is_cuda, name
        actual = actual.cpu().numpy()
        target = getattr(expected, name)
        assert actual.dtype == target.dtype, (name, actual.dtype, target.dtype)
        if name == "vwap":
            np.testing.assert_allclose(
                actual, target, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL, err_msg=name
            )
        else:
            np.testing.assert_array_equal(actual, target, err_msg=name)
