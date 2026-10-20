"""Weeks 3-4: the GPU engine must agree with the CPU reference (TF-05, TF-06).

This file is the *harness* for the student's kernels. It is agent-owned
plumbing; the kernels it exercises are not. It skips with a precise reason
on a machine without CUDA + Triton, or while ``kernels.py`` is unimplemented.

Parity rule (PRD section 6): integer counts and signed sums must match
exactly; VWAP within contract.VWAP_RTOL / VWAP_ATOL.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate, ops, reference, windows

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu_reference(gpu_ready: None, profile: str, seed: int) -> None:
    import torch

    trades = generate.generate_trades(seed=seed, n_buckets=1000, profile=profile)
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 1000, contract.DEFAULT_BUCKET_WIDTH_US
    )
    expected = reference.aggregate_reference(trades, offsets)

    got = ops.aggregate(
        {k: torch.as_tensor(trades[k].copy()).cuda() for k in trades.dtype.names},
        torch.as_tensor(offsets).cuda(),
    )

    np.testing.assert_array_equal(got["event_count"].cpu().numpy(), expected.event_count)
    np.testing.assert_array_equal(got["total_quantity"].cpu().numpy(), expected.total_quantity)
    np.testing.assert_array_equal(got["signed_quantity"].cpu().numpy(), expected.signed_quantity)
    np.testing.assert_array_equal(got["valid"].cpu().numpy(), expected.valid)

    valid = expected.valid
    np.testing.assert_allclose(
        got["vwap"].cpu().numpy()[valid],
        expected.vwap[valid],
        rtol=contract.VWAP_RTOL,
        atol=contract.VWAP_ATOL,
    )


def test_oversized_bucket_takes_the_two_pass_path(gpu_ready: None) -> None:
    """TF-06: one bucket far larger than a single program should handle."""
    pytest.skip("week 4: enable once the student's two-pass reduction exists")


def test_all_empty_buckets_do_not_crash(gpu_ready: None) -> None:
    pytest.skip("week 4: enable once the student's kernel exists")
