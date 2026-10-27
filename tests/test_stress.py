"""Tests for ``tapeforge.ops`` at scale: a million records per profile through the gate.

Oracle: ``reference.aggregate_reference_naive`` on the CPU path and
``reference.aggregate_reference`` on the GPU path. Slow: run with
``python -m pytest -q -m slow``. The CPU cases run everywhere; the GPU cases
additionally need the ``gpu_ready`` fixture and skip with its reason otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate, kernels, ops, reference, windows

pytestmark = pytest.mark.slow

W = contract.DEFAULT_BUCKET_WIDTH_US

# Mean records per bucket; burst is the measured mean of the generator, which
# only needs to land the record count in the right order of magnitude.
_RECORDS_PER_BUCKET = {"sparse": 0.5, "uniform": 50.0, "burst": 100.0}


def _workload(n_records: int, profile: str, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Generate roughly ``n_records`` trades of ``profile`` and offsets spanning every bucket."""
    n_buckets = max(1, round(n_records / _RECORDS_PER_BUCKET[profile]))
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile=profile)
    return trades, windows.build_offsets(trades["timestamp"], 0, n_buckets, W)


def _assert_same(got: dict[str, object], expected: reference.Features) -> None:
    """Assert counts, sums and validity exactly and VWAP within the contract tolerances."""
    for name in ("event_count", "total_quantity", "signed_quantity", "valid"):
        np.testing.assert_array_equal(got[name].cpu().numpy(), getattr(expected, name))  # type: ignore[attr-defined]
    np.testing.assert_allclose(
        got["vwap"].cpu().numpy(),  # type: ignore[attr-defined]
        expected.vwap,
        rtol=contract.VWAP_RTOL,
        atol=contract.VWAP_ATOL,
    )


@pytest.mark.parametrize("n_records", [100_000, 1_000_000])
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
def test_no_incorrect_features_at_scale(n_records: int, profile: str) -> None:
    """The CPU path matches the plain-loop oracle and conserves records at scale."""
    pytest.importorskip("torch")
    trades, offsets = _workload(n_records, profile)
    got = ops.aggregate(trades, offsets)
    expected = reference.aggregate_reference_naive(trades, offsets)
    _assert_same(got, expected)

    # Conservation: every record in the window is counted exactly once.
    window = slice(int(offsets[0]), int(offsets[-1]))
    assert int(got["event_count"].sum()) == offsets[-1] - offsets[0]
    assert int(got["total_quantity"].sum()) == int(trades["quantity"][window].sum())
    assert bool((got["signed_quantity"].abs() <= got["total_quantity"]).all())


@pytest.mark.gpu
@pytest.mark.parametrize("n_records", [100_000, 1_000_000])
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
def test_gpu_no_incorrect_features_at_scale(
    gpu_ready: None, n_records: int, profile: str
) -> None:
    """The CUDA path matches the reference on both dispatch paths at scale."""
    import torch

    trades, offsets = _workload(n_records, profile)
    columns = {k: torch.as_tensor(trades[k].copy()).cuda() for k in trades.dtype.names}
    gpu_offsets = torch.as_tensor(offsets).cuda()
    # Burst workloads carry buckets beyond the one-pass tile and take the
    # two-pass path; sparse and uniform stay one-pass. Assert which, so the
    # suite is known to reach the parity comparison on both paths.
    longest = int(np.diff(offsets).max())
    expected_path = "two_pass" if profile == "burst" else "one_pass"
    assert kernels.plan_dispatch(longest).path == expected_path, (profile, longest)
    expected = reference.aggregate_reference(trades, offsets)
    _assert_same(ops.aggregate(columns, gpu_offsets), expected)


def _real_magnitude_bucket(n: int) -> np.ndarray:
    """Build ``n`` records at the BTCUSDT 2025-01-02 extremes: highest price, largest quantity.

    Sides alternate so the signed quantity sums to zero for even ``n``.
    """
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = round(97_839.50 * contract.PRICE_SCALE)
    trades["quantity"] = round(18.89042 * contract.QUANTITY_SCALE)
    trades["buyer_is_maker"] = np.arange(n) % 2 == 1
    return trades


def test_integer_overflow_bound_is_respected_at_extremes() -> None:
    """400k worst-case records in one second stay exact; 1M would wrap and must raise."""
    pytest.importorskip("torch")
    n = 400_000
    trades = _real_magnitude_bucket(n)
    offsets = np.array([0, n], dtype=np.int64)
    price, qty = int(trades["price"][0]), int(trades["quantity"][0])
    assert n * price * qty <= contract.INT64_MAX

    got = ops.aggregate(trades, offsets)
    assert int(got["total_quantity"][0]) == n * qty
    assert int(got["signed_quantity"][0]) == 0
    assert float(got["vwap"][0]) == pytest.approx(price / contract.PRICE_SCALE,
                                                  rel=contract.VWAP_RTOL)

    n_big = 1_000_000
    assert n_big * price * qty > contract.INT64_MAX
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        ops.aggregate(_real_magnitude_bucket(n_big), np.array([0, n_big], dtype=np.int64))
