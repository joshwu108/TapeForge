"""Parity cases for buckets longer than one tile: the two-pass reduction.

These tests pin the contract the two-pass path must meet before it exists.
Each case drives ``ops.aggregate`` on CUDA tensors and compares every output
with the plain-loop oracle, exactly like test_gpu_parity.py. Cases whose
longest bucket fits the one-pass tile run (and must pass) today; cases beyond
``kernels.MAX_ONE_PASS_BUCKET`` carry the strict ``TWO_PASS_MISSING`` marker
and start failing (XPASS) the day the reduction lands, at which point the
marker is removed and the file becomes an ordinary parity suite.

What is covered, per the design notes for the two-pass path:

  * chunk boundaries: bucket lengths CHUNK-1, CHUNK, CHUNK+1, 2*CHUNK and
    3*CHUNK+17 (a ragged last chunk) for every plausible CHUNK;
  * mixed-size and empty buckets side by side, so chunk offsets repeat;
  * clipped windows: records before offsets[0] and after offsets[-1] with
    long buckets inside, and a long bucket that is the very last one;
  * exact integer partial sums near the int64 bound, so combining partials
    cannot lose a bit;
  * the overflow bound still refusing a long bucket that could wrap;
  * the partial-VWAP averaging trap: two halves of one bucket with different
    quantities, where a mean of per-chunk VWAPs is wrong and only integer
    partials divided once give the right answer.

The last trap is also pinned on the CPU here, so the oracle the GPU is held
to is itself checked against hand arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TWO_PASS_MISSING, assert_gpu_parity, cuda_inputs

from tapeforge import contract, kernels, ops, reference, windows

W = contract.DEFAULT_BUCKET_WIDTH_US
P = contract.PRICE_SCALE
Q = contract.QUANTITY_SCALE
LIMIT = kernels.MAX_ONE_PASS_BUCKET

# Tile sizes the two-pass path might reasonably chunk by. The boundary cases
# below are generated for each, so the suite does not depend on which one
# the implementation picks.
CHUNK_CANDIDATES = (1024, 2048, 4096)


def _needs_two_pass(longest: int) -> list[pytest.MarkDecorator]:
    return [TWO_PASS_MISSING] if longest > LIMIT else []


def _buckets(lengths: list[int], before: int = 0, after: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Trades with ``lengths[b]`` records in bucket ``b``, plus records outside the window.

    Prices and quantities vary record by record so that any chunk-sized
    sub-range has a different mean price and a different total quantity from
    its neighbours: a reduction that mixes partials up, drops a chunk, or
    averages partial VWAPs cannot match the oracle by luck.
    """
    n_buckets = len(lengths)
    n = before + sum(lengths) + after
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    seconds = np.concatenate(
        [
            np.full(before, -1, dtype=np.int64),
            np.repeat(np.arange(n_buckets, dtype=np.int64), lengths),
            np.full(after, n_buckets + 3, dtype=np.int64),
        ]
    )
    i = np.arange(n, dtype=np.int64)
    trades["timestamp"] = seconds * W + (i % 997) * 1_000
    trades["timestamp"].sort()  # the per-record jitter must not reorder buckets
    trades["price"] = 100 * P + i % 97 + (i // 1000) * 3
    trades["quantity"] = 1 + i % 1009 + (i // 512) * 7
    trades["buyer_is_maker"] = i % 3 == 0
    trades["agg_trade_id"] = i
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, W)
    assert np.diff(offsets).tolist() == lengths, "fixture built wrong bucket lengths"
    assert offsets[0] == before and n - offsets[-1] == after
    return trades, offsets


def _trap_bucket(half: int) -> tuple[np.ndarray, np.ndarray]:
    """One bucket of ``2 * half`` records: first half 1 unit at 100, second half 3 units at 200.

    True VWAP = (100 * half + 200 * 3 * half) / (4 * half) = 175 for every
    ``half``. The mean of the two half-VWAPs is 150. The two halves are exactly
    one chunk each when CHUNK == half, which is the case the trap targets.
    """
    n = 2 * half
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = np.where(np.arange(n) < half, 100 * P, 200 * P)
    trades["quantity"] = np.where(np.arange(n) < half, 1 * Q, 3 * Q)
    return trades, np.array([0, n], dtype=np.int64)


def _gpu(trades: np.ndarray, offsets: np.ndarray) -> None:
    assert_gpu_parity(ops.aggregate(*cuda_inputs(trades, offsets)), trades, offsets)


# --- the oracle itself, on the CPU -----------------------------------------


@pytest.mark.parametrize("half", CHUNK_CANDIDATES)
def test_oracle_resolves_the_averaging_trap_to_175(half: int) -> None:
    trades, offsets = _trap_bucket(half)
    got = reference.aggregate_reference_naive(trades, offsets)
    assert got.vwap[0] == 175.0
    assert got.total_quantity[0] == 4 * half * Q
    mean_of_partial_vwaps = (100.0 + 200.0) / 2
    assert mean_of_partial_vwaps == 150.0 != got.vwap[0]


def test_fixture_builder_places_records_outside_the_window() -> None:
    trades, offsets = _buckets([3, 0, 5000, 2], before=4, after=6)
    assert offsets.tolist() == [4, 7, 7, 5007, 5009]
    assert trades.size == 4 + 5005 + 6


# --- chunk boundaries --------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "length",
    [
        pytest.param(n, id=f"{chunk}{sign}", marks=_needs_two_pass(n))
        for chunk in CHUNK_CANDIDATES
        for sign, n in (
            ("-1", chunk - 1),
            ("", chunk),
            ("+1", chunk + 1),
            ("x2", 2 * chunk),
            ("x3+17", 3 * chunk + 17),
        )
    ],
)
def test_single_bucket_around_each_chunk_boundary(gpu_ready: None, length: int) -> None:
    _gpu(*_buckets([length]))


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_one_bucket_longer_than_the_real_busiest_second_by_far(gpu_ready: None) -> None:
    """28,000 records in one bucket: the measured tail of the burst generator."""
    _gpu(*_buckets([28_000]))


# --- mixed-size and empty buckets ----------------------------------------


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_long_buckets_interleaved_with_empty_and_short_ones(gpu_ready: None) -> None:
    lengths = [0, 5, 9_000, 0, 0, LIMIT + 1, 1, 0, 12_345, 3, 0, 0]
    _gpu(*_buckets(lengths))


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_every_bucket_is_long(gpu_ready: None) -> None:
    _gpu(*_buckets([LIMIT + 1, 2 * LIMIT, LIMIT + 1023, 3 * LIMIT + 1]))


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_all_records_in_the_last_bucket_which_is_long(gpu_ready: None) -> None:
    _gpu(*_buckets([0, 0, 0, 0, 0, 10_000]))


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_all_records_in_the_first_bucket_which_is_long(gpu_ready: None) -> None:
    _gpu(*_buckets([10_000, 0, 0, 0]))


# --- clipped windows -------------------------------------------------------


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_records_before_and_after_the_window_with_long_buckets_inside(gpu_ready: None) -> None:
    _gpu(*_buckets([7, LIMIT + 5, 0, 2 * LIMIT + 1], before=1_500, after=6_000))


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_long_bucket_ending_exactly_at_the_last_record(gpu_ready: None) -> None:
    _gpu(*_buckets([2, 0, LIMIT + 1], before=3, after=0))


@pytest.mark.gpu
def test_records_before_and_after_the_window_with_one_pass_buckets_inside(gpu_ready: None) -> None:
    """The clipped-window case that runs today; must keep passing after dispatch changes."""
    _gpu(*_buckets([7, 1_500, 0, 4_000], before=1_500, after=6_000))


# --- exact partial sums and the overflow bound ---------------------------


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_long_bucket_just_inside_the_int64_bound_is_exact(gpu_ready: None) -> None:
    """Every partial is a sub-sum of a bounded bucket sum; recombination must lose nothing."""
    n = 2 * LIMIT
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30)
    trades["buyer_is_maker"] = np.arange(n) % 2 == 0
    offsets = np.array([0, n], dtype=np.int64)
    assert n * 2**30 * int(trades["quantity"][0]) <= contract.INT64_MAX
    got = ops.aggregate(*cuda_inputs(trades, offsets))
    assert got["total_quantity"].item() == int(trades["quantity"].sum())
    assert got["signed_quantity"].item() == 0
    assert_gpu_parity(got, trades, offsets)


@pytest.mark.gpu
def test_long_bucket_past_the_int64_bound_raises_overflow_not_size_error(gpu_ready: None) -> None:
    """Validation runs before any tile or chunk decision: the overflow check wins."""
    n = 2 * LIMIT
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30) + 1
    offsets = np.array([0, n], dtype=np.int64)
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        ops.aggregate(*cuda_inputs(trades, offsets))


# --- the partial-VWAP averaging trap ---------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "half",
    [pytest.param(c, id=str(c), marks=_needs_two_pass(2 * c)) for c in CHUNK_CANDIDATES],
)
def test_vwap_of_two_unequal_halves_is_175_not_150(gpu_ready: None, half: int) -> None:
    trades, offsets = _trap_bucket(half)
    got = ops.aggregate(*cuda_inputs(trades, offsets))
    vwap = got["vwap"].item()
    assert vwap == pytest.approx(175.0, rel=contract.VWAP_RTOL, abs=contract.VWAP_ATOL)
    assert abs(vwap - 150.0) > 1.0, "VWAP looks like a mean of per-chunk VWAPs"
    assert_gpu_parity(got, trades, offsets)


@pytest.mark.gpu
@TWO_PASS_MISSING
def test_ragged_last_chunk_cannot_be_averaged_either(gpu_ready: None) -> None:
    """Three full chunks of 1 unit at 100, then 17 records of 3 units at 200.

    True VWAP = (100*3C + 200*51) / (3C + 51); a mean over four chunk VWAPs
    would be 125, far off for every C.
    """
    for chunk in CHUNK_CANDIDATES:
        n_head = 3 * chunk
        n = n_head + 17
        trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
        trades["price"] = np.where(np.arange(n) < n_head, 100 * P, 200 * P)
        trades["quantity"] = np.where(np.arange(n) < n_head, 1 * Q, 3 * Q)
        offsets = np.array([0, n], dtype=np.int64)
        got = ops.aggregate(*cuda_inputs(trades, offsets))
        expected = (100 * n_head + 200 * 51) / (n_head + 51)
        assert got["vwap"].item() == pytest.approx(expected, rel=contract.VWAP_RTOL)
        assert_gpu_parity(got, trades, offsets)
