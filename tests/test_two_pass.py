"""Tests for the two-pass reduction in ``tapeforge.kernels``: buckets longer than one tile.

Each GPU case drives ``ops.aggregate`` on CUDA tensors and compares every
output with the plain-loop oracle, exactly like test_gpu_parity.py, and
asserts through ``kernels.plan_dispatch`` which path its fixture takes, so a
case meant for the two-pass kernels cannot silently pass through the
one-pass tile (or the reverse). A spy test additionally checks that the
two-pass launcher is the one invoked for a long bucket.

What is covered:

  * chunk boundaries: bucket lengths CHUNK-1, CHUNK, CHUNK+1, 2*CHUNK and
    3*CHUNK+17 (a ragged last chunk) for the configured ``kernels.CHUNK_SIZE``
    and the other plausible chunk sizes; lengths that fit a one-pass tile are
    additionally pushed through the two-pass kernels with
    ``prepared.prepare(path="two_pass")``, otherwise the configured chunk
    size would never be exercised on a single bucket;
  * mixed-size and empty buckets side by side, so chunk offsets repeat;
  * clipped windows: records before offsets[0] and after offsets[-1] with
    long buckets inside, and a long bucket that is the very last one;
  * pass-2 accumulation over several tiles of partials: buckets with more
    chunks than ``kernels.COMBINE_BLOCK_SIZE``, including a ragged last tile
    and two such buckets separated by empty and short ones;
  * signed quantity on long buckets (mixed, all sell-initiated), zero
    quantity throughout a long bucket and in whole leading chunks;
  * exact integer partial sums near the int64 bound, so combining partials
    cannot lose a bit, and the overflow bound still refusing a long bucket
    that could wrap;
  * the partial-VWAP averaging trap: two halves of one bucket with different
    quantities, where a mean of per-chunk VWAPs is wrong and only integer
    partials divided once give the right answer.

The last trap is also pinned on the CPU here, so the oracle the GPU is held
to is itself checked against closed-form arithmetic. Host-side schedule logic
(``build_chunk_metadata``, ``plan_dispatch``) is tested on the CPU in
test_kernels.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import assert_gpu_parity, cuda_inputs

from tapeforge import contract, kernels, ops, prepared, reference, windows

W = contract.DEFAULT_BUCKET_WIDTH_US
P = contract.PRICE_SCALE
Q = contract.QUANTITY_SCALE
LIMIT = kernels.MAX_ONE_PASS_BUCKET
CHUNK = kernels.CHUNK_SIZE
TILE = kernels.COMBINE_BLOCK_SIZE

# Chunk sizes the boundary cases are generated for: the configured one plus
# the other plausible powers of two, so the suite keeps its coverage if the
# kernel maintainer retunes CHUNK_SIZE.
CHUNK_CANDIDATES = tuple(sorted({CHUNK, 1024, 2048, 4096}))


def _buckets(lengths: list[int], before: int = 0, after: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build trades with ``lengths[b]`` records in bucket ``b``, plus records outside the window.

    Prices and quantities vary record by record so that any chunk-sized
    sub-range has a different mean price and a different total quantity from
    its neighbours: a reduction that mixes partials up, drops a chunk, or
    averages partial VWAPs cannot match the oracle by luck. Every third
    record is sell-initiated, so signed quantity differs from total quantity.

    Args:
        lengths: Record count of each window bucket, in order.
        before: Records placed one second before the window start.
        after: Records placed three seconds after the window end.

    Returns:
        The trade array and the offsets for the window ``[0, len(lengths) * W)``.
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
    trades["timestamp"].sort()  # The per-record jitter must not reorder buckets.
    trades["price"] = 100 * P + i % 97 + (i // 1000) * 3
    trades["quantity"] = 1 + i % 1009 + (i // 512) * 7
    trades["buyer_is_maker"] = i % 3 == 0
    trades["agg_trade_id"] = i
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, W)
    assert np.diff(offsets).tolist() == lengths, "fixture built wrong bucket lengths"
    assert offsets[0] == before and n - offsets[-1] == after
    return trades, offsets


def _trap_bucket(half: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one bucket of ``2 * half`` records: 1 unit at 100 per record, then 3 units at 200.

    True VWAP = (100 * half + 200 * 3 * half) / (4 * half) = 175 for every
    ``half``. The mean of the two half-VWAPs is 150. The two halves are exactly
    one chunk each when CHUNK == half, which is the case the trap targets.
    """
    n = 2 * half
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = np.where(np.arange(n) < half, 100 * P, 200 * P)
    trades["quantity"] = np.where(np.arange(n) < half, 1 * Q, 3 * Q)
    return trades, np.array([0, n], dtype=np.int64)


def _path_of(offsets: np.ndarray) -> str:
    """Return the dispatch path the automatic plan picks for the longest bucket."""
    return kernels.plan_dispatch(int(np.diff(offsets).max())).path


def _gpu(trades: np.ndarray, offsets: np.ndarray, path: str = "two_pass") -> dict[str, Any]:
    """Run the public API on CUDA, assert the dispatch path, and check parity."""
    assert _path_of(offsets) == path, f"fixture takes the {_path_of(offsets)} path, not {path}"
    got = ops.aggregate(*cuda_inputs(trades, offsets))
    assert_gpu_parity(got, trades, offsets)
    return got


def _both_paths(trades: np.ndarray, offsets: np.ndarray) -> dict[str, Any]:
    """Run the public API on its automatic path, forcing two-pass as well when auto is one-pass.

    Boundary and trap fixtures sized around ``CHUNK`` fit a one-pass tile
    whenever ``CHUNK`` is well below ``MAX_ONE_PASS_BUCKET``; forcing the
    two-pass path through ``tapeforge.prepared`` is the only way those sizes
    reach the chunk kernels. Returns the two-pass result.
    """
    columns, dev_offsets = cuda_inputs(trades, offsets)
    auto = ops.aggregate(columns, dev_offsets)
    assert_gpu_parity(auto, trades, offsets)
    if _path_of(offsets) == "two_pass":
        return auto
    forced = prepared.prepare(columns, dev_offsets, path="two_pass")
    assert forced.path == "two_pass" and forced.n_chunks > 0
    got = forced.execute()
    assert_gpu_parity(got, trades, offsets)
    return got


# --- the oracle itself, on the CPU -----------------------------------------


@pytest.mark.parametrize("half", CHUNK_CANDIDATES)
def test_oracle_resolves_the_averaging_trap_to_175(half: int) -> None:
    """The plain-loop oracle itself gives 175, not the 150 a mean of half-VWAPs would."""
    trades, offsets = _trap_bucket(half)
    got = reference.aggregate_reference_naive(trades, offsets)
    assert got.vwap[0] == 175.0
    assert got.total_quantity[0] == 4 * half * Q
    mean_of_partial_vwaps = (100.0 + 200.0) / 2
    assert mean_of_partial_vwaps == 150.0 != got.vwap[0]


def test_fixture_builder_places_records_outside_the_window() -> None:
    """``_buckets`` puts ``before`` records ahead of the window and ``after`` records past it."""
    trades, offsets = _buckets([3, 0, 5000, 2], before=4, after=6)
    assert offsets.tolist() == [4, 7, 7, 5007, 5009]
    assert trades.size == 4 + 5005 + 6


def test_configured_chunk_size_is_among_the_boundary_candidates() -> None:
    """``CHUNK_SIZE`` is a power of two within the one-pass limit and is always exercised."""
    assert CHUNK in CHUNK_CANDIDATES
    assert CHUNK & (CHUNK - 1) == 0 and CHUNK <= LIMIT


# --- which launcher actually runs ------------------------------------------


@pytest.mark.gpu
def test_two_pass_launcher_runs_for_a_long_bucket_and_one_pass_for_a_short_one(
    gpu_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher the dispatch plan names is the one that executes, verified by spies."""
    calls: list[str] = []
    real_one, real_two = kernels.launch_one_pass_unchecked, kernels.launch_two_pass_unchecked

    def spy_one(*args: Any, **kwargs: Any) -> None:
        calls.append("one_pass")
        real_one(*args, **kwargs)

    def spy_two(*args: Any, **kwargs: Any) -> None:
        calls.append("two_pass")
        real_two(*args, **kwargs)

    monkeypatch.setattr(kernels, "launch_one_pass_unchecked", spy_one)
    monkeypatch.setattr(kernels, "launch_two_pass_unchecked", spy_two)

    _gpu(*_buckets([LIMIT + 1]), path="two_pass")
    assert calls == ["two_pass"]
    _gpu(*_buckets([LIMIT]), path="one_pass")
    assert calls == ["two_pass", "one_pass"]


# --- chunk boundaries --------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "length",
    [
        pytest.param(n, id=f"{chunk}{sign}")
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
    """Lengths at and around every chunk multiple, through the two-pass kernels.

    Lengths within the one-pass limit also run on their automatic (one-pass)
    path; the two-pass result is what the chunk-boundary claim is about.
    """
    _both_paths(*_buckets([length]))


@pytest.mark.gpu
def test_one_bucket_longer_than_the_real_busiest_second_by_far(gpu_ready: None) -> None:
    """A 28,000-record bucket, the measured tail of the burst generator, matches the oracle."""
    _gpu(*_buckets([28_000]))


# --- mixed-size and empty buckets ----------------------------------------


@pytest.mark.gpu
def test_long_buckets_interleaved_with_empty_and_short_ones(gpu_ready: None) -> None:
    """Chunk offsets that repeat across empty and short buckets still map to the right bucket."""
    lengths = [0, 5, 9_000, 0, 0, LIMIT + 1, 1, 0, 12_345, 3, 0, 0]
    got = _gpu(*_buckets(lengths))
    assert got["valid"].tolist() == [n > 0 for n in lengths]


@pytest.mark.gpu
def test_every_bucket_is_long(gpu_ready: None) -> None:
    """Consecutive multi-chunk buckets with no short bucket between them match the oracle."""
    _gpu(*_buckets([LIMIT + 1, 2 * LIMIT, LIMIT + 1023, 3 * LIMIT + 1]))


@pytest.mark.gpu
def test_all_records_in_the_last_bucket_which_is_long(gpu_ready: None) -> None:
    """A long bucket preceded only by empty buckets starts its chunks at offset zero."""
    _gpu(*_buckets([0, 0, 0, 0, 0, 10_000]))


@pytest.mark.gpu
def test_all_records_in_the_first_bucket_which_is_long(gpu_ready: None) -> None:
    """A long first bucket followed only by empty buckets leaves the trailing buckets zeroed."""
    _gpu(*_buckets([10_000, 0, 0, 0]))


# --- clipped windows -------------------------------------------------------


@pytest.mark.gpu
def test_records_before_and_after_the_window_with_long_buckets_inside(gpu_ready: None) -> None:
    """Chunks start at ``offsets[0]``, not record 0, and stop at ``offsets[-1]``."""
    _gpu(*_buckets([7, LIMIT + 5, 0, 2 * LIMIT + 1], before=1_500, after=6_000))


@pytest.mark.gpu
def test_long_bucket_ending_exactly_at_the_last_record(gpu_ready: None) -> None:
    """A long final bucket whose last chunk ends at the array end keeps every record."""
    _gpu(*_buckets([2, 0, LIMIT + 1], before=3, after=0))


@pytest.mark.gpu
def test_records_before_and_after_the_window_with_one_pass_buckets_inside(gpu_ready: None) -> None:
    """The clipped-window case on the one-pass path, so a dispatch change cannot regress it."""
    _gpu(*_buckets([7, 1_500, 0, 4_000], before=1_500, after=6_000), path="one_pass")


# --- pass 2 over several tiles of partials ---------------------------------


def _n_chunks(length: int) -> int:
    """Return the number of ``CHUNK``-sized chunks a bucket of ``length`` records needs."""
    return -(-length // CHUNK)


@pytest.mark.gpu
def test_pass_2_accumulates_over_several_tiles_with_a_ragged_last_tile(gpu_ready: None) -> None:
    """A bucket with more chunks than one pass-2 tile exercises the in-program tile loop.

    (3 * TILE + 5) full chunks plus a 17-record ragged chunk give
    3 * TILE + 6 partials: three full pass-2 tiles and a six-wide remainder.
    """
    length = (3 * TILE + 5) * CHUNK + 17
    assert _n_chunks(length) == 3 * TILE + 6 > TILE
    trades, offsets = _buckets([length])
    import torch

    schedule = kernels.build_chunk_metadata(torch.from_numpy(offsets), CHUNK)
    assert schedule.n_chunks == 3 * TILE + 6
    got = _gpu(trades, offsets)
    assert got["event_count"].item() == length


@pytest.mark.gpu
def test_pass_2_tile_boundaries_do_not_leak_between_adjacent_long_buckets(gpu_ready: None) -> None:
    """Two multi-tile buckets separated by empty and short buckets do not share partials.

    The second long bucket has exactly TILE + 1 chunks (one full tile plus a
    one-partial remainder) and starts at a chunk index that is not a tile
    multiple, so a pass-2 program that read past its own slice would pick
    up a neighbour's partial.
    """
    first = (TILE + 3) * CHUNK + 5
    second = (TILE + 1) * CHUNK
    lengths = [first, 0, 3, second, 0]
    assert _n_chunks(first) == TILE + 4 and _n_chunks(second) == TILE + 1
    got = _gpu(*_buckets(lengths))
    assert got["event_count"].tolist() == lengths


# --- signed and zero quantities on long buckets ----------------------------


@pytest.mark.gpu
def test_long_bucket_entirely_sell_initiated_has_signed_equal_to_minus_total(
    gpu_ready: None,
) -> None:
    """With every record sell-initiated, the signed quantity is exactly minus the total."""
    trades, offsets = _buckets([2 * LIMIT + 3])
    trades["buyer_is_maker"] = True
    got = _gpu(trades, offsets)
    assert got["signed_quantity"].item() == -got["total_quantity"].item() < 0


@pytest.mark.gpu
def test_long_bucket_with_zero_quantity_everywhere_is_valid_with_vwap_zero(
    gpu_ready: None,
) -> None:
    """Records that carry no quantity still set the count and validity; the VWAP is 0.0."""
    n = 2 * LIMIT + 5
    trades, offsets = _buckets([n])
    trades["quantity"] = 0
    got = _gpu(trades, offsets)
    assert got["event_count"].item() == n
    assert got["total_quantity"].item() == 0 == got["signed_quantity"].item()
    assert got["vwap"].item() == 0.0
    assert bool(got["valid"].item())


@pytest.mark.gpu
def test_long_bucket_whose_leading_chunks_carry_no_quantity(gpu_ready: None) -> None:
    """Whole chunks with zero partials do not disturb the single division in pass 2."""
    head, tail = 2 * CHUNK, 3 * CHUNK + 11
    n = head + tail
    trades, offsets = _buckets([n])
    trades["quantity"][:head] = 0
    trades["price"][head:] = 250 * P
    trades["quantity"][head:] = 2 * Q
    got = _gpu(trades, offsets)
    assert got["vwap"].item() == pytest.approx(250.0, rel=contract.VWAP_RTOL)
    assert got["total_quantity"].item() == tail * 2 * Q


# --- exact partial sums and the overflow bound ---------------------------


@pytest.mark.gpu
def test_long_bucket_just_inside_the_int64_bound_is_exact(gpu_ready: None) -> None:
    """Every partial is a sub-sum of a bounded bucket sum, so recombination loses nothing."""
    n = 2 * LIMIT
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30)
    trades["buyer_is_maker"] = np.arange(n) % 2 == 0
    offsets = np.array([0, n], dtype=np.int64)
    assert n * 2**30 * int(trades["quantity"][0]) <= contract.INT64_MAX
    got = _gpu(trades, offsets)
    assert got["total_quantity"].item() == int(trades["quantity"].sum())
    assert got["signed_quantity"].item() == 0


@pytest.mark.gpu
def test_long_bucket_past_the_int64_bound_raises_overflow_not_size_error(gpu_ready: None) -> None:
    """Validation runs before any tile or chunk decision, so the overflow check reports first."""
    n = 2 * LIMIT
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30) + 1
    offsets = np.array([0, n], dtype=np.int64)
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        ops.aggregate(*cuda_inputs(trades, offsets))


# --- the partial-VWAP averaging trap ---------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("half", CHUNK_CANDIDATES, ids=str)
def test_vwap_of_two_unequal_halves_is_175_not_150(gpu_ready: None, half: int) -> None:
    """Two unequal halves, one chunk each when ``half == CHUNK``, give 175 on the two-pass path."""
    trades, offsets = _trap_bucket(half)
    got = _both_paths(trades, offsets)
    vwap = got["vwap"].item()
    assert vwap == pytest.approx(175.0, rel=contract.VWAP_RTOL, abs=contract.VWAP_ATOL)
    assert abs(vwap - 150.0) > 1.0, "VWAP looks like a mean of per-chunk VWAPs"


@pytest.mark.gpu
@pytest.mark.parametrize("chunk", CHUNK_CANDIDATES, ids=str)
def test_ragged_last_chunk_cannot_be_averaged_either(gpu_ready: None, chunk: int) -> None:
    """A ragged last chunk with a different price is weighted by quantity, not averaged.

    Three full chunks of 1 unit at 100 are followed by 17 records of 3 units
    at 200. True VWAP = (100*3C + 200*51) / (3C + 51); a mean over four chunk
    VWAPs would be 125, far off for every C.
    """
    n_head = 3 * chunk
    n = n_head + 17
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = np.where(np.arange(n) < n_head, 100 * P, 200 * P)
    trades["quantity"] = np.where(np.arange(n) < n_head, 1 * Q, 3 * Q)
    offsets = np.array([0, n], dtype=np.int64)
    got = _both_paths(trades, offsets)
    expected = (100 * n_head + 200 * 51) / (n_head + 51)
    assert got["vwap"].item() == pytest.approx(expected, rel=contract.VWAP_RTOL)
