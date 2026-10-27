"""Tests for the host-side launcher logic of ``tapeforge.kernels``, runnable without a GPU.

Covers the one-pass tile choice, explicit-capacity rejection, the dispatch
plan and the two-pass chunk schedule. These are plain Python or torch-on-CPU,
so they are pinned on every machine. The kernels themselves are exercised in
test_gpu_parity.py and test_two_pass.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import kernels

LIMIT = kernels.MAX_ONE_PASS_BUCKET
CHUNK = kernels.CHUNK_SIZE


# --- one-pass tile ---------------------------------------------------------


@pytest.mark.parametrize(
    ("max_len", "expected"),
    [(0, 16), (1, 16), (16, 16), (17, 32), (1024, 1024), (1025, 2048), (LIMIT, LIMIT)],
)
def test_auto_block_size_is_the_smallest_power_of_two_that_fits(
    max_len: int, expected: int
) -> None:
    """The tile is the smallest power of two that holds ``max_len``, with a floor of 16."""
    assert kernels.select_block_size(max_len) == expected


def test_bucket_one_past_the_one_pass_limit_does_not_fit_a_tile() -> None:
    with pytest.raises(kernels.BucketTooLargeError, match=f"{LIMIT + 1} records"):
        kernels.select_block_size(LIMIT + 1)


def test_bucket_too_large_error_is_a_value_error() -> None:
    assert issubclass(kernels.BucketTooLargeError, ValueError)


def test_explicit_block_size_is_honoured_and_enforced() -> None:
    """An explicit ``block_size`` is used as given and is a hard capacity limit."""
    assert kernels.select_block_size(1024, block_size=1024) == 1024
    assert kernels.select_block_size(3, block_size=1024) == 1024
    with pytest.raises(kernels.BucketTooLargeError, match="at most 1024"):
        kernels.select_block_size(1025, block_size=1024)


@pytest.mark.parametrize("block_size", [0, -4, 1000, 3])
def test_explicit_block_size_must_be_a_power_of_two(block_size: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        kernels.select_block_size(1, block_size=block_size)


def test_explicit_block_size_cannot_exceed_the_one_pass_limit() -> None:
    with pytest.raises(ValueError, match="exceeds the one-pass limit"):
        kernels.select_block_size(1, block_size=2 * LIMIT)


# --- dispatch plan ---------------------------------------------------------


@pytest.mark.parametrize("max_len", [0, 1, 1024, LIMIT])
def test_auto_dispatch_is_one_pass_up_to_the_limit(max_len: int) -> None:
    """Buckets up to ``MAX_ONE_PASS_BUCKET`` dispatch one-pass with the automatic tile."""
    plan = kernels.plan_dispatch(max_len)
    assert plan.path == "one_pass"
    assert plan.block_size == kernels.select_block_size(max_len)
    assert plan.chunk_size is None


@pytest.mark.parametrize("max_len", [LIMIT + 1, 2 * LIMIT, 28_000, 400_000])
def test_auto_dispatch_is_two_pass_beyond_the_limit(max_len: int) -> None:
    """Buckets beyond ``MAX_ONE_PASS_BUCKET`` dispatch two-pass with ``CHUNK_SIZE`` chunks."""
    plan = kernels.plan_dispatch(max_len)
    assert plan == kernels.DispatchPlan(path="two_pass", block_size=None, chunk_size=CHUNK)


def test_explicit_block_size_forces_one_pass_and_is_a_capacity_check() -> None:
    """Forcing one-pass never falls back to two-pass; an oversized bucket raises instead."""
    assert kernels.plan_dispatch(900, block_size=1024).path == "one_pass"
    with pytest.raises(kernels.BucketTooLargeError, match="at most 1024"):
        kernels.plan_dispatch(1025, block_size=1024)
    with pytest.raises(kernels.BucketTooLargeError, match="at most 4096"):
        kernels.plan_dispatch(LIMIT + 1, path="one_pass")


def test_forced_two_pass_applies_to_short_buckets_too() -> None:
    plan = kernels.plan_dispatch(5, path="two_pass")
    assert plan.path == "two_pass" and plan.chunk_size == CHUNK


def test_forced_two_pass_rejects_an_explicit_block_size() -> None:
    with pytest.raises(ValueError, match="one-pass path only"):
        kernels.plan_dispatch(5, block_size=1024, path="two_pass")


def test_unknown_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="path must be"):
        kernels.plan_dispatch(5, path="three_pass")  # type: ignore[arg-type]


def test_chunk_size_is_a_power_of_two_within_the_one_pass_tile_limit() -> None:
    """Pass 1 reduces a chunk with one tile, so ``CHUNK_SIZE`` must itself be a legal tile."""
    assert kernels.select_block_size(CHUNK, block_size=CHUNK) == CHUNK
    assert kernels.COMBINE_BLOCK_SIZE > 0
    assert kernels.COMBINE_BLOCK_SIZE & (kernels.COMBINE_BLOCK_SIZE - 1) == 0


# --- two-pass chunk schedule (torch on the CPU) ----------------------------


def _schedule(lengths: list[int], before: int = 0, chunk: int = CHUNK) -> kernels.ChunkMetadata:
    """Build the chunk schedule for buckets of the given ``lengths``.

    Args:
        lengths: Record count of each bucket, in order.
        before: Number of records preceding the window, so ``offsets[0] == before``.
        chunk: Chunk size passed to ``build_chunk_metadata``.
    """
    torch = pytest.importorskip("torch")
    offsets = np.concatenate([[before], before + np.cumsum(lengths)]).astype(np.int64)
    return kernels.build_chunk_metadata(torch.from_numpy(offsets), chunk)


def _ranges(schedule: kernels.ChunkMetadata) -> list[tuple[int, int]]:
    """Return each chunk's ``(start, end)`` record range in schedule order."""
    return list(zip(schedule.chunk_starts.tolist(), schedule.chunk_ends.tolist(), strict=True))


@pytest.mark.parametrize(
    "lengths",
    [
        [0, 5, 9_000, 0, 0, LIMIT + 1, 1, 0, 12_345, 3, 0, 0],
        [CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK, 3 * CHUNK + 17],
        [0, 0, 0],
        [28_000],
    ],
    ids=["mixed", "boundaries", "all-empty", "single-long"],
)
def test_chunk_schedule_covers_every_bucket_exactly_once(lengths: list[int]) -> None:
    """Chunks tile each bucket contiguously, never cross a bucket and never exceed ``CHUNK``."""
    before = 1_500
    schedule = _schedule(lengths, before=before)
    # Each bucket needs ceil(n / CHUNK) chunks; an empty bucket needs none.
    expected_counts = [-(-n // CHUNK) for n in lengths]
    assert schedule.n_chunks == sum(expected_counts)
    assert schedule.chunk_offsets.tolist() == [0, *np.cumsum(expected_counts).tolist()]
    assert schedule.chunk_size == CHUNK

    ranges = _ranges(schedule)
    start = before
    for b, length in enumerate(lengths):
        first, last = schedule.chunk_offsets[b].item(), schedule.chunk_offsets[b + 1].item()
        covered = ranges[first:last]
        cursor = start
        for lo, hi in covered:
            assert lo == cursor, "chunks must be contiguous within a bucket"
            assert 0 < hi - lo <= CHUNK, "every chunk holds 1..CHUNK records"
            cursor = hi
        assert cursor == start + length, "chunks must cover the bucket exactly"
        start += length


def test_chunk_schedule_last_chunk_is_ragged_and_earlier_ones_full() -> None:
    """Only the final chunk of a bucket may be shorter than ``CHUNK``."""
    schedule = _schedule([3 * CHUNK + 17])
    assert [hi - lo for lo, hi in _ranges(schedule)] == [CHUNK, CHUNK, CHUNK, 17]


def test_chunk_schedule_for_an_exact_multiple_has_no_empty_tail_chunk() -> None:
    """A bucket of exactly ``k * CHUNK`` records yields ``k`` chunks, never an empty tail."""
    schedule = _schedule([2 * CHUNK])
    assert [hi - lo for lo, hi in _ranges(schedule)] == [CHUNK, CHUNK]


def test_chunk_schedule_respects_a_clipped_window() -> None:
    """Records before offsets[0] are never chunked; chunk indices start at offsets[0]."""
    schedule = _schedule([CHUNK + 1], before=7)
    assert _ranges(schedule) == [(7, 7 + CHUNK), (7 + CHUNK, 7 + CHUNK + 1)]


def test_chunk_schedule_rejects_a_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _schedule([10], chunk=0)


def test_chunk_schedule_dtypes_are_int64() -> None:
    """Chunk tensors are int64 so the kernels can index records without overflow."""
    torch = pytest.importorskip("torch")
    schedule = _schedule([CHUNK + 1, 0, 3])
    for tensor in (schedule.chunk_starts, schedule.chunk_ends, schedule.chunk_offsets):
        assert tensor.dtype == torch.int64


# --- without Triton ---------------------------------------------------------


@pytest.mark.skipif(kernels.TRITON_AVAILABLE, reason="triton is installed on this machine")
def test_launcher_without_triton_raises_a_clear_error() -> None:
    """Without Triton the launcher raises ``RuntimeError`` naming the missing dependency."""
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        kernels.launch_aggregate({}, torch.zeros(2, dtype=torch.int64))
