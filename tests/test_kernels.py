"""Host-side kernel launcher logic that runs without a GPU.

The one-pass tile-size choice and the oversized-bucket rejection are plain
Python, so they are pinned here on every machine. The kernel itself is
exercised in test_gpu_parity.py.
"""

from __future__ import annotations

import pytest

from tapeforge import kernels

LIMIT = kernels.MAX_ONE_PASS_BUCKET


@pytest.mark.parametrize(
    ("max_len", "expected"),
    [(0, 16), (1, 16), (16, 16), (17, 32), (1024, 1024), (1025, 2048), (LIMIT, LIMIT)],
)
def test_auto_block_size_is_the_smallest_power_of_two_that_fits(
    max_len: int, expected: int
) -> None:
    assert kernels.select_block_size(max_len) == expected


def test_bucket_one_past_the_one_pass_limit_is_rejected() -> None:
    with pytest.raises(kernels.BucketTooLargeError, match=f"{LIMIT + 1} records"):
        kernels.select_block_size(LIMIT + 1)


def test_bucket_too_large_error_is_a_value_error() -> None:
    assert issubclass(kernels.BucketTooLargeError, ValueError)


def test_explicit_block_size_is_honoured_and_enforced() -> None:
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


@pytest.mark.skipif(kernels.TRITON_AVAILABLE, reason="triton is installed on this machine")
def test_launcher_without_triton_raises_a_clear_error() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        kernels.launch_aggregate({}, torch.zeros(2, dtype=torch.int64))
