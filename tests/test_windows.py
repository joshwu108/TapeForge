"""Week 2: the no-padding window index."""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, windows

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="week 2: delete this marker to begin the RED phase",
)

SEC = contract.US_PER_SECOND


def test_prd_offsets_example() -> None:
    """PRD section 4: offsets [0, 2, 2, 7] -- bucket 1 is empty."""
    ts = np.array([0, 0, 2, 2, 2, 2, 2], dtype=np.int64) * SEC
    offsets = windows.build_offsets(ts, 0, 3, SEC)
    np.testing.assert_array_equal(offsets, np.array([0, 2, 2, 7]))


def test_offsets_length_is_n_buckets_plus_one() -> None:
    ts = np.arange(10, dtype=np.int64) * SEC
    assert windows.build_offsets(ts, 0, 10, SEC).shape == (11,)


def test_offsets_are_monotonic_and_cover_every_record() -> None:
    ts = np.sort(np.random.default_rng(0).integers(0, 10 * SEC, size=1000)).astype(np.int64)
    offsets = windows.build_offsets(ts, 0, 10, SEC)
    assert np.all(np.diff(offsets) >= 0)
    assert offsets[0] == 0
    assert offsets[-1] == ts.size


def test_records_outside_the_window_are_excluded() -> None:
    ts = np.array([-5, 1, 2, 99], dtype=np.int64) * SEC
    offsets = windows.build_offsets(ts, 0, 3, SEC)
    assert offsets[-1] - offsets[0] == 2  # only t=1s and t=2s are inside


def test_bucket_edges_are_exact_multiples_of_width() -> None:
    edges = windows.bucket_edges(start_us=0, n_buckets=4, width_us=SEC)
    np.testing.assert_array_equal(edges, np.array([0, SEC, 2 * SEC, 3 * SEC, 4 * SEC]))
