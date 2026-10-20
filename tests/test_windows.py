"""Week 2: the no-padding window index."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tapeforge import contract, windows

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


def test_offsets_are_indices_into_the_input_not_a_compacted_copy() -> None:
    """offsets[0] is nonzero when records precede the window. Documented, tested."""
    ts = np.array([-5, -3, 1, 2], dtype=np.int64) * SEC
    offsets = windows.build_offsets(ts, 0, 3, SEC)
    assert offsets[0] == 2
    assert offsets[-1] == 4


def test_a_window_entirely_after_the_data_is_all_empty() -> None:
    ts = np.array([0, 1], dtype=np.int64) * SEC
    offsets = windows.build_offsets(ts, 100 * SEC, 5, SEC)
    np.testing.assert_array_equal(windows.bucket_lengths(offsets), np.zeros(5, dtype=np.int64))


def test_empty_input_gives_all_zero_offsets() -> None:
    offsets = windows.build_offsets(np.array([], dtype=np.int64), 0, 4, SEC)
    np.testing.assert_array_equal(offsets, np.zeros(5, dtype=np.int64))


# --- bucket_lengths ------------------------------------------------------

def test_bucket_lengths_matches_the_prd_example() -> None:
    np.testing.assert_array_equal(
        windows.bucket_lengths(np.array([0, 2, 2, 7], dtype=np.int64)), [2, 0, 5]
    )


def test_bucket_lengths_rejects_a_degenerate_offsets_array() -> None:
    with pytest.raises(ValueError, match="at least 2 entries"):
        windows.bucket_lengths(np.array([0], dtype=np.int64))


# --- validation ----------------------------------------------------------

def test_non_monotonic_input_names_the_offending_record() -> None:
    ts = np.array([0, 5, 3], dtype=np.int64) * SEC
    with pytest.raises(ValueError, match=r"record 2 \(3000000\) precedes record 1"):
        windows.build_offsets(ts, 0, 10, SEC)


def test_float_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="int64"):
        windows.build_offsets(np.array([0.0, 1.0]), 0, 2, SEC)


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        windows.build_offsets(np.zeros((2, 2), dtype=np.int64), 0, 2, SEC)


def test_degenerate_geometry_is_rejected() -> None:
    ts = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="n_buckets must be positive"):
        windows.build_offsets(ts, 0, 0, SEC)
    with pytest.raises(ValueError, match="width_us must be positive"):
        windows.build_offsets(ts, 0, 2, 0)


# --- cross-check against an independent brute-force implementation -------

def _naive_offsets(ts: np.ndarray, start: int, n_buckets: int, width: int) -> list[int]:
    """Deliberately dumb oracle: a plain loop, no searchsorted, no numpy tricks."""
    out = []
    for b in range(n_buckets + 1):
        edge = start + b * width
        count = 0
        for t in ts:
            if t < edge:
                count += 1
        out.append(count)
    return out


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
def test_matches_a_brute_force_oracle_on_generated_data(profile: str) -> None:
    from tapeforge import generate

    trades = generate.generate_trades(seed=21, n_buckets=40, profile=profile)
    offsets = windows.build_offsets(trades["timestamp"], 0, 40, SEC)
    assert offsets.tolist() == _naive_offsets(trades["timestamp"], 0, 40, SEC)


def test_matches_a_brute_force_oracle_on_the_real_fixture() -> None:
    """Real archive rows, not just synthetic ones."""
    from pathlib import Path

    from tapeforge import ingest

    fixture = Path(__file__).parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"
    trades, _ = ingest.load_agg_trades(fixture)
    start = int(trades["timestamp"][0]) // SEC * SEC
    offsets = windows.build_offsets(trades["timestamp"], start, 4, SEC)
    assert offsets.tolist() == _naive_offsets(trades["timestamp"], start, 4, SEC)
    # The fixture was built with second 1 empty on purpose.
    np.testing.assert_array_equal(windows.bucket_lengths(offsets), [3, 0, 2, 0])


# --- properties ----------------------------------------------------------

@given(
    st.lists(st.integers(min_value=-10, max_value=40), min_size=0, max_size=200),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=5),
)
@settings(max_examples=300, deadline=None)
def test_properties_hold_for_arbitrary_input(
    raw: list[int], n_buckets: int, width_s: int
) -> None:
    ts = np.sort(np.array(raw, dtype=np.int64)) * SEC
    width = width_s * SEC
    offsets = windows.build_offsets(ts, 0, n_buckets, width)
    lengths = windows.bucket_lengths(offsets)

    assert offsets.shape == (n_buckets + 1,)
    assert np.all(lengths >= 0)                       # monotonic offsets
    assert offsets[-1] <= ts.size                     # never past the end
    in_window = int(np.sum((ts >= 0) & (ts < n_buckets * width)))
    assert int(lengths.sum()) == in_window            # covers every in-window record

    # Every record in a bucket's slice really belongs to that bucket.
    for b in range(n_buckets):
        lo, hi = int(offsets[b]), int(offsets[b + 1])
        slice_ts = ts[lo:hi]
        assert np.all(slice_ts >= b * width)
        assert np.all(slice_ts < (b + 1) * width)
