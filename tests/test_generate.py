"""Week 1: the seeded generator (TF-01)."""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="week 1: delete this marker to begin the RED phase",
)


def test_same_seed_reproduces_identical_records() -> None:
    a = generate.generate_trades(seed=7, n_buckets=50)
    b = generate.generate_trades(seed=7, n_buckets=50)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = generate.generate_trades(seed=7, n_buckets=50)
    b = generate.generate_trades(seed=8, n_buckets=50)
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_output_matches_the_trade_dtype() -> None:
    t = generate.generate_trades(seed=1, n_buckets=10)
    assert t.dtype == contract.TRADE_DTYPE


def test_timestamps_are_sorted_non_decreasing() -> None:
    t = generate.generate_trades(seed=1, n_buckets=100, profile="burst")
    assert np.all(np.diff(t["timestamp"]) >= 0)


def test_sparse_profile_produces_empty_buckets() -> None:
    t = generate.generate_trades(seed=3, n_buckets=200, profile="sparse")
    occupied = np.unique(t["timestamp"] // contract.DEFAULT_BUCKET_WIDTH_US)
    assert occupied.size < 200


def test_burst_profile_produces_a_heavy_tail() -> None:
    t = generate.generate_trades(seed=4, n_buckets=500, profile="burst")
    _, counts = np.unique(t["timestamp"] // contract.DEFAULT_BUCKET_WIDTH_US, return_counts=True)
    assert counts.max() > 10 * np.median(counts)


def test_duplicate_timestamps_occur() -> None:
    t = generate.generate_trades(seed=5, n_buckets=100, profile="uniform")
    assert np.any(np.diff(t["timestamp"]) == 0)


def test_agg_trade_ids_are_unique_and_increasing() -> None:
    t = generate.generate_trades(seed=6, n_buckets=100)
    assert np.all(np.diff(t["agg_trade_id"]) > 0)
