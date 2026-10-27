"""Tests for ``tapeforge.generate``: determinism, dtype, profile shape, validation and CLI.

Reproducibility is checked both in-process and across a fresh interpreter;
profile tests assert the statistical shape each profile promises (empty buckets,
heavy tail, duplicate timestamps) rather than exact values.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate


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


def test_determinism_survives_a_fresh_interpreter() -> None:
    """Generation is reproducible across processes, not only across calls.

    An in-process check would pass even if the generator read a module-level
    RNG, so this spawns a new interpreter twice and compares the raw bytes.
    """
    import subprocess
    import sys

    script = (
        "import hashlib;"
        "from tapeforge.generate import generate_trades;"
        "print(hashlib.sha256("
        "generate_trades(seed=11, n_buckets=200, profile='burst').tobytes()"
        ").hexdigest())"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(digests) == 1


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        generate.generate_trades(seed=1, n_buckets=10, profile="moon")  # type: ignore[arg-type]


def test_non_positive_bucket_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets must be positive"):
        generate.generate_trades(seed=1, n_buckets=0)


def test_quantities_are_strictly_positive() -> None:
    t = generate.generate_trades(seed=9, n_buckets=200, profile="burst")
    assert np.all(t["quantity"] > 0)


def test_start_offset_shifts_every_timestamp() -> None:
    base = generate.generate_trades(seed=2, n_buckets=20)
    shifted = generate.generate_trades(seed=2, n_buckets=20, start_us=5 * contract.US_PER_SECOND)
    np.testing.assert_array_equal(
        shifted["timestamp"] - 5 * contract.US_PER_SECOND, base["timestamp"]
    )


def test_width_must_be_a_multiple_of_the_event_tick() -> None:
    with pytest.raises(ValueError, match="event tick"):
        generate.generate_trades(seed=1, n_buckets=5, width_us=1_500)


def test_non_positive_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="width_us must be positive"):
        generate.generate_trades(seed=1, n_buckets=5, width_us=0)


def test_all_empty_output_still_has_the_right_dtype() -> None:
    """A sparse run short enough to draw nothing still returns ``TRADE_DTYPE``, not a bare array."""
    empty = generate.generate_trades(seed=1, n_buckets=1, profile="sparse", width_us=1_000)
    assert empty.dtype == contract.TRADE_DTYPE


def test_cli_writes_a_loadable_reproducible_sample(tmp_path, capsys) -> None:
    """The documented CLI command writes a sample equal to the in-process generator output."""
    out = tmp_path / "sample.npy"
    assert generate.main(["--seed", "7", "--seconds", "10", "--profile", "burst",
                          "--out", str(out)]) == 0
    reloaded = np.load(out)
    assert reloaded.dtype == contract.TRADE_DTYPE
    np.testing.assert_array_equal(
        reloaded, generate.generate_trades(seed=7, n_buckets=10, profile="burst")
    )
    printed = capsys.readouterr().out
    assert "records" in printed and "non-empty" in printed
