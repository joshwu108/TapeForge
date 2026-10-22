"""The benchmark baseline and harness are measured against, so they get tests too.

A wrong baseline makes every speedup figure wrong; the PyTorch baseline is
checked here against the plain-loop oracle on CPU, including the window
edge cases where a scatter-based implementation is most likely to slip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tapeforge import contract, reference, windows

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import baselines  # noqa: E402
import harness  # noqa: E402
import workload  # noqa: E402

W = contract.DEFAULT_BUCKET_WIDTH_US


def _run_baseline(trades: np.ndarray, offsets: np.ndarray) -> dict[str, object]:
    return baselines.torch_index_add(workload.host_columns(trades), torch.from_numpy(offsets))


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize("seed", [1, 2])
def test_torch_baseline_matches_the_naive_loop(profile: str, seed: int) -> None:
    trades, offsets, info = workload.make_workload(5_000, profile, seed)
    assert info.n_records == trades.size
    workload.check_features(
        _run_baseline(trades, offsets), reference.aggregate_reference_naive(trades, offsets)
    )


def test_torch_baseline_ignores_records_outside_the_window() -> None:
    trades = np.zeros(6, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = [0, W, W, 2 * W + 5, 4 * W, 9 * W]
    trades["price"] = [1, 10_000, 10_200, 500, 700, 9]
    trades["quantity"] = [7, 200_000, 100_000, 0, 3, 8]
    trades["buyer_is_maker"] = [False, False, True, True, True, False]
    offsets = windows.build_offsets(trades["timestamp"], W, 5, W)
    assert offsets[0] > 0 and offsets[-1] < trades.size
    workload.check_features(
        _run_baseline(trades, offsets), reference.aggregate_reference_naive(trades, offsets)
    )


def test_torch_baseline_with_no_records() -> None:
    empty = np.zeros(0, dtype=contract.TRADE_DTYPE)
    offsets = np.zeros(4, dtype=np.int64)
    workload.check_features(
        _run_baseline(empty, offsets), reference.aggregate_reference_naive(empty, offsets)
    )


def test_correctness_gate_rejects_a_wrong_dtype_and_a_wrong_value() -> None:
    trades, offsets, _ = workload.make_workload(500, "uniform", 3)
    expected = reference.aggregate_reference(trades, offsets)
    good = workload.features_dict(expected)
    with pytest.raises(AssertionError, match="dtype"):
        workload.check_features({**good, "event_count": good["event_count"].astype(np.int32)},
                                expected)
    bad_total = good["total_quantity"].copy()
    bad_total[0] += 1
    with pytest.raises(AssertionError):
        workload.check_features({**good, "total_quantity": bad_total}, expected)


def test_environment_captures_provenance() -> None:
    env = harness.capture_environment()
    assert env.python and env.platform and env.numpy and env.torch
    assert env.git_commit is None or env.git_dirty is not None


def test_record_appends_without_touching_the_committed_results(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    harness.record("kernel", "x", {"p50_ms": 1.0}, {"n": 1}, path=path)
    harness.record("kernel", "y", {"p50_ms": 2.0}, {"n": 2}, path=path)
    import json

    entries = json.loads(path.read_text())
    assert [e["name"] for e in entries] == ["x", "y"]
    assert entries[0]["environment"]["python"]
