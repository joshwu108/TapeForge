"""Tests of the training-pipeline benchmark: gates, skips, threshold accounting and records.

Timings themselves are not asserted; only that every section produces a
well-formed row, that unavailable hardware yields a skip instead of a number,
and that a run which never reaches the threshold is reported as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import training  # noqa: E402
from test_execution.ppo_configs import write_configs  # noqa: E402

BENCHMARK = """
name = "training-test"
scope = "train"
ppo_config = "{ppo}"
backends = {backends}
device = "{device}"
sections = {sections}
model_seed = 0
total_steps = 192
output_dir = "{root}/bench-runs"
feature_episodes = 5
env_step_vector_steps = 30
core_batch_sizes = [4, 32]
core_vector_steps = 10

[threshold]
metric = "validation_mean_shortfall_bps"
max = {threshold}
"""


def _config(root: Path, *, backends: str = '["serial", "batched"]', device: str = "cpu",
            sections: str = '["features", "env_step", "batched_core", "policy"]',
            threshold: float = -1000.0) -> Path:
    ppo = write_configs(root)
    path = root / "benchmark.toml"
    path.write_text(BENCHMARK.format(ppo=ppo, root=root, backends=backends, device=device,
                                     sections=sections, threshold=threshold))
    return path


def test_every_stage_section_produces_a_gated_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results.json"
    assert training.main(["--config", str(_config(tmp_path)), "--out", str(out)]) == 0
    entries = {e["name"]: e for e in json.loads(out.read_text())}
    assert set(entries) == {"features", "env_step/serial", "env_step/batched",
                            "batched_core/b4", "batched_core/b32", "policy"}
    for entry in entries.values():
        assert entry["scope"] == "train" and entry["metrics"]["status"] == "ok"
        assert entry["environment"]["torch"] == torch.__version__
        assert len(entry["config"]["unavoidable_differences"]) == 3
    features = entries["features"]["metrics"]
    assert features["n_episodes"] == 5 and features["slice_buckets"] == 28
    assert features["precomputed_ms_per_episode"] > 0 and features["on_demand_ms_per_episode"] > 0
    assert entries["env_step/serial"]["metrics"]["env_steps_per_second"] > 0
    assert entries["batched_core/b32"]["metrics"]["batch_size"] == 32
    assert "features:" in capsys.readouterr().out


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA skip path")
def test_an_unavailable_device_yields_skipped_rows_not_numbers(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    config = _config(tmp_path, sections='["env_step", "batched_core"]')
    assert training.main(["--config", str(config), "--out", str(out), "--device", "cuda"]) == 0
    entries = {e["name"]: e["metrics"] for e in json.loads(out.read_text())}
    assert entries["env_step/serial"]["status"] == "ok"
    for name in ("env_step/batched", "batched_core"):
        assert entries[name]["status"] == "skipped" and "unavailable" in entries[name]["reason"]
        assert "env_steps_per_second" not in entries[name]


def test_a_failed_parity_gate_blocks_the_batched_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tapeforge.execution import batched

    monkeypatch.setattr(
        batched, "_fee", lambda notional, fee_bps: batched._floor_div(notional * fee_bps, 10_000)
    )
    out = tmp_path / "results.json"
    config = _config(tmp_path, sections='["batched_core"]')
    assert training.main(["--config", str(config), "--out", str(out)]) == 1
    (entry,) = json.loads(out.read_text())
    assert entry["metrics"]["status"] == "incorrect" and entry["metrics"]["n_mismatches"] > 0


def test_end_to_end_reports_stage_times_and_an_unreached_threshold(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    config = _config(tmp_path, backends='["serial"]', sections='["end_to_end"]')
    assert training.main(["--config", str(config), "--out", str(out)]) == 0
    (entry,) = json.loads(out.read_text())
    metrics = entry["metrics"]
    assert entry["name"] == "end_to_end/serial" and metrics["env_steps"] == 192
    assert set(metrics["seconds"]) == {"data_load", "statistics", "setup", "collect", "update",
                                       "evaluate", "checkpoint", "train"}
    assert metrics["max_rss_bytes_main"] > 0 and metrics["process_wall_seconds"] > 0
    # No policy has a shortfall below -1000 bps here: the run must say it never got there.
    assert metrics["threshold"]["reached"] is False and "best_value" in metrics["threshold"]

    run_dir = Path(metrics["run_dir"])
    best = metrics["threshold"]["best_value"]
    reached = training.time_to_threshold(run_dir, "validation_mean_shortfall_bps", best)
    assert reached["reached"] is True and reached["value"] == best
    assert reached["train_plus_evaluation_seconds"] > reached["train_seconds"] >= 0.0
