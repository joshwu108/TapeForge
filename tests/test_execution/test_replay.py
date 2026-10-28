"""Tests for tapeforge.execution.replay: an episode identifier rebuilds the recorded episode
exactly, the step rows expose what the policy saw and what happened, and bad identifiers fail
with a reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tapeforge.execution import replay as replay_module
from tapeforge.execution.experiment import run_experiment
from tapeforge.execution.replay import (
    STEP_COLUMNS,
    format_replay,
    replay_episode,
    step_rows,
    verify_against_record,
)
from tapeforge.execution.runner import summarize_episode
from test_execution.configs import synthetic_config


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the synthetic experiment once for the module."""
    return run_experiment(synthetic_config(), output_dir=tmp_path_factory.mktemp("runs"))


@pytest.mark.parametrize("policy", ["twap", "immediate", "cap_aware_twap"])
def test_replay_reproduces_the_recorded_accounting(run_dir: Path, policy: str) -> None:
    replay = replay_episode(run_dir, "validation/s2/i1", policy)
    assert replay.episode_id == "validation/s2/i1" and replay.policy == policy
    assert verify_against_record(replay, run_dir) == []
    record = json.loads((run_dir / "records" / f"{policy}-validation-seed2.json").read_text())
    assert record["episodes"][1]["summary"] == summarize_episode(replay.trajectory).to_json()
    assert replay.draw.to_json() == record["episodes"][1]["draw"]


def test_step_rows_show_observation_availability_actions_fills_and_rewards(run_dir: Path) -> None:
    replay = replay_episode(run_dir, "train/s1/i0", "twap")
    rows = step_rows(replay.trajectory, replay.market)
    assert len(rows) == len(replay.trajectory.results)
    assert all(tuple(row) == STEP_COLUMNS for row in rows)
    spec = replay.spec
    for t, (row, result, state_after) in enumerate(
        zip(rows, replay.trajectory.results, replay.trajectory.states[1:], strict=True)
    ):
        assert row["step"] == t and row["visible_bucket"] == t - 1 - spec.data_delay_buckets
        assert row["fill_bucket"] == t + spec.latency_buckets
        assert row["submitted_qty_ticks"] == result.submitted_qty_ticks
        assert row["fill_qty_ticks"] == result.fill_qty_ticks <= row["cap_qty_ticks"] or (
            row["cap_qty_ticks"] == 0 and row["fill_qty_ticks"] == 0
        )
        assert row["fee_ticks"] == result.fee_ticks
        assert row["fill_price_ticks"] == result.fill_price_ticks
        assert row["reward_ticks"] == -(row["cost"] + row["risk"] + row["terminal"])
        assert row["remaining_after"] == spec.target_qty_ticks - state_after.filled_qty_ticks
        assert row["cash_after"] == state_after.cash_ticks
        assert row["inventory_after"] == state_after.inventory_ticks
        assert row["visible_valid"] == bool(replay.market.valid[replay.market.index(t - 1)])
    last = rows[-1]
    assert last["terminated"] != last["truncated"]
    text = format_replay(replay)
    assert "ending: terminated" in text or "ending: truncated" in text
    assert "shortfall" in text and "reward sum" in text and "arrival price" in text


def test_the_terminal_handling_is_spelled_out_for_residual_inventory(run_dir: Path) -> None:
    replay = replay_episode(run_dir, "train/s1/i0", "twap")
    summary = summarize_episode(replay.trajectory)
    text = format_replay(replay)
    if summary.residual_qty_ticks and summary.terminated:
        assert "terminal mark" in text and "residual term" in text
    elif summary.completed:
        assert "completed 0" in text


def test_unknown_identifiers_and_policies_fail_with_a_reason(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="not of the form"):
        replay_episode(run_dir, "validation-2-1", "twap")
    with pytest.raises(ValueError, match="is not in the run"):
        replay_episode(run_dir, "validation/s9/i0", "twap")
    with pytest.raises(ValueError, match="is not in the run"):
        replay_episode(run_dir, "validation/s2/i400", "twap")
    with pytest.raises(ValueError, match="unknown policy"):
        replay_episode(run_dir, "validation/s2/i0", "vwap")


def test_cli_prints_the_table_json_and_verification(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert replay_module.main([str(run_dir), "--episode", "train/s2/i3", "--policy", "immediate",
                               "--verify"]) == 0
    out = capsys.readouterr().out
    assert "submitted_qty_ticks" in out and "matches the recorded episode summary" in out
    assert replay_module.main([str(run_dir), "--episode", "train/s2/i3", "--policy", "twap",
                               "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episode_id"] == "train/s2/i3" and payload["steps"][0]["step"] == 0
    assert replay_module.main([str(run_dir), "--episode", "train/s2/i99", "--policy", "twap"]) == 1
    assert "not in the run" in capsys.readouterr().err
