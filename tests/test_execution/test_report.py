"""Tests for tapeforge.execution.report: outcome classes, hand-calculated group and paired
statistics, independent reconstruction from the records, the day-count gate on uncertainty, the
held-out rule and the exported files.

Hand-calculated cases use rows built directly; the end-to-end cases use the
synthetic experiment of ``test_execution.configs``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tapeforge.execution import report as report_module
from tapeforge.execution.experiment import run_experiment
from tapeforge.execution.report import (
    REPORT_SCHEMA,
    EpisodeRow,
    build_report,
    day_cluster_bootstrap,
    episode_rows,
    group_statistics,
    load_records,
    paired_differences,
    settlement_label,
    write_report,
)
from test_execution.configs import DATES, synthetic_config


def _row(policy: str, episode_id: str, date: str, outcome: str, shortfall: int,
         terminal: int = 0, target: int = 1_000, arrival: int = 10_000,
         risk: int = 0) -> EpisodeRow:
    """Build a row whose shortfall in bps is ``shortfall / (arrival * target) * 1e4``.

    A ``completed`` row filled everything; ``settled`` filled half and settled
    the other half (fee 40 of the 100 total); ``incomplete`` filled half under
    the legacy rule; ``truncated`` filled half and settled nothing.
    """
    role, seed, index = episode_id.split("/")
    filled = target if outcome == "completed" else target // 2
    settled = target - filled if outcome == "settled" else 0
    settlement_fee = 40 if settled else 0
    return EpisodeRow(
        policy=policy, role=role, held_out=False, episode_id=episode_id, seed=int(seed[1:]),
        index=int(index[1:]), date=date, side=1, rho_bps=500, target_qty_ticks=target,
        arrival_price_ticks=arrival, terminal_rule="terminal_settlement",
        terminal_surcharge_bps=25, outcome=outcome,
        completed_by_fills=outcome == "completed", completed=outcome in ("completed", "settled"),
        n_steps=3, filled_qty_ticks=filled, settled_qty_ticks=settled,
        residual_qty_ticks=target - filled - settled, filled_fraction=filled / target,
        settlement_fraction=settled / target, fill_fees_ticks=100 - settlement_fee,
        settlement_fees_ticks=settlement_fee, fees_ticks=100, shortfall_ticks=shortfall,
        shortfall_bps=shortfall * 10_000 / (arrival * target),
        fill_shortfall_ticks=shortfall - terminal, terminal_component_ticks=terminal,
        terminal_price_ticks=10_010 if terminal else 0,
        settlement_price_ticks=10_036 if settled else 0, settlement_exceeds_cap=False,
        n_capped_steps=1, risk_sum_ticks=risk, reward_sum_ticks=-shortfall - risk,
    )


# --- hand-calculated statistics -------------------------------------------------------


def test_group_statistics_keep_the_four_outcome_classes_apart() -> None:
    # Arrival notional 10,000 x 1,000 = 1e7 cash ticks, so 1,000 ticks = 1 bps.
    rows = [
        _row("twap", "train/s1/i0", DATES[0], "completed", 2_000, risk=500),
        _row("twap", "train/s1/i1", DATES[0], "completed", 4_000),
        _row("twap", "train/s1/i2", DATES[1], "settled", 9_000, terminal=3_000),
        _row("twap", "train/s1/i3", DATES[1], "incomplete", 6_000, terminal=1_000),
        _row("twap", "train/s1/i4", DATES[1], "truncated", 1_000),
    ]
    stats = group_statistics(rows)
    assert (stats["n_episodes"], stats["n_completed"], stats["n_settled"], stats["n_incomplete"],
            stats["n_truncated"]) == (5, 2, 1, 1, 1)
    # Completion by fills (2 of 5) and final completion (3 of 5: the settled one counts).
    assert stats["completion_by_fills_rate"] == 0.4 and stats["final_completion_rate"] == 0.6
    assert stats["truncation_rate"] == 0.2
    assert stats["n_days"] == 2 and stats["mean_steps"] == 3.0 and stats["mean_capped_steps"] == 1.0
    terminated = stats["terminated"]
    assert terminated["n"] == 4 and terminated["total_shortfall_ticks"] == 21_000
    assert terminated["mean_shortfall_bps"] == pytest.approx(5.25)   # (2 + 4 + 9 + 6) / 4.
    assert terminated["median_shortfall_bps"] == 4.0                 # Nearest rank of 4 values.
    assert terminated["p95_shortfall_bps"] == terminated["max_shortfall_bps"] == 9.0
    assert terminated["mean_terminal_component_bps"] == pytest.approx(1.0)   # 4,000 / 4 rows.
    assert terminated["total_terminal_component_ticks"] == 4_000
    assert terminated["mean_fill_shortfall_bps"] == pytest.approx(4.25)       # 17,000 / 4.
    assert terminated["mean_reward_bps"] == pytest.approx(-(21_000 + 500) / 4 / 1_000)
    assert terminated["mean_shortfall_bps_uncertainty"]["status"] == "insufficient_days"
    assert stats["completed"] == {"n": 2, "mean_shortfall_bps": 3.0, "median_shortfall_bps": 2.0}
    assert stats["settled"] == {"n": 1, "mean_shortfall_bps": 9.0, "mean_settlement_fraction": 0.5,
                                "mean_settlement_cost_bps": 3.0}
    assert stats["truncated"]["n"] == 1 and stats["truncated"]["mean_fill_shortfall_bps"] == 1.0
    assert stats["truncated"]["mean_filled_fraction"] == 0.5
    assert stats["mean_filled_fraction"] == pytest.approx((1 + 1 + 0.5 + 0.5 + 0.5) / 5)
    assert stats["mean_settlement_fraction"] == pytest.approx(0.5 / 5)
    assert stats["mean_residual_fraction"] == pytest.approx((0 + 0 + 0 + 0.5 + 0.5) / 5)
    assert stats["settlement_label"] == "heavy"   # 0.10 of the target on average.
    assert stats["n_settlements_exceeding_cap"] == 0
    assert stats["total_fees_ticks"] == 500 and stats["mean_fees_bps"] == pytest.approx(0.1)
    assert stats["total_fill_fees_ticks"] == 460 and stats["total_settlement_fees_ticks"] == 40
    assert stats["mean_settlement_fees_bps"] == pytest.approx(0.04 / 5)
    assert stats["total_risk_ticks"] == 500 and stats["mean_risk_bps"] == pytest.approx(0.1)
    # Six completed rows and one half-settled row average 0.5 / 7 < 0.10 of the target: light.
    light = group_statistics(
        [_row("twap", f"train/s1/i{i}", DATES[0], "completed", 2_000) for i in range(6)]
        + [rows[2]]
    )
    assert light["settlement_label"] == "light" and light["n_settled"] == 1
    assert group_statistics(rows[:2])["settlement_label"] == "none"


def test_settlement_label_thresholds() -> None:
    assert settlement_label(None, 0) == "none" and settlement_label(0.0, 0) == "none"
    assert settlement_label(0.05, 3) == "light"
    assert settlement_label(0.10, 1) == "heavy" and settlement_label(0.9, 1) == "heavy"


def test_group_statistics_of_an_empty_group_are_none_not_zero() -> None:
    stats = group_statistics([])
    assert stats["n_episodes"] == 0 and stats["completion_by_fills_rate"] is None
    assert stats["final_completion_rate"] is None and stats["settlement_label"] == "none"
    assert stats["terminated"]["mean_shortfall_bps"] is None
    assert stats["terminated"]["median_shortfall_bps"] is None


def test_paired_differences_are_taken_on_identical_episode_ids_only() -> None:
    rows = [
        _row("twap", "train/s1/i0", DATES[0], "completed", 2_000),
        _row("twap", "train/s1/i1", DATES[0], "completed", 4_000),
        _row("twap", "train/s1/i2", DATES[1], "truncated", 1_000),
        _row("twap", "train/s1/i3", DATES[1], "settled", 6_000, terminal=1_000),
        _row("immediate", "train/s1/i0", DATES[0], "completed", 5_000),   # +3 bps.
        _row("immediate", "train/s1/i1", DATES[0], "completed", 3_000),   # -1 bps.
        _row("immediate", "train/s1/i2", DATES[1], "completed", 2_000),   # Reference truncated.
        _row("immediate", "train/s1/i3", DATES[1], "completed", 6_000),   # 0.
    ]
    (paired,) = paired_differences(rows, "twap")
    assert paired["policy"] == "immediate" and paired["reference"] == "twap"
    assert paired["n_pairs"] == 3 and paired["n_excluded_truncated"] == 1
    assert paired["n_missing"] == 0 and paired["n_both_completed_by_fills"] == 2
    assert paired["n_either_settled"] == 1
    assert paired["mean_diff_bps"] == pytest.approx((3 - 1 + 0) / 3)
    assert paired["median_diff_bps"] == 0.0 and paired["total_diff_ticks"] == 2_000
    assert paired["fraction_policy_cheaper"] == pytest.approx(1 / 3)
    assert paired["fraction_equal"] == pytest.approx(1 / 3)
    assert paired["mean_diff_bps_uncertainty"]["status"] == "insufficient_days"
    with pytest.raises(ValueError, match="not in the run"):
        paired_differences(rows, "vwap")


def test_day_cluster_bootstrap_needs_enough_days_and_is_seeded() -> None:
    few = {d: [1.0, 2.0] for d in ("a", "b", "c", "d")}
    assert day_cluster_bootstrap(few) == {"status": "insufficient_days", "n_days": 4,
                                         "min_days": 5}
    enough = {f"d{i}": [float(i), float(i) + 1.0] for i in range(6)}
    first = day_cluster_bootstrap(enough, n_resamples=500, seed=3)
    second = day_cluster_bootstrap(enough, n_resamples=500, seed=3)
    assert first == second and first["status"] == "ok" and first["n_days"] == 6
    assert first["ci95_low"] <= 3.0 <= first["ci95_high"]      # The pooled mean is 3.0.
    assert first["ci95_low"] >= 0.5 and first["ci95_high"] <= 5.5   # Between the day means.
    constant = {f"d{i}": [7.0] * 3 for i in range(5)}
    interval = day_cluster_bootstrap(constant)
    assert interval["ci95_low"] == interval["ci95_high"] == 7.0
    assert day_cluster_bootstrap({"a": []}, min_days=1)["status"] == "insufficient_days"


# --- end to end on the synthetic experiment --------------------------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the synthetic experiment with a held-out day once for the module."""
    config = synthetic_config(
        split={"train": [DATES[0]], "validation": [DATES[1]], "test": [DATES[2]]}
    )
    return run_experiment(config, include_test=True, output_dir=tmp_path_factory.mktemp("runs"))


def test_report_metrics_are_reconstructed_from_the_records_by_a_plain_loop(run_dir: Path) -> None:
    report = build_report(run_dir)
    records = load_records(run_dir)
    assert report["schema"] == REPORT_SCHEMA and report["reference_policy"] == "twap"
    for group in report["per_policy"]:
        episodes = [
            e for r in records if r["config"]["policy"] == group["policy"]
            and r["config"]["role"] == group["role"] for e in r["episodes"]
        ]
        assert group["n_episodes"] == len(episodes)
        terminated = [e["summary"] for e in episodes if not e["summary"]["truncated"]]
        completed = [s for s in terminated if s["completed_by_fills"]]
        settled = [s for s in terminated if s["settled_qty_ticks"] and not s["completed_by_fills"]]
        assert group["n_completed"] == len(completed) and group["n_settled"] == len(settled)
        assert group["final_completion_rate"] == pytest.approx(
            sum(e["summary"]["completed"] for e in episodes) / len(episodes)
        )
        assert group["mean_settlement_fraction"] == pytest.approx(
            sum(e["summary"]["settlement_fraction"] for e in episodes) / len(episodes)
        )
        assert group["total_settlement_fees_ticks"] == sum(
            e["summary"]["settlement_fees_ticks"] for e in episodes
        )
        assert group["n_truncated"] == sum(e["summary"]["truncated"] for e in episodes)
        bps = sorted(s["shortfall_bps"] for s in terminated)
        assert group["terminated"]["mean_shortfall_bps"] == pytest.approx(sum(bps) / len(bps))
        assert group["terminated"]["median_shortfall_bps"] == bps[math.ceil(0.5 * len(bps)) - 1]
        assert group["terminated"]["p95_shortfall_bps"] == bps[math.ceil(0.95 * len(bps)) - 1]
        assert group["terminated"]["total_shortfall_ticks"] == sum(
            s["shortfall_ticks"] for s in terminated
        )
        assert group["total_fees_ticks"] == sum(e["summary"]["fees_ticks"] for e in episodes)
        assert group["mean_steps"] == pytest.approx(
            sum(e["summary"]["n_steps"] for e in episodes) / len(episodes)
        )
    by_day = {(g["policy"], g["role"], g["date"]): g for g in report["per_day"]}
    for group in report["per_policy"]:
        days = [g for k, g in by_day.items() if k[:2] == (group["policy"], group["role"])]
        assert sum(g["n_episodes"] for g in days) == group["n_episodes"]
        assert sum(g["terminated"]["total_shortfall_ticks"] for g in days) == (
            group["terminated"]["total_shortfall_ticks"]
        )


def test_paired_rows_match_a_direct_join_of_the_records(run_dir: Path) -> None:
    report = build_report(run_dir, reference="immediate")
    rows = episode_rows(load_records(run_dir))
    for paired in report["paired"]:
        mine = {r.episode_id: r for r in rows if r.policy == paired["policy"]
                and r.role == paired["role"]}
        theirs = {r.episode_id: r for r in rows if r.policy == "immediate"
                  and r.role == paired["role"]}
        diffs = [mine[i].shortfall_bps - theirs[i].shortfall_bps for i in theirs
                 if mine[i].terminated and theirs[i].terminated]
        assert paired["n_pairs"] == len(diffs) == len(theirs)
        assert paired["mean_diff_bps"] == pytest.approx(sum(diffs) / len(diffs))


def test_held_out_records_are_excluded_unless_requested(run_dir: Path) -> None:
    plain = build_report(run_dir)
    assert plain["roles"] == ["train", "validation"] and not plain["include_test"]
    assert all(g["role"] != "test" for g in plain["per_policy"] + plain["per_day"])
    assert all(p["role"] != "test" for p in plain["paired"])
    held = build_report(run_dir, include_test=True)
    assert held["roles"] == ["train", "validation", "test"]
    assert any(g["role"] == "test" and g["held_out"] for g in held["per_policy"])


def test_one_day_groups_report_insufficient_uncertainty(run_dir: Path) -> None:
    report = build_report(run_dir)
    for group in report["per_policy"]:
        uncertainty = group["terminated"]["mean_shortfall_bps_uncertainty"]
        assert uncertainty == {"status": "insufficient_days", "n_days": 1, "min_days": 5}
    lowered = build_report(run_dir, min_days=1)
    for group in lowered["per_policy"]:
        assert group["terminated"]["mean_shortfall_bps_uncertainty"]["status"] == "ok"


def test_write_report_exports_json_csv_and_valid_rst(run_dir: Path, tmp_path: Path) -> None:
    docutils = pytest.importorskip("docutils.core")
    report = build_report(run_dir)
    rows = episode_rows(load_records(run_dir))
    paths = write_report(report, tmp_path / "report", rows)
    assert set(paths) == {"json", "policies", "days", "paired", "episodes", "rst"}
    assert json.loads(paths["json"].read_text())["schema"] == REPORT_SCHEMA
    with paths["episodes"].open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == len(rows) and set(episodes[0]) == set(report_module.EPISODE_COLUMNS)
    assert {e["outcome"] for e in episodes} <= set(report_module.OUTCOMES)
    with paths["policies"].open() as handle:
        policies = list(csv.DictReader(handle))
    assert {p["policy"] for p in policies} == {"twap", "immediate", "cap_aware_twap"}
    assert "terminated.mean_shortfall_bps" in policies[0] and "settlement_label" in policies[0]
    text = paths["rst"].read_text()
    assert "Smoke run" in text and "insufficient" in text and "Paired differences" in text
    assert "terminal_settlement" in text and "settlement frac" in text
    assert "completed by fills" in text and "either settled" in text
    assert not any(line != line.rstrip() for line in text.splitlines())
    problems: list[str] = []

    class Collector:
        def write(self, message: str) -> None:
            problems.append(message)

    docutils.publish_doctree(
        text, settings_overrides={"report_level": 2, "halt_level": 5,
                                  "warning_stream": Collector(), "file_insertion_enabled": False},
    )
    assert not problems, "".join(problems)
    assert not list((tmp_path / "report").glob("*.tmp"))


def test_diagnostic_and_settlement_heavy_policies_are_labelled(tmp_path: Path) -> None:
    config = synthetic_config(policies={"names": ["twap", "zero_action"]})
    run_dir = run_experiment(config, output_dir=tmp_path)
    report = build_report(run_dir)
    zero = [g for g in report["per_policy"] if g["policy"] == "zero_action"]
    assert zero and all(g["diagnostic"] and g["settlement_label"] == "heavy" for g in zero)
    assert all(g["mean_settlement_fraction"] == 1.0 and g["completion_by_fills_rate"] == 0.0
               and g["final_completion_rate"] == 1.0 for g in zero)
    assert all(not g["diagnostic"] for g in report["per_policy"] if g["policy"] == "twap")
    text = report_module.render_rst(report)
    assert "zero_action (diagnostic)" in text and "Diagnostic policies" in text
    assert "Settlement-heavy" in text and "``zero_action`` (train)" in text
    (paired,) = [p for p in report["paired"] if p["role"] == "train"]
    assert paired["policy"] == "zero_action" and paired["n_either_settled"] >= 1


def test_cli_writes_the_report_under_the_run_directory(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report_module.main([str(run_dir), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert (run_dir / "report" / "report.rst").exists() and "rst:" in out
    assert report_module.main([str(run_dir), "--reference", "vwap", "--quiet"]) == 1
    assert "not in the run" in capsys.readouterr().err


def test_episode_rows_carry_the_record_fields_exactly(run_dir: Path) -> None:
    records = load_records(run_dir)
    rows = episode_rows(records)
    record = records[0]
    first = next(r for r in rows if r.policy == record["config"]["policy"]
                 and r.role == record["config"]["role"] and r.index == 0)
    summary = record["episodes"][0]["summary"]
    assert first.episode_id == record["episode_ids"][0]
    assert first.shortfall_ticks == summary["shortfall_ticks"]
    assert first.fill_shortfall_ticks == (
        summary["shortfall_ticks"] - summary["terminal_component_ticks"]
    )
    assert replace(first, policy="x").policy == "x"   # Rows are frozen dataclasses.
