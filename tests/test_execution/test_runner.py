"""Replay runner: observations, trajectories, summaries, run records and reproducibility."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W
from tapeforge.execution.baselines import immediate, twap
from tapeforge.execution.days import DayTape, synthetic_day
from tapeforge.execution.runner import (
    RUN_RECORD_SCHEMA,
    EpisodeSummary,
    Observation,
    Trajectory,
    code_revision,
    nearest_rank_percentile,
    observation_for,
    run_episode,
    run_policy,
    summarize_episode,
    summarize_set,
    write_run_record,
)
from tapeforge.execution.simulator import OBSERVATION_FIELDS, observe, reset
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import SplitDates, Splits, build_splits
from tapeforge.execution.state import MarketSlice
from test_execution.policies import rollout, twap_policy
from test_execution.tapes import (
    DEEP,
    P0,
    Q_STAR,
    slice_from_rows,
    worksheet_spec,
    worksheet_tape,
)

SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, latency_buckets=0,
                   arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02", "2024-01-03")


def _synthetic_splits(seed: int = 1) -> Splits:
    days = [
        synthetic_day(seed, DATES[0], 80, start_us=0),
        synthetic_day(seed + 1, DATES[1], 80, start_us=80 * W),
        synthetic_day(seed + 2, DATES[2], 60, start_us=160 * W),
    ]
    return build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],)), SPEC)


# --- observations and trajectories -----------------------------------------------------


def test_observation_bundles_the_vector_with_exact_integers() -> None:
    spec = worksheet_spec(side=-1, arrival_lookback=1)
    market = worksheet_tape()
    state = reset(spec, market)
    observation = observation_for(state, market, spec)
    assert isinstance(observation, Observation)
    np.testing.assert_array_equal(observation.vector, observe(state, market, spec))
    assert observation.remaining_qty_ticks == Q_STAR and observation.time_remaining_buckets == 3
    assert observation.target_qty_ticks == Q_STAR and observation.side == -1
    assert observation.newest_bucket_total_qty_ticks == DEEP   # bucket -1 (d = 0)
    assert observation.participation_cap_bps == 1_000


def test_run_episode_matches_the_plain_rollout_helper() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    trajectory = run_episode(spec, market, twap)
    states, results = rollout(spec, market, twap_policy)
    assert isinstance(trajectory, Trajectory)
    assert list(trajectory.states) == states and list(trajectory.results) == results
    assert len(trajectory.observations) == len(trajectory.actions) == 3
    assert trajectory.observations[0].shape == (len(OBSERVATION_FIELDS),)
    assert trajectory.final_state is states[-1] or trajectory.final_state == states[-1]
    assert trajectory.last_result.terminated


def test_run_episode_stops_on_truncation() -> None:
    spec = worksheet_spec(side=1)
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), (DEEP, 10_005)])  # one bucket short
    trajectory = run_episode(spec, market, twap)
    assert trajectory.last_result.truncated and len(trajectory.results) == 3
    assert trajectory.final_state.filled_qty_ticks == 200_000


def test_policy_sees_only_what_observe_exposes() -> None:
    seen: list[Observation] = []

    def spy(observation: Observation) -> float:
        seen.append(observation)
        return 0.0

    spec, market = worksheet_spec(side=1), worksheet_tape()
    run_episode(spec, market, spy)
    assert [o.time_remaining_buckets for o in seen] == [3, 2, 1]
    assert all(o.remaining_qty_ticks == Q_STAR for o in seen)
    assert [o.newest_bucket_total_qty_ticks for o in seen] == [DEEP, DEEP, DEEP]


# --- episode summaries -------------------------------------------------------------------


def test_episode_summary_of_the_worksheet_buy_table() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    summary = summarize_episode(run_episode(spec, market, twap))
    assert isinstance(summary, EpisodeSummary)
    assert summary.shortfall_ticks == 3_800_800 and summary.completed
    assert summary.shortfall_bps == pytest.approx(12.669, abs=1e-3)
    assert summary.fees_ticks == 3_000_800 and summary.exec_notional_ticks == 3_000_800_000
    assert summary.filled_qty_ticks == Q_STAR and summary.residual_qty_ticks == 0
    assert summary.n_steps == 3 and summary.terminated and not summary.truncated
    assert summary.reward_sum_ticks == -3_800_800 and summary.cost_sum_ticks == 3_800_800
    assert summary.risk_sum_ticks == 0 and summary.terminal_component_ticks == 0
    assert summary.n_capped_steps == 0 and summary.terminal_price_ticks == 0


def test_episode_summary_includes_the_residual_term_when_the_horizon_ends() -> None:
    spec, market = worksheet_spec(side=1, residual_mu=5), worksheet_tape()
    summary = summarize_episode(run_episode(spec, market, lambda _: 0.0))
    # Nothing filled: IS = side x (P_T - P_0) x Q* + mu x Q* = 10 x 300,000 + 5 x 300,000.
    assert summary.shortfall_ticks == 3_000_000 + 1_500_000
    assert summary.terminal_price_ticks == 10_010 and not summary.completed
    assert summary.residual_qty_ticks == Q_STAR and summary.terminal_component_ticks == 4_500_000
    assert summary.reward_sum_ticks == -summary.shortfall_ticks


def test_episode_summary_of_a_truncated_episode_reports_the_fills_only() -> None:
    spec = worksheet_spec(side=1)
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), (DEEP, 10_005)])
    summary = summarize_episode(run_episode(spec, market, twap))
    assert summary.truncated and not summary.terminated and not summary.completed
    assert summary.shortfall_ticks == (2 * 100_000 + 5 * 100_000) + 1_000_200 + 1_000_500
    assert summary.residual_qty_ticks == 100_000 and summary.terminal_price_ticks == 0


def test_episode_summary_counts_capped_fills() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    summary = summarize_episode(run_episode(spec, market, immediate))
    assert summary.n_capped_steps == 1 and summary.n_steps == 2 and summary.completed


# --- set summaries -------------------------------------------------------------------


def test_nearest_rank_percentile_is_the_textbook_definition() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert nearest_rank_percentile(values, 0) == 1.0
    assert nearest_rank_percentile(values, 50) == 3.0
    assert nearest_rank_percentile(values, 95) == 5.0
    assert nearest_rank_percentile(values, 100) == 5.0
    assert nearest_rank_percentile([7.0], 99) == 7.0
    assert nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0   # ceil(2) = 2nd value
    with pytest.raises(ValueError, match="undefined"):
        nearest_rank_percentile([], 50)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        nearest_rank_percentile(values, 101)


def test_set_summary_matches_a_plain_loop_over_the_trajectories() -> None:
    splits = _synthetic_splits()
    record, trajectories = run_policy(splits, "train", twap, "twap", SPEC, n_episodes=12, seed=4)
    summaries = [summarize_episode(t) for t in trajectories]
    got = summarize_set(summaries)
    assert record["summary"] == got.to_json()

    n = len(trajectories)
    bps, completed, truncated, fees, shortfall, residual_fraction, capped = [], 0, 0, 0, 0, 0.0, 0
    for trajectory in trajectories:
        final, spec = trajectory.final_state, trajectory.spec
        is_ticks = spec.side * (
            final.exec_notional_ticks - final.arrival_price_ticks * final.filled_qty_ticks
        ) + final.fees_ticks
        left = spec.target_qty_ticks - final.filled_qty_ticks
        last = trajectory.results[-1]
        if left and last.terminated:
            is_ticks += spec.side * (last.terminal_price_ticks - final.arrival_price_ticks) * left
        bps.append(is_ticks * 10_000 / (final.arrival_price_ticks * spec.target_qty_ticks))
        completed += left == 0
        truncated += last.truncated
        fees += final.fees_ticks
        shortfall += is_ticks
        residual_fraction += left / spec.target_qty_ticks
        for r in trajectory.results:
            capped += r.fill_qty_ticks == r.cap_qty_ticks < r.submitted_qty_ticks
    ordered = sorted(bps)
    assert got.n_episodes == n and got.n_completed == completed and got.n_truncated == truncated
    assert got.completion_rate == completed / n
    assert got.mean_shortfall_bps == sum(bps) / n
    assert got.p50_shortfall_bps == ordered[math.ceil(0.5 * n) - 1]
    assert got.p95_shortfall_bps == ordered[math.ceil(0.95 * n) - 1]
    assert got.p99_shortfall_bps == ordered[math.ceil(0.99 * n) - 1]
    assert got.max_shortfall_bps == ordered[-1]
    assert got.total_shortfall_ticks == shortfall and got.total_fees_ticks == fees
    assert got.mean_residual_fraction == residual_fraction / n
    assert got.mean_capped_steps == capped / n


def test_set_summary_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        summarize_set([])


# --- run records ---------------------------------------------------------------------


def _without_volatile_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in ("created_utc", "code_revision")}


def test_run_record_has_the_documented_shape() -> None:
    splits = _synthetic_splits()
    record, trajectories = run_policy(splits, "validation", twap, "twap", SPEC, 5, seed=9)
    assert record["schema"] == RUN_RECORD_SCHEMA
    assert set(record) == {"schema", "created_utc", "code_revision", "config", "split",
                           "observation_statistics", "episodes", "summary"}
    assert record["config"] == {
        "policy": "twap", "role": "validation", "n_episodes": 5, "seed": 9,
        "rho_choices_bps": [200, 500, 1_000],
        "template": {
            "horizon_buckets": 5, "participation_cap_bps": 1_000, "slippage_ticks": 0,
            "fee_bps": 10, "latency_buckets": 0, "data_delay_buckets": 0, "risk_lambda": 0,
            "residual_mu": 0, "arrival_lookback": 3, "fill_rule": "bucket_vwap_capped",
        },
    }
    assert record["split"] == splits.record.to_json()
    assert record["observation_statistics"] is None
    assert len(record["episodes"]) == len(trajectories) == 5
    episode = record["episodes"][0]
    assert set(episode) == {"draw", "spec", "summary"}
    assert episode["draw"]["role"] == "validation" and episode["draw"]["index"] == 0
    assert episode["spec"]["target_qty_ticks"] == trajectories[0].spec.target_qty_ticks
    assert episode["summary"] == summarize_episode(trajectories[0]).to_json()
    assert record["code_revision"].keys() == {"commit", "dirty"}
    json.dumps(record)  # every value is JSON-serialisable


def test_same_seed_gives_byte_identical_records_and_different_seeds_differ(tmp_path: Path) -> None:
    splits = _synthetic_splits()
    first, _ = run_policy(splits, "train", twap, "twap", SPEC, 8, seed=21)
    second, _ = run_policy(splits, "train", twap, "twap", SPEC, 8, seed=21)
    first_path = write_run_record(first, tmp_path)
    second_path = write_run_record(second, tmp_path)
    assert first_path != second_path and first_path.parent == tmp_path
    strip = _without_volatile_fields
    assert json.dumps(strip(json.loads(first_path.read_text())), sort_keys=True) == json.dumps(
        strip(json.loads(second_path.read_text())), sort_keys=True
    )
    other, _ = run_policy(splits, "train", twap, "twap", SPEC, 8, seed=22)
    assert strip(other) != strip(first)
    assert [e["draw"]["position"] for e in other["episodes"]] != [
        e["draw"]["position"] for e in first["episodes"]
    ]


def test_run_record_file_name_carries_policy_role_and_seed(tmp_path: Path) -> None:
    record, _ = run_policy(_synthetic_splits(), "train", immediate, "immediate", SPEC, 2, seed=3)
    path = write_run_record(record, tmp_path / "nested" / "runs")
    assert path.exists() and path.name.endswith("-immediate-train-seed3.json")
    assert path.name[:16].replace("T", "").isdigit() or path.name[:8].isdigit()
    loaded = json.loads(path.read_text())
    assert loaded["config"]["seed"] == 3 and loaded["summary"]["n_episodes"] == 2


def test_run_policy_validates_its_inputs() -> None:
    splits = _synthetic_splits()
    with pytest.raises(ValueError, match="n_episodes must be positive"):
        run_policy(splits, "train", twap, "twap", SPEC, 0, seed=1)
    with pytest.raises(ValueError, match="geometry"):
        run_policy(splits, "train", twap, "twap", EpisodeSpec(side=1, target_qty_ticks=1), 1, 1)
    with pytest.raises(ValueError, match="no admissible episode start"):
        run_policy(splits, "test", twap, "twap", SPEC, 1, seed=1)


def test_code_revision_reports_a_commit_or_none() -> None:
    revision = code_revision()
    assert set(revision) == {"commit", "dirty"}
    if revision["commit"] is not None:
        assert len(revision["commit"]) == 40 and isinstance(revision["dirty"], bool)


def test_run_record_stores_observation_statistics_verbatim() -> None:
    stats = {"fields": ["a"], "mean": [0.0], "std": [1.0]}
    record, _ = run_policy(_synthetic_splits(), "train", twap, "twap", SPEC, 1, seed=1,
                           observation_statistics=stats)
    assert record["observation_statistics"] == stats


def test_runs_directory_is_ignored_by_git() -> None:
    gitignore = (Path(__file__).parent.parent.parent / ".gitignore").read_text().splitlines()
    assert "runs/" in gitignore


def test_market_slice_passed_to_run_episode_is_not_modified() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    before = market.total_quantity.copy()
    run_episode(spec, market, immediate)
    assert isinstance(market, MarketSlice)
    np.testing.assert_array_equal(market.total_quantity, before)


def test_day_tape_round_trip_through_a_run_record_source(tmp_path: Path) -> None:
    day = DayTape(DATES[0], 0, W, slice_from_rows([], [(DEEP, P0)] * 40), "rows")
    splits = build_splits([day], SplitDates((DATES[0],), ()), SPEC)
    record, _ = run_policy(splits, "train", twap, "twap", SPEC, 1, seed=1)
    assert record["split"]["days"][0]["source"] == "rows"
