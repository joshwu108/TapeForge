"""Baselines: TWAP completes when liquidity allows, immediate hits the cap, the cap-aware rule."""

from __future__ import annotations

import pytest

from tapeforge.execution.baselines import (
    BASELINES,
    cap_aware_twap,
    cap_aware_twap_quantity,
    immediate,
    twap,
    twap_quantity,
)
from tapeforge.execution.runner import Observation, run_episode, summarize_episode
from tapeforge.execution.spec import EpisodeSpec
from test_execution.tapes import DEEP, P0, Q_STAR, slice_from_rows, worksheet_spec, worksheet_tape


def _observation(remaining: int, time_remaining: int, newest_total: int = DEEP,
                 cap_bps: int = 1_000) -> Observation:
    import numpy as np

    return Observation(
        vector=np.zeros(10), remaining_qty_ticks=remaining,
        time_remaining_buckets=time_remaining, target_qty_ticks=Q_STAR, side=1,
        newest_bucket_total_qty_ticks=newest_total, participation_cap_bps=cap_bps,
    )


def test_twap_quantity_is_the_floor_share_and_the_last_step_takes_everything() -> None:
    assert twap_quantity(300_000, 3) == 100_000
    assert twap_quantity(10, 3) == 3 and twap_quantity(7, 2) == 3 and twap_quantity(4, 1) == 4
    assert twap_quantity(0, 5) == 0
    with pytest.raises(ValueError, match="time_remaining"):
        twap_quantity(10, 0)
    with pytest.raises(ValueError, match="non-negative"):
        twap_quantity(-1, 2)


def test_twap_completes_exactly_when_liquidity_allows() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    trajectory = run_episode(spec, market, twap)
    assert [r.fill_qty_ticks for r in trajectory.results] == [100_000, 100_000, 100_000]
    assert summarize_episode(trajectory).completed
    # A target that does not divide: 10 ticks over 3 steps submits 3, 3, 4 and completes.
    small = worksheet_spec(side=-1, target_qty_ticks=10)
    trajectory = run_episode(small, market, twap)
    assert [r.submitted_qty_ticks for r in trajectory.results] == [3, 3, 4]
    assert trajectory.final_state.filled_qty_ticks == 10


def test_twap_leaves_a_residual_only_when_the_cap_binds() -> None:
    spec = worksheet_spec(side=1)
    thin = slice_from_rows([(DEEP, P0)], [(500_000, 10_002), (500_000, 10_005), (500_000, 10_001),
                                          (DEEP, 10_010)])  # cap 50,000 per bucket
    trajectory = run_episode(spec, thin, twap)
    # Shares: 300,000 / 3, then 250,000 / 2, then all 200,000 left; each fills the 50,000 cap.
    assert [r.fill_qty_ticks for r in trajectory.results] == [50_000, 50_000, 50_000]
    assert [r.submitted_qty_ticks for r in trajectory.results] == [100_000, 125_000, 200_000]
    summary = summarize_episode(trajectory)
    assert summary.residual_qty_ticks == 150_000 and summary.n_capped_steps == 3


def test_immediate_hits_the_cap_when_it_binds() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    trajectory = run_episode(spec, market, immediate)
    assert trajectory.actions == (1.0, 1.0)
    first, second = trajectory.results
    assert first.submitted_qty_ticks == Q_STAR
    assert first.fill_qty_ticks == first.cap_qty_ticks == 200_000
    assert second.fill_qty_ticks == 100_000 and second.terminated
    assert summarize_episode(trajectory).n_capped_steps == 1
    assert immediate(_observation(5, 1)) == 1.0


def test_cap_aware_quantity_equals_twap_when_liquidity_is_deep() -> None:
    # Deep book: the future can absorb 2 x 200,000 >= 300,000, so no catch-up.
    assert cap_aware_twap_quantity(300_000, 3, DEEP, 1_000) == twap_quantity(300_000, 3)
    assert cap_aware_twap(_observation(300_000, 3)) == twap(_observation(300_000, 3))


def test_cap_aware_quantity_front_loads_what_the_future_cannot_absorb() -> None:
    # Cap estimate 50,000 per bucket, 2 future decisions: 100,000 fits, 200,000 must go now.
    assert cap_aware_twap_quantity(300_000, 3, 500_000, 1_000) == 200_000
    # Last decision: everything that is left, whatever the estimate.
    assert cap_aware_twap_quantity(120_000, 1, 500_000, 1_000) == 120_000
    # An empty newest bucket estimates zero liquidity: submit everything.
    assert cap_aware_twap_quantity(300_000, 3, 0, 1_000) == 300_000
    # Never more than what remains.
    assert cap_aware_twap_quantity(10, 3, 0, 1_000) == 10
    with pytest.raises(ValueError, match="non-negative"):
        cap_aware_twap_quantity(10, 3, -1, 1_000)


def test_cap_aware_twap_submits_the_catch_up_quantity_on_a_thin_tape() -> None:
    spec = worksheet_spec(side=1)
    thin = slice_from_rows([(500_000, P0)], [(500_000, 10_002), (500_000, 10_005),
                                             (500_000, 10_001), (DEEP, 10_010)])
    plain = run_episode(spec, thin, twap)
    aware = run_episode(spec, thin, cap_aware_twap)
    # Estimate 50,000 per bucket. t=0: 300,000 - 2 x 50,000 = 200,000 must go now;
    # t=1: 250,000 - 50,000 = 200,000; t=2: everything left (200,000).
    assert [r.submitted_qty_ticks for r in aware.results] == [200_000, 200_000, 200_000]
    # Over-submitting is free under the fill rule, so both fill the same capped 150,000.
    assert aware.final_state.filled_qty_ticks == plain.final_state.filled_qty_ticks == 150_000
    assert aware.results[0].fill_qty_ticks == 50_000   # the real cap still binds


def test_cap_aware_twap_catches_up_after_a_thin_bucket() -> None:
    """Deep history, one thin bucket visible: the policy raises its share, then relaxes."""
    spec = EpisodeSpec(side=1, target_qty_ticks=300_000, horizon_buckets=4, arrival_lookback=1)
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), (200_000, 10_005), (DEEP, 10_001),
                                            (DEEP, 10_003), (DEEP, 10_010)])
    trajectory = run_episode(spec, market, cap_aware_twap)
    submitted = [r.submitted_qty_ticks for r in trajectory.results]
    fills = [r.fill_qty_ticks for r in trajectory.results]
    # t=0 sees bucket -1 (deep): share 75,000, fills 75,000. t=1 sees bucket 0 (deep): share
    # 75,000, but bucket 1 is thin and fills only its cap, 20,000. t=2 sees that thin bucket:
    # 205,000 left and one later decision estimated at 20,000, so 185,000 goes now and fills
    # in the deep bucket 2. t=3 submits the last 20,000.
    assert submitted == [75_000, 75_000, 185_000, 20_000]
    assert fills == [75_000, 20_000, 185_000, 20_000]
    assert trajectory.final_state.filled_qty_ticks == 300_000


def test_baseline_registry_names_the_three_policies() -> None:
    assert set(BASELINES) == {"twap", "immediate", "cap_aware_twap"}
    assert BASELINES["twap"] is twap and BASELINES["immediate"] is immediate
    assert BASELINES["cap_aware_twap"] is cap_aware_twap


def test_baselines_return_actions_inside_the_unit_interval() -> None:
    for policy in BASELINES.values():
        cases = [(1, 1, 0), (7, 3, 10), (300_000, 300, DEEP), (0, 2, 5)]
        for remaining, time_left, newest in cases:
            action = policy(_observation(remaining, time_left, newest))
            assert 0.0 <= action <= 1.0
