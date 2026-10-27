"""Tests for leakage controls at the episode level: buckets after the visible range never reach
the target, the arrival price, an observation, or a validation episode's market data.

Perturbations overwrite buckets of a synthetic day tape through
``MarketSlice.with_bucket`` and compare the derived episode against the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W
from tapeforge.execution.baselines import twap
from tapeforge.execution.days import DayTape, episode_at, synthetic_day
from tapeforge.execution.episodes import required_window
from tapeforge.execution.runner import run_episode
from tapeforge.execution.sampling import materialize, sample_episodes
from tapeforge.execution.simulator import observe, reset
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import SplitDates, build_splits
from tapeforge.execution.state import MarketSlice

SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02")


def _perturb_from(tape: MarketSlice, first_bucket: int) -> MarketSlice:
    """Return ``tape`` with every bucket at or after ``first_bucket`` overwritten.

    The replacement buckets carry a large quantity, a heavy sell imbalance and
    a price far from the original, so any read of them changes the result.
    """
    perturbed = tape
    for bucket in range(first_bucket, tape.n_buckets):
        perturbed = perturbed.with_bucket(
            bucket, event_count=97, total_quantity=5_000_000, signed_quantity=-4_000_000,
            vwap_ticks=777_777,
        )
    return perturbed


@pytest.mark.parametrize("delay", [0, 1])
def test_buckets_after_the_visible_range_do_not_change_target_arrival_or_observations(
    delay: int,
) -> None:
    spec_template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=3,
                                data_delay_buckets=delay)
    tape = synthetic_day(seed=9, day=DATES[0], n_buckets=60, start_us=0).tape
    start = 20
    spec, market = episode_at(tape, start, -1, 500, spec_template)
    trajectory = run_episode(spec, market, twap)
    # The target reads the ``horizon_buckets`` buckets before the start and nothing from the
    # start on. (With ``data_delay_buckets`` 1 the newest of those buckets is not yet visible
    # to the policy; the reference volume still includes it, by design.)
    future_spec, _ = episode_at(_perturb_from(tape, start), start, -1, 500, spec_template)
    assert future_spec.target_qty_ticks == spec.target_qty_ticks
    for t, state in enumerate(trajectory.states[:-1]):
        visible_end = start + t - delay   # First bucket the decision at t may not read.
        perturbed_tape = _perturb_from(tape, visible_end)
        other_spec, other_market = episode_at(perturbed_tape, start, -1, 500, spec_template)
        other_spec = spec   # The target may differ with delay 1 (above); the geometry is identical.
        assert reset(other_spec, other_market).arrival_price_ticks == state.arrival_price_ticks
        np.testing.assert_array_equal(
            observe(state, other_market, other_spec), trajectory.observations[t]
        )


def test_perturbing_the_newest_visible_bucket_does_change_the_observation() -> None:
    tape = synthetic_day(seed=9, day=DATES[0], n_buckets=60, start_us=0).tape
    spec, market = episode_at(tape, 20, 1, 500, SPEC)
    state = reset(spec, market)
    baseline = observe(state, market, spec)
    louder = _perturb_from(tape, 19)   # Bucket -1 is visible at t = 0 with zero data delay.
    _, other_market = episode_at(louder, 20, 1, 500, SPEC)
    assert not np.array_equal(observe(reset(spec, other_market), other_market, spec), baseline)


def test_validation_episodes_read_only_their_own_segment() -> None:
    """Every bucket a validation episode reads lies in validation days; training data cannot leak.

    Swapping the training day for different data leaves every validation draw's
    spec and market identical.
    """
    train = DayTape(DATES[0], 0, W, synthetic_day(1, DATES[0], 70, start_us=0).tape, "train")
    validation = synthetic_day(2, DATES[1], 70, start_us=70 * W)
    splits = build_splits([train, validation], SplitDates((DATES[0],), (DATES[1],)), SPEC)
    other_train = DayTape(DATES[0], 0, W, synthetic_day(5, DATES[0], 70, start_us=0).tape, "other")
    other_splits = build_splits([other_train, validation], SplitDates((DATES[0],), (DATES[1],)),
                                SPEC)
    assert splits.record.segments == other_splits.record.segments
    for draw in sample_episodes(splits.record, "validation", 20, seed=3):
        spec, market = materialize(splits, draw, SPEC)
        other_spec, other_market = materialize(other_splits, draw, SPEC)
        assert spec == other_spec
        np.testing.assert_array_equal(market.total_quantity, other_market.total_quantity)
        np.testing.assert_array_equal(market.vwap_ticks, other_market.vwap_ticks)
        n_before, n_after = required_window(SPEC)
        assert market.origin == n_before and market.n_after == n_after
