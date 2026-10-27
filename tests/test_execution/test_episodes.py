"""Tests for tapeforge.execution.episodes: the required window, the reference volume, the target
quantity derived from it, and episode construction from a trade array.

Rounding cases are pinned with exact integers; ``_day`` generates a uniform
synthetic day for the end-to-end checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate, reference, windows
from tapeforge.execution import (
    EpisodeSpec,
    build_market_slice,
    make_episode,
    reference_volume_ticks,
    required_window,
    reset,
    target_from_reference_volume,
)
from tapeforge.execution.episodes import round_half_up_bps
from test_execution.policies import rollout, twap_policy
from test_execution.tapes import DEEP, P0, slice_from_rows

W = contract.DEFAULT_BUCKET_WIDTH_US


def test_required_window_covers_history_fills_and_the_terminal_search() -> None:
    assert required_window(EpisodeSpec(side=1, target_qty_ticks=1)) == (300, 360)
    spec = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=2,
                       latency_buckets=1, data_delay_buckets=0)
    assert required_window(spec) == (5, 8)
    deep_lookback = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5,
                                arrival_lookback=7, data_delay_buckets=2)
    assert required_window(deep_lookback) == (9, 12)


@pytest.mark.parametrize(
    "value, rate_bps, expected",
    [(1_234_567, 200, 24_691), (1_234_567, 500, 61_728), (1_234_567, 1_000, 123_457),
     (25, 200, 1), (24, 200, 0), (75, 200, 2), (0, 1_000, 0)],
)
def test_round_half_up_bps(value: int, rate_bps: int, expected: int) -> None:
    assert round_half_up_bps(value, rate_bps) == expected


@pytest.mark.parametrize("rho_bps, expected", [(200, 24_691), (500, 61_728), (1_000, 123_456)])
def test_target_for_each_accepted_rho(rho_bps: int, expected: int) -> None:
    """A 10 % target fraction rounds up to 123,457 and is then clipped by the cap (floor, 123,456).

    With the cap lifted to 100 % every fraction gives the half-up rounded value.
    """
    assert target_from_reference_volume(1_234_567, rho_bps, 1_000) == expected
    assert target_from_reference_volume(1_234_567, rho_bps, 10_000) == round_half_up_bps(
        1_234_567, rho_bps
    )


def test_target_is_floored_at_one_tick() -> None:
    assert target_from_reference_volume(0, 1_000, 1_000) == 1
    assert target_from_reference_volume(24, 200, 1_000) == 1   # 0.48 ticks rounds to 0, floored.
    assert target_from_reference_volume(25, 200, 1_000) == 1   # 0.5 rounds up.


def test_target_is_capped_at_the_participation_cap_times_the_reference_volume() -> None:
    assert target_from_reference_volume(1_000_000, 5_000, 1_000) == 100_000  # 50 % asked, 10 % cap.
    assert target_from_reference_volume(1_000_000, 1_000, 1_000) == 100_000  # Fraction equals cap.
    assert target_from_reference_volume(1_000_005, 1_000, 1_000) == 100_000  # Rounds up, then cap.
    with pytest.raises(ValueError, match="non-negative"):
        target_from_reference_volume(-1, 200, 1_000)


def test_reference_volume_sums_exactly_the_h_buckets_before_t0() -> None:
    market = slice_from_rows([(5, P0), (7, P0), (11, P0), None], [(DEEP, P0), (DEEP, P0)])
    assert reference_volume_ticks(market, 4) == 23
    assert reference_volume_ticks(market, 2) == 11
    with pytest.raises(ValueError, match="needs 5 buckets"):
        reference_volume_ticks(market, 5)


def _day(seed: int = 5, n_buckets: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Generate a uniform-profile synthetic trade day and its bucket offsets."""
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile="uniform")
    return trades, windows.build_offsets(trades["timestamp"], 0, n_buckets, W)


TEMPLATE = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=2)


def test_make_episode_sizes_q_star_from_the_preceding_volume() -> None:
    trades, offsets = _day()
    spec, market = make_episode(trades, offsets, start_bucket=10, side=-1, rho_bps=500,
                                template=TEMPLATE)
    features = reference.aggregate_reference(trades, offsets)
    v_ref = int(features.total_quantity[5:10].sum())
    assert market.origin == 5 and market.n_after == 7
    assert reference_volume_ticks(market, 5) == v_ref
    assert spec.target_qty_ticks == target_from_reference_volume(v_ref, 500, 1_000)
    assert spec.side == -1 and spec.horizon_buckets == 5 and spec.arrival_lookback == 2
    np.testing.assert_array_equal(market.total_quantity, features.total_quantity[5:17])


def test_make_episode_default_template_is_the_accepted_configuration() -> None:
    trades, offsets = _day(n_buckets=700)
    spec, market = make_episode(trades, offsets, start_bucket=300, side=1, rho_bps=200)
    assert spec.horizon_buckets == 300 and market.origin == 300 and market.n_after == 360
    assert spec.target_qty_ticks >= 1


def test_perturbing_a_bucket_after_t0_leaves_q_star_unchanged() -> None:
    trades, offsets = _day()
    spec, _ = make_episode(trades, offsets, 10, 1, 1_000, template=TEMPLATE)
    louder = trades.copy()
    in_bucket_12 = slice(int(offsets[12]), int(offsets[13]))
    louder["quantity"][in_bucket_12] *= 50
    other, _ = make_episode(louder, offsets, 10, 1, 1_000, template=TEMPLATE)
    assert other.target_qty_ticks == spec.target_qty_ticks
    earlier = trades.copy()
    earlier["quantity"][int(offsets[7]) : int(offsets[8])] *= 50
    changed, _ = make_episode(earlier, offsets, 10, 1, 1_000, template=TEMPLATE)
    assert changed.target_qty_ticks != spec.target_qty_ticks


def test_build_market_slice_clips_at_the_end_of_the_day() -> None:
    trades, offsets = _day()
    market = build_market_slice(trades, offsets, start_bucket=36, n_before=5, n_after=7)
    assert market.origin == 5 and market.n_after == 4  # Buckets 36..39 only.
    with pytest.raises(ValueError, match="buckets of history"):
        build_market_slice(trades, offsets, start_bucket=3, n_before=5, n_after=7)
    with pytest.raises(ValueError, match="past the 40-bucket window"):
        build_market_slice(trades, offsets, start_bucket=40, n_before=5, n_after=7)


def test_an_episode_built_from_generated_trades_runs_end_to_end() -> None:
    trades, offsets = _day()
    spec, market = make_episode(trades, offsets, 10, 1, 1_000, template=TEMPLATE)
    state = reset(spec, market)
    assert state.arrival_price_ticks > 0
    states, results = rollout(spec, market, twap_policy)
    assert results[-1].terminated or results[-1].truncated
    assert states[-1].filled_qty_ticks <= spec.target_qty_ticks
