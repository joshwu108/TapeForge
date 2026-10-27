"""Tests for tapeforge.execution.sampling: a draw is a pure function of (seed, role, index), draws
cover the whole side and target-fraction domain, and ``materialize`` cuts the episode a draw names.

``_splits`` builds three flat single-price days whose prices identify the day a
materialised episode came from.
"""

from __future__ import annotations

import pytest

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W
from tapeforge.execution.days import DayTape
from tapeforge.execution.sampling import (
    EpisodeDraw,
    draw_episode,
    materialize,
    sample_episodes,
)
from tapeforge.execution.spec import ACCEPTED_TARGET_FRACTIONS_BPS, EpisodeSpec
from tapeforge.execution.splits import SplitDates, Splits, build_splits
from test_execution.tapes import DEEP, P0, slice_from_rows

SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, latency_buckets=1,
                   arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02", "2024-01-03")


def _splits() -> Splits:
    """Build splits over three flat days (two train, one validation) with distinct prices."""
    days = [
        DayTape(DATES[0], 0, W, slice_from_rows([], [(DEEP, P0)] * 60), "a"),
        DayTape(DATES[1], 60 * W, W, slice_from_rows([], [(DEEP + 1_000, P0 + 5)] * 60), "b"),
        DayTape(DATES[2], 120 * W, W, slice_from_rows([], [(DEEP, P0 + 9)] * 40), "c"),
    ]
    return build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],)), SPEC)


def test_a_draw_is_a_pure_function_of_seed_role_and_index() -> None:
    record = _splits().record
    first = draw_episode(record, "train", seed=7, index=3)
    again = draw_episode(record, "train", seed=7, index=3)
    assert first == again
    assert sample_episodes(record, "train", 10, seed=7)[3] == first
    assert sample_episodes(record, "train", 4, seed=7)[3] == first   # Independent of n.
    assert draw_episode(record, "train", seed=8, index=3) != first


def test_draws_differ_across_indices_and_roles() -> None:
    record = _splits().record
    train = sample_episodes(record, "train", 50, seed=1)
    assert len({(d.position, d.side, d.rho_bps) for d in train}) > 1
    validation = sample_episodes(record, "validation", 50, seed=1)
    assert all(d.role == "validation" for d in validation)
    assert [d.position for d in validation] != [d.position for d in train]


def test_draw_fields_are_consistent_with_the_record() -> None:
    record = _splits().record
    for draw in sample_episodes(record, "train", 200, seed=3):
        assert isinstance(draw, EpisodeDraw)
        assert 0 <= draw.position < record.n_starts("train")
        assert (draw.segment, draw.start_bucket) == record.start_at("train", draw.position)
        assert draw.date == record.segments[draw.segment].date_at(draw.start_bucket)
        assert draw.side in (1, -1) and draw.rho_bps in ACCEPTED_TARGET_FRACTIONS_BPS
        assert draw.seed == 3 and draw.role == "train"
    assert [d.index for d in sample_episodes(record, "train", 5, seed=3)] == [0, 1, 2, 3, 4]


def test_sides_and_rhos_are_sampled_from_the_whole_domain() -> None:
    draws = sample_episodes(_splits().record, "train", 600, seed=11)
    buys = sum(1 for d in draws if d.side == 1)
    assert 240 <= buys <= 360   # Equal probability; a 5-sigma band around 300.
    counts = {
        rho: sum(1 for d in draws if d.rho_bps == rho) for rho in ACCEPTED_TARGET_FRACTIONS_BPS
    }
    assert all(130 <= c <= 270 for c in counts.values()), counts
    positions = {d.position for d in draws}
    assert min(positions) == 0 and max(positions) == _splits().record.n_starts("train") - 1


def test_custom_rho_choices_and_validation() -> None:
    record = _splits().record
    draw = draw_episode(record, "train", 1, 0, rho_choices_bps=(777,))
    assert draw.rho_bps == 777
    with pytest.raises(ValueError, match="must not be empty"):
        draw_episode(record, "train", 1, 0, rho_choices_bps=())
    with pytest.raises(ValueError, match="unknown role"):
        draw_episode(record, "holdout", 1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        draw_episode(record, "train", -1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        draw_episode(record, "train", 1, -1)
    with pytest.raises(ValueError, match="no admissible episode start"):
        draw_episode(record, "test", 1, 0)
    with pytest.raises(ValueError, match="n_episodes"):
        sample_episodes(record, "train", -1, 1)
    assert sample_episodes(record, "train", 0, 1) == ()


def test_materialize_cuts_the_episode_the_draw_names() -> None:
    splits = _splits()
    draw = draw_episode(splits.record, "validation", seed=5, index=2)
    spec, market = materialize(splits, draw, SPEC)
    assert spec.side == draw.side and spec.horizon_buckets == 5
    expected_target = min(5 * DEEP * draw.rho_bps // 10_000, 5 * DEEP * 1_000 // 10_000)
    assert spec.target_qty_ticks == expected_target
    assert set(market.vwap_ticks.tolist()) == {P0 + 9}   # Validation day only.
    assert market.origin == splits.record.n_before and market.n_after == splits.record.n_after


def test_draw_is_json_serialisable() -> None:
    draw = draw_episode(_splits().record, "train", 2, 9)
    as_json = draw.to_json()
    assert as_json["index"] == 9 and as_json["seed"] == 2 and set(as_json) == {
        "seed", "role", "index", "position", "segment", "start_bucket", "date", "side", "rho_bps"
    }
