"""Chronological splits: hand-built boundaries, embargo widths, purge, and the invariants."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W
from tapeforge.execution.days import DayTape
from tapeforge.execution.episodes import required_window
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import (
    ROLES,
    Segment,
    SplitDates,
    SplitRecord,
    Splits,
    build_splits,
    embargo_buckets,
    purge_window,
)
from test_execution.tapes import DEEP, P0, slice_from_rows

# Hand-built geometry: H = 5, l = 1, L = 3, d = 0 -> n_before 5, n_after 9,
# purge window [t_0 - 3, t_0 + 7], embargo E = max(6, 3) = 6.
SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, latency_buckets=1,
                   arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05")


def _day(date: str, n_buckets: int, start_us: int, price: int = P0) -> DayTape:
    tape = slice_from_rows([], [(DEEP, price)] * n_buckets)
    return DayTape(date=date, start_us=start_us, width_us=W, tape=tape, source=f"rows {date}")


def _four_days() -> list[DayTape]:
    """Three contiguous 50-bucket days, then a day after a one-day gap."""
    return [
        _day(DATES[0], 50, 0),
        _day(DATES[1], 50, 50 * W),
        _day(DATES[2], 50, 100 * W),
        _day(DATES[3], 50, 200 * W),
    ]


def test_embargo_and_purge_window_follow_the_accepted_formulas() -> None:
    assert embargo_buckets(SPEC) == 6
    assert embargo_buckets(EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=2,
                                       arrival_lookback=60)) == 60
    assert purge_window(20, SPEC) == (17, 27)
    default = EpisodeSpec(side=1, target_qty_ticks=1)
    assert embargo_buckets(default) == 300 and purge_window(1_000, default) == (940, 1_301)


def test_hand_built_boundaries_give_the_expected_segments_and_ranges() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                   (DATES[3],)), SPEC)
    record = splits.record
    assert record.embargo_buckets == 6 and (record.n_before, record.n_after) == (5, 9)
    assert (record.purge_before, record.purge_after) == (3, 7)
    assert [s.role for s in record.segments] == ["train", "validation", "test"]
    train, validation, test = record.segments
    # Two contiguous train days are one 100-bucket segment with no embargo at the start.
    assert train.dates == DATES[:2] and train.n_buckets == 100 and train.embargo_before == 0
    assert (train.first_start, train.end_start) == (5, 92)
    # Validation follows a different role: its first 6 buckets are embargoed.
    assert validation.embargo_before == 6
    assert (validation.first_start, validation.end_start) == (11, 42)
    # The test day is separated by a calendar gap and still embargoed (role change).
    assert test.embargo_before == 6 and (test.first_start, test.end_start) == (11, 42)
    assert record.n_starts("train") == 87 and record.n_starts("validation") == 31
    assert record.n_starts("test") == 31
    assert len(splits.tapes) == 3 and splits.tapes[0].n_buckets == 100


def test_embargo_width_separates_the_last_episode_before_from_the_first_after() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                   (DATES[3],)), SPEC)
    train, validation, _ = splits.record.segments
    last_train_start = train.end_start - 1
    # The slice (9 after) is longer than the purge window (7 after) here, so the slice is
    # what touches the boundary; the purge window ends two buckets short of it.
    assert last_train_start + splits.record.n_after == train.n_buckets
    assert purge_window(last_train_start, SPEC)[1] == train.n_buckets - 2
    first_validation_start = validation.first_start
    # The whole slice of the first validation episode begins E buckets into its segment.
    assert first_validation_start - splits.record.n_before == embargo_buckets(SPEC)
    assert purge_window(first_validation_start, SPEC)[0] >= embargo_buckets(SPEC)


def test_boundary_episodes_are_purged() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                   (DATES[3],)), SPEC)
    for segment in splits.record.segments:
        lo, hi = segment.first_start, segment.end_start
        for start in range(lo, hi):
            a, b = purge_window(start, SPEC)
            assert segment.embargo_before <= a and b < segment.n_buckets
            assert start - splits.record.n_before >= segment.embargo_before
            assert start + splits.record.n_after <= segment.n_buckets
        # One bucket outside the range on either side breaks a rule.
        purge_crosses = purge_window(hi, SPEC)[1] >= segment.n_buckets
        assert purge_crosses or hi + splits.record.n_after > segment.n_buckets
        assert lo - 1 - splits.record.n_before < segment.embargo_before


def test_no_start_appears_in_two_roles_and_positions_enumerate_every_start() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0],), (DATES[1],), (DATES[2], DATES[3])),
                          SPEC)
    seen: dict[tuple[int, int], str] = {}
    for role in ROLES:
        for position in range(splits.record.n_starts(role)):
            key = splits.record.start_at(role, position)
            assert key not in seen
            seen[key] = role
    assert len(seen) == sum(splits.record.n_starts(r) for r in ROLES)
    # Test role spans two segments (the gap splits it); positions run through both in order.
    test_segments = splits.record.segments_for("test")
    assert [s.index for s in test_segments] == [2, 3]
    assert splits.record.start_at("test", 0) == (2, test_segments[0].first_start)
    assert splits.record.start_at("test", test_segments[0].n_starts) == (
        3, test_segments[1].first_start
    )
    with pytest.raises(IndexError, match="starts of role"):
        splits.record.start_at("test", splits.record.n_starts("test"))
    with pytest.raises(IndexError, match="negative"):
        splits.record.start_at("train", -1)


def test_test_period_is_empty_when_not_provided() -> None:
    splits = build_splits(_four_days()[:3], SplitDates((DATES[0], DATES[1]), (DATES[2],)), SPEC)
    assert splits.record.n_starts("test") == 0 and splits.record.segments_for("test") == ()
    assert splits.record.dates_for("test") == ()
    with pytest.raises(IndexError):
        splits.record.start_at("test", 0)


def test_a_segment_too_short_for_an_episode_has_no_starts() -> None:
    days = [_day(DATES[0], 50, 0), _day(DATES[1], 15, 50 * W)]  # 15 < E + n_before + ...
    splits = build_splits(days, SplitDates((DATES[0],), (DATES[1],)), SPEC)
    short = splits.record.segments[1]
    assert short.n_starts == 0 and short.first_start == short.end_start == 11
    assert splits.record.n_starts("validation") == 0


def test_same_role_days_separated_by_a_gap_form_two_segments_without_embargo() -> None:
    days = [_day(DATES[0], 50, 0), _day(DATES[2], 50, 100 * W), _day(DATES[3], 50, 200 * W)]
    splits = build_splits(days, SplitDates((DATES[0], DATES[2]), (DATES[3],)), SPEC)
    first, second, validation = splits.record.segments
    assert first.role == second.role == "train" and second.embargo_before == 0
    assert validation.embargo_before == 6


def test_split_dates_validation() -> None:
    with pytest.raises(ValueError, match="listed under train and validation"):
        SplitDates((DATES[0],), (DATES[0],))
    with pytest.raises(ValueError, match="listed under"):
        SplitDates((DATES[0], DATES[0]), ())
    dates = SplitDates([DATES[0]], [DATES[1]])  # lists are accepted and frozen as tuples
    assert dates.train == (DATES[0],) and dates.role_of(DATES[1]) == "validation"
    assert dates.role_of("1999-01-01") is None


def test_build_splits_rejects_bad_assignments() -> None:
    days = _four_days()
    with pytest.raises(ValueError, match="at least one day"):
        build_splits([], SplitDates((), ()), SPEC)
    with pytest.raises(ValueError, match="not assigned to any role"):
        build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],)), SPEC)
    with pytest.raises(ValueError, match="has no day tape"):
        build_splits(days[:3], SplitDates((DATES[0], DATES[1]), (DATES[2],), ("2030-01-01",)), SPEC)
    with pytest.raises(ValueError, match="must be chronological"):
        build_splits(days, SplitDates((DATES[0], DATES[2]), (DATES[1],), (DATES[3],)), SPEC)
    with pytest.raises(ValueError, match="must be chronological"):
        build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[3],), (DATES[2],)), SPEC)
    overlapping = [_day(DATES[0], 50, 0), _day(DATES[1], 50, 20 * W)]
    with pytest.raises(ValueError, match="overlap"):
        build_splits(overlapping, SplitDates((DATES[0],), (DATES[1],)), SPEC)
    mislabelled = [_day(DATES[1], 50, 0), _day(DATES[0], 50, 50 * W)]
    with pytest.raises(ValueError, match="calendar order"):
        build_splits(mislabelled, SplitDates((DATES[1],), (DATES[0],)), SPEC)
    with pytest.raises(ValueError, match="unknown role"):
        build_splits(days[:1], SplitDates((DATES[0],), ()), SPEC).record.n_starts("holdout")


def test_record_is_json_serialisable_and_round_trips_its_numbers() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                   (DATES[3],)), SPEC)
    text = json.dumps(splits.record.to_json(), sort_keys=True)
    loaded = json.loads(text)
    assert loaded["embargo_buckets"] == 6 and len(loaded["segments"]) == 3
    assert loaded["days"][0] == {
        "date": DATES[0], "role": "train", "start_us": 0, "width_us": W, "n_buckets": 50,
        "source": f"rows {DATES[0]}",
    }
    assert loaded["segments"][1]["first_start"] == 11
    assert json.dumps(build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                           (DATES[3],)), SPEC).record.to_json(),
                      sort_keys=True) == text


def test_segment_date_at_maps_local_buckets_to_days() -> None:
    splits = build_splits(_four_days(), SplitDates((DATES[0], DATES[1]), (DATES[2],),
                                                   (DATES[3],)), SPEC)
    train = splits.record.segments[0]
    assert train.date_at(0) == DATES[0] and train.date_at(49) == DATES[0]
    assert train.date_at(50) == DATES[1] and train.date_at(99) == DATES[1]
    with pytest.raises(IndexError):
        train.date_at(100)


def test_episode_from_splits_is_cut_inside_its_segment() -> None:
    days = [_day(DATES[0], 50, 0, price=P0), _day(DATES[1], 50, 50 * W, price=P0 + 100),
            _day(DATES[2], 50, 100 * W, price=P0 + 200)]
    splits = build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],)), SPEC)
    spec, market = splits.episode("train", 0, -1, 500, SPEC)
    assert spec.side == -1 and market.origin == 5 and market.n_after == 9
    assert int(market.vwap_ticks[market.index(-1)]) == P0
    spec, market = splits.episode("train", splits.record.n_starts("train") - 1, 1, 200, SPEC)
    assert set(market.vwap_ticks.tolist()) == {P0 + 100}   # entirely inside day 2
    spec, market = splits.episode("validation", 0, 1, 1_000, SPEC)
    assert set(market.vwap_ticks.tolist()) == {P0 + 200}
    with pytest.raises(ValueError, match="geometry"):
        splits.episode("train", 0, 1, 500, EpisodeSpec(side=1, target_qty_ticks=1))
    # The record and tapes are what build_splits returned; Splits adds no state.
    assert isinstance(splits, Splits) and isinstance(splits.record, SplitRecord)


# --- Hypothesis: random day structures keep the invariants -------------------------


@st.composite
def structures(draw: st.DrawFn) -> tuple[list[DayTape], SplitDates, EpisodeSpec]:
    n_days = draw(st.integers(1, 5))
    lengths = draw(st.lists(st.integers(1, 70), min_size=n_days, max_size=n_days))
    gaps = draw(st.lists(st.booleans(), min_size=n_days, max_size=n_days))
    n_train = draw(st.integers(0, n_days))
    n_validation = draw(st.integers(0, n_days - n_train))
    roles = ["train"] * n_train + ["validation"] * n_validation
    roles += ["test"] * (n_days - len(roles))
    spec = EpisodeSpec(
        side=1, target_qty_ticks=1,
        horizon_buckets=draw(st.integers(1, 8)),
        latency_buckets=draw(st.integers(0, 1)),
        data_delay_buckets=draw(st.integers(0, 1)),
        arrival_lookback=draw(st.integers(1, 5)),
    )
    days, start = [], 0
    for i, (n, gap) in enumerate(zip(lengths, gaps, strict=True)):
        start += 3 * W if gap else 0
        days.append(_day(f"2024-02-{i + 1:02d}", n, start))
        start += n * W
    dates = SplitDates(
        tuple(d.date for d, r in zip(days, roles, strict=True) if r == "train"),
        tuple(d.date for d, r in zip(days, roles, strict=True) if r == "validation"),
        tuple(d.date for d, r in zip(days, roles, strict=True) if r == "test"),
    )
    return days, dates, spec


def _rules_hold(segment: Segment, start: int, record: SplitRecord, spec: EpisodeSpec) -> bool:
    a, b = purge_window(start, spec)
    data = segment.embargo_before <= start - record.n_before and (
        start + record.n_after <= segment.n_buckets
    )
    purge = segment.embargo_before <= a and b <= segment.n_buckets - 1
    return data and purge


@given(structures())
@settings(max_examples=300, deadline=None)
def test_split_invariants_hold_for_random_structures(
    structure: tuple[list[DayTape], SplitDates, EpisodeSpec]
) -> None:
    days, dates, spec = structure
    splits = build_splits(days, dates, spec)
    record = splits.record
    assert record.embargo_buckets == embargo_buckets(spec)
    assert (record.n_before, record.n_after) == required_window(spec)
    assert [d.date for d in record.days] == [d.date for d in days]

    # Segments partition the days in order; neighbours differ in role or are not contiguous.
    assert [d for s in record.segments for d in s.dates] == [d.date for d in days]
    by_date = {d.date: d for d in days}
    for earlier, later in zip(record.segments, record.segments[1:], strict=False):
        boundary_is_role_change = earlier.role != later.role
        contiguous = by_date[earlier.dates[-1]].is_contiguous_with(by_date[later.dates[0]])
        assert boundary_is_role_change or not contiguous
        assert later.embargo_before == (record.embargo_buckets if boundary_is_role_change else 0)
    assert record.segments[0].embargo_before == 0

    seen: set[tuple[int, int]] = set()
    for role in ROLES:
        for position in range(record.n_starts(role)):
            key = splits.record.start_at(role, position)
            assert key not in seen
            seen.add(key)
            segment = record.segments[key[0]]
            assert segment.role == role
            assert _rules_hold(segment, key[1], record, spec)
    for segment in record.segments:
        assert 0 <= segment.first_start <= segment.end_start
        if segment.n_starts:
            assert not _rules_hold(segment, segment.first_start - 1, record, spec)
            assert not _rules_hold(segment, segment.end_start, record, spec)
            # The slice cut at either end of the range is complete, never clipped.
            for start in (segment.first_start, segment.end_start - 1):
                _, market = splits.episode(segment.role, start - segment.first_start + sum(
                    s.n_starts for s in record.segments_for(segment.role) if s.index < segment.index
                ), 1, 500, spec)
                assert market.origin == record.n_before and market.n_after == record.n_after
    assert len(seen) == sum(record.n_starts(r) for r in ROLES)
