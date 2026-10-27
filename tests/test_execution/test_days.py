"""Tests for tapeforge.execution.days: UTC day geometry, the synthetic, trade-array and archive
adapters, contiguity and concatenation, slice cutting, and episodes cut from a day tape.

Synthetic days are checked bucket for bucket against the reference feature
engine; the archive adapter runs on the five-record fixture under ``tests/fixtures``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tapeforge import contract, generate, reference, windows
from tapeforge.execution import EpisodeSpec, make_episode, reset
from tapeforge.execution.days import (
    BUCKETS_PER_DAY,
    DayTape,
    concat_tapes,
    cut_slice,
    day_from_archive,
    day_from_trades,
    episode_at,
    synthetic_day,
    utc_day_start_us,
)
from tapeforge.execution.state import MarketSlice
from test_execution.tapes import DEEP, P0, slice_from_rows

W = contract.DEFAULT_BUCKET_WIDTH_US
FIXTURE_ZIP = Path(__file__).parent.parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"


def test_utc_day_start_is_midnight_in_microseconds() -> None:
    assert utc_day_start_us("1970-01-01") == 0
    assert utc_day_start_us("2023-12-01") == 1_701_388_800 * contract.US_PER_SECOND
    assert utc_day_start_us("2025-01-02") == 1_735_776_000 * contract.US_PER_SECOND
    with pytest.raises(ValueError):
        utc_day_start_us("2023/12/01")


def test_synthetic_day_is_deterministic_and_labelled() -> None:
    a = synthetic_day(seed=3, day="2024-01-01", n_buckets=20)
    b = synthetic_day(seed=3, day="2024-01-01", n_buckets=20)
    assert a.n_buckets == 20 and a.start_us == utc_day_start_us("2024-01-01")
    assert a.end_us == a.start_us + 20 * W
    np.testing.assert_array_equal(a.tape.total_quantity, b.tape.total_quantity)
    np.testing.assert_array_equal(a.tape.vwap_ticks, b.tape.vwap_ticks)
    assert "seed=3" in a.source
    other = synthetic_day(seed=4, day="2024-01-01", n_buckets=20)
    assert not np.array_equal(a.tape.total_quantity, other.tape.total_quantity)


def test_synthetic_day_matches_the_reference_features_bucket_for_bucket() -> None:
    day = synthetic_day(seed=11, day="2024-01-01", n_buckets=30, profile="burst", start_us=0)
    trades = generate.generate_trades(seed=11, n_buckets=30, profile="burst", start_us=0)
    offsets = windows.build_offsets(trades["timestamp"], 0, 30, W)
    features = reference.aggregate_reference(trades, offsets)
    np.testing.assert_array_equal(day.tape.event_count, features.event_count)
    np.testing.assert_array_equal(day.tape.total_quantity, features.total_quantity)
    np.testing.assert_array_equal(day.tape.signed_quantity, features.signed_quantity)
    assert day.tape.origin == 0


def test_day_tape_validates_label_and_geometry() -> None:
    tape = slice_from_rows([], [(DEEP, P0)] * 3)
    with pytest.raises(ValueError):
        DayTape(date="not-a-date", start_us=0, width_us=W, tape=tape)
    with pytest.raises(ValueError, match="width_us"):
        DayTape(date="2024-01-01", start_us=0, width_us=0, tape=tape)
    with pytest.raises(ValueError, match="start_us"):
        DayTape(date="2024-01-01", start_us=-1, width_us=W, tape=tape)
    with pytest.raises(ValueError, match="origin 0"):
        DayTape(date="2024-01-01", start_us=0, width_us=W, tape=slice_from_rows([(DEEP, P0)], []))


def test_contiguity_is_end_equals_next_start_at_the_same_width() -> None:
    first = synthetic_day(1, "2024-01-01", 10, start_us=0)
    second = synthetic_day(2, "2024-01-02", 10, start_us=10 * W)
    gap = synthetic_day(3, "2024-01-03", 10, start_us=25 * W)
    assert first.is_contiguous_with(second)
    assert not second.is_contiguous_with(gap)
    assert not first.is_contiguous_with(first)


def test_concat_tapes_joins_contiguous_days_in_order() -> None:
    first = synthetic_day(1, "2024-01-01", 10, start_us=0)
    second = synthetic_day(2, "2024-01-02", 7, start_us=10 * W)
    joined = concat_tapes([first, second])
    assert joined.n_buckets == 17 and joined.origin == 0
    np.testing.assert_array_equal(joined.total_quantity[:10], first.tape.total_quantity)
    np.testing.assert_array_equal(joined.vwap_ticks[10:], second.tape.vwap_ticks)
    assert concat_tapes([first]) is first.tape
    with pytest.raises(ValueError, match="only contiguous days"):
        concat_tapes([second, first])
    with pytest.raises(ValueError, match="at least one day"):
        concat_tapes([])


def test_cut_slice_reindexes_relative_to_the_new_origin() -> None:
    tape = slice_from_rows([], [(DEEP, P0 + i) for i in range(8)])
    cut = cut_slice(tape, 2, 7, origin=3)
    assert cut.n_buckets == 5 and cut.origin == 3 and cut.n_after == 2
    assert int(cut.vwap_ticks[cut.index(0)]) == P0 + 5   # Tape bucket 5 is episode bucket 0.
    assert int(cut.vwap_ticks[cut.index(-3)]) == P0 + 2
    for first, last in [(3, 3), (-1, 4), (0, 9), (5, 2)]:
        with pytest.raises(ValueError, match="not inside"):
            cut_slice(tape, first, last, 0)


def test_episode_at_agrees_with_make_episode_on_the_same_trades() -> None:
    trades = generate.generate_trades(seed=5, n_buckets=40, profile="uniform", start_us=0)
    offsets = windows.build_offsets(trades["timestamp"], 0, 40, W)
    day = synthetic_day(seed=5, day="2024-01-01", n_buckets=40, start_us=0)
    template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=2)
    for start in (5, 10, 36):
        spec_a, market_a = make_episode(trades, offsets, start, -1, 500, template=template)
        spec_b, market_b = episode_at(day.tape, start, -1, 500, template=template)
        assert spec_a == spec_b
        assert market_a.origin == market_b.origin and market_a.n_after == market_b.n_after
        np.testing.assert_array_equal(market_a.total_quantity, market_b.total_quantity)
        np.testing.assert_array_equal(market_a.vwap_ticks, market_b.vwap_ticks)
        np.testing.assert_array_equal(market_a.signed_quantity, market_b.signed_quantity)
    assert reset(spec_b, market_b).arrival_price_ticks > 0


def test_episode_at_uses_the_accepted_defaults_without_a_template() -> None:
    tape = slice_from_rows([], [(DEEP, P0)] * 700)
    spec, market = episode_at(tape, 300, 1, 200)
    assert spec.horizon_buckets == 300 and market.origin == 300 and market.n_after == 360
    assert spec.target_qty_ticks == 300 * DEEP * 200 // 10_000


def test_episode_at_rejects_starts_without_history_or_past_the_tape() -> None:
    tape = slice_from_rows([], [(DEEP, P0)] * 12)
    template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=2)
    with pytest.raises(ValueError, match="buckets of history"):
        episode_at(tape, 4, 1, 500, template=template)
    with pytest.raises(ValueError, match="past the 12-bucket tape"):
        episode_at(tape, 12, 1, 500, template=template)
    _, clipped = episode_at(tape, 10, 1, 500, template=template)
    assert clipped.n_after == 2   # The slice ends with the tape, so the episode truncates.


# --- whole-day adapters -------------------------------------------------------------


def _day_trades(day: str, seconds: tuple[int, ...]) -> np.ndarray:
    """Build ``TRADE_DTYPE`` records at the given bucket offsets into UTC day ``day``.

    Every record carries the same price and quantity; only the timestamps vary.
    """
    trades = np.zeros(len(seconds), dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = utc_day_start_us(day) + np.array(seconds, dtype=np.int64) * W
    trades["price"] = 10_000
    trades["quantity"] = 100
    trades["agg_trade_id"] = np.arange(len(seconds))
    return trades


def test_day_from_trades_buckets_a_full_utc_day() -> None:
    trades = _day_trades("2024-01-01", (0, 0, 1, 86_399))
    day = day_from_trades(trades, "2024-01-01", source="hand-built")
    assert day.n_buckets == BUCKETS_PER_DAY == 86_400
    assert day.start_us == utc_day_start_us("2024-01-01") and day.source == "hand-built"
    assert day.tape.event_count[0] == 2 and day.tape.event_count[1] == 1
    assert day.tape.event_count[86_399] == 1 and int(day.tape.event_count.sum()) == 4
    assert bool(day.tape.valid[2]) is False


def test_day_from_trades_rejects_records_outside_the_day() -> None:
    before = _day_trades("2024-01-01", (0, 1))
    before["timestamp"][0] -= 1
    with pytest.raises(ValueError, match="1 of 2 records fall outside UTC day 2024-01-01"):
        day_from_trades(before, "2024-01-01")
    after = _day_trades("2024-01-01", (0, 86_400))
    with pytest.raises(ValueError, match="outside"):
        day_from_trades(after, "2024-01-01")


def test_day_from_trades_rejects_a_width_that_does_not_divide_the_day() -> None:
    with pytest.raises(ValueError, match="must divide"):
        day_from_trades(_day_trades("2024-01-01", (0,)), "2024-01-01", width_us=7_000_000)
    with pytest.raises(ValueError, match="TRADE_DTYPE"):
        day_from_trades(np.zeros(1, dtype=np.int64), "2024-01-01")


def test_day_from_archive_reads_the_date_from_the_file_name() -> None:
    day, provenance = day_from_archive(FIXTURE_ZIP)
    assert day.date == "2025-01-02" and provenance.n_records == 5
    assert int(day.tape.event_count.sum()) == 5
    assert provenance.sha256 in day.source and FIXTURE_ZIP.name in day.source


def test_day_from_archive_needs_a_date_for_renamed_files(tmp_path: Path) -> None:
    renamed = tmp_path / "whatever.zip"
    renamed.write_bytes(FIXTURE_ZIP.read_bytes())
    with pytest.raises(ValueError, match="pass day="):
        day_from_archive(renamed)
    day, _ = day_from_archive(renamed, day="2025-01-02")
    assert day.date == "2025-01-02"
    with pytest.raises(ValueError, match="outside UTC day 2025-01-03"):
        day_from_archive(renamed, day="2025-01-03")


def test_market_slice_type_is_preserved_through_the_adapters() -> None:
    day = synthetic_day(seed=1, day="2024-01-01", n_buckets=5, start_us=0)
    assert isinstance(day.tape, MarketSlice)
    assert not day.tape.total_quantity.flags.writeable
