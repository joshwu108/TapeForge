"""The downloaded futures day (BTCUSDT USD-M, 2023-12-01) through the day tape and the baselines.

Skips with a precise reason when the archive is absent, so a fresh checkout
passes without data. Nothing here is a performance claim; the figures are
properties of the day that the loader must reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tapeforge.execution.baselines import BASELINES
from tapeforge.execution.days import BUCKETS_PER_DAY, DayTape, day_from_archive
from tapeforge.execution.runner import run_policy, write_run_record
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import SplitDates, Splits, build_splits

REPO = Path(__file__).resolve().parent.parent.parent
FUTURES_DAY = REPO / "data" / "futures-um" / "aggTrades" / "BTCUSDT-aggTrades-2023-12-01.zip"
DATE = "2023-12-01"
TEMPLATE = EpisodeSpec(side=1, target_qty_ticks=1)   # the accepted defaults: H 300, L 60


def _stable(record: dict) -> dict:
    """The record without the two fields that differ between otherwise identical runs."""
    return {k: v for k, v in record.items() if k not in ("created_utc", "code_revision")}


@pytest.fixture(scope="module")
def futures_day() -> DayTape:
    if not FUTURES_DAY.exists():
        pytest.skip(f"{FUTURES_DAY.relative_to(REPO)} is not downloaded (step 01 data)")
    day, provenance = day_from_archive(FUTURES_DAY)
    assert provenance.date == DATE and provenance.timestamp_unit == "ms"
    return day


@pytest.fixture(scope="module")
def single_day_splits(futures_day: DayTape) -> Splits:
    return build_splits([futures_day], SplitDates((DATE,), ()), TEMPLATE)


def test_the_day_loads_into_86400_one_second_buckets(futures_day: DayTape) -> None:
    assert futures_day.date == DATE and futures_day.n_buckets == BUCKETS_PER_DAY
    assert int(futures_day.tape.event_count.sum()) == 1_038_461   # every record is in the day
    assert int((~futures_day.tape.valid).sum()) == 1_044           # empty seconds
    assert int(futures_day.tape.event_count.max()) == 2_918        # busiest second
    assert futures_day.tape.vwap_ticks[futures_day.tape.valid].min() >= 3_760_000  # 37,600.00


def test_a_single_train_day_has_the_expected_admissible_starts(single_day_splits: Splits) -> None:
    record = single_day_splits.record
    assert record.embargo_buckets == 300 and (record.n_before, record.n_after) == (300, 360)
    segment, = record.segments
    # No embargo at the start of the timeline; first start needs 300 buckets of history,
    # last start needs 360 buckets after it (the slice rule binds, not the purge window).
    assert (segment.first_start, segment.end_start) == (300, BUCKETS_PER_DAY - 359)
    assert record.n_starts("train") == 85_741 and record.n_starts("validation") == 0


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_run_reproducibly_on_the_real_day(
    single_day_splits: Splits, name: str, tmp_path: Path
) -> None:
    first, trajectories = run_policy(single_day_splits, "train", BASELINES[name], name, TEMPLATE,
                                     n_episodes=6, seed=2023)
    second, _ = run_policy(single_day_splits, "train", BASELINES[name], name, TEMPLATE,
                           n_episodes=6, seed=2023)
    assert _stable(first) == _stable(second)
    path = write_run_record(first, tmp_path)
    assert json.loads(path.read_text())["summary"]["n_episodes"] == 6
    for trajectory in trajectories:
        assert trajectory.last_result.terminated   # a full day never truncates inside the ranges
        assert trajectory.final_state.filled_qty_ticks <= trajectory.spec.target_qty_ticks
    summary = first["summary"]
    assert summary["n_truncated"] == 0 and 0.0 <= summary["completion_rate"] <= 1.0
