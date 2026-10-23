"""Day tapes: one UTC day of bucketed features, and episodes cut from a tape.

A ``DayTape`` is the feature engine's output for one calendar day as a
``MarketSlice`` with ``origin == 0`` (array index == bucket index), plus the
calendar label and bucket geometry the split logic needs: ``date`` (the UTC
day, ``YYYY-MM-DD``), ``start_us`` (the timestamp of bucket 0) and
``width_us``. Two days are *contiguous* when one ends where the next starts;
``concat_tapes`` joins contiguous days into one longer tape so an episode
may run across midnight inside a split.

Adapters:

* ``synthetic_day``: a ``generate.generate_trades`` day, so splits and
  baselines are testable without data; the number of buckets is a parameter
  and the geometry is whatever the caller asks for.
* ``day_from_trades``: a full UTC day (86,400 one-second buckets by default)
  from any ``TRADE_DTYPE`` array whose records fall inside that day.
* ``day_from_archive``: the above for a downloaded aggTrades archive, through
  ``tapeforge.ingest.load_agg_trades``; every record must belong to the day
  the file name claims.

``episode_at`` is ``episodes.make_episode`` for a tape that is already
bucketed: it cuts the slice ``required_window`` demands around ``t_0`` and
sizes Q* from the reference volume, with no recomputation of features.

Units follow the data contract: int64 ticks everywhere, VWAP quantised by
``vwap_ticks_from_numerators``. This module reads nothing after the end of a
tape and does not know about splits, sampling or policies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np

from tapeforge import generate, ingest, reference, windows
from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US, TRADE_DTYPE, US_PER_SECOND
from tapeforge.execution.episodes import (
    reference_volume_ticks,
    required_window,
    target_from_reference_volume,
)
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice

__all__ = [
    "BUCKETS_PER_DAY",
    "DayTape",
    "concat_tapes",
    "cut_slice",
    "day_from_archive",
    "day_from_trades",
    "episode_at",
    "synthetic_day",
    "utc_day_start_us",
]

SECONDS_PER_DAY: Final[int] = 86_400
BUCKETS_PER_DAY: Final[int] = SECONDS_PER_DAY  # at the default one-second width


def utc_day_start_us(day: str) -> int:
    """Microseconds since the epoch at 00:00:00 UTC of ``day`` (``YYYY-MM-DD``).

    Raises:
        ValueError: ``day`` is not an ISO calendar date.
    """
    parsed = date.fromisoformat(day)
    midnight = datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
    return int(midnight.timestamp()) * US_PER_SECOND


@dataclass(frozen=True, eq=False)
class DayTape:
    """One day's bucket features with their calendar position.

    ``tape.origin`` must be 0 so that bucket ``b`` is array index ``b``.
    ``source`` is a short provenance string carried into run records (the
    generator seed and profile, or the archive name and digest).
    """

    date: str
    start_us: int
    width_us: int
    tape: MarketSlice
    source: str = ""

    def __post_init__(self) -> None:
        """Validate the calendar label and geometry.

        Raises:
            ValueError: ``date`` is not ``YYYY-MM-DD``, ``width_us`` is not
                positive, ``start_us`` is negative, or the slice does not have
                ``origin == 0``.
        """
        date.fromisoformat(self.date)
        if self.width_us <= 0:
            raise ValueError(f"width_us must be positive, got {self.width_us}")
        if self.start_us < 0:
            raise ValueError(f"start_us must be non-negative, got {self.start_us}")
        if self.tape.origin != 0:
            raise ValueError(f"a day tape's slice must have origin 0, got {self.tape.origin}")

    @property
    def n_buckets(self) -> int:
        """Buckets in the day (86,400 for a full day at the default width)."""
        return self.tape.n_buckets

    @property
    def end_us(self) -> int:
        """Timestamp just past the last bucket (the next day's ``start_us`` when contiguous)."""
        return self.start_us + self.n_buckets * self.width_us

    def is_contiguous_with(self, following: DayTape) -> bool:
        """True when ``following`` starts exactly where this day ends, at the same width."""
        return self.width_us == following.width_us and self.end_us == following.start_us


def _tape_from_trades(trades: np.ndarray, offsets: np.ndarray) -> MarketSlice:
    features = reference.aggregate_reference(trades, offsets)
    numerators = reference.vwap_numerators(trades, offsets)
    return MarketSlice.from_features(features, numerators, origin=0)


def synthetic_day(
    seed: int,
    day: str,
    n_buckets: int,
    profile: generate.Profile = "uniform",
    start_us: int | None = None,
    width_us: int = DEFAULT_BUCKET_WIDTH_US,
) -> DayTape:
    """A seeded synthetic day: ``generate.generate_trades`` bucketed into a tape.

    ``start_us`` defaults to midnight UTC of ``day``; pass it explicitly to
    build short contiguous test days. Deterministic in ``seed`` and the other
    arguments.

    Raises:
        ValueError: whatever ``generate_trades`` rejects, or a bad ``day``.
    """
    start = utc_day_start_us(day) if start_us is None else start_us
    trades = generate.generate_trades(
        seed=seed, n_buckets=n_buckets, profile=profile, start_us=start, width_us=width_us
    )
    offsets = windows.build_offsets(trades["timestamp"], start, n_buckets, width_us)
    return DayTape(
        date=day,
        start_us=start,
        width_us=width_us,
        tape=_tape_from_trades(trades, offsets),
        source=f"synthetic seed={seed} profile={profile} n_buckets={n_buckets}",
    )


def day_from_trades(
    trades: np.ndarray,
    day: str,
    width_us: int = DEFAULT_BUCKET_WIDTH_US,
    source: str = "",
) -> DayTape:
    """Bucket a timestamp-sorted ``TRADE_DTYPE`` array over the whole UTC day ``day``.

    The day is ``[midnight, midnight + 86,400 s)`` cut into ``width_us``
    buckets (``width_us`` must divide the day). Every record must lie inside
    the day: a record outside it means the array is not the day it claims to
    be, and silently ignoring it would make the tape disagree with its label.

    Raises:
        ValueError: wrong dtype, ``width_us`` does not divide a day, or any
            record outside the day (the message counts them).
    """
    if trades.dtype != TRADE_DTYPE:
        raise ValueError(f"trades must have TRADE_DTYPE, got {trades.dtype}")
    day_us = SECONDS_PER_DAY * US_PER_SECOND
    if width_us <= 0 or day_us % width_us:
        raise ValueError(f"width_us {width_us} must divide the {day_us} us day")
    start = utc_day_start_us(day)
    timestamps = trades["timestamp"]
    outside = int(np.count_nonzero((timestamps < start) | (timestamps >= start + day_us)))
    if outside:
        raise ValueError(
            f"{outside} of {trades.size} records fall outside UTC day {day}; "
            "the array is not that day"
        )
    n_buckets = day_us // width_us
    offsets = windows.build_offsets(trades["timestamp"], start, n_buckets, width_us)
    return DayTape(
        date=day, start_us=start, width_us=width_us, tape=_tape_from_trades(trades, offsets),
        source=source,
    )


def day_from_archive(
    path: str | Path, day: str | None = None, width_us: int = DEFAULT_BUCKET_WIDTH_US
) -> tuple[DayTape, ingest.Provenance]:
    """Load a downloaded aggTrades archive and bucket it over its UTC day.

    ``day`` defaults to the date in the archive's file name
    (``SYMBOL-aggTrades-YYYY-MM-DD.zip``); pass it when the file is renamed.
    Returns the tape and the ingest provenance (sha256, record count,
    timestamp unit, resort flag). The futures and spot archives share the
    7-column layout, so the trade ingest handles both.

    Raises:
        ValueError: no date given and none in the file name, plus whatever
            ``load_agg_trades`` and ``day_from_trades`` reject.
    """
    trades, provenance = ingest.load_agg_trades(path)
    if day is None:
        if provenance.date == "unknown":
            raise ValueError(
                f"{Path(path).name} does not follow SYMBOL-aggTrades-YYYY-MM-DD; pass day="
            )
        day = provenance.date
    source = f"aggTrades {Path(path).name} sha256={provenance.sha256}"
    return day_from_trades(trades, day, width_us, source=source), provenance


# --- slicing ------------------------------------------------------------------


def cut_slice(tape: MarketSlice, first: int, last: int, origin: int) -> MarketSlice:
    """Buckets ``[first, last)`` of a tape as a new slice with the given ``origin``.

    ``tape`` is indexed by array position (an ``origin == 0`` tape, or a
    concatenation of days). The copy is validated like any slice.

    Raises:
        ValueError: ``[first, last)`` is empty or not inside the tape.
    """
    if not 0 <= first < last <= tape.n_buckets:
        raise ValueError(
            f"bucket range [{first}, {last}) is not inside the {tape.n_buckets}-bucket tape"
        )
    return MarketSlice(
        event_count=tape.event_count[first:last],
        total_quantity=tape.total_quantity[first:last],
        signed_quantity=tape.signed_quantity[first:last],
        vwap_ticks=tape.vwap_ticks[first:last],
        valid=tape.valid[first:last],
        origin=origin,
    )


def concat_tapes(days: tuple[DayTape, ...] | list[DayTape]) -> MarketSlice:
    """Join contiguous days, in order, into one ``origin == 0`` slice.

    Raises:
        ValueError: no days, or two neighbours that are not contiguous
            (the message names the dates).
    """
    if not days:
        raise ValueError("concat_tapes needs at least one day")
    for earlier, later in zip(days, days[1:], strict=False):
        if not earlier.is_contiguous_with(later):
            raise ValueError(
                f"day {earlier.date} ends at {earlier.end_us} us but {later.date} starts at "
                f"{later.start_us} us (width {later.width_us}); only contiguous days concatenate"
            )
    if len(days) == 1:
        return days[0].tape
    return MarketSlice(
        event_count=np.concatenate([d.tape.event_count for d in days]),
        total_quantity=np.concatenate([d.tape.total_quantity for d in days]),
        signed_quantity=np.concatenate([d.tape.signed_quantity for d in days]),
        vwap_ticks=np.concatenate([d.tape.vwap_ticks for d in days]),
        valid=np.concatenate([d.tape.valid for d in days]),
        origin=0,
    )


def episode_at(
    tape: MarketSlice,
    start_bucket: int,
    side: int,
    rho_bps: int,
    template: EpisodeSpec | None = None,
) -> tuple[EpisodeSpec, MarketSlice]:
    """Cut the episode whose first decision bucket is ``start_bucket`` of ``tape``.

    Same contract as ``episodes.make_episode`` (``template`` supplies every
    field except side and target; Q* from the H buckets before the start),
    but on an already bucketed tape: the slice is ``required_window`` buckets
    around the start, clipped at the end of the tape (a clipped slice makes
    the episode truncate, not fail).

    Raises:
        ValueError: too little history before the start, or a start past the tape.
    """
    base = EpisodeSpec(side=side, target_qty_ticks=1) if template is None else replace(
        template, side=side, target_qty_ticks=1
    )
    n_before, n_after = required_window(base)
    if start_bucket < n_before:
        raise ValueError(
            f"start bucket {start_bucket} has only {start_bucket} buckets of history, "
            f"{n_before} are needed"
        )
    if start_bucket >= tape.n_buckets:
        raise ValueError(f"start bucket {start_bucket} is past the {tape.n_buckets}-bucket tape")
    last = min(start_bucket + n_after, tape.n_buckets)
    market = cut_slice(tape, start_bucket - n_before, last, origin=n_before)
    v_ref = reference_volume_ticks(market, base.horizon_buckets)
    target = target_from_reference_volume(v_ref, rho_bps, base.participation_cap_bps)
    return replace(base, target_qty_ticks=target), market
