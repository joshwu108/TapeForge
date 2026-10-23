"""Chronological train / validation / test splits over whole UTC days, with leakage controls.

Days are assigned to a role by calendar date, explicitly (``SplitDates``;
there are no defaults and the test role may be empty). The order must be
chronological: every train day precedes every validation day, which
precedes every test day. Contiguous days of one role form a *segment*
(``days.concat_tapes``), inside which an episode may run across midnight;
a calendar gap or a change of role ends the segment.

Where an episode may start inside a segment of ``n`` buckets, for a spec
with horizon H, latency l, arrival lookback L and data delay d:

* **data**: the slice ``episodes.required_window`` needs,
  ``[t_0 - n_before, t_0 + n_after)``, lies inside the segment, so a
  split never reads another split's buckets, not even for V_ref;
* **purge**: the window ``[t_0 - L, t_0 + H + l + 1]`` (closed) lies inside
  the segment, so no episode's history, fills or terminal mark cross a
  boundary (the slice rule implies this for L >= 2; both are enforced);
* **embargo**: the first ``E = max(H + l, L)`` buckets of a segment that
  follows a segment of a *different* role are unusable, so the last
  episode before a boundary and the first after it are at least E buckets
  apart. The embargo sits in the later segment, like the embargo that
  follows a held-out period in purged cross-validation; two same-role
  segments separated by a gap have no embargo between them because
  nothing can be learned across a boundary where no episode is cut.

The admissible starts of a segment are therefore one half-open range
``[first_start, end_start)`` in segment-local buckets, possibly empty. The
``SplitRecord`` is a frozen, JSON-serialisable description of the days, the
segments, the ranges and the parameters that produced them; ``Splits`` pairs
it with the segment tapes so episodes can be cut. Sampling and policies live
elsewhere; nothing here is random.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from tapeforge.execution.days import DayTape, concat_tapes, episode_at
from tapeforge.execution.episodes import required_window
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice

__all__ = [
    "ROLES",
    "DayRecord",
    "Segment",
    "SplitDates",
    "SplitRecord",
    "Splits",
    "build_splits",
    "embargo_buckets",
    "purge_window",
]

ROLES: Final[tuple[str, ...]] = ("train", "validation", "test")


def embargo_buckets(spec: EpisodeSpec) -> int:
    """``max(H + l, L)``: the accepted embargo width in buckets."""
    return max(spec.horizon_buckets + spec.latency_buckets, spec.arrival_lookback)


def purge_window(start_bucket: int, spec: EpisodeSpec) -> tuple[int, int]:
    """Closed bucket interval ``[t_0 - L, t_0 + H + l + 1]`` that must not cross a boundary."""
    return (
        start_bucket - spec.arrival_lookback,
        start_bucket + spec.horizon_buckets + spec.latency_buckets + 1,
    )


@dataclass(frozen=True)
class SplitDates:
    """The explicit role of every day. Dates are ``YYYY-MM-DD``; ``test`` may be empty."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Raises ``ValueError`` on a date listed twice or in two roles."""
        for role in ROLES:
            object.__setattr__(self, role, tuple(getattr(self, role)))
        seen: dict[str, str] = {}
        for role in ROLES:
            for day in getattr(self, role):
                if day in seen:
                    raise ValueError(f"date {day} is listed under {seen[day]} and {role}")
                seen[day] = role

    def role_of(self, day: str) -> str | None:
        """The role a date is assigned to, or ``None`` when it is not listed."""
        for role in ROLES:
            if day in getattr(self, role):
                return role
        return None


@dataclass(frozen=True)
class DayRecord:
    """What the record keeps about one day (no features)."""

    date: str
    role: str
    start_us: int
    width_us: int
    n_buckets: int
    source: str


@dataclass(frozen=True)
class Segment:
    """A maximal run of contiguous same-role days and its admissible episode starts.

    Buckets are segment-local: bucket ``b`` of the segment is bucket
    ``b - sum(day_n_buckets[:k])`` of its ``k``-th day. ``embargo_before`` is
    ``E`` when a segment of another role precedes this one, else 0.
    ``[first_start, end_start)`` is the admissible range of ``t_0``;
    ``end_start == first_start`` means the segment is too short to hold an
    episode under the rules.
    """

    index: int
    role: str
    dates: tuple[str, ...]
    day_n_buckets: tuple[int, ...]
    n_buckets: int
    embargo_before: int
    first_start: int
    end_start: int

    @property
    def n_starts(self) -> int:
        """Admissible episode starts in this segment (0 when it is too short)."""
        return self.end_start - self.first_start

    def date_at(self, bucket: int) -> str:
        """Calendar date of a segment-local bucket.

        Raises:
            IndexError: the bucket is outside the segment.
        """
        if not 0 <= bucket < self.n_buckets:
            raise IndexError(f"bucket {bucket} is outside the {self.n_buckets}-bucket segment")
        offset = 0
        for day, n in zip(self.dates, self.day_n_buckets, strict=True):
            if bucket < offset + n:
                return day
            offset += n
        raise AssertionError("day_n_buckets do not sum to n_buckets")  # pragma: no cover


@dataclass(frozen=True)
class SplitRecord:
    """Frozen, JSON-serialisable description of a split (``to_json``)."""

    horizon_buckets: int
    latency_buckets: int
    arrival_lookback: int
    data_delay_buckets: int
    embargo_buckets: int
    n_before: int
    n_after: int
    purge_before: int  # L: buckets before t_0 in the purge window
    purge_after: int   # H + l + 1: buckets after t_0 in the purge window (closed)
    days: tuple[DayRecord, ...]
    segments: tuple[Segment, ...]

    def segments_for(self, role: str) -> tuple[Segment, ...]:
        """The role's segments in chronological order.

        Raises:
            ValueError: unknown role.
        """
        _check_role(role)
        return tuple(s for s in self.segments if s.role == role)

    def n_starts(self, role: str) -> int:
        """Admissible episode starts of a role, summed over its segments."""
        return sum(s.n_starts for s in self.segments_for(role))

    def start_at(self, role: str, position: int) -> tuple[int, int]:
        """``(segment index, t_0)`` of the ``position``-th admissible start of a role.

        Positions enumerate the role's segments in chronological order and
        each segment's range in increasing ``t_0``, so a position is a stable
        name for a start as long as the record is unchanged.

        Raises:
            IndexError: ``position`` is outside ``[0, n_starts(role))``.
        """
        if position < 0:
            raise IndexError(f"position {position} is negative")
        remaining = position
        for segment in self.segments_for(role):
            if remaining < segment.n_starts:
                return segment.index, segment.first_start + remaining
            remaining -= segment.n_starts
        raise IndexError(f"position {position} >= {self.n_starts(role)} starts of role {role!r}")

    def dates_for(self, role: str) -> tuple[str, ...]:
        """Calendar dates assigned to a role, in order (empty for an unused role)."""
        return tuple(d.date for d in self.days if d.role == role)

    def matches(self, spec: EpisodeSpec) -> bool:
        """Whether ``spec`` has the geometry (H, l, L, d) the ranges were computed for."""
        return (
            spec.horizon_buckets == self.horizon_buckets
            and spec.latency_buckets == self.latency_buckets
            and spec.arrival_lookback == self.arrival_lookback
            and spec.data_delay_buckets == self.data_delay_buckets
        )

    def to_json(self) -> dict[str, Any]:
        """Plain dict of built-in types (tuples become lists under ``json.dumps``)."""
        return asdict(self)


@dataclass(frozen=True, eq=False)
class Splits:
    """A ``SplitRecord`` with the segment tapes it describes (one ``origin == 0`` slice each)."""

    record: SplitRecord
    tapes: tuple[MarketSlice, ...]

    def episode(
        self, role: str, position: int, side: int, rho_bps: int, template: EpisodeSpec
    ) -> tuple[EpisodeSpec, MarketSlice]:
        """Cut the episode at the ``position``-th admissible start of ``role``.

        Raises:
            ValueError: ``template`` does not have the record's geometry.
            IndexError: ``position`` out of range.
        """
        if not self.record.matches(template):
            raise ValueError(
                "template geometry (H, l, L, d) = "
                f"({template.horizon_buckets}, {template.latency_buckets}, "
                f"{template.arrival_lookback}, {template.data_delay_buckets}) differs from the "
                f"split record's ({self.record.horizon_buckets}, {self.record.latency_buckets}, "
                f"{self.record.arrival_lookback}, {self.record.data_delay_buckets})"
            )
        segment, start = self.record.start_at(role, position)
        return episode_at(self.tapes[segment], start, side, rho_bps, template)


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")


def _ordered_days(days: Sequence[DayTape], dates: SplitDates) -> list[tuple[DayTape, str]]:
    """Sort the days, check uniqueness, assignment and chronological roles."""
    if not days:
        raise ValueError("build_splits needs at least one day")
    ordered = sorted(days, key=lambda d: (d.start_us, d.date))
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.date <= earlier.date or later.start_us < earlier.end_us:
            raise ValueError(
                f"days {earlier.date} and {later.date} are not in strictly increasing calendar "
                "order or overlap in time"
            )
    known = {d.date for d in ordered}
    for role in ROLES:
        for day in getattr(dates, role):
            if day not in known:
                raise ValueError(f"{role} date {day} has no day tape")
    assigned = []
    for day in ordered:
        assigned_role = dates.role_of(day.date)
        if assigned_role is None:
            raise ValueError(f"day {day.date} is not assigned to any role")
        assigned.append((day, assigned_role))
    latest_so_far = ""
    for role in ROLES:
        role_dates = _dates_of(assigned, role)
        if role_dates and latest_so_far and min(role_dates) <= latest_so_far:
            raise ValueError(
                f"{role} day {min(role_dates)} does not follow every earlier role's day "
                f"({latest_so_far}); splits must be chronological"
            )
        if role_dates:
            latest_so_far = max(role_dates)
    return assigned


def _dates_of(assigned: Sequence[tuple[DayTape, str]], role: str) -> tuple[str, ...]:
    return tuple(day.date for day, r in assigned if r == role)


def _group_segments(assigned: Sequence[tuple[DayTape, str]]) -> list[list[tuple[DayTape, str]]]:
    groups: list[list[tuple[DayTape, str]]] = []
    for day, role in assigned:
        if groups and groups[-1][-1][1] == role and groups[-1][-1][0].is_contiguous_with(day):
            groups[-1].append((day, role))
        else:
            groups.append([(day, role)])
    return groups


def _start_range(
    n_buckets: int, embargo_before: int, n_before: int, n_after: int, purge_after: int
) -> tuple[int, int]:
    first = embargo_before + n_before
    end = n_buckets - max(purge_after, n_after - 1)
    return (first, end) if end > first else (first, first)


def build_splits(days: Sequence[DayTape], dates: SplitDates, template: EpisodeSpec) -> Splits:
    """Assign days to roles and compute the admissible episode starts (module docstring).

    ``template`` supplies H, l, L and d; its side and target are ignored.

    Raises:
        ValueError: no days; a date listed twice, unassigned or without a
            tape; days out of calendar order or overlapping; roles not
            chronological.
    """
    assigned = _ordered_days(days, dates)
    n_before, n_after = required_window(template)
    purge_before, purge_after = (
        template.arrival_lookback,
        template.horizon_buckets + template.latency_buckets + 1,
    )
    embargo = embargo_buckets(template)

    segments: list[Segment] = []
    tapes: list[MarketSlice] = []
    previous_role: str | None = None
    for index, group in enumerate(_group_segments(assigned)):
        role = group[0][1]
        day_tapes = [day for day, _ in group]
        n_buckets = sum(d.n_buckets for d in day_tapes)
        embargo_before = embargo if previous_role not in (None, role) else 0
        first, end = _start_range(n_buckets, embargo_before, n_before, n_after, purge_after)
        segments.append(
            Segment(
                index=index,
                role=role,
                dates=tuple(d.date for d in day_tapes),
                day_n_buckets=tuple(d.n_buckets for d in day_tapes),
                n_buckets=n_buckets,
                embargo_before=embargo_before,
                first_start=first,
                end_start=end,
            )
        )
        tapes.append(concat_tapes(day_tapes))
        previous_role = role

    record = SplitRecord(
        horizon_buckets=template.horizon_buckets,
        latency_buckets=template.latency_buckets,
        arrival_lookback=template.arrival_lookback,
        data_delay_buckets=template.data_delay_buckets,
        embargo_buckets=embargo,
        n_before=n_before,
        n_after=n_after,
        purge_before=purge_before,
        purge_after=purge_after,
        days=tuple(
            DayRecord(
                date=day.date, role=role, start_us=day.start_us, width_us=day.width_us,
                n_buckets=day.n_buckets, source=day.source,
            )
            for day, role in assigned
        ),
        segments=tuple(segments),
    )
    return Splits(record=record, tapes=tuple(tapes))
