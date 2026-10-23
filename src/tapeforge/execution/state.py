"""Immutable data carried through an episode: the tape slice, the state, a step.

``MarketSlice`` holds the per-bucket features of exactly the buckets one
episode may touch, indexed relative to the episode start ``t_0``:

    array index i  <->  episode bucket b = i - origin

Buckets ``b < 0`` precede the first decision (reference volume, arrival
price), buckets ``0 <= b < H`` are the decision buckets, and buckets ``b >= H``
hold latency fills and the terminal mark. All integer fields are int64 ticks;
``vwap_ticks`` is the bucket VWAP rounded half-up to a price tick by
``vwap_ticks_from_numerators`` (exact integer arithmetic, no floating point).

``State`` and ``StepResult`` are frozen dataclasses of Python ``int`` so that
``==`` is exact equality of the accounting and a replayed episode is
byte-identical to the original. Nothing here is mutated; ``simulator.step``
returns new objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tapeforge.contract import PRICE_SCALE
from tapeforge.reference import Features

__all__ = [
    "MarketSlice",
    "State",
    "StepResult",
    "vwap_ticks_from_numerators",
]


def vwap_ticks_from_numerators(numerators: np.ndarray, total_quantity: np.ndarray) -> np.ndarray:
    """Bucket VWAP in whole price ticks: ``round_half_up(numerator / total)``.

    ``numerators`` is the exact int64 ``sum(price_ticks * quantity_ticks)`` per
    bucket (``reference.vwap_numerators``) and ``total_quantity`` the exact
    quantity sum. Buckets with zero quantity map to 0, matching the engine's
    ``vwap == 0.0`` convention. The rounding is done with integer arithmetic
    only, ``(2 * numerator + total) // (2 * total)``, so the result does not
    depend on floating-point evaluation order and a half-tick VWAP always
    rounds up. This quantisation is part of the synthetic fill assumption: a
    fill price must lie on the exchange tick grid.

    Raises:
        ValueError: shape or dtype mismatch, or a negative input.
    """
    numerators = np.asarray(numerators)
    total_quantity = np.asarray(total_quantity)
    if numerators.shape != total_quantity.shape or numerators.ndim != 1:
        raise ValueError(
            f"numerators {numerators.shape} and total_quantity {total_quantity.shape} "
            "must be 1-D and the same length"
        )
    if numerators.dtype != np.int64 or total_quantity.dtype != np.int64:
        raise ValueError("numerators and total_quantity must be int64")
    if numerators.size and (numerators.min() < 0 or total_quantity.min() < 0):
        raise ValueError("numerators and total_quantity must be non-negative")
    ticks = np.zeros(numerators.shape, dtype=np.int64)
    weighted = total_quantity > 0
    doubled = 2 * total_quantity[weighted]
    ticks[weighted] = (2 * numerators[weighted] + total_quantity[weighted]) // doubled
    return ticks


def _read_only_int64(values: np.ndarray, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.int64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class MarketSlice:
    """Features of the buckets one episode may read, relative to ``origin``.

    ``origin`` is the number of buckets before the first decision bucket, so
    episode bucket ``b`` lives at array index ``b + origin``. Arrays are
    copied and made read-only at construction; invalid buckets must be all
    zeros, exactly as the feature engine emits them.
    """

    event_count: np.ndarray      # int64, records per bucket
    total_quantity: np.ndarray   # int64 quantity ticks
    signed_quantity: np.ndarray  # int64 quantity ticks, taker-signed
    vwap_ticks: np.ndarray       # int64 price ticks, 0 where no quantity traded
    valid: np.ndarray            # bool, False for an empty bucket
    origin: int

    def __post_init__(self) -> None:
        """Copy, freeze and validate the arrays.

        Raises:
            ValueError: ragged lengths, wrong dtypes, a negative value, an
                ``origin`` outside ``[0, n_buckets]``, or an invalid bucket
                carrying non-zero features.
        """
        n = len(self.event_count)
        for name in ("event_count", "total_quantity", "signed_quantity", "vwap_ticks"):
            array = _read_only_int64(getattr(self, name), name)
            if array.size != n:
                raise ValueError(f"{name} has {array.size} buckets, event_count has {n}")
            object.__setattr__(self, name, array)
        valid = np.array(self.valid, copy=True)
        if valid.dtype != np.bool_ or valid.shape != (n,):
            raise ValueError(
                f"valid must be a bool array of shape ({n},), got {valid.dtype} {valid.shape}"
            )
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        if isinstance(self.origin, bool) or not isinstance(self.origin, int | np.integer):
            raise TypeError(f"origin must be an integer, got {self.origin!r}")
        object.__setattr__(self, "origin", int(self.origin))
        if not 0 <= self.origin <= n:
            raise ValueError(f"origin must be in [0, {n}], got {self.origin}")
        if n and min(
            int(self.event_count.min()), int(self.total_quantity.min()), int(self.vwap_ticks.min())
        ) < 0:
            raise ValueError("event_count, total_quantity and vwap_ticks must be non-negative")
        if np.any((self.event_count > 0) != self.valid):
            raise ValueError("valid must be exactly event_count > 0")
        empty = ~self.valid
        if np.any(self.total_quantity[empty]) or np.any(self.signed_quantity[empty]) or np.any(
            self.vwap_ticks[empty]
        ):
            raise ValueError("invalid buckets must carry zero quantity, signed quantity and VWAP")
        if np.any(np.abs(self.signed_quantity) > self.total_quantity):
            raise ValueError("|signed_quantity| cannot exceed total_quantity")
        if np.any(self.vwap_ticks[self.total_quantity == 0]):
            raise ValueError("a bucket with no quantity must have vwap_ticks == 0")

    @classmethod
    def from_features(cls, features: Features, numerators: np.ndarray, origin: int) -> MarketSlice:
        """Build from the feature engine's output plus the exact VWAP numerators.

        ``features`` is ``reference.aggregate_reference(trades, offsets)`` and
        ``numerators`` is ``reference.vwap_numerators(trades, offsets)`` on
        the same inputs; the float ``features.vwap`` is not used, so the slice
        never depends on floating-point rounding.
        """
        return cls(
            event_count=features.event_count,
            total_quantity=features.total_quantity,
            signed_quantity=features.signed_quantity,
            vwap_ticks=vwap_ticks_from_numerators(numerators, features.total_quantity),
            valid=features.valid,
            origin=origin,
        )

    @property
    def n_buckets(self) -> int:
        """Buckets in the slice, before and after ``t_0`` together."""
        return int(self.event_count.size)

    @property
    def n_after(self) -> int:
        """Buckets at or after the first decision bucket (``b >= 0``)."""
        return self.n_buckets - self.origin

    def has_bucket(self, bucket: int) -> bool:
        """Whether episode bucket ``bucket`` (relative to ``t_0``) is in the slice."""
        return -self.origin <= bucket < self.n_after

    def index(self, bucket: int) -> int:
        """Array index of episode bucket ``bucket``.

        Raises:
            IndexError: the bucket is not in the slice.
        """
        if not self.has_bucket(bucket):
            raise IndexError(
                f"bucket {bucket} is outside the slice [{-self.origin}, {self.n_after})"
            )
        return bucket + self.origin

    def vwap(self, bucket: int) -> float:
        """VWAP of a bucket in quote units, for display only (never for accounting)."""
        return int(self.vwap_ticks[self.index(bucket)]) / PRICE_SCALE

    def with_bucket(
        self,
        bucket: int,
        *,
        event_count: int | None = None,
        total_quantity: int | None = None,
        signed_quantity: int | None = None,
        vwap_ticks: int | None = None,
    ) -> MarketSlice:
        """Return a copy with one bucket's fields replaced (tests perturb tapes this way).

        ``valid`` is recomputed from ``event_count``. Fields left ``None`` keep
        their value; the result is validated like any other slice.
        """
        i = self.index(bucket)
        columns = {
            "event_count": self.event_count.copy(),
            "total_quantity": self.total_quantity.copy(),
            "signed_quantity": self.signed_quantity.copy(),
            "vwap_ticks": self.vwap_ticks.copy(),
        }
        replacements = {
            "event_count": event_count,
            "total_quantity": total_quantity,
            "signed_quantity": signed_quantity,
            "vwap_ticks": vwap_ticks,
        }
        for name, value in replacements.items():
            if value is not None:
                columns[name][i] = value
        return MarketSlice(valid=columns["event_count"] > 0, origin=self.origin, **columns)


@dataclass(frozen=True)
class State:
    """Accounting state after ``step`` decisions. Every field is a Python ``int``.

    ``cash_ticks`` falls by ``side * notional + fee`` on every fill and
    ``inventory_ticks`` rises by ``side * fill``; ``exec_notional_ticks``,
    ``filled_qty_ticks`` and ``fees_ticks`` accumulate the same fills so the
    implementation shortfall can be read without replaying. The mark price is
    the VWAP of the newest *valid* bucket the agent has been allowed to see,
    and ``mark_age`` counts how many visible buckets have passed since it (0
    when the newest visible bucket is itself valid).
    """

    step: int                     # decisions taken so far, 0 .. H
    inventory_ticks: int          # signed position, quantity ticks
    cash_ticks: int               # signed, cash ticks
    fees_ticks: int               # cumulative fees paid, cash ticks
    filled_qty_ticks: int         # cumulative filled quantity, >= 0
    exec_notional_ticks: int      # cumulative sum(fill_price * fill_qty), cash ticks
    arrival_price_ticks: int      # P_0, fixed at reset
    mark_price_ticks: int         # last valid visible VWAP, price ticks
    mark_age: int                 # visible buckets since the mark bucket closed


@dataclass(frozen=True)
class StepResult:
    """What one decision did, with the reward split into its three terms.

    Signs: ``cost_component`` is ``side * (notional - P_0 * fill) + fee``
    (positive is a cost for both sides), ``risk_component`` is
    ``lambda * remaining_after`` (``>= 0``) and ``terminal_component`` is the
    residual term at the horizon (zero before it and when nothing is left).
    ``reward_ticks == -(cost_component + risk_component + terminal_component)``
    exactly, so over a terminated episode
    ``sum(reward) == -IS - lambda * sum(remaining_after)``.

    ``terminated`` and ``truncated`` are never both set. ``fill_price_ticks``
    and ``terminal_price_ticks`` are 0 when there was no fill or no terminal
    mark was needed or available.
    """

    step: int                     # the decision index t, 0 .. H - 1
    fill_bucket: int              # t + latency; the bucket the order executed in
    submitted_qty_ticks: int      # floor(action * remaining_before)
    cap_qty_ticks: int            # liquidity cap of the fill bucket (0 when invalid)
    fill_qty_ticks: int
    fill_price_ticks: int
    notional_ticks: int
    fee_ticks: int
    cost_component: int
    risk_component: int
    terminal_component: int
    terminal_price_ticks: int
    reward_ticks: int
    terminated: bool
    truncated: bool
