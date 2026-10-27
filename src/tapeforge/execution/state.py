"""Immutable data carried through an episode: the tape slice, the state and a step record.

``MarketSlice`` holds the per-bucket features of exactly the buckets one
episode may touch, indexed relative to the episode's first decision bucket::

    array index i  <->  episode bucket b = i - origin

Buckets ``b < 0`` precede the first decision (reference volume, arrival
price), buckets ``0 <= b < horizon_buckets`` are the decision buckets, and
buckets ``b >= horizon_buckets`` hold latency fills and the terminal mark.

Data contract:
    All integer fields are int64 ticks. ``vwap_ticks`` is the bucket VWAP
    rounded half-up to a price tick by ``vwap_ticks_from_numerators`` using
    exact integer arithmetic; no floating point enters the slice.

Invariants:
    ``State`` and ``StepResult`` are frozen dataclasses of Python ``int``, so
    ``==`` is exact equality of the accounting and a replayed episode is
    byte-identical to the original. Nothing in this module is mutated;
    ``simulator.step`` returns new objects.
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
    """Return each bucket's VWAP in whole price ticks, rounded half-up.

    ``numerators`` is the exact int64 ``sum(price_ticks * quantity_ticks)``
    per bucket (``reference.vwap_numerators``) and ``total_quantity`` the
    exact quantity sum. Buckets with zero quantity map to 0, matching the
    feature engine's ``vwap == 0.0`` convention. The rounding uses integer
    arithmetic only, ``(2 * numerator + total) // (2 * total)``, so the result
    does not depend on floating-point evaluation order and a half-tick VWAP
    always rounds up. The quantisation is part of the synthetic fill
    assumption: a fill price must lie on the exchange tick grid.

    Args:
        numerators: 1-D int64 array of per-bucket VWAP numerators.
        total_quantity: 1-D int64 array of per-bucket total quantity, in
            quantity ticks, with the same length as ``numerators``.

    Returns:
        1-D int64 array of price ticks, one per bucket.

    Raises:
        ValueError: The arrays differ in shape, are not 1-D, are not int64, or
            contain a negative value.
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
    """Return a read-only int64 copy of ``values``, which must be 1-D (``ValueError``)."""
    array = np.array(values, dtype=np.int64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class MarketSlice:
    """Per-bucket features of the buckets one episode may read, relative to ``origin``.

    Arrays are copied and made read-only at construction, and the instance is
    frozen, so a slice can be shared between episodes and replays. Invalid
    buckets must be all zeros, exactly as the feature engine emits them.
    Equality is identity (``eq=False``); compare the arrays directly.

    Attributes:
        event_count: int64, records per bucket.
        total_quantity: int64 quantity ticks traded in the bucket.
        signed_quantity: int64 quantity ticks, taker-signed; its magnitude
            never exceeds ``total_quantity``.
        vwap_ticks: int64 price ticks; 0 where no quantity traded.
        valid: bool, ``True`` exactly where ``event_count > 0``.
        origin: Number of buckets before the first decision bucket, so
            episode bucket ``b`` lives at array index ``b + origin``. In
            ``[0, n_buckets]``.
    """

    event_count: np.ndarray      # int64; records per bucket.
    total_quantity: np.ndarray   # int64 quantity ticks.
    signed_quantity: np.ndarray  # int64 quantity ticks, taker-signed.
    vwap_ticks: np.ndarray       # int64 price ticks; 0 where no quantity traded.
    valid: np.ndarray            # bool; False for an empty bucket.
    origin: int

    def __post_init__(self) -> None:
        """Copy, freeze and validate the arrays.

        Raises:
            TypeError: ``origin`` is not an integer.
            ValueError: Ragged lengths, wrong dtypes, a negative value, an
                ``origin`` outside ``[0, n_buckets]``, ``valid`` disagreeing
                with ``event_count > 0``, an invalid bucket carrying non-zero
                features, ``|signed_quantity| > total_quantity``, or a
                zero-quantity bucket with a non-zero VWAP.
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
        """Build a slice from the feature engine's output and the exact VWAP numerators.

        Args:
            features: ``reference.aggregate_reference(trades, offsets)``.
            numerators: ``reference.vwap_numerators(trades, offsets)`` on the
                same inputs. The float ``features.vwap`` is not used, so the
                slice never depends on floating-point rounding.
            origin: Number of buckets before the first decision bucket.

        Returns:
            A validated, read-only ``MarketSlice``.
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
        """Return the number of buckets in the slice, before and after the first decision."""
        return int(self.event_count.size)

    @property
    def n_after(self) -> int:
        """Return the number of buckets at or after the first decision bucket (``b >= 0``)."""
        return self.n_buckets - self.origin

    def has_bucket(self, bucket: int) -> bool:
        """Return whether episode bucket ``bucket`` lies inside the slice."""
        return -self.origin <= bucket < self.n_after

    def index(self, bucket: int) -> int:
        """Return the array index of episode bucket ``bucket``.

        Raises:
            IndexError: The bucket is not in the slice.
        """
        if not self.has_bucket(bucket):
            raise IndexError(
                f"bucket {bucket} is outside the slice [{-self.origin}, {self.n_after})"
            )
        return bucket + self.origin

    def vwap(self, bucket: int) -> float:
        """Return a bucket's VWAP in quote units, for display only (never for accounting)."""
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
        """Return a copy with one bucket's fields replaced.

        Tests perturb tapes this way. ``valid`` is recomputed from
        ``event_count``. Fields left ``None`` keep their value; the result is
        validated like any other slice.

        Raises:
            IndexError: ``bucket`` is not in the slice.
            ValueError: The replacement violates a slice invariant.
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
    ``inventory_ticks`` rises by ``side * fill``. ``exec_notional_ticks``,
    ``filled_qty_ticks`` and ``fees_ticks`` accumulate the same fills, so the
    implementation shortfall can be read without replaying. The mark price is
    the VWAP of the newest *priced* bucket (valid, with quantity) the policy
    has been allowed to see, and ``mark_age`` counts how many visible buckets
    have passed since it (0 when the newest visible bucket is itself priced).

    Attributes:
        step: Decisions taken so far, in ``[0, horizon_buckets]``.
        inventory_ticks: Signed position, in quantity ticks.
        cash_ticks: Signed cash balance, in cash ticks; zero at reset.
        fees_ticks: Cumulative fees paid, in cash ticks.
        filled_qty_ticks: Cumulative filled quantity, in quantity ticks;
            never negative.
        exec_notional_ticks: Cumulative ``sum(fill_price * fill_qty)``, in
            cash ticks.
        arrival_price_ticks: Arrival price in price ticks, fixed at reset.
        mark_price_ticks: VWAP of the newest priced visible bucket, in price
            ticks.
        mark_age: Visible buckets since the mark bucket closed.
    """

    step: int                     # Decisions taken so far, 0 .. horizon_buckets.
    inventory_ticks: int          # Signed position, quantity ticks.
    cash_ticks: int               # Signed, cash ticks.
    fees_ticks: int               # Cumulative fees paid, cash ticks.
    filled_qty_ticks: int         # Cumulative filled quantity, >= 0.
    exec_notional_ticks: int      # Cumulative sum(fill_price * fill_qty), cash ticks.
    arrival_price_ticks: int      # Arrival price, fixed at reset.
    mark_price_ticks: int         # Newest priced visible VWAP, price ticks.
    mark_age: int                 # Visible buckets since the mark bucket closed.


@dataclass(frozen=True)
class StepResult:
    """Record of one decision, with the reward split into its three terms.

    Signs: ``cost_component`` is
    ``side * (notional_ticks - arrival_price_ticks * fill_qty_ticks) + fee_ticks``
    (positive is a cost for both sides), ``risk_component`` is
    ``risk_lambda * remaining_after`` (``>= 0``) and ``terminal_component`` is
    the residual term at the horizon (zero before it and when nothing is
    left). ``reward_ticks == -(cost_component + risk_component +
    terminal_component)`` exactly, so over a terminated episode
    ``sum(reward_ticks) == -shortfall - risk_lambda * sum(remaining_after)``,
    with ``shortfall`` the implementation shortfall.

    ``terminated`` and ``truncated`` are never both set. ``fill_price_ticks``
    and ``terminal_price_ticks`` are 0 when there was no fill or no terminal
    mark was needed or available.

    Attributes:
        step: The decision index ``t``, in ``[0, horizon_buckets)``.
        fill_bucket: ``step + latency_buckets``; the bucket the order
            executed in.
        submitted_qty_ticks: ``floor(action * remaining_before)``, in
            quantity ticks.
        cap_qty_ticks: Liquidity cap of the fill bucket, in quantity ticks
            (0 when the bucket is not priced).
        fill_qty_ticks: Quantity filled, ``min(submitted, cap)``.
        fill_price_ticks: Fill price in price ticks, including slippage; 0
            without a fill.
        notional_ticks: ``fill_price_ticks * fill_qty_ticks``, in cash ticks.
        fee_ticks: Taker fee on the notional, in cash ticks.
        cost_component: Shortfall of this fill against the arrival price plus
            the fee, in cash ticks.
        risk_component: ``risk_lambda * remaining_after``, in cash ticks.
        terminal_component: Residual term charged at the horizon, in cash
            ticks; zero otherwise.
        terminal_price_ticks: Terminal mark used for the residual term; 0
            when none was needed or available.
        reward_ticks: ``-(cost_component + risk_component +
            terminal_component)``.
        terminated: The target was filled or the horizon was reached, with a
            terminal mark available if one was needed.
        truncated: The fill bucket or a needed terminal mark lies outside the
            slice.
    """

    step: int                     # The decision index t, 0 .. horizon_buckets - 1.
    fill_bucket: int              # step + latency_buckets; the bucket the order executed in.
    submitted_qty_ticks: int      # floor(action * remaining_before).
    cap_qty_ticks: int            # Liquidity cap of the fill bucket (0 when not priced).
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
