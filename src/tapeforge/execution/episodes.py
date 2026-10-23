"""Episode construction: the tape window an episode needs, V_ref and the target Q*.

Given a day's trades and bucket offsets, ``make_episode`` cuts the
``MarketSlice`` around a start bucket ``t_0`` and sizes the target from the
volume *before* ``t_0``:

    V_ref = total quantity over the H buckets preceding t_0   (causal)
    Q*    = round_half_up(rho * V_ref), rho in basis points,
            floored at one tick, capped at participation_cap * V_ref

Using the episode's own future volume for V_ref would leak; using the
preceding H buckets is known at ``t_0``. The cap is the participation cap
applied to the reference volume, the quantity the fill rule could complete
over H buckets of volume like the reference window, so completion is
feasible by construction when the future resembles the past. The one-tick
floor is applied last so Q* is always positive.

Window: ``required_window`` returns how many buckets before and after ``t_0``
the slice must carry: ``max(H, L + d)`` before (reference volume and arrival
lookback under the data delay) and ``H + l + L`` after (every fill bucket plus
the terminal-mark search). A day that ends early yields a shorter slice and a
truncated episode; too little history before ``t_0`` is an error.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tapeforge import reference
from tapeforge.execution.spec import BPS_DENOMINATOR, EpisodeSpec
from tapeforge.execution.state import MarketSlice

__all__ = [
    "build_market_slice",
    "make_episode",
    "reference_volume_ticks",
    "required_window",
    "round_half_up_bps",
    "target_from_reference_volume",
]


def required_window(spec: EpisodeSpec) -> tuple[int, int]:
    """``(n_before, n_after)``: buckets the slice needs around ``t_0`` for ``spec``."""
    n_before = max(spec.horizon_buckets, spec.arrival_lookback + spec.data_delay_buckets)
    n_after = spec.horizon_buckets + spec.latency_buckets + spec.arrival_lookback
    return n_before, n_after


def round_half_up_bps(value: int, rate_bps: int) -> int:
    """``round_half_up(value * rate_bps / 10_000)`` in integer arithmetic."""
    return (2 * value * rate_bps + BPS_DENOMINATOR) // (2 * BPS_DENOMINATOR)


def reference_volume_ticks(market: MarketSlice, horizon_buckets: int) -> int:
    """V_ref: total quantity over buckets ``-H .. -1`` of the slice, in quantity ticks.

    Raises:
        ValueError: the slice carries fewer than ``horizon_buckets`` buckets before ``t_0``.
    """
    if market.origin < horizon_buckets:
        raise ValueError(
            f"reference volume needs {horizon_buckets} buckets before the first decision, "
            f"the slice has {market.origin}"
        )
    start = market.origin - horizon_buckets
    return int(market.total_quantity[start : market.origin].sum())


def target_from_reference_volume(v_ref_ticks: int, rho_bps: int, cap_bps: int) -> int:
    """Q* from V_ref: ``round_half_up(rho * V_ref)``, capped at ``cap * V_ref``, at least one tick.

    Raises:
        ValueError: a negative volume or rate.
    """
    if min(v_ref_ticks, rho_bps, cap_bps) < 0:
        raise ValueError("reference volume and rates must be non-negative")
    target = round_half_up_bps(v_ref_ticks, rho_bps)
    feasible = v_ref_ticks * cap_bps // BPS_DENOMINATOR
    return max(1, min(target, feasible))


def build_market_slice(
    trades: np.ndarray, offsets: np.ndarray, start_bucket: int, n_before: int, n_after: int
) -> MarketSlice:
    """Cut the slice for an episode whose first decision bucket is ``start_bucket``.

    Features come from ``reference.aggregate_reference`` over all of
    ``offsets``; the slice then keeps buckets ``[start - n_before, start +
    n_after)``, clipped at the end of the window (fewer buckets after the
    start mean the episode will truncate, not fail).

    Raises:
        ValueError: ``start_bucket - n_before < 0`` or ``start_bucket`` is past
            the window, plus whatever ``aggregate_reference`` rejects.
    """
    n_buckets = offsets.size - 1
    first = start_bucket - n_before
    if first < 0:
        raise ValueError(
            f"start bucket {start_bucket} has only {start_bucket} buckets of history, "
            f"{n_before} are needed"
        )
    if start_bucket >= n_buckets:
        raise ValueError(f"start bucket {start_bucket} is past the {n_buckets}-bucket window")
    last = min(start_bucket + n_after, n_buckets)
    features = reference.aggregate_reference(trades, offsets)
    numerators = reference.vwap_numerators(trades, offsets)
    window = reference.Features(
        event_count=features.event_count[first:last],
        total_quantity=features.total_quantity[first:last],
        signed_quantity=features.signed_quantity[first:last],
        vwap=features.vwap[first:last],
        valid=features.valid[first:last],
    )
    return MarketSlice.from_features(window, numerators[first:last], origin=n_before)


def make_episode(
    trades: np.ndarray,
    offsets: np.ndarray,
    start_bucket: int,
    side: int,
    rho_bps: int,
    template: EpisodeSpec | None = None,
) -> tuple[EpisodeSpec, MarketSlice]:
    """Slice the tape at ``start_bucket`` and size the target from the preceding volume.

    ``template`` supplies every spec field except ``side`` and
    ``target_qty_ticks`` (its own values for those two are ignored); ``None``
    means the accepted defaults. Returns the finished spec and its slice;
    ``simulator.reset`` still validates the arrival price and the int64
    bounds.
    """
    base = EpisodeSpec(side=side, target_qty_ticks=1) if template is None else replace(
        template, side=side, target_qty_ticks=1
    )
    n_before, n_after = required_window(base)
    market = build_market_slice(trades, offsets, start_bucket, n_before, n_after)
    v_ref = reference_volume_ticks(market, base.horizon_buckets)
    target = target_from_reference_volume(v_ref, rho_bps, base.participation_cap_bps)
    return replace(base, target_qty_ticks=target), market
