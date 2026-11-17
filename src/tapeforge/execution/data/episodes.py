"""Episode construction: the tape window an episode needs and target sizing.

Given a day's trades and bucket offsets, ``make_episode`` cuts the
``MarketSlice`` around a start bucket and sizes ``target_qty_ticks`` from the
volume *before* the start::

    v_ref_ticks      = total quantity over the horizon_buckets buckets
                       preceding the start (causal)
    target_qty_ticks = round_half_up(rho_bps * v_ref_ticks), rho_bps in
                       basis points, capped at participation_cap_bps *
                       v_ref_ticks, then floored at one tick

Using the episode's own future volume for the reference volume would leak;
the preceding ``horizon_buckets`` buckets are known at the start. The cap is
the participation cap applied to the reference volume: the quantity the fill
rule could complete over ``horizon_buckets`` buckets of volume like the
reference window, so completion is feasible by construction when the future
resembles the past. The one-tick floor is applied last, so the target is
always positive.

Window:
    ``required_window`` returns how many buckets before and after the start
    the slice must carry: ``max(horizon_buckets, arrival_lookback +
    data_delay_buckets)`` before (reference volume, and the arrival lookback
    under the data delay) and ``horizon_buckets + latency_buckets +
    arrival_lookback`` after (every fill bucket plus the terminal-mark
    search). A day that ends early yields a shorter slice and a truncated
    episode; too little history before the start is an error.
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
    """Return ``(n_before, n_after)``: the buckets the slice needs around the start.

    See the module docstring for the formulas.
    """
    n_before = max(spec.horizon_buckets, spec.arrival_lookback + spec.data_delay_buckets)
    n_after = spec.horizon_buckets + spec.latency_buckets + spec.arrival_lookback
    return n_before, n_after


def round_half_up_bps(value: int, rate_bps: int) -> int:
    """Return ``round_half_up(value * rate_bps / 10_000)`` in integer arithmetic."""
    return (2 * value * rate_bps + BPS_DENOMINATOR) // (2 * BPS_DENOMINATOR)


def reference_volume_ticks(market: MarketSlice, horizon_buckets: int) -> int:
    """Return the reference volume in quantity ticks.

    Sums ``total_quantity`` over episode buckets ``-horizon_buckets .. -1`` of
    the slice.

    Raises:
        ValueError: The slice carries fewer than ``horizon_buckets`` buckets
            before the first decision.
    """
    if market.origin < horizon_buckets:
        raise ValueError(
            f"reference volume needs {horizon_buckets} buckets before the first decision, "
            f"the slice has {market.origin}"
        )
    start = market.origin - horizon_buckets
    return int(market.total_quantity[start : market.origin].sum())


def target_from_reference_volume(v_ref_ticks: int, rho_bps: int, cap_bps: int) -> int:
    """Return the target quantity sized from the reference volume.

    Computes ``round_half_up(rho_bps * v_ref_ticks)``, capped at
    ``floor(cap_bps * v_ref_ticks)``, then floored at one tick.

    Args:
        v_ref_ticks: Reference volume in quantity ticks.
        rho_bps: Target fraction of the reference volume, in basis points.
        cap_bps: Participation cap in basis points.

    Returns:
        The target in quantity ticks; always at least 1.

    Raises:
        ValueError: A negative volume or rate.
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
    ``offsets``; the slice then keeps buckets
    ``[start_bucket - n_before, start_bucket + n_after)``, clipped at the end
    of the window. Fewer buckets after the start mean the episode will
    truncate, not fail.

    Raises:
        ValueError: ``start_bucket - n_before < 0``, ``start_bucket`` is past
            the window, or ``aggregate_reference`` rejects the inputs.
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
    selects the default production configuration.

    Returns:
        The finished spec and its slice. ``simulator.reset`` still validates
        the arrival price and the int64 bounds.
    """
    base = EpisodeSpec(side=side, target_qty_ticks=1) if template is None else replace(
        template, side=side, target_qty_ticks=1
    )
    n_before, n_after = required_window(base)
    market = build_market_slice(trades, offsets, start_bucket, n_before, n_after)
    v_ref = reference_volume_ticks(market, base.horizon_buckets)
    target = target_from_reference_volume(v_ref, rho_bps, base.participation_cap_bps)
    return replace(base, target_qty_ticks=target), market
