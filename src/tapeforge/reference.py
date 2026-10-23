"""NumPy reference features: the correctness oracle every other path is held to.

``aggregate_reference`` computes the five per-bucket features of the data
contract from a timestamp-sorted ``TRADE_DTYPE`` array and the bucket
offsets of ``windows.build_offsets``; ``aggregate_reference_naive`` computes
the same features with a plain Python loop and exists only to validate the
vectorised version. ``vwap_numerators`` exposes the exact int64 VWAP
numerator for consumers that must round on the integer price grid.

Data contract: ``event_count``, ``total_quantity`` and ``signed_quantity``
are exact int64 (quantity ticks, taker-signed for the last); ``vwap`` is a
float64 in quote units whose only floating-point operation is the final
division by the bucket's quantity; ``valid`` is ``event_count > 0``. A
bucket with records but no quantity is valid with ``vwap == 0.0``. Callers
are responsible for the int64 accumulation bound
(``contract.check_exact_accumulation_bound``); this module does not check
it, so it stays usable as an oracle on inputs the engine would reject.

Torch-free. Nothing here is tuned for speed beyond staying vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tapeforge.contract import (
    PRICE_SCALE,
    SIGN_BUY_INITIATED,
    SIGN_SELL_INITIATED,
    TRADE_DTYPE,
)

# The engine's convention for a bucket whose records carry no quantity; the
# GPU kernels and the streaming aggregator emit the same value.
VWAP_WHEN_NO_QUANTITY: float = 0.0


@dataclass(frozen=True)
class Features:
    """Per-bucket features. All arrays have shape ``(n_buckets,)``.

    Frozen: features are returned as new objects, never mutated in place.
    """

    event_count: np.ndarray      # int64, exact
    total_quantity: np.ndarray   # int64 quantity ticks, exact
    signed_quantity: np.ndarray  # int64 quantity ticks, exact, may be negative
    vwap: np.ndarray             # float64 price units, 0.0 where not valid
    valid: np.ndarray            # bool, False for empty buckets


def taker_sign(buyer_is_maker: np.ndarray) -> np.ndarray:
    """Map the exchange's ``buyer_is_maker`` flag to the taker sign, as int64.

    ``buyer_is_maker == True`` means the seller was the aggressor, so the
    record counts as sell-initiated (``SIGN_SELL_INITIATED``); otherwise it
    is buy-initiated. The flag is kept raw in ``TRADE_DTYPE`` and converted
    here, in one place, so every implementation shares the convention.

    Raises:
        ValueError: the column is not a bool array.
    """
    flags = np.asarray(buyer_is_maker)
    if flags.dtype != np.bool_:
        raise ValueError(f"buyer_is_maker must be bool, got {flags.dtype}")
    return np.where(flags, SIGN_SELL_INITIATED, SIGN_BUY_INITIATED).astype(np.int64)


def _validate(trades: np.ndarray, offsets: np.ndarray) -> None:
    if trades.dtype != TRADE_DTYPE:
        raise ValueError(f"trades must have TRADE_DTYPE, got {trades.dtype}")
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"offsets must be 1-D with at least 2 entries, got {offsets.shape}")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be non-decreasing; came from a different trade array?")
    if offsets.size and (offsets[0] < 0 or offsets[-1] > trades.size):
        raise ValueError(
            f"offsets span [{int(offsets[0])}, {int(offsets[-1])}] but there are "
            f"{trades.size} records"
        )


def _segment_sums(values: np.ndarray, offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Exact per-segment int64 sums, handling empty segments and the tail.

    ``np.add.reduceat`` needs the *full* offsets array, not ``offsets[:-1]``:
    each segment is bounded by the next index, and only the final segment runs
    to the end of the array. Passing one offset per boundary and discarding
    that final segment is what keeps every bucket's end boundary intact.
    """
    if values.size == 0:
        return np.zeros(lengths.size, dtype=np.int64)
    # reduceat rejects an index equal to the array length, which is exactly
    # where the last boundary sits when the window ends at the final record;
    # a trailing zero gives that boundary a valid (empty) segment to start.
    padded = np.append(values, 0) if int(offsets[-1]) == values.size else values

    # Segment i is sum(padded[offsets[i]:offsets[i + 1]]); the final segment
    # (from offsets[-1] to the end) is the tail after the window and is dropped.
    # reduceat returns the single element, not zero, for an empty segment, so
    # empty buckets are zeroed explicitly from the lengths.
    sums = np.add.reduceat(padded, offsets)[:-1]
    return np.where(lengths > 0, sums, 0).astype(np.int64)


def aggregate_reference(trades: np.ndarray, offsets: np.ndarray) -> Features:
    """Compute per-bucket features from a sorted trade array and window offsets.

    ``trades`` is a structured array of ``contract.TRADE_DTYPE``; ``offsets``
    comes from ``tapeforge.windows.build_offsets``. Records outside
    ``[offsets[0], offsets[-1])`` are never read.

    Raises:
        ValueError: wrong dtype, offsets not 1-D / decreasing / outside the
            record range.
    """
    _validate(trades, offsets)
    lengths = np.diff(offsets).astype(np.int64)
    valid = lengths > 0

    # Per-record products in int64 before any reduction, so the sums are exact
    # within the contract bound and the sign never passes through a float.
    quantity = trades["quantity"].astype(np.int64)
    signed = quantity * taker_sign(trades["buyer_is_maker"])
    numerator = trades["price"].astype(np.int64) * quantity

    total_quantity = _segment_sums(quantity, offsets, lengths)
    signed_quantity = _segment_sums(signed, offsets, lengths)
    numerator_sum = _segment_sums(numerator, offsets, lengths)

    # The one floating-point step: numerator / quantity / PRICE_SCALE, in this
    # order, once per bucket. The kernels and the streaming aggregator use the
    # same order so results agree within the contract tolerance.
    vwap = np.full(lengths.size, VWAP_WHEN_NO_QUANTITY, dtype=np.float64)
    weighted = total_quantity > 0
    vwap[weighted] = (
        numerator_sum[weighted].astype(np.float64) / total_quantity[weighted] / PRICE_SCALE
    )

    return Features(
        event_count=lengths,
        total_quantity=total_quantity,
        signed_quantity=signed_quantity,
        vwap=vwap,
        valid=valid,
    )


def vwap_numerators(trades: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Exact int64 ``sum(price_ticks * quantity_ticks)`` per bucket, shape ``(n_buckets,)``.

    This is the VWAP numerator before the engine's single floating-point
    division. Consumers that need a VWAP on the integer price grid (the
    execution simulator's fill price) round this exactly instead of rounding
    the float ``Features.vwap``. Same validation as ``aggregate_reference``;
    the caller is responsible for the int64 accumulation bound.
    """
    _validate(trades, offsets)
    lengths = np.diff(offsets).astype(np.int64)
    numerator = trades["price"].astype(np.int64) * trades["quantity"].astype(np.int64)
    return _segment_sums(numerator, offsets, lengths)


def aggregate_reference_naive(trades: np.ndarray, offsets: np.ndarray) -> Features:
    """The same features, computed with a plain Python loop.

    Deliberately the dumbest correct implementation in the project: Python
    integers, one bucket at a time, no NumPy reductions. It exists to validate
    ``aggregate_reference``, and is far too slow for anything else.
    """
    _validate(trades, offsets)
    n_buckets = offsets.size - 1
    counts = np.zeros(n_buckets, dtype=np.int64)
    totals = np.zeros(n_buckets, dtype=np.int64)
    signeds = np.zeros(n_buckets, dtype=np.int64)
    vwaps = np.full(n_buckets, VWAP_WHEN_NO_QUANTITY, dtype=np.float64)

    for bucket in range(n_buckets):
        total = 0
        signed = 0
        numerator = 0
        count = 0
        for i in range(int(offsets[bucket]), int(offsets[bucket + 1])):
            quantity = int(trades["quantity"][i])
            sign = SIGN_SELL_INITIATED if trades["buyer_is_maker"][i] else SIGN_BUY_INITIATED
            count += 1
            total += quantity
            signed += quantity * sign
            numerator += int(trades["price"][i]) * quantity
        counts[bucket] = count
        totals[bucket] = total
        signeds[bucket] = signed
        if total > 0:
            vwaps[bucket] = numerator / total / PRICE_SCALE

    return Features(
        event_count=counts,
        total_quantity=totals,
        signed_quantity=signeds,
        vwap=vwaps,
        valid=counts > 0,
    )
