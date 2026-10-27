"""Streaming (live-path) trade aggregation on the CPU, independent of CUDA.

``StreamingAggregator`` computes the same five features as
``reference.aggregate_reference`` over the window
``windows.build_offsets(timestamps, start_us, n_buckets, width_us)``, but
from a stream of timestamp-ordered chunks instead of one array. A chunk can
be a single record (the per-event case) or a whole day; the result is
identical either way, and identical to the batch result.

Invariants:
    Chunks must arrive in timestamp order, and each chunk must be internally
    non-decreasing. Buckets are emitted at most once, in order, as soon as
    the newest timestamp proves they can receive nothing more. A rejected
    chunk leaves the aggregator unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from tapeforge import windows
from tapeforge.contract import (
    DEFAULT_BUCKET_WIDTH_US,
    PRICE_SCALE,
    TRADE_DTYPE,
    check_exact_accumulation_bound,
)
from tapeforge.reference import VWAP_WHEN_NO_QUANTITY, Features, taker_sign

__all__ = ["OutOfOrderError", "StreamingAggregator", "aggregate_stream", "concat_features"]


class OutOfOrderError(ValueError):
    """A chunk is not timestamp-ordered, or starts before the previous chunk ended."""


class StreamingAggregator:
    """Accumulator of per-bucket features over timestamp-ordered chunks of trades.

    ``start_us``, ``n_buckets`` and ``width_us`` define the same fixed window
    as ``windows.build_offsets``; the geometry is validated at construction.
    Records outside the window are counted in ``records_seen`` but contribute
    to no bucket.

    Args:
        start_us: Timestamp of the first bucket edge, in microseconds.
        n_buckets: Number of buckets in the window; must be positive.
        width_us: Bucket width in microseconds; must be positive.

    Raises:
        ValueError: ``n_buckets`` or ``width_us`` is not positive.
    """

    def __init__(
        self, start_us: int, n_buckets: int, width_us: int = DEFAULT_BUCKET_WIDTH_US
    ) -> None:
        self._edges = windows.bucket_edges(start_us, n_buckets, width_us)
        self._start_us = int(start_us)
        self._width_us = int(width_us)
        self._n_buckets = int(n_buckets)
        self._end_us = int(self._edges[-1])

        self._count = np.zeros(n_buckets, dtype=np.int64)
        self._total = np.zeros(n_buckets, dtype=np.int64)
        self._signed = np.zeros(n_buckets, dtype=np.int64)
        self._numerator = np.zeros(n_buckets, dtype=np.int64)

        self._emitted = 0
        self._records_seen = 0
        self._last_timestamp: int | None = None
        self._max_price = 0
        self._max_quantity = 0
        self._finished = False

    # --- Read-only state ----------------------------------------------------

    @property
    def n_buckets(self) -> int:
        """Return the buckets in the window: the total rows emitted over the stream."""
        return self._n_buckets

    @property
    def emitted_buckets(self) -> int:
        """Return how many leading buckets have already been returned."""
        return self._emitted

    @property
    def records_seen(self) -> int:
        """Return the number of records pushed so far, inside the window or not."""
        return self._records_seen

    @property
    def last_timestamp(self) -> int | None:
        """Return the timestamp of the last record pushed; the floor for the next chunk."""
        return self._last_timestamp

    @property
    def is_finished(self) -> bool:
        """Return ``True`` after ``finalize``; further ``push`` calls raise."""
        return self._finished

    # --- Stream operations --------------------------------------------------

    def push(self, chunk: np.ndarray) -> Features:
        """Absorb one ordered chunk and return the buckets it closed.

        On any error the aggregator is unchanged.

        Args:
            chunk: 1-D ``TRADE_DTYPE`` array with non-decreasing timestamps,
                starting at or after ``last_timestamp``.

        Returns:
            The buckets closed by this chunk, in order; possibly none.

        Raises:
            RuntimeError: The stream was already finalized.
            ValueError: Wrong dtype or shape, or a negative price or quantity.
            OutOfOrderError: Timestamps decrease within the chunk, or the
                chunk starts before the previous chunk's last record.
            OverflowError: Absorbing the chunk could make an int64 bucket sum
                wrap (``contract.check_exact_accumulation_bound``).
        """
        if self._finished:
            raise RuntimeError(
                "the stream was finalized; create a new StreamingAggregator for more data"
            )
        timestamps = self._validated_timestamps(chunk)
        if chunk.size == 0:
            return self._emit(self._emitted)

        price = chunk["price"].astype(np.int64)
        quantity = chunk["quantity"].astype(np.int64)
        min_price, min_quantity = int(price.min()), int(quantity.min())
        if min_price < 0 or min_quantity < 0:
            raise ValueError(
                f"price and quantity must be non-negative ticks, got min price {min_price}, "
                f"min quantity {min_quantity}"
            )

        inside = (timestamps >= self._start_us) & (timestamps < self._end_us)
        bucket = ((timestamps[inside] - self._start_us) // self._width_us).astype(np.int64)
        added_counts = np.bincount(bucket, minlength=self._n_buckets).astype(np.int64)

        # The bound is checked first, on the state as it would be after this
        # chunk, so a rejected chunk leaves nothing half-applied.
        max_price = max(self._max_price, int(price.max()))
        max_quantity = max(self._max_quantity, int(quantity.max()))
        check_exact_accumulation_bound(
            int((self._count + added_counts).max()), max_price, max_quantity
        )

        in_quantity = quantity[inside]
        self._count += added_counts
        np.add.at(self._total, bucket, in_quantity)
        np.add.at(self._signed, bucket, in_quantity * taker_sign(chunk["buyer_is_maker"][inside]))
        np.add.at(self._numerator, bucket, price[inside] * in_quantity)
        self._max_price, self._max_quantity = max_price, max_quantity
        self._last_timestamp = int(timestamps[-1])
        self._records_seen += int(chunk.size)

        # Buckets whose upper edge is at or before the newest timestamp can
        # receive nothing more. edges[1:] holds the n_buckets upper edges, so
        # the count is capped at n_buckets automatically.
        closed = int(np.searchsorted(self._edges[1:], self._last_timestamp, side="right"))
        return self._emit(closed)

    def finalize(self) -> Features:
        """End the stream and return every bucket not yet emitted.

        Trailing buckets that never saw a record at or beyond their upper
        edge are emitted as they stand: empty, or partially filled by the
        last records of the stream. Idempotent: a second call returns zero
        buckets.
        """
        self._finished = True
        return self._emit(self._n_buckets)

    # --- Internals ----------------------------------------------------------

    def _validated_timestamps(self, chunk: np.ndarray) -> np.ndarray:
        """Check the chunk's dtype, shape and ordering; return its contiguous timestamps."""
        if not isinstance(chunk, np.ndarray) or chunk.dtype != TRADE_DTYPE:
            got = chunk.dtype if isinstance(chunk, np.ndarray) else type(chunk).__name__
            raise ValueError(f"chunk must be a TRADE_DTYPE array, got {got}")
        if chunk.ndim != 1:
            raise ValueError(f"chunk must be 1-D, got shape {chunk.shape}")
        timestamps = np.ascontiguousarray(chunk["timestamp"])
        if timestamps.size == 0:
            return timestamps
        decreasing = np.flatnonzero(np.diff(timestamps) < 0)
        if decreasing.size:
            first = int(decreasing[0])
            raise OutOfOrderError(
                f"timestamps must be non-decreasing within a chunk; record {first + 1} "
                f"({int(timestamps[first + 1])}) precedes record {first} "
                f"({int(timestamps[first])}). Sort at ingest, not here."
            )
        if self._last_timestamp is not None and int(timestamps[0]) < self._last_timestamp:
            raise OutOfOrderError(
                f"chunk starts at {int(timestamps[0])} but the previous chunk ended at "
                f"{self._last_timestamp}; chunks must arrive in timestamp order. "
                "The chunk was not absorbed."
            )
        return timestamps

    def _emit(self, upto: int) -> Features:
        """Return buckets ``[emitted, upto)`` as copies and advance the emitted mark."""
        lo, hi = self._emitted, max(self._emitted, upto)
        self._emitted = hi
        count = self._count[lo:hi].copy()
        total = self._total[lo:hi].copy()
        numerator = self._numerator[lo:hi]
        vwap = np.full(hi - lo, VWAP_WHEN_NO_QUANTITY, dtype=np.float64)
        weighted = total > 0
        # Same operation order as reference.aggregate_reference: the only
        # floating-point step, applied once per bucket, never to partials.
        vwap[weighted] = numerator[weighted].astype(np.float64) / total[weighted] / PRICE_SCALE
        return Features(
            event_count=count,
            total_quantity=total,
            signed_quantity=self._signed[lo:hi].copy(),
            vwap=vwap,
            valid=count > 0,
        )


def concat_features(parts: Sequence[Features]) -> Features:
    """Join emitted bucket runs, in order, into one ``Features`` covering all buckets.

    An empty sequence yields zero-length arrays of the contract dtypes.
    """
    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return Features(
            event_count=empty,
            total_quantity=empty,
            signed_quantity=empty,
            vwap=np.zeros(0, dtype=np.float64),
            valid=np.zeros(0, dtype=np.bool_),
        )
    return Features(
        event_count=np.concatenate([p.event_count for p in parts]),
        total_quantity=np.concatenate([p.total_quantity for p in parts]),
        signed_quantity=np.concatenate([p.signed_quantity for p in parts]),
        vwap=np.concatenate([p.vwap for p in parts]),
        valid=np.concatenate([p.valid for p in parts]),
    )


def aggregate_stream(
    chunks: Iterable[np.ndarray],
    start_us: int,
    n_buckets: int,
    width_us: int = DEFAULT_BUCKET_WIDTH_US,
) -> Features:
    """Run a whole chunked stream through ``StreamingAggregator`` and finalize.

    Intended for tests and offline replay. The result has ``n_buckets`` rows
    and equals ``reference.aggregate_reference`` on the concatenated chunks
    with ``windows.build_offsets(..., start_us, n_buckets, width_us)``.
    """
    aggregator = StreamingAggregator(start_us, n_buckets, width_us)
    parts = [aggregator.push(chunk) for chunk in chunks]
    parts.append(aggregator.finalize())
    return concat_features(parts)
