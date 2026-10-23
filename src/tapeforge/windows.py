"""Bucket window index: boundary offsets, no padding.

For ``n_buckets`` fixed-width time buckets over a timestamp-sorted trade
array, ``build_offsets`` returns an ``offsets`` array of length
``n_buckets + 1`` such that bucket ``b`` owns records
``[offsets[b], offsets[b + 1])``. Empty buckets are a repeated offset, so
nothing is padded to the longest bucket.

    offsets = [0, 2, 2, 7]
      bucket 0 -> records 0..1
      bucket 1 -> empty
      bucket 2 -> records 2..6

Cost is one ``np.searchsorted`` of the ``n_buckets + 1`` edges into the
timestamp column: O((n_buckets + 1) log n), no per-record Python work.
"""

from __future__ import annotations

import numpy as np


def bucket_edges(start_us: int, n_buckets: int, width_us: int) -> np.ndarray:
    """Return the ``n_buckets + 1`` bucket boundary timestamps (int64 microseconds).

    Edge ``b`` opens bucket ``b``; the last edge closes the window. Buckets
    are half-open: a record exactly on an edge belongs to the bucket the edge
    opens.

    Raises:
        ValueError: on a non-positive bucket count or width.
    """
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")
    if width_us <= 0:
        raise ValueError(f"width_us must be positive, got {width_us}")
    return start_us + np.arange(n_buckets + 1, dtype=np.int64) * np.int64(width_us)


def build_offsets(
    timestamps: np.ndarray,
    start_us: int,
    n_buckets: int,
    width_us: int,
) -> np.ndarray:
    """Return int64 offsets of shape ``(n_buckets + 1,)``.

    Preconditions are validated rather than assumed: ``timestamps`` must be a
    1-D int64 array in non-decreasing order. Out-of-order input would produce
    silently wrong offsets, so it is an error instead.

    Raises:
        ValueError: if ``timestamps`` is not 1-D int64 and non-decreasing, or
            if the bucket geometry is degenerate.
    """
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {timestamps.shape}")
    if timestamps.dtype != np.int64:
        raise ValueError(
            f"timestamps must be int64 in {np.datetime_data(np.dtype('M8[us]'))[0]}, "
            f"got {timestamps.dtype}"
        )
    if timestamps.size and np.any(np.diff(timestamps) < 0):
        first = int(np.flatnonzero(np.diff(timestamps) < 0)[0])
        raise ValueError(
            f"timestamps must be non-decreasing; record {first + 1} "
            f"({int(timestamps[first + 1])}) precedes record {first} "
            f"({int(timestamps[first])}). Sort at ingest, not here."
        )

    edges = bucket_edges(start_us, n_buckets, width_us)
    return np.searchsorted(timestamps, edges, side="left").astype(np.int64)


def bucket_lengths(offsets: np.ndarray) -> np.ndarray:
    """Records per bucket, shape ``(n_buckets,)``. Zero for empty buckets.

    The distribution of this array is what decides the one-pass vs two-pass
    kernel dispatch threshold, so it is a first-class output rather than
    something each caller recomputes.
    """
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"offsets must be 1-D with at least 2 entries, got {offsets.shape}")
    return np.diff(offsets).astype(np.int64)
