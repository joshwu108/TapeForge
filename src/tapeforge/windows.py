"""Bucket window index: boundary offsets, no padding.

For ``n_buckets`` fixed-width time buckets over a timestamp-sorted trade
array, build an ``offsets`` array of length ``n_buckets + 1`` such that
bucket ``b`` owns records ``[offsets[b], offsets[b + 1])``. Empty buckets
are a repeated offset, so nothing is padded to the longest bucket.

    offsets = [0, 2, 2, 7]
      bucket 0 -> records 0..1
      bucket 1 -> empty
      bucket 2 -> records 2..6

Implementation plan (week 2, TDD): ``np.searchsorted`` of the bucket edges
into the sorted timestamp column gives the offsets in one vectorized pass.
"""

from __future__ import annotations

import numpy as np


def bucket_edges(
    start_us: int,
    n_buckets: int,
    width_us: int,
) -> np.ndarray:
    """Return the ``n_buckets + 1`` bucket boundary timestamps (int64)."""
    raise NotImplementedError("week 2: implement after tests/test_windows.py is RED")


def build_offsets(
    timestamps: np.ndarray,
    start_us: int,
    n_buckets: int,
    width_us: int,
) -> np.ndarray:
    """Return int64 offsets of shape ``(n_buckets + 1,)``.

    Preconditions (validated, not assumed): ``timestamps`` is int64,
    1-D, and non-decreasing. Records outside the window are excluded.
    """
    raise NotImplementedError("week 2: implement after tests/test_windows.py is RED")
