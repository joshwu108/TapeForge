"""Readable NumPy/CPU reference features: the correctness oracle.

Deliberately simple and slow. Every GPU result is compared against this,
so clarity beats speed here -- do not optimize this module.

Features per bucket (PRD section 3): event_count, total_quantity,
signed_quantity, vwap, valid. Units and the taker-side sign convention come
from ``tapeforge.contract``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Features:
    """Per-bucket features. All arrays have shape ``(n_buckets,)``.

    Frozen: features are returned as new objects, never mutated in place.
    """

    event_count: np.ndarray      # int64, exact
    total_quantity: np.ndarray   # int64 quantity ticks, exact
    signed_quantity: np.ndarray  # int64 quantity ticks, exact, may be negative
    vwap: np.ndarray             # float64, undefined where not valid
    valid: np.ndarray            # bool, False for empty buckets


def taker_sign(buyer_is_maker: np.ndarray) -> np.ndarray:
    """Map the raw ``buyer_is_maker`` flag to +1 buy-initiated / -1 sell-initiated."""
    raise NotImplementedError("week 2: implement after tests/test_reference.py is RED")


def aggregate_reference(trades: np.ndarray, offsets: np.ndarray) -> Features:
    """Compute per-bucket features from a sorted trade array and window offsets.

    ``trades`` is a structured array of ``contract.TRADE_DTYPE``;
    ``offsets`` comes from ``tapeforge.windows.build_offsets``.
    """
    raise NotImplementedError("week 2: implement after tests/test_reference.py is RED")
