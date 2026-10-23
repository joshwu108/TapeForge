"""Shared tape builders for the execution simulator tests.

The worksheet's worked examples are the specification: every tape here is
written as literal per-bucket rows ``(total_quantity, vwap_ticks)`` so a test
reads like the table it checks. ``None`` is an empty (invalid) bucket.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence

import numpy as np

from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice

Row = tuple[int, int] | None

# Worksheet conventions: P_0 = 10,000 ticks, 10 bps taker fee, ceil rounding,
# 10 % participation cap, d = 0, l = 0. Execution VWAPs 10,002 / 10,005 / 10,001.
P0 = 10_000
EXEC_VWAPS = (10_002, 10_005, 10_001)
DEEP = 2_000_000  # 20 BTC per bucket: the 10 % cap (200,000) never binds for TWAP
Q_STAR = 300_000  # 3.00000 BTC
P_T = 10_010


def slice_from_rows(before: Sequence[Row], after: Sequence[Row]) -> MarketSlice:
    """Slice with ``before`` as buckets ``-len(before) .. -1`` and ``after`` as ``0 ..``.

    Every non-empty row is one record with no taker imbalance; a row
    ``(0, 0)`` is a valid bucket whose only record had zero quantity.
    """
    rows = list(before) + list(after)
    count = np.array([0 if r is None else 1 for r in rows], dtype=np.int64)
    total = np.array([0 if r is None else r[0] for r in rows], dtype=np.int64)
    vwap = np.array([0 if r is None else r[1] for r in rows], dtype=np.int64)
    return MarketSlice(
        event_count=count,
        total_quantity=total,
        signed_quantity=np.zeros(len(rows), dtype=np.int64),
        vwap_ticks=vwap,
        valid=count > 0,
        origin=len(before),
    )


def worksheet_tape(
    totals: Sequence[int] = (DEEP, DEEP, DEEP),
    vwaps: Sequence[int] = EXEC_VWAPS,
    after: Sequence[Row] = ((DEEP, P_T),),
) -> MarketSlice:
    """Arrival bucket at P_0, the three execution buckets, then the terminal bucket."""
    return slice_from_rows([(DEEP, P0)], [*zip(totals, vwaps, strict=True), *after])


def worksheet_spec(side: int = 1, **overrides: int) -> EpisodeSpec:
    """Example (a)/(b) spec: Q* = 300,000, H = 3, lookback 1, free coefficients off."""
    params: dict[str, int] = {
        "side": side,
        "target_qty_ticks": Q_STAR,
        "horizon_buckets": 3,
        "arrival_lookback": 1,
    }
    params.update(overrides)
    return EpisodeSpec(**params)


def frozen_bytes(objects: Sequence[object]) -> bytes:
    """Byte-level identity for determinism tests."""
    return pickle.dumps(list(objects), protocol=pickle.HIGHEST_PROTOCOL)
