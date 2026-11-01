"""Shared tape and spec builders for the execution simulator tests.

Every tape is written as literal per-bucket rows ``(total_quantity, vwap_ticks)``
so a test reads like the table it checks. ``None`` is an empty (invalid)
bucket. The default tape and spec reproduce the hand-verified reference
fixture whose fills, fees and shortfalls are asserted exactly in
``test_oracles``.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence

import numpy as np

from tapeforge.execution.spec import TERMINAL_RULE_RESIDUAL_PEROLD, EpisodeSpec
from tapeforge.execution.state import MarketSlice

Row = tuple[int, int] | None

# Reference-fixture conventions: arrival price 10,000 ticks, 10 bps taker fee with
# ceiling rounding, 10 % participation cap, zero data delay and zero latency.
# Execution VWAPs are 10,002 / 10,005 / 10,001.
P0 = 10_000  # Arrival price, in price ticks.
EXEC_VWAPS = (10_002, 10_005, 10_001)
DEEP = 2_000_000  # 20 BTC per bucket: the 10 % cap (200,000) never binds for TWAP.
Q_STAR = 300_000  # Target quantity, 3.00000 BTC in quantity ticks.
P_T = 10_010  # Terminal mark, in price ticks.


def slice_from_rows(before: Sequence[Row], after: Sequence[Row]) -> MarketSlice:
    """Build a slice with ``before`` as buckets ``-len(before) .. -1`` and ``after`` as ``0 ..``.

    Every non-empty row is one record with no taker imbalance; a row
    ``(0, 0)`` is a valid bucket whose only record had zero quantity.

    Args:
        before: Rows for the history buckets, oldest first.
        after: Rows for bucket 0 onwards.

    Returns:
        A ``MarketSlice`` with ``origin == len(before)``.
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
    """Build the reference tape: an arrival bucket at ``P0``, the execution buckets, then ``after``.

    Args:
        totals: Total quantity per execution bucket, in quantity ticks.
        vwaps: VWAP per execution bucket, in price ticks.
        after: Rows appended after the execution buckets; the default is one
            deep terminal bucket at ``P_T``.
    """
    return slice_from_rows([(DEEP, P0)], [*zip(totals, vwaps, strict=True), *after])


def worksheet_spec(side: int = 1, **overrides: int | str) -> EpisodeSpec:
    """Build the reference spec: target 300,000, horizon 3, arrival lookback 1.

    The terminal rule is the spec default (``terminal_settlement`` with a zero
    surcharge); ``risk_lambda`` keeps its zero default, which switches the
    risk term off. ``overrides`` replace any field.
    """
    params: dict[str, int | str] = {
        "side": side,
        "target_qty_ticks": Q_STAR,
        "horizon_buckets": 3,
        "arrival_lookback": 1,
    }
    params.update(overrides)
    return EpisodeSpec(**params)  # type: ignore[arg-type]


def legacy_spec(side: int = 1, **overrides: int | str) -> EpisodeSpec:
    """Build the reference spec under the legacy ``residual_perold`` rule (``residual_mu`` 0)."""
    return worksheet_spec(side, terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD, **overrides)


def frozen_bytes(objects: Sequence[object]) -> bytes:
    """Return a canonical pickle of ``objects`` for byte-level identity checks."""
    return pickle.dumps(list(objects), protocol=pickle.HIGHEST_PROTOCOL)
