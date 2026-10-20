"""Week 2: CPU reference features against hand-calculated values.

Every test here is written before the implementation. They currently xfail
with NotImplementedError. When you start week 2, delete the
``pytestmark`` line, watch them fail for real (RED), then implement.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, reference, windows

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="week 2: delete this marker to begin the RED phase",
)


def _trades(rows: list[tuple[int, float, float, bool]]) -> np.ndarray:
    """Build a TRADE_DTYPE array from (second, price, quantity, buyer_is_maker)."""
    out = np.zeros(len(rows), dtype=contract.TRADE_DTYPE)
    for i, (second, price, qty, bim) in enumerate(rows):
        out[i] = (
            second * contract.US_PER_SECOND,
            round(price * contract.PRICE_SCALE),
            round(qty * contract.QUANTITY_SCALE),
            bim,
            i,
        )
    return out


def test_prd_toy_example() -> None:
    """PRD section 3: 2 records, total 3, signed +1, vwap 100.666..."""
    trades = _trades([(1, 100.0, 2.0, False), (1, 102.0, 1.0, True)])
    offsets = windows.build_offsets(
        trades["timestamp"],
        start_us=1 * contract.US_PER_SECOND,
        n_buckets=1,
        width_us=contract.DEFAULT_BUCKET_WIDTH_US,
    )
    f = reference.aggregate_reference(trades, offsets)

    assert f.event_count[0] == 2
    assert f.total_quantity[0] == 3 * contract.QUANTITY_SCALE
    assert f.signed_quantity[0] == 1 * contract.QUANTITY_SCALE
    assert f.vwap[0] == pytest.approx(
        (100 * 2 + 102 * 1) / 3, rel=contract.VWAP_RTOL, abs=contract.VWAP_ATOL
    )
    assert bool(f.valid[0]) is True


def test_empty_bucket_is_invalid_and_zeroed() -> None:
    trades = _trades([(0, 100.0, 1.0, False), (2, 100.0, 1.0, False)])
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 3, contract.DEFAULT_BUCKET_WIDTH_US
    )
    f = reference.aggregate_reference(trades, offsets)

    assert bool(f.valid[1]) is False
    assert f.event_count[1] == 0
    assert f.total_quantity[1] == 0
    assert f.signed_quantity[1] == 0


def test_taker_sign_follows_buyer_is_maker() -> None:
    """buyer_is_maker=True means the seller was the aggressor -> -1."""
    signs = reference.taker_sign(np.array([True, False]))
    assert signs[0] == contract.SIGN_SELL_INITIATED
    assert signs[1] == contract.SIGN_BUY_INITIATED


def test_all_sell_initiated_gives_negative_signed_quantity() -> None:
    trades = _trades([(0, 100.0, 1.0, True), (0, 101.0, 2.0, True)])
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 1, contract.DEFAULT_BUCKET_WIDTH_US
    )
    f = reference.aggregate_reference(trades, offsets)
    assert f.signed_quantity[0] == -3 * contract.QUANTITY_SCALE


def test_zero_quantity_rows_do_not_produce_nan_vwap() -> None:
    trades = _trades([(0, 100.0, 0.0, False)])
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 1, contract.DEFAULT_BUCKET_WIDTH_US
    )
    f = reference.aggregate_reference(trades, offsets)
    assert f.event_count[0] == 1
    assert not np.isnan(f.vwap[0])


def test_identical_timestamps_stay_in_one_bucket() -> None:
    trades = _trades([(5, 100.0, 1.0, False)] * 100)
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 10, contract.DEFAULT_BUCKET_WIDTH_US
    )
    f = reference.aggregate_reference(trades, offsets)
    assert f.event_count[5] == 100
    assert f.event_count.sum() == 100


def test_burst_of_100_records_matches_closed_form() -> None:
    rows = [(3, 100.0 + i, 1.0, i % 2 == 1) for i in range(100)]
    trades = _trades(rows)
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 5, contract.DEFAULT_BUCKET_WIDTH_US
    )
    f = reference.aggregate_reference(trades, offsets)

    assert f.event_count[3] == 100
    assert f.total_quantity[3] == 100 * contract.QUANTITY_SCALE
    assert f.signed_quantity[3] == 0  # 50 buys, 50 sells, equal size
    expected_vwap = float(np.mean([100.0 + i for i in range(100)]))
    assert f.vwap[3] == pytest.approx(expected_vwap, rel=contract.VWAP_RTOL)


# Out-of-order input is rejected by windows.build_offsets, which owns that
# precondition; see test_windows.py::test_non_monotonic_input_names_the_
# offending_record. The reference never sees unsorted records.
