"""CPU reference features checked against hand calculations and an independent loop."""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, reference, windows


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


def test_toy_example() -> None:
    """The hand-calculated toy example: 2 records, total 3, signed +1, vwap 100.666..."""
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


# --- the np.add.reduceat traps, pinned as regressions --------------------

def test_last_bucket_running_to_the_end_of_the_array_keeps_every_record() -> None:
    """Regression: clipping the final offset silently dropped one record.

    A bucket whose slice ends exactly at len(trades), with empty buckets
    after it, is the case where reduceat needs a sentinel rather than a
    clamped index.
    """
    trades = _trades([(3, 100.0, 1.0, False)] * 100)
    offsets = windows.build_offsets(trades["timestamp"], 0, 5, contract.DEFAULT_BUCKET_WIDTH_US)
    assert offsets.tolist() == [0, 0, 0, 0, 100, 100]
    f = reference.aggregate_reference(trades, offsets)
    assert f.event_count[3] == 100
    assert f.total_quantity[3] == 100 * contract.QUANTITY_SCALE


def test_empty_bucket_between_two_full_ones() -> None:
    trades = _trades([(0, 100.0, 1.0, False), (2, 200.0, 2.0, False)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 3, contract.DEFAULT_BUCKET_WIDTH_US)
    f = reference.aggregate_reference(trades, offsets)
    np.testing.assert_array_equal(f.event_count, [1, 0, 1])
    np.testing.assert_array_equal(f.valid, [True, False, True])
    assert f.vwap[1] == 0.0


def test_records_before_the_window_are_not_counted() -> None:
    """offsets[0] > 0: the reduction must start at the window, not at index 0."""
    trades = _trades([(0, 500.0, 9.0, False), (5, 100.0, 1.0, False)])
    offsets = windows.build_offsets(
        trades["timestamp"], 5 * contract.US_PER_SECOND, 1, contract.DEFAULT_BUCKET_WIDTH_US
    )
    assert offsets[0] == 1
    f = reference.aggregate_reference(trades, offsets)
    assert f.event_count[0] == 1
    assert f.total_quantity[0] == 1 * contract.QUANTITY_SCALE
    assert f.vwap[0] == pytest.approx(100.0)


def test_records_after_the_window_are_not_counted() -> None:
    """offsets[-1] < len(trades): the tail must not leak into the last bucket."""
    trades = _trades([(0, 100.0, 1.0, False), (9, 999.0, 50.0, False)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 2, contract.DEFAULT_BUCKET_WIDTH_US)
    assert offsets[-1] == 1
    f = reference.aggregate_reference(trades, offsets)
    assert f.total_quantity.sum() == 1 * contract.QUANTITY_SCALE
    assert f.vwap[0] == pytest.approx(100.0)


def test_every_bucket_empty_is_handled() -> None:
    trades = _trades([(50, 100.0, 1.0, False)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 5, contract.DEFAULT_BUCKET_WIDTH_US)
    f = reference.aggregate_reference(trades, offsets)
    assert not f.valid.any()
    assert f.total_quantity.sum() == 0


def test_no_records_at_all_is_handled() -> None:
    empty = np.zeros(0, dtype=contract.TRADE_DTYPE)
    offsets = windows.build_offsets(empty["timestamp"], 0, 3, contract.DEFAULT_BUCKET_WIDTH_US)
    f = reference.aggregate_reference(empty, offsets)
    assert f.event_count.shape == (3,)
    assert not f.valid.any()


# --- the vectorized implementation against the naive loop ----------------

def _assert_same(a: reference.Features, b: reference.Features) -> None:
    np.testing.assert_array_equal(a.event_count, b.event_count)
    np.testing.assert_array_equal(a.total_quantity, b.total_quantity)
    np.testing.assert_array_equal(a.signed_quantity, b.signed_quantity)
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_allclose(
        a.vwap, b.vwap, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL
    )


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vectorized_matches_the_naive_loop(profile: str, seed: int) -> None:
    from tapeforge import generate

    trades = generate.generate_trades(seed=seed, n_buckets=60, profile=profile)
    offsets = windows.build_offsets(trades["timestamp"], 0, 60, contract.DEFAULT_BUCKET_WIDTH_US)
    _assert_same(
        reference.aggregate_reference(trades, offsets),
        reference.aggregate_reference_naive(trades, offsets),
    )


def test_vectorized_matches_the_naive_loop_on_real_archive_rows() -> None:
    from pathlib import Path

    from tapeforge import ingest

    fixture = Path(__file__).parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"
    trades, _ = ingest.load_agg_trades(fixture)
    start = int(trades["timestamp"][0]) // contract.US_PER_SECOND * contract.US_PER_SECOND
    offsets = windows.build_offsets(trades["timestamp"], start, 4, contract.DEFAULT_BUCKET_WIDTH_US)
    _assert_same(
        reference.aggregate_reference(trades, offsets),
        reference.aggregate_reference_naive(trades, offsets),
    )


def test_hand_calculated_values_for_the_real_fixture() -> None:
    """Independent arithmetic on the committed 5-row fixture.

    bucket 0: 94591.78 x 2.0 (buy), 94591.78 x 0.5 (sell), 94591.79 x 1.25 (buy)
      count 3, total 3.75, signed 2.0 - 0.5 + 1.25 = 2.75
      vwap = (94591.78*2 + 94591.78*0.5 + 94591.79*1.25) / 3.75
    bucket 2: 94590.00 x 0.001 (sell), 94589.50 x 3.0 (buy)
      count 2, total 3.001, signed 3.0 - 0.001 = 2.999
    """
    from pathlib import Path

    from tapeforge import ingest

    fixture = Path(__file__).parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"
    trades, _ = ingest.load_agg_trades(fixture)
    start = int(trades["timestamp"][0]) // contract.US_PER_SECOND * contract.US_PER_SECOND
    offsets = windows.build_offsets(trades["timestamp"], start, 4, contract.DEFAULT_BUCKET_WIDTH_US)
    f = reference.aggregate_reference(trades, offsets)

    Q = contract.QUANTITY_SCALE
    np.testing.assert_array_equal(f.event_count, [3, 0, 2, 0])
    np.testing.assert_array_equal(f.valid, [True, False, True, False])
    assert f.total_quantity[0] == round(3.75 * Q)
    assert f.signed_quantity[0] == round(2.75 * Q)
    assert f.total_quantity[2] == round(3.001 * Q)
    assert f.signed_quantity[2] == round(2.999 * Q)
    expected_vwap0 = (94591.78 * 2 + 94591.78 * 0.5 + 94591.79 * 1.25) / 3.75
    assert f.vwap[0] == pytest.approx(expected_vwap0, rel=1e-12)
    expected_vwap2 = (94590.00 * 0.001 + 94589.50 * 3.0) / 3.001
    assert f.vwap[2] == pytest.approx(expected_vwap2, rel=1e-12)


# --- sign convention and validation --------------------------------------

def test_signed_quantity_never_exceeds_total_quantity_in_magnitude() -> None:
    from tapeforge import generate

    trades = generate.generate_trades(seed=31, n_buckets=100, profile="burst")
    offsets = windows.build_offsets(trades["timestamp"], 0, 100, contract.DEFAULT_BUCKET_WIDTH_US)
    f = reference.aggregate_reference(trades, offsets)
    assert np.all(np.abs(f.signed_quantity) <= f.total_quantity)


def test_taker_sign_rejects_a_non_boolean_column() -> None:
    with pytest.raises(ValueError, match="must be bool"):
        reference.taker_sign(np.array([0, 1]))


def test_wrong_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="TRADE_DTYPE"):
        reference.aggregate_reference(np.zeros(3), np.array([0, 3]))


def test_offsets_spanning_past_the_record_count_are_rejected() -> None:
    trades = _trades([(0, 100.0, 1.0, False)])
    with pytest.raises(ValueError, match="but there are 1 records"):
        reference.aggregate_reference(trades, np.array([0, 5], dtype=np.int64))


def test_decreasing_offsets_are_rejected() -> None:
    trades = _trades([(0, 100.0, 1.0, False)])
    with pytest.raises(ValueError, match="non-decreasing"):
        reference.aggregate_reference(trades, np.array([1, 0], dtype=np.int64))


def test_degenerate_offsets_are_rejected() -> None:
    trades = _trades([(0, 100.0, 1.0, False)])
    with pytest.raises(ValueError, match="at least 2 entries"):
        reference.aggregate_reference(trades, np.array([0], dtype=np.int64))
