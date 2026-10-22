"""The streaming CPU aggregator must equal the batch reference for any chunking.

Oracles: hand-calculated values, ``reference.aggregate_reference_naive`` (a
plain loop) and ``reference.aggregate_reference`` (the batch path). Parity is
asserted bitwise for every output, including ``vwap``, because the streaming
path performs the one floating-point division in the same operation order.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tapeforge import contract, generate, reference, streaming, windows
from tapeforge.streaming import OutOfOrderError, StreamingAggregator, aggregate_stream

W = contract.DEFAULT_BUCKET_WIDTH_US
P = contract.PRICE_SCALE
Q = contract.QUANTITY_SCALE
EMPTY = np.zeros(0, dtype=contract.TRADE_DTYPE)


def _trades(rows: list[tuple[int, float, float, bool]]) -> np.ndarray:
    """TRADE_DTYPE array from (second, price, quantity, buyer_is_maker)."""
    out = np.zeros(len(rows), dtype=contract.TRADE_DTYPE)
    for i, (second, price, qty, bim) in enumerate(rows):
        out[i] = (second * contract.US_PER_SECOND, round(price * P), round(qty * Q), bim, i)
    return out


def _batch(
    trades: np.ndarray, start_us: int, n_buckets: int, width_us: int = W
) -> reference.Features:
    offsets = windows.build_offsets(trades["timestamp"], start_us, n_buckets, width_us)
    return reference.aggregate_reference(trades, offsets)


def _assert_bitwise(got: reference.Features, expected: reference.Features) -> None:
    for name in contract.FEATURE_NAMES:
        actual, target = getattr(got, name), getattr(expected, name)
        assert actual.dtype == target.dtype, (name, actual.dtype, target.dtype)
        np.testing.assert_array_equal(actual, target, err_msg=name, strict=True)


def _chunks(trades: np.ndarray, cuts: list[int]) -> list[np.ndarray]:
    return [np.ascontiguousarray(part) for part in np.split(trades, sorted(cuts))]


# --- hand-calculated semantics -------------------------------------------


def test_prd_toy_example_in_one_chunk() -> None:
    trades = _trades([(1, 100.0, 2.0, False), (1, 102.0, 1.0, True)])
    got = aggregate_stream([trades], contract.US_PER_SECOND, 1)
    assert got.event_count.tolist() == [2]
    assert got.total_quantity.tolist() == [3 * Q]
    assert got.signed_quantity.tolist() == [1 * Q]
    assert got.vwap[0] == pytest.approx((100 * 2 + 102) / 3, rel=contract.VWAP_RTOL)
    assert got.valid.tolist() == [True]


def test_prd_toy_example_one_record_per_chunk() -> None:
    trades = _trades([(1, 100.0, 2.0, False), (1, 102.0, 1.0, True)])
    per_event = [trades[i : i + 1] for i in range(trades.size)]
    _assert_bitwise(aggregate_stream(per_event, contract.US_PER_SECOND, 1),
                    _batch(trades, contract.US_PER_SECOND, 1))


def test_bucket_spanning_three_chunks_is_emitted_once_complete() -> None:
    """State carries across chunk boundaries; the bucket closes only on a later timestamp."""
    trades = _trades([(3, 100.0 + i, 1.0, i % 2 == 1) for i in range(100)])
    agg = StreamingAggregator(0, 5)

    first = agg.push(trades[:30])
    # Seeing second 3 proves seconds 0-2 are over: three empty buckets close.
    assert first.event_count.tolist() == [0, 0, 0]
    assert not first.valid.any()
    assert agg.emitted_buckets == 3

    assert agg.push(trades[30:60]).event_count.size == 0
    assert agg.push(trades[60:]).event_count.size == 0
    assert agg.emitted_buckets == 3

    tail = agg.finalize()
    assert tail.event_count.tolist() == [100, 0]
    assert tail.total_quantity.tolist() == [100 * Q, 0]
    assert tail.signed_quantity.tolist() == [0, 0]
    assert tail.vwap[0] == pytest.approx(float(np.mean([100.0 + i for i in range(100)])),
                                         rel=contract.VWAP_RTOL)
    assert tail.valid.tolist() == [True, False]
    assert agg.is_finished


def test_a_record_on_the_upper_edge_closes_the_bucket_and_belongs_to_the_next() -> None:
    agg = StreamingAggregator(0, 3)
    first = _trades([(0, 100.0, 1.0, False)])
    assert agg.push(first).event_count.size == 0  # bucket 0 still open

    on_edge = np.zeros(1, dtype=contract.TRADE_DTYPE)
    on_edge["timestamp"], on_edge["price"], on_edge["quantity"] = W, 200 * P, Q
    closed = agg.push(on_edge)
    assert closed.event_count.tolist() == [1]
    assert closed.vwap.tolist() == [100.0]

    rest = agg.finalize()
    assert rest.event_count.tolist() == [1, 0]
    assert rest.vwap.tolist() == [200.0, 0.0]


def test_records_before_the_window_are_ignored_but_still_ordered() -> None:
    trades = _trades([(0, 500.0, 9.0, False), (1, 600.0, 9.0, False), (5, 100.0, 1.0, False)])
    got = aggregate_stream([trades[:2], trades[2:]], 5 * contract.US_PER_SECOND, 1)
    assert got.event_count.tolist() == [1]
    assert got.total_quantity.tolist() == [1 * Q]
    assert got.vwap.tolist() == [100.0]


def test_a_record_past_the_window_closes_every_bucket() -> None:
    trades = _trades([(0, 100.0, 1.0, False), (9, 999.0, 50.0, False)])
    agg = StreamingAggregator(0, 2)
    assert agg.push(trades[:1]).event_count.size == 0
    closed = agg.push(trades[1:])
    assert closed.event_count.tolist() == [1, 0]
    assert closed.total_quantity.sum() == 1 * Q
    assert agg.emitted_buckets == 2
    assert agg.finalize().event_count.size == 0


def test_empty_chunks_change_nothing() -> None:
    trades = _trades([(0, 100.0, 1.0, False), (2, 200.0, 2.0, True)])
    got = aggregate_stream([EMPTY, trades[:1], EMPTY, EMPTY, trades[1:], EMPTY], 0, 3)
    _assert_bitwise(got, _batch(trades, 0, 3))


def test_no_records_at_all_gives_all_empty_buckets() -> None:
    got = aggregate_stream([], 0, 4)
    assert got.event_count.tolist() == [0, 0, 0, 0]
    assert not got.valid.any()
    assert got.vwap.tolist() == [0.0] * 4
    _assert_bitwise(got, _batch(EMPTY, 0, 4))


def test_finalize_is_idempotent_and_push_after_it_raises() -> None:
    agg = StreamingAggregator(0, 2)
    agg.push(_trades([(0, 100.0, 1.0, False)]))
    assert agg.finalize().event_count.tolist() == [1, 0]
    assert agg.finalize().event_count.size == 0
    with pytest.raises(RuntimeError, match="finalized"):
        agg.push(_trades([(1, 100.0, 1.0, False)]))


def test_zero_quantity_records_count_but_do_not_move_vwap() -> None:
    trades = _trades([(0, 100.0, 2.0, False), (0, 999.0, 0.0, True), (1, 101.0, 0.0, True)])
    got = aggregate_stream([trades[:1], trades[1:2], trades[2:]], 0, 2)
    assert got.event_count.tolist() == [2, 1]
    assert got.valid.tolist() == [True, True]
    assert got.vwap.tolist() == pytest.approx([100.0, 0.0])
    assert not np.isnan(got.vwap).any()


def test_averaging_trap_vwap_comes_from_integer_partials_not_chunk_means() -> None:
    """Chunk A: 1 unit at 100. Chunk B: 3 units at 200. VWAP is 175, not mean(100, 200) = 150."""
    chunk_a = _trades([(0, 100.0, 1.0, False)])
    chunk_b = _trades([(0, 200.0, 1.0, False)] * 3)
    got = aggregate_stream([chunk_a, chunk_b], 0, 1)
    assert got.vwap[0] == 175.0
    assert got.vwap[0] != 150.0
    _assert_bitwise(got, _batch(np.concatenate([chunk_a, chunk_b]), 0, 1))


# --- ordering is enforced, never repaired --------------------------------


def test_decreasing_timestamps_within_a_chunk_are_rejected() -> None:
    bad = _trades([(0, 100.0, 1.0, False), (5, 100.0, 1.0, False), (3, 100.0, 1.0, False)])
    with pytest.raises(OutOfOrderError, match=r"record 2 \(3000000\) precedes record 1"):
        StreamingAggregator(0, 10).push(bad)


def test_a_chunk_starting_before_the_previous_one_ended_is_rejected_atomically() -> None:
    first = _trades([(0, 100.0, 1.0, False), (4, 100.0, 1.0, False)])
    late = _trades([(2, 100.0, 1.0, False), (6, 100.0, 1.0, False)])
    ok = _trades([(4, 100.0, 1.0, False), (7, 100.0, 1.0, False)])

    agg = StreamingAggregator(0, 10)
    parts = [agg.push(first)]
    with pytest.raises(OutOfOrderError, match="chunks must arrive in timestamp order"):
        agg.push(late)
    assert agg.records_seen == 2
    assert agg.last_timestamp == 4 * contract.US_PER_SECOND

    # The rejected chunk left no trace: the stream continues and matches the
    # batch result over the chunks that were actually accepted.
    parts.append(agg.push(ok))
    parts.append(agg.finalize())
    got = streaming.concat_features(parts)
    assert got.event_count.sum() == 4
    _assert_bitwise(got, _batch(np.concatenate([first, ok]), 0, 10))


def test_equal_timestamps_across_a_chunk_boundary_are_fine() -> None:
    trades = _trades([(2, 100.0, 1.0, False)] * 6)
    _assert_bitwise(aggregate_stream([trades[:4], trades[4:]], 0, 4), _batch(trades, 0, 4))


# --- input validation ------------------------------------------------------


def test_wrong_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="TRADE_DTYPE"):
        StreamingAggregator(0, 1).push(np.zeros(3))


def test_two_dimensional_chunk_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        StreamingAggregator(0, 1).push(np.zeros((2, 2), dtype=contract.TRADE_DTYPE))


def test_negative_price_or_quantity_is_rejected() -> None:
    bad = _trades([(0, 100.0, 1.0, False)])
    bad["quantity"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        StreamingAggregator(0, 1).push(bad)


def test_degenerate_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets must be positive"):
        StreamingAggregator(0, 0)
    with pytest.raises(ValueError, match="width_us must be positive"):
        StreamingAggregator(0, 1, 0)


# --- exact integer sums and the overflow bound -----------------------------


def test_sums_just_inside_the_int64_bound_combine_exactly_across_chunks() -> None:
    n = 64
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30)
    trades["buyer_is_maker"] = np.arange(n) % 2 == 0
    assert n * 2**30 * int(trades["quantity"][0]) <= contract.INT64_MAX

    got = aggregate_stream(_chunks(trades, [7, 8, 40, 63]), 0, 1)
    assert int(got.total_quantity[0]) == int(trades["quantity"].sum())
    assert int(got.signed_quantity[0]) == 0
    _assert_bitwise(got, _batch(trades, 0, 1))


def test_a_chunk_that_could_overflow_is_rejected_before_any_state_changes() -> None:
    n = 4
    price, qty = 2**30, contract.INT64_MAX // (n * 2**30) + 1
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"], trades["quantity"] = price, qty

    agg = StreamingAggregator(0, 1)
    agg.push(trades[:2])  # two records: bound is 2 * price * qty, still inside int64
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        agg.push(trades[2:])  # a fourth record would push the bound over
    assert agg.records_seen == 2
    assert agg.finalize().total_quantity.tolist() == [2 * qty]


def test_overflow_bound_uses_the_longest_bucket_not_the_record_count() -> None:
    n = 1_000
    price, qty = 2**30, contract.INT64_MAX // (2 * 2**30)
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = np.arange(n, dtype=np.int64) // 2 * W  # 500 buckets of 2 records
    trades["price"], trades["quantity"] = price, qty
    got = aggregate_stream(_chunks(trades, [333, 666]), 0, 500)
    assert (got.total_quantity == 2 * qty).all()


def test_out_of_window_records_still_count_towards_the_bound() -> None:
    """A huge value after the window would make the batch validation raise; so does the stream."""
    n = 4
    inside = np.zeros(n, dtype=contract.TRADE_DTYPE)
    inside["price"], inside["quantity"] = 2**30, contract.INT64_MAX // (n * 2**30)
    after = np.zeros(1, dtype=contract.TRADE_DTYPE)
    after["timestamp"], after["price"] = 50 * W, 2**30
    after["quantity"] = contract.INT64_MAX // 2**30
    agg = StreamingAggregator(0, 1)
    agg.push(inside)
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        agg.push(after)


# --- parity with the batch reference ---------------------------------------


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fixed_chunkings_match_batch(profile: str, seed: int) -> None:
    n_buckets = 40
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile=profile)
    expected = _batch(trades, 0, n_buckets)
    n = trades.size
    chunkings = {
        "single": [],
        "halves": [n // 2],
        "per_record": list(range(1, n)),
        "ragged": [1, 2, 3, n // 3, n // 3 + 1, n - 1],
    }
    for name, cuts in chunkings.items():
        got = aggregate_stream(_chunks(trades, cuts), 0, n_buckets)
        assert got.event_count.size == n_buckets, name
        _assert_bitwise(got, expected)


def test_per_record_chunks_match_batch_on_the_real_fixture() -> None:
    from pathlib import Path

    from tapeforge import ingest

    fixture = Path(__file__).parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"
    trades, _ = ingest.load_agg_trades(fixture)
    start = int(trades["timestamp"][0]) // contract.US_PER_SECOND * contract.US_PER_SECOND
    per_event = [trades[i : i + 1] for i in range(trades.size)]
    got = aggregate_stream(per_event, start, 4)
    _assert_bitwise(got, _batch(trades, start, 4))
    _assert_bitwise(
        got, reference.aggregate_reference_naive(
            trades, windows.build_offsets(trades["timestamp"], start, 4, W)
        )
    )


def test_clipped_window_with_records_on_both_sides_matches_batch() -> None:
    trades = generate.generate_trades(seed=9, n_buckets=30, profile="uniform")
    start, n_buckets = 7 * W, 12  # buckets 0-6 precede the window, 19-29 follow it
    expected = _batch(trades, start, n_buckets)
    assert expected.event_count.sum() < trades.size
    got = aggregate_stream(_chunks(trades, [5, 400, 401, 900]), start, n_buckets)
    _assert_bitwise(got, expected)


# --- properties: any chunking, any window --------------------------------

_profiles = st.sampled_from(["sparse", "uniform", "burst"])


@given(
    profile=_profiles,
    seed=st.integers(min_value=0, max_value=10_000),
    n_buckets=st.integers(min_value=1, max_value=25),
    cuts=st.lists(st.integers(min_value=0, max_value=5_000), max_size=12),
    start_shift=st.integers(min_value=-2, max_value=3),
    window=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=250, deadline=None)
def test_any_chunking_of_generated_trades_matches_batch(
    profile: str, seed: int, n_buckets: int, cuts: list[int], start_shift: int, window: int
) -> None:
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile=profile)
    chunks = _chunks(trades, [min(c, trades.size) for c in cuts])
    start = start_shift * W
    _assert_bitwise(aggregate_stream(chunks, start, window), _batch(trades, start, window))


@given(
    seconds=st.lists(st.integers(min_value=-3, max_value=12), min_size=0, max_size=60),
    prices=st.lists(st.integers(min_value=0, max_value=10_000 * P), min_size=60, max_size=60),
    quantities=st.lists(st.integers(min_value=0, max_value=100 * Q), min_size=60, max_size=60),
    flags=st.lists(st.booleans(), min_size=60, max_size=60),
    cuts=st.lists(st.integers(min_value=0, max_value=60), max_size=8),
    n_buckets=st.integers(min_value=1, max_value=10),
    width_s=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=250, deadline=None)
def test_any_chunking_of_arbitrary_ordered_records_matches_batch(
    seconds: list[int],
    prices: list[int],
    quantities: list[int],
    flags: list[bool],
    cuts: list[int],
    n_buckets: int,
    width_s: int,
) -> None:
    n = len(seconds)
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = np.sort(np.array(seconds, dtype=np.int64)) * contract.US_PER_SECOND
    trades["price"] = prices[:n]
    trades["quantity"] = quantities[:n]
    trades["buyer_is_maker"] = flags[:n]
    width = width_s * contract.US_PER_SECOND
    chunks = _chunks(trades, [min(c, n) for c in cuts])
    _assert_bitwise(
        aggregate_stream(chunks, 0, n_buckets, width), _batch(trades, 0, n_buckets, width)
    )


@pytest.mark.slow
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    cuts=st.lists(st.integers(min_value=0, max_value=20_000), max_size=40),
)
@settings(max_examples=1_000, deadline=None)
def test_one_thousand_random_chunkings_of_a_burst_day_match_batch(
    seed: int, cuts: list[int]
) -> None:
    trades = generate.generate_trades(seed=seed, n_buckets=200, profile="burst")
    chunks = _chunks(trades, [min(c, trades.size) for c in cuts])
    _assert_bitwise(aggregate_stream(chunks, 0, 200), _batch(trades, 0, 200))
