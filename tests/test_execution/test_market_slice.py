"""MarketSlice construction from the feature engine and its exact VWAP quantisation."""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate, reference, windows
from tapeforge.execution.state import MarketSlice, vwap_ticks_from_numerators
from test_execution.tapes import slice_from_rows

I64 = np.int64


def _i64(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=I64)


@pytest.mark.parametrize(
    "numerator, total, expected",
    [
        (30_006, 3, 10_002),   # exact
        (30_007, 3, 10_002),   # 10,002.33 rounds down
        (30_008, 3, 10_003),   # 10,002.67 rounds up
        (20_005, 2, 10_003),   # 10,002.5: half rounds up
        (0, 5, 0),             # zero-quantity numerator
        (0, 0, 0),             # no quantity at all
    ],
)
def test_vwap_ticks_round_half_up_in_integers(numerator: int, total: int, expected: int) -> None:
    got = vwap_ticks_from_numerators(_i64([numerator]), _i64([total]))
    assert got.dtype == I64
    assert got.tolist() == [expected]


def test_vwap_ticks_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        vwap_ticks_from_numerators(_i64([1, 2]), _i64([1]))
    with pytest.raises(ValueError, match="int64"):
        vwap_ticks_from_numerators(np.array([1.0]), _i64([1]))
    with pytest.raises(ValueError, match="non-negative"):
        vwap_ticks_from_numerators(_i64([-1]), _i64([1]))


@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
def test_from_features_agrees_with_the_float_vwap_to_half_a_tick(profile: str) -> None:
    trades = generate.generate_trades(seed=11, n_buckets=60, profile=profile)
    offsets = windows.build_offsets(trades["timestamp"], 0, 60, contract.DEFAULT_BUCKET_WIDTH_US)
    features = reference.aggregate_reference(trades, offsets)
    numerators = reference.vwap_numerators(trades, offsets)
    market = MarketSlice.from_features(features, numerators, origin=10)

    assert market.n_buckets == 60 and market.n_after == 50
    np.testing.assert_array_equal(market.event_count, features.event_count)
    np.testing.assert_array_equal(market.total_quantity, features.total_quantity)
    np.testing.assert_array_equal(market.signed_quantity, features.signed_quantity)
    np.testing.assert_array_equal(market.valid, features.valid)
    distance = np.abs(market.vwap_ticks - features.vwap * contract.PRICE_SCALE)
    assert np.all(distance <= 0.5 + 1e-6)
    assert np.all(market.vwap_ticks[~market.valid] == 0)


def test_vwap_numerators_match_the_naive_loop() -> None:
    trades = generate.generate_trades(seed=3, n_buckets=20, profile="burst")
    offsets = windows.build_offsets(trades["timestamp"], 0, 20, contract.DEFAULT_BUCKET_WIDTH_US)
    got = reference.vwap_numerators(trades, offsets)
    expected = [
        sum(int(p) * int(q) for p, q in zip(
            trades["price"][offsets[b] : offsets[b + 1]],
            trades["quantity"][offsets[b] : offsets[b + 1]], strict=True,
        ))
        for b in range(20)
    ]
    assert got.dtype == I64
    assert got.tolist() == expected


def test_arrays_are_copied_and_read_only() -> None:
    source = _i64([5, 0])
    market = slice_from_rows([(5, 100)], [None])
    market2 = MarketSlice(
        event_count=_i64([1, 0]), total_quantity=source, signed_quantity=_i64([0, 0]),
        vwap_ticks=_i64([100, 0]), valid=np.array([True, False]), origin=1,
    )
    source[0] = 999
    assert market2.total_quantity.tolist() == [5, 0]
    with pytest.raises(ValueError, match="read-only"):
        market.total_quantity[0] = 1
    assert market.vwap(-1) == 1.0  # 100 ticks is 1.00 quote units


def test_indexing_is_relative_to_origin() -> None:
    market = slice_from_rows([(1, 10), (2, 20)], [(3, 30), None, (5, 50)])
    assert market.origin == 2 and market.n_after == 3
    assert market.index(-2) == 0 and market.index(0) == 2 and market.index(2) == 4
    assert market.has_bucket(-2) and market.has_bucket(2)
    assert not market.has_bucket(-3) and not market.has_bucket(3)
    with pytest.raises(IndexError, match="outside the slice"):
        market.index(3)


def test_with_bucket_replaces_one_bucket_and_recomputes_valid() -> None:
    market = slice_from_rows([(1, 10)], [(3, 30), None])
    emptied = market.with_bucket(0, event_count=0, total_quantity=0, vwap_ticks=0)
    assert emptied.valid.tolist() == [True, False, False]
    assert market.valid.tolist() == [True, True, False]  # the original is untouched
    filled = market.with_bucket(1, event_count=2, total_quantity=7, vwap_ticks=70)
    assert filled.valid.tolist() == [True, True, True]
    assert filled.total_quantity.tolist() == [1, 3, 7]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"total_quantity": _i64([1])}, "has 1 buckets"),
        ({"valid": np.array([1, 0])}, "bool array"),
        ({"origin": 3}, "origin must be in"),
        ({"origin": 1.5}, "origin must be an integer"),
        ({"vwap_ticks": _i64([-1, 0])}, "non-negative"),
        ({"valid": np.array([True, True])}, "valid must be exactly"),
        ({"vwap_ticks": _i64([10, 3])}, "invalid buckets must carry zero"),
        ({"signed_quantity": _i64([6, 0])}, "cannot exceed total_quantity"),
        ({"event_count": _i64([1, 1]), "valid": np.array([True, True]),
          "vwap_ticks": _i64([10, 3])}, "no quantity must have vwap_ticks == 0"),
    ],
)
def test_inconsistent_slices_are_rejected(kwargs: dict[str, object], match: str) -> None:
    params: dict[str, object] = {
        "event_count": _i64([1, 0]),
        "total_quantity": _i64([5, 0]),
        "signed_quantity": _i64([0, 0]),
        "vwap_ticks": _i64([10, 0]),
        "valid": np.array([True, False]),
        "origin": 1,
    }
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=match):
        MarketSlice(**params)  # type: ignore[arg-type]


def test_two_dimensional_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        MarketSlice(
            event_count=np.zeros((2, 2), dtype=I64), total_quantity=_i64([0, 0]),
            signed_quantity=_i64([0, 0]), vwap_ticks=_i64([0, 0]),
            valid=np.array([False, False]), origin=0,
        )
