"""The PyTorch-facing API (TF-07): input formats, dispatch, and CPU correctness.

Every correctness check here compares against an independent oracle --
hand-calculated values or ``reference.aggregate_reference_naive`` (a plain
Python loop) -- never against the vectorized reference that the CPU path
itself calls. GPU dispatch is covered in test_gpu_parity.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from conftest import TOY_EXPECTED

from tapeforge import contract, generate, ops, reference, windows

torch = pytest.importorskip("torch")

Q = contract.QUANTITY_SCALE
P = contract.PRICE_SCALE
W = contract.DEFAULT_BUCKET_WIDTH_US

DTYPES = {
    "event_count": torch.int64,
    "total_quantity": torch.int64,
    "signed_quantity": torch.int64,
    "vwap": torch.float64,
    "valid": torch.bool,
}


def _trades(rows: list[tuple[int, float, float, bool]]) -> np.ndarray:
    """TRADE_DTYPE array from (second, price, quantity, buyer_is_maker)."""
    out = np.zeros(len(rows), dtype=contract.TRADE_DTYPE)
    for i, (second, price, qty, bim) in enumerate(rows):
        out[i] = (second * contract.US_PER_SECOND, round(price * P), round(qty * Q), bim, i)
    return out


def _as_numpy(trades: np.ndarray, offsets: np.ndarray) -> tuple[Any, Any]:
    return trades, offsets


def _as_tensors(trades: np.ndarray, offsets: np.ndarray) -> tuple[Any, Any]:
    columns = {name: torch.from_numpy(trades[name].copy()) for name in trades.dtype.names}
    return columns, torch.from_numpy(offsets.copy())


FORMATS = pytest.mark.parametrize("to_format", [_as_numpy, _as_tensors], ids=["numpy", "tensor"])
Formatter = Callable[[np.ndarray, np.ndarray], tuple[Any, Any]]


def _run(to_format: Formatter, trades: np.ndarray, offsets: np.ndarray) -> dict[str, Any]:
    return ops.aggregate(*to_format(trades, offsets))


def _assert_matches(got: dict[str, Any], expected: reference.Features) -> None:
    for name in ("event_count", "total_quantity", "signed_quantity", "valid"):
        np.testing.assert_array_equal(got[name].numpy(), getattr(expected, name), err_msg=name)
    np.testing.assert_allclose(
        got["vwap"].numpy(), expected.vwap, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL
    )


# --- output contract ------------------------------------------------------


@FORMATS
def test_returns_documented_keys_shapes_and_dtypes(to_format: Formatter) -> None:
    trades = generate.generate_trades(seed=4, n_buckets=7, profile="uniform")
    offsets = windows.build_offsets(trades["timestamp"], 0, 7, W)
    got = _run(to_format, trades, offsets)

    assert tuple(got) == contract.FEATURE_NAMES
    for name, dtype in DTYPES.items():
        assert isinstance(got[name], torch.Tensor), name
        assert got[name].dtype == dtype, name
        assert got[name].shape == (7,), name
        assert got[name].device.type == "cpu", name


@FORMATS
def test_prd_toy_example(to_format: Formatter) -> None:
    trades = _trades([(1, 100.0, 2.0, False), (1, 102.0, 1.0, True)])
    offsets = windows.build_offsets(trades["timestamp"], 1 * contract.US_PER_SECOND, 1, W)
    got = _run(to_format, trades, offsets)

    assert got["event_count"].item() == TOY_EXPECTED["event_count"]
    assert got["total_quantity"].item() == round(TOY_EXPECTED["total_quantity"] * Q)
    assert got["signed_quantity"].item() == round(TOY_EXPECTED["signed_quantity"] * Q)
    assert got["vwap"].item() == pytest.approx(
        TOY_EXPECTED["vwap"], rel=contract.VWAP_RTOL, abs=contract.VWAP_ATOL
    )
    assert got["valid"].item() is TOY_EXPECTED["valid"]


@FORMATS
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize("seed", [1, 2, 3, 11])
def test_matches_the_naive_loop_across_seeds_and_profiles(
    to_format: Formatter, profile: str, seed: int
) -> None:
    trades = generate.generate_trades(seed=seed, n_buckets=40, profile=profile)
    offsets = windows.build_offsets(trades["timestamp"], 0, 40, W)
    _assert_matches(
        _run(to_format, trades, offsets), reference.aggregate_reference_naive(trades, offsets)
    )


# --- edge cases -----------------------------------------------------------


@FORMATS
def test_empty_buckets_are_invalid_and_zeroed(to_format: Formatter) -> None:
    trades = _trades([(0, 100.0, 1.0, False), (3, 200.0, 2.0, True)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 4, W)
    got = _run(to_format, trades, offsets)

    assert got["event_count"].tolist() == [1, 0, 0, 1]
    assert got["valid"].tolist() == [True, False, False, True]
    assert got["total_quantity"].tolist() == [Q, 0, 0, 2 * Q]
    assert got["signed_quantity"].tolist() == [Q, 0, 0, -2 * Q]
    assert got["vwap"].tolist() == [100.0, 0.0, 0.0, 200.0]


@FORMATS
def test_no_records_at_all(to_format: Formatter) -> None:
    empty = np.zeros(0, dtype=contract.TRADE_DTYPE)
    offsets = windows.build_offsets(empty["timestamp"], 0, 3, W)
    got = _run(to_format, empty, offsets)

    assert got["event_count"].tolist() == [0, 0, 0]
    assert not got["valid"].any()
    assert got["vwap"].tolist() == [0.0, 0.0, 0.0]


@FORMATS
def test_records_outside_the_window_leave_every_bucket_empty(to_format: Formatter) -> None:
    trades = _trades([(0, 100.0, 1.0, False), (50, 100.0, 1.0, False)])
    offsets = windows.build_offsets(trades["timestamp"], 10 * contract.US_PER_SECOND, 5, W)
    assert offsets.tolist() == [1, 1, 1, 1, 1, 1]
    got = _run(to_format, trades, offsets)
    assert not got["valid"].any()
    assert got["total_quantity"].sum().item() == 0


@FORMATS
def test_zero_quantity_rows_count_but_give_zero_vwap(to_format: Formatter) -> None:
    trades = _trades([(0, 100.0, 0.0, False), (0, 105.0, 0.0, True), (1, 101.0, 0.0, True)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 2, W)
    got = _run(to_format, trades, offsets)

    assert got["event_count"].tolist() == [2, 1]
    assert got["valid"].tolist() == [True, True]
    assert got["total_quantity"].tolist() == [0, 0]
    assert got["signed_quantity"].tolist() == [0, 0]
    assert got["vwap"].tolist() == [0.0, 0.0]
    assert not torch.isnan(got["vwap"]).any()


@FORMATS
def test_zero_quantity_rows_do_not_move_vwap(to_format: Formatter) -> None:
    trades = _trades([(0, 100.0, 2.0, False), (0, 999.0, 0.0, True)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 1, W)
    got = _run(to_format, trades, offsets)
    assert got["event_count"].item() == 2
    assert got["vwap"].item() == pytest.approx(100.0, rel=contract.VWAP_RTOL)


@FORMATS
def test_sign_convention_follows_buyer_is_maker(to_format: Formatter) -> None:
    """buyer_is_maker=True: the seller hit the bid, so the trade is sell-initiated."""
    trades = _trades(
        [
            (0, 100.0, 1.0, True),   # all sell-initiated
            (0, 100.0, 2.0, True),
            (1, 100.0, 1.5, False),  # all buy-initiated
            (2, 100.0, 4.0, False),  # mixed: +4 - 1.25
            (2, 100.0, 1.25, True),
        ]
    )
    offsets = windows.build_offsets(trades["timestamp"], 0, 3, W)
    got = _run(to_format, trades, offsets)
    assert got["signed_quantity"].tolist() == [
        contract.SIGN_SELL_INITIATED * 3 * Q,
        contract.SIGN_BUY_INITIATED * round(1.5 * Q),
        round(2.75 * Q),
    ]


@FORMATS
def test_window_boundaries_are_half_open(to_format: Formatter) -> None:
    """A record exactly on an edge belongs to the later bucket: [start, end)."""
    trades = np.zeros(4, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = [W - 1, W, 2 * W - 1, 2 * W]
    trades["price"] = [100 * P, 200 * P, 300 * P, 400 * P]
    trades["quantity"] = Q
    trades["agg_trade_id"] = np.arange(4)
    offsets = windows.build_offsets(trades["timestamp"], 0, 2, W)
    got = _run(to_format, trades, offsets)

    assert offsets.tolist() == [0, 1, 3]  # the record at 2W is outside the window
    assert got["event_count"].tolist() == [1, 2]
    assert got["vwap"].tolist() == pytest.approx([100.0, 250.0], rel=contract.VWAP_RTOL)


@FORMATS
def test_large_single_bucket_is_exact_on_cpu(to_format: Formatter) -> None:
    """The CPU path has no bucket-size limit: 20,000 records in one second."""
    n = 20_000
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 100 * P + np.arange(n) % 7
    trades["quantity"] = 1 + np.arange(n) % 13
    trades["buyer_is_maker"] = np.arange(n) % 3 == 0
    offsets = np.array([0, n], dtype=np.int64)
    _assert_matches(
        _run(to_format, trades, offsets), reference.aggregate_reference_naive(trades, offsets)
    )


# --- overflow bound -------------------------------------------------------


@FORMATS
def test_sums_just_inside_the_int64_bound_are_exact(to_format: Formatter) -> None:
    n = 4
    price, qty = 2**30, (contract.INT64_MAX // (n * 2**30))
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"], trades["quantity"] = price, qty
    trades["buyer_is_maker"] = [True, False, False, False]
    got = _run(to_format, trades, np.array([0, n], dtype=np.int64))

    assert n * price * qty <= contract.INT64_MAX
    assert got["total_quantity"].item() == n * qty
    assert got["signed_quantity"].item() == 2 * qty
    assert got["vwap"].item() == pytest.approx(price / P, rel=contract.VWAP_RTOL)


@FORMATS
def test_sums_past_the_int64_bound_raise_instead_of_wrapping(to_format: Formatter) -> None:
    n = 4
    price, qty = 2**30, (contract.INT64_MAX // (n * 2**30)) + 1
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"], trades["quantity"] = price, qty
    with pytest.raises(OverflowError, match="int64 accumulation bound"):
        _run(to_format, trades, np.array([0, n], dtype=np.int64))


def test_overflow_bound_uses_the_longest_bucket_not_the_record_count() -> None:
    """Many short buckets with large values are fine; only per-bucket sums accumulate."""
    n = 1_000
    price, qty = 2**30, contract.INT64_MAX // (2 * 2**30)
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"], trades["quantity"] = price, qty
    offsets = np.arange(0, n + 1, 2, dtype=np.int64)  # 500 buckets of 2 records
    got = ops.aggregate(trades, offsets)
    assert (got["total_quantity"] == 2 * qty).all()


# --- input validation -----------------------------------------------------


def _cpu_inputs() -> tuple[dict[str, Any], Any]:
    trades = _trades([(0, 100.0, 1.0, False), (0, 101.0, 2.0, True), (1, 99.0, 1.0, False)])
    return _as_tensors(trades, np.array([0, 2, 3], dtype=np.int64))


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda c, o: c.pop("price"), ValueError, "missing columns"),
        (lambda c, o: c.update(price=c["price"].to(torch.int32)), ValueError, "torch.int64"),
        (lambda c, o: c.update(buyer_is_maker=c["buyer_is_maker"].long()), ValueError, "bool"),
        (lambda c, o: c.update(price=c["price"][None, :]), ValueError, "1-D"),
        (lambda c, o: c.update(quantity=c["quantity"][:2]), ValueError, "equal length"),
        (
            lambda c, o: c.update(price=torch.arange(6, dtype=torch.int64)[::2]),
            ValueError,
            "contiguous",
        ),
        (lambda c, o: c.update(price=c["price"].numpy()), TypeError, "torch.Tensor"),
        (lambda c, o: c.update(quantity=-c["quantity"]), ValueError, "non-negative"),
        (lambda c, o: c.update(price=-c["price"]), ValueError, "non-negative"),
    ],
    ids=[
        "missing",
        "int32-price",
        "int-flag",
        "2d",
        "ragged",
        "strided",
        "numpy-column",
        "negative-qty",
        "negative-price",
    ],
)
def test_invalid_columns_are_rejected(mutate: Any, error: type, match: str) -> None:
    columns, offsets = _cpu_inputs()
    mutate(columns, offsets)
    with pytest.raises(error, match=match):
        ops.aggregate(columns, offsets)


@pytest.mark.parametrize(
    ("offsets", "error", "match"),
    [
        ([0, 3, 2], ValueError, "non-decreasing"),
        ([0, 2, 4], ValueError, "but there are 3 records"),
        ([-1, 2, 3], ValueError, "but there are 3 records"),
        ([0], ValueError, "at least 2 entries"),
    ],
)
def test_invalid_offsets_are_rejected(offsets: list[int], error: type, match: str) -> None:
    columns, _ = _cpu_inputs()
    with pytest.raises(error, match=match):
        ops.aggregate(columns, torch.tensor(offsets, dtype=torch.int64))


def test_int32_offsets_are_rejected() -> None:
    columns, _ = _cpu_inputs()
    with pytest.raises(ValueError, match="offsets must be int64"):
        ops.aggregate(columns, torch.tensor([0, 2, 3], dtype=torch.int32))


def test_strided_offsets_are_rejected() -> None:
    columns, _ = _cpu_inputs()
    strided = torch.tensor([0, 9, 2, 9, 3], dtype=torch.int64)[::2]
    with pytest.raises(ValueError, match="offsets must be contiguous"):
        ops.aggregate(columns, strided)


def test_numpy_int32_offsets_are_rejected() -> None:
    trades = _trades([(0, 100.0, 1.0, False)])
    with pytest.raises(ValueError, match="offsets must be int64"):
        ops.aggregate(trades, np.array([0, 1], dtype=np.int32))


def test_wrong_structured_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="TRADE_DTYPE"):
        ops.aggregate(np.zeros(3), np.array([0, 3], dtype=np.int64))


def test_mixed_numpy_and_tensor_inputs_are_rejected() -> None:
    trades = _trades([(0, 100.0, 1.0, False)])
    with pytest.raises(TypeError, match="NumPy trades need NumPy offsets"):
        ops.aggregate(trades, torch.tensor([0, 1]))
    columns, _ = _as_tensors(trades, np.array([0, 1], dtype=np.int64))
    with pytest.raises(TypeError, match="tensor trades need tensor offsets"):
        ops.aggregate(columns, np.array([0, 1], dtype=np.int64))


def test_unsupported_trade_container_is_rejected() -> None:
    with pytest.raises(TypeError, match="structured array or a mapping"):
        ops.aggregate([(0, 1, 1, False, 0)], np.array([0, 1], dtype=np.int64))


def test_unsupported_device_is_rejected_not_silently_moved() -> None:
    columns, offsets = _cpu_inputs()
    meta = {name: col.to("meta") for name, col in columns.items()}
    with pytest.raises(ValueError, match="unsupported device"):
        ops.aggregate(meta, offsets.to("meta"))


def test_columns_on_another_device_than_offsets_are_rejected() -> None:
    columns, offsets = _cpu_inputs()
    columns["price"] = columns["price"].to("meta")
    with pytest.raises(ValueError, match="all inputs must share one device"):
        ops.aggregate(columns, offsets)


def test_extra_columns_are_ignored() -> None:
    columns, offsets = _cpu_inputs()
    columns["note"] = torch.zeros(1)
    assert ops.aggregate(columns, offsets)["event_count"].tolist() == [2, 1]
