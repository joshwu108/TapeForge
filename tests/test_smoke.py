"""Week 0: the repository itself is wired correctly."""

from __future__ import annotations

import numpy as np

import tapeforge
from tapeforge import contract


def test_package_imports_with_version() -> None:
    assert tapeforge.__version__


def test_every_module_imports() -> None:
    """A stub must still import: the harness depends on it."""
    for name in (
        "contract",
        "generate",
        "ingest",
        "kernels",
        "model",
        "ops",
        "reference",
        "windows",
    ):
        __import__(f"tapeforge.{name}")


def test_sign_convention_is_explicit_and_opposed() -> None:
    assert contract.SIGN_BUY_INITIATED == 1
    assert contract.SIGN_SELL_INITIATED == -1


def test_trade_dtype_carries_the_documented_fields() -> None:
    assert contract.TRADE_DTYPE.names == (
        "timestamp",
        "price",
        "quantity",
        "buyer_is_maker",
        "agg_trade_id",
    )
    assert contract.TRADE_DTYPE["timestamp"] == np.int64
    assert contract.TRADE_DTYPE["price"] == np.int64
    assert contract.TRADE_DTYPE["quantity"] == np.int64


def test_scaled_integer_products_fit_in_int64() -> None:
    """Overflow bound of the VWAP numerator, stated as a test (PRD section 6)."""
    max_i64 = np.iinfo(np.int64).max
    # One bucket's numerator is sum(price_ticks * quantity_ticks).
    worst_single_product = contract.PRICE_SCALE * contract.QUANTITY_SCALE
    assert worst_single_product < max_i64
    # Headroom left for accumulation across a bucket, in units of worst-case rows.
    assert max_i64 // worst_single_product >= 1
