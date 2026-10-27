"""Tests for package import and the data contract.

Covers module imports (with and without torch/triton installed), the sign
convention, the trade dtype and the int64 overflow bound at measured BTCUSDT
magnitudes.
"""

from __future__ import annotations

import numpy as np

import tapeforge
from tapeforge import contract


def test_package_imports_with_version() -> None:
    assert tapeforge.__version__


def test_every_module_imports() -> None:
    """Every module imports even while it is a stub, because the harness imports all of them."""
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


def test_scaled_integer_products_fit_in_int64_at_real_magnitudes() -> None:
    """The int64 overflow bound holds at measured BTCUSDT magnitudes, not at unit values.

    Multiplying ``PRICE_SCALE`` by ``QUANTITY_SCALE`` only checks a price of 1.0
    and a quantity of 1.0, which passes for scales that overflow int64 on a
    single row of real BTCUSDT data. The assertions use measured extremes.
    """
    max_i64 = np.iinfo(np.int64).max

    # Measured on BTCUSDT 2025-01-02: the highest price and largest quantity.
    worst_price_ticks = round(97_839.50 * contract.PRICE_SCALE)
    worst_quantity_ticks = round(18.89042 * contract.QUANTITY_SCALE)
    worst_row = worst_price_ticks * worst_quantity_ticks
    assert worst_row < max_i64, "a single row overflows the VWAP numerator"

    # The busiest second of that day, scaled to the chosen factors.
    assert max_i64 > contract.MEASURED_MAX_BUCKET_NUMERATOR
    assert max_i64 // contract.MEASURED_MAX_BUCKET_NUMERATOR > 1_000, (
        "less than 1000x headroom on the measured worst bucket: too tight for "
        "a busier symbol or a wider bucket"
    )


def test_scales_match_the_precision_the_canonical_day_actually_uses() -> None:
    """BTCUSDT uses 2 price decimals and 5 quantity decimals, as measured on the canonical day."""
    assert contract.PRICE_SCALE == 10**2
    assert contract.QUANTITY_SCALE == 10**5


def test_package_imports_without_torch_or_triton() -> None:
    """Every module imports with torch and triton absent, as in CI; GPU code paths are guarded."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['triton'] = None\n"
        "import tapeforge.kernels as k, tapeforge.ops, tapeforge.reference, tapeforge.windows\n"
        "assert k.TRITON_AVAILABLE is False\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
