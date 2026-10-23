"""Package imports and data contract checks."""

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


def test_scaled_integer_products_fit_in_int64_at_real_magnitudes() -> None:
    """The contract's overflow bound, asserted against measured data, not unit values.

    The original version of this test multiplied PRICE_SCALE by
    QUANTITY_SCALE -- a price of 1.0 and a quantity of 1.0 -- and so passed
    happily while the committed scales overflowed int64 on a single row of
    real BTCUSDT data. See planning/experiments.md E-002.
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
    """BTCUSDT: 2 price decimals, 5 quantity decimals (measured)."""
    assert contract.PRICE_SCALE == 10**2
    assert contract.QUANTITY_SCALE == 10**5


def test_package_imports_without_torch_or_triton() -> None:
    """CI installs no torch; every module must still import (GPU code is guarded)."""
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
