"""The TapeForge data contract: dtypes, sign convention, scaling, tolerances."""

from __future__ import annotations
from typing import Final
import numpy as np

# internal time unit normalized
TIME_UNIT: Final[str] = "us"
US_PER_SECOND: Final[int] = 1_000_000

DEFAULT_BUCKET_WIDTH_US: Final[int] = 1 * US_PER_SECOND
PRICE_SCALE: Final[int] = 100          # 2 decimals: BTCUSDT tick size 0.01
QUANTITY_SCALE: Final[int] = 100_000   # 5 decimals: BTCUSDT step size 0.00001

# Exact worst-case bucket numerator measured on the canonical day
MEASURED_MAX_BUCKET_NUMERATOR: Final[int] = 95_214_009_975_192
SIGN_BUY_INITIATED: Final[int] = +1
SIGN_SELL_INITIATED: Final[int] = -1

VWAP_RTOL: Final[float] = 1e-9
VWAP_ATOL: Final[float] = 1e-12

# --- Canonical dtypes ----------------------------------------------------
TRADE_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("timestamp", np.int64),       # TIME_UNIT since epoch, non-decreasing
        ("price", np.int64),           # price * PRICE_SCALE
        ("quantity", np.int64),        # quantity * QUANTITY_SCALE
        ("buyer_is_maker", np.bool_),  # raw exchange flag, not the derived sign
        ("agg_trade_id", np.int64),    # for deterministic replay / dedup
    ]
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "event_count",
    "total_quantity",
    "signed_quantity",
    "vwap",
    "valid",
)
