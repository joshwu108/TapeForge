"""The TapeForge data contract: dtypes, units, sign convention, scaling, tolerances.

Every module reads its units from here. Timestamps are int64 microseconds
since the Unix epoch; prices and quantities are int64 *ticks*, the decimal
value multiplied by ``PRICE_SCALE`` / ``QUANTITY_SCALE``, so sums are exact
integers. The scales are those of BTCUSDT (2 and 5 decimals) and are not
per-symbol yet: an instrument with more decimals is rejected at ingest
rather than truncated.

The taker sign is derived from the exchange's ``buyer_is_maker`` flag: a
record where the buyer was the maker was sell-initiated and carries
``SIGN_SELL_INITIATED``. ``check_exact_accumulation_bound`` is the single
int64 overflow rule shared by the CPU reference, the kernels and the
streaming aggregator. Nothing in this module computes features.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Internal time unit after ingest normalisation. Archives arrive in ms or us;
# everything downstream sees us.
TIME_UNIT: Final[str] = "us"
US_PER_SECOND: Final[int] = 1_000_000

DEFAULT_BUCKET_WIDTH_US: Final[int] = 1 * US_PER_SECOND
PRICE_SCALE: Final[int] = 100          # 2 decimals: BTCUSDT tick size 0.01
QUANTITY_SCALE: Final[int] = 100_000   # 5 decimals: BTCUSDT step size 0.00001

# Largest sum(price_ticks * quantity_ticks) of any one-second bucket on the
# canonical day (BTCUSDT spot 2025-01-02). Documents how far inside int64 the
# chosen scales sit; the runtime bound below is what is enforced.
MEASURED_MAX_BUCKET_NUMERATOR: Final[int] = 95_214_009_975_192
SIGN_BUY_INITIATED: Final[int] = +1
SIGN_SELL_INITIATED: Final[int] = -1

INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)

# VWAP is the only floating-point feature. Implementations are compared with
# numpy.testing.assert_allclose at these tolerances; the integer features
# must match exactly.
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


def check_exact_accumulation_bound(
    max_bucket_len: int, max_price_ticks: int, max_quantity_ticks: int
) -> None:
    """Raise ``OverflowError`` if int64 bucket sums could wrap around.

    Sufficient, not tight: every bucket's VWAP numerator is bounded by the
    longest bucket times the largest price and quantity anywhere in the
    input, which also bounds the quantity sums. Evaluated with Python
    integers, so the check itself cannot overflow.

    Raises:
        ValueError: a negative bound.
        OverflowError: the product exceeds ``INT64_MAX``; the message carries
            the three factors.
    """
    if min(max_bucket_len, max_price_ticks, max_quantity_ticks) < 0:
        raise ValueError("bucket length, price and quantity bounds must be non-negative")
    worst = max_bucket_len * max_price_ticks * max_quantity_ticks
    if worst > INT64_MAX:
        raise OverflowError(
            f"int64 accumulation bound exceeded: longest bucket {max_bucket_len} x "
            f"max price {max_price_ticks} ticks x max quantity {max_quantity_ticks} ticks "
            f"= {worst} > {INT64_MAX}; results would not be exact"
        )
