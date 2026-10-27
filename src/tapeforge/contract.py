"""Data contract shared by every TapeForge module: dtypes, units, signs, scales, tolerances.

Every other module takes its units from here and defines none of its own.
Timestamps are int64 microseconds since the Unix epoch. Prices and quantities
are int64 ticks: the decimal value multiplied by ``PRICE_SCALE`` or
``QUANTITY_SCALE``, so bucket sums are exact integers. The scales are those
of BTCUSDT (2 and 5 decimals) and are not configurable per symbol; an
instrument with more decimals is rejected at ingest rather than truncated.

Data contract:
    The taker sign is derived from the exchange's ``buyer_is_maker`` flag. A
    record whose buyer was the maker was sell-initiated and carries
    ``SIGN_SELL_INITIATED``; every other record carries ``SIGN_BUY_INITIATED``.

Invariants:
    ``check_exact_accumulation_bound`` is the single int64 overflow rule
    shared by the CPU reference, the GPU kernels and the streaming
    aggregator, so all three accept and reject the same inputs.

Not covered:
    No features are computed here. The reference implementation lives in
    ``tapeforge.reference``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Internal time unit after ingest normalisation. Archives arrive in milliseconds
# or microseconds; every module downstream of ingest sees microseconds.
TIME_UNIT: Final[str] = "us"
US_PER_SECOND: Final[int] = 1_000_000  # Microseconds per second.

DEFAULT_BUCKET_WIDTH_US: Final[int] = 1 * US_PER_SECOND  # One-second buckets.
PRICE_SCALE: Final[int] = 100          # Ticks per price unit: 2 decimals, BTCUSDT tick 0.01.
QUANTITY_SCALE: Final[int] = 100_000   # Ticks per quantity unit: 5 decimals, step 0.00001.

# Largest sum(price_ticks * quantity_ticks) observed in any one-second bucket of
# the reference day (BTCUSDT spot 2025-01-02). Documents the headroom the chosen
# scales leave inside int64; the runtime bound enforced at validation time is
# ``check_exact_accumulation_bound``.
MEASURED_MAX_BUCKET_NUMERATOR: Final[int] = 95_214_009_975_192
SIGN_BUY_INITIATED: Final[int] = +1    # Taker sign when the buyer was the aggressor.
SIGN_SELL_INITIATED: Final[int] = -1   # Taker sign when the buyer was the maker.

INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)  # Upper bound for exact sums.

# VWAP is the only floating-point feature. Implementations are compared with
# ``numpy.testing.assert_allclose`` at these tolerances; the integer features
# must match exactly.
VWAP_RTOL: Final[float] = 1e-9   # Relative tolerance for VWAP comparisons.
VWAP_ATOL: Final[float] = 1e-12  # Absolute tolerance for VWAP comparisons.

# --- Canonical dtypes ----------------------------------------------------
TRADE_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("timestamp", np.int64),       # TIME_UNIT since the epoch; non-decreasing.
        ("price", np.int64),           # Price ticks: price * PRICE_SCALE.
        ("quantity", np.int64),        # Quantity ticks: quantity * QUANTITY_SCALE.
        ("buyer_is_maker", np.bool_),  # Raw exchange flag, not the derived sign.
        ("agg_trade_id", np.int64),    # Exchange id, for deterministic replay and dedup.
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
    """Raise ``OverflowError`` if any int64 bucket sum could wrap around.

    The bound is sufficient, not tight: every bucket's VWAP numerator is at
    most the longest bucket times the largest price and the largest quantity
    anywhere in the input, and that product also bounds the quantity sums.
    The check runs on Python integers, so it cannot overflow itself.

    Args:
        max_bucket_len: Records in the longest bucket.
        max_price_ticks: Largest price in the input, in price ticks.
        max_quantity_ticks: Largest quantity in the input, in quantity ticks.

    Raises:
        ValueError: One of the bounds is negative.
        OverflowError: The product exceeds ``INT64_MAX``. The message carries
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
