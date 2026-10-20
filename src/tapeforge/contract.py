"""The TapeForge data contract: dtypes, sign convention, scaling, tolerances.

Every number that crosses a module boundary is described here, so the CPU
reference and the GPU kernels cannot silently disagree about units.

PENDING STUDENT SIGN-OFF
------------------------
The values below are the defaults *derived from PRD section 3*. They encode
market-data semantics and precision policy, which the PRD assigns to the
student, not to an agent. Confirm or change each item marked ``DECISION``
and record the rationale in ``planning/decisions.md`` before week 2 work
depends on it.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# --- Timestamps ----------------------------------------------------------
# DECISION: canonical internal time unit.
# Binance spot aggTrades archives are microseconds from 2025-01-01 onward and
# milliseconds before that (PRD section 3). Ingest normalizes to one unit.
TIME_UNIT: Final[str] = "us"
US_PER_SECOND: Final[int] = 1_000_000

# DECISION: default bucket width. 1 second per the PRD worked example.
DEFAULT_BUCKET_WIDTH_US: Final[int] = 1 * US_PER_SECOND

# --- Scaled-integer price / quantity -------------------------------------
# DECISION: scale factors. Scaled integers make count and signed-sum results
# bit-exact between CPU and GPU; only VWAP needs a float tolerance.
# Overflow bound must be stated here once the factors are fixed:
#   max_total_quantity_ticks * PRICE_SCALE must fit in int64.
PRICE_SCALE: Final[int] = 100_000_000       # 1e8 ticks per unit price
QUANTITY_SCALE: Final[int] = 100_000_000    # 1e8 ticks per unit quantity

# --- Taker-side sign convention ------------------------------------------
# DECISION (PRD section 3): derive the taker side from ``buyer_is_maker``.
# buyer_is_maker == True  -> the buyer rested on the book, the *seller* was the
#                            aggressor -> sell-initiated -> sign -1
# buyer_is_maker == False -> the buyer was the aggressor -> buy-initiated -> +1
# ``signed_quantity`` is buy-initiated quantity minus sell-initiated quantity.
SIGN_BUY_INITIATED: Final[int] = +1
SIGN_SELL_INITIATED: Final[int] = -1

# --- Numerical tolerance for VWAP ----------------------------------------
# DECISION: CPU/GPU parity tolerance for the one floating-point feature.
# Counts and signed sums are compared exactly; VWAP uses these bounds.
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
