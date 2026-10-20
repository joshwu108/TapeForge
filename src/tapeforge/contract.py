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
# DECISION D-003 (ACCEPTED 2026-10-19, from measurement -- see
# planning/experiments.md E-002). Scaled integers make every feature except
# VWAP bit-exact between CPU and GPU, and keep the VWAP numerator exact too,
# so VWAP's only floating-point operation is its final division.
#
# Chosen against real BTCUSDT 2025-01-02 (1,299,165 aggregate-trade records),
# where price uses 2 significant decimals and quantity 5:
#
#   worst single row  price_ticks * qty_ticks        = 1.824e13
#   worst bucket sum  sum over the busiest second    = 9.5214e13
#   int64 max                                        = 9.223e18   -> 96,869x
#
# The rejected 1e8/1e8 default overflowed int64 on a *single* row by ~2,000x.
#
# SYMBOL CONSTRAINT: these scales are exact for BTCUSDT-like instruments.
# A low-priced symbol (SHIBUSDT prices carry 8 decimals) cannot be
# represented, and ``ingest.decimal_strings_to_ticks`` raises rather than
# truncating -- the limitation fails loudly instead of corrupting features.
# Per-symbol scales from the exchange tick/step size are the documented next
# step if another symbol is ever needed.
PRICE_SCALE: Final[int] = 100          # 2 decimals: BTCUSDT tick size 0.01
QUANTITY_SCALE: Final[int] = 100_000   # 5 decimals: BTCUSDT step size 0.00001

# Exact worst-case bucket numerator measured on the canonical day, pinned by a
# test so a future scale change cannot silently reintroduce the overflow.
# Recompute with:
#   num = trades["price"] * trades["quantity"]
#   np.add.reduceat(num, build_offsets(...)[:-1]).max()
MEASURED_MAX_BUCKET_NUMERATOR: Final[int] = 95_214_009_975_192

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
