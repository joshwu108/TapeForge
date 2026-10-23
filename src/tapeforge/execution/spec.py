"""Episode specification for the reference execution simulator.

An ``EpisodeSpec`` fixes everything about an episode that is not market data:
side, target, horizon, fill rule parameters, latency, data delay, fees and the
two free reward coefficients. It is a frozen dataclass of Python integers so
that two specs compare equal field by field and a spec can be hashed into a
replay key.

Units (``tapeforge.contract``): quantities in quantity ticks (``QUANTITY_SCALE``
per unit), prices in price ticks (``PRICE_SCALE`` per quote unit), cash in cash
ticks (one price tick times one quantity tick). Rates that would naturally be
written as fractions (participation cap, fee, target fraction) are integer
basis points so that every cap, fee and target is computed with integer
arithmetic and the only rounding in the accounting is the fee ceiling.

This module does not know about market data; ``check_episode_bounds`` takes
the tape's maxima as arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final, Literal

import numpy as np

from tapeforge.contract import INT64_MAX, PRICE_SCALE, QUANTITY_SCALE

__all__ = [
    "ACCEPTED_FEE_BPS",
    "ACCEPTED_HORIZON_BUCKETS",
    "ACCEPTED_PARTICIPATION_CAP_BPS",
    "ACCEPTED_TARGET_FRACTIONS_BPS",
    "BPS_DENOMINATOR",
    "CASH_SCALE",
    "FILL_RULE_BUCKET_VWAP_CAPPED",
    "SIDE_BUY",
    "SIDE_SELL",
    "EpisodeSpec",
    "FillRule",
    "Side",
    "check_episode_bounds",
    "fee_ticks",
    "participation_cap_ticks",
]

Side = Literal[1, -1]
SIDE_BUY: Final[int] = 1    # trade toward a larger position: buy from zero to +Q*
SIDE_SELL: Final[int] = -1  # trade toward a smaller position: liquidate +Q* to zero

# Rates are integer basis points: 10_000 bps = 100 %.
BPS_DENOMINATOR: Final[int] = 10_000

# One cash tick is one price tick times one quantity tick (1e-7 quote units for
# BTCUSDT). Notionals are ``price_ticks * quantity_ticks`` with no rounding.
CASH_SCALE: Final[int] = PRICE_SCALE * QUANTITY_SCALE

# The accepted first configuration. Coefficients that are experiment sweeps
# rather than fixed assumptions (risk lambda, residual mu) have no accepted
# value and default to zero, which switches the corresponding term off.
ACCEPTED_HORIZON_BUCKETS: Final[int] = 300
ACCEPTED_PARTICIPATION_CAP_BPS: Final[int] = 1_000   # 10 % of a bucket's total quantity
ACCEPTED_FEE_BPS: Final[int] = 10   # taker fee; an assumed tier, not read from the exchange
ACCEPTED_TARGET_FRACTIONS_BPS: Final[tuple[int, ...]] = (200, 500, 1_000)  # rho

# Fill rules are named strings, not a plugin registry. A second rule (for
# example fill-at-touch once best bid/ask data exists) is added as a second
# literal and a second branch in ``simulator.step``.
FillRule = Literal["bucket_vwap_capped"]
FILL_RULE_BUCKET_VWAP_CAPPED: Final[str] = "bucket_vwap_capped"
_FILL_RULES: Final[frozenset[str]] = frozenset({FILL_RULE_BUCKET_VWAP_CAPPED})


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything about an episode except the tape.

    ``target_qty_ticks`` is Q*, the quantity to trade. A buy starts with zero
    inventory and trades toward +Q*; a sell starts with +Q* and trades toward
    zero. ``horizon_buckets`` is H, the number of decisions; decision ``t``
    (``0 <= t < H``) is taken at the boundary opening bucket ``t``, may read
    buckets ``<= t - 1 - data_delay_buckets`` and fills in bucket
    ``t + latency_buckets``.

    ``risk_lambda`` is charged per step in cash ticks per quantity tick of the
    quantity still unfilled after the step; ``residual_mu`` is charged once at
    the horizon in cash ticks per quantity tick of residual. Both are free
    coefficients with no accepted default; zero disables the term.
    ``arrival_lookback`` is L: how many buckets before the first visible bucket
    are searched for a valid arrival price, and how many buckets after the
    last fill bucket are searched for a valid terminal mark.
    """

    side: int
    target_qty_ticks: int
    horizon_buckets: int = ACCEPTED_HORIZON_BUCKETS
    participation_cap_bps: int = ACCEPTED_PARTICIPATION_CAP_BPS
    slippage_ticks: int = 0
    fee_bps: int = ACCEPTED_FEE_BPS
    latency_buckets: int = 0
    data_delay_buckets: int = 0
    risk_lambda: int = 0
    residual_mu: int = 0
    arrival_lookback: int = 60
    fill_rule: str = FILL_RULE_BUCKET_VWAP_CAPPED

    def __post_init__(self) -> None:
        """Normalize NumPy integers to ``int`` and reject anything out of domain.

        Raises:
            TypeError: a numeric field is not an integer (``bool`` and floats
                included), or ``fill_rule`` is not a string.
            ValueError: a field is outside its domain (see the messages).
        """
        for field in fields(self):
            if field.name == "fill_rule":
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{field.name} must be an integer, got {value!r}")
            object.__setattr__(self, field.name, int(value))
        if self.side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(
                f"side must be {SIDE_BUY} (buy) or {SIDE_SELL} (sell), got {self.side}"
            )
        if self.target_qty_ticks < 1:
            raise ValueError(
                f"target_qty_ticks must be at least one tick, got {self.target_qty_ticks}"
            )
        if self.horizon_buckets < 1:
            raise ValueError(f"horizon_buckets must be positive, got {self.horizon_buckets}")
        if not 0 <= self.participation_cap_bps <= BPS_DENOMINATOR:
            raise ValueError(
                f"participation_cap_bps must be in [0, {BPS_DENOMINATOR}], "
                f"got {self.participation_cap_bps}"
            )
        if self.arrival_lookback < 1:
            raise ValueError(f"arrival_lookback must be positive, got {self.arrival_lookback}")
        for name in (
            "slippage_ticks", "fee_bps", "latency_buckets", "data_delay_buckets",
            "risk_lambda", "residual_mu",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not isinstance(self.fill_rule, str):
            raise TypeError(f"fill_rule must be a string, got {type(self.fill_rule).__name__}")
        if self.fill_rule not in _FILL_RULES:
            raise ValueError(
                f"unknown fill_rule {self.fill_rule!r}; implemented: {sorted(_FILL_RULES)}"
            )

    @property
    def initial_inventory_ticks(self) -> int:
        """Zero for a buy; +Q* for a sell (liquidation to zero)."""
        return 0 if self.side == SIDE_BUY else self.target_qty_ticks

    @property
    def target_inventory_ticks(self) -> int:
        """+Q* for a buy; zero for a sell."""
        return self.target_qty_ticks if self.side == SIDE_BUY else 0


def participation_cap_ticks(total_quantity_ticks: int, cap_bps: int) -> int:
    """Largest fill the liquidity rule allows in a bucket: ``floor(cap * total)``.

    Integer arithmetic, so 10 % of 600,000 ticks is exactly 60,000 and never
    59,999 through a floating-point product.
    """
    return total_quantity_ticks * cap_bps // BPS_DENOMINATOR


def fee_ticks(notional_ticks: int, fee_bps: int) -> int:
    """Taker fee in cash ticks, rounded *up*: ``ceil(notional * fee_bps / 10_000)``.

    The ceiling is the only rounding in the simulator's accounting. It never
    under-charges, and it is what makes
    ``fee * 10_000 >= notional * fee_bps`` and
    ``fee * 10_000 < notional * fee_bps + 10_000`` exact test invariants.
    A notional of 5,002,500 at 10 bps is 5,002.5 and charges 5,003.
    """
    return -(-notional_ticks * fee_bps // BPS_DENOMINATOR)


def check_episode_bounds(
    spec: EpisodeSpec, max_price_ticks: int, max_total_quantity_ticks: int
) -> None:
    """Raise ``OverflowError`` if any int64 accumulator of the episode could wrap.

    ``max_price_ticks`` and ``max_total_quantity_ticks`` are the largest VWAP
    (after slippage) and bucket quantity anywhere in the tape the episode may
    fill against. The sufficient conditions, evaluated with Python integers:

    1. ``H * cap * max_price <= INT64_MAX``: cash, exec notional and the
       shortfall numerator cannot wrap, since no step fills more than ``cap``
       and no fill prices above ``max_price``.
    2. ``cap * max_price * fee_bps <= INT64_MAX``: the fee intermediate
       product cannot wrap.
    3. ``max(lambda, mu) * Q* <= INT64_MAX``: the risk and residual
       components, each at most a coefficient times the target.

    Raises:
        ValueError: a negative maximum.
        OverflowError: a condition fails; the message names it and the value.
    """
    if min(max_price_ticks, max_total_quantity_ticks) < 0:
        raise ValueError("price and quantity maxima must be non-negative")
    cap = participation_cap_ticks(max_total_quantity_ticks, spec.participation_cap_bps)
    worst_price = max_price_ticks + spec.slippage_ticks
    cash_bound = spec.horizon_buckets * cap * worst_price
    if cash_bound > INT64_MAX:
        raise OverflowError(
            f"int64 cash bound exceeded: horizon {spec.horizon_buckets} x cap {cap} ticks x "
            f"max price {worst_price} ticks = {cash_bound} > {INT64_MAX}"
        )
    fee_bound = cap * worst_price * spec.fee_bps
    if fee_bound > INT64_MAX:
        raise OverflowError(
            f"int64 fee bound exceeded: cap {cap} ticks x max price {worst_price} ticks x "
            f"fee {spec.fee_bps} bps = {fee_bound} > {INT64_MAX}"
        )
    coefficient_bound = max(spec.risk_lambda, spec.residual_mu) * spec.target_qty_ticks
    if coefficient_bound > INT64_MAX:
        raise OverflowError(
            f"int64 reward bound exceeded: max(lambda {spec.risk_lambda}, mu {spec.residual_mu}) "
            f"x target {spec.target_qty_ticks} ticks = {coefficient_bound} > {INT64_MAX}"
        )
