"""Episode specification for the reference execution simulator.

``EpisodeSpec`` fixes everything about an episode that is not market data:
side, target quantity, horizon, fill-rule parameters, latency, data delay,
fees and the two experiment reward coefficients. It is a frozen dataclass of
Python integers, so two specs compare equal field by field and a spec can be
hashed into a replay key.

Data contract:
    Units follow ``tapeforge.contract``: quantities in quantity ticks
    (``QUANTITY_SCALE`` per unit), prices in price ticks (``PRICE_SCALE`` per
    quote unit), cash in cash ticks (one price tick times one quantity tick).
    Rates that are naturally fractions (participation cap, fee, target
    fraction) are integer basis points, so every cap, fee and target is
    computed with integer arithmetic and the only rounding in the accounting
    is the fee ceiling.

Not covered:
    Market data. ``check_episode_bounds`` takes the tape's maxima as
    arguments rather than reading a tape.
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
SIDE_BUY: Final[int] = 1    # Buy from zero inventory up to ``target_qty_ticks``.
SIDE_SELL: Final[int] = -1  # Liquidate ``target_qty_ticks`` of inventory down to zero.

# Rates are integer basis points: 10_000 bps = 100 %.
BPS_DENOMINATOR: Final[int] = 10_000

# One cash tick is one price tick times one quantity tick (1e-7 quote units for
# BTCUSDT). Notionals are ``price_ticks * quantity_ticks`` with no rounding.
CASH_SCALE: Final[int] = PRICE_SCALE * QUANTITY_SCALE

# Default production configuration. ``risk_lambda`` and ``residual_mu`` are
# experiment parameters with no production default; zero disables their
# reward terms.
ACCEPTED_HORIZON_BUCKETS: Final[int] = 300
ACCEPTED_PARTICIPATION_CAP_BPS: Final[int] = 1_000   # 10 % of a bucket's total quantity.
ACCEPTED_FEE_BPS: Final[int] = 10   # Taker fee; an assumed tier, not read from the exchange.
ACCEPTED_TARGET_FRACTIONS_BPS: Final[tuple[int, ...]] = (200, 500, 1_000)  # Target fractions.

# Fill rules are named strings, not a plugin registry. A second rule (for
# example fill-at-touch, once best bid/ask data exists) is added as a second
# literal and a second branch in ``simulator.step``.
FillRule = Literal["bucket_vwap_capped"]
FILL_RULE_BUCKET_VWAP_CAPPED: Final[str] = "bucket_vwap_capped"
_FILL_RULES: Final[frozenset[str]] = frozenset({FILL_RULE_BUCKET_VWAP_CAPPED})


@dataclass(frozen=True)
class EpisodeSpec:
    """Immutable configuration of one execution episode, excluding market data.

    Two specs with equal fields compare equal and hash identically, so a spec
    can serve as part of a replay key. All numeric fields are plain ``int``;
    NumPy integers are normalised in ``__post_init__``.

    Attributes:
        side: ``SIDE_BUY`` (1) or ``SIDE_SELL`` (-1).
        target_qty_ticks: Quantity to trade, in quantity ticks; at least 1. A
            buy starts with zero inventory and trades toward this amount; a
            sell starts holding this amount and trades toward zero.
        horizon_buckets: Number of decisions; positive. Decision ``t``
            (``0 <= t < horizon_buckets``) is taken at the boundary opening
            bucket ``t``, may read buckets ``<= t - 1 - data_delay_buckets``
            and fills in bucket ``t + latency_buckets``.
        participation_cap_bps: Largest fill in one bucket as basis points of
            that bucket's total quantity, in ``[0, BPS_DENOMINATOR]``.
        slippage_ticks: Price ticks added against the side to every fill
            price (``side * slippage_ticks``); non-negative.
        fee_bps: Taker fee in basis points of the fill notional, rounded up
            by ``fee_ticks``; non-negative.
        latency_buckets: Buckets between a decision and its fill bucket;
            non-negative.
        data_delay_buckets: Buckets withheld from the observation: decision
            ``t`` reads buckets ``<= t - 1 - data_delay_buckets``;
            non-negative.
        risk_lambda: Per-step charge in cash ticks per quantity tick of the
            quantity still unfilled after the step. Experiment parameter;
            zero disables the term.
        residual_mu: One-time charge at the horizon in cash ticks per
            quantity tick of residual inventory. Experiment parameter; zero
            disables the term.
        arrival_lookback: How many buckets before the first visible bucket
            are searched for a valid arrival price, and how many buckets
            after the last fill bucket are searched for a valid terminal
            mark; positive.
        fill_rule: Name of the fill rule; only
            ``FILL_RULE_BUCKET_VWAP_CAPPED`` is implemented.
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
        """Normalise NumPy integers to ``int`` and reject out-of-domain values.

        Raises:
            TypeError: A numeric field is not an integer (``bool`` and floats
                included), or ``fill_rule`` is not a string.
            ValueError: A field is outside its domain; the message names the
                field and the offending value.
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
        """Return the inventory at the start of the episode, in quantity ticks.

        Zero for a buy; ``target_qty_ticks`` for a sell (liquidation to zero).
        """
        return 0 if self.side == SIDE_BUY else self.target_qty_ticks

    @property
    def target_inventory_ticks(self) -> int:
        """Return the inventory the episode trades toward, in quantity ticks.

        ``target_qty_ticks`` for a buy; zero for a sell.
        """
        return self.target_qty_ticks if self.side == SIDE_BUY else 0


def participation_cap_ticks(total_quantity_ticks: int, cap_bps: int) -> int:
    """Return the largest fill the liquidity rule allows in one bucket.

    Computes ``floor(total_quantity_ticks * cap_bps / BPS_DENOMINATOR)`` in
    integer arithmetic, so 10 % of 600,000 ticks is exactly 60,000 and never
    59,999 through a floating-point product.

    Args:
        total_quantity_ticks: The bucket's total traded quantity, in quantity
            ticks.
        cap_bps: Participation cap in basis points.

    Returns:
        The cap in quantity ticks.
    """
    return total_quantity_ticks * cap_bps // BPS_DENOMINATOR


def fee_ticks(notional_ticks: int, fee_bps: int) -> int:
    """Return the taker fee in cash ticks, rounded up.

    Computes ``ceil(notional_ticks * fee_bps / BPS_DENOMINATOR)``. The ceiling
    is the only rounding in the simulator's accounting. It never
    under-charges, and it makes ``fee * 10_000 >= notional * fee_bps`` and
    ``fee * 10_000 < notional * fee_bps + 10_000`` exact invariants that the
    tests check. A notional of 5,002,500 at 10 bps is 5,002.5 and charges
    5,003.
    """
    return -(-notional_ticks * fee_bps // BPS_DENOMINATOR)


def check_episode_bounds(
    spec: EpisodeSpec, max_price_ticks: int, max_total_quantity_ticks: int
) -> None:
    """Raise ``OverflowError`` if any int64 accumulator of the episode could wrap.

    ``max_price_ticks`` and ``max_total_quantity_ticks`` are the largest VWAP
    and the largest bucket quantity anywhere in the tape the episode may fill
    against; ``slippage_ticks`` is added to the price here. With ``cap`` the
    participation cap of the largest bucket and ``worst_price`` the largest
    VWAP plus ``slippage_ticks``, the sufficient conditions, evaluated with
    Python integers, are:

    1. ``horizon_buckets * cap * worst_price <= INT64_MAX``: cash, executed
       notional and the shortfall numerator cannot wrap, since no step fills
       more than ``cap`` and no fill prices above ``worst_price``.
    2. ``cap * worst_price * fee_bps <= INT64_MAX``: the fee intermediate
       product cannot wrap.
    3. ``max(risk_lambda, residual_mu) * target_qty_ticks <= INT64_MAX``: the
       risk and residual components are each at most a coefficient times the
       target.

    Raises:
        ValueError: A maximum is negative.
        OverflowError: A condition fails; the message names it and the value.
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
