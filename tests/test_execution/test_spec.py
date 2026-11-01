"""Tests for tapeforge.execution.spec: EpisodeSpec defaults, validation and normalisation, the fee
ceiling, the participation-cap floor, the settlement price rounding and the int64 bounds check.

Rounding and bound cases are pinned with exact integers; no fixtures are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract
from tapeforge.execution.spec import (
    ACCEPTED_FEE_BPS,
    ACCEPTED_HORIZON_BUCKETS,
    ACCEPTED_PARTICIPATION_CAP_BPS,
    BPS_DENOMINATOR,
    CASH_SCALE,
    TERMINAL_RULE_RESIDUAL_PEROLD,
    TERMINAL_RULE_SETTLEMENT,
    EpisodeSpec,
    check_episode_bounds,
    fee_ticks,
    participation_cap_ticks,
    settlement_price_ticks,
)


def test_defaults_are_the_accepted_configuration() -> None:
    spec = EpisodeSpec(side=1, target_qty_ticks=300_000)
    assert spec.horizon_buckets == ACCEPTED_HORIZON_BUCKETS == 300
    assert spec.participation_cap_bps == ACCEPTED_PARTICIPATION_CAP_BPS == 1_000
    assert spec.fee_bps == ACCEPTED_FEE_BPS == 10
    assert spec.slippage_ticks == 0
    assert spec.latency_buckets == 0 and spec.data_delay_buckets == 0
    assert spec.risk_lambda == 0 and spec.residual_mu == 0
    assert spec.fill_rule == "bucket_vwap_capped"
    # The primary terminal rule is the default; the surcharge has no accepted value and is 0.
    assert spec.terminal_rule == TERMINAL_RULE_SETTLEMENT == "terminal_settlement"
    assert spec.terminal_surcharge_bps == 0 and spec.settles_at_horizon
    assert CASH_SCALE == contract.PRICE_SCALE * contract.QUANTITY_SCALE == 10_000_000


def test_terminal_rules_are_explicit_and_their_parameters_exclusive() -> None:
    legacy = EpisodeSpec(side=1, target_qty_ticks=1, terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD,
                         residual_mu=20)
    assert not legacy.settles_at_horizon and legacy.residual_mu == 20
    settling = EpisodeSpec(side=1, target_qty_ticks=1, terminal_surcharge_bps=25)
    assert settling.settles_at_horizon and settling.terminal_surcharge_bps == 25
    # residual_mu belongs to the legacy rule; a settlement spec must not carry it silently.
    with pytest.raises(ValueError, match="residual_mu 5 is a parameter of the legacy"):
        EpisodeSpec(side=1, target_qty_ticks=1, residual_mu=5)
    # The surcharge belongs to settlement; the legacy rule settles nothing.
    with pytest.raises(ValueError, match="applies to 'terminal_settlement' only"):
        EpisodeSpec(side=1, target_qty_ticks=1, terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD,
                    terminal_surcharge_bps=1)
    with pytest.raises(ValueError, match="unknown terminal_rule"):
        EpisodeSpec(side=1, target_qty_ticks=1, terminal_rule="forced_liquidation")
    with pytest.raises(TypeError, match="terminal_rule must be a string"):
        EpisodeSpec(side=1, target_qty_ticks=1, terminal_rule=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="terminal_surcharge_bps must be below 10000"):
        EpisodeSpec(side=1, target_qty_ticks=1, terminal_surcharge_bps=BPS_DENOMINATOR)
    with pytest.raises(ValueError, match="terminal_surcharge_bps must be non-negative"):
        EpisodeSpec(side=1, target_qty_ticks=1, terminal_surcharge_bps=-1)


def test_inventory_endpoints_follow_the_side() -> None:
    buy = EpisodeSpec(side=1, target_qty_ticks=5)
    sell = EpisodeSpec(side=-1, target_qty_ticks=5)
    assert (buy.initial_inventory_ticks, buy.target_inventory_ticks) == (0, 5)
    assert (sell.initial_inventory_ticks, sell.target_inventory_ticks) == (5, 0)


def test_numpy_integers_are_normalised_to_int() -> None:
    spec = EpisodeSpec(side=np.int64(-1), target_qty_ticks=np.int32(7), horizon_buckets=np.int64(2))
    assert type(spec.side) is int and type(spec.target_qty_ticks) is int
    assert spec == EpisodeSpec(side=-1, target_qty_ticks=7, horizon_buckets=2)
    assert hash(spec) == hash(EpisodeSpec(side=-1, target_qty_ticks=7, horizon_buckets=2))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"side": 0}, "side must be"),
        ({"side": 2}, "side must be"),
        ({"target_qty_ticks": 0}, "at least one tick"),
        ({"horizon_buckets": 0}, "horizon_buckets must be positive"),
        ({"participation_cap_bps": 10_001}, "participation_cap_bps must be in"),
        ({"participation_cap_bps": -1}, "participation_cap_bps must be in"),
        ({"arrival_lookback": 0}, "arrival_lookback must be positive"),
        ({"fee_bps": -1}, "fee_bps must be non-negative"),
        ({"latency_buckets": -1}, "latency_buckets must be non-negative"),
        ({"data_delay_buckets": -1}, "data_delay_buckets must be non-negative"),
        ({"risk_lambda": -1}, "risk_lambda must be non-negative"),
        ({"residual_mu": -1}, "residual_mu must be non-negative"),
        ({"slippage_ticks": -1}, "slippage_ticks must be non-negative"),
        ({"fill_rule": "fill_at_touch"}, "unknown fill_rule"),
        ({"terminal_rule": "none"}, "unknown terminal_rule"),
    ],
)
def test_out_of_domain_fields_raise(overrides: dict[str, object], match: str) -> None:
    params: dict[str, object] = {"side": 1, "target_qty_ticks": 100}
    params.update(overrides)
    with pytest.raises(ValueError, match=match):
        EpisodeSpec(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1.0, 0.5, True, "1", None])
def test_non_integer_fields_raise_type_error(bad: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        EpisodeSpec(side=1, target_qty_ticks=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EpisodeSpec(side=1, target_qty_ticks=1, fill_rule=3)  # type: ignore[arg-type]


def test_spec_is_immutable() -> None:
    spec = EpisodeSpec(side=1, target_qty_ticks=1)
    with pytest.raises(AttributeError):
        spec.side = -1  # type: ignore[misc]


# --- fee rounding -----------------------------------------------------------------


@pytest.mark.parametrize(
    "notional, fee_bps, expected",
    [
        (1_000_200_000, 10, 1_000_200),  # Exact.
        (5_002_500, 10, 5_003),          # 5,002.5 rounds up: the reference-fixture case.
        (5_002_400, 10, 5_003),          # 5,002.4 rounds up too: ceiling, not half-up.
        (1, 10, 1),                      # One cash tick of notional still costs a tick.
        (0, 10, 0),
        (123_456_789, 0, 0),             # Zero fee tier.
        (10_000, 10_000, 10_000),        # 100 % fee.
    ],
)
def test_fee_is_the_ceiling_in_cash_ticks(notional: int, fee_bps: int, expected: int) -> None:
    fee = fee_ticks(notional, fee_bps)
    assert fee == expected
    assert fee * 10_000 >= notional * fee_bps
    assert fee * 10_000 < notional * fee_bps + 10_000


# --- participation cap --------------------------------------------------------------


@pytest.mark.parametrize(
    "total, cap_bps, expected",
    [(600_000, 1_000, 60_000), (400_000, 1_000, 40_000), (5_000, 1_000, 500), (9, 1_000, 0),
     (2_000_000, 1_000, 200_000), (7, 10_000, 7), (7, 0, 0)],
)
def test_participation_cap_is_an_exact_floor(total: int, cap_bps: int, expected: int) -> None:
    assert participation_cap_ticks(total, cap_bps) == expected


# --- case (d): int64 overflow bounds ------------------------------------------------

MAX_PRICE = 100_000_00           # 100,000.00 USDT in price ticks.
THOUSAND_BTC = 1_000 * 100_000   # 1,000 BTC in quantity ticks.


def test_example_d_realistic_extremes_pass() -> None:
    """A 10^8-tick cap (1,000 BTC) at 10^7 price ticks over 300 buckets stays inside int64.

    The cash bound is 3e17 and the fee bound 1e16, both below ``INT64_MAX``.
    """
    spec = EpisodeSpec(side=1, target_qty_ticks=THOUSAND_BTC)
    max_total = THOUSAND_BTC * 10  # The 10 % cap is then exactly 1,000 BTC.
    assert participation_cap_ticks(max_total, spec.participation_cap_bps) == 10**8
    assert 300 * 10**8 * 10**7 == 3 * 10**17 < contract.INT64_MAX
    assert 10**8 * 10**7 * 10 == 10**16 < contract.INT64_MAX
    check_episode_bounds(spec, MAX_PRICE, max_total)


def test_cash_bound_raises_for_the_million_record_stress_pattern() -> None:
    """One bucket of 1,000,000 records at the canonical day's price and size extremes overflows.

    The cap times the price over 300 steps exceeds ``INT64_MAX``, so the bounds
    check must raise before any arithmetic wraps.
    """
    price = round(97_839.50 * contract.PRICE_SCALE)
    total = 1_000_000 * round(18.89042 * contract.QUANTITY_SCALE)
    spec = EpisodeSpec(side=1, target_qty_ticks=1)
    cap = participation_cap_ticks(total, spec.participation_cap_bps)
    assert spec.horizon_buckets * cap * price > contract.INT64_MAX
    with pytest.raises(OverflowError, match="int64 cash bound exceeded"):
        check_episode_bounds(spec, price, total)


def test_fee_bound_is_checked_separately_from_cash() -> None:
    price, total = 10**9, 10**10  # Cap 10^9; cap x price = 10^18 fits, times fee 10 does not.
    spec = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=1)
    with pytest.raises(OverflowError, match="int64 fee bound exceeded"):
        check_episode_bounds(spec, price, total)
    check_episode_bounds(
        EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=1, fee_bps=0), price, total
    )


def test_slippage_counts_toward_the_worst_price() -> None:
    """Cap 10^9 x price 10^9 = 10^18 fits; a slippage of 9 x 10^9 ticks pushes it to 10^19."""
    spec = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=1, fee_bps=0, slippage_ticks=1)
    check_episode_bounds(spec, 10**9 - 1, 10**10)
    with pytest.raises(OverflowError, match="cash bound"):
        check_episode_bounds(
            EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=1, fee_bps=0,
                        slippage_ticks=9 * 10**9), 10**9, 10**10,
        )


def test_reward_coefficient_bound() -> None:
    spec = EpisodeSpec(side=1, target_qty_ticks=10**10, residual_mu=10**9,
                       terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD)
    with pytest.raises(OverflowError, match="int64 reward bound exceeded"):
        check_episode_bounds(spec, 1, 1)


# --- settlement price -----------------------------------------------------------------


@pytest.mark.parametrize(
    "mark, side, surcharge, expected",
    [
        (10_010, 1, 0, 10_010),      # No surcharge: the mark itself.
        (10_010, -1, 0, 10_010),
        (10_010, 1, 25, 10_036),     # 10,035.025 rounded up against the buyer.
        (10_010, -1, 25, 9_984),     # 9,984.975 rounded down against the seller.
        (10_010, 1, 10, 10_021),     # 10,020.01 -> 10,021.
        (10_010, -1, 10, 9_999),     # 9,999.99 -> 9,999.
        (10_010, 1, 50, 10_061),     # 10,060.05 -> 10,061.
        (10_010, -1, 50, 9_959),     # 9,959.95 -> 9,959.
        (9_990, 1, 25, 10_015),      # 10,014.975 -> 10,015.
        (4_000, 1, 25, 4_010),       # Exact product: no rounding needed.
        (4_000, -1, 25, 3_990),
        (1, -1, 0, 1),               # A one-tick mark survives without a surcharge.
    ],
)
def test_settlement_price_moves_the_mark_against_the_side_and_rounds_against_it(
    mark: int, side: int, surcharge: int, expected: int
) -> None:
    price = settlement_price_ticks(mark, side, surcharge)
    assert price == expected
    # Never favourable: a buyer pays at least the exact adjusted mark, a seller gets at most it.
    exact = mark * (BPS_DENOMINATOR + side * surcharge) / BPS_DENOMINATOR
    assert (price >= exact) if side == 1 else (price <= exact)
    assert abs(price - exact) < 1


def test_settlement_price_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="is not positive"):
        settlement_price_ticks(1, -1, 1)        # floor(0.9999) = 0.
    with pytest.raises(ValueError, match="terminal mark must be positive"):
        settlement_price_ticks(0, 1, 0)
    with pytest.raises(ValueError, match="side must be"):
        settlement_price_ticks(10, 0, 0)
    with pytest.raises(ValueError, match="surcharge must be in"):
        settlement_price_ticks(10, 1, BPS_DENOMINATOR)
    with pytest.raises(ValueError, match="surcharge must be in"):
        settlement_price_ticks(10, 1, -1)


def test_settlement_bound_counts_the_settled_target_at_the_surcharged_price() -> None:
    """The residual can be settled in full at the buy-side settlement price of the max VWAP."""
    spec = EpisodeSpec(side=-1, target_qty_ticks=10**9, horizon_buckets=1,
                       participation_cap_bps=0, fee_bps=0, terminal_surcharge_bps=25)
    # 10**9 ticks x ceil(10**10 x 1.0025) = 1.0025e19 > INT64_MAX although nothing can fill.
    with pytest.raises(OverflowError, match="int64 cash bound exceeded"):
        check_episode_bounds(spec, 10**10, 10**6)
    check_episode_bounds(spec, 9 * 10**8, 10**6)   # 9.0225e17 fits.
    legacy = EpisodeSpec(side=-1, target_qty_ticks=10**9, horizon_buckets=1,
                         participation_cap_bps=0, fee_bps=0,
                         terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD)
    check_episode_bounds(legacy, 921 * 10**7, 10**6)   # 9.21e18 at the mark itself fits...
    with pytest.raises(OverflowError, match="int64 cash bound exceeded"):
        check_episode_bounds(spec, 921 * 10**7, 10**6)   # ...but 9.233e18 at 1.0025 x it does not.
    # The settlement fee product is checked on its own; the legacy residual pays no fee.
    fee_heavy = EpisodeSpec(side=1, target_qty_ticks=10**9, horizon_buckets=1,
                            participation_cap_bps=0, fee_bps=10_000)
    with pytest.raises(OverflowError, match="int64 fee bound exceeded"):
        check_episode_bounds(fee_heavy, 10**9, 10**6)
    check_episode_bounds(EpisodeSpec(side=1, target_qty_ticks=10**9, horizon_buckets=1,
                                     participation_cap_bps=0, fee_bps=10_000,
                                     terminal_rule=TERMINAL_RULE_RESIDUAL_PEROLD), 10**9, 10**6)


def test_negative_maxima_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        check_episode_bounds(EpisodeSpec(side=1, target_qty_ticks=1), -1, 1)
