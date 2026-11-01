"""Tests for tapeforge.execution.simulator against the hand-verified reference fixture, asserted
exactly in cash ticks.

Cases: (a) a buy under TWAP, (b) the same episode as a sell, (c) partial
completion under the three legacy residual rules and under terminal settlement
at 0 / 10 / 25 / 50 bps for both sides, (e) an invalid bucket and the
5,002.5 -> 5,003 fee case, plus the TWAP, immediate and all-zero policies on
the case (a) tape under both terminal rules. Every expected value is a row of
the reference tables in ``test_execution.tapes`` or a hand calculation
written out next to the assertion.
"""

from __future__ import annotations

import pytest

from tapeforge.execution.simulator import (
    EpisodeOver,
    fill_shortfall,
    implementation_shortfall,
    residual_fixed_penalty,
    residual_forced_liquidation,
    residual_perold,
    settlement_shortfall,
    shortfall_bps,
    step,
)
from test_execution.policies import (
    immediate_policy,
    rollout,
    schedule_policy,
    twap_policy,
    zero_policy,
)
from test_execution.tapes import (
    DEEP,
    P0,
    P_T,
    Q_STAR,
    legacy_spec,
    slice_from_rows,
    worksheet_spec,
    worksheet_tape,
)

# --- case (a): buy 3.00000 BTC over 3 buckets under TWAP -----------------------

BUY_TABLE = [
    # Fill qty, notional,      fee,       cash after,     inventory, remaining.
    (100_000, 1_000_200_000, 1_000_200, -1_001_200_200, 100_000, 200_000),
    (100_000, 1_000_500_000, 1_000_500, -2_002_700_700, 200_000, 100_000),
    (100_000, 1_000_100_000, 1_000_100, -3_003_800_800, 300_000, 0),
]
BUY_IS = 3_800_800


def test_example_a_buy_twap_table() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    states, results = rollout(spec, market, twap_policy)
    assert len(results) == 3
    for row, state, result in zip(BUY_TABLE, states[1:], results, strict=True):
        fill, notional, fee, cash, inventory, remaining = row
        assert result.fill_qty_ticks == fill
        assert result.notional_ticks == notional
        assert result.fee_ticks == fee
        assert state.cash_ticks == cash
        assert state.inventory_ticks == inventory
        assert spec.target_qty_ticks - state.filled_qty_ticks == remaining
    final = states[-1]
    assert final.exec_notional_ticks == 3_000_800_000
    assert final.fill_fees_ticks == 3_000_800
    assert implementation_shortfall(final, spec) == BUY_IS
    assert shortfall_bps(BUY_IS, P0, Q_STAR) == pytest.approx(12.669, abs=1e-3)
    # The paper portfolio at the arrival price minus the real portfolio is the shortfall.
    assert final.cash_ticks + final.inventory_ticks * P0 == -BUY_IS
    assert results[-1].terminated and not results[-1].truncated


def test_example_a_reward_decomposition() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    _, results = rollout(spec, market, twap_policy)
    assert [r.cost_component for r in results] == [1_200_200, 1_500_500, 1_100_100]
    assert [r.risk_component for r in results] == [0, 0, 0]
    assert [r.terminal_component for r in results] == [0, 0, 0]
    assert sum(r.reward_ticks for r in results) == -BUY_IS


def test_example_a_risk_term_accrues_on_remaining_quantity() -> None:
    """The risk term charges ``risk_lambda`` per remaining quantity tick after each step.

    Remaining quantity after each step is 2/3, 1/3 and 0 of the target, so the
    total risk charge is ``risk_lambda`` times the target.
    """
    spec, market = worksheet_spec(side=1, risk_lambda=7), worksheet_tape()
    _, results = rollout(spec, market, twap_policy)
    assert [r.risk_component for r in results] == [7 * 200_000, 7 * 100_000, 0]
    assert sum(r.reward_ticks for r in results) == -BUY_IS - 7 * Q_STAR


# --- case (b): the same episode as a sell --------------------------------------

SELL_TABLE = [
    (100_000, 1_000_200_000, 1_000_200, 999_199_800, 200_000, 200_000),
    (100_000, 1_000_500_000, 1_000_500, 1_998_699_300, 100_000, 100_000),
    (100_000, 1_000_100_000, 1_000_100, 2_997_799_200, 0, 0),
]
SELL_IS = 2_200_800


def test_example_b_sell_mirror_table() -> None:
    spec, market = worksheet_spec(side=-1), worksheet_tape()
    states, results = rollout(spec, market, twap_policy)
    assert states[0].inventory_ticks == Q_STAR
    for row, state, result in zip(SELL_TABLE, states[1:], results, strict=True):
        fill, notional, fee, cash, inventory, remaining = row
        assert result.fill_qty_ticks == fill
        assert result.notional_ticks == notional
        assert result.fee_ticks == fee
        assert state.cash_ticks == cash
        assert state.inventory_ticks == inventory
        assert spec.target_qty_ticks - state.filled_qty_ticks == remaining
    final = states[-1]
    assert implementation_shortfall(final, spec) == SELL_IS
    assert SELL_IS != BUY_IS  # A simulator that forgets the sign flip reports 12.67 bps twice.
    assert shortfall_bps(SELL_IS, P0, Q_STAR) == pytest.approx(7.336, abs=1e-3)
    # Paper proceeds at the arrival price minus real cash are the shortfall.
    assert P0 * Q_STAR - final.cash_ticks == SELL_IS
    assert sum(r.reward_ticks for r in results) == -SELL_IS


# --- case (c): partial completion because the liquidity cap binds --------------

PARTIAL_TABLE = [
    # Total,   cap,    submitted, fill,   notional,    fee,     cash after,     remaining.
    (600_000, 60_000, 100_000, 60_000, 600_120_000, 600_120, -600_720_120, 240_000),
    (400_000, 40_000, 100_000, 40_000, 400_200_000, 400_200, -1_001_320_320, 200_000),
    (800_000, 80_000, 100_000, 80_000, 800_080_000, 800_080, -1_802_200_400, 120_000),
]
IS_FILLED = 2_200_400
RESIDUAL = 120_000


def _partial(mu: int = 0) -> tuple[list, list, object]:
    """Run the case (c) schedule on the thin tape under the legacy rule with ``residual_mu`` mu."""
    spec = legacy_spec(side=1, residual_mu=mu)
    market = worksheet_tape(totals=[row[0] for row in PARTIAL_TABLE])
    states, results = rollout(spec, market, schedule_policy(100_000))
    return states, results, spec


def test_example_c_cap_binds_and_the_horizon_ends_with_residual() -> None:
    states, results, spec = _partial()
    for row, state, result in zip(PARTIAL_TABLE, states[1:], results, strict=True):
        total, cap, submitted, fill, notional, fee, cash, remaining = row
        assert result.cap_qty_ticks == cap
        assert result.submitted_qty_ticks == submitted
        assert result.fill_qty_ticks == fill
        assert result.notional_ticks == notional
        assert result.fee_ticks == fee
        assert state.cash_ticks == cash
        assert spec.target_qty_ticks - state.filled_qty_ticks == remaining
    final = states[-1]
    assert final.filled_qty_ticks == 180_000
    assert final.exec_notional_ticks == 1_800_400_000
    assert final.fill_fees_ticks == 1_800_400
    assert final.settled_qty_ticks == 0 and final.settlement_fees_ticks == 0
    assert results[-1].terminated and not results[-1].truncated
    assert results[-1].terminal_price_ticks == P_T
    assert results[-1].settlement_qty_ticks == 0 and not results[-1].settlement_exceeds_cap


def test_example_c_residual_rules_reproduce_the_table() -> None:
    # Rule R1: forced buy of the 120,000 residual at terminal mark + kappa = 10,020, with fees.
    assert residual_forced_liquidation(1, P0, P_T, RESIDUAL, kappa=10, fee_bps=10) == 3_602_400
    # Rule R2: the Perold residual term (the residual-inventory term of the
    # implementation-shortfall decomposition) with ``mu`` 0 and 20.
    assert residual_perold(1, P0, P_T, RESIDUAL, mu=0) == 1_200_000
    assert residual_perold(1, P0, P_T, RESIDUAL, mu=20) == 3_600_000
    # Rule R3: a fixed penalty per residual tick only.
    assert residual_fixed_penalty(RESIDUAL, mu=20) == 2_400_000
    totals = {
        "R1": IS_FILLED + 3_602_400,
        "R2_mu0": IS_FILLED + 1_200_000,
        "R2_mu20": IS_FILLED + 3_600_000,
        "R3": IS_FILLED + 2_400_000,
    }
    assert totals == {"R1": 5_802_800, "R2_mu0": 3_400_400, "R2_mu20": 5_800_400, "R3": 4_600_400}
    bps = {k: round(shortfall_bps(v, P0, Q_STAR), 2) for k, v in totals.items()}
    assert bps == {"R1": 19.34, "R2_mu0": 11.33, "R2_mu20": 19.33, "R3": 15.33}


def test_example_c_accepted_rule_is_perold_with_mu() -> None:
    states, results, spec = _partial(mu=20)
    assert results[-1].terminal_component == 3_600_000
    assert implementation_shortfall(states[-1], spec, P_T) == 5_800_400
    assert sum(r.reward_ticks for r in results) == -5_800_400
    states0, results0, spec0 = _partial(mu=0)
    assert results0[-1].terminal_component == 1_200_000
    assert implementation_shortfall(states0[-1], spec0, P_T) == 3_400_400


def test_example_c_falling_terminal_price_rewards_inaction_when_mu_is_zero() -> None:
    """A terminal mark of 9,990 with ``residual_mu`` 0 pushes the shortfall below the filled value.

    This is the documented incentive of the legacy residual rule, not a defect:
    a zero ``residual_mu`` rewards leaving quantity unfilled when the price falls.
    """
    spec = legacy_spec(side=1, residual_mu=0)
    market = worksheet_tape(totals=[row[0] for row in PARTIAL_TABLE], after=[(DEEP, 9_990)])
    states, results = rollout(spec, market, schedule_policy(100_000))
    assert results[-1].terminal_component == -1_200_000
    assert implementation_shortfall(states[-1], spec, 9_990) == 1_000_400
    assert IS_FILLED > 1_000_400


def test_shortfall_with_residual_needs_a_terminal_price() -> None:
    states, _, spec = _partial()
    with pytest.raises(ValueError, match="needs a terminal price"):
        implementation_shortfall(states[-1], spec)


# --- case (e): zero-liquidity and sub-tick-liquidity buckets --------------------


def test_example_e_invalid_bucket_consumes_the_decision_without_a_fill() -> None:
    spec = worksheet_spec(side=1)
    market = worksheet_tape(totals=[DEEP, 0, DEEP], vwaps=[10_002, 0, 10_001])
    market = market.with_bucket(1, event_count=0, total_quantity=0, vwap_ticks=0)
    assert not market.valid[market.index(1)]
    states, results = rollout(spec, market, schedule_policy(100_000))
    skipped = results[1]
    assert skipped.cap_qty_ticks == 0
    assert skipped.submitted_qty_ticks == 100_000
    assert skipped.fill_qty_ticks == 0
    assert skipped.fee_ticks == 0
    assert states[2].cash_ticks == states[1].cash_ticks
    assert states[2].filled_qty_ticks == states[1].filled_qty_ticks
    # Time advanced: the mark is the last valid VWAP (bucket 0) with age 1.
    assert states[2].step == 2
    assert states[2].mark_price_ticks == 10_002
    assert states[2].mark_age == 1
    assert states[1].mark_age == 0


def test_example_e_sub_tick_liquidity_fee_rounds_up_to_5003() -> None:
    spec = worksheet_spec(side=1)
    market = worksheet_tape(totals=[DEEP, 5_000, DEEP], vwaps=[10_002, 10_005, 10_001])
    _, results = rollout(spec, market, schedule_policy(100_000))
    tiny = results[1]
    assert tiny.cap_qty_ticks == 500
    assert tiny.fill_qty_ticks == 500
    assert tiny.fill_price_ticks == 10_005
    assert tiny.notional_ticks == 5_002_500
    assert tiny.fee_ticks == 5_003  # ceil(5,002.5); floor would give 5,002.


# --- reference policies on the case (a) tape ------------------------------------


def test_twap_baseline_gives_the_tabulated_shortfall() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    states, _ = rollout(spec, market, twap_policy)
    assert implementation_shortfall(states[-1], spec) == BUY_IS


def test_immediate_baseline_fills_as_fast_as_the_cap_allows() -> None:
    """Submitting everything fills 200,000 (the 10 % cap) at 10,002, then 100,000 at 10,005.

    The fully filled shortfall exceeds the TWAP shortfall on this tape, and a
    further step after termination raises ``EpisodeOver``.
    """
    spec, market = worksheet_spec(side=1), worksheet_tape()
    states, results = rollout(spec, market, immediate_policy)
    assert [r.submitted_qty_ticks for r in results] == [300_000, 100_000]
    assert [r.fill_qty_ticks for r in results] == [200_000, 100_000]
    assert [r.notional_ticks for r in results] == [2_000_400_000, 1_000_500_000]
    assert [r.fee_ticks for r in results] == [2_000_400, 1_000_500]
    assert [r.terminated for r in results] == [False, True]
    fully_filled_is = (3_000_900_000 - P0 * Q_STAR) + 3_000_900
    assert fully_filled_is == 3_900_900
    assert implementation_shortfall(states[-1], spec) == fully_filled_is
    assert fully_filled_is > BUY_IS  # Speed costs more than patience on this tape.
    with pytest.raises(EpisodeOver):
        step(states[-1], 1.0, market, spec)


def test_all_zero_baseline_pays_only_the_residual_term_under_the_legacy_rule() -> None:
    spec, market = legacy_spec(side=1, residual_mu=20), worksheet_tape()
    states, results = rollout(spec, market, zero_policy)
    assert [r.fill_qty_ticks for r in results] == [0, 0, 0]
    assert states[-1].cash_ticks == 0 and states[-1].fill_fees_ticks == 0
    assert states[-1].settled_qty_ticks == 0 and states[-1].inventory_ticks == 0
    residual = residual_perold(1, P0, P_T, Q_STAR, mu=20)
    assert residual == (P_T - P0) * Q_STAR + 20 * Q_STAR == 9_000_000
    assert results[-1].terminal_component == residual
    assert implementation_shortfall(states[-1], spec, P_T) == residual
    assert sum(r.reward_ticks for r in results) == -residual


# --- case (s): terminal settlement of the case (c) residual ----------------------------
#
# The case (c) schedule fills 180,000 of 300,000 and leaves 120,000 at the horizon. Under
# terminal settlement the residual is settled at the terminal bucket (DEEP = 2,000,000 at
# P_T = 10,010; its 10 % cap is 200,000 >= 120,000) at the surcharged mark. Per row:
# surcharge, settlement price, notional = price x 120,000, fee = ceil(notional x 10 / 10,000),
# settlement cost = side x (notional - 10,000 x 120,000) + fee, total IS = fills IS + cost.
# Buy fills IS is 2,200,400 (case c); sell fills IS is
# -(1,800,400,000 - 1,800,000,000) + 1,800,400 = 1,400,400.
BUY_SETTLEMENT_TABLE = [
    #  bps, price,  notional,       fee,       cost,       total IS
    (0, 10_010, 1_201_200_000, 1_201_200, 2_401_200, 4_601_600),   # 10,010.000 exact.
    (10, 10_021, 1_202_520_000, 1_202_520, 3_722_520, 5_922_920),  # 10,020.01 -> 10,021.
    (25, 10_036, 1_204_320_000, 1_204_320, 5_524_320, 7_724_720),  # 10,035.025 -> 10,036.
    (50, 10_061, 1_207_320_000, 1_207_320, 8_527_320, 10_727_720),  # 10,060.05 -> 10,061.
]
SELL_SETTLEMENT_TABLE = [
    (0, 10_010, 1_201_200_000, 1_201_200, 1_200, 1_401_600),      # Favourable move nets the fee.
    (10, 9_999, 1_199_880_000, 1_199_880, 1_319_880, 2_720_280),  # 9,999.99 -> 9,999.
    (25, 9_984, 1_198_080_000, 1_198_080, 3_118_080, 4_518_480),  # 9,984.975 -> 9,984.
    (50, 9_959, 1_195_080_000, 1_195_080, 6_115_080, 7_515_480),  # 9,959.95 -> 9,959.
]
PARTIAL_TOTALS = [row[0] for row in PARTIAL_TABLE]


@pytest.mark.parametrize("row", BUY_SETTLEMENT_TABLE)
def test_example_s_buy_residual_is_settled_at_the_surcharged_mark(row: tuple[int, ...]) -> None:
    bps, price, notional, fee, cost, total = row
    spec = worksheet_spec(side=1, terminal_surcharge_bps=bps)
    states, results = rollout(spec, worksheet_tape(totals=PARTIAL_TOTALS), schedule_policy(100_000))
    last, final = results[-1], states[-1]
    assert [r.fill_qty_ticks for r in results] == [60_000, 40_000, 80_000]
    assert last.terminated and not last.truncated
    assert last.terminal_price_ticks == P_T
    assert last.settlement_qty_ticks == RESIDUAL == 120_000
    assert last.settlement_price_ticks == price
    assert last.settlement_notional_ticks == notional == price * RESIDUAL
    assert last.settlement_fee_ticks == fee
    assert last.terminal_component == cost
    assert last.settlement_cap_qty_ticks == 200_000 and not last.settlement_exceeds_cap
    # Ordinary fills and the settlement stay in separate accounts; cash and inventory hold both.
    assert final.filled_qty_ticks == 180_000 and final.settled_qty_ticks == RESIDUAL
    assert final.exec_notional_ticks == 1_800_400_000 and final.fill_fees_ticks == 1_800_400
    assert final.settlement_notional_ticks == notional and final.settlement_fees_ticks == fee
    assert final.inventory_ticks == Q_STAR
    assert final.cash_ticks == -(1_800_400_000 + notional) - 1_800_400 - fee
    assert spec.target_qty_ticks - final.filled_qty_ticks - final.settled_qty_ticks == 0
    # Shortfall decomposition and identities.
    assert fill_shortfall(final, spec) == IS_FILLED
    assert settlement_shortfall(final, spec) == cost
    assert implementation_shortfall(final, spec) == total == IS_FILLED + cost
    assert final.cash_ticks + final.inventory_ticks * P0 == -total   # Paper portfolio identity.
    assert sum(r.reward_ticks for r in results) == -total
    # The legacy opportunity-cost term is not charged on top of the settlement.
    assert total - IS_FILLED == cost != residual_perold(1, P0, P_T, RESIDUAL, 0) + cost


@pytest.mark.parametrize("row", SELL_SETTLEMENT_TABLE)
def test_example_s_sell_residual_is_settled_below_the_mark(row: tuple[int, ...]) -> None:
    bps, price, notional, fee, cost, total = row
    spec = worksheet_spec(side=-1, terminal_surcharge_bps=bps)
    states, results = rollout(spec, worksheet_tape(totals=PARTIAL_TOTALS), schedule_policy(100_000))
    last, final = results[-1], states[-1]
    assert last.settlement_qty_ticks == RESIDUAL and last.settlement_price_ticks == price
    assert last.settlement_notional_ticks == notional and last.settlement_fee_ticks == fee
    assert last.terminal_component == cost
    assert final.inventory_ticks == 0   # Liquidated to zero through fills plus settlement.
    assert final.cash_ticks == 1_800_400_000 + notional - 1_800_400 - fee
    assert fill_shortfall(final, spec) == 1_400_400
    assert implementation_shortfall(final, spec) == total == 1_400_400 + cost
    assert P0 * Q_STAR - final.cash_ticks == total   # Paper proceeds minus real cash.
    assert sum(r.reward_ticks for r in results) == -total


def test_example_s_settlement_fee_follows_the_same_ceiling_rule() -> None:
    """A 500-tick residual settled at 10,005 has notional 5,002,500 and pays 5,003, not 5,002."""
    spec = worksheet_spec(side=1, target_qty_ticks=500)
    _, results = rollout(spec, worksheet_tape(after=[(DEEP, 10_005)]), zero_policy)
    last = results[-1]
    assert last.settlement_qty_ticks == 500 and last.settlement_price_ticks == 10_005
    assert last.settlement_notional_ticks == 5_002_500
    assert last.settlement_fee_ticks == 5_003
    assert last.terminal_component == (5_002_500 - 500 * P0) + 5_003 == 2_500 + 5_003


def test_example_s_zero_action_settles_the_whole_target_above_the_cap() -> None:
    """Never trading settles 300,000 at 10,036: 200,000 is the terminal bucket's cap, so the
    synthetic settlement exceeds it and the record says so. Cost 10,800,000 + fee 3,010,800."""
    spec, market = worksheet_spec(side=1, terminal_surcharge_bps=25), worksheet_tape()
    states, results = rollout(spec, market, zero_policy)
    last, final = results[-1], states[-1]
    assert [r.fill_qty_ticks for r in results] == [0, 0, 0]
    assert last.settlement_qty_ticks == Q_STAR and last.settlement_price_ticks == 10_036
    assert last.settlement_notional_ticks == 3_010_800_000
    assert last.settlement_fee_ticks == 3_010_800
    assert last.settlement_cap_qty_ticks == 200_000 and last.settlement_exceeds_cap
    assert last.terminal_component == 10_800_000 + 3_010_800 == 13_810_800
    assert final.filled_qty_ticks == 0 and final.settled_qty_ticks == Q_STAR
    assert final.inventory_ticks == Q_STAR and final.fill_fees_ticks == 0
    assert implementation_shortfall(final, spec) == 13_810_800
    assert shortfall_bps(13_810_800, P0, Q_STAR) == pytest.approx(46.036, abs=1e-3)
    # Under the legacy rule with mu 0 the same inaction paid 3,000,000 and no fee at all.
    assert residual_perold(1, P0, P_T, Q_STAR, 0) == 3_000_000


def test_example_s_favourable_terminal_move_makes_inaction_cheap_but_not_fee_free() -> None:
    """With the mark at 9,990 a buy settles below arrival: -3,000,000 of price gain offsets the
    2,997,000 fee at 0 bps (cost -3,000); at 25 bps the surcharge dominates (cost 7,504,500).

    Favourable market movement can make delaying execution beneficial; the
    settlement rule does not and should not force every zero-action episode to
    lose. What it removes is the fee avoidance: the residual pays the fee.
    """
    falling = worksheet_tape(after=[(DEEP, 9_990)])
    states, results = rollout(worksheet_spec(side=1), falling, zero_policy)
    assert results[-1].settlement_price_ticks == 9_990
    assert results[-1].settlement_notional_ticks == 2_997_000_000
    assert results[-1].settlement_fee_ticks == 2_997_000
    assert results[-1].terminal_component == -3_000_000 + 2_997_000 == -3_000
    assert states[-1].settlement_fees_ticks == 2_997_000 > 0
    _, surcharged = rollout(worksheet_spec(side=1, terminal_surcharge_bps=25), falling, zero_policy)
    assert surcharged[-1].settlement_price_ticks == 10_015
    assert surcharged[-1].terminal_component == 4_500_000 + 3_004_500 == 7_504_500


def test_example_s_early_completion_settles_nothing() -> None:
    spec, market = worksheet_spec(side=1, terminal_surcharge_bps=25), worksheet_tape()
    states, results = rollout(spec, market, twap_policy)
    last, final = results[-1], states[-1]
    assert last.terminated and final.filled_qty_ticks == Q_STAR
    assert last.settlement_qty_ticks == last.settlement_price_ticks == last.terminal_component == 0
    assert last.terminal_price_ticks == 0 and not last.settlement_exceeds_cap
    assert final.settled_qty_ticks == 0 and final.settlement_fees_ticks == 0
    assert implementation_shortfall(final, spec) == BUY_IS   # Same as under the legacy rule.
    _, immediate_results = rollout(spec, market, immediate_policy)
    assert immediate_results[-1].terminated and immediate_results[-1].settlement_qty_ticks == 0


def test_example_s_missing_terminal_data_truncates_and_settles_nothing() -> None:
    spec = worksheet_spec(side=1, terminal_surcharge_bps=25)
    market = worksheet_tape(totals=PARTIAL_TOTALS, after=[])   # No bucket after the horizon.
    states, results = rollout(spec, market, schedule_policy(100_000))
    last, final = results[-1], states[-1]
    assert last.truncated and not last.terminated
    assert last.settlement_qty_ticks == 0 and last.terminal_component == 0
    assert last.settlement_price_ticks == 0 and last.terminal_price_ticks == 0
    assert final.settled_qty_ticks == 0 and final.filled_qty_ticks == 180_000
    assert spec.target_qty_ticks - final.filled_qty_ticks == RESIDUAL   # Not completed.
    assert fill_shortfall(final, spec) == IS_FILLED
    with pytest.raises(ValueError, match="neither filled nor settled"):
        implementation_shortfall(final, spec, P_T)   # No settled shortfall exists to report.


def test_example_s_invalid_settlement_price_is_rejected_not_truncated() -> None:
    """A one-tick terminal mark cannot absorb a sell surcharge: the step raises."""
    spec = worksheet_spec(side=-1, terminal_surcharge_bps=1)
    market = worksheet_tape(after=[(DEEP, 1)])
    with pytest.raises(ValueError, match="settlement price 0 is not positive"):
        rollout(spec, market, zero_policy)


def test_example_s_risk_term_is_charged_on_the_pre_settlement_residual() -> None:
    """The horizon step's risk term prices the 120,000 still unfilled after its fill; the
    settlement that follows in the same step does not reduce it, so both terminal rules
    charge identical risk."""
    settling = worksheet_spec(side=1, risk_lambda=7, terminal_surcharge_bps=25)
    legacy = legacy_spec(side=1, risk_lambda=7)
    market = worksheet_tape(totals=PARTIAL_TOTALS)
    _, settled = rollout(settling, market, schedule_policy(100_000))
    _, unsettled = rollout(legacy, market, schedule_policy(100_000))
    assert [r.risk_component for r in settled] == [7 * 240_000, 7 * 200_000, 7 * 120_000]
    assert [r.risk_component for r in settled] == [r.risk_component for r in unsettled]
    assert sum(r.reward_ticks for r in settled) == -(7_724_720 + 7 * 560_000)


def test_sub_tick_slice_rows_helper_matches_the_worksheet_layout() -> None:
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), None])
    assert market.origin == 1
    assert market.n_after == 2
    assert market.valid.tolist() == [True, True, False]
