"""Tests for tapeforge.execution.simulator against the hand-verified reference fixture, asserted
exactly in cash ticks.

Cases: (a) a buy under TWAP, (b) the same episode as a sell, (c) partial
completion under the three residual rules, (e) an invalid bucket and the
5,002.5 -> 5,003 fee case, plus the TWAP, immediate and all-zero policies on
the case (a) tape. Every expected value is a row of the reference tables in
``test_execution.tapes``.
"""

from __future__ import annotations

import pytest

from tapeforge.execution.simulator import (
    EpisodeOver,
    implementation_shortfall,
    residual_fixed_penalty,
    residual_forced_liquidation,
    residual_perold,
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
    assert final.fees_ticks == 3_000_800
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
    """Run the case (c) fixed schedule on the thin tape with ``residual_mu`` set to ``mu``."""
    spec = worksheet_spec(side=1, residual_mu=mu)
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
    assert final.fees_ticks == 1_800_400
    assert results[-1].terminated and not results[-1].truncated
    assert results[-1].terminal_price_ticks == P_T


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

    This is the documented incentive of the residual rule, not a defect: a zero
    ``residual_mu`` rewards leaving quantity unfilled when the price falls.
    """
    spec = worksheet_spec(side=1, residual_mu=0)
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


def test_all_zero_baseline_pays_only_the_residual_term() -> None:
    spec, market = worksheet_spec(side=1, residual_mu=20), worksheet_tape()
    states, results = rollout(spec, market, zero_policy)
    assert [r.fill_qty_ticks for r in results] == [0, 0, 0]
    assert states[-1].cash_ticks == 0 and states[-1].fees_ticks == 0
    residual = residual_perold(1, P0, P_T, Q_STAR, mu=20)
    assert residual == (P_T - P0) * Q_STAR + 20 * Q_STAR == 9_000_000
    assert results[-1].terminal_component == residual
    assert implementation_shortfall(states[-1], spec, P_T) == residual
    assert sum(r.reward_ticks for r in results) == -residual


def test_sub_tick_slice_rows_helper_matches_the_worksheet_layout() -> None:
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), None])
    assert market.origin == 1
    assert market.n_after == 2
    assert market.valid.tolist() == [True, True, False]
