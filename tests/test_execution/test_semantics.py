"""Tests for tapeforge.execution.simulator transition semantics: per-step invariants, termination
versus truncation, reset, action conversion, causality, latency and determinism.

``assert_step_invariants`` is shared with the property tests in ``test_properties``.
All tapes come from ``test_execution.tapes``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tapeforge.execution.simulator import (
    OBSERVATION_FIELDS,
    EpisodeOver,
    action_for_quantity,
    implementation_shortfall,
    is_done,
    observe,
    reset,
    step,
    terminal_mark,
)
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice, State, StepResult
from test_execution.policies import (
    Policy,
    immediate_policy,
    rollout,
    schedule_policy,
    twap_policy,
    zero_policy,
)
from test_execution.tapes import (
    DEEP,
    P0,
    frozen_bytes,
    slice_from_rows,
    worksheet_spec,
    worksheet_tape,
)


def assert_step_invariants(
    before: State, after: State, result: StepResult, market: MarketSlice, spec: EpisodeSpec
) -> None:
    """Assert the invariants every transition from ``before`` to ``after`` must satisfy.

    Submitted quantity is bounded by what remains; the fill by the remaining
    quantity and the participation cap of the fill bucket (zero for an invalid
    or missing bucket); the fee by the ceiling rule; inventory by the side and
    the fill count. Terminated and truncated are mutually exclusive, the reward
    is the negated sum of its components, and the running accounting identity
    holds.
    """
    remaining_before = spec.target_qty_ticks - before.filled_qty_ticks
    assert 0 <= result.submitted_qty_ticks <= remaining_before
    assert result.fill_qty_ticks <= min(remaining_before, result.cap_qty_ticks)
    if market.has_bucket(result.fill_bucket):
        i = market.index(result.fill_bucket)
        if not market.valid[i]:
            assert result.fill_qty_ticks == 0 and result.cap_qty_ticks == 0
        else:
            cap = int(market.total_quantity[i]) * spec.participation_cap_bps // 10_000
            assert result.cap_qty_ticks == cap
    else:
        assert result.truncated and result.fill_qty_ticks == 0
    assert result.fee_ticks * 10_000 >= result.notional_ticks * spec.fee_bps
    assert result.fee_ticks * 10_000 < result.notional_ticks * spec.fee_bps + 10_000
    assert result.notional_ticks == result.fill_price_ticks * result.fill_qty_ticks
    assert abs(after.inventory_ticks) <= spec.target_qty_ticks
    assert 0 <= after.filled_qty_ticks <= spec.target_qty_ticks
    expected_inventory = spec.initial_inventory_ticks + spec.side * after.filled_qty_ticks
    assert after.inventory_ticks == expected_inventory
    assert not (result.terminated and result.truncated)
    assert result.reward_ticks == -(
        result.cost_component + result.risk_component + result.terminal_component
    )
    # Running accounting identity: the paper portfolio at the arrival price minus the real
    # one equals the shortfall of the fills so far (residual term excluded).
    paper = after.cash_ticks + after.inventory_ticks * P0_of(after)
    initial = spec.initial_inventory_ticks * P0_of(after)
    shortfall_so_far = spec.side * (
        after.exec_notional_ticks - P0_of(after) * after.filled_qty_ticks
    ) + after.fees_ticks
    assert paper - initial == -shortfall_so_far


def P0_of(state: State) -> int:
    """Return the arrival price of ``state``, in price ticks."""
    return state.arrival_price_ticks


def _run_checked(
    spec: EpisodeSpec, market: MarketSlice, policy: Policy
) -> tuple[list[State], list[StepResult]]:
    """Roll out ``policy`` and assert the step invariants on every transition."""
    states, results = rollout(spec, market, policy)
    for before, after, result in zip(states[:-1], states[1:], results, strict=True):
        assert_step_invariants(before, after, result, market, spec)
    return states, results


# --- invariants on the reference tape ------------------------------------------------


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize(
    "policy", [twap_policy, immediate_policy, zero_policy, schedule_policy(70_001)]
)
def test_invariants_hold_on_the_worksheet_tape(side: int, policy: Policy) -> None:
    spec = worksheet_spec(side=side, risk_lambda=3, residual_mu=5)
    states, results = _run_checked(spec, worksheet_tape(), policy)
    assert results[-1].terminated
    total_risk = sum(r.risk_component for r in results)
    shortfall = implementation_shortfall(states[-1], spec, results[-1].terminal_price_ticks or None)
    assert sum(r.reward_ticks for r in results) == -shortfall - total_risk


# --- termination vs truncation ---------------------------------------------------------


def test_completion_terminates_early_and_stops_fills() -> None:
    """Submitting everything fills 200,000 (the cap) then 100,000: done after 2 of 3 steps.

    Stepping a finished episode raises ``EpisodeOver``.
    """
    spec, market = worksheet_spec(side=1), worksheet_tape()
    states, results = rollout(spec, market, immediate_policy)
    assert [r.fill_qty_ticks for r in results] == [200_000, 100_000]
    assert [r.terminated for r in results] == [False, True]
    assert not results[-1].truncated
    assert is_done(states[-1], spec)
    with pytest.raises(EpisodeOver, match="episode is over"):
        step(states[-1], 0.0, market, spec)


def test_horizon_end_terminates_even_with_residual() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    states, results = rollout(spec, market, zero_policy)
    assert [r.terminated for r in results] == [False, False, True]
    assert states[-1].step == spec.horizon_buckets
    assert results[-1].terminal_price_ticks == 10_010
    with pytest.raises(EpisodeOver):
        step(states[-1], 0.0, market, spec)


def test_short_slice_truncates_when_the_fill_bucket_is_missing() -> None:
    spec = worksheet_spec(side=1)
    market = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), (DEEP, 10_005)])  # Horizon 3 needs 3.
    states, results = rollout(spec, market, schedule_policy(100_000))
    assert len(results) == 3
    assert [r.fill_qty_ticks for r in results] == [100_000, 100_000, 0]
    last = results[-1]
    assert last.truncated and not last.terminated
    assert last.cap_qty_ticks == 0 and last.notional_ticks == 0
    assert last.cost_component == 0
    assert states[-1].step == 3  # Time still advanced.
    # No terminal price exists, so the shortfall of the fills is what can be reported.
    assert sum(r.reward_ticks for r in results) == -(
        spec.side * (states[-1].exec_notional_ticks - P0 * states[-1].filled_qty_ticks)
        + states[-1].fees_ticks
    )


def test_missing_terminal_mark_truncates_instead_of_terminating() -> None:
    spec = worksheet_spec(side=1)
    market = worksheet_tape(after=[])  # Fills exist; no bucket after the horizon.
    _, results = rollout(spec, market, zero_policy)
    assert results[-1].truncated and not results[-1].terminated
    assert results[-1].terminal_component == 0 and results[-1].terminal_price_ticks == 0
    assert terminal_mark(market, spec) is None


def test_terminal_mark_skips_unpriced_buckets_within_the_lookback() -> None:
    spec = worksheet_spec(side=1, arrival_lookback=3)
    market = slice_from_rows(
        [(DEEP, P0), (DEEP, P0), (DEEP, P0)],
        [(DEEP, 10_002), (DEEP, 10_005), (DEEP, 10_001), None, (0, 0), (DEEP, 10_020)],
    )
    assert terminal_mark(market, spec) == 10_020
    _, results = rollout(spec, market, zero_policy)
    assert results[-1].terminated and results[-1].terminal_price_ticks == 10_020
    # With lookback 2 the search covers buckets 3 and 4 only and never reaches 10,020.
    beyond = worksheet_spec(side=1, arrival_lookback=2)
    assert terminal_mark(market, beyond) is None


def test_completion_in_the_last_bucket_needs_no_terminal_mark() -> None:
    spec = worksheet_spec(side=1)
    market = worksheet_tape(after=[])
    _, results = rollout(spec, market, twap_policy)
    assert results[-1].terminated and results[-1].terminal_component == 0


def test_a_terminated_or_truncated_flag_is_set_at_most_once_per_episode() -> None:
    spec = worksheet_spec(side=1)
    for market in (worksheet_tape(), worksheet_tape(after=[])):
        _, results = rollout(spec, market, zero_policy)
        flags = [(r.terminated, r.truncated) for r in results]
        assert flags[:-1] == [(False, False)] * (len(flags) - 1)
        assert sum(flags[-1]) == 1


# --- reset ---------------------------------------------------------------------------


def test_reset_fixes_the_arrival_price_from_the_last_priced_bucket_within_lookback() -> None:
    spec = worksheet_spec(side=1, arrival_lookback=3)
    market = slice_from_rows([(DEEP, 9_990), None, (0, 0)], [(DEEP, 10_002)])
    state = reset(spec, market)
    assert state.arrival_price_ticks == 9_990 and state.mark_age == 2
    assert state.cash_ticks == 0 and state.filled_qty_ticks == 0 and state.step == 0


def test_reset_rejects_a_slice_without_a_priced_arrival_bucket() -> None:
    spec = worksheet_spec(side=1, arrival_lookback=2)
    with pytest.raises(ValueError, match="arrival price is undefined"):
        reset(spec, slice_from_rows([None, (0, 0)], [(DEEP, 10_002)]))
    with pytest.raises(ValueError, match="are needed"):
        reset(spec, slice_from_rows([(DEEP, P0)], [(DEEP, 10_002)]))


def test_reset_runs_the_overflow_check_on_the_slice_maxima() -> None:
    spec = worksheet_spec(side=1, horizon_buckets=1)
    with pytest.raises(OverflowError, match="int64 cash bound"):
        reset(spec, slice_from_rows([(DEEP, P0)], [(10**12, 10**9)]))


def test_reset_twice_gives_equal_states() -> None:
    spec, market = worksheet_spec(side=-1), worksheet_tape()
    assert reset(spec, market) == reset(spec, market)
    assert frozen_bytes([reset(spec, market)]) == frozen_bytes([reset(spec, market)])


# --- action conversion -------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
def test_actions_outside_the_unit_interval_raise(bad: float) -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    with pytest.raises(ValueError, match=r"finite fraction in \[0, 1\]"):
        step(reset(spec, market), bad, market, spec)


def test_non_numeric_actions_raise_type_error() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    for bad in ("0.5", None, True):
        with pytest.raises(TypeError, match="real number"):
            step(reset(spec, market), bad, market, spec)  # type: ignore[arg-type]


def test_action_is_floored_to_quantity_ticks_once() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    _, result = step(reset(spec, market), 0.5, market, spec)
    assert result.submitted_qty_ticks == 150_000
    _, result = step(reset(spec, market), 1 / 7, market, spec)
    assert result.submitted_qty_ticks == math.floor(300_000 / 7) == 42_857
    _, result = step(reset(spec, market), np.float32(0.25), market, spec)
    assert result.submitted_qty_ticks == 75_000
    _, result = step(reset(spec, market), 1, market, spec)  # Integer 1 is a valid fraction.
    assert result.submitted_qty_ticks == 300_000


@pytest.mark.parametrize("remaining", [1, 3, 7, 300_000, 999_983, 2**40 + 1])
@pytest.mark.parametrize("fraction", [0, 1, 2, 3, 5, 7, 11])
def test_action_for_quantity_round_trips_through_the_floor(remaining: int, fraction: int) -> None:
    quantity = min(remaining, remaining * fraction // 11)
    action = action_for_quantity(quantity, remaining)
    assert 0.0 <= action <= 1.0
    assert math.floor(action * remaining) == quantity


def test_action_for_quantity_rejects_overfill() -> None:
    with pytest.raises(ValueError, match="must be in"):
        action_for_quantity(5, 4)
    assert action_for_quantity(0, 0) == 0.0


# --- causality -------------------------------------------------------------------


def _long_tape(n_before: int = 4, n_after: int = 8) -> MarketSlice:
    """Build a tape whose every bucket has a distinct quantity and price."""
    rows_before = [(DEEP + 1_000 * i, P0 - i) for i in range(n_before)]
    rows_after = [(DEEP + 7_000 * i, P0 + 3 * i + 1) for i in range(n_after)]
    return slice_from_rows(rows_before, rows_after)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_observation_reads_only_buckets_up_to_t_minus_1_minus_d(delay: int) -> None:
    spec = worksheet_spec(side=1, horizon_buckets=5, data_delay_buckets=delay, arrival_lookback=2)
    market = _long_tape()
    states, _ = rollout(spec, market, schedule_policy(20_000))
    for t, state in enumerate(states[:-1]):
        baseline = observe(state, market, spec)
        assert baseline.shape == (len(OBSERVATION_FIELDS),)
        newest = t - 1 - delay
        for bucket in range(newest + 1, market.n_after):  # Every bucket >= t - delay.
            perturbed = market.with_bucket(bucket, total_quantity=1, vwap_ticks=1)
            np.testing.assert_array_equal(observe(state, perturbed, spec), baseline)
        changed = market.with_bucket(newest, total_quantity=1, vwap_ticks=1)
        assert not np.array_equal(observe(state, changed, spec), baseline)


def test_future_fills_do_not_change_earlier_observations() -> None:
    spec = worksheet_spec(side=1, horizon_buckets=5, arrival_lookback=2)
    market = _long_tape()
    states, _ = rollout(spec, market, schedule_policy(20_000))
    for t in range(1, 5):
        perturbed = market
        for bucket in range(t, market.n_after):
            perturbed = perturbed.with_bucket(bucket, total_quantity=50_000, vwap_ticks=12_345)
        other_states, _ = rollout(spec, perturbed, schedule_policy(20_000))
        assert other_states[t] == states[t]
        np.testing.assert_array_equal(observe(other_states[t], perturbed, spec),
                                      observe(states[t], market, spec))


def test_observation_fields_are_what_the_layout_says() -> None:
    spec = worksheet_spec(side=-1, horizon_buckets=4, arrival_lookback=2)
    market = slice_from_rows([(DEEP, P0), (600_000, 10_100)], [(DEEP, 10_002)] * 5)
    market = market.with_bucket(-1, event_count=3, signed_quantity=-150_000)
    obs = dict(zip(OBSERVATION_FIELDS, observe(reset(spec, market), market, spec), strict=True))
    assert obs["remaining_fraction"] == 1.0
    assert obs["time_remaining_fraction"] == 1.0
    assert obs["side"] == -1.0
    assert obs["mark_return"] == 0.0  # The arrival bucket is the newest priced bucket.
    assert obs["mark_age"] == 0.0
    assert obs["bucket_valid"] == 1.0
    assert obs["bucket_event_count"] == 3.0
    assert obs["bucket_quantity_fraction"] == 2.0  # 600,000 / 300,000.
    assert obs["bucket_signed_fraction"] == -0.25
    assert obs["bucket_return"] == pytest.approx(0.0)  # VWAP 10,100 against arrival 10,100.


def test_observation_of_an_empty_newest_bucket_is_zeroed_but_the_mark_persists() -> None:
    spec = worksheet_spec(side=1, horizon_buckets=3, arrival_lookback=2)
    market = slice_from_rows([(DEEP, P0), None], [(DEEP, 10_002)] * 4)
    state = reset(spec, market)
    assert state.mark_age == 1
    obs = dict(zip(OBSERVATION_FIELDS, observe(state, market, spec), strict=True))
    assert obs["bucket_valid"] == 0.0 and obs["bucket_event_count"] == 0.0
    assert obs["bucket_quantity_fraction"] == 0.0 and obs["bucket_signed_fraction"] == 0.0
    assert obs["bucket_return"] == 0.0 and obs["mark_return"] == 0.0 and obs["mark_age"] == 1.0


# --- latency -----------------------------------------------------------------------


def test_latency_one_fills_a_decision_at_t_in_bucket_t_plus_1() -> None:
    spec = worksheet_spec(side=1, latency_buckets=1)
    market = worksheet_tape(after=[(DEEP, 10_020), (DEEP, 10_030)])
    _, results = rollout(spec, market, schedule_policy(100_000))
    assert [r.fill_bucket for r in results] == [1, 2, 3]
    assert [r.fill_price_ticks for r in results] == [10_005, 10_001, 10_020]
    assert results[-1].terminated


def test_latency_shifts_fills_by_one_bucket_on_a_shifted_tape() -> None:
    rows = [(DEEP, 10_002), (DEEP, 10_005), (DEEP, 10_001), (DEEP, 10_020), (DEEP, 10_030)]
    direct = slice_from_rows([(DEEP, P0)], rows)
    shifted = slice_from_rows([(DEEP, P0)], [(DEEP, P0), *rows])
    _, immediate = rollout(worksheet_spec(side=1), direct, schedule_policy(100_000))
    lagged = worksheet_spec(side=1, latency_buckets=1)
    _, delayed = rollout(lagged, shifted, schedule_policy(100_000))
    assert [r.fill_price_ticks for r in delayed] == [r.fill_price_ticks for r in immediate]
    assert [r.fill_bucket for r in delayed] == [b + 1 for b in (r.fill_bucket for r in immediate)]


def test_latency_terminal_mark_starts_after_the_last_fill_bucket() -> None:
    spec = worksheet_spec(side=1, latency_buckets=1)
    market = worksheet_tape(after=[(DEEP, 10_020), (DEEP, 10_030)])
    assert terminal_mark(market, spec) == 10_030  # Bucket horizon + latency = 4, not bucket 3.


# --- determinism -------------------------------------------------------------------


def test_identical_inputs_give_byte_identical_sequences() -> None:
    spec, market = worksheet_spec(side=-1, risk_lambda=2, residual_mu=9), worksheet_tape(
        totals=[600_000, 400_000, 800_000]
    )
    first = rollout(spec, market, schedule_policy(100_000))
    second = rollout(spec, market, schedule_policy(100_000))
    assert first == second
    assert frozen_bytes(first[0] + first[1]) == frozen_bytes(second[0] + second[1])


def test_states_and_results_are_immutable_python_ints() -> None:
    spec, market = worksheet_spec(side=1), worksheet_tape()
    state, result = step(reset(spec, market), 0.5, market, spec)
    for obj in (state, result):
        for name, value in vars(obj).items():
            assert type(value) in (int, bool), (name, type(value))
    with pytest.raises(AttributeError):
        state.cash_ticks = 0  # type: ignore[misc]


def test_slippage_moves_the_fill_price_against_the_agent() -> None:
    market = worksheet_tape()
    buy_spec = worksheet_spec(side=1, slippage_ticks=3)
    sell_spec = worksheet_spec(side=-1, slippage_ticks=3)
    _, buy = step(reset(buy_spec, market), 1.0, market, buy_spec)
    _, sell = step(reset(sell_spec, market), 1.0, market, sell_spec)
    assert buy.fill_price_ticks == 10_002 + 3 and sell.fill_price_ticks == 10_002 - 3
    assert buy.fill_qty_ticks == sell.fill_qty_ticks == 200_000  # The 10 % cap binds.
