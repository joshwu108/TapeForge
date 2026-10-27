"""Property tests for tapeforge.execution.simulator: random tapes, random actions and both sides
must keep every step invariant, the reward identity, causality and byte-identical replay.

The per-step invariants are shared with ``test_semantics.assert_step_invariants``;
``episodes`` draws a spec, a slice with a priced arrival bucket and one action per
decision.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tapeforge.execution.simulator import implementation_shortfall, is_done, observe, reset, step
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice
from test_execution.tapes import frozen_bytes
from test_execution.test_semantics import assert_step_invariants

# A bucket is None (empty), (count, total, signed, vwap) with total > 0, or a
# zero-quantity valid bucket (count, 0, 0, 0).
_priced = st.tuples(
    st.integers(1, 50), st.integers(1, 5_000_000), st.just(0), st.integers(1, 20_000)
).flatmap(
    lambda t: st.integers(-t[1], t[1]).map(lambda signed: (t[0], t[1], signed, t[3]))
)
_bucket = st.one_of(st.none(), _priced, st.integers(1, 3).map(lambda c: (c, 0, 0, 0)))


def _slice(before: list, after: list) -> MarketSlice:
    """Build a slice from ``(count, total, signed, vwap)`` rows; ``None`` is an empty bucket."""
    rows = before + after
    count = np.array([0 if r is None else r[0] for r in rows], dtype=np.int64)
    total = np.array([0 if r is None else r[1] for r in rows], dtype=np.int64)
    signed = np.array([0 if r is None else r[2] for r in rows], dtype=np.int64)
    vwap = np.array([0 if r is None else r[3] for r in rows], dtype=np.int64)
    return MarketSlice(event_count=count, total_quantity=total, signed_quantity=signed,
                       vwap_ticks=vwap, valid=count > 0, origin=len(before))


@st.composite
def episodes(draw: st.DrawFn) -> tuple[EpisodeSpec, MarketSlice, list[float]]:
    """Draw a spec, a slice whose arrival bucket is priced, and one action per decision."""
    horizon = draw(st.integers(1, 6))
    latency = draw(st.integers(0, 1))
    delay = draw(st.integers(0, 1))
    lookback = draw(st.integers(1, 3))
    spec = EpisodeSpec(
        side=draw(st.sampled_from([1, -1])),
        target_qty_ticks=draw(st.integers(1, 2_000_000)),
        horizon_buckets=horizon,
        participation_cap_bps=draw(st.sampled_from([0, 1_000, 2_500, 10_000])),
        slippage_ticks=draw(st.integers(0, 2)),
        fee_bps=draw(st.sampled_from([0, 10, 25])),
        latency_buckets=latency,
        data_delay_buckets=delay,
        risk_lambda=draw(st.integers(0, 50)),
        residual_mu=draw(st.integers(0, 50)),
        arrival_lookback=lookback,
    )
    n_before = lookback + delay + draw(st.integers(0, 2))
    before = draw(st.lists(_bucket, min_size=n_before, max_size=n_before))
    before[-1 - delay] = draw(_priced)  # The arrival price must exist.
    n_after = draw(st.integers(0, horizon + latency + lookback + 1))
    after = draw(st.lists(_bucket, min_size=n_after, max_size=n_after))
    actions = draw(st.lists(st.floats(0.0, 1.0), min_size=horizon, max_size=horizon))
    return spec, _slice(before, after), actions


@given(episodes())
@settings(max_examples=400, deadline=None)
def test_invariants_reward_identity_causality_and_replay(
    episode: tuple[EpisodeSpec, MarketSlice, list[float]]
) -> None:
    spec, market, actions = episode
    state = reset(spec, market)
    assert state.arrival_price_ticks > 0
    states, results = [state], []
    for t, action in enumerate(actions):
        if is_done(state, spec):
            break
        observation = observe(state, market, spec)
        assert observation.shape == (10,) and np.all(np.isfinite(observation))
        # Causality: any bucket at or after ``t - data_delay_buckets`` is invisible here.
        for bucket in range(t - spec.data_delay_buckets, market.n_after):
            if market.has_bucket(bucket):
                altered = market.with_bucket(bucket, event_count=1, total_quantity=1,
                                             signed_quantity=0, vwap_ticks=1)
                np.testing.assert_array_equal(observe(state, altered, spec), observation)
        state, result = step(state, action, market, spec)
        assert_step_invariants(states[-1], state, result, market, spec)
        states.append(state)
        results.append(result)
        if result.terminated or result.truncated:
            break

    final, last = states[-1], results[-1]
    assert sum(1 for r in results if r.terminated or r.truncated) == 1
    total_risk = sum(r.risk_component for r in results)
    if last.terminated:
        terminal = last.terminal_price_ticks if last.terminal_price_ticks else None
        shortfall = implementation_shortfall(final, spec, terminal)
    else:  # Truncated: the shortfall of the fills so far, with no residual term.
        shortfall = spec.side * (
            final.exec_notional_ticks - final.arrival_price_ticks * final.filled_qty_ticks
        ) + final.fees_ticks
    assert sum(r.reward_ticks for r in results) == -shortfall - total_risk

    # Deterministic replay: the same inputs reproduce the sequences byte for byte.
    replay_state = reset(spec, market)
    replay_states, replay_results = [replay_state], []
    for action in actions[: len(results)]:
        replay_state, replay_result = step(replay_state, action, market, spec)
        replay_states.append(replay_state)
        replay_results.append(replay_result)
    assert replay_states == states and replay_results == results
    assert frozen_bytes(replay_states + replay_results) == frozen_bytes(states + results)


@given(episodes())
@settings(max_examples=150, deadline=None)
def test_latency_one_uses_the_next_bucket_price(
    episode: tuple[EpisodeSpec, MarketSlice, list[float]]
) -> None:
    spec, market, actions = episode
    state = reset(spec, market)
    for action in actions:
        if is_done(state, spec):
            break
        state, result = step(state, action, market, spec)
        assert result.fill_bucket == result.step + spec.latency_buckets
        if result.fill_qty_ticks:
            i = market.index(result.fill_bucket)
            assert result.fill_price_ticks == (
                int(market.vwap_ticks[i]) + spec.side * spec.slippage_ticks
            )
        if result.terminated or result.truncated:
            break
