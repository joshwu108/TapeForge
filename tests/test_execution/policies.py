"""Rollout helper and the fixed policies the simulator tests use as baselines."""

from __future__ import annotations

from collections.abc import Callable

from tapeforge.execution.simulator import (
    action_for_quantity,
    is_done,
    remaining_qty_ticks,
    reset,
    step,
)
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.state import MarketSlice, State, StepResult

Policy = Callable[[State, EpisodeSpec], float]


def rollout(
    spec: EpisodeSpec, market: MarketSlice, policy: Policy
) -> tuple[list[State], list[StepResult]]:
    """Run until a result is terminated or truncated. Returns every state (reset first)."""
    state = reset(spec, market)
    states, results = [state], []
    while not is_done(state, spec):
        state, result = step(state, policy(state, spec), market, spec)
        states.append(state)
        results.append(result)
        if result.terminated or result.truncated:
            break
    return states, results


def twap_policy(state: State, spec: EpisodeSpec) -> float:
    """Submit ``remaining / time_remaining``: 1/3, 1/2, 1 of what is left."""
    remaining = remaining_qty_ticks(state, spec)
    time_remaining = spec.horizon_buckets - state.step
    return action_for_quantity(remaining // time_remaining, remaining)


def schedule_policy(quantity: int) -> Policy:
    """Submit a fixed quantity per step (the worksheet's ``Q* / H`` schedule)."""

    def policy(state: State, spec: EpisodeSpec) -> float:
        remaining = remaining_qty_ticks(state, spec)
        return action_for_quantity(min(quantity, remaining), remaining)

    return policy


def immediate_policy(state: State, spec: EpisodeSpec) -> float:
    """Submit everything that is left; the participation cap decides the fill."""
    return 1.0


def zero_policy(state: State, spec: EpisodeSpec) -> float:
    """Never trade; isolates the residual term and the risk term."""
    return 0.0
