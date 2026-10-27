"""Test fixtures for the simulator: a rollout helper and fixed reference policies.

A ``Policy`` maps ``(State, EpisodeSpec)`` to the action fraction in ``[0, 1]``
that ``step`` expects. The policies here are deliberately simple so that the
expected fills, fees and rewards can be derived by hand.
"""

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
    """Run ``policy`` on ``market`` until a step reports terminated or truncated.

    Returns:
        The visited states, starting with the reset state, and the step results
        in order; ``len(states) == len(results) + 1``.
    """
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
    """Submit the floor of ``remaining / time_remaining`` at each decision.

    Over three decisions this is 1/3, 1/2 and all of what is left, so a target
    that divides evenly completes exactly at the horizon.
    """
    remaining = remaining_qty_ticks(state, spec)
    time_remaining = spec.horizon_buckets - state.step
    return action_for_quantity(remaining // time_remaining, remaining)


def schedule_policy(quantity: int) -> Policy:
    """Build a policy that submits ``quantity`` per step, capped at what remains.

    With ``quantity = target_qty_ticks // horizon_buckets`` this is the even
    schedule of the hand-verified reference fixture.
    """

    def policy(state: State, spec: EpisodeSpec) -> float:
        remaining = remaining_qty_ticks(state, spec)
        return action_for_quantity(min(quantity, remaining), remaining)

    return policy


def immediate_policy(state: State, spec: EpisodeSpec) -> float:
    """Submit everything that is left; the participation cap decides the fill."""
    return 1.0


def zero_policy(state: State, spec: EpisodeSpec) -> float:
    """Never trade, which isolates the residual-inventory and risk reward terms."""
    return 0.0
