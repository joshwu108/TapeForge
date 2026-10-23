"""Signal-free baseline policies: TWAP, immediate, and a participation-cap-aware TWAP.

Each baseline is a plain function ``Observation -> action`` for
``runner.run_episode``. They read the exact integers on the observation
(remaining quantity, time remaining, the newest visible bucket's quantity)
and convert a quantity to the simulator's fractional action with
``simulator.action_for_quantity``, so the submitted quantity is exactly the
one computed here. None of them looks at prices; they are the reference
points a learned policy has to beat, not strategies.

* ``twap``: submit ``remaining // time_remaining`` each step (1/H, then
  1/(H-1) of what is left, ...). The last decision submits everything left,
  so TWAP completes exactly whenever the fill rule lets it.
* ``immediate``: submit everything at every step; the participation cap is
  what spreads it out.
* ``cap_aware_twap``: TWAP, raised to whatever the estimated future liquidity
  cannot absorb. The liquidity estimate is the participation cap of the
  newest visible bucket (``floor(cap_bps * quantity / 10_000)``), assumed to
  persist: with ``k`` decisions after this one, the future can absorb about
  ``k * cap_estimate``, so ``max(0, remaining - k * cap_estimate)`` must be
  submitted now to stay feasible. When liquidity is deep the policy is TWAP;
  when the newest bucket was empty the estimate is 0 and everything is
  submitted (the fill is still capped by the real bucket). This definition is
  a proposal recorded for the owner, not an accepted specification.
"""

from __future__ import annotations

from typing import Final

from tapeforge.execution.runner import Observation, Policy
from tapeforge.execution.simulator import action_for_quantity
from tapeforge.execution.spec import participation_cap_ticks

__all__ = [
    "BASELINES",
    "cap_aware_twap",
    "cap_aware_twap_quantity",
    "immediate",
    "twap",
    "twap_quantity",
]


def twap_quantity(remaining_qty_ticks: int, time_remaining_buckets: int) -> int:
    """``remaining // time_remaining``: the TWAP share of what is left, in ticks.

    Raises:
        ValueError: no time remaining or a negative quantity.
    """
    if time_remaining_buckets < 1:
        raise ValueError(f"time_remaining_buckets must be positive, got {time_remaining_buckets}")
    if remaining_qty_ticks < 0:
        raise ValueError(f"remaining_qty_ticks must be non-negative, got {remaining_qty_ticks}")
    return remaining_qty_ticks // time_remaining_buckets


def twap(observation: Observation) -> float:
    """Time-weighted schedule over the remaining decisions (module docstring)."""
    quantity = twap_quantity(observation.remaining_qty_ticks, observation.time_remaining_buckets)
    return action_for_quantity(quantity, observation.remaining_qty_ticks)


def immediate(observation: Observation) -> float:
    """Submit the whole remaining quantity; fills are limited by the cap alone."""
    return 1.0


def cap_aware_twap_quantity(
    remaining_qty_ticks: int,
    time_remaining_buckets: int,
    newest_bucket_total_qty_ticks: int,
    participation_cap_bps: int,
) -> int:
    """TWAP share, raised to ``remaining - (time_remaining - 1) * cap_estimate`` when positive.

    Raises:
        ValueError: as ``twap_quantity``, or a negative bucket quantity.
    """
    share = twap_quantity(remaining_qty_ticks, time_remaining_buckets)
    if newest_bucket_total_qty_ticks < 0:
        raise ValueError(
            "newest_bucket_total_qty_ticks must be non-negative, "
            f"got {newest_bucket_total_qty_ticks}"
        )
    cap_estimate = participation_cap_ticks(newest_bucket_total_qty_ticks, participation_cap_bps)
    future_capacity = (time_remaining_buckets - 1) * cap_estimate
    needed_now = max(0, remaining_qty_ticks - future_capacity)
    return min(remaining_qty_ticks, max(share, needed_now))


def cap_aware_twap(observation: Observation) -> float:
    """TWAP that front-loads what the estimated future liquidity cannot absorb."""
    quantity = cap_aware_twap_quantity(
        observation.remaining_qty_ticks,
        observation.time_remaining_buckets,
        observation.newest_bucket_total_qty_ticks,
        observation.participation_cap_bps,
    )
    return action_for_quantity(quantity, observation.remaining_qty_ticks)


# Names used in run records; a learned policy is recorded under its own name.
BASELINES: Final[dict[str, Policy]] = {
    "twap": twap,
    "immediate": immediate,
    "cap_aware_twap": cap_aware_twap,
}
