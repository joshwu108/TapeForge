"""Reference execution simulator: pure transitions with exact integer accounting.

The agent must trade ``target_qty_ticks`` (Q*) within ``horizon_buckets`` (H)
one-bucket decisions. Decision ``t`` is taken at the boundary that opens
bucket ``t``, may read buckets ``<= t - 1 - d`` (``d`` = data delay) and
executes in bucket ``t + l`` (``l`` = latency). ``reset``, ``observe`` and
``step`` are pure functions over frozen dataclasses: the same
``(spec, market, actions)`` always produces identical ``State`` and
``StepResult`` sequences.

Fill rule ``"bucket_vwap_capped"`` (synthetic, the only rule implemented):
the submitted quantity fills at the execution bucket's VWAP rounded to a
price tick, plus ``side * slippage_ticks``, capped at ``participation_cap_bps``
of that bucket's total quantity; nothing fills in a bucket with no records or
no quantity. It does not model market impact, the market's reaction to the
agent, passive orders or queue position; a taker order would not actually get
the bucket VWAP. The rule is named so a best-bid/ask fill-at-touch rule can
be added beside it once quote data exists.

Accounting (cash ticks = price ticks x quantity ticks): ``cash -= side *
notional + fee``, ``inventory += side * fill``, ``fee = ceil(notional *
fee_bps / 10_000)``. The fee ceiling is the only rounding; the action is the
only float and is converted once, by ``floor(action * remaining)``.

Reward per step: ``-(cost + risk + terminal)`` where ``cost = side *
(notional - P_0 * fill) + fee``, ``risk = lambda * remaining_after`` and
``terminal`` is the Perold residual term ``side * (P_T - P_0) * residual + mu
* residual`` at the horizon. Summed over a terminated episode this is exactly
``-IS - lambda * sum(remaining_after)`` with ``IS`` the implementation
shortfall against the arrival price ``P_0``.

Flags: ``remaining == 0`` or ``step == H`` terminates. A fill bucket or a
terminal mark that the slice does not contain truncates instead (the tape
ran out before the task did); never both.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from tapeforge.execution.spec import (
    FILL_RULE_BUCKET_VWAP_CAPPED,
    EpisodeSpec,
    check_episode_bounds,
    fee_ticks,
    participation_cap_ticks,
)
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "OBSERVATION_FIELDS",
    "EpisodeOver",
    "action_for_quantity",
    "implementation_shortfall",
    "is_done",
    "newest_visible_bucket",
    "observe",
    "remaining_qty_ticks",
    "reset",
    "residual_fixed_penalty",
    "residual_forced_liquidation",
    "residual_perold",
    "shortfall_bps",
    "step",
    "terminal_mark",
]

# Layout of the observation vector returned by ``observe`` (float64).
OBSERVATION_FIELDS: Final[tuple[str, ...]] = (
    "remaining_fraction",        # remaining / Q*
    "time_remaining_fraction",   # (H - step) / H
    "side",                      # +1 buy, -1 sell
    "mark_return",               # mark_price / P_0 - 1
    "mark_age",                  # visible buckets since the mark bucket
    "bucket_valid",              # newest visible bucket had records
    "bucket_event_count",        # its record count
    "bucket_quantity_fraction",  # its total quantity / Q*
    "bucket_signed_fraction",    # its signed / total quantity (0 when no quantity)
    "bucket_return",             # its VWAP / P_0 - 1 (0 when no quantity)
)


class EpisodeOver(RuntimeError):
    """``step`` was called after the target was filled or the horizon ended."""


# --- episode geometry ---------------------------------------------------------


def newest_visible_bucket(state_step: int, spec: EpisodeSpec) -> int:
    """The latest episode bucket a decision taken after ``state_step`` steps may read."""
    return state_step - 1 - spec.data_delay_buckets


def remaining_qty_ticks(state: State, spec: EpisodeSpec) -> int:
    """Quantity still to trade: ``Q* - filled``; never negative."""
    return spec.target_qty_ticks - state.filled_qty_ticks


def is_done(state: State, spec: EpisodeSpec) -> bool:
    """True once the target is filled or ``H`` decisions have been taken.

    Truncation is reported on the ``StepResult`` only: a state whose tape ran
    out mid-episode is not ``done`` here, and stepping it truncates again.
    """
    return remaining_qty_ticks(state, spec) == 0 or state.step >= spec.horizon_buckets


def _priced(market: MarketSlice, index: int) -> bool:
    # A bucket can be valid (it had records) yet carry no quantity; its VWAP is
    # 0 by convention and must never be used as a price.
    return bool(market.valid[index]) and int(market.total_quantity[index]) > 0


def _latest_priced_bucket(
    market: MarketSlice, newest: int, lookback: int
) -> tuple[int, int] | None:
    """Price and age of the newest priced bucket in ``newest, newest - 1, ...``.

    Searches at most ``lookback`` buckets, skips buckets past the end of the
    slice (they count toward the age) and stops at its start.
    """
    for age in range(lookback):
        bucket = newest - age
        if bucket < -market.origin:
            return None
        if bucket >= market.n_after:
            continue
        i = market.index(bucket)
        if _priced(market, i):
            return int(market.vwap_ticks[i]), age
    return None


def terminal_mark(market: MarketSlice, spec: EpisodeSpec) -> int | None:
    """P_T: VWAP of the first priced bucket after the last fill bucket.

    The search starts at bucket ``H + l`` and covers ``arrival_lookback``
    buckets or the end of the slice, whichever comes first. ``None`` when the
    slice holds no such bucket, which truncates an episode that still has
    residual inventory.
    """
    first = spec.horizon_buckets + spec.latency_buckets
    for bucket in range(first, first + spec.arrival_lookback):
        if not market.has_bucket(bucket):
            return None
        i = market.index(bucket)
        if _priced(market, i):
            return int(market.vwap_ticks[i])
    return None


# --- reset / observe / step -----------------------------------------------------


def reset(spec: EpisodeSpec, market: MarketSlice) -> State:
    """Initial state: zero cash, initial inventory, arrival price fixed.

    The arrival price ``P_0`` is the VWAP of the newest priced bucket at or
    before ``-1 - d`` within ``arrival_lookback`` buckets; its age is the
    initial ``mark_age``.

    Raises:
        ValueError: the slice has fewer than ``arrival_lookback + d`` buckets
            before the first decision, or none of them is priced.
        OverflowError: ``check_episode_bounds`` fails for this tape.
    """
    needed = spec.arrival_lookback + spec.data_delay_buckets
    if market.origin < needed:
        raise ValueError(
            f"the slice carries {market.origin} buckets before the first decision but "
            f"arrival lookback {spec.arrival_lookback} + data delay {spec.data_delay_buckets} "
            f"= {needed} are needed"
        )
    if market.n_buckets:
        check_episode_bounds(
            spec, int(market.vwap_ticks.max()), int(market.total_quantity.max())
        )
    found = _latest_priced_bucket(market, newest_visible_bucket(0, spec), spec.arrival_lookback)
    if found is None:
        raise ValueError(
            f"no priced bucket within {spec.arrival_lookback} buckets before the first "
            "decision; the arrival price is undefined"
        )
    arrival, age = found
    return State(
        step=0,
        inventory_ticks=spec.initial_inventory_ticks,
        cash_ticks=0,
        fees_ticks=0,
        filled_qty_ticks=0,
        exec_notional_ticks=0,
        arrival_price_ticks=arrival,
        mark_price_ticks=arrival,
        mark_age=age,
    )


def observe(state: State, market: MarketSlice, spec: EpisodeSpec) -> np.ndarray:
    """Observation vector (``OBSERVATION_FIELDS``) from buckets ``<= step - 1 - d`` only.

    The visible part of the tape is sliced off before anything is read, so a
    later bucket cannot leak in by construction. Floats are fine here: the
    observation is the policy's view, not the accounting.
    """
    newest = newest_visible_bucket(state.step, spec)
    end = market.index(newest) + 1
    count = market.event_count[:end]
    total = market.total_quantity[:end]
    signed = market.signed_quantity[:end]
    vwap = market.vwap_ticks[:end]
    valid = market.valid[:end]

    arrival = state.arrival_price_ticks
    bucket_total = int(total[-1])
    quantity_fraction = bucket_total / spec.target_qty_ticks
    signed_fraction = int(signed[-1]) / bucket_total if bucket_total else 0.0
    bucket_return = int(vwap[-1]) / arrival - 1.0 if bucket_total else 0.0
    return np.array(
        [
            remaining_qty_ticks(state, spec) / spec.target_qty_ticks,
            (spec.horizon_buckets - state.step) / spec.horizon_buckets,
            float(spec.side),
            state.mark_price_ticks / arrival - 1.0,
            float(state.mark_age),
            float(valid[-1]),
            float(count[-1]),
            quantity_fraction,
            signed_fraction,
            bucket_return,
        ],
        dtype=np.float64,
    )


def _submitted_qty(action: float, remaining: int) -> int:
    if isinstance(action, bool) or not isinstance(action, int | float | np.floating | np.integer):
        raise TypeError(f"action must be a real number in [0, 1], got {action!r}")
    value = float(action)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"action must be a finite fraction in [0, 1], got {value!r}")
    # The only float-to-integer conversion in the simulator.
    return math.floor(value * remaining)


def _fill_bucket_vwap_capped(
    market: MarketSlice, index: int, spec: EpisodeSpec, submitted: int
) -> tuple[int, int, int]:
    """Return ``(cap, fill, price)`` under the capped bucket-VWAP rule."""
    if not _priced(market, index):
        return 0, 0, 0
    cap = participation_cap_ticks(int(market.total_quantity[index]), spec.participation_cap_bps)
    fill = min(submitted, cap)
    price = int(market.vwap_ticks[index]) + spec.side * spec.slippage_ticks if fill else 0
    return cap, fill, price


def _mark_after(
    state: State, market: MarketSlice, spec: EpisodeSpec, new_step: int
) -> tuple[int, int]:
    found = _latest_priced_bucket(
        market, newest_visible_bucket(new_step, spec), market.n_buckets
    )
    if found is None:  # cannot happen after a successful reset; keep the old mark
        return state.mark_price_ticks, state.mark_age + 1
    return found


def step(
    state: State, action: float, market: MarketSlice, spec: EpisodeSpec
) -> tuple[State, StepResult]:
    """Apply one decision and return the new state and the step record.

    ``action`` is the fraction of the remaining quantity to submit, in
    ``[0, 1]``; it is floored to quantity ticks once and then every number is
    an integer. See the module docstring for the fill rule, accounting and
    reward decomposition.

    Raises:
        EpisodeOver: the target is filled or ``H`` decisions were taken.
        TypeError, ValueError: ``action`` is not a finite real in ``[0, 1]``.
    """
    if is_done(state, spec):
        raise EpisodeOver(
            f"episode is over after {state.step} steps with "
            f"{remaining_qty_ticks(state, spec)} ticks remaining; call reset for a new one"
        )
    remaining_before = remaining_qty_ticks(state, spec)
    submitted = _submitted_qty(action, remaining_before)
    new_step = state.step + 1
    fill_bucket = state.step + spec.latency_buckets
    side, arrival = spec.side, state.arrival_price_ticks

    cap = fill = price = notional = fee = 0
    truncated = not market.has_bucket(fill_bucket)
    if not truncated:
        if spec.fill_rule != FILL_RULE_BUCKET_VWAP_CAPPED:  # pragma: no cover - spec validates
            raise ValueError(f"unknown fill_rule {spec.fill_rule!r}")
        index = market.index(fill_bucket)
        cap, fill, price = _fill_bucket_vwap_capped(market, index, spec, submitted)
        if fill:
            notional = price * fill
            fee = fee_ticks(notional, spec.fee_bps)

    remaining_after = remaining_before - fill
    cost = side * (notional - arrival * fill) + fee
    risk = spec.risk_lambda * remaining_after

    terminated = not truncated and (remaining_after == 0 or new_step == spec.horizon_buckets)
    terminal = 0
    terminal_price = 0
    if terminated and remaining_after > 0:
        mark = terminal_mark(market, spec)
        if mark is None:
            terminated, truncated = False, True
        else:
            terminal_price = mark
            terminal = residual_perold(side, arrival, mark, remaining_after, spec.residual_mu)

    mark_price, mark_age = _mark_after(state, market, spec, new_step)
    new_state = State(
        step=new_step,
        inventory_ticks=state.inventory_ticks + side * fill,
        cash_ticks=state.cash_ticks - side * notional - fee,
        fees_ticks=state.fees_ticks + fee,
        filled_qty_ticks=state.filled_qty_ticks + fill,
        exec_notional_ticks=state.exec_notional_ticks + notional,
        arrival_price_ticks=arrival,
        mark_price_ticks=mark_price,
        mark_age=mark_age,
    )
    result = StepResult(
        step=state.step,
        fill_bucket=fill_bucket,
        submitted_qty_ticks=submitted,
        cap_qty_ticks=cap,
        fill_qty_ticks=fill,
        fill_price_ticks=price,
        notional_ticks=notional,
        fee_ticks=fee,
        cost_component=cost,
        risk_component=risk,
        terminal_component=terminal,
        terminal_price_ticks=terminal_price,
        reward_ticks=-(cost + risk + terminal),
        terminated=terminated,
        truncated=truncated,
    )
    return new_state, result


# --- shortfall and residual rules ----------------------------------------------


def residual_perold(side: int, arrival: int, terminal: int, residual: int, mu: int) -> int:
    """R2 (accepted): opportunity cost of the residual at ``P_T`` plus ``mu * residual``.

    ``side * (P_T - P_0) * residual`` is positive when the price moved against
    the agent; with ``mu == 0`` a favourable move makes the term negative, so
    inaction is rewarded. That is the documented reason ``mu`` is swept, not
    a bug.
    """
    return side * (terminal - arrival) * residual + mu * residual


def residual_forced_liquidation(
    side: int, arrival: int, terminal: int, residual: int, kappa: int, fee_bps: int
) -> int:
    """R1 (for comparison only): trade the residual at ``P_T + side * kappa`` with fees.

    Shortfall of that forced fill against ``P_0``: ``side * (notional - P_0 *
    residual) + fee``. Not used by ``step``; kept as a pure function so the
    worksheet table can be checked and the rules compared on the same tape.
    """
    notional = (terminal + side * kappa) * residual
    return side * (notional - arrival * residual) + fee_ticks(notional, fee_bps)


def residual_fixed_penalty(residual: int, mu: int) -> int:
    """R3 (for comparison only): ``mu * residual``, blind to where the price went."""
    return mu * residual


def implementation_shortfall(
    state: State, spec: EpisodeSpec, terminal_price_ticks: int | None = None
) -> int:
    """IS in cash ticks: ``side * (exec_notional - P_0 * filled) + fees + residual term``.

    Positive is a cost for both sides. The residual term is the accepted
    Perold rule at ``terminal_price_ticks`` with ``spec.residual_mu``; it is
    zero when nothing is left, in which case the price may be omitted.

    Raises:
        ValueError: residual inventory remains and no terminal price was given.
    """
    filled = state.filled_qty_ticks
    shortfall = spec.side * (state.exec_notional_ticks - state.arrival_price_ticks * filled)
    shortfall += state.fees_ticks
    residual = remaining_qty_ticks(state, spec)
    if residual == 0:
        return shortfall
    if terminal_price_ticks is None:
        raise ValueError(
            f"{residual} ticks remain unfilled; the shortfall needs a terminal price"
        )
    return shortfall + residual_perold(
        spec.side, state.arrival_price_ticks, terminal_price_ticks, residual, spec.residual_mu
    )


def shortfall_bps(shortfall_ticks: int, arrival_price_ticks: int, target_qty_ticks: int) -> float:
    """Shortfall as basis points of the arrival notional ``P_0 * Q*`` (reporting only)."""
    return shortfall_ticks * 10_000 / (arrival_price_ticks * target_qty_ticks)


def action_for_quantity(quantity_ticks: int, remaining_ticks: int) -> float:
    """Smallest action in ``[0, 1]`` whose floor conversion submits exactly ``quantity_ticks``.

    Baselines are written in quantities (TWAP submits ``Q* / H`` per step);
    ``floor(quantity / remaining * remaining)`` can land one tick short through
    floating-point rounding, so the ratio is nudged by ulps until the
    simulator's own conversion returns the requested quantity.

    Raises:
        ValueError: ``quantity_ticks`` is outside ``[0, remaining_ticks]``.
    """
    if not 0 <= quantity_ticks <= remaining_ticks:
        raise ValueError(f"quantity {quantity_ticks} must be in [0, {remaining_ticks}]")
    if remaining_ticks == 0 or quantity_ticks == remaining_ticks:
        return 1.0 if quantity_ticks else 0.0
    action = quantity_ticks / remaining_ticks
    while math.floor(action * remaining_ticks) < quantity_ticks:
        action = math.nextafter(action, 1.0)
    while math.floor(action * remaining_ticks) > quantity_ticks:
        action = math.nextafter(action, 0.0)
    return action
