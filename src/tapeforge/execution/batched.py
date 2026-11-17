"""Batched tensor implementation of the reference transition rules.

``step`` and ``observe`` apply ``simulator.step`` and ``simulator.observe`` to
many independent episodes at once, on whatever device the tensors live on.
The rules are the reference simulator's (``SIMULATOR_VERSION``); this module
defines none of its own, and ``execution.parity`` compares the two after
every transition. The CPU reference does not import this module.

Layout:
    ``EpisodeBatch`` holds ``K`` episodes: their slices padded to a common
    length as ``[K, N]`` int64 tensors, their specs as ``[K]`` tensors, and
    per-episode constants. ``BatchedState`` holds ``B`` environments, each
    pointing at one episode (``episode``), so many environments can share a
    bank of episodes and an environment is reset by pointing it elsewhere
    (``reset_where``). All accounting tensors are int64.

Division of work:
    Everything that depends only on immutable inputs is computed once, at
    construction, by the reference functions themselves: the arrival price
    and its age (``simulator.reset``, including its history and int64 bound
    checks), the terminal-mark bucket (``simulator.terminal_mark_bucket``)
    and the settlement price (``spec.settlement_price_ticks``). Stepping and
    observing are tensor code and touch no host memory: no ``.item()``, no
    ``.cpu()``, no Python branch on a tensor value. The one exception is
    ``step(..., validate=True)``, which checks the action range with a single
    host synchronisation so an invalid action raises as it does in the
    reference; a caller that bounds actions on the device passes
    ``validate=False``.

Supported geometry and limits:
    * Episodes may differ in every spec field (side, target, horizon,
      latency, data delay, fees, caps, terminal rule) and in slice length.
    * Prices, bucket quantities and targets must be below 2**53. The
      reference converts the action with ``floor(float64(action) * remaining)``
      and builds observations from integer ratios; below 2**53 the integers
      convert to float64 exactly, so the tensor arithmetic performs the same
      IEEE operations and gives the same quantity ticks and observation
      values. Actions must be float64 tensors: a float32 policy output has
      to be widened by the caller, exactly as the reference widens it.
    * A settling episode whose settlement price is not positive is rejected
      at construction. The reference raises for it only when the settlement
      is reached; rejecting early never changes a transition.
    * Stepping an ended environment is a masked no-op (zeros in the result,
      state unchanged) where the reference raises ``EpisodeOver``.
    * Memory is ``K * N * 41`` bytes for the padded slices (four int64
      feature columns, one int64 index column and one bool column) plus 138
      bytes of constants per episode; ``EpisodeBatch.from_episodes`` refuses
      to exceed ``memory_limit_bytes``.

What parity means:
    Agreement with the reference on identical inputs. It says nothing about
    realism: replaying recorded market data does not model the market's
    response to the agent's own orders, in either implementation.

Dependencies:
    ``torch`` (``pip install -e '.[torch]'``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Final

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.batched needs torch; install it with pip install -e '.[torch]'"
    ) from exc

from tapeforge.execution import simulator
from tapeforge.execution.simulator import OBSERVATION_FIELDS, terminal_mark_bucket
from tapeforge.execution.spec import (
    BPS_DENOMINATOR,
    EpisodeSpec,
    participation_cap_ticks,
    settlement_price_ticks,
)
from tapeforge.execution.state import MarketSlice

__all__ = [
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "EXACT_FLOAT_LIMIT",
    "BatchedState",
    "BatchedStepResult",
    "EpisodeBatch",
    "observe",
    "remaining_qty_ticks",
    "reset",
    "reset_where",
    "step",
]

# Integers below this convert to float64 exactly; see "Supported geometry".
EXACT_FLOAT_LIMIT: Final[int] = 2**53
DEFAULT_MEMORY_LIMIT_BYTES: Final[int] = 4 * 1024**3
_BYTES_PER_BUCKET: Final[int] = 4 * 8 + 8 + 1      # feature columns, last_priced, priced
_BYTES_PER_EPISODE: Final[int] = 17 * 8 + 2        # int64 constants and the two bool flags

Tensor = torch.Tensor


@dataclass(frozen=True, eq=False)
class EpisodeBatch:
    """Immutable inputs of ``K`` episodes on one device; build with ``from_episodes``.

    Slice tensors are ``[K, N]`` with ``N`` the longest slice; padding
    buckets are zero and unpriced, and ``n_buckets`` marks where each slice
    really ends. Spec and constant tensors are ``[K]`` int64 unless noted.

    Attributes:
        event_count: Records per bucket.
        total_quantity: Quantity ticks per bucket.
        signed_quantity: Taker-signed quantity ticks per bucket.
        vwap_ticks: Bucket VWAP in price ticks; 0 where no quantity traded.
        priced: bool; the bucket has records and quantity.
        last_priced: Array index of the newest priced bucket at or before
            each position, or -1 when there is none.
        n_buckets: Real length of each slice.
        origin: Buckets before the first decision bucket.
        side, target_qty_ticks, horizon_buckets, participation_cap_bps,
        slippage_ticks, fee_bps, latency_buckets, data_delay_buckets,
        risk_lambda, residual_mu: The ``EpisodeSpec`` fields.
        settles: bool; the terminal rule is ``terminal_settlement``.
        arrival_price_ticks: Arrival price fixed at reset.
        arrival_age: Age of the arrival bucket (the initial mark age).
        has_terminal_mark: bool; the slice holds a terminal-mark bucket.
        terminal_mark_ticks: Its VWAP; 0 without one.
        terminal_cap_qty_ticks: Its participation cap; 0 without one.
        settlement_price_ticks: The surcharged mark a settling episode
            settles at; 0 for a legacy-rule episode or without a mark.
    """

    event_count: Tensor
    total_quantity: Tensor
    signed_quantity: Tensor
    vwap_ticks: Tensor
    priced: Tensor
    last_priced: Tensor
    n_buckets: Tensor
    origin: Tensor
    side: Tensor
    target_qty_ticks: Tensor
    horizon_buckets: Tensor
    participation_cap_bps: Tensor
    slippage_ticks: Tensor
    fee_bps: Tensor
    latency_buckets: Tensor
    data_delay_buckets: Tensor
    risk_lambda: Tensor
    residual_mu: Tensor
    settles: Tensor
    arrival_price_ticks: Tensor
    arrival_age: Tensor
    has_terminal_mark: Tensor
    terminal_mark_ticks: Tensor
    terminal_cap_qty_ticks: Tensor
    settlement_price_ticks: Tensor

    @property
    def n_episodes(self) -> int:
        """Return ``K``, the number of episodes."""
        return int(self.n_buckets.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device every tensor lives on."""
        return self.n_buckets.device

    @property
    def nbytes(self) -> int:
        """Return the bytes of tensor storage this batch occupies."""
        return sum(
            getattr(self, f.name).numel() * getattr(self, f.name).element_size()
            for f in fields(self)
        )

    @staticmethod
    def estimate_bytes(n_episodes: int, max_buckets: int) -> int:
        """Return the storage ``from_episodes`` needs for ``K`` episodes of ``N`` buckets."""
        return n_episodes * (max_buckets * _BYTES_PER_BUCKET + _BYTES_PER_EPISODE)

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[tuple[EpisodeSpec, MarketSlice]],
        device: str | torch.device = "cpu",
        memory_limit_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
    ) -> EpisodeBatch:
        """Validate ``episodes`` with the reference functions and move them to ``device``.

        Raises:
            ValueError: No episodes; the reference ``reset`` rejects one (too
                little history, no arrival price); a value is not below
                2**53; a settling episode's settlement price is not positive;
                or the batch would exceed ``memory_limit_bytes``.
            OverflowError: The reference int64 bound check fails for one.
        """
        if not episodes:
            raise ValueError("a batch needs at least one episode")
        n_max = max(market.n_buckets for _, market in episodes)
        needed = cls.estimate_bytes(len(episodes), n_max)
        if memory_limit_bytes is not None and needed > memory_limit_bytes:
            raise ValueError(
                f"{len(episodes)} episodes of up to {n_max} buckets need {needed} bytes, above "
                f"the limit of {memory_limit_bytes}; use fewer episodes or raise the limit"
            )
        columns = {
            name: np.zeros((len(episodes), n_max), dtype=np.int64)
            for name in ("event_count", "total_quantity", "signed_quantity", "vwap_ticks")
        }
        constants: dict[str, list[int]] = {
            name: [] for name in (
                "n_buckets", "origin", "side", "target_qty_ticks", "horizon_buckets",
                "participation_cap_bps", "slippage_ticks", "fee_bps", "latency_buckets",
                "data_delay_buckets", "risk_lambda", "residual_mu", "settles",
                "arrival_price_ticks", "arrival_age", "has_terminal_mark",
                "terminal_mark_ticks", "terminal_cap_qty_ticks", "settlement_price_ticks",
            )
        }
        for row, (spec, market) in enumerate(episodes):
            initial = simulator.reset(spec, market)   # history, arrival and int64 bound checks
            largest = max(
                spec.target_qty_ticks, int(market.total_quantity.max()),
                int(market.vwap_ticks.max()),
            )
            if largest >= EXACT_FLOAT_LIMIT:
                raise ValueError(
                    f"episode {row} holds a price, quantity or target of {largest}, not below "
                    "2**53; the batched simulator supports only values float64 represents exactly"
                )
            n = market.n_buckets
            for name, column in columns.items():
                column[row, :n] = getattr(market, name)
            mark_bucket = terminal_mark_bucket(market, spec)
            mark = cap = settlement = 0
            if mark_bucket is not None:
                index = market.index(mark_bucket)
                mark = int(market.vwap_ticks[index])
                cap = participation_cap_ticks(
                    int(market.total_quantity[index]), spec.participation_cap_bps
                )
                if spec.settles_at_horizon:
                    settlement = settlement_price_ticks(
                        mark, spec.side, spec.terminal_surcharge_bps
                    )
            values = {
                "n_buckets": n, "origin": market.origin, "side": spec.side,
                "target_qty_ticks": spec.target_qty_ticks,
                "horizon_buckets": spec.horizon_buckets,
                "participation_cap_bps": spec.participation_cap_bps,
                "slippage_ticks": spec.slippage_ticks, "fee_bps": spec.fee_bps,
                "latency_buckets": spec.latency_buckets,
                "data_delay_buckets": spec.data_delay_buckets,
                "risk_lambda": spec.risk_lambda, "residual_mu": spec.residual_mu,
                "settles": int(spec.settles_at_horizon),
                "arrival_price_ticks": initial.arrival_price_ticks,
                "arrival_age": initial.mark_age,
                "has_terminal_mark": int(mark_bucket is not None),
                "terminal_mark_ticks": mark, "terminal_cap_qty_ticks": cap,
                "settlement_price_ticks": settlement,
            }
            for name, value in values.items():
                constants[name].append(value)

        priced = (columns["event_count"] > 0) & (columns["total_quantity"] > 0)
        positions = np.where(priced, np.arange(n_max, dtype=np.int64), -1)
        tensors: dict[str, Tensor] = {
            name: torch.from_numpy(column).to(device) for name, column in columns.items()
        }
        tensors["priced"] = torch.from_numpy(priced).to(device)
        tensors["last_priced"] = torch.from_numpy(np.maximum.accumulate(positions, axis=1)).to(
            device
        )
        for name, values_list in constants.items():
            tensor = torch.tensor(values_list, dtype=torch.int64, device=device)
            tensors[name] = tensor.bool() if name in ("settles", "has_terminal_mark") else tensor
        return cls(**tensors)


@dataclass(frozen=True, eq=False)
class BatchedState:
    """Accounting state of ``B`` environments; the fields of ``State`` as ``[B]`` int64 tensors.

    Two fields have no counterpart in ``State``: ``episode`` is the row of
    the ``EpisodeBatch`` each environment runs, and ``ended`` (bool) records
    that an earlier step terminated or truncated it. An ended environment is
    not stepped again until ``reset_where`` gives it a new episode.
    """

    episode: Tensor
    ended: Tensor
    step: Tensor
    inventory_ticks: Tensor
    cash_ticks: Tensor
    fill_fees_ticks: Tensor
    filled_qty_ticks: Tensor
    exec_notional_ticks: Tensor
    settled_qty_ticks: Tensor
    settlement_notional_ticks: Tensor
    settlement_fees_ticks: Tensor
    arrival_price_ticks: Tensor
    mark_price_ticks: Tensor
    mark_age: Tensor


@dataclass(frozen=True, eq=False)
class BatchedStepResult:
    """One decision of ``B`` environments; the fields of ``StepResult`` as ``[B]`` tensors.

    ``settlement_exceeds_cap``, ``terminated`` and ``truncated`` are bool,
    the rest int64. ``active`` (bool) marks the environments that were
    actually stepped; every other field is zero or ``False`` where it is
    unset.
    """

    active: Tensor
    step: Tensor
    fill_bucket: Tensor
    submitted_qty_ticks: Tensor
    cap_qty_ticks: Tensor
    fill_qty_ticks: Tensor
    fill_price_ticks: Tensor
    notional_ticks: Tensor
    fee_ticks: Tensor
    cost_component: Tensor
    risk_component: Tensor
    terminal_component: Tensor
    terminal_price_ticks: Tensor
    settlement_qty_ticks: Tensor
    settlement_price_ticks: Tensor
    settlement_notional_ticks: Tensor
    settlement_fee_ticks: Tensor
    settlement_cap_qty_ticks: Tensor
    settlement_exceeds_cap: Tensor
    reward_ticks: Tensor
    terminated: Tensor
    truncated: Tensor


def _floor_div(numerator: Tensor, denominator: Tensor | int) -> Tensor:
    """Return the exact integer floor quotient (Python ``//`` on int64 tensors)."""
    return torch.div(numerator, denominator, rounding_mode="floor")


def _fee(notional: Tensor, fee_bps: Tensor) -> Tensor:
    """Return ``ceil(notional * fee_bps / 10_000)`` as ``spec.fee_ticks`` computes it."""
    return -_floor_div(-(notional * fee_bps), BPS_DENOMINATOR)


def reset(batch: EpisodeBatch, episode: Tensor) -> BatchedState:
    """Return the initial state of environments running the episodes ``episode`` (``[B]`` int64).

    Zero cash, the side's initial inventory, and the arrival price and age
    the reference ``reset`` computed at construction.
    """
    zeros = torch.zeros_like(episode)
    target, side = batch.target_qty_ticks[episode], batch.side[episode]
    arrival = batch.arrival_price_ticks[episode]
    return BatchedState(
        episode=episode,
        ended=torch.zeros_like(episode, dtype=torch.bool),
        step=zeros,
        inventory_ticks=torch.where(side > 0, zeros, target),
        cash_ticks=zeros,
        fill_fees_ticks=zeros,
        filled_qty_ticks=zeros,
        exec_notional_ticks=zeros,
        settled_qty_ticks=zeros,
        settlement_notional_ticks=zeros,
        settlement_fees_ticks=zeros,
        arrival_price_ticks=arrival,
        mark_price_ticks=arrival,
        mark_age=batch.arrival_age[episode],
    )


def reset_where(
    state: BatchedState, mask: Tensor, episode: Tensor, batch: EpisodeBatch
) -> BatchedState:
    """Return ``state`` with the environments in ``mask`` restarted on ``episode``.

    ``episode`` is ``[B]``; its entries outside ``mask`` are ignored. The
    other environments keep their state bit for bit.
    """
    fresh = reset(batch, torch.where(mask, episode, state.episode))
    return BatchedState(**{
        f.name: torch.where(mask, getattr(fresh, f.name), getattr(state, f.name))
        for f in fields(BatchedState)
    })


def remaining_qty_ticks(state: BatchedState, batch: EpisodeBatch) -> Tensor:
    """Return the quantity still to trade: target minus filled minus settled."""
    return (
        batch.target_qty_ticks[state.episode] - state.filled_qty_ticks - state.settled_qty_ticks
    )


def observe(state: BatchedState, batch: EpisodeBatch) -> tuple[Tensor, Tensor]:
    """Return the float64 observations (``[B, 10]``, ``OBSERVATION_FIELDS``) and their validity.

    Row ``i`` equals ``simulator.observe`` of environment ``i``: it reads the
    bucket ``step - 1 - data_delay_buckets`` and nothing newer. The second
    tensor (bool ``[B]``) is ``False`` where that bucket lies past the end
    of the slice, which only happens for an ended environment; the reference
    raises ``IndexError`` there and the row holds no meaningful values.
    """
    episode = state.episode
    target = batch.target_qty_ticks[episode].double()
    horizon = batch.horizon_buckets[episode]
    arrival = state.arrival_price_ticks.double()
    newest = state.step - 1 - batch.data_delay_buckets[episode] + batch.origin[episode]
    available = newest < batch.n_buckets[episode]
    index = torch.minimum(newest, batch.n_buckets[episode] - 1)
    total = batch.total_quantity[episode, index]
    traded = total > 0
    total_safe = torch.where(traded, total, torch.ones_like(total)).double()
    zero = torch.zeros_like(arrival)
    observation = torch.stack(
        [
            remaining_qty_ticks(state, batch).double() / target,
            (horizon - state.step).double() / horizon.double(),
            batch.side[episode].double(),
            state.mark_price_ticks.double() / arrival - 1.0,
            state.mark_age.double(),
            (batch.event_count[episode, index] > 0).double(),
            batch.event_count[episode, index].double(),
            total.double() / target,
            torch.where(traded, batch.signed_quantity[episode, index].double() / total_safe, zero),
            torch.where(traded, batch.vwap_ticks[episode, index].double() / arrival - 1.0, zero),
        ],
        dim=1,
    )
    assert observation.shape[1] == len(OBSERVATION_FIELDS)
    return observation, available


def step(
    state: BatchedState, actions: Tensor, batch: EpisodeBatch, *, validate: bool = True
) -> tuple[BatchedState, BatchedStepResult]:
    """Apply one decision to every environment that has not ended.

    ``actions`` is a float64 ``[B]`` tensor on the batch's device: the
    fraction of the remaining quantity to submit, in ``[0, 1]``. Environments
    whose episode ended earlier, or that are done by the reference
    ``is_done``, are left untouched and report zeros.

    Args:
        state: Current state.
        actions: Fractions in ``[0, 1]``.
        batch: The episodes ``state.episode`` indexes.
        validate: Check that every action is finite and in ``[0, 1]`` and
            raise otherwise. This reads one boolean back from the device
            (a host synchronisation per step). With ``False`` the caller
            guarantees the range; out-of-range values then give undefined
            quantities rather than an error.

    Raises:
        TypeError: ``actions`` is not a float64 tensor of shape ``[B]`` on
            the batch's device.
        ValueError: ``validate`` and an action is not a finite fraction in
            ``[0, 1]``.
    """
    if actions.dtype != torch.float64 or actions.shape != state.step.shape:
        raise TypeError(
            f"actions must be a float64 tensor of shape {tuple(state.step.shape)}, got "
            f"{actions.dtype} {tuple(actions.shape)}; widen float32 policy outputs explicitly"
        )
    if actions.device != state.step.device:
        raise TypeError(f"actions are on {actions.device}, the state on {state.step.device}")
    if validate and bool(torch.any(~((actions >= 0.0) & (actions <= 1.0)))):
        raise ValueError("actions must be finite fractions in [0, 1]")

    episode = state.episode
    side, target = batch.side[episode], batch.target_qty_ticks[episode]
    horizon, fee_bps = batch.horizon_buckets[episode], batch.fee_bps[episode]
    origin, n_buckets = batch.origin[episode], batch.n_buckets[episode]
    arrival = state.arrival_price_ticks
    zeros = torch.zeros_like(state.step)

    remaining_before = target - state.filled_qty_ticks - state.settled_qty_ticks
    active = ~state.ended & (remaining_before > 0) & (state.step < horizon)
    # The only float-to-integer conversion, as in the reference: floor(action * remaining).
    submitted = torch.floor(actions * remaining_before.double()).to(torch.int64)
    new_step = state.step + 1
    fill_bucket = state.step + batch.latency_buckets[episode]

    fill_index = fill_bucket + origin
    in_slice = fill_index < n_buckets
    fill_index = torch.minimum(fill_index, n_buckets - 1)
    priced = in_slice & batch.priced[episode, fill_index]
    cap = torch.where(
        priced,
        _floor_div(
            batch.total_quantity[episode, fill_index] * batch.participation_cap_bps[episode],
            BPS_DENOMINATOR,
        ),
        zeros,
    )
    fill = torch.minimum(submitted, cap)
    filled = fill > 0
    price = torch.where(
        filled, batch.vwap_ticks[episode, fill_index] + side * batch.slippage_ticks[episode], zeros
    )
    notional = price * fill
    fee = torch.where(filled, _fee(notional, fee_bps), zeros)

    remaining_after = remaining_before - fill
    cost = side * (notional - arrival * fill) + fee
    risk = batch.risk_lambda[episode] * remaining_after

    truncated = ~in_slice
    terminated = in_slice & ((remaining_after == 0) | (new_step == horizon))
    needs_terminal = terminated & (remaining_after > 0)
    has_mark = batch.has_terminal_mark[episode]
    truncated = truncated | (needs_terminal & ~has_mark)
    terminated = terminated & ~(needs_terminal & ~has_mark)
    settles = batch.settles[episode]
    settle = needs_terminal & has_mark & settles
    legacy = needs_terminal & has_mark & ~settles
    mark = batch.terminal_mark_ticks[episode]

    settled_qty = torch.where(settle, remaining_after, zeros)
    settlement_price = torch.where(settle, batch.settlement_price_ticks[episode], zeros)
    settlement_notional = settlement_price * settled_qty
    settlement_fee = _fee(settlement_notional, fee_bps)
    settlement_cap = torch.where(settle, batch.terminal_cap_qty_ticks[episode], zeros)
    terminal = torch.where(
        settle,
        side * (settlement_notional - arrival * settled_qty) + settlement_fee,
        torch.where(
            legacy,
            side * (mark - arrival) * remaining_after
            + batch.residual_mu[episode] * remaining_after,
            zeros,
        ),
    )
    terminal_price = torch.where(settle | legacy, mark, zeros)

    # Mark after the step: the newest priced bucket at or before new_step - 1 - data_delay.
    # Buckets past the end of the slice still count toward the age, and a search longer
    # than the slice finds nothing, as in the reference's bounded loop.
    newest = new_step - 1 - batch.data_delay_buckets[episode] + origin
    found_at = batch.last_priced[episode, torch.minimum(newest, n_buckets - 1)]
    age = newest - found_at
    found = (found_at >= 0) & (age < n_buckets)
    mark_price = torch.where(
        found, batch.vwap_ticks[episode, torch.clamp(found_at, min=0)], state.mark_price_ticks
    )
    mark_age = torch.where(found, age, state.mark_age + 1)

    def kept(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(active, new, old)

    def shown(value: Tensor) -> Tensor:
        return torch.where(active, value, torch.zeros_like(value))

    new_state = BatchedState(
        episode=episode,
        ended=state.ended | (active & (terminated | truncated)),
        step=kept(new_step, state.step),
        inventory_ticks=kept(
            state.inventory_ticks + side * (fill + settled_qty), state.inventory_ticks
        ),
        cash_ticks=kept(
            state.cash_ticks - side * (notional + settlement_notional) - fee - settlement_fee,
            state.cash_ticks,
        ),
        fill_fees_ticks=kept(state.fill_fees_ticks + fee, state.fill_fees_ticks),
        filled_qty_ticks=kept(state.filled_qty_ticks + fill, state.filled_qty_ticks),
        exec_notional_ticks=kept(state.exec_notional_ticks + notional, state.exec_notional_ticks),
        settled_qty_ticks=kept(state.settled_qty_ticks + settled_qty, state.settled_qty_ticks),
        settlement_notional_ticks=kept(
            state.settlement_notional_ticks + settlement_notional, state.settlement_notional_ticks
        ),
        settlement_fees_ticks=kept(
            state.settlement_fees_ticks + settlement_fee, state.settlement_fees_ticks
        ),
        arrival_price_ticks=arrival,
        mark_price_ticks=kept(mark_price, state.mark_price_ticks),
        mark_age=kept(mark_age, state.mark_age),
    )
    result = BatchedStepResult(
        active=active,
        step=shown(state.step),
        fill_bucket=shown(fill_bucket),
        submitted_qty_ticks=shown(submitted),
        cap_qty_ticks=shown(cap),
        fill_qty_ticks=shown(fill),
        fill_price_ticks=shown(price),
        notional_ticks=shown(notional),
        fee_ticks=shown(fee),
        cost_component=shown(cost),
        risk_component=shown(risk),
        terminal_component=shown(terminal),
        terminal_price_ticks=shown(terminal_price),
        settlement_qty_ticks=shown(settled_qty),
        settlement_price_ticks=shown(settlement_price),
        settlement_notional_ticks=shown(settlement_notional),
        settlement_fee_ticks=shown(settlement_fee),
        settlement_cap_qty_ticks=shown(settlement_cap),
        settlement_exceeds_cap=active & (settled_qty > settlement_cap),
        reward_ticks=shown(-(cost + risk + terminal)),
        terminated=active & terminated,
        truncated=active & truncated,
    )
    return new_state, result
