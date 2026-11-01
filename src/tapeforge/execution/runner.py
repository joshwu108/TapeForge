"""Policy replay through episodes, summaries and JSON run records.

A *policy* is any callable ``Observation -> action`` with the action a
fraction of the remaining quantity in ``[0, 1]`` (``simulator.step``'s
domain). ``Observation`` carries the float vector a learned policy consumes
(``simulator.observe``) next to the exact integers behind it (remaining
quantity, time remaining, the newest visible bucket's quantity), so the
quantity-based baselines never round through a float. Everything the policy
sees is computed from buckets ``<= t - 1 - data_delay_buckets``; the runner
adds no information of its own.

``run_episode`` returns the full ``Trajectory`` (states, observations,
actions, step results); ``summarize_episode`` reduces it to integer
accounting plus the shortfall in basis points; ``summarize_set`` aggregates a
set of episodes with nearest-rank percentiles (``sorted[ceil(p * n) - 1]``,
no interpolation), so a plain loop can recompute every figure.

``run_policy`` samples a split role, replays one policy and writes a JSON
run record (configuration, seeds, split record, code revision, per-episode
draws and summaries, set summary) under a runs directory that Git ignores.

Invariants:
    Two runs with the same inputs produce identical records except for the
    ``created_utc`` and ``code_revision`` fields. This module performs no
    training.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from tapeforge.execution.sampling import EpisodeDraw, materialize, sample_episodes
from tapeforge.execution.simulator import (
    SIMULATOR_VERSION,
    fill_shortfall,
    implementation_shortfall,
    is_done,
    newest_visible_bucket,
    observe,
    remaining_qty_ticks,
    reset,
    settlement_shortfall,
    shortfall_bps,
    step,
)
from tapeforge.execution.spec import ACCEPTED_TARGET_FRACTIONS_BPS, EpisodeSpec
from tapeforge.execution.splits import Splits
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "RUN_RECORD_SCHEMA",
    "EpisodeSummary",
    "Observation",
    "Policy",
    "SetSummary",
    "Trajectory",
    "code_revision",
    "nearest_rank_percentile",
    "observation_for",
    "run_episode",
    "run_policy",
    "summarize_episode",
    "summarize_set",
    "write_run_record",
]

# Schema tag of every run record. Version 2 adds the terminal rule, the
# settlement accounting and the split of fees and completion into ordinary
# fills and terminal settlement; version 1 records are not read by this code.
RUN_RECORD_SCHEMA: Final[str] = "tapeforge.execution.run/2"
TAIL_PERCENTILES: Final[tuple[int, ...]] = (50, 95, 99)  # Percentiles reported by SetSummary.


@dataclass(frozen=True, eq=False)
class Observation:
    """What a policy may see before one decision.

    The integer fields are exact and visible at the same time as ``vector``:
    the remaining quantity and horizon are episode state,
    ``newest_bucket_total_qty_ticks`` is the quantity of bucket
    ``t - 1 - data_delay_buckets`` (0 when it had none), and the cap rate
    lets a baseline estimate the next fill's liquidity limit.

    Attributes:
        vector: ``simulator.observe`` output (``OBSERVATION_FIELDS``,
            float64).
        remaining_qty_ticks: Quantity still to trade, in quantity ticks.
        time_remaining_buckets: Decisions left, ``horizon_buckets - step``.
        target_qty_ticks: The episode's target, in quantity ticks.
        side: ``SIDE_BUY`` or ``SIDE_SELL``.
        newest_bucket_total_qty_ticks: Total quantity of the newest visible
            bucket, in quantity ticks.
        participation_cap_bps: The spec's participation cap, in basis
            points.
    """

    vector: np.ndarray
    remaining_qty_ticks: int
    time_remaining_buckets: int
    target_qty_ticks: int
    side: int
    newest_bucket_total_qty_ticks: int
    participation_cap_bps: int


# A policy maps an observation to the fraction of the remaining quantity to submit.
Policy = Callable[[Observation], float]


def observation_for(state: State, market: MarketSlice, spec: EpisodeSpec) -> Observation:
    """Bundle the observation vector with the exact integers behind it."""
    newest = market.index(newest_visible_bucket(state.step, spec))
    return Observation(
        vector=observe(state, market, spec),
        remaining_qty_ticks=remaining_qty_ticks(state, spec),
        time_remaining_buckets=spec.horizon_buckets - state.step,
        target_qty_ticks=spec.target_qty_ticks,
        side=spec.side,
        newest_bucket_total_qty_ticks=int(market.total_quantity[newest]),
        participation_cap_bps=spec.participation_cap_bps,
    )


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Everything one replay produced.

    Attributes:
        spec: The episode's spec.
        states: ``states[0]`` is the reset state and ``states[i + 1]`` the
            state after decision ``i``; at most ``horizon_buckets + 1``
            entries.
        observations: The vector shown before each decision.
        actions: The float each decision submitted.
        results: The ``StepResult`` of each decision.
    """

    spec: EpisodeSpec
    states: tuple[State, ...]
    observations: tuple[np.ndarray, ...]
    actions: tuple[float, ...]
    results: tuple[StepResult, ...]

    @property
    def final_state(self) -> State:
        """Return the state after the last decision (the reset state if none was taken)."""
        return self.states[-1]

    @property
    def last_result(self) -> StepResult:
        """Return the step that ended the episode.

        Its ``terminated`` and ``truncated`` flags say how it ended.

        Raises:
            IndexError: The trajectory holds no step, which ``run_episode``
                never produces (a fresh episode always takes at least one).
        """
        return self.results[-1]


def run_episode(spec: EpisodeSpec, market: MarketSlice, policy: Policy) -> Trajectory:
    """Reset, then step the policy until the episode terminates or truncates.

    Raises:
        ValueError: ``reset`` finds no arrival price, or ``step`` rejects the
            action (not a finite fraction in ``[0, 1]``).
        OverflowError: The int64 bounds check in ``reset`` fails.
    """
    state = reset(spec, market)
    states, observations, actions, results = [state], [], [], []
    while not is_done(state, spec):
        observation = observation_for(state, market, spec)
        action = float(policy(observation))
        state, result = step(state, action, market, spec)
        observations.append(observation.vector)
        actions.append(action)
        states.append(state)
        results.append(result)
        if result.terminated or result.truncated:
            break
    return Trajectory(
        spec=spec,
        states=tuple(states),
        observations=tuple(observations),
        actions=tuple(actions),
        results=tuple(results),
    )


# --- summaries -----------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSummary:
    """Integer accounting of one episode plus the shortfall in basis points.

    Ordinary fills and the terminal settlement are kept apart everywhere, and
    completion is recorded twice: ``completed_by_fills`` says whether the
    ordinary fills alone reached the target, ``completed`` whether the
    target was reached by the end (fills plus settlement). A policy with
    ``completed`` on every episode but a large ``settlement_fraction``
    reached the target through the synthetic boundary condition, not
    through execution.

    Attributes:
        n_steps: Decisions taken.
        terminated: The episode ended by filling the target or reaching the
            horizon (with a terminal mark where one was needed).
        truncated: The tape ran out before the task did.
        terminal_rule: ``spec.terminal_rule``.
        terminal_surcharge_bps: ``spec.terminal_surcharge_bps``.
        arrival_price_ticks: Arrival price in price ticks.
        terminal_price_ticks: Terminal mark used for the terminal term; 0
            when none was needed or available.
        target_qty_ticks: Target quantity in quantity ticks.
        filled_qty_ticks: Quantity of ordinary fills, in quantity ticks.
        settled_qty_ticks: Quantity settled at the horizon, in quantity
            ticks.
        residual_qty_ticks: ``target - filled - settled``: quantity neither
            filled nor settled (legacy rule at the horizon, or truncation).
        completed_by_fills: ``filled_qty_ticks == target_qty_ticks``.
        completed: ``residual_qty_ticks == 0`` (final completion).
        filled_fraction: ``filled_qty_ticks / target_qty_ticks``.
        settlement_fraction: ``settled_qty_ticks / target_qty_ticks``.
        exec_notional_ticks: Cumulative ordinary-fill notional in cash ticks.
        settlement_price_ticks: Settlement price in price ticks; 0 without a
            settlement.
        settlement_notional_ticks: Settlement notional in cash ticks.
        settlement_exceeds_cap: The settlement traded more than the terminal
            bucket's participation cap.
        fill_fees_ticks: Fees of ordinary fills in cash ticks.
        settlement_fees_ticks: Fee of the settlement in cash ticks.
        fees_ticks: ``fill_fees_ticks + settlement_fees_ticks``.
        fill_shortfall_ticks: Shortfall of the ordinary fills alone (fills
            against the arrival price plus their fees), in cash ticks.
        terminal_component_ticks: Terminal term of the last step: the
            settlement cost (its shortfall against arrival including the fee)
            or the legacy residual term; 0 when none was charged.
        shortfall_ticks: Implementation shortfall against the arrival price,
            in cash ticks: ``fill_shortfall_ticks + terminal_component_ticks``
            for a terminated episode, ``fill_shortfall_ticks`` alone when it
            truncated (no terminal term exists).
        shortfall_bps: ``shortfall_ticks`` as basis points of the arrival
            notional.
        cost_sum_ticks: Sum of ``cost_component`` over the steps.
        risk_sum_ticks: Sum of ``risk_component`` over the steps.
        reward_sum_ticks: Sum of ``reward_ticks`` over the steps; equals
            ``-(shortfall_ticks + risk_sum_ticks)`` for a terminated episode.
        n_capped_steps: Fills cut by the participation cap
            (``fill_qty_ticks == cap_qty_ticks < submitted_qty_ticks``).
    """

    n_steps: int
    terminated: bool
    truncated: bool
    terminal_rule: str
    terminal_surcharge_bps: int
    arrival_price_ticks: int
    terminal_price_ticks: int
    target_qty_ticks: int
    filled_qty_ticks: int
    settled_qty_ticks: int
    residual_qty_ticks: int
    completed_by_fills: bool
    completed: bool
    filled_fraction: float
    settlement_fraction: float
    exec_notional_ticks: int
    settlement_price_ticks: int
    settlement_notional_ticks: int
    settlement_exceeds_cap: bool
    fill_fees_ticks: int
    settlement_fees_ticks: int
    fees_ticks: int
    fill_shortfall_ticks: int
    terminal_component_ticks: int
    shortfall_ticks: int
    shortfall_bps: float
    cost_sum_ticks: int
    risk_sum_ticks: int
    reward_sum_ticks: int
    n_capped_steps: int

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict of built-in types; integers stay exact under ``json.dumps``."""
        return asdict(self)


def summarize_episode(trajectory: Trajectory) -> EpisodeSummary:
    """Reduce a trajectory to its accounting; every integer is exact.

    A truncated episode reports the shortfall of its fills alone, because no
    terminal mark exists to settle or price the residual. For a terminated
    episode the shortfall is ``implementation_shortfall`` and equals
    ``fill_shortfall + terminal_component`` by construction.
    """
    spec, final, last = trajectory.spec, trajectory.final_state, trajectory.last_result
    residual = remaining_qty_ticks(final, spec)
    fills_only = fill_shortfall(final, spec)
    if last.truncated:
        shortfall = fills_only
    else:
        shortfall = implementation_shortfall(
            final, spec, last.terminal_price_ticks if residual else None
        )
        # Decomposition identity: the terminal component of the last step is exactly the
        # settlement shortfall (or the legacy residual term), so the two definitions agree.
        if shortfall != fills_only + last.terminal_component or (
            spec.settles_at_horizon and settlement_shortfall(final, spec) != last.terminal_component
        ):  # pragma: no cover - guards the simulator's own invariant
            raise AssertionError("shortfall decomposition does not match the step records")
    results = trajectory.results
    return EpisodeSummary(
        n_steps=len(results),
        terminated=last.terminated,
        truncated=last.truncated,
        terminal_rule=spec.terminal_rule,
        terminal_surcharge_bps=spec.terminal_surcharge_bps,
        arrival_price_ticks=final.arrival_price_ticks,
        terminal_price_ticks=last.terminal_price_ticks,
        target_qty_ticks=spec.target_qty_ticks,
        filled_qty_ticks=final.filled_qty_ticks,
        settled_qty_ticks=final.settled_qty_ticks,
        residual_qty_ticks=residual,
        completed_by_fills=final.filled_qty_ticks == spec.target_qty_ticks,
        completed=residual == 0,
        filled_fraction=final.filled_qty_ticks / spec.target_qty_ticks,
        settlement_fraction=final.settled_qty_ticks / spec.target_qty_ticks,
        exec_notional_ticks=final.exec_notional_ticks,
        settlement_price_ticks=last.settlement_price_ticks,
        settlement_notional_ticks=final.settlement_notional_ticks,
        settlement_exceeds_cap=last.settlement_exceeds_cap,
        fill_fees_ticks=final.fill_fees_ticks,
        settlement_fees_ticks=final.settlement_fees_ticks,
        fees_ticks=final.total_fees_ticks,
        fill_shortfall_ticks=fills_only,
        terminal_component_ticks=last.terminal_component,
        shortfall_ticks=shortfall,
        shortfall_bps=shortfall_bps(shortfall, final.arrival_price_ticks, spec.target_qty_ticks),
        cost_sum_ticks=sum(r.cost_component for r in results),
        risk_sum_ticks=sum(r.risk_component for r in results),
        reward_sum_ticks=sum(r.reward_ticks for r in results),
        n_capped_steps=sum(
            1 for r in results if r.fill_qty_ticks == r.cap_qty_ticks < r.submitted_qty_ticks
        ),
    )


def nearest_rank_percentile(values: Sequence[float], percentile: int) -> float:
    """Return the nearest-rank percentile: ``sorted(values)[ceil(percentile / 100 * n) - 1]``.

    The 0th percentile is the minimum; there is no interpolation.

    Raises:
        ValueError: No values, or a percentile outside ``[0, 100]``.
    """
    if not values:
        raise ValueError("a percentile of no values is undefined")
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class SetSummary:
    """Aggregate of an episode set.

    Percentiles are nearest-rank over ``shortfall_bps``. Completion is
    aggregated twice, by ordinary fills and finally, so reliance on the
    terminal settlement is visible next to the cost.

    Attributes:
        n_episodes: Episodes in the set.
        n_completed_by_fills: Episodes whose ordinary fills reached the
            target.
        n_completed: Episodes with ``completed`` set (final completion).
        n_settled: Episodes with a non-zero settled quantity.
        n_truncated: Episodes with ``truncated`` set.
        completion_by_fills_rate: ``n_completed_by_fills / n_episodes``.
        completion_rate: ``n_completed / n_episodes``.
        mean_shortfall_bps: Mean of ``shortfall_bps``.
        p50_shortfall_bps: 50th nearest-rank percentile of ``shortfall_bps``.
        p95_shortfall_bps: 95th nearest-rank percentile.
        p99_shortfall_bps: 99th nearest-rank percentile.
        max_shortfall_bps: Largest ``shortfall_bps``.
        total_shortfall_ticks: Sum of ``shortfall_ticks``, in cash ticks.
        total_fill_fees_ticks: Sum of ``fill_fees_ticks``, in cash ticks.
        total_settlement_fees_ticks: Sum of ``settlement_fees_ticks``.
        total_fees_ticks: Sum of ``fees_ticks``, in cash ticks.
        total_terminal_component_ticks: Sum of ``terminal_component_ticks``
            (settlement cost or legacy residual term), in cash ticks.
        total_risk_ticks: Sum of ``risk_sum_ticks``, in cash ticks.
        mean_filled_fraction: Mean of ``filled_fraction``.
        mean_settlement_fraction: Mean of ``settlement_fraction``.
        mean_residual_fraction: Mean of
            ``residual_qty_ticks / target_qty_ticks``.
        mean_capped_steps: Mean of ``n_capped_steps``.
    """

    n_episodes: int
    n_completed_by_fills: int
    n_completed: int
    n_settled: int
    n_truncated: int
    completion_by_fills_rate: float
    completion_rate: float
    mean_shortfall_bps: float
    p50_shortfall_bps: float
    p95_shortfall_bps: float
    p99_shortfall_bps: float
    max_shortfall_bps: float
    total_shortfall_ticks: int
    total_fill_fees_ticks: int
    total_settlement_fees_ticks: int
    total_fees_ticks: int
    total_terminal_component_ticks: int
    total_risk_ticks: int
    mean_filled_fraction: float
    mean_settlement_fraction: float
    mean_residual_fraction: float
    mean_capped_steps: float

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict of built-in types for the run record."""
        return asdict(self)


def summarize_set(summaries: Sequence[EpisodeSummary]) -> SetSummary:
    """Aggregate episode summaries.

    An empty set is an error, not a summary of zeros.

    Raises:
        ValueError: No summaries.
    """
    if not summaries:
        raise ValueError("summarize_set needs at least one episode")
    n = len(summaries)
    bps = [s.shortfall_bps for s in summaries]
    p50, p95, p99 = (nearest_rank_percentile(bps, p) for p in TAIL_PERCENTILES)
    n_completed = sum(1 for s in summaries if s.completed)
    n_by_fills = sum(1 for s in summaries if s.completed_by_fills)
    return SetSummary(
        n_episodes=n,
        n_completed_by_fills=n_by_fills,
        n_completed=n_completed,
        n_settled=sum(1 for s in summaries if s.settled_qty_ticks),
        n_truncated=sum(1 for s in summaries if s.truncated),
        completion_by_fills_rate=n_by_fills / n,
        completion_rate=n_completed / n,
        mean_shortfall_bps=sum(bps) / n,
        p50_shortfall_bps=p50,
        p95_shortfall_bps=p95,
        p99_shortfall_bps=p99,
        max_shortfall_bps=max(bps),
        total_shortfall_ticks=sum(s.shortfall_ticks for s in summaries),
        total_fill_fees_ticks=sum(s.fill_fees_ticks for s in summaries),
        total_settlement_fees_ticks=sum(s.settlement_fees_ticks for s in summaries),
        total_fees_ticks=sum(s.fees_ticks for s in summaries),
        total_terminal_component_ticks=sum(s.terminal_component_ticks for s in summaries),
        total_risk_ticks=sum(s.risk_sum_ticks for s in summaries),
        mean_filled_fraction=sum(s.filled_fraction for s in summaries) / n,
        mean_settlement_fraction=sum(s.settlement_fraction for s in summaries) / n,
        mean_residual_fraction=sum(s.residual_qty_ticks / s.target_qty_ticks for s in summaries)
        / n,
        mean_capped_steps=sum(s.n_capped_steps for s in summaries) / n,
    )


# --- run records ----------------------------------------------------------------------


def code_revision() -> dict[str, Any]:
    """Return ``{"commit", "dirty"}`` of the checkout this module runs from.

    Both values are ``None`` when ``git`` is unavailable or the module does
    not live in a repository.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=here
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def _template_json(template: EpisodeSpec) -> dict[str, Any]:
    """Return the template's fields except side and target, which every draw overrides."""
    return {
        f.name: getattr(template, f.name)
        for f in fields(template)
        if f.name not in ("side", "target_qty_ticks")
    }


def run_policy(
    splits: Splits,
    role: str,
    policy: Policy,
    policy_name: str,
    template: EpisodeSpec,
    n_episodes: int,
    seed: int,
    rho_choices_bps: tuple[int, ...] = ACCEPTED_TARGET_FRACTIONS_BPS,
    observation_statistics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[Trajectory, ...]]:
    """Sample ``n_episodes`` of ``role``, replay ``policy`` on each and build the run record.

    Args:
        splits: Split tapes and record.
        role: One of ``ROLES``.
        policy: The policy to replay.
        policy_name: Name stored in the record's ``config``.
        template: Spec template; side and target come from each draw.
        n_episodes: Number of draws; positive.
        seed: Sampling seed.
        rho_choices_bps: Target fractions the draws choose from.
        observation_statistics: Stored verbatim in the record when given
            (see ``normalization``); the runner does not apply it, since the
            baselines read raw observations.

    Returns:
        The JSON-ready record and the trajectories. The trajectories are not
        stored in the record, which keeps per-episode summaries only.

    Raises:
        ValueError: ``n_episodes < 1``, a template that does not match the
            split geometry, or a role without admissible starts.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    draws = sample_episodes(splits.record, role, n_episodes, seed, rho_choices_bps)
    trajectories: list[Trajectory] = []
    episodes: list[dict[str, Any]] = []
    for draw in draws:
        spec, market = materialize(splits, draw, template)
        trajectory = run_episode(spec, market, policy)
        trajectories.append(trajectory)
        episodes.append(_episode_json(draw, spec, summarize_episode(trajectory)))
    record = {
        "schema": RUN_RECORD_SCHEMA,
        "simulator_version": SIMULATOR_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "config": {
            "policy": policy_name,
            "role": role,
            "n_episodes": n_episodes,
            "seed": seed,
            "rho_choices_bps": list(rho_choices_bps),
            "template": _template_json(template),
        },
        "split": splits.record.to_json(),
        "observation_statistics": observation_statistics,
        "episodes": episodes,
        "summary": summarize_set([summarize_episode(t) for t in trajectories]).to_json(),
    }
    return record, tuple(trajectories)


def _episode_json(draw: EpisodeDraw, spec: EpisodeSpec, summary: EpisodeSummary) -> dict[str, Any]:
    """Return the per-episode entry of a run record: draw, resolved side and target, summary."""
    return {
        "draw": draw.to_json(),
        "spec": {"side": spec.side, "target_qty_ticks": spec.target_qty_ticks},
        "summary": summary.to_json(),
    }


def write_run_record(record: dict[str, Any], runs_dir: str | Path = "runs") -> Path:
    """Write a record as indented, key-sorted JSON and return its path.

    The file name is ``<created>-<policy>-<role>-seed<seed>[-<k>].json``; a
    numeric suffix avoids overwriting a record written in the same second.
    The directory is created if needed and is meant to be ignored by Git.
    """
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    created = record["created_utc"].replace("-", "").replace(":", "").replace("+0000", "Z")
    config = record["config"]
    stem = f"{created}-{config['policy']}-{config['role']}-seed{config['seed']}"
    path = root / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = root / f"{stem}-{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path
