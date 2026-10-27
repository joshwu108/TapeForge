"""Replay a policy through episodes and summarise the outcome, exactly and reproducibly.

A *policy* is any callable ``Observation -> action`` with the action a
fraction of the remaining quantity in ``[0, 1]`` (``simulator.step``'s
domain). ``Observation`` carries the float vector a learned policy consumes
(``simulator.observe``) next to the exact integers behind it (remaining
quantity, time remaining, the newest visible bucket's quantity) so that the
quantity-based baselines never round through a float. Everything the policy
sees is computed from buckets ``<= t - 1 - d``; the runner adds no
information of its own.

``run_episode`` returns the full ``Trajectory`` (states, observations,
actions, step results); ``summarize_episode`` reduces it to integer
accounting plus the shortfall in basis points; ``summarize_set`` aggregates a
set of episodes with nearest-rank percentiles (``sorted[ceil(p * n) - 1]``,
no interpolation) so a plain loop can recompute every figure.

``run_policy`` samples a split role, replays one policy, and writes a JSON
run record (configuration, seeds, split record, code revision, per-episode
draws and summaries, set summary) under a runs directory that Git ignores.
Two runs with the same inputs produce identical records except for the
``created_utc`` and ``code_revision`` fields. Nothing here learns.
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
    implementation_shortfall,
    is_done,
    newest_visible_bucket,
    observe,
    remaining_qty_ticks,
    reset,
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

RUN_RECORD_SCHEMA: Final[str] = "tapeforge.execution.run/1"
TAIL_PERCENTILES: Final[tuple[int, ...]] = (50, 95, 99)


@dataclass(frozen=True, eq=False)
class Observation:
    """What a policy may see before one decision.

    ``vector`` is ``simulator.observe`` (``OBSERVATION_FIELDS``, float64).
    The integer fields are exact and visible at the same time: the remaining
    quantity and horizon are episode state, ``newest_bucket_total_qty_ticks``
    is the quantity of bucket ``t - 1 - d`` (0 when it had none), and the
    cap rate lets a baseline estimate the next fill's liquidity limit.
    """

    vector: np.ndarray
    remaining_qty_ticks: int
    time_remaining_buckets: int
    target_qty_ticks: int
    side: int
    newest_bucket_total_qty_ticks: int
    participation_cap_bps: int


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
    """Everything one replay produced. ``states[0]`` is the reset state."""

    spec: EpisodeSpec
    states: tuple[State, ...]
    observations: tuple[np.ndarray, ...]
    actions: tuple[float, ...]
    results: tuple[StepResult, ...]

    @property
    def final_state(self) -> State:
        """The state after the last decision (the reset state if none was taken)."""
        return self.states[-1]

    @property
    def last_result(self) -> StepResult:
        """The step that ended the episode; its flags say whether it terminated or truncated.

        Raises:
            IndexError: the trajectory holds no step, which ``run_episode``
                never produces (a fresh episode always takes at least one).
        """
        return self.results[-1]


def run_episode(spec: EpisodeSpec, market: MarketSlice, policy: Policy) -> Trajectory:
    """Reset, then step the policy until the episode terminates or truncates.

    Raises:
        Whatever ``reset`` and ``step`` raise (undefined arrival price, int64
        bounds, an action outside ``[0, 1]``).
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

    ``shortfall_ticks`` is the implementation shortfall against ``P_0``
    including the residual term when the episode terminated with residual
    inventory, and the shortfall of the fills alone when it truncated.
    ``n_capped_steps`` counts fills cut by the participation cap
    (``fill == cap < submitted``).
    """

    n_steps: int
    terminated: bool
    truncated: bool
    arrival_price_ticks: int
    terminal_price_ticks: int
    target_qty_ticks: int
    filled_qty_ticks: int
    residual_qty_ticks: int
    completed: bool
    exec_notional_ticks: int
    fees_ticks: int
    shortfall_ticks: int
    shortfall_bps: float
    cost_sum_ticks: int
    risk_sum_ticks: int
    terminal_component_ticks: int
    reward_sum_ticks: int
    n_capped_steps: int

    def to_json(self) -> dict[str, Any]:
        """Plain dict of built-in types; integers stay exact under ``json.dumps``."""
        return asdict(self)


def summarize_episode(trajectory: Trajectory) -> EpisodeSummary:
    """Reduce a trajectory to its accounting; every integer is exact."""
    spec, final, last = trajectory.spec, trajectory.final_state, trajectory.last_result
    residual = remaining_qty_ticks(final, spec)
    if last.truncated:
        shortfall = spec.side * (
            final.exec_notional_ticks - final.arrival_price_ticks * final.filled_qty_ticks
        ) + final.fees_ticks
    else:
        shortfall = implementation_shortfall(
            final, spec, last.terminal_price_ticks if residual else None
        )
    results = trajectory.results
    return EpisodeSummary(
        n_steps=len(results),
        terminated=last.terminated,
        truncated=last.truncated,
        arrival_price_ticks=final.arrival_price_ticks,
        terminal_price_ticks=last.terminal_price_ticks,
        target_qty_ticks=spec.target_qty_ticks,
        filled_qty_ticks=final.filled_qty_ticks,
        residual_qty_ticks=residual,
        completed=residual == 0,
        exec_notional_ticks=final.exec_notional_ticks,
        fees_ticks=final.fees_ticks,
        shortfall_ticks=shortfall,
        shortfall_bps=shortfall_bps(shortfall, final.arrival_price_ticks, spec.target_qty_ticks),
        cost_sum_ticks=sum(r.cost_component for r in results),
        risk_sum_ticks=sum(r.risk_component for r in results),
        terminal_component_ticks=last.terminal_component,
        reward_sum_ticks=sum(r.reward_ticks for r in results),
        n_capped_steps=sum(
            1 for r in results if r.fill_qty_ticks == r.cap_qty_ticks < r.submitted_qty_ticks
        ),
    )


def nearest_rank_percentile(values: Sequence[float], percentile: int) -> float:
    """``sorted(values)[ceil(percentile / 100 * n) - 1]``; the 0th percentile is the minimum.

    Raises:
        ValueError: no values, or a percentile outside ``[0, 100]``.
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
    """Aggregate of an episode set. Percentiles are nearest-rank over ``shortfall_bps``."""

    n_episodes: int
    n_completed: int
    n_truncated: int
    completion_rate: float
    mean_shortfall_bps: float
    p50_shortfall_bps: float
    p95_shortfall_bps: float
    p99_shortfall_bps: float
    max_shortfall_bps: float
    total_shortfall_ticks: int
    total_fees_ticks: int
    mean_residual_fraction: float
    mean_capped_steps: float

    def to_json(self) -> dict[str, Any]:
        """Plain dict of built-in types for the run record."""
        return asdict(self)


def summarize_set(summaries: Sequence[EpisodeSummary]) -> SetSummary:
    """Aggregate episode summaries; an empty set is an error, not a summary of zeros.

    Raises:
        ValueError: no summaries.
    """
    if not summaries:
        raise ValueError("summarize_set needs at least one episode")
    n = len(summaries)
    bps = [s.shortfall_bps for s in summaries]
    p50, p95, p99 = (nearest_rank_percentile(bps, p) for p in TAIL_PERCENTILES)
    return SetSummary(
        n_episodes=n,
        n_completed=sum(1 for s in summaries if s.completed),
        n_truncated=sum(1 for s in summaries if s.truncated),
        completion_rate=sum(1 for s in summaries if s.completed) / n,
        mean_shortfall_bps=sum(bps) / n,
        p50_shortfall_bps=p50,
        p95_shortfall_bps=p95,
        p99_shortfall_bps=p99,
        max_shortfall_bps=max(bps),
        total_shortfall_ticks=sum(s.shortfall_ticks for s in summaries),
        total_fees_ticks=sum(s.fees_ticks for s in summaries),
        mean_residual_fraction=sum(s.residual_qty_ticks / s.target_qty_ticks for s in summaries)
        / n,
        mean_capped_steps=sum(s.n_capped_steps for s in summaries) / n,
    )


# --- run records ----------------------------------------------------------------------


def code_revision() -> dict[str, Any]:
    """``{"commit", "dirty"}`` of the checkout this module runs from; ``None`` without git."""
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
    """The template's fields except side and target, which every draw overrides."""
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
    """Sample ``n_episodes`` of ``role``, replay ``policy`` on each, build the run record.

    Returns the JSON-ready record and the trajectories (not stored in the
    record, which keeps per-episode summaries only). ``observation_statistics``
    is stored verbatim when given (see ``normalization``); the runner does not
    apply it, since the baselines read raw observations.

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
