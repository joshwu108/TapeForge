"""Environment checker: Gymnasium's checker, a random-action smoke run and simulator parity.

``check_environment`` builds the environment an experiment configuration
describes (same splits, same template, same seeded draws), runs Gymnasium's
``check_env``, drives random actions through it and, on every step, compares
the adapter's exact integer ``info`` with a direct ``simulator.step`` on the
same episode. It also checks the per-step invariants a learner relies on:
observation shape, dtype and finiteness, exact reward against the integer
components, the terminated/truncated contract, the terminal observation flag
and, at the end of a terminated episode, the reward-sum identity
``sum(reward_ticks) == -(shortfall + risk)``.

Random actions are not a policy; the run demonstrates that the environment
holds its contract under arbitrary valid inputs, which is what a trainer
needs before its first update. The command exits non-zero with the first
failed check named, so it can gate a training run.

Not covered:
    No learning, no checkpoint, no GPU. The checker needs the optional
    ``rl`` extra.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tapeforge.execution.experiment import ConfigError, load_config, rebuild_splits
from tapeforge.execution.gym_env import ExecutionEnv, SplitPool
from tapeforge.execution.normalization import load_statistics
from tapeforge.execution.runner import run_episode, summarize_episode
from tapeforge.execution.simulator import OBSERVATION_FIELDS, reset, step

__all__ = [
    "CheckFailure",
    "CheckReport",
    "check_environment",
    "main",
]


class CheckFailure(AssertionError):
    """A check failed; the message names the check, the episode and the step."""


@dataclass(frozen=True)
class CheckReport:
    """What the checker ran and found.

    Attributes:
        config: Configuration path.
        role: Split role the episodes came from.
        seed: Sampling seed of the pool.
        n_episodes: Episodes driven with random actions.
        n_steps: Steps taken in total.
        n_terminated: Episodes that terminated.
        n_truncated: Episodes that truncated.
        n_settled: Episodes whose horizon step settled quantity.
        checker_warnings: Messages Gymnasium's ``check_env`` emitted.
        statistics_file: Frozen statistics applied, if any.
        simulator_version: Version the environment reports.
    """

    config: str
    role: str
    seed: int
    n_episodes: int
    n_steps: int
    n_terminated: int
    n_truncated: int
    n_settled: int
    checker_warnings: tuple[str, ...]
    statistics_file: str | None
    simulator_version: str

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict of built-in types."""
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    """Raise ``CheckFailure`` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise CheckFailure(message)


def _check_episode(
    env: ExecutionEnv, index: int, rng: np.random.Generator, n_fields: int
) -> tuple[int, bool, bool, bool]:
    """Drive one episode with random actions and compare every step with the simulator.

    Returns:
        ``(n_steps, terminated, truncated, settled)``.
    """
    observation, info = env.reset(options={"episode": index})
    entry = env.pool[index]
    spec, market = entry.spec, entry.market
    state = reset(spec, market)
    tag = f"episode {entry.episode_id}"
    _require(info["episode_id"] == entry.episode_id, f"{tag}: reset info names another episode")
    _require(info["arrival_price_ticks"] == state.arrival_price_ticks, f"{tag}: arrival differs")
    _require(
        observation.shape == (n_fields,) and observation.dtype == np.float32,
        f"{tag}: reset observation has shape {observation.shape} dtype {observation.dtype}",
    )
    _require(bool(np.all(np.isfinite(observation))), f"{tag}: reset observation is not finite")
    rewards_ticks: list[int] = []
    actions: list[float] = []
    n_steps = 0
    settled = False
    while True:
        action = float(rng.uniform(0.0, 1.0)) if rng.uniform() < 0.8 else float(rng.integers(2))
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(np.float32(action))
        # The adapter passes the float32 value through; the direct call must see the same one.
        state, result = step(state, float(np.float32(action)), market, spec)
        n_steps += 1
        where = f"{tag} step {result.step}"
        _require(info["step_result"] == asdict(result), f"{where}: step_result differs")
        _require(info["state"] == asdict(state), f"{where}: state differs")
        _require(info["reward_ticks"] == result.reward_ticks, f"{where}: reward_ticks differs")
        components = info["reward_components_ticks"]
        component_sum = components["cost"] + components["risk"] + components["terminal"]
        _require(result.reward_ticks == -component_sum, f"{where}: reward != -component sum")
        notional = state.arrival_price_ticks * spec.target_qty_ticks
        expected = result.reward_ticks * 10_000 / notional
        if env.reward_unit == "ticks":
            expected = float(result.reward_ticks)
        _require(reward == expected, f"{where}: scalar reward {reward} is not {expected}")
        _require(
            observation.shape == (n_fields,) and observation.dtype == np.float32
            and bool(np.all(np.isfinite(observation))),
            f"{where}: observation shape/dtype/finiteness",
        )
        _require(not (terminated and truncated), f"{where}: terminated and truncated both set")
        _require(
            terminated == result.terminated and truncated == result.truncated,
            f"{where}: flags differ from the simulator",
        )
        rewards_ticks.append(result.reward_ticks)
        settled = settled or result.settlement_qty_ticks > 0
        if terminated or truncated:
            _require(
                info["terminal_observation_available"] in (True, False),
                f"{where}: terminal_observation_available missing",
            )
            break
    # Reward-sum identity against the summary of the same action sequence.
    trajectory = run_episode(spec, market, _replay(actions))
    summary = summarize_episode(trajectory)
    _require(
        sum(rewards_ticks) == summary.reward_sum_ticks,
        f"{tag}: adapter reward sum {sum(rewards_ticks)} != summary {summary.reward_sum_ticks}",
    )
    if summary.terminated:
        _require(
            summary.reward_sum_ticks == -(summary.shortfall_ticks + summary.risk_sum_ticks),
            f"{tag}: reward sum is not -(shortfall + risk)",
        )
        if spec.settles_at_horizon:
            _require(summary.completed, f"{tag}: terminated under settlement without completion")
    else:
        _require(summary.settled_qty_ticks == 0, f"{tag}: truncated episode settled quantity")
    return n_steps, summary.terminated, summary.truncated, settled


def _replay(actions: Sequence[float]) -> Any:
    """Return a policy that replays ``actions`` (as float32 values) in order."""
    iterator = iter(actions)

    def policy(_observation: Any) -> float:
        return float(np.float32(next(iterator)))

    return policy


def check_environment(
    config_path: str | Path,
    role: str = "train",
    n_episodes: int = 8,
    seed: int = 0,
    statistics_path: str | Path | None = None,
    reward_unit: str = "bps",
    run_gym_checker: bool = True,
) -> CheckReport:
    """Build the environment of a configuration and run every check.

    Args:
        config_path: Experiment configuration (TOML, schema 2).
        role: Role to draw episodes from; ``"test"`` is refused.
        n_episodes: Episodes driven with random actions; positive.
        seed: Pool seed and random-action seed.
        statistics_path: Frozen statistics file to apply (``load_statistics``).
        reward_unit: ``"bps"`` or ``"ticks"``.
        run_gym_checker: Run Gymnasium's ``check_env`` first.

    Raises:
        CheckFailure: A check failed; the message names it.
        ConfigError: The configuration is invalid or names the test role.
        ValueError: The statistics file is invalid, or the pool cannot be
            built.
    """
    if role == "test":
        raise ConfigError("the checker never draws from the held-out test role")
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    config = load_config(config_path)
    splits = rebuild_splits(config)
    statistics = None if statistics_path is None else load_statistics(statistics_path)
    pool = SplitPool(splits, role, config.template, seed, n_episodes, config.rho_choices_bps)
    unit: Any = reward_unit
    env = ExecutionEnv(pool, statistics=statistics, reward_unit=unit)
    checker_warnings: list[str] = []
    if run_gym_checker:
        from gymnasium.utils.env_checker import check_env

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            checked = ExecutionEnv(pool, statistics=statistics, reward_unit=unit, cycle=True)
            check_env(checked, skip_render_check=True)
        checker_warnings = [str(w.message) for w in caught]
    rng = np.random.default_rng(seed)
    n_steps = n_terminated = n_truncated = n_settled = 0
    for index in range(n_episodes):
        steps, terminated, truncated, settled = _check_episode(
            env, index, rng, len(OBSERVATION_FIELDS)
        )
        n_steps += steps
        n_terminated += terminated
        n_truncated += truncated
        n_settled += settled
    env.close()
    return CheckReport(
        config=str(config_path), role=role, seed=seed, n_episodes=n_episodes, n_steps=n_steps,
        n_terminated=n_terminated, n_truncated=n_truncated, n_settled=n_settled,
        checker_warnings=tuple(checker_warnings), statistics_file=(
            None if statistics_path is None else str(statistics_path)
        ), simulator_version=env.simulator_version,
    )


def main(argv: list[str] | None = None) -> int:
    """``<config.toml> [--role R] [--episodes N] [--seed S] [--statistics F] [--no-gym-checker]``.

    Returns:
        0 when every check passed, 1 with the failed check on stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.env_check")
    parser.add_argument("config", help="experiment configuration (schema 2)")
    parser.add_argument("--role", default="train", help="split role to draw from (never test)")
    parser.add_argument("--episodes", type=int, default=8,
                        help="episodes driven with random actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--statistics", default=None, help="frozen statistics JSON to apply")
    parser.add_argument("--reward-unit", default="bps", choices=("bps", "ticks"))
    parser.add_argument("--no-gym-checker", action="store_true", help="skip gymnasium check_env")
    args = parser.parse_args(argv)
    try:
        report = check_environment(
            args.config, args.role, args.episodes, args.seed, args.statistics, args.reward_unit,
            run_gym_checker=not args.no_gym_checker,
        )
    except (CheckFailure, ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"environment check passed: {report.n_episodes} random-action episodes, "
        f"{report.n_steps} steps, {report.n_terminated} terminated, {report.n_truncated} "
        f"truncated, {report.n_settled} settled at the horizon; simulator "
        f"{report.simulator_version}; statistics {report.statistics_file or 'none'}"
    )
    for message in report.checker_warnings:
        print(f"  gymnasium checker warning: {message}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
