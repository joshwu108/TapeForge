"""Training-side environment for the PPO trainer: the action map, episode summaries, vector
environments.

Action map:
    The policy acts in ``[-1, 1]`` (a symmetric box, as the library expects
    of a Gaussian policy) and ``policy_action_to_env`` maps it to the
    environment's fraction of remaining quantity, ``(clip(a, -1, 1) + 1) / 2``,
    in float64. The rollout buffer stores the unclipped Gaussian sample with
    the log-probability of that sample, and the PPO update evaluates the same
    unclipped sample, so stored and recomputed log-probabilities refer to one
    distribution; the clip and the affine map are a deterministic part of the
    environment as the learner sees it. Evaluation applies the same function
    to the policy mean. Samples beyond the bounds all act as the bound, so
    exact 0 and 1 are reachable; the price is that exploration noise is
    wasted there.

Seeding:
    Environment ``i`` of model seed ``s`` selects episodes with its own
    generator derived from ``(s, i)``, so two model seeds never share a
    selection stream. The pools themselves are draws of the training
    configuration's ``training.seeds``.

Dependencies:
    ``stable-baselines3`` (``pip install -e '.[train]'``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.rl.ppo_env needs stable-baselines3; install it with "
        "pip install -e '.[train]'"
    ) from exc

from tapeforge.execution.data.splits import Splits
from tapeforge.execution.research.experiment import ExperimentConfig
from tapeforge.execution.rl.evaluation import TrainingConfig
from tapeforge.execution.rl.gym_env import ExecutionEnv, SplitPool
from tapeforge.execution.rl.normalization import ObservationStatistics
from tapeforge.execution.rl.ppo_config import PPOConfig

__all__ = [
    "ACTION_MAP",
    "PolicyActionEnv",
    "environment_seed",
    "make_vec_env",
    "policy_action_to_env",
]

ACTION_MAP: Final[str] = "env_action = (clip(policy_action, -1, 1) + 1) / 2, float64"


def policy_action_to_env(action: Any) -> float:
    """Map one policy action in ``[-1, 1]`` to the environment's fraction in ``[0, 1]``.

    ``(clip(a, -1, 1) + 1) / 2`` in float64, whatever the input precision,
    so training and evaluation convert identically.

    Raises:
        ValueError: The action does not hold exactly one value, or the value
            is not finite (a diverged policy must not be clipped into range).
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise ValueError(f"policy action must hold one value, got shape {np.shape(action)}")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError(f"policy action must be finite, got {value!r}")
    return (min(max(value, -1.0), 1.0) + 1.0) / 2.0


class PolicyActionEnv(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """``ExecutionEnv`` as the learner sees it: actions in ``[-1, 1]``, an episode summary.

    Observations, rewards and episode ends pass through unchanged. On the
    step that ends an episode ``info["episode_summary"]`` carries the
    episode's reward sum (in the environment's reward unit), its shortfall
    and terminal term in basis points of the arrival notional, the fractions
    of the target reached through ordinary fills and through the terminal
    settlement, and how it ended. The shortfall is the sum of the step cost
    components plus the terminal component (fills only when truncated),
    which is the simulator's own decomposition.

    Args:
        env: The environment to wrap.
        detailed_info: Pass ``ExecutionEnv``'s full step info through
            (exact step record and state on every step). Off by default:
            the vector environment deep-copies every info dict on every
            step, and with the full records that copy cost several times
            the simulator step itself in profiles. The summary is always
            present on the last step.
    """

    def __init__(self, env: ExecutionEnv, *, detailed_info: bool = False) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.detailed_info = detailed_info
        self._target_qty_ticks = 0
        self._reward_sum = 0.0
        self._cost_ticks = 0
        self._risk_ticks = 0
        self._steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._target_qty_ticks = int(info["target_qty_ticks"])
        self._reward_sum, self._cost_ticks, self._risk_ticks, self._steps = 0.0, 0, 0, 0
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        fraction = np.array([policy_action_to_env(action)], dtype=np.float64)
        observation, reward, terminated, truncated, info = self.env.step(fraction)
        components = info["reward_components_ticks"]
        self._reward_sum += float(reward)
        self._cost_ticks += components["cost"]
        self._risk_ticks += components["risk"]
        self._steps += 1
        lean: dict[str, Any] = info if self.detailed_info else {}
        if terminated or truncated:
            state = info["state"]
            notional = state["arrival_price_ticks"] * self._target_qty_ticks
            terminal = components["terminal"]
            lean["episode_summary"] = {
                "episode_id": info["episode_id"],
                "n_steps": self._steps,
                "terminated": terminated,
                "truncated": truncated,
                "reward_sum": self._reward_sum,
                "shortfall_bps": (self._cost_ticks + terminal) * 10_000 / notional,
                "terminal_bps": terminal * 10_000 / notional,
                "risk_bps": self._risk_ticks * 10_000 / notional,
                "filled_fraction": state["filled_qty_ticks"] / self._target_qty_ticks,
                "settlement_fraction": state["settled_qty_ticks"] / self._target_qty_ticks,
            }
        return observation, float(reward), terminated, truncated, lean

    def snapshot(self) -> dict[str, Any]:
        """Return the collection position of this environment (see ``ExecutionEnv.snapshot``)."""
        inner: ExecutionEnv = self.env.unwrapped  # type: ignore[assignment]
        return {
            "env": inner.snapshot(),
            "target_qty_ticks": self._target_qty_ticks,
            "reward_sum": self._reward_sum,
            "cost_ticks": self._cost_ticks,
            "risk_ticks": self._risk_ticks,
            "steps": self._steps,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Continue from a ``snapshot`` of an environment built over the same pool."""
        inner: ExecutionEnv = self.env.unwrapped  # type: ignore[assignment]
        inner.restore(snapshot["env"])
        self._target_qty_ticks = int(snapshot["target_qty_ticks"])
        self._reward_sum = float(snapshot["reward_sum"])
        self._cost_ticks = int(snapshot["cost_ticks"])
        self._risk_ticks = int(snapshot["risk_ticks"])
        self._steps = int(snapshot["steps"])


def environment_seed(model_seed: int, index: int) -> int:
    """Return the episode-selection seed of environment ``index`` under ``model_seed``."""
    return int(np.random.SeedSequence([model_seed, index]).generate_state(1)[0])


def _environment_factory(
    splits: Splits, experiment: ExperimentConfig, training: TrainingConfig, pool_seed: int,
    pool_episodes: int, statistics: ObservationStatistics, selection_seed: int,
) -> Callable[[], PolicyActionEnv]:
    """Return a picklable constructor of one seeded training environment."""

    def factory() -> PolicyActionEnv:
        pool = SplitPool(
            splits, training.train_role, experiment.template, pool_seed, pool_episodes,
            experiment.rho_choices_bps,
        )
        env = PolicyActionEnv(ExecutionEnv(
            pool, selection=training.selection,  # type: ignore[arg-type]
            cycle=True, statistics=statistics,
            reward_unit=training.reward_unit,  # type: ignore[arg-type]
        ))
        # Seeds the selection generator once; the vector environment's own
        # resets pass no seed and therefore continue this stream.
        env.reset(seed=selection_seed)
        return env

    return factory


def make_vec_env(
    config: PPOConfig, experiment: ExperimentConfig, training: TrainingConfig, splits: Splits,
    statistics: ObservationStatistics, model_seed: int,
) -> VecEnv:
    """Build the training environments of one model seed for the configured backend.

    Environment ``i`` draws from the pool of ``training.train_seeds[i %
    len]`` (``config.pool_episodes`` draws of the training role) with its own
    selection stream.

    The ``batched`` backend instead builds one tensor environment over the
    union of the pools (``batched_env.make_batched_vec_env``).
    """
    if config.backend == "batched":
        # Imported here: the CPU backends must not require the tensor simulator.
        from tapeforge.execution.rl.batched_env import make_batched_vec_env

        return make_batched_vec_env(config, experiment, training, splits, statistics, model_seed)
    factories = [
        _environment_factory(
            splits, experiment, training,
            training.train_seeds[i % len(training.train_seeds)], config.pool_episodes,
            statistics, environment_seed(model_seed, i),
        )
        for i in range(config.n_envs)
    ]
    if config.backend == "serial":
        return DummyVecEnv(factories)  # type: ignore[arg-type]
    return SubprocVecEnv(factories)  # type: ignore[arg-type]
