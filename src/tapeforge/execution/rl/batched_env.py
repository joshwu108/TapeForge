"""Vector environment over the batched simulator, with the interface the PPO trainer uses.

``BatchedExecutionVecEnv`` steps ``n_envs`` environments with
``execution.batched`` on the device of its ``EpisodeBatch`` and presents them
as a Stable-Baselines3 ``VecEnv``. It mirrors ``PolicyActionEnv`` over
``ExecutionEnv``: the same action map, the same frozen-statistics
normalisation, the same reward unit, the same ``terminated`` / ``truncated``
reporting and the same ``episode_summary``. The accounting is exact int64 on
the device.

What stays on the device and what does not:
    Simulator state, market lookups, action conversion, rewards, automatic
    resets and episode selection are device tensors. The library's rollout
    buffer lives in host memory, so each step moves one action array to the
    device and the observations, rewards and done flags back, and the rows
    of environments that ended are fetched for ``info``. Those transfers are
    a property of the trainer integration, not of the simulator; the
    benchmark reports them separately.

Differences from the CPU environments:
    * Episodes are drawn uniformly from one bank (the union of the training
      pools) with a device generator, so the sequence of episodes differs
      from the CPU backends even for the same seed. The distribution is the
      same when every environment has the same number of pools.
    * The float reward is computed in float64 from the exact integer reward;
      it equals the CPU value whenever ``reward_ticks * 10_000`` and
      ``arrival_price * target`` are below 2**53 and can differ in the last
      bit beyond that. ``reward_ticks`` itself is exact.
    * The collection position is one snapshot of the whole vector
      (``collection_snapshot`` / ``restore_collection``), not one per
      environment: the simulator state, the episode accumulators, the
      pending observations and the selection generator, all copied to host
      memory. Restoring it on CPU tensors continues bit-identically (a test
      checks this). On CUDA the simulator integers and the generator are
      restored exactly, but the trainer's floating-point update is not
      guaranteed deterministic there, so a resumed CUDA run is a valid
      continuation of the same schedule, not a bit-identical replay.

Compiled transition:
    With ``compile_step=True`` the device part of one vector step (action
    map, ``batched.step``, rewards, accumulators, the observation of the
    ended state, the episode summary, the masked reset and the next
    observation) is one function compiled with ``torch.compile(fullgraph=True,
    dynamic=False)``. It computes the same tensors as the eager function: all
    accounting is int64, and the float64 steps are the same elementwise
    operations, which the compiler does not reorder; a test compares the two
    exactly, and on CUDA the parity of compiled and eager steps must be
    re-checked on the device. Episode draws are taken from the selection
    generator before the function runs, in the same order as the eager path,
    so the generator state is unchanged by compilation. The first step pays
    the compilation.

Dependencies:
    ``stable-baselines3`` and ``torch`` (``pip install -e '.[train]'``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import numpy as np

try:
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.rl.batched_env needs stable-baselines3; install it with "
        "pip install -e '.[train]'"
    ) from exc

from tapeforge.execution import batched
from tapeforge.execution.data.splits import Splits
from tapeforge.execution.research.experiment import ConfigError, ExperimentConfig
from tapeforge.execution.rl.evaluation import TrainingConfig
from tapeforge.execution.rl.gym_env import SplitPool
from tapeforge.execution.rl.normalization import ObservationStatistics
from tapeforge.execution.rl.ppo_config import PPOConfig
from tapeforge.execution.rl.ppo_env import environment_seed
from tapeforge.execution.simulator import OBSERVATION_FIELDS

__all__ = ["BatchedExecutionVecEnv", "make_batched_vec_env"]

_SUMMARY_FIELDS = ("reward_sum", "shortfall_bps", "terminal_bps", "risk_bps", "filled_fraction",
                   "settlement_fraction", "n_steps")


class BatchedExecutionVecEnv(VecEnv):
    """``n_envs`` execution environments stepped as tensors; see the module docstring.

    Args:
        batch: The episode bank; its device is the simulation device.
        n_envs: Environments stepped together.
        seed: Seed of the episode-selection generator.
        statistics: Frozen training statistics applied to observations;
            ``None`` returns raw values.
        reward_unit: ``"bps"`` or ``"ticks"``.
        episode_ids: One identifier per bank row, reported in
            ``episode_summary``; row numbers when ``None``.
    """

    def __init__(
        self,
        batch: batched.EpisodeBatch,
        n_envs: int,
        *,
        seed: int,
        statistics: ObservationStatistics | None = None,
        reward_unit: str = "bps",
        episode_ids: Sequence[str] | None = None,
        compile_step: bool = False,
    ) -> None:
        """Build the spaces and the selection generator; nothing is stepped until ``reset``.

        Raises:
            ValueError: ``n_envs < 1``, an unknown reward unit, statistics
                over other fields, or identifiers that do not match the bank.
        """
        if n_envs < 1:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        if reward_unit not in ("bps", "ticks"):
            raise ValueError(f"reward_unit must be 'bps' or 'ticks', got {reward_unit!r}")
        if statistics is not None and tuple(statistics.fields) != OBSERVATION_FIELDS:
            raise ValueError("statistics must cover exactly OBSERVATION_FIELDS, in order")
        if episode_ids is not None and len(episode_ids) != batch.n_episodes:
            raise ValueError("episode_ids must name every episode of the batch")
        n_fields = len(OBSERVATION_FIELDS)
        super().__init__(
            n_envs,
            spaces.Box(-np.inf, np.inf, shape=(n_fields,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        )
        self.batch = batch
        self.reward_unit = reward_unit
        self._ids = None if episode_ids is None else list(episode_ids)
        device = batch.device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._mean: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        if statistics is not None:
            std = torch.tensor(statistics.std, dtype=torch.float64, device=device)
            self._mean = torch.tensor(statistics.mean, dtype=torch.float64, device=device)
            self._scale = torch.where(std > 0, std, torch.ones_like(std))
        self._state: batched.BatchedState | None = None
        self._last_observation = torch.zeros((n_envs, n_fields), dtype=torch.float32,
                                             device=device)
        self._reward_sum = torch.zeros(n_envs, dtype=torch.float64, device=device)
        self._cost_ticks = torch.zeros(n_envs, dtype=torch.int64, device=device)
        self._risk_ticks = torch.zeros(n_envs, dtype=torch.int64, device=device)
        self._steps = torch.zeros(n_envs, dtype=torch.int64, device=device)
        self._actions: np.ndarray | None = None
        self.compile_step = compile_step
        self._core = (
            torch.compile(self._transition, fullgraph=True, dynamic=False)
            if compile_step else self._transition
        )

    # --- helpers ---------------------------------------------------------------------------

    def _draw(self) -> torch.Tensor:
        """Return one uniformly drawn bank row per environment."""
        return torch.randint(
            self.batch.n_episodes, (self.num_envs,), generator=self._generator,
            device=self.batch.device,
        )

    def _observation(self, state: batched.BatchedState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the float32 observations of ``state`` and whether each could be computed."""
        observation, available = batched.observe(state, self.batch)
        if self._mean is not None and self._scale is not None:
            observation = (observation - self._mean) / self._scale
        return observation.to(torch.float32), available

    def _start(self, episodes: torch.Tensor) -> np.ndarray:
        """Start every environment on the given bank rows and return the observations."""
        self._state = batched.reset(self.batch, episodes)
        for accumulator in (self._reward_sum, self._cost_ticks, self._risk_ticks, self._steps):
            accumulator.zero_()
        self._last_observation, _ = self._observation(self._state)
        return self._last_observation.cpu().numpy()

    def _bank_fingerprint(self) -> str:
        """Return a digest of the bank's size and episode identifiers.

        A restored collection position indexes bank rows, so it is only valid
        for a bank with the same rows in the same order; the identifiers stand
        in for the market data, which ``SplitPool`` rebuilds deterministically.
        Without identifiers only the bank size is checked; the trainer always
        passes them (``make_batched_vec_env``).
        """
        digest = hashlib.sha256(str(self.batch.n_episodes).encode())
        for identifier in self._ids or ():
            digest.update(b"\0" + identifier.encode())
        return digest.hexdigest()

    def collection_snapshot(self) -> dict[str, Any]:
        """Return the collection position of every environment, in host memory.

        It holds the simulator state, the per-episode accumulators, the last
        observations and the selection generator state; no market data.

        Raises:
            RuntimeError: Before the first reset.
        """
        state = self.state
        return {
            "schema": "tapeforge.execution.batched-collection/1",
            "device": self.batch.device.type,
            "n_envs": self.num_envs,
            "reward_unit": self.reward_unit,
            "bank": self._bank_fingerprint(),
            "state": {f.name: getattr(state, f.name).cpu().clone() for f in fields(state)},
            "reward_sum": self._reward_sum.cpu().clone(),
            "cost_ticks": self._cost_ticks.cpu().clone(),
            "risk_ticks": self._risk_ticks.cpu().clone(),
            "steps": self._steps.cpu().clone(),
            "last_observation": self._last_observation.cpu().clone(),
            "generator": self._generator.get_state().clone(),
        }

    def restore_collection(self, snapshot: Mapping[str, Any]) -> np.ndarray:
        """Continue from ``collection_snapshot()`` of an environment over the same bank.

        Returns the restored last observations, as ``reset`` would.

        Everything is checked before any state changes, so a refused
        snapshot leaves the environment as it was.

        Raises:
            ValueError: A snapshot of another bank, environment count, reward
                unit or device type. The selection generator's state is
                device-specific, so a CPU snapshot cannot continue on CUDA or
                the reverse. Snapshots without a ``device`` entry are checked
                through the size of the generator state instead.
        """
        device = self.batch.device
        expected = {"n_envs": self.num_envs, "reward_unit": self.reward_unit,
                    "bank": self._bank_fingerprint()}
        differing = sorted(k for k, v in expected.items() if snapshot.get(k) != v)
        generator = snapshot["generator"]
        if snapshot.get("device", device.type) != device.type or (
            generator.numel() != self._generator.get_state().numel()
        ):
            differing.append("device")
        if differing:
            raise ValueError(f"the collection snapshot differs in {differing}")
        self._state = batched.BatchedState(
            **{name: tensor.to(device).clone() for name, tensor in snapshot["state"].items()}
        )
        self._reward_sum = snapshot["reward_sum"].to(device).clone()
        self._cost_ticks = snapshot["cost_ticks"].to(device).clone()
        self._risk_ticks = snapshot["risk_ticks"].to(device).clone()
        self._steps = snapshot["steps"].to(device).clone()
        self._last_observation = snapshot["last_observation"].to(device).clone()
        self._generator.set_state(generator)
        self._actions = None
        return self._last_observation.cpu().numpy()

    @property
    def state(self) -> batched.BatchedState:
        """Return the exact simulator state of every environment.

        Raises:
            RuntimeError: Before the first reset.
        """
        if self._state is None:
            raise RuntimeError("no episode is running; call reset() first")
        return self._state

    # --- VecEnv interface ----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start every environment on a freshly drawn episode."""
        return self._start(self._draw())

    def reset_to(self, episodes: Sequence[int]) -> np.ndarray:
        """Start environment ``i`` on bank row ``episodes[i]`` (for tests and parity checks).

        Raises:
            ValueError: Not one valid row per environment.
        """
        rows = torch.as_tensor(list(episodes), dtype=torch.int64, device=self.batch.device)
        if rows.shape != (self.num_envs,) or bool(
            torch.any((rows < 0) | (rows >= self.batch.n_episodes))
        ):
            raise ValueError("reset_to needs one valid bank row per environment")
        return self._start(rows)

    def step_async(self, actions: np.ndarray) -> None:
        """Store the policy actions (``[n_envs, 1]`` in ``[-1, 1]``) for ``step_wait``."""
        self._actions = np.asarray(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Apply the stored actions; ended environments restart on a new episode.

        Raises:
            RuntimeError: Before ``reset`` or without ``step_async``.
            ValueError: An action is not finite, or the array is not one
                value per environment.
        """
        if self._state is None or self._actions is None:
            raise RuntimeError("call reset() and step_async() before step_wait()")
        actions, self._actions = self._actions, None
        if actions.size != self.num_envs:
            raise ValueError(f"expected one action per environment, got shape {actions.shape}")
        # Checked on the host copy, so the range guarantee below needs no device read-back.
        if not np.isfinite(actions).all():
            raise ValueError("policy actions must be finite")
        device = self.batch.device
        raw = torch.as_tensor(actions, device=device).reshape(-1).to(torch.float64)
        before = self._state
        draws = self._draw()  # Drawn before the transition, as the eager order requires.
        (
            self._state, reward, done, terminal_observation, summary, result_truncated,
            self._last_observation, self._reward_sum, self._cost_ticks, self._risk_ticks,
            self._steps,
        ) = self._core(
            before, raw, draws, self._reward_sum, self._cost_ticks, self._risk_ticks,
            self._steps, self._last_observation,
        )

        dones = done.cpu().numpy()
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        ended = np.flatnonzero(dones)
        if ended.size:
            rows = torch.as_tensor(ended, device=device)
            terminal_host = terminal_observation[rows].cpu().numpy()
            summary_host = summary[rows].cpu().numpy()
            truncated_host = result_truncated[rows].cpu().numpy()
            episode_host = before.episode[rows].cpu().numpy()
            for j, index in enumerate(ended):
                values = dict(zip(_SUMMARY_FIELDS, summary_host[j].tolist(), strict=True))
                values["n_steps"] = int(values["n_steps"])
                row = int(episode_host[j])
                infos[index] = {
                    "terminal_observation": terminal_host[j],
                    "TimeLimit.truncated": bool(truncated_host[j]),
                    "episode_summary": {
                        "episode_id": f"bank/{row}" if self._ids is None else self._ids[row],
                        "terminated": not bool(truncated_host[j]),
                        "truncated": bool(truncated_host[j]),
                        **values,
                    },
                }
        return (
            self._last_observation.cpu().numpy(),
            reward.cpu().numpy(),
            dones,
            infos,
        )

    def _transition(
        self, before: batched.BatchedState, raw: torch.Tensor, draws: torch.Tensor,
        reward_sum: torch.Tensor, cost_ticks: torch.Tensor, risk_ticks: torch.Tensor,
        steps: torch.Tensor, last_observation: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Device part of one vector step, without host transfers or in-place updates.

        Returns the next state, the float32 rewards, the done flags, the
        terminal observations, the episode summaries (``_SUMMARY_FIELDS``),
        the truncation flags, the next observations and the four updated
        accumulators. Pure in its tensor arguments, so it can be compiled.
        """
        fraction = (torch.clamp(raw, -1.0, 1.0) + 1.0) / 2.0      # ppo_env.ACTION_MAP
        state, result = batched.step(before, fraction, self.batch, validate=False)

        notional = (
            state.arrival_price_ticks.double()
            * self.batch.target_qty_ticks[state.episode].double()
        )
        reward = result.reward_ticks.double()
        if self.reward_unit == "bps":
            reward = reward * 10_000.0 / notional
        done = result.terminated | result.truncated
        reward_sum = reward_sum + reward
        cost_ticks = cost_ticks + result.cost_component
        risk_ticks = risk_ticks + result.risk_component
        steps = steps + result.active.to(torch.int64)

        final_observation, available = self._observation(state)
        terminal_observation = torch.where(
            available.unsqueeze(1), final_observation, last_observation
        )
        target = self.batch.target_qty_ticks[state.episode].double()
        summary = torch.stack(
            [
                reward_sum,
                (cost_ticks + result.terminal_component).double() * 10_000.0 / notional,
                result.terminal_component.double() * 10_000.0 / notional,
                risk_ticks.double() * 10_000.0 / notional,
                state.filled_qty_ticks.double() / target,
                state.settled_qty_ticks.double() / target,
                steps.double(),
            ],
            dim=1,
        )

        next_state = batched.reset_where(state, done, draws, self.batch)
        reward_sum = torch.where(done, torch.zeros_like(reward_sum), reward_sum)
        cost_ticks = torch.where(done, torch.zeros_like(cost_ticks), cost_ticks)
        risk_ticks = torch.where(done, torch.zeros_like(risk_ticks), risk_ticks)
        steps = torch.where(done, torch.zeros_like(steps), steps)
        next_observation = torch.where(
            done.unsqueeze(1), self._observation(next_state)[0], terminal_observation
        )
        return (next_state, reward.to(torch.float32), done, terminal_observation, summary,
                result.truncated, next_observation, reward_sum, cost_ticks, risk_ticks, steps)

    def close(self) -> None:
        """Drop the running state; the episode bank is untouched."""
        self._state = None

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """Return ``render_mode`` (``None``) per environment; there are no per-environment objects.

        Raises:
            AttributeError: Any other attribute.
        """
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise AttributeError(f"the batched environments have no attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Not supported: the environments are rows of tensors (``AttributeError``)."""
        raise AttributeError("the batched environments have no settable attributes")

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        """Not supported (``NotImplementedError``): the collection position is one
        snapshot of the whole vector, ``collection_snapshot``."""
        raise NotImplementedError(
            f"the batched backend has no per-environment method {method_name!r}; use "
            "collection_snapshot() / restore_collection() for the whole vector"
        )

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:
        """Return ``False`` per environment: nothing wraps a tensor row."""
        return [False] * self.num_envs


def make_batched_vec_env(
    config: PPOConfig, experiment: ExperimentConfig, training: TrainingConfig, splits: Splits,
    statistics: ObservationStatistics, model_seed: int,
    memory_limit_bytes: int | None = batched.DEFAULT_MEMORY_LIMIT_BYTES,
) -> BatchedExecutionVecEnv:
    """Build the batched training environments of one model seed on ``config.device``.

    The bank is every draw of every training pool (``config.pool_episodes``
    draws of each ``training.train_seeds`` entry), materialised once.

    Raises:
        ConfigError: ``training.selection`` is not ``"random"``; the batched
            environments only draw uniformly.
        ValueError: The bank would exceed ``memory_limit_bytes``.
    """
    if training.selection != "random":
        raise ConfigError("the batched backend supports training.selection = 'random' only")
    episodes, identifiers = [], []
    for pool_seed in training.train_seeds:
        pool = SplitPool(splits, training.train_role, experiment.template, pool_seed,
                         config.pool_episodes, experiment.rho_choices_bps)
        for index in range(len(pool)):
            entry = pool[index]
            episodes.append((entry.spec, entry.market))
            identifiers.append(entry.episode_id)
    batch = batched.EpisodeBatch.from_episodes(episodes, config.device, memory_limit_bytes)
    return BatchedExecutionVecEnv(
        batch, config.n_envs, seed=environment_seed(model_seed, 0), statistics=statistics,
        reward_unit=training.reward_unit, episode_ids=identifiers,
        compile_step=config.compile_simulator,
    )
