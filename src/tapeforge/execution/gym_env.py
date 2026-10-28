"""Gymnasium adapter over the reference simulator: a thin CPU environment, no learning.

``ExecutionEnv`` wraps ``simulator.reset`` / ``observe`` / ``step`` for one
episode at a time drawn from an ``EpisodePool``. It adds nothing to the
financial logic: fills, fees, cash, inventory, the reward decomposition and
the terminated/truncated flags are the simulator's own, and ``info`` carries
them as exact integers next to the float reward.

Spaces (``gymnasium.spaces.Box``):
    observation  shape ``(len(OBSERVATION_FIELDS),)`` = ``(10,)``, float32.
                 The fields and their order are ``OBSERVATION_FIELDS``:
                 remaining fraction ``[0, 1]``, time-remaining fraction
                 ``[0, 1]``, side ``{-1, +1}``, mark return (unbounded),
                 mark age ``[0, inf)``, newest visible bucket's validity
                 ``{0, 1}``, record count ``[0, inf)``, quantity over
                 target ``[0, inf)``, taker imbalance ``[-1, 1]`` and
                 return against arrival (unbounded). The float64 vector of
                 ``observe`` is optionally standardised with frozen training
                 statistics (``normalization.ObservationStatistics``) and
                 then cast to float32; without statistics the raw values are
                 returned and the bounds above apply.
    action       shape ``(1,)``, float32, ``[0, 1]``: the fraction of the
                 remaining quantity to submit, floored to quantity ticks by
                 the simulator (``floor(action * remaining)``). A scalar is
                 accepted too. An action outside ``[0, 1]``, non-finite, or
                 of another shape raises ``ValueError``; nothing is clipped.

Reward:
    The simulator's ``reward_ticks`` (an exact integer in cash ticks,
    ``-(cost + risk + terminal)``) is returned as a Python float in basis
    points of the arrival notional::

        reward = reward_ticks * 10_000 / (arrival_price_ticks * target_qty_ticks)

    so a learner sees numbers of order one and the sum over a terminated
    episode is ``-shortfall_bps - risk`` in the same unit. ``info["reward_ticks"]``
    and ``info["reward_components_ticks"]`` keep the exact integers;
    ``reward_unit="ticks"`` returns the integer itself as a float (exact
    below 2**53).

Episode end:
    ``terminated`` and ``truncated`` are the simulator's flags, never both.
    On a truncated episode no residual term and no terminal mark are
    invented; the info reports the fills so far. The terminal observation is
    ``observe`` of the final state when its newest visible bucket exists in
    the slice; when it does not (a truncation with zero data delay and zero
    latency cuts the tape exactly there) the last observation the agent saw
    is returned again and ``info["terminal_observation_available"]`` is
    ``False``. No wrapper time limit exists: the horizon is part of the task
    and ends it with ``terminated``.

Episode pools and seeding:
    ``reset(seed=s)`` reseeds the environment and, in ``"sequential"``
    selection, restarts the pool from its first episode, so the same seed
    replays the same sequence. ``reset()`` without a seed continues: the
    next pool entry in sequential mode (``EpisodePoolExhausted`` once the
    pool is used up unless ``cycle=True``), or a draw from the environment's
    generator in ``"random"`` mode. ``reset(options={"episode": i})`` or
    ``{"episode": "<id>"}`` selects an entry explicitly. Stepping before the
    first reset or after the episode ended raises.

Causality:
    The observation is ``simulator.observe``, which cuts the tape at bucket
    ``t - 1 - data_delay_buckets`` before reading; nothing after it reaches
    the policy, and ``info`` exposes the simulator's exact state, not future
    buckets.

Dependencies:
    ``gymnasium`` is optional (``pip install -e '.[rl]'``). Importing this
    module without it raises ``ImportError`` with that hint; importing
    ``tapeforge.execution`` does not import this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.gym_env needs gymnasium; install it with pip install -e '.[rl]'"
    ) from exc

from tapeforge.execution.normalization import ObservationStatistics, normalize
from tapeforge.execution.sampling import draw_episode, materialize
from tapeforge.execution.simulator import (
    OBSERVATION_FIELDS,
    SIMULATOR_VERSION,
    EpisodeOver,
    is_done,
    newest_visible_bucket,
    observe,
    remaining_qty_ticks,
    reset,
    step,
)
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import Splits
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "ACTION_SHAPE",
    "OBSERVATION_HIGH",
    "OBSERVATION_LOW",
    "EpisodePool",
    "EpisodePoolExhausted",
    "ExecutionEnv",
    "ListPool",
    "SplitPool",
    "reward_from_ticks",
]

ACTION_SHAPE: Final[tuple[int, ...]] = (1,)

# Per-field bounds of the raw (unnormalised) observation, aligned with OBSERVATION_FIELDS.
OBSERVATION_LOW: Final[np.ndarray] = np.array(
    [0.0, 0.0, -1.0, -np.inf, 0.0, 0.0, 0.0, 0.0, -1.0, -np.inf], dtype=np.float32
)
OBSERVATION_HIGH: Final[np.ndarray] = np.array(
    [1.0, 1.0, 1.0, np.inf, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf], dtype=np.float32
)

Selection = Literal["sequential", "random"]
RewardUnit = Literal["bps", "ticks"]


class EpisodePoolExhausted(RuntimeError):
    """Raised by ``reset`` when a sequential pool has no episode left and ``cycle`` is off."""


@dataclass(frozen=True)
class PoolEntry:
    """One episode of a pool: its identifier, spec and slice."""

    episode_id: str
    spec: EpisodeSpec
    market: MarketSlice


class EpisodePool:
    """The episodes an environment draws from; an indexable sequence of ``PoolEntry``.

    Subclasses implement ``__len__``, ``__getitem__`` and ``episode_id``.
    Entries must be deterministic in their index so that ``reset(seed)``
    replays.
    """

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, index: int) -> PoolEntry:  # pragma: no cover - abstract
        raise NotImplementedError

    def episode_id(self, index: int) -> str:
        """Return the identifier of entry ``index`` without materialising it."""
        return self[index].episode_id

    def index_of(self, episode_id: str) -> int:
        """Return the index of ``episode_id``.

        Raises:
            KeyError: No entry has that identifier.
        """
        for i in range(len(self)):
            if self.episode_id(i) == episode_id:
                return i
        raise KeyError(f"episode {episode_id!r} is not in the pool")


class ListPool(EpisodePool):
    """A pool over explicit ``(episode_id, spec, market)`` entries, for tests and small sets."""

    def __init__(self, entries: Sequence[tuple[str, EpisodeSpec, MarketSlice]]) -> None:
        """Store the entries.

        Raises:
            ValueError: No entries, or duplicate identifiers.
        """
        if not entries:
            raise ValueError("a pool needs at least one episode")
        ids = [e[0] for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("episode identifiers must be unique")
        self._entries = tuple(PoolEntry(*e) for e in entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> PoolEntry:
        return self._entries[index]


class SplitPool(EpisodePool):
    """A pool of seeded draws from one split role, materialised on access.

    Entry ``i`` is draw ``(seed, role, i)`` of ``sampling.draw_episode``,
    identified as ``"<role>/s<seed>/i<i>"`` (the experiment identifier), so
    a pool with ``n_episodes`` entries contains exactly the episodes a run
    record with the same seed lists.
    """

    def __init__(
        self,
        splits: Splits,
        role: str,
        template: EpisodeSpec,
        seed: int,
        n_episodes: int,
        rho_choices_bps: tuple[int, ...] | None = None,
    ) -> None:
        """Validate the arguments; nothing is cut until an entry is accessed.

        Raises:
            ValueError: ``n_episodes < 1``, a template that does not match
                the split geometry, or a role without admissible starts.
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")
        if not splits.record.matches(template):
            raise ValueError("template geometry (H, l, L, d) differs from the split record's")
        if splits.record.n_starts(role) == 0:
            raise ValueError(f"role {role!r} has no admissible episode start")
        self._splits, self._role, self._template = splits, role, template
        self._seed, self._n = seed, n_episodes
        self._rho = rho_choices_bps

    def __len__(self) -> int:
        return self._n

    def episode_id(self, index: int) -> str:
        if not 0 <= index < self._n:
            raise IndexError(f"index {index} outside the {self._n}-episode pool")
        return f"{self._role}/s{self._seed}/i{index}"

    def __getitem__(self, index: int) -> PoolEntry:
        episode_id = self.episode_id(index)
        draw = (
            draw_episode(self._splits.record, self._role, self._seed, index)
            if self._rho is None
            else draw_episode(self._splits.record, self._role, self._seed, index, self._rho)
        )
        spec, market = materialize(self._splits, draw, self._template)
        return PoolEntry(episode_id, spec, market)


def reward_from_ticks(
    reward_ticks: int, spec: EpisodeSpec, arrival_price_ticks: int, unit: str
) -> float:
    """Convert an exact reward in cash ticks to the float the environment returns.

    ``"bps"``: ``reward_ticks * 10_000 / (arrival_price_ticks *
    target_qty_ticks)``. ``"ticks"``: ``float(reward_ticks)``.

    Raises:
        ValueError: Unknown unit.
    """
    if unit == "bps":
        return reward_ticks * 10_000 / (arrival_price_ticks * spec.target_qty_ticks)
    if unit == "ticks":
        return float(reward_ticks)
    raise ValueError(f"reward_unit must be 'bps' or 'ticks', got {unit!r}")


class ExecutionEnv(gym.Env[np.ndarray, np.ndarray]):
    """One optimal-execution episode at a time, driven by the reference simulator.

    Args:
        pool: Episodes to draw from.
        selection: ``"sequential"`` (pool order; a seed restarts it) or
            ``"random"`` (uniform over the pool from ``np_random``).
        cycle: In sequential mode, wrap around instead of raising
            ``EpisodePoolExhausted``.
        statistics: Frozen training statistics applied to every
            observation; ``None`` returns raw values. Never fitted here.
        reward_unit: ``"bps"`` (default) or ``"ticks"``.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        pool: EpisodePool,
        *,
        selection: Selection = "sequential",
        cycle: bool = False,
        statistics: ObservationStatistics | None = None,
        reward_unit: RewardUnit = "bps",
    ) -> None:
        """Build the spaces; no episode is cut until ``reset``.

        Raises:
            ValueError: Empty pool, unknown selection or reward unit, or
                statistics whose fields are not ``OBSERVATION_FIELDS``.
        """
        super().__init__()
        if len(pool) == 0:
            raise ValueError("the episode pool is empty")
        if selection not in ("sequential", "random"):
            raise ValueError(f"selection must be 'sequential' or 'random', got {selection!r}")
        reward_from_ticks(0, EpisodeSpec(side=1, target_qty_ticks=1), 1, reward_unit)
        if statistics is not None and tuple(statistics.fields) != OBSERVATION_FIELDS:
            raise ValueError("statistics must cover exactly OBSERVATION_FIELDS, in order")
        self.pool = pool
        self.selection: Selection = selection
        self.cycle = cycle
        self.statistics = statistics
        self.reward_unit: RewardUnit = reward_unit
        n_fields = len(OBSERVATION_FIELDS)
        if statistics is None:
            self.observation_space = spaces.Box(OBSERVATION_LOW, OBSERVATION_HIGH, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(n_fields,), dtype=np.float32
            )
        self.action_space = spaces.Box(0.0, 1.0, shape=ACTION_SHAPE, dtype=np.float32)
        self._cursor = 0
        self._entry: PoolEntry | None = None
        self._state: State | None = None
        self._last_observation: np.ndarray | None = None
        self._done = False

    # --- helpers ---------------------------------------------------------------------

    @property
    def simulator_version(self) -> str:
        """Return the version tag of the transition semantics in use."""
        return SIMULATOR_VERSION

    @property
    def current_episode_id(self) -> str | None:
        """Return the identifier of the running episode, or ``None`` before the first reset."""
        return None if self._entry is None else self._entry.episode_id

    @property
    def state(self) -> State:
        """Return the simulator's exact state of the running episode.

        Raises:
            RuntimeError: Before the first reset.
        """
        if self._state is None:
            raise RuntimeError("no episode is running; call reset() first")
        return self._state

    def _select(self, options: dict[str, Any] | None) -> int:
        """Return the pool index the next episode comes from.

        Raises:
            EpisodePoolExhausted: Sequential selection ran past the pool.
            IndexError: An explicit index is out of range.
            KeyError: An explicit identifier is unknown.
            TypeError: ``options["episode"]`` is neither int nor str.
        """
        if options and "episode" in options:
            chosen = options["episode"]
            if isinstance(chosen, bool) or not isinstance(chosen, int | str):
                raise TypeError("options['episode'] must be a pool index or an episode identifier")
            index = self.pool.index_of(chosen) if isinstance(chosen, str) else int(chosen)
            if not 0 <= index < len(self.pool):
                raise IndexError(f"episode index {index} outside the {len(self.pool)}-episode pool")
            return index
        if self.selection == "random":
            return int(self.np_random.integers(len(self.pool)))
        if self._cursor >= len(self.pool):
            if not self.cycle:
                raise EpisodePoolExhausted(
                    f"all {len(self.pool)} episodes of the pool were used; reset(seed=...) "
                    "restarts it, cycle=True wraps around"
                )
            self._cursor = 0
        index = self._cursor
        self._cursor += 1
        return index

    def _observation(self, state: State) -> np.ndarray:
        """Return the float32 observation of ``state``, standardised when statistics are set."""
        assert self._entry is not None
        vector = observe(state, self._entry.market, self._entry.spec)
        if self.statistics is not None:
            vector = normalize(vector, self.statistics)
        return vector.astype(np.float32)

    def _terminal_observation(self, state: State) -> tuple[np.ndarray, bool]:
        """Return the final observation and whether it could be computed from the tape."""
        assert self._entry is not None and self._last_observation is not None
        if self._entry.market.has_bucket(newest_visible_bucket(state.step, self._entry.spec)):
            return self._observation(state), True
        return self._last_observation.copy(), False

    @staticmethod
    def _action_value(action: Any) -> float:
        """Validate an action and return it as a Python float in ``[0, 1]``.

        Raises:
            ValueError: Wrong shape, non-finite, or outside ``[0, 1]``.
            TypeError: Not a number or array of numbers.
        """
        if isinstance(action, bool):
            raise TypeError("action must be a number or a shape-(1,) array, got bool")
        if isinstance(action, int | float | np.floating | np.integer):
            value = float(action)
        else:
            array = np.asarray(action)
            if array.shape not in (ACTION_SHAPE, ()):
                raise ValueError(
                    f"action must have shape {ACTION_SHAPE} or be a scalar, got {array.shape}"
                )
            if not np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
                raise TypeError(f"action must be numeric, got dtype {array.dtype}")
            value = float(array.reshape(-1)[0])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"action must be a finite fraction in [0, 1], got {value!r}")
        return value

    # --- gymnasium API ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start the next episode and return its first observation and info.

        See the module docstring for how the episode is chosen. ``info``
        carries the episode identifier, the spec, the arrival price, the
        initial inventory and the target as exact integers.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._cursor = 0
        index = self._select(options)
        entry = self.pool[index]
        state = reset(entry.spec, entry.market)
        self._entry, self._state, self._done = entry, state, False
        observation = self._observation(state)
        self._last_observation = observation
        info = {
            "episode_id": entry.episode_id,
            "pool_index": index,
            "spec": asdict(entry.spec),
            "arrival_price_ticks": state.arrival_price_ticks,
            "mark_age": state.mark_age,
            "initial_inventory_ticks": state.inventory_ticks,
            "target_qty_ticks": entry.spec.target_qty_ticks,
            "remaining_qty_ticks": remaining_qty_ticks(state, entry.spec),
            "newest_visible_bucket": newest_visible_bucket(state.step, entry.spec),
            "simulator_version": SIMULATOR_VERSION,
        }
        return observation.copy(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one decision.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` with the
            reward in ``reward_unit`` and ``info`` holding the exact
            ``StepResult`` fields, the exact state after the step, the reward
            decomposition in ticks and ``terminal_observation_available``.

        Raises:
            RuntimeError: No episode is running (before the first reset), or
                the episode already ended.
            ValueError: Invalid action (see ``_action_value``).
        """
        if self._entry is None or self._state is None:
            raise RuntimeError("no episode is running; call reset() first")
        if self._done:
            raise EpisodeOver(
                f"episode {self._entry.episode_id} has ended; call reset() for the next one"
            )
        value = self._action_value(action)
        spec, market = self._entry.spec, self._entry.market
        state, result = step(self._state, value, market, spec)
        self._state = state
        done = result.terminated or result.truncated
        if done:
            self._done = True
            observation, available = self._terminal_observation(state)
        else:
            observation, available = self._observation(state), True
        self._last_observation = observation
        reward = reward_from_ticks(
            result.reward_ticks, spec, state.arrival_price_ticks, self.reward_unit
        )
        info = self._step_info(state, result, available)
        return observation.copy(), reward, result.terminated, result.truncated, info

    def _step_info(self, state: State, result: StepResult, available: bool) -> dict[str, Any]:
        """Build the info dict of one step from exact integers."""
        assert self._entry is not None
        spec = self._entry.spec
        return {
            "episode_id": self._entry.episode_id,
            "step_result": asdict(result),
            "state": asdict(state),
            "remaining_qty_ticks": remaining_qty_ticks(state, spec),
            "reward_ticks": result.reward_ticks,
            "reward_components_ticks": {
                "cost": result.cost_component,
                "risk": result.risk_component,
                "terminal": result.terminal_component,
            },
            "reward_unit": self.reward_unit,
            "newest_visible_bucket": newest_visible_bucket(state.step, spec),
            "terminal_observation_available": available,
            "is_done": is_done(state, spec),
        }

    def close(self) -> None:
        """Release the running episode; the pool is untouched."""
        self._entry, self._state, self._last_observation = None, None, None
        self._done = False
