"""Strict configuration of the optional PPO trainer (``rl.ppo``).

``PPOConfig`` holds what the optimiser needs and nothing about the data: the
network, the optimiser, rollout collection, the PPO update, the evaluation
schedule and the model-training seeds. Data, split, normalisation and
evaluation episodes come from the training configuration it names
(``evaluation.TrainingConfig``), which in turn names the experiment
configuration. Every value is read from the file; nothing falls back to a
library default, so a run cannot pick up, for example, a discount factor it
did not ask for.

Objective:
    The task is a finite-horizon, undiscounted cost: the per-step rewards of
    a terminated episode sum to minus the implementation shortfall (plus the
    risk term). ``gamma = 1.0`` optimises exactly that sum. A smaller value
    discounts the terminal settlement relative to earlier fills and is a
    different objective; it is accepted only because it is written in the
    file and recorded with every checkpoint.

Seeds:
    ``model_seeds`` seed network initialisation, action sampling, minibatch
    order and the per-environment episode-selection streams. They are
    distinct from the episode-sampling seeds of the training configuration
    (``training.seeds`` name the training pools, ``evaluation.seeds`` the
    fixed evaluation episodes), so the same evaluation set is replayed for
    every model seed.

This module imports no trainer; it is usable without the ``train`` extra.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from tapeforge.execution.research.experiment import ConfigError

__all__ = [
    "ACTIVATIONS",
    "BACKENDS",
    "PPO_CONFIG_SCHEMA",
    "SELECTION_METRICS",
    "PPOConfig",
    "load_ppo_config",
    "ppo_config_from_mapping",
]

PPO_CONFIG_SCHEMA: Final[str] = "tapeforge.execution.ppo-config/1"

ACTIVATIONS: Final[tuple[str, ...]] = ("tanh", "relu")
# How the training environments are stepped: one process in turn, one worker
# process per environment, or the batched tensor simulator (needs torch).
BACKENDS: Final[tuple[str, ...]] = ("serial", "subprocess", "batched")
# Checkpoints are compared on the validation role only; lower is better.
SELECTION_METRICS: Final[tuple[str, ...]] = ("validation_mean_shortfall_bps",)


@dataclass(frozen=True)
class PPOConfig:
    """One PPO training setup; see the module docstring for what it does not contain.

    Attributes:
        name: Run name; a training run directory is
            ``<output_dir>/<name>-seed<model_seed>-<utc>``.
        training_config: Path of the training configuration
            (``tapeforge.execution.training-config/1``).
        model_seeds: Model-training seeds this setup is meant to be run with.
        device: Torch device of the policy (``"cpu"``, ``"cuda"``, ...).
        torch_threads: Intra-op threads torch may use; recorded because it
            affects both speed and bit-exact reproducibility.
        output_dir: Root of training run directories (ignored by Git).
        hidden_sizes: Width of each hidden layer; the actor and the critic
            are separate MLPs of this shape.
        activation: One of ``ACTIVATIONS``.
        log_std_init: Initial log standard deviation of the Gaussian policy,
            in units of the policy action (see ``rl.ppo`` for the map to the
            environment action).
        learning_rate: Adam step size; constant over training.
        adam_eps: Adam epsilon.
        backend: One of ``BACKENDS``.
        n_envs: Environments stepped per rollout.
        n_steps: Decisions collected per environment per rollout.
        pool_episodes: Size of each training episode pool (draws
            ``0 .. pool_episodes - 1`` of a training seed).
        batch_size: Minibatch size; divides ``n_envs * n_steps``.
        n_epochs: Passes over each rollout.
        gamma: Discount factor in ``(0, 1]``; 1.0 is the undiscounted task.
        gae_lambda: GAE parameter in ``[0, 1]``.
        clip_range: PPO clipping parameter; constant over training.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Global gradient-norm clip.
        normalize_advantage: Standardise advantages per minibatch.
        target_kl: Stop an update's epochs early past ``1.5 * target_kl``;
            ``None`` disables the check.
        total_steps: Environment steps to train for; a multiple of
            ``n_envs * n_steps``.
        eval_every_rollouts: A checkpoint is saved and evaluated every this
            many policy versions (and before the first and after the last).
        selection_metric: One of ``SELECTION_METRICS``.
        compile_simulator: Optional ``rollout.compile_simulator``: compile
            the batched backend's vector step with ``torch.compile``
            (``batched`` backend only). It computes the same tensors as the
            eager step; see ``rl.batched_env``.
        evaluation_device: Optional ``schedule.evaluation_device``: the
            device of the policy copy that scores checkpoints on validation;
            ``None`` uses ``device``. The episodes, statistics, metric and
            selection rule are unchanged, but the policy's float32 outputs
            on another device can differ in the last bits, so a selection
            score is not bit-identical to one computed on ``device``.

    Both optional settings are omitted from ``to_json`` at their defaults,
    so configurations that do not use them keep their stored form.
    """

    name: str
    training_config: str
    model_seeds: tuple[int, ...]
    device: str
    torch_threads: int
    output_dir: str
    hidden_sizes: tuple[int, ...]
    activation: str
    log_std_init: float
    learning_rate: float
    adam_eps: float
    backend: str
    n_envs: int
    n_steps: int
    pool_episodes: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    normalize_advantage: bool
    target_kl: float | None
    total_steps: int
    eval_every_rollouts: int
    selection_metric: str
    compile_simulator: bool = False
    evaluation_device: str | None = None

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            ConfigError: A value outside its domain; the message names the key.
        """
        if not self.name or any(c in self.name for c in "/\\ "):
            raise ConfigError(f"name must be a non-empty file-name fragment, got {self.name!r}")
        seeds = self.model_seeds
        if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) < len(seeds):
            raise ConfigError("model_seeds must be a non-empty list of distinct non-negative ints")
        if self.torch_threads < 1:
            raise ConfigError("torch_threads must be positive")
        if not self.hidden_sizes or any(w < 1 for w in self.hidden_sizes):
            raise ConfigError("network.hidden_sizes must list positive layer widths")
        if self.activation not in ACTIVATIONS:
            raise ConfigError(f"network.activation must be one of {ACTIVATIONS}")
        if self.backend not in BACKENDS:
            raise ConfigError(f"rollout.backend must be one of {BACKENDS}")
        for key, value in (("rollout.n_envs", self.n_envs), ("rollout.n_steps", self.n_steps),
                           ("rollout.pool_episodes", self.pool_episodes),
                           ("update.batch_size", self.batch_size),
                           ("update.n_epochs", self.n_epochs),
                           ("schedule.total_steps", self.total_steps),
                           ("schedule.eval_every_rollouts", self.eval_every_rollouts)):
            if value < 1:
                raise ConfigError(f"{key} must be positive, got {value}")
        for key, number in (("network.log_std_init", self.log_std_init),
                            ("optimizer.learning_rate", self.learning_rate),
                            ("optimizer.adam_eps", self.adam_eps), ("update.gamma", self.gamma),
                            ("update.gae_lambda", self.gae_lambda),
                            ("update.clip_range", self.clip_range),
                            ("update.entropy_coef", self.entropy_coef),
                            ("update.value_coef", self.value_coef),
                            ("update.max_grad_norm", self.max_grad_norm)):
            if not math.isfinite(number):
                raise ConfigError(f"{key} must be finite, got {number!r}")
        if self.learning_rate <= 0 or self.adam_eps <= 0:
            raise ConfigError("optimizer.learning_rate and optimizer.adam_eps must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError(f"update.gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ConfigError(f"update.gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.clip_range <= 0 or self.max_grad_norm <= 0:
            raise ConfigError("update.clip_range and update.max_grad_norm must be positive")
        if self.entropy_coef < 0 or self.value_coef < 0:
            raise ConfigError("update.entropy_coef and update.value_coef must be non-negative")
        if self.target_kl is not None and not self.target_kl > 0:
            raise ConfigError("update.target_kl must be positive when given")
        rollout = self.n_envs * self.n_steps
        if rollout % self.batch_size:
            raise ConfigError(
                f"update.batch_size {self.batch_size} must divide n_envs * n_steps = {rollout}; "
                "a ragged last minibatch would weight its samples differently"
            )
        if self.total_steps % rollout:
            raise ConfigError(
                f"schedule.total_steps {self.total_steps} must be a multiple of n_envs * "
                f"n_steps = {rollout}; training advances in whole rollouts"
            )
        if self.selection_metric not in SELECTION_METRICS:
            raise ConfigError(f"schedule.selection_metric must be one of {SELECTION_METRICS}")
        if not isinstance(self.compile_simulator, bool):
            raise ConfigError("rollout.compile_simulator must be a boolean")
        if self.compile_simulator and self.backend != "batched":
            raise ConfigError("rollout.compile_simulator applies to the batched backend only")
        if self.evaluation_device is not None and not self.evaluation_device:
            raise ConfigError("schedule.evaluation_device must name a device")

    @property
    def rollout_steps(self) -> int:
        """Return the environment steps of one rollout, ``n_envs * n_steps``."""
        return self.n_envs * self.n_steps

    def to_json(self) -> dict[str, Any]:
        """Return the configuration as plain values with its schema tag."""
        values = asdict(self)
        if not values["compile_simulator"]:
            del values["compile_simulator"]
        if values["evaluation_device"] is None:
            del values["evaluation_device"]
        return {"schema": PPO_CONFIG_SCHEMA, **values}


def _section(mapping: Mapping[str, Any], name: str, keys: Sequence[str],
             optional: Sequence[str] = ()) -> Mapping[str, Any]:
    """Return section ``name`` after checking it holds exactly ``keys`` (plus ``optional``).

    Raises:
        ConfigError: The section is missing, or a key is missing or unknown.
    """
    if name not in mapping or not isinstance(mapping[name], Mapping):
        raise ConfigError(f"missing required section [{name}]")
    section: Mapping[str, Any] = mapping[name]
    unknown = sorted(set(section) - set(keys) - set(optional))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {unknown}; allowed: {list(keys)}")
    missing = [k for k in keys if k not in section]
    if missing:
        raise ConfigError(f"missing required key(s) in [{name}]: {missing}")
    return section


def _integer(value: Any, key: str) -> int:
    """Return ``value`` as an int, rejecting bools and non-integers (``ConfigError``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer, got {value!r}")
    return value


def _number(value: Any, key: str) -> float:
    """Return ``value`` as a float, rejecting bools and non-numbers (``ConfigError``)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key} must be a number, got {value!r}")
    return float(value)


def _integers(value: Any, key: str) -> tuple[int, ...]:
    """Return a list of integers as a tuple (``ConfigError`` otherwise)."""
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list of integers")
    return tuple(_integer(v, key) for v in value)


def ppo_config_from_mapping(mapping: Mapping[str, Any]) -> PPOConfig:
    """Build a ``PPOConfig`` from a parsed TOML mapping, strictly.

    Every key except ``update.target_kl``, ``rollout.compile_simulator`` and
    ``schedule.evaluation_device`` is required.

    Raises:
        ConfigError: Wrong schema, a missing or unknown key, or a bad value.
    """
    top = ("schema", "name", "training_config", "model_seeds", "device", "torch_threads",
           "output_dir")
    sections = ("network", "optimizer", "rollout", "update", "schedule")
    unknown = sorted(set(mapping) - set(top) - set(sections))
    if unknown:
        raise ConfigError(f"unknown key(s) at top level: {unknown}")
    if mapping.get("schema") != PPO_CONFIG_SCHEMA:
        raise ConfigError(f"schema {mapping.get('schema')!r} is not {PPO_CONFIG_SCHEMA!r}")
    missing = [k for k in top if k not in mapping]
    if missing:
        raise ConfigError(f"missing required top-level key(s): {missing}")
    network = _section(mapping, "network", ("hidden_sizes", "activation", "log_std_init"))
    optimizer = _section(mapping, "optimizer", ("learning_rate", "adam_eps"))
    rollout = _section(mapping, "rollout", ("backend", "n_envs", "n_steps", "pool_episodes"),
                       optional=("compile_simulator",))
    update = _section(
        mapping, "update",
        ("batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "entropy_coef",
         "value_coef", "max_grad_norm", "normalize_advantage"),
        optional=("target_kl",),
    )
    schedule = _section(
        mapping, "schedule", ("total_steps", "eval_every_rollouts", "selection_metric"),
        optional=("evaluation_device",),
    )
    if not isinstance(update["normalize_advantage"], bool):
        raise ConfigError("update.normalize_advantage must be a boolean")
    target_kl = update.get("target_kl")
    return PPOConfig(
        name=str(mapping["name"]),
        training_config=str(mapping["training_config"]),
        model_seeds=_integers(mapping["model_seeds"], "model_seeds"),
        device=str(mapping["device"]),
        torch_threads=_integer(mapping["torch_threads"], "torch_threads"),
        output_dir=str(mapping["output_dir"]),
        hidden_sizes=_integers(network["hidden_sizes"], "network.hidden_sizes"),
        activation=str(network["activation"]),
        log_std_init=_number(network["log_std_init"], "network.log_std_init"),
        learning_rate=_number(optimizer["learning_rate"], "optimizer.learning_rate"),
        adam_eps=_number(optimizer["adam_eps"], "optimizer.adam_eps"),
        backend=str(rollout["backend"]),
        n_envs=_integer(rollout["n_envs"], "rollout.n_envs"),
        n_steps=_integer(rollout["n_steps"], "rollout.n_steps"),
        pool_episodes=_integer(rollout["pool_episodes"], "rollout.pool_episodes"),
        batch_size=_integer(update["batch_size"], "update.batch_size"),
        n_epochs=_integer(update["n_epochs"], "update.n_epochs"),
        gamma=_number(update["gamma"], "update.gamma"),
        gae_lambda=_number(update["gae_lambda"], "update.gae_lambda"),
        clip_range=_number(update["clip_range"], "update.clip_range"),
        entropy_coef=_number(update["entropy_coef"], "update.entropy_coef"),
        value_coef=_number(update["value_coef"], "update.value_coef"),
        max_grad_norm=_number(update["max_grad_norm"], "update.max_grad_norm"),
        normalize_advantage=update["normalize_advantage"],
        target_kl=None if target_kl is None else _number(target_kl, "update.target_kl"),
        total_steps=_integer(schedule["total_steps"], "schedule.total_steps"),
        eval_every_rollouts=_integer(
            schedule["eval_every_rollouts"], "schedule.eval_every_rollouts"
        ),
        selection_metric=str(schedule["selection_metric"]),
        compile_simulator=rollout.get("compile_simulator", False),
        evaluation_device=(None if schedule.get("evaluation_device") is None
                           else str(schedule["evaluation_device"])),
    )


def load_ppo_config(path: str | Path) -> PPOConfig:
    """Parse a PPO configuration file.

    Raises:
        ConfigError: Invalid TOML or ``ppo_config_from_mapping`` fails.
        FileNotFoundError: No such file.
    """
    try:
        mapping = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return ppo_config_from_mapping(mapping)
