"""PPO training on ``ExecutionEnv`` through Stable-Baselines3: a thin, optional integration.

The learner is Stable-Baselines3's ``PPO`` with separate MLP actor and critic.
This module adds only what connects it to TapeForge: environment
construction from a split, the action map, rollout diagnostics, checkpoints
with provenance, validation-based checkpoint selection and evaluation into a
baseline run directory. Fills, fees, settlement, rewards and episode ends are
the reference simulator's; nothing here changes them.

Action map:
    See ``rl.ppo_env``: the policy acts in ``[-1, 1]`` and the map to the
    environment's ``[0, 1]`` fraction is deterministic and shared by training
    and evaluation.

Observations:
    ``ExecutionEnv`` standardises with statistics fitted once on training
    draws (``evaluation.training_statistics``) and frozen. The same
    ``ObservationStatistics`` object is stored in every checkpoint's
    provenance and applied by ``checkpoint_policy``; no running normaliser
    exists anywhere.

Episode ends:
    ``terminated`` (target reached or horizon, including the settlement step)
    ends the return: no value is bootstrapped. ``truncated`` (the tape ran
    out) is passed on as a time-limit truncation, for which the library adds
    ``gamma * V(terminal observation)`` to the last reward. Episodes cut by
    the split builder never truncate, so on a normal training split this path
    is exercised only by tests. The time-remaining fraction is part of the
    observation, so the finite-horizon value function is well defined.

Seeding:
    ``train`` seeds Python, NumPy and torch globally with the model seed
    (network initialisation, action noise, minibatch order) and gives
    environment ``i`` its own episode-selection stream derived from
    ``(model_seed, i)``. Training pools are draws of the training
    configuration's ``training.seeds``; evaluation episodes are those of its
    ``evaluation.seeds``. The three kinds of seed never mix.

Checkpoints and resuming:
    A checkpoint directory ``checkpoints/v<k>`` is written at rollout
    boundaries, where ``k`` is the policy version (completed updates). It
    holds ``model.zip`` (policy and value weights, optimiser state and
    hyperparameters, as saved by the library), ``provenance.json``
    (``evaluation.checkpoint_provenance``: configuration, split and data
    hashes, frozen statistics, simulator version) and ``collection.pt`` (the
    collection position: every environment's running episode and selection
    generator, the pending observations, the global torch / NumPy / Python
    generator states, the CUDA generator states when CUDA is in use, and the
    log counters). The ``batched`` backend stores one snapshot of the whole
    vector under ``batched_environments`` instead of per-environment ones.

    Loading ``model.zip`` alone restores the policy for evaluation; it does
    not continue training exactly. ``train(resume_from=...)`` restores the
    collection position as well and then continues bit-identically to the
    uninterrupted run. That guarantee holds for the ``serial`` and
    ``subprocess`` backends on CPU with the same software versions, thread
    count and configuration (tests check the serial, subprocess and
    CPU-tensor batched cases); it is not claimed across library versions,
    devices or thread counts. On CUDA a resumed run restores the same
    simulator state and generators but the update's floating-point
    reductions may differ, so it is a continuation of the same schedule, not
    a bit-identical replay. Checkpoints the ``batched`` backend wrote before
    it stored a collection snapshot can be evaluated but not resumed.
    ``collection.pt`` is a pickle: load only checkpoints you wrote.

Dependencies:
    ``stable-baselines3`` (``pip install -e '.[train]'``). Importing this
    module without it raises ``ImportError`` with that hint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

try:
    import gymnasium as gym
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.rl.ppo needs stable-baselines3; install it with "
        "pip install -e '.[train]'"
    ) from exc

from tapeforge.execution.data.datasets import DatasetError
from tapeforge.execution.data.splits import Splits, build_splits
from tapeforge.execution.research.experiment import (
    ConfigError,
    ExperimentConfig,
    acquire_days,
    config_from_mapping,
    config_mapping_from_manifest,
    create_run_directory,
    load_config,
    load_manifest,
    rebuild_splits,
    write_json_atomic,
)
from tapeforge.execution.rl.evaluation import (
    TrainingConfig,
    checkpoint_provenance,
    load_evaluation_manifest,
    load_training_config,
    training_statistics,
)
from tapeforge.execution.rl.normalization import ObservationStatistics, save_statistics
from tapeforge.execution.rl.ppo_config import PPOConfig, load_ppo_config
from tapeforge.execution.rl.ppo_env import ACTION_MAP, make_vec_env
from tapeforge.execution.rl.ppo_eval import (
    checkpoint_policy,
    evaluate_checkpoint,
    sha256_of_file,
)
from tapeforge.execution.runner import code_revision, run_policy
from tapeforge.execution.simulator import SIMULATOR_VERSION

__all__ = [
    "CHECKPOINTS_SUBDIR",
    "TRAINING_RUN_SCHEMA",
    "TrainingData",
    "prepare_training",
    "train",
]

TRAINING_RUN_SCHEMA: Final[str] = "tapeforge.execution.ppo-run/1"
CHECKPOINTS_SUBDIR: Final[str] = "checkpoints"
_ACTIVATIONS: Final[dict[str, type[torch.nn.Module]]] = {
    "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU,
}


# --- rollout diagnostics, checkpoints and selection ---------------------------------------------


def _cuda_generator_states() -> list[Any] | None:
    """Return every CUDA device's generator state, or ``None`` when CUDA is not in use.

    Only read once CUDA has been initialised (a policy on ``cuda`` does that),
    so a CPU run never initialises a device here.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        return list(torch.cuda.get_rng_state_all())
    return None


def _generator_states() -> tuple[Any, Any, Any, Any]:
    """Return the global torch, NumPy, Python and CUDA generator states.

    The library draws action noise from torch's global generator of the
    policy device (CPU or CUDA) and minibatch order from NumPy's legacy
    global one, so these are the states that decide how training continues.
    """
    return (torch.get_rng_state(), np.random.get_state(), random.getstate(),  # noqa: NPY002
            _cuda_generator_states())


def _set_generator_states(states: Sequence[Any]) -> None:
    """Restore the states returned by ``_generator_states``.

    A missing or ``None`` CUDA entry (a CPU run, or a checkpoint written
    before CUDA states were stored) leaves the CUDA generators unchanged.
    """
    torch.set_rng_state(states[0])
    np.random.set_state(states[1])  # noqa: NPY002
    random.setstate(states[2])
    cuda = states[3] if len(states) > 3 else None
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def _batched_env(env: VecEnv) -> Any:
    """Return the batched vector environment under any ``VecEnvWrapper`` layers.

    Raises:
        ConfigError: ``env`` does not wrap a batched environment.
    """
    while not hasattr(env, "collection_snapshot"):
        inner = getattr(env, "venv", None)
        if inner is None:
            raise ConfigError("the training environment stores no collection position")
        env = inner
    return env


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean of ``values`` or ``None`` when there are none."""
    return float(np.mean(values)) if values else None


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line."""
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return the rows of a JSON-lines file (empty when it does not exist)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _TrainingLog(BaseCallback):
    """Rollout diagnostics, scheduled checkpoints and validation for one training run.

    Rollout ``k`` (0-based) is collected by policy version ``k`` and followed
    by update ``k``, which produces version ``k + 1``. One row per rollout is
    appended to ``rollouts.jsonl`` once its update has finished; versions on
    the schedule are checkpointed and evaluated before they collect anything.
    With ``resume_state`` set (a loaded ``collection.pt``) the run starts
    from that collection position instead of fresh resets.
    Validation and checkpointing run with the global generators saved and
    restored around them, so they cannot perturb training.
    """

    def __init__(self, run_dir: Path, config: PPOConfig, evaluate: Callable[[PPO], dict[str, Any]],
                 provenance: Callable[[int, int, str], dict[str, Any]]) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.config = config
        self._evaluate = evaluate
        self._provenance = provenance
        self.resume_state: dict[str, Any] | None = None
        self.version = 0
        self.seconds = {"collect": 0.0, "update": 0.0, "evaluate": 0.0, "checkpoint": 0.0}
        self.evaluations: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._mark = 0.0
        self._skip_boundary: int | None = None

    # -- library hooks ------------------------------------------------------------------------

    def _on_training_start(self) -> None:
        """Restore the collection position of ``resume_state``, if any."""
        state = self.resume_state
        if state is None:
            return
        if state["environments"] is None:
            _batched_env(self.training_env).restore_collection(state["batched_environments"])
        else:
            for index, snapshot in enumerate(state["environments"]):
                self.training_env.env_method("restore", snapshot, indices=[index])
        model: PPO = self.model  # type: ignore[assignment]
        model._last_obs = state["last_obs"]
        model._last_episode_starts = state["last_episode_starts"]
        self.version = int(state["version"])
        self.seconds = dict(state["seconds"])
        self.evaluations = list(state["evaluations"])
        self._skip_boundary = self.version
        # Last: everything above may have consumed random numbers.
        _set_generator_states((state["torch_rng"], state["numpy_rng"], state["python_rng"],
                               state.get("cuda_rng")))

    def _on_rollout_start(self) -> None:
        self._finish_pending()
        if self.version != self._skip_boundary and (
            self.version % self.config.eval_every_rollouts == 0
        ):
            self._boundary()
        self._episodes = []
        self._mark = time.perf_counter()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self._episodes.append(info["episode_summary"])
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        collect = now - self._mark
        self.seconds["collect"] += collect
        episodes = self._episodes
        self._pending = {
            "rollout": self.version,
            "policy_version": self.version,
            "env_steps": int(self.model.num_timesteps),
            "collect_seconds": collect,
            "episodes_finished": len(episodes),
            "episodes_truncated": sum(1 for e in episodes if e["truncated"]),
            "mean_reward_sum": _mean([e["reward_sum"] for e in episodes]),
            "mean_shortfall_bps": _mean([e["shortfall_bps"] for e in episodes]),
            "mean_terminal_bps": _mean([e["terminal_bps"] for e in episodes]),
            "mean_filled_fraction": _mean([e["filled_fraction"] for e in episodes]),
            "mean_settlement_fraction": _mean([e["settlement_fraction"] for e in episodes]),
            "settled_episode_rate": _mean(
                [float(e["settlement_fraction"] > 0) for e in episodes]
            ),
        }
        self.version += 1
        self._mark = now

    def _on_training_end(self) -> None:
        self._finish_pending()
        if self.version != self._skip_boundary and not (
            self.evaluations and self.evaluations[-1]["policy_version"] == self.version
        ):
            self._boundary()

    # -- bookkeeping -----------------------------------------------------------------------------

    def _finish_pending(self) -> None:
        """Complete the row of the last rollout with the diagnostics of the update after it."""
        if self._pending is None:
            return
        update = time.perf_counter() - self._mark
        self.seconds["update"] += update
        logged = self.model.logger.name_to_value
        entropy_loss = logged.get("train/entropy_loss")
        row = {
            **self._pending,
            "update_seconds": update,
            "train_seconds": self.seconds["collect"] + self.seconds["update"],
            "policy_entropy": None if entropy_loss is None else -float(entropy_loss),
            "approx_kl": _optional(logged.get("train/approx_kl")),
            "clip_fraction": _optional(logged.get("train/clip_fraction")),
            "value_loss": _optional(logged.get("train/value_loss")),
            "policy_gradient_loss": _optional(logged.get("train/policy_gradient_loss")),
            "explained_variance": _optional(logged.get("train/explained_variance")),
            "action_std": _optional(logged.get("train/std")),
            "n_updates": _optional(logged.get("train/n_updates")),
        }
        _append_jsonl(self.run_dir / "rollouts.jsonl", row)
        self._pending = None

    def _boundary(self) -> None:
        """Checkpoint and evaluate the current policy version without disturbing training."""
        generators = _generator_states()
        started = time.perf_counter()
        name = f"v{self.version:05d}"
        checkpoint_dir = self.run_dir / CHECKPOINTS_SUBDIR / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model: PPO = self.model  # type: ignore[assignment]
        model.save(checkpoint_dir / "model.zip")
        model_sha256 = sha256_of_file(checkpoint_dir / "model.zip")
        write_json_atomic(
            checkpoint_dir / "provenance.json",
            self._provenance(self.version, int(model.num_timesteps), model_sha256),
        )
        saved = time.perf_counter()
        result = self._evaluate(model)
        evaluated = time.perf_counter()
        self.seconds["checkpoint"] += saved - started
        self.seconds["evaluate"] += evaluated - saved
        self.evaluations.append({
            "policy_version": self.version,
            "env_steps": int(model.num_timesteps),
            "checkpoint": f"{CHECKPOINTS_SUBDIR}/{name}",
            "model_sha256": model_sha256,
            "train_seconds": self.seconds["collect"] + self.seconds["update"],
            "evaluate_seconds": evaluated - saved,
            **result,
        })
        _append_jsonl(self.run_dir / "evaluations.jsonl", self.evaluations[-1])
        write_json_atomic(self.run_dir / "selection.json", self.selection())
        # The collection position is written last so that it includes this evaluation.
        environments = self._environment_snapshots()
        torch.save(
            {
                "environments": environments,
                "batched_environments": (
                    None if environments is not None
                    else _batched_env(self.training_env).collection_snapshot()
                ),
                "last_obs": model._last_obs,
                "last_episode_starts": model._last_episode_starts,
                "torch_rng": generators[0],
                "numpy_rng": generators[1],
                "python_rng": generators[2],
                "cuda_rng": generators[3],
                "version": self.version,
                "seconds": dict(self.seconds),
                "evaluations": list(self.evaluations),
            },
            checkpoint_dir / "collection.pt",
        )
        _set_generator_states(generators)

    def _environment_snapshots(self) -> list[Any] | None:
        """Return every environment's collection position, or ``None`` on the batched backend.

        The batched backend's position is ``_batched_env(...).collection_snapshot()``.
        """
        try:
            return list(self.training_env.env_method("snapshot"))
        except NotImplementedError:
            return None

    def selection(self) -> dict[str, Any]:
        """Return the checkpoint-selection record: the rule, every evaluation and the choice.

        The selected checkpoint has the lowest metric; the earliest wins a tie.
        """
        metric = self.config.selection_metric
        best = min(self.evaluations, key=lambda e: (e[metric], e["policy_version"]))
        return {
            "rule": (
                f"lowest {metric} over the checkpoints evaluated every "
                f"{self.config.eval_every_rollouts} policy versions and after the last update; "
                "earliest wins a tie; the test role is never consulted"
            ),
            "metric": metric,
            "selected": {k: best[k] for k in ("policy_version", "env_steps", "checkpoint",
                                               "model_sha256", metric)},
            "evaluations": [
                {k: e[k] for k in ("policy_version", "env_steps", "checkpoint", metric)}
                for e in self.evaluations
            ],
        }


def _optional(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when the library logged nothing."""
    return None if value is None else float(value)


# --- training ------------------------------------------------------------------------------


def _validation_evaluator(
    experiment: ExperimentConfig, training: TrainingConfig, splits: Splits,
    statistics: ObservationStatistics, metric: str,
) -> Callable[[PPO], dict[str, Any]]:
    """Return the function that scores a model on the fixed validation episodes.

    The episodes are draws ``0 .. n - 1`` of every ``evaluation.seeds`` entry
    in the validation role, with ``n`` the experiment's validation count:
    exactly the identifiers a baseline run of the same experiment replays.
    """
    n_episodes = experiment.n_episodes("validation")

    def evaluate(model: PPO) -> dict[str, Any]:
        policy = checkpoint_policy(model, statistics)
        per_seed = {}
        for seed in training.eval_seeds:
            record, _ = run_policy(
                splits, "validation", policy, "ppo", experiment.template, n_episodes, seed,
                experiment.rho_choices_bps,
            )
            per_seed[str(seed)] = {
                k: record["summary"][k]
                for k in ("n_episodes", "n_truncated", "mean_shortfall_bps", "p50_shortfall_bps",
                          "p95_shortfall_bps", "completion_by_fills_rate",
                          "mean_settlement_fraction")
            }
        return {
            metric: float(np.mean([s["mean_shortfall_bps"] for s in per_seed.values()])),
            "validation": per_seed,
        }

    return evaluate


def _drop_rows_from(path: Path, key: str, first_dropped: int) -> None:
    """Rewrite a JSON-lines file without the rows whose ``key`` is ``>= first_dropped``."""
    rows = [r for r in _read_jsonl(path) if r[key] < first_dropped]
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


@dataclass(frozen=True, eq=False)
class TrainingData:
    """What every model seed of one configuration shares: data, splits, frozen statistics.

    Attributes:
        training: The training configuration ``PPOConfig.training_config`` names.
        experiment: Its experiment configuration, with downloads disabled.
        splits: The split tapes.
        data_provenance: The ``data`` list an experiment manifest would carry.
        statistics: Observation statistics fitted on training draws only.
        data_seconds: Wall time of loading the days and building the splits.
        statistics_seconds: Wall time of fitting the statistics.
    """

    training: TrainingConfig
    experiment: ExperimentConfig
    splits: Splits
    data_provenance: tuple[dict[str, Any], ...]
    statistics: ObservationStatistics
    data_seconds: float
    statistics_seconds: float


def prepare_training(config: PPOConfig) -> TrainingData:
    """Load the data a PPO configuration trains on and fit its frozen statistics.

    The result can be passed to ``train`` for several model seeds so the
    days are read once.

    Raises:
        ConfigError: The training configuration does not evaluate the
            validation role, which checkpoint selection needs.
        DatasetError: The experiment's data is not available locally;
            training never downloads.
    """
    training = load_training_config(config.training_config)
    if "validation" not in training.eval_roles:
        raise ConfigError("checkpoint selection needs 'validation' in evaluation.roles")
    experiment = load_config(training.experiment_config)
    experiment = replace(experiment, data=replace(experiment.data, allow_download=False))
    started = time.perf_counter()
    tapes, data_provenance = acquire_days(experiment)
    splits = build_splits(tapes, experiment.split, experiment.template)
    data_seconds = time.perf_counter() - started
    started = time.perf_counter()
    statistics = training_statistics(experiment, splits, training)
    return TrainingData(
        training=training, experiment=experiment, splits=splits,
        data_provenance=data_provenance, statistics=statistics, data_seconds=data_seconds,
        statistics_seconds=time.perf_counter() - started,
    )


def train(
    config: PPOConfig,
    model_seed: int,
    *,
    data: TrainingData | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    total_steps: int | None = None,
    evaluation_manifest_path: str | Path | None = None,
    vec_env_builder: Callable[..., VecEnv] | None = None,
) -> Path:
    """Train one model seed and return its run directory.

    The directory holds ``config.json`` (the resolved PPO, training and
    experiment configurations), ``statistics.json`` (the frozen training
    statistics), ``rollouts.jsonl`` (one row per rollout: environment steps,
    wall time, the policy version that collected it, episode reward,
    shortfall and settlement dependence, and the entropy, approximate KL,
    clip fraction, value loss and explained variance of the update after
    it), ``evaluations.jsonl`` and ``selection.json`` (validation scores of
    the scheduled checkpoints and the one selected), ``checkpoints/`` and
    ``summary.json`` (step counts and the time spent loading data, building
    environments and the model, collecting, updating, evaluating and
    checkpointing).

    Args:
        config: The PPO configuration.
        model_seed: Seeds initialisation, action noise, minibatch order and
            the environments' episode selection.
        data: ``prepare_training(config)`` when several seeds share one
            load; loaded here when ``None``.
        output_dir: Overrides ``config.output_dir``. Ignored when resuming.
        resume_from: A checkpoint directory of an earlier run of the same
            configuration and seed; training continues in that run's
            directory from the checkpoint's policy version. Log rows and
            evaluations written after that version are dropped.
        total_steps: Overrides ``config.total_steps`` (a multiple of the
            rollout size); the only setting a resumed run may change.
        evaluation_manifest_path: A file written by ``evaluation manifest``;
            its digest is recorded in every checkpoint so a later evaluation
            can check it is judged on the same episodes.
        vec_env_builder: Replaces ``make_vec_env`` (same arguments); the
            batched backend and tests use it.

    Raises:
        ConfigError: An inconsistent configuration, or a resume whose
            configuration or seed differs from the checkpoint's.
        DatasetError: The experiment's data is not available locally.
    """
    if total_steps is not None:
        config = replace(config, total_steps=total_steps)
    if data is None:
        data = prepare_training(config)
    training, experiment, splits = data.training, data.experiment, data.splits
    statistics, data_provenance = data.statistics, data.data_provenance
    evaluation_digest = None
    if evaluation_manifest_path is not None:
        evaluation_digest = load_evaluation_manifest(evaluation_manifest_path)["digest"]

    resolved = {
        "schema": TRAINING_RUN_SCHEMA,
        "model_seed": model_seed,
        "ppo": config.to_json(),
        "training": training.to_json(),
        "experiment": json.loads(json.dumps(experiment.to_json())),
        "evaluation_digest": evaluation_digest,
        "action_map": ACTION_MAP,
        "simulator_version": SIMULATOR_VERSION,
    }
    # JSON-normalised (tuples become lists) so it compares equal to its own file round trip.
    resolved = json.loads(json.dumps(resolved))
    resume_state: dict[str, Any] | None = None
    if resume_from is None:
        run_dir = create_run_directory(
            output_dir or config.output_dir, f"{config.name}-seed{model_seed}"
        )
        write_json_atomic(run_dir / "config.json", resolved)
        save_statistics(statistics, run_dir / "statistics.json")
    else:
        checkpoint_dir = Path(resume_from)
        run_dir = checkpoint_dir.parent.parent
        stored = json.loads((run_dir / "config.json").read_text())
        for payload in (stored, resolved):
            payload["ppo"] = {k: v for k, v in payload["ppo"].items() if k != "total_steps"}
        if stored != resolved:
            differing = sorted(k for k in resolved if stored.get(k) != resolved[k])
            raise ConfigError(
                f"cannot resume {run_dir}: the configuration differs in {differing}; only "
                "schedule.total_steps may change"
            )
        resolved["ppo"] = json.loads(json.dumps(config.to_json()))
        write_json_atomic(run_dir / "config.json", resolved)
        resume_state = torch.load(checkpoint_dir / "collection.pt", weights_only=False)
        assert resume_state is not None
        if resume_state["environments"] is None and (
            resume_state.get("batched_environments") is None
        ):
            raise ConfigError(
                f"{checkpoint_dir} was written by a batched backend that stored no "
                "collection position; it can be evaluated but not resumed"
            )
        _drop_rows_from(run_dir / "rollouts.jsonl", "rollout", int(resume_state["version"]))
        _drop_rows_from(
            run_dir / "evaluations.jsonl", "policy_version", int(resume_state["version"]) + 1
        )

    torch.set_num_threads(config.torch_threads)
    set_random_seed(model_seed)
    setup_started = time.perf_counter()
    builder = make_vec_env if vec_env_builder is None else vec_env_builder
    vec_env = builder(config, experiment, training, splits, statistics, model_seed)

    def provenance(version: int, env_steps: int, model_sha256: str) -> dict[str, Any]:
        return checkpoint_provenance(
            experiment, splits, data_provenance, model_seed, statistics,
            reward_unit=training.reward_unit, evaluation_digest=evaluation_digest,
            training=training,
            extra={
                "framework": f"stable-baselines3 {stable_baselines3.__version__}",
                "torch": torch.__version__,
                "ppo_config": config.to_json(),
                "action_map": ACTION_MAP,
                "policy_version": version,
                "env_steps": env_steps,
                "model_sha256": model_sha256,
            },
        )

    log = _TrainingLog(
        run_dir, config,
        _validation_evaluator(experiment, training, splits, statistics, config.selection_metric),
        provenance,
    )
    try:
        if resume_state is None:
            model = PPO(
                "MlpPolicy", vec_env, learning_rate=config.learning_rate, n_steps=config.n_steps,
                batch_size=config.batch_size, n_epochs=config.n_epochs, gamma=config.gamma,
                gae_lambda=config.gae_lambda, clip_range=config.clip_range,
                normalize_advantage=config.normalize_advantage, ent_coef=config.entropy_coef,
                vf_coef=config.value_coef, max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl, device=config.device, seed=None, verbose=0,
                policy_kwargs={
                    "net_arch": {"pi": list(config.hidden_sizes), "vf": list(config.hidden_sizes)},
                    "activation_fn": _ACTIVATIONS[config.activation],
                    "log_std_init": config.log_std_init,
                    "optimizer_kwargs": {"eps": config.adam_eps},
                },
            )
            remaining = config.total_steps
        else:
            model = PPO.load(Path(resume_from) / "model.zip",  # type: ignore[arg-type]
                             env=vec_env, device=config.device)
            log.resume_state = resume_state
            remaining = config.total_steps - int(model.num_timesteps)
            if remaining < 0:
                raise ConfigError(
                    f"the checkpoint is at {model.num_timesteps} steps, past total_steps "
                    f"{config.total_steps}"
                )
        started = time.perf_counter()
        setup_seconds = started - setup_started
        model.learn(total_timesteps=remaining, callback=log,
                    reset_num_timesteps=resume_state is None)
        learn_seconds = time.perf_counter() - started
    finally:
        vec_env.close()

    train_seconds = log.seconds["collect"] + log.seconds["update"]
    write_json_atomic(run_dir / "selection.json", log.selection())
    write_json_atomic(run_dir / "summary.json", {
        "schema": TRAINING_RUN_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "model_seed": model_seed,
        "resumed_from": None if resume_from is None else str(resume_from),
        "env_steps": int(model.num_timesteps),
        "policy_versions": log.version,
        "backend": config.backend,
        "n_envs": config.n_envs,
        "device": config.device,
        "torch_threads": config.torch_threads,
        "seconds": {
            "data_load": data.data_seconds,
            "setup": setup_seconds,
            "statistics": data.statistics_seconds,
            **log.seconds,
            "train": train_seconds,
            "learn_call_this_session": learn_seconds,
        },
        "env_steps_per_collect_second": (
            model.num_timesteps / log.seconds["collect"] if log.seconds["collect"] else None
        ),
        "env_steps_per_train_second": (
            model.num_timesteps / train_seconds if train_seconds else None
        ),
        "selected": log.selection()["selected"],
        "versions": {
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
            "gymnasium": gym.__version__, "numpy": np.__version__,
        },
    })
    return run_dir


# --- command line ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``train <ppo.toml> [--seed N ...]`` or ``evaluate <baseline run> NAME=CHECKPOINT ...``.

    Returns:
        0 on success; 1 on a configuration, data or provenance failure, with
        the reason on stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.rl.ppo")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="train one or more model seeds")
    train_parser.add_argument("config", help="PPO configuration (TOML)")
    train_parser.add_argument("--seed", type=int, action="append", dest="seeds", default=None,
                              help="model-training seed; repeat to train several on one data "
                                   "load (default: every model_seeds entry)")
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--resume", default=None, metavar="CHECKPOINT_DIR",
                              help="continue the run that wrote this checkpoint (one seed)")
    train_parser.add_argument("--total-steps", type=int, default=None,
                              help="override schedule.total_steps")
    train_parser.add_argument("--evaluation-manifest", default=None,
                              help="record this evaluation set's digest in every checkpoint")
    evaluate_parser = sub.add_parser(
        "evaluate", help="replay checkpoints on a baseline run's episodes"
    )
    evaluate_parser.add_argument("run_dir", help="baseline run directory to add the records to")
    evaluate_parser.add_argument(
        "policies", nargs="+", metavar="NAME=CHECKPOINT",
        help="record name and a checkpoint directory, or a training run (uses its selection)",
    )
    evaluate_parser.add_argument("--role", action="append", dest="roles", default=None)
    evaluate_parser.add_argument("--include-test", action="store_true",
                                 help="evaluate the held-out role (needs authorization)")
    evaluate_parser.add_argument(
        "--allow-setting-mismatch", action="store_true",
        help="evaluate under a simulator template the checkpoints were not trained on",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = load_ppo_config(args.config)
            seeds = list(config.model_seeds) if args.seeds is None else args.seeds
            if args.resume is not None and len(seeds) != 1:
                raise ConfigError("--resume continues one run; give exactly one --seed")
            data = prepare_training(config)
            for seed in seeds:
                run_dir = train(
                    config, seed, data=data, output_dir=args.output_dir, resume_from=args.resume,
                    total_steps=args.total_steps,
                    evaluation_manifest_path=args.evaluation_manifest,
                )
                summary = json.loads((run_dir / "summary.json").read_text())
                seconds = summary["seconds"]
                print(f"training run: {run_dir}")
                print(f"  seed {seed}: {summary['env_steps']} steps, "
                      f"{summary['policy_versions']} policy versions; collect "
                      f"{seconds['collect']:.1f}s, update {seconds['update']:.1f}s, evaluate "
                      f"{seconds['evaluate']:.1f}s")
                print(f"  selected {summary['selected']}")
            return 0
        pairs = [item.partition("=") for item in args.policies]
        if any(not name or not checkpoint for name, _, checkpoint in pairs):
            raise ValueError("each policy is given as NAME=CHECKPOINT")
        manifest = load_manifest(args.run_dir)
        splits = rebuild_splits(config_from_mapping(config_mapping_from_manifest(
            manifest["config"]
        )))
        for name, _, checkpoint in pairs:
            written = evaluate_checkpoint(
                checkpoint, args.run_dir, name, roles=args.roles,
                include_test=args.include_test,
                strict_settings=not args.allow_setting_mismatch, splits=splits,
            )
            for path in written:
                print(f"record: {path}")
        return 0
    except (ConfigError, DatasetError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
