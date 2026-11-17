"""Loading and evaluating PPO checkpoints against a baseline run.

A checkpoint directory holds ``model.zip`` (policy and value weights and the
optimiser state, as saved by Stable-Baselines3) and ``provenance.json``
(``evaluation.checkpoint_provenance``). ``load_checkpoint`` restores the
policy for evaluation only: it does not restore the collection position a
training run needs to continue exactly (see ``rl.ppo``).

``evaluate_checkpoint`` replays the deterministic policy on the episodes a
baseline run directory lists, with the checkpoint's own frozen observation
statistics, and registers the records there so ``research.report`` pairs the
policy with the baselines on identical episode identifiers.

Dependencies:
    ``stable-baselines3`` (``pip install -e '.[train]'``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tapeforge.execution.rl.ppo_eval needs stable-baselines3; install it with "
        "pip install -e '.[train]'"
    ) from exc

from tapeforge.execution.data.splits import Splits
from tapeforge.execution.research.experiment import load_manifest
from tapeforge.execution.rl.evaluation import (
    canonical_sha256,
    evaluate_policy,
    evaluation_manifest,
    validate_checkpoint_provenance,
    vector_policy,
)
from tapeforge.execution.rl.normalization import ObservationStatistics, statistics_from_json
from tapeforge.execution.rl.ppo_env import policy_action_to_env
from tapeforge.execution.runner import Policy

__all__ = [
    "checkpoint_policy",
    "evaluate_checkpoint",
    "load_checkpoint",
    "selected_checkpoint",
    "sha256_of_file",
]


def checkpoint_policy(model: PPO, statistics: ObservationStatistics | None) -> Policy:
    """Return the deterministic evaluation policy of a trained model.

    The policy standardises ``simulator.observe`` with ``statistics`` (the
    frozen training statistics), takes the Gaussian mean and applies
    ``policy_action_to_env``: the training-time observation and action path
    without the sampling noise.
    """

    def act(vector: np.ndarray) -> float:
        action, _ = model.policy.predict(vector, deterministic=True)
        return policy_action_to_env(action)

    return vector_policy(act, statistics)


def sha256_of_file(path: Path) -> str:
    """Return the sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
    checkpoint_dir: str | Path, device: str = "cpu"
) -> tuple[PPO, dict[str, Any], ObservationStatistics | None]:
    """Load a checkpoint for evaluation: the model, its provenance and its frozen statistics.

    This restores the policy, the value function and the optimiser state. It
    does not restore the collection position; see the module docstring for
    what exact continuation needs.

    Raises:
        FileNotFoundError: The directory lacks ``model.zip`` or
            ``provenance.json``.
        ValueError: The provenance fails ``validate_checkpoint_provenance``
            or does not match the stored model file.
    """
    root = Path(checkpoint_dir)
    provenance: dict[str, Any] = json.loads((root / "provenance.json").read_text())
    validate_checkpoint_provenance(provenance)
    model_path = root / "model.zip"
    expected = (provenance.get("extra") or {}).get("model_sha256")
    if expected != sha256_of_file(model_path):
        raise ValueError(f"{model_path} does not match the sha256 recorded in its provenance")
    model = PPO.load(model_path, device=device)
    statistics_json = provenance["observation_statistics"]
    statistics = None if statistics_json is None else statistics_from_json(statistics_json)
    return model, provenance, statistics


def selected_checkpoint(path: str | Path) -> Path:
    """Return the checkpoint directory ``path`` names.

    ``path`` is either a checkpoint directory or a training run directory,
    whose ``selection.json`` names the checkpoint chosen on validation.

    Raises:
        FileNotFoundError: Neither a checkpoint nor a run with a selection.
    """
    root = Path(path)
    if (root / "model.zip").exists():
        return root
    selection = root / "selection.json"
    if not selection.exists():
        raise FileNotFoundError(
            f"{root} is neither a checkpoint directory nor a training run with selection.json"
        )
    name: str = json.loads(selection.read_text())["selected"]["checkpoint"]
    return root / name


def evaluate_checkpoint(
    checkpoint: str | Path,
    run_dir: str | Path,
    policy_name: str,
    *,
    roles: Sequence[str] | None = None,
    include_test: bool = False,
    strict_settings: bool = True,
    splits: Splits | None = None,
) -> list[Path]:
    """Replay a checkpoint on a baseline run's episodes and register its records there.

    The checkpoint must have been trained on the world the run evaluated:
    the data digests and the split record must match, and with
    ``strict_settings`` so must the simulator template and, when the
    checkpoint names one, the evaluation-set digest. ``strict_settings=False``
    permits a run whose template differs (a sensitivity run that evaluates a
    policy under settings it was not trained on), including a geometry
    change that re-derives the split record from the same days and roles;
    the differing keys are recorded with the evaluation. A run over other
    days or another role assignment is always refused. Observations are
    standardised with the checkpoint's own frozen statistics.

    Returns:
        The record paths written by ``evaluation.evaluate_policy``.

    Raises:
        ValueError: A mismatch described above, or anything
            ``evaluate_policy`` rejects (the test role without
            ``include_test`` among them).
    """
    checkpoint_dir = selected_checkpoint(checkpoint)
    model, provenance, statistics = load_checkpoint(checkpoint_dir)
    manifest = load_manifest(run_dir)
    # A template change to the episode geometry (latency, data delay, ...) recomputes the
    # admissible starts, so the split record differs although the days and roles do not.
    same_split = canonical_sha256(manifest["split"]) == provenance["split_sha256"]
    same_days = manifest["config"]["split"] == provenance["config"]["split"]
    if not same_split and (strict_settings or not same_days):
        raise ValueError("the run's split record differs from the one the checkpoint trained on")
    digests = [d.get("sha256") or f"seed={d.get('seed')}" for d in manifest["data"]]
    if digests != provenance["data_digests"]:
        raise ValueError("the run's data digests differ from the checkpoint's")
    trained, evaluated = provenance["config"]["template"], manifest["config"]["template"]
    mismatch = sorted(
        k for k in set(trained) | set(evaluated) if trained.get(k) != evaluated.get(k)
    )
    digest = evaluation_manifest(run_dir, include_test=include_test)["digest"]
    if strict_settings:
        if mismatch:
            raise ValueError(
                f"the run's simulator template differs from the checkpoint's in {mismatch}; "
                "pass strict_settings=False to evaluate under settings it was not trained on"
            )
        expected = provenance["evaluation_digest"]
        if expected is not None and not include_test and expected != digest:
            raise ValueError(
                "the run's evaluation set differs from the one the checkpoint was selected on"
            )
    return evaluate_policy(
        run_dir, checkpoint_policy(model, statistics), policy_name, roles=roles,
        include_test=include_test, statistics=statistics, splits=splits,
        provenance={
            **provenance,
            "evaluated_checkpoint": str(checkpoint_dir),
            "evaluated_set_digest": digest,
            "template_keys_differing_from_training": mismatch,
            "split_record_differs_from_training": not same_split,
        },
    )
