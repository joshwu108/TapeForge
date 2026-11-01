"""Policy evaluation against the baseline records: fixed episode manifests, an external-callable
interface, strict training/evaluation configuration and a model-agnostic checkpoint provenance.

What a trainer needs before its first update, and nothing of the trainer itself:

* ``TrainingConfig`` / ``load_training_config``: a strict TOML description of
  which experiment configuration supplies the episodes, which role and seeds
  train, which reference baseline fits the frozen observation statistics,
  and which roles and seeds evaluate. Unknown keys are errors; the test role
  is refused unless named explicitly.
* ``evaluation_manifest`` / ``write_evaluation_manifest`` /
  ``load_evaluation_manifest``: the fixed list of evaluation episodes of a
  run directory (the same ``<role>/s<seed>/i<index>`` identifiers every
  baseline replayed), with the split record, data digests and the simulator
  version, hashed so a checkpoint can name the exact set it was judged on.
* ``vector_policy``: adapts a callable over the learner's float32
  observation vector (standardised with frozen statistics when given) to the
  runner's ``Policy``; the returned action is validated, never clipped.
* ``evaluate_policy``: replays any ``Policy`` on the episodes a run
  directory lists, writes run records with the same schema as the baselines
  into that directory and registers them in its manifest, so ``report``
  pairs the policy against the baselines on identical episode identifiers.
* ``checkpoint_provenance`` / ``validate_checkpoint_provenance``: the
  provenance a checkpoint must carry to be evaluated later (configuration,
  split and data hashes, normalisation, seed, action and reward conventions,
  simulator version); the weights themselves are the trainer's business.

Not covered:
    No neural policy, optimiser, rollout buffer, PPO loop or checkpoint
    selection. Nothing here fits statistics on anything but the training
    role.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from tapeforge.execution.baselines import BASELINES
from tapeforge.execution.experiment import (
    MANIFEST_NAME,
    RECORDS_SUBDIR,
    ConfigError,
    ExperimentConfig,
    config_from_mapping,
    config_mapping_from_manifest,
    load_manifest,
    rebuild_splits,
    write_json_atomic,
)
from tapeforge.execution.normalization import (
    STATISTICS_SCHEMA,
    ObservationStatistics,
    normalize,
    observation_statistics,
    statistics_from_json,
)
from tapeforge.execution.runner import (
    Observation,
    Policy,
    code_revision,
    run_policy,
)
from tapeforge.execution.sampling import sample_episodes
from tapeforge.execution.simulator import OBSERVATION_FIELDS, SIMULATOR_VERSION
from tapeforge.execution.splits import ROLES, Splits

__all__ = [
    "CHECKPOINT_PROVENANCE_SCHEMA",
    "EVALUATION_MANIFEST_SCHEMA",
    "TRAINING_CONFIG_SCHEMA",
    "TrainingConfig",
    "VectorPolicy",
    "canonical_sha256",
    "checkpoint_provenance",
    "evaluate_policy",
    "evaluation_manifest",
    "load_evaluation_manifest",
    "load_training_config",
    "training_config_from_mapping",
    "training_statistics",
    "validate_checkpoint_provenance",
    "vector_policy",
    "write_evaluation_manifest",
]

TRAINING_CONFIG_SCHEMA: Final[str] = "tapeforge.execution.training-config/1"
EVALUATION_MANIFEST_SCHEMA: Final[str] = "tapeforge.execution.evaluation-manifest/1"
CHECKPOINT_PROVENANCE_SCHEMA: Final[str] = "tapeforge.execution.checkpoint-provenance/1"

# Learner-facing conventions, recorded verbatim in every checkpoint provenance
# so an evaluation can refuse a checkpoint trained under other conventions.
ACTION_CONVENTION: Final[dict[str, Any]] = {
    "space": "Box(0, 1, shape=(1,), float32)",
    "meaning": "fraction of the remaining quantity to submit at this decision",
    "quantity": "floor(action * remaining_qty_ticks), the only float-to-integer conversion",
    "out_of_range": "raises ValueError; the environment never clips",
}
REWARD_CONVENTION: Final[dict[str, Any]] = {
    "per_step": "-(cost + risk + terminal) in cash ticks, exact integers in info",
    "scalar_units": ("bps", "ticks"),
    "bps": "reward_ticks * 10_000 / (arrival_price_ticks * target_qty_ticks)",
    "episode_sum_terminated": "-(implementation shortfall + risk_lambda * sum(remaining_after))",
    "truncated": "no terminal term, nothing settled; fills-only shortfall",
}

_POLICY_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def canonical_sha256(payload: Any) -> str:
    """Return the sha256 of ``payload`` serialised as compact, key-sorted JSON."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


# --- training / evaluation configuration ----------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """What a trainer and its evaluation read; validated strictly, nothing inferred.

    Attributes:
        experiment_config: Path of the experiment configuration (schema 2)
            that supplies data, split, template and episode plan.
        policy_name: Name the evaluated policy's records are written under;
            must not be a baseline name.
        train_role: Role the trainer samples from; always ``"train"``.
        train_seeds: Seeds of the training pools; distinct, non-negative.
        selection: ``"random"`` or ``"sequential"`` for training resets.
        reward_unit: ``"bps"`` or ``"ticks"``.
        statistics_policy: Baseline whose training observations fit the
            frozen statistics.
        statistics_seed: Seed of the training draws the statistics are
            fitted on.
        statistics_file: Optional path the statistics are saved to and
            loaded from.
        eval_roles: Roles evaluated; ``"test"`` only with ``include_test``.
        eval_seeds: Seeds evaluated; distinct, non-negative.
        include_test: Whether the held-out role may be evaluated.
    """

    experiment_config: str
    policy_name: str
    train_role: str
    train_seeds: tuple[int, ...]
    selection: str
    reward_unit: str
    statistics_policy: str
    statistics_seed: int
    statistics_file: str | None
    eval_roles: tuple[str, ...]
    eval_seeds: tuple[int, ...]
    include_test: bool

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            ConfigError: A bad policy name or a baseline name, a training
                role other than ``"train"``, bad seeds, an unknown selection,
                reward unit, statistics policy or role, or the test role
                without ``include_test``.
        """
        if not _POLICY_NAME.match(self.policy_name):
            raise ConfigError(f"policy_name must match {_POLICY_NAME.pattern}")
        if self.policy_name in BASELINES:
            raise ConfigError(f"policy_name {self.policy_name!r} is a baseline name")
        if self.train_role != "train":
            raise ConfigError("training.role must be 'train'; statistics and learning never "
                              "touch validation or test data")
        for name, seeds in (("training.seeds", self.train_seeds),
                            ("evaluation.seeds", self.eval_seeds)):
            if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) < len(seeds):
                raise ConfigError(f"{name} must be a non-empty list of distinct non-negative ints")
        if self.selection not in ("random", "sequential"):
            raise ConfigError("training.selection must be 'random' or 'sequential'")
        if self.reward_unit not in ("bps", "ticks"):
            raise ConfigError("training.reward_unit must be 'bps' or 'ticks'")
        if self.statistics_policy not in BASELINES:
            raise ConfigError(f"normalization.reference_policy must be one of {sorted(BASELINES)}")
        if self.statistics_seed < 0:
            raise ConfigError("normalization.seed must be non-negative")
        if not self.eval_roles:
            raise ConfigError("evaluation.roles must name at least one role")
        for role in self.eval_roles:
            if role not in ROLES:
                raise ConfigError(f"unknown role {role!r} in evaluation.roles")
            if role == "test" and not self.include_test:
                raise ConfigError("evaluation.roles names 'test' without evaluation.include_test "
                                  "= true; the held-out period is never evaluated by default")

    def to_json(self) -> dict[str, Any]:
        """Return the configuration as plain values with its schema tag."""
        return {"schema": TRAINING_CONFIG_SCHEMA, **asdict(self)}


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str], section: str) -> None:
    """Raise ``ConfigError`` when ``mapping`` holds a key outside ``allowed``."""
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {unknown}; allowed: {list(allowed)}")


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    """Return ``mapping[key]`` or raise ``ConfigError`` naming ``section.key``."""
    if key not in mapping:
        raise ConfigError(f"missing required key {section}.{key}")
    return mapping[key]


def _ints(values: Any, key: str) -> tuple[int, ...]:
    """Coerce a list of integers, rejecting bools and floats."""
    if not isinstance(values, list):
        raise ConfigError(f"{key} must be a list")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{key} must hold integers, got {v!r}")
        out.append(int(v))
    return tuple(out)


def training_config_from_mapping(mapping: Mapping[str, Any]) -> TrainingConfig:
    """Build a ``TrainingConfig`` from a parsed TOML mapping, strictly.

    Raises:
        ConfigError: Wrong schema, unknown or missing keys, or an invalid value.
    """
    _reject_unknown(mapping, ("schema", "experiment_config", "policy_name", "training",
                              "normalization", "evaluation"), "top level")
    if mapping.get("schema") != TRAINING_CONFIG_SCHEMA:
        raise ConfigError(f"schema {mapping.get('schema')!r} is not {TRAINING_CONFIG_SCHEMA!r}")
    training = _require(mapping, "training", "top level")
    _reject_unknown(training, ("role", "seeds", "selection", "reward_unit"), "training")
    normalization = _require(mapping, "normalization", "top level")
    _reject_unknown(normalization, ("reference_policy", "seed", "statistics_file"), "normalization")
    evaluation = _require(mapping, "evaluation", "top level")
    _reject_unknown(evaluation, ("roles", "seeds", "include_test"), "evaluation")
    roles = _require(evaluation, "roles", "evaluation")
    if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
        raise ConfigError("evaluation.roles must be a list of strings")
    statistics_file = normalization.get("statistics_file")
    return TrainingConfig(
        experiment_config=str(_require(mapping, "experiment_config", "top level")),
        policy_name=str(_require(mapping, "policy_name", "top level")),
        train_role=str(_require(training, "role", "training")),
        train_seeds=_ints(_require(training, "seeds", "training"), "training.seeds"),
        selection=str(training.get("selection", "random")),
        reward_unit=str(training.get("reward_unit", "bps")),
        statistics_policy=str(_require(normalization, "reference_policy", "normalization")),
        statistics_seed=int(_require(normalization, "seed", "normalization")),
        statistics_file=None if statistics_file is None else str(statistics_file),
        eval_roles=tuple(roles),
        eval_seeds=_ints(_require(evaluation, "seeds", "evaluation"), "evaluation.seeds"),
        include_test=bool(evaluation.get("include_test", False)),
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Parse a training/evaluation TOML file.

    Raises:
        ConfigError: Invalid TOML or ``training_config_from_mapping`` fails.
        FileNotFoundError: No such file.
    """
    try:
        mapping = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return training_config_from_mapping(mapping)


def training_statistics(
    config: ExperimentConfig, splits: Splits, training: TrainingConfig
) -> ObservationStatistics:
    """Fit the frozen statistics the training configuration names, on training draws only.

    The draws are ``config.n_episodes("train")`` episodes of
    ``training.statistics_seed`` under the configuration's target fractions;
    the reference baseline replays them. ``observation_statistics`` refuses
    any non-training draw.
    """
    draws = sample_episodes(
        splits.record, "train", config.n_episodes("train"), training.statistics_seed,
        config.rho_choices_bps,
    )
    policy = training.statistics_policy
    return observation_statistics(splits, draws, config.template, BASELINES[policy], policy)


# --- evaluation manifests ---------------------------------------------------------------------


def evaluation_manifest(run_dir: str | Path, include_test: bool = False) -> dict[str, Any]:
    """Return the fixed evaluation episode set of a run directory (``EVALUATION_MANIFEST_SCHEMA``).

    The episodes are exactly the identifiers the run's manifest lists per
    role and seed (what every baseline replayed), without the test role
    unless ``include_test``. ``digest`` hashes the split record, the data
    digests, the template and the episode list, so a checkpoint provenance
    can name the set.

    Raises:
        ValueError: The manifest cannot be read under this version.
    """
    manifest = load_manifest(run_dir)
    episodes = {
        role: by_seed for role, by_seed in manifest["episodes"].items()
        if include_test or role != "test"
    }
    data = [{k: d.get(k) for k in ("source", "date", "sha256", "n_bytes", "seed", "market")}
            for d in manifest["data"]]
    body = {
        "source_run": str(run_dir),
        "experiment_name": manifest["name"],
        "label": manifest["label"],
        "simulator_version": manifest["simulator"]["version"],
        "template": manifest["config"]["template"],
        "rho_choices_bps": manifest["config"]["rho_choices_bps"],
        "split": manifest["split"],
        "data": data,
        "episodes": episodes,
        "n_episodes": sum(len(ids) for by_seed in episodes.values() for ids in by_seed.values()),
    }
    return {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "digest": canonical_sha256(
            {k: body[k] for k in ("simulator_version", "template", "split", "data", "episodes")}
        ),
        **body,
    }


def write_evaluation_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write an evaluation manifest atomically."""
    return write_json_atomic(Path(path), dict(manifest))


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate an evaluation manifest.

    Raises:
        ValueError: Wrong schema, another simulator version, or a digest that
            does not match the content (the file was edited).
    """
    manifest: dict[str, Any] = json.loads(Path(path).read_text())
    if manifest.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise ValueError(f"{path} has schema {manifest.get('schema')!r}; expected "
                         f"{EVALUATION_MANIFEST_SCHEMA!r}")
    if manifest.get("simulator_version") != SIMULATOR_VERSION:
        raise ValueError(f"{path} lists simulator {manifest.get('simulator_version')!r}; this "
                         f"code evaluates under {SIMULATOR_VERSION!r}")
    expected = canonical_sha256(
        {k: manifest[k] for k in ("simulator_version", "template", "split", "data", "episodes")}
    )
    if manifest.get("digest") != expected:
        raise ValueError(f"{path}: digest {manifest.get('digest')!r} does not match the content")
    return manifest


# --- external policies -------------------------------------------------------------------------

# A callable over the learner's observation vector returning the action (a scalar or a
# shape-(1,) array); what a trained policy exposes once its weights are loaded.
VectorPolicy = Callable[[np.ndarray], Any]


def _action_value(value: Any) -> float:
    """Return the action as a Python float in ``[0, 1]`` without clipping.

    Raises:
        TypeError: Not a number or a numeric array.
        ValueError: Wrong shape, non-finite, or outside ``[0, 1]``.
    """
    if isinstance(value, bool):
        raise TypeError("action must be a number or a shape-(1,) array, got bool")
    if isinstance(value, int | float | np.floating | np.integer):
        action = float(value)
    else:
        array = np.asarray(value)
        if array.shape not in ((1,), ()):
            raise ValueError(f"action must be a scalar or shape (1,), got shape {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
            raise TypeError(f"action must be numeric, got dtype {array.dtype}")
        action = float(array.reshape(-1)[0])
    if not math.isfinite(action) or not 0.0 <= action <= 1.0:
        raise ValueError(f"policy returned {action!r}; actions must be finite fractions in [0, 1] "
                         "and are never clipped here")
    return action


def vector_policy(
    fn: VectorPolicy, statistics: ObservationStatistics | None = None
) -> Policy:
    """Adapt a callable over the float32 observation vector to the runner's ``Policy``.

    The vector handed to ``fn`` is what ``ExecutionEnv`` would return:
    ``simulator.observe`` standardised with ``statistics`` when given, cast to
    float32. The result is validated by ``_action_value``: a value outside
    ``[0, 1]`` or non-finite raises instead of being clipped, so a policy
    that needs clipping must clip itself, exactly as it would before
    ``env.step``.
    """
    if statistics is not None and tuple(statistics.fields) != OBSERVATION_FIELDS:
        raise ValueError("statistics must cover exactly OBSERVATION_FIELDS, in order")

    def policy(observation: Observation) -> float:
        vector = observation.vector
        if statistics is not None:
            vector = normalize(vector, statistics)
        return _action_value(fn(vector.astype(np.float32)))

    return policy


# --- evaluation into a run directory ----------------------------------------------------------


def evaluate_policy(
    run_dir: str | Path,
    policy: Policy,
    policy_name: str,
    *,
    roles: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    include_test: bool = False,
    statistics: ObservationStatistics | None = None,
    provenance: Mapping[str, Any] | None = None,
    splits: Splits | None = None,
) -> list[Path]:
    """Replay ``policy`` on the episodes a run directory lists and register the records.

    One record per ``(role, seed)`` is written as
    ``records/<policy_name>-<role>-seed<seed>.json`` with the baseline
    record schema, and the run's ``manifest.json`` gains one ``runs`` entry
    per record plus an ``evaluations`` entry carrying ``provenance`` (a
    checkpoint provenance, typically) and the statistics used. The episode
    identifiers of every record are checked against the manifest, so the
    policy is paired with the baselines on identical episodes.

    Args:
        run_dir: Directory written by ``experiment.run_experiment``.
        policy: Any ``Policy`` (``vector_policy`` adapts a learner).
        policy_name: Record name; not a baseline name, not already present.
        roles: Roles to evaluate; the manifest's evaluated roles by default,
            minus ``"test"`` unless ``include_test``.
        seeds: Seeds to evaluate; the manifest's seeds by default.
        include_test: Allow the held-out role.
        statistics: Frozen statistics the policy used; stored in the record.
        provenance: Stored verbatim in the manifest's ``evaluations`` entry.
        splits: Already rebuilt splits (saves the data load).

    Returns:
        The written record paths.

    Raises:
        ValueError: A bad or colliding policy name, an unknown role or seed,
            the test role without ``include_test``, data whose split record
            differs from the manifest's, or episode identifiers that differ.
    """
    if not _POLICY_NAME.match(policy_name) or policy_name in BASELINES:
        raise ValueError(f"policy_name {policy_name!r} must match {_POLICY_NAME.pattern} and "
                         "must not be a baseline name")
    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    if any(entry["policy"] == policy_name for entry in manifest["runs"]):
        raise ValueError(f"policy {policy_name!r} already has records in {run_path}")
    config = config_from_mapping(config_mapping_from_manifest(manifest["config"]))
    evaluated = [r for r in manifest["roles_evaluated"] if include_test or r != "test"]
    chosen_roles = list(evaluated if roles is None else roles)
    for role in chosen_roles:
        if role == "test" and not include_test:
            raise ValueError("the test role needs include_test=True")
        if role not in manifest["episodes"]:
            raise ValueError(f"role {role!r} has no episodes in the run")
    chosen_seeds = list(config.seeds if seeds is None else seeds)
    for seed in chosen_seeds:
        if seed not in config.seeds:
            raise ValueError(f"seed {seed} is not one of the run's seeds {list(config.seeds)}")
    if splits is None:
        splits = rebuild_splits(config)
    if json.loads(json.dumps(splits.record.to_json())) != manifest["split"]:
        raise ValueError("the rebuilt split record differs from the manifest's; the data changed")

    written: list[Path] = []
    entries: list[dict[str, Any]] = []
    for role in chosen_roles:
        for seed in chosen_seeds:
            listed = [e["id"] for e in manifest["episodes"][role][str(seed)]]
            started = time.perf_counter()
            record, _ = run_policy(
                splits, role, policy, policy_name, config.template, len(listed), seed,
                config.rho_choices_bps,
                observation_statistics=None if statistics is None else statistics.to_json(),
            )
            seconds = time.perf_counter() - started
            ids = [f"{role}/s{seed}/i{e['draw']['index']}" for e in record["episodes"]]
            if ids != listed:
                raise ValueError(f"episode identifiers of {role} seed {seed} differ from the "
                                 "manifest's; the policy was not evaluated on the same episodes")
            record["episode_ids"] = ids
            record["held_out"] = role == "test"
            record["evaluated_externally"] = True
            file_name = f"{policy_name}-{role}-seed{seed}.json"
            written.append(write_json_atomic(run_path / RECORDS_SUBDIR / file_name, record))
            entries.append({
                "policy": policy_name, "role": role, "seed": seed, "held_out": role == "test",
                "n_episodes": len(ids), "record_file": f"{RECORDS_SUBDIR}/{file_name}",
                "seconds": seconds, "summary": record["summary"],
            })
    manifest["runs"] = list(manifest["runs"]) + entries
    manifest.setdefault("evaluations", []).append({
        "policy": policy_name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "roles": chosen_roles,
        "seeds": chosen_seeds,
        "record_files": [e["record_file"] for e in entries],
        "observation_statistics": None if statistics is None else statistics.to_json(),
        "provenance": None if provenance is None else dict(provenance),
    })
    write_json_atomic(run_path / MANIFEST_NAME, manifest)
    return written


# --- checkpoint provenance -------------------------------------------------------------------


def checkpoint_provenance(
    config: ExperimentConfig,
    splits: Splits,
    data_provenance: Sequence[Mapping[str, Any]],
    seed: int,
    statistics: ObservationStatistics | None,
    *,
    reward_unit: str = "bps",
    evaluation_digest: str | None = None,
    training: TrainingConfig | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return what a checkpoint must record to be evaluated later.

    The result carries ``CHECKPOINT_PROVENANCE_SCHEMA``.

    Model-agnostic: nothing about the architecture or the weights. Everything
    the evaluation depends on is either embedded (configuration, statistics,
    conventions) or hashed (split record, data digests, evaluation episode
    set), so a later run can check it got the same world.

    Args:
        config: The experiment configuration the trainer used.
        splits: Its splits (the record is hashed).
        data_provenance: The ``data`` list of the experiment manifest.
        seed: The trainer's seed.
        statistics: Frozen statistics the policy observes through, or
            ``None`` for raw observations.
        reward_unit: Scalar reward unit the trainer optimised.
        evaluation_digest: ``evaluation_manifest(...)["digest"]`` of the set
            the checkpoint is judged on, if fixed already.
        training: The training configuration, if one was used.
        extra: Free-form trainer fields (framework, step count, ...), stored
            verbatim; never read by the evaluation.

    Raises:
        ValueError: Unknown reward unit or statistics over other fields.
    """
    if reward_unit not in REWARD_CONVENTION["scalar_units"]:
        raise ValueError(f"reward_unit must be one of {REWARD_CONVENTION['scalar_units']}")
    if statistics is not None and tuple(statistics.fields) != OBSERVATION_FIELDS:
        raise ValueError("statistics must cover exactly OBSERVATION_FIELDS, in order")
    # Everything embedded is JSON-normalised (tuples become lists) so the in-memory record
    # equals its file round trip and the hashes are taken over exactly what is stored.
    config_json = json.loads(json.dumps(config.to_json()))
    split_json = json.loads(json.dumps(splits.record.to_json()))
    digests = [d.get("sha256") or f"seed={d.get('seed')}" for d in data_provenance]
    statistics_json = (
        None if statistics is None else json.loads(json.dumps(statistics.to_json()))
    )
    template = config.template
    provenance = {
        "schema": CHECKPOINT_PROVENANCE_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulator_version": SIMULATOR_VERSION,
        "code_revision": code_revision(),
        "config": config_json,
        "config_sha256": canonical_sha256(config_json),
        "split_sha256": canonical_sha256(split_json),
        "data_digests": digests,
        "data_sha256": canonical_sha256(digests),
        "seed": int(seed),
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_statistics": statistics_json,
        "observation_statistics_sha256": (
            None if statistics_json is None else canonical_sha256(statistics_json)
        ),
        "action_convention": dict(ACTION_CONVENTION),
        "reward_convention": {
            **REWARD_CONVENTION,
            "unit": reward_unit,
            "terminal_rule": template.terminal_rule,
            "terminal_surcharge_bps": template.terminal_surcharge_bps,
            "residual_mu": template.residual_mu,
            "risk_lambda": template.risk_lambda,
        },
        "evaluation_digest": evaluation_digest,
        "training_config": None if training is None else training.to_json(),
        "extra": None if extra is None else dict(extra),
    }
    normalised: dict[str, Any] = json.loads(json.dumps(provenance))
    return normalised


def validate_checkpoint_provenance(provenance: Mapping[str, Any]) -> None:
    """Check a provenance record is complete, internally consistent and for this simulator.

    Raises:
        ValueError: Wrong schema or simulator version, a missing key, a hash
            that does not match the embedded content, statistics that fail
            ``statistics_from_json``, conventions other than the current
            ones, or a reward unit that is not supported.
    """
    if provenance.get("schema") != CHECKPOINT_PROVENANCE_SCHEMA:
        raise ValueError(f"provenance schema {provenance.get('schema')!r} is not "
                         f"{CHECKPOINT_PROVENANCE_SCHEMA!r}")
    if provenance.get("simulator_version") != SIMULATOR_VERSION:
        raise ValueError(f"checkpoint was trained under simulator "
                         f"{provenance.get('simulator_version')!r}; this code runs "
                         f"{SIMULATOR_VERSION!r} and does not evaluate it")
    required = ("config", "config_sha256", "split_sha256", "data_digests", "data_sha256", "seed",
                "observation_fields", "observation_statistics", "action_convention",
                "reward_convention", "evaluation_digest", "training_config", "extra")
    missing = [k for k in required if k not in provenance]
    if missing:
        raise ValueError(f"provenance is missing {missing}")
    if canonical_sha256(provenance["config"]) != provenance["config_sha256"]:
        raise ValueError("config_sha256 does not match the embedded configuration")
    if canonical_sha256(provenance["data_digests"]) != provenance["data_sha256"]:
        raise ValueError("data_sha256 does not match data_digests")
    if list(provenance["observation_fields"]) != list(OBSERVATION_FIELDS):
        raise ValueError("observation_fields differ from OBSERVATION_FIELDS")
    statistics = provenance["observation_statistics"]
    if statistics is not None:
        statistics_from_json(statistics)
        if canonical_sha256(statistics) != provenance.get("observation_statistics_sha256"):
            raise ValueError("observation_statistics_sha256 does not match the statistics")
        if statistics.get("schema") != STATISTICS_SCHEMA:  # pragma: no cover - checked above
            raise ValueError("statistics schema mismatch")
    if provenance["action_convention"] != ACTION_CONVENTION:
        raise ValueError("action_convention differs from the current environment contract")
    reward = provenance["reward_convention"]
    if reward.get("unit") not in REWARD_CONVENTION["scalar_units"]:
        raise ValueError(f"reward unit {reward.get('unit')!r} is not supported")
    for key, value in REWARD_CONVENTION.items():
        if key == "scalar_units":
            if list(reward.get(key, ())) != list(value):
                raise ValueError("reward_convention.scalar_units differs from the current contract")
        elif reward.get(key) != value:
            raise ValueError(f"reward_convention.{key} differs from the current contract")
    # The embedded configuration must still parse under this code's schema.
    config_from_mapping(config_mapping_from_manifest(provenance["config"]))
    if not isinstance(provenance["seed"], int) or isinstance(provenance["seed"], bool):
        raise ValueError("seed must be an integer")
