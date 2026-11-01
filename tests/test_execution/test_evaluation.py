"""Tests for tapeforge.execution.evaluation and the versioned statistics files: strict training
configuration, fixed evaluation manifests, the external-callable interface writing baseline-
comparable records, and the model-agnostic checkpoint provenance.

The end-to-end cases run the synthetic experiment of ``test_execution.configs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tapeforge.execution.baselines import BASELINES, twap
from tapeforge.execution.evaluation import (
    CHECKPOINT_PROVENANCE_SCHEMA,
    EVALUATION_MANIFEST_SCHEMA,
    TRAINING_CONFIG_SCHEMA,
    canonical_sha256,
    checkpoint_provenance,
    evaluate_policy,
    evaluation_manifest,
    load_evaluation_manifest,
    load_training_config,
    training_config_from_mapping,
    training_statistics,
    validate_checkpoint_provenance,
    vector_policy,
    write_evaluation_manifest,
)
from tapeforge.execution.experiment import (
    ConfigError,
    acquire_days,
    load_manifest,
    rebuild_splits,
    run_experiment,
)
from tapeforge.execution.normalization import (
    STATISTICS_SCHEMA,
    load_statistics,
    observation_statistics,
    save_statistics,
    statistics_from_json,
)
from tapeforge.execution.report import build_report, episode_rows, load_records
from tapeforge.execution.runner import observation_for, run_episode, summarize_episode
from tapeforge.execution.sampling import sample_episodes
from tapeforge.execution.simulator import OBSERVATION_FIELDS, SIMULATOR_VERSION
from test_execution.configs import synthetic_config


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the synthetic experiment once for the module."""
    return run_experiment(synthetic_config(), output_dir=tmp_path_factory.mktemp("runs"))


def _training_mapping(**overrides: object) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schema": TRAINING_CONFIG_SCHEMA,
        "experiment_config": "experiments/configs/smoke-synthetic.toml",
        "policy_name": "ppo-smoke",
        "training": {"role": "train", "seeds": [1], "selection": "random", "reward_unit": "bps"},
        "normalization": {"reference_policy": "twap", "seed": 1},
        "evaluation": {"roles": ["validation"], "seeds": [1]},
    }
    mapping.update(overrides)
    return mapping


# --- training configuration --------------------------------------------------------------------


def test_training_config_is_strict(tmp_path: Path) -> None:
    config = training_config_from_mapping(_training_mapping())
    assert config.policy_name == "ppo-smoke" and config.train_role == "train"
    assert config.eval_roles == ("validation",) and not config.include_test
    assert config.to_json()["schema"] == TRAINING_CONFIG_SCHEMA
    with pytest.raises(ConfigError, match="unknown key"):
        training_config_from_mapping(_training_mapping(extra=1))
    with pytest.raises(ConfigError, match="training.role must be 'train'"):
        training_config_from_mapping(_training_mapping(training={"role": "validation",
                                                                 "seeds": [1]}))
    with pytest.raises(ConfigError, match="is a baseline name"):
        training_config_from_mapping(_training_mapping(policy_name="twap"))
    with pytest.raises(ConfigError, match="never evaluated by default"):
        training_config_from_mapping(_training_mapping(evaluation={"roles": ["test"],
                                                                   "seeds": [1]}))
    assert training_config_from_mapping(_training_mapping(evaluation={
        "roles": ["test"], "seeds": [1], "include_test": True})).include_test
    with pytest.raises(ConfigError, match="reference_policy"):
        training_config_from_mapping(_training_mapping(normalization={"reference_policy": "ppo",
                                                                      "seed": 1}))
    with pytest.raises(ConfigError, match="distinct non-negative"):
        training_config_from_mapping(_training_mapping(training={"role": "train",
                                                                 "seeds": [1, 1]}))
    with pytest.raises(ConfigError, match="schema"):
        training_config_from_mapping(_training_mapping(schema="x/9"))
    path = tmp_path / "train.toml"
    path.write_text(
        f'schema = "{TRAINING_CONFIG_SCHEMA}"\nexperiment_config = "e.toml"\n'
        'policy_name = "p"\n[training]\nrole = "train"\nseeds = [2]\n'
        '[normalization]\nreference_policy = "twap"\nseed = 2\n'
        '[evaluation]\nroles = ["train", "validation"]\nseeds = [2]\n'
    )
    assert load_training_config(path).eval_roles == ("train", "validation")
    path.write_text("[[[")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_training_config(path)


def test_training_statistics_are_fitted_on_training_draws_only() -> None:
    config = synthetic_config()
    splits = rebuild_splits(config)
    training = training_config_from_mapping(_training_mapping())
    stats = training_statistics(config, splits, training)
    draws = sample_episodes(splits.record, "train", config.n_episodes("train"), 1,
                            config.rho_choices_bps)
    expected = observation_statistics(splits, draws, config.template, twap, "twap")
    assert stats == expected and stats.policy == "twap" and stats.seed == 1
    assert all(d.role == "train" for d in draws)


# --- statistics files ---------------------------------------------------------------------------


def test_statistics_round_trip_through_a_versioned_file(tmp_path: Path) -> None:
    config = synthetic_config()
    splits = rebuild_splits(config)
    draws = sample_episodes(splits.record, "train", 4, 1, config.rho_choices_bps)
    stats = observation_statistics(splits, draws, config.template, twap, "twap")
    payload = stats.to_json()
    assert payload["schema"] == STATISTICS_SCHEMA
    assert payload["simulator_version"] == SIMULATOR_VERSION
    path = save_statistics(stats, tmp_path / "stats" / "statistics.json")
    assert load_statistics(path) == stats
    assert json.loads(path.read_text()) == json.loads(json.dumps(payload))
    assert not list(path.parent.glob("*.tmp"))
    for key, value, match in (
        ("schema", "x", "schema"),
        ("simulator_version", "tapeforge.execution.simulator/1", "fitted under simulator"),
        ("fields", list(OBSERVATION_FIELDS[:-1]) + ["other"], "not OBSERVATION_FIELDS"),
        ("mean", payload["mean"][:-1], "one value per field"),
        ("std", [-1.0] * len(OBSERVATION_FIELDS), "non-negative"),
        ("mean", [float("nan")] * len(OBSERVATION_FIELDS), "finite"),
        ("n_episodes", 0, "positive"),
    ):
        broken = dict(payload)
        broken[key] = value
        with pytest.raises(ValueError, match=match):
            statistics_from_json(broken)
    with pytest.raises(ValueError, match="missing"):
        statistics_from_json({k: v for k, v in payload.items() if k != "policy"})
    path.write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_statistics(path)
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_statistics(path)


# --- evaluation manifests -----------------------------------------------------------------------


def test_evaluation_manifest_lists_the_baselines_episodes_and_is_tamper_evident(
    run_dir: Path, tmp_path: Path
) -> None:
    manifest = evaluation_manifest(run_dir)
    assert manifest["schema"] == EVALUATION_MANIFEST_SCHEMA
    assert manifest["simulator_version"] == SIMULATOR_VERSION
    experiment = load_manifest(run_dir)
    assert manifest["episodes"] == experiment["episodes"]   # No test role in this run.
    assert manifest["n_episodes"] == sum(
        len(ids) for by_seed in experiment["episodes"].values() for ids in by_seed.values()
    )
    assert manifest["split"] == experiment["split"]
    path = write_evaluation_manifest(manifest, tmp_path / "evaluation.json")
    assert load_evaluation_manifest(path)["digest"] == manifest["digest"]
    edited = dict(manifest)
    edited["episodes"] = {"train": edited["episodes"]["train"]}
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="digest"):
        load_evaluation_manifest(path)
    edited = dict(manifest)
    edited["simulator_version"] = "tapeforge.execution.simulator/1"
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="lists simulator"):
        load_evaluation_manifest(path)
    edited = dict(manifest)
    edited["schema"] = "x"
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="schema"):
        load_evaluation_manifest(path)


def test_held_out_episodes_enter_the_evaluation_manifest_only_on_request(tmp_path: Path) -> None:
    dates = synthetic_config().split
    config = synthetic_config(split={"train": [dates.train[0]], "validation": [dates.train[1]],
                                     "test": [dates.validation[0]]})
    run = run_experiment(config, include_test=True, output_dir=tmp_path)
    assert "test" not in evaluation_manifest(run)["episodes"]
    assert "test" in evaluation_manifest(run, include_test=True)["episodes"]
    with_test = evaluation_manifest(run, include_test=True)
    assert evaluation_manifest(run)["digest"] != with_test["digest"]


# --- external policies ---------------------------------------------------------------------------


def test_vector_policy_feeds_the_learner_vector_and_never_clips(run_dir: Path) -> None:
    config = synthetic_config()
    splits = rebuild_splits(config)
    draws = sample_episodes(splits.record, "train", 3, 1, config.rho_choices_bps)
    stats = observation_statistics(splits, draws, config.template, twap, "twap")
    seen: list[np.ndarray] = []

    def constant(vector: np.ndarray) -> np.ndarray:
        seen.append(vector)
        return np.array([0.25], dtype=np.float32)

    policy = vector_policy(constant, stats)
    entry = splits.episode("train", 0, 1, 500, config.template)
    spec, market = entry
    trajectory = run_episode(spec, market, policy)
    assert len(seen) == len(trajectory.results) and all(v.dtype == np.float32 for v in seen)
    assert all(v.shape == (len(OBSERVATION_FIELDS),) for v in seen)
    # The vector is the standardised observation, not the raw one.
    first = observation_for(trajectory.states[0], market, spec).vector
    mean, std = np.array(stats.mean), np.array(stats.std)
    np.testing.assert_allclose(seen[0], ((first - mean) / np.where(std > 0, std, 1.0)),
                               rtol=1e-5, atol=1e-6)
    assert trajectory.actions[0] == pytest.approx(0.25)
    for bad in (1.5, -0.1, float("nan"), np.array([0.1, 0.2]), True, "x"):
        with pytest.raises((ValueError, TypeError)):
            run_episode(spec, market, vector_policy(lambda _v, b=bad: b))
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS"):
        vector_policy(constant, stats.__class__(fields=("a",), mean=(0.0,), std=(1.0,),
                                                n_observations=1, n_episodes=1, policy="twap",
                                                seed=1))


def test_evaluate_policy_writes_baseline_comparable_records_on_identical_episodes(
    run_dir: Path,
) -> None:
    """A TWAP re-implemented as an external vector policy reproduces the twap records exactly."""
    config = synthetic_config()
    splits = rebuild_splits(config)

    def twap_from_vector(vector: np.ndarray) -> float:
        # remaining fraction and time fraction are fields 0 and 1; the exact quantity is not
        # visible to a vector policy, so this reproduces TWAP only when it divides evenly.
        raise AssertionError("not used")

    del twap_from_vector
    stats = observation_statistics(
        splits, sample_episodes(splits.record, "train", 4, 1, config.rho_choices_bps),
        config.template, twap, "twap",
    )
    provenance = checkpoint_provenance(config, splits, acquire_days(config)[1], 7, stats)
    written = evaluate_policy(
        run_dir, BASELINES["twap"], "twap-external", statistics=stats, provenance=provenance,
        splits=splits,
    )
    manifest = load_manifest(run_dir)
    assert len(written) == 2 * len(config.seeds)   # train and validation, every seed.
    assert all(p.exists() for p in written)
    entries = [e for e in manifest["runs"] if e["policy"] == "twap-external"]
    assert len(entries) == len(written)
    for entry in entries:
        external = json.loads((run_dir / entry["record_file"]).read_text())
        baseline = json.loads(
            (run_dir / f"records/twap-{entry['role']}-seed{entry['seed']}.json").read_text()
        )
        assert external["episode_ids"] == baseline["episode_ids"]
        assert external["episodes"] == baseline["episodes"]   # Same draws, same accounting.
        assert external["summary"] == baseline["summary"] == entry["summary"]
        assert external["evaluated_externally"]
        assert external["observation_statistics"] == json.loads(json.dumps(stats.to_json()))
    (evaluation,) = manifest["evaluations"]
    assert evaluation["policy"] == "twap-external"
    assert evaluation["provenance"] == json.loads(json.dumps(provenance))
    # The report pairs the external policy against the reference on identical identifiers.
    report = build_report(run_dir, reference="twap")
    paired = [p for p in report["paired"] if p["policy"] == "twap-external"]
    assert paired and all(p["mean_diff_bps"] == 0.0 and p["fraction_equal"] == 1.0 for p in paired)
    assert any(g["policy"] == "twap-external" for g in report["per_policy"])
    rows = episode_rows(load_records(run_dir))
    assert {r.policy for r in rows} >= {"twap", "twap-external"}
    with pytest.raises(ValueError, match="already has records"):
        evaluate_policy(run_dir, BASELINES["twap"], "twap-external", splits=splits)
    with pytest.raises(ValueError, match="must not be a baseline name"):
        evaluate_policy(run_dir, BASELINES["twap"], "immediate", splits=splits)
    with pytest.raises(ValueError, match="needs include_test"):
        evaluate_policy(run_dir, BASELINES["twap"], "other", roles=["test"], splits=splits)
    with pytest.raises(ValueError, match="not one of the run's seeds"):
        evaluate_policy(run_dir, BASELINES["twap"], "other", seeds=[99], splits=splits)


def test_evaluate_policy_matches_a_direct_replay_of_a_vector_policy(tmp_path: Path) -> None:
    config = synthetic_config(policies={"names": ["twap"]}, episodes={
        "seeds": [3], "episodes_per_day": 2, "evaluate_roles": ["validation"],
    })
    run = run_experiment(config, output_dir=tmp_path)
    splits = rebuild_splits(config)
    half = vector_policy(lambda vector: np.float32(0.5 * float(vector[0] > 0)))
    evaluate_policy(run, half, "half-remaining", roles=["validation"], splits=splits)
    record = json.loads((run / "records" / "half-remaining-validation-seed3.json").read_text())
    for episode in record["episodes"]:
        draw = episode["draw"]
        spec, market = splits.episode("validation", draw["position"], draw["side"],
                                      draw["rho_bps"], config.template)
        assert summarize_episode(run_episode(spec, market, half)).to_json() == episode["summary"]


# --- checkpoint provenance ----------------------------------------------------------------------


def test_checkpoint_provenance_is_complete_consistent_and_validated(run_dir: Path) -> None:
    config = synthetic_config()
    splits = rebuild_splits(config)
    _, data = acquire_days(config)
    stats = observation_statistics(
        splits, sample_episodes(splits.record, "train", 4, 1, config.rho_choices_bps),
        config.template, twap, "twap",
    )
    training = training_config_from_mapping(_training_mapping())
    digest = evaluation_manifest(run_dir)["digest"]
    provenance = checkpoint_provenance(
        config, splits, data, 11, stats, evaluation_digest=digest, training=training,
        extra={"framework": "none", "updates": 0},
    )
    validate_checkpoint_provenance(provenance)
    assert provenance["schema"] == CHECKPOINT_PROVENANCE_SCHEMA
    assert provenance["simulator_version"] == SIMULATOR_VERSION and provenance["seed"] == 11
    assert provenance["config_sha256"] == canonical_sha256(config.to_json())
    assert provenance == json.loads(json.dumps(provenance))   # Already JSON-normalised.
    assert provenance["split_sha256"] == canonical_sha256(
        json.loads(json.dumps(splits.record.to_json()))
    )
    assert provenance["data_digests"] == [f"seed={d['seed']}" for d in data]
    assert provenance["observation_statistics"] == json.loads(json.dumps(stats.to_json()))
    assert provenance["reward_convention"]["terminal_rule"] == "terminal_settlement"
    assert provenance["reward_convention"]["terminal_surcharge_bps"] == 25
    assert provenance["reward_convention"]["unit"] == "bps"
    assert provenance["action_convention"]["out_of_range"].startswith("raises")
    assert provenance["evaluation_digest"] == digest
    assert provenance["training_config"] == json.loads(json.dumps(training.to_json()))
    json.dumps(provenance)
    raw = checkpoint_provenance(config, splits, data, 0, None, reward_unit="ticks")
    validate_checkpoint_provenance(raw)
    assert raw["observation_statistics"] is None and raw["observation_statistics_sha256"] is None
    for mutate, match in (
        (lambda p: p.update(schema="x"), "schema"),
        (lambda p: p.update(simulator_version="tapeforge.execution.simulator/1"), "trained under"),
        (lambda p: p.pop("seed"), "missing"),
        (lambda p: p.update(config_sha256="0" * 64), "config_sha256"),
        (lambda p: p.update(data_sha256="0" * 64), "data_sha256"),
        (lambda p: p.update(observation_fields=["a"]), "observation_fields"),
        (lambda p: p.update(observation_statistics_sha256="0" * 64), "statistics_sha256"),
        (lambda p: p["action_convention"].update(out_of_range="clipped"), "action_convention"),
        (lambda p: p["reward_convention"].update(unit="usd"), "not supported"),
        (lambda p: p["reward_convention"].update(per_step="other"), "reward_convention.per_step"),
        (lambda p: p.update(seed="7"), "seed must be"),
    ):
        broken = json.loads(json.dumps(provenance))
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            validate_checkpoint_provenance(broken)
    with pytest.raises(ValueError, match="reward_unit"):
        checkpoint_provenance(config, splits, data, 0, None, reward_unit="usd")
