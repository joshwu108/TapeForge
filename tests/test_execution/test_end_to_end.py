"""End-to-end synthetic regression: configuration -> experiment -> report -> replay -> fixed
evaluation set -> frozen statistics -> external policy evaluation -> paired report -> environment
parity, on one small synthetic world under the primary terminal rule.

Every stage reads what the previous one wrote on disk, so a change that
breaks a schema, an identifier, a hash or an accounting identity anywhere in
the chain fails here before it reaches a training run. Nothing is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tapeforge.execution.baselines import BASELINES
from tapeforge.execution.evaluation import (
    checkpoint_provenance,
    evaluate_policy,
    evaluation_manifest,
    load_evaluation_manifest,
    training_config_from_mapping,
    training_statistics,
    validate_checkpoint_provenance,
    vector_policy,
    write_evaluation_manifest,
)
from tapeforge.execution.experiment import (
    acquire_days,
    load_config,
    load_manifest,
    rebuild_splits,
    run_experiment,
)
from tapeforge.execution.normalization import load_statistics, save_statistics
from tapeforge.execution.replay import replay_episode, verify_against_record
from tapeforge.execution.report import build_report, episode_rows, load_records, write_report
from tapeforge.execution.runner import observation_for, run_episode, summarize_episode
from tapeforge.execution.simulator import OBSERVATION_FIELDS, SIMULATOR_VERSION
from test_execution.configs import synthetic_mapping

CONFIG_TOML = """
schema = "tapeforge.execution.experiment-config/2"
name = "e2e"
label = "smoke"
description = "end-to-end regression world"

[data]
source = "synthetic"
market = "synthetic"

[data.synthetic]
n_buckets = 300
profile = "uniform"
days = [
  { date = "2024-01-01", seed = 1 },
  { date = "2024-01-02", seed = 2 },
  { date = "2024-01-03", seed = 3 },
  { date = "2024-01-04", seed = 4 },
]

[split]
train = ["2024-01-01", "2024-01-02"]
validation = ["2024-01-03"]
test = ["2024-01-04"]

[episodes]
seeds = [1, 2]
episodes_per_day = 3
rho_choices_bps = [200, 500, 1000]
evaluate_roles = ["train", "validation"]

[policies]
names = ["twap", "immediate", "cap_aware_twap", "zero_action"]

[template]
horizon_buckets = 12
arrival_lookback = 4
risk_lambda = 3
terminal_rule = "terminal_settlement"
terminal_surcharge_bps = 25
"""


def test_end_to_end_synthetic_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from tapeforge.execution.env_check import check_environment
    from tapeforge.execution.gym_env import ExecutionEnv, SplitPool

    config_path = tmp_path / "e2e.toml"
    config_path.write_text(CONFIG_TOML)
    config = load_config(config_path)
    assert config.template.settles_at_horizon and config.template.risk_lambda == 3

    # 1. Experiment: identical draws for every policy, held-out test excluded by default.
    run_dir = run_experiment(config, output_dir=tmp_path / "runs", command=["e2e"])
    manifest = load_manifest(run_dir)
    assert manifest["simulator"]["version"] == SIMULATOR_VERSION
    assert manifest["roles_evaluated"] == ["train", "validation"]
    assert "test" not in manifest["episodes"]
    ids = {role: {seed: [e["id"] for e in listed] for seed, listed in by_seed.items()}
           for role, by_seed in manifest["episodes"].items()}
    for entry in manifest["runs"]:
        record = json.loads((run_dir / entry["record_file"]).read_text())
        assert record["episode_ids"] == ids[entry["role"]][str(entry["seed"])]
        for episode in record["episodes"]:
            summary = episode["summary"]
            assert summary["terminal_rule"] == "terminal_settlement"
            if not summary["truncated"]:
                assert summary["completed"]
                assert summary["filled_qty_ticks"] + summary["settled_qty_ticks"] == (
                    summary["target_qty_ticks"]
                )
                assert summary["reward_sum_ticks"] == -(
                    summary["shortfall_ticks"] + summary["risk_sum_ticks"]
                )
                assert summary["shortfall_ticks"] == (
                    summary["fill_shortfall_ticks"] + summary["terminal_component_ticks"]
                )

    # 2. Report: outcome classes, settlement reliance and pairing on identical identifiers.
    report = build_report(run_dir)
    paths = write_report(report, run_dir / "report", episode_rows(load_records(run_dir)))
    assert (run_dir / "report" / "report.rst").exists() and "episodes" in paths
    zero = [g for g in report["per_policy"] if g["policy"] == "zero_action"]
    assert all(g["diagnostic"] and g["final_completion_rate"] == 1.0
               and g["completion_by_fills_rate"] == 0.0 and g["settlement_label"] == "heavy"
               for g in zero)
    for paired in report["paired"]:
        assert paired["reference"] == "twap" and paired["n_missing"] == 0

    # 3. Replay: every recorded episode of every policy replays to its record exactly.
    splits = rebuild_splits(config)
    for entry in manifest["runs"]:
        for episode_id in ids[entry["role"]][str(entry["seed"])][:2]:
            replay = replay_episode(run_dir, episode_id, entry["policy"], splits)
            assert verify_against_record(replay, run_dir) == []

    # 4. Fixed evaluation set and frozen training statistics, saved and reloaded.
    evaluation = evaluation_manifest(run_dir)
    eval_path = write_evaluation_manifest(evaluation, tmp_path / "evaluation.json")
    assert load_evaluation_manifest(eval_path)["episodes"] == manifest["episodes"]
    training = training_config_from_mapping({
        "schema": "tapeforge.execution.training-config/1",
        "experiment_config": str(config_path), "policy_name": "e2e-policy",
        "training": {"role": "train", "seeds": [1]},
        "normalization": {"reference_policy": "twap", "seed": 1,
                          "statistics_file": str(tmp_path / "statistics.json")},
        "evaluation": {"roles": ["train", "validation"], "seeds": [1, 2]},
    })
    statistics = training_statistics(config, splits, training)
    stats_path = save_statistics(statistics, training.statistics_file or "")
    loaded = load_statistics(stats_path)
    assert loaded == statistics and tuple(loaded.fields) == OBSERVATION_FIELDS

    # 5. External policy through the vector interface, with provenance, paired in the report.
    def quarter(vector: np.ndarray) -> np.ndarray:
        assert vector.dtype == np.float32 and vector.shape == (len(OBSERVATION_FIELDS),)
        return np.array([0.25], dtype=np.float32)

    _, data = acquire_days(config)
    provenance = checkpoint_provenance(
        config, splits, data, seed=1, statistics=loaded, evaluation_digest=evaluation["digest"],
        training=training,
    )
    validate_checkpoint_provenance(provenance)
    policy = vector_policy(quarter, loaded)
    written = evaluate_policy(run_dir, policy, training.policy_name, roles=training.eval_roles,
                              seeds=training.eval_seeds, statistics=loaded,
                              provenance=provenance, splits=splits)
    assert len(written) == 4
    after = build_report(run_dir)
    mine = {g["role"]: g for g in after["per_policy"] if g["policy"] == training.policy_name}
    expected_n = {role: config.n_episodes(role) * len(config.seeds) for role in mine}
    assert expected_n == {"train": 12, "validation": 6}
    assert {role: g["n_episodes"] for role, g in mine.items()} == expected_n
    paired = {p["role"]: p for p in after["paired"] if p["policy"] == training.policy_name}
    assert {role: p["n_pairs"] + p["n_excluded_truncated"] for role, p in paired.items()} == (
        expected_n
    )
    record_path = run_dir / "records" / f"{training.policy_name}-train-seed1.json"
    record = json.loads(record_path.read_text())
    assert record["episode_ids"] == ids["train"]["1"]

    # 6. Environment parity: the adapter reproduces the direct replay of the same policy on the
    #    same evaluation episodes, exact integers in info, scaled scalar rewards, flags.
    pool = SplitPool(splits, "validation", config.template, 2, len(ids["validation"]["2"]),
                     config.rho_choices_bps)
    env = ExecutionEnv(pool, statistics=loaded)
    for index in range(len(pool)):
        entry = pool[index]
        assert entry.episode_id == ids["validation"]["2"][index]
        trajectory = run_episode(entry.spec, entry.market, policy)
        summary = summarize_episode(trajectory)
        obs, info = env.reset(options={"episode": index})
        rewards: list[float] = []
        done = False
        while not done:
            action = quarter(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        assert info["state"]["filled_qty_ticks"] == summary.filled_qty_ticks
        assert info["state"]["settled_qty_ticks"] == summary.settled_qty_ticks
        assert info["step_result"]["terminated"] == summary.terminated
        assert info["step_result"]["truncated"] == summary.truncated
        notional = summary.arrival_price_ticks * summary.target_qty_ticks
        assert sum(rewards) == pytest.approx(summary.reward_sum_ticks * 10_000 / notional)
        assert isinstance(info["reward_ticks"], int)
    # The same baselines through direct replay and through the environment agree too.
    for name in ("twap", "zero_action"):
        entry = pool[0]
        direct = summarize_episode(run_episode(entry.spec, entry.market, BASELINES[name]))
        env.reset(options={"episode": 0})
        done = False
        total = 0
        while not done:
            action = BASELINES[name](observation_for(env.state, entry.market, entry.spec))
            _, _, terminated, truncated, info = env.step(np.array([action]))
            total += info["reward_ticks"]
            done = terminated or truncated
        assert total == direct.reward_sum_ticks

    # 7. The checker command passes on this configuration with the frozen statistics.
    check = check_environment(config_path, "validation", n_episodes=3, seed=5,
                              statistics_path=stats_path, run_gym_checker=False)
    assert check.n_episodes == 3 and check.n_terminated + check.n_truncated == 3

    # 8. Nothing fitted on validation data: the statistics equal a fresh training-only fit.
    assert training_statistics(config, splits, training) == loaded
    assert synthetic_mapping()["schema"] == "tapeforge.execution.experiment-config/2"
