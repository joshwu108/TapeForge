"""Tests of the optional PPO trainer: configuration, the action map, the training-side
environment against the CPU reference, training, checkpoints, resuming and evaluation.

No test asserts anything about historical profitability. Learning is checked
on a constructed tape where the right timing is obvious by arithmetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tapeforge.execution import EpisodeSpec, run_episode, run_experiment, twap, zero_action
from tapeforge.execution.research.experiment import ConfigError, load_config
from tapeforge.execution.research.report import build_report
from tapeforge.execution.rl.ppo_config import (
    PPO_CONFIG_SCHEMA,
    load_ppo_config,
    ppo_config_from_mapping,
)
from tapeforge.execution.runner import summarize_episode
from tapeforge.execution.simulator import reset, step
from test_execution.ppo_configs import write_configs
from test_execution.tapes import slice_from_rows, worksheet_spec, worksheet_tape

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from tapeforge.execution.rl.evaluation import vector_policy  # noqa: E402
from tapeforge.execution.rl.gym_env import ExecutionEnv, ListPool, reward_from_ticks  # noqa: E402
from tapeforge.execution.rl.normalization import ObservationStatistics  # noqa: E402
from tapeforge.execution.rl.ppo import train  # noqa: E402
from tapeforge.execution.rl.ppo_env import PolicyActionEnv, policy_action_to_env  # noqa: E402
from tapeforge.execution.rl.ppo_eval import (  # noqa: E402
    checkpoint_policy,
    evaluate_checkpoint,
    load_checkpoint,
    selected_checkpoint,
)
from tapeforge.execution.simulator import OBSERVATION_FIELDS  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent.parent


# --- configuration ----------------------------------------------------------------------------


def _mapping(tmp_path: Path) -> dict[str, object]:
    return tomllib.loads(write_configs(tmp_path).read_text())


def test_the_ppo_configuration_is_strict_and_has_no_silent_defaults(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path)
    config = ppo_config_from_mapping(mapping)
    assert config.gamma == 1.0 and config.rollout_steps == 96
    assert config.to_json()["schema"] == PPO_CONFIG_SCHEMA
    assert load_ppo_config(tmp_path / "ppo.toml") == config

    for section, key in (("update", "gamma"), ("update", "clip_range"), ("rollout", "n_steps"),
                         ("network", "hidden_sizes"), ("schedule", "total_steps"),
                         ("optimizer", "learning_rate")):
        broken = _mapping(tmp_path)
        del broken[section][key]  # type: ignore[attr-defined]
        with pytest.raises(ConfigError, match=key):
            ppo_config_from_mapping(broken)
    broken = _mapping(tmp_path)
    broken["update"]["discount"] = 0.99  # type: ignore[index]
    with pytest.raises(ConfigError, match="unknown key"):
        ppo_config_from_mapping(broken)


def test_the_tracked_pilot_configuration_keeps_the_objective_and_the_seeds_apart() -> None:
    from tapeforge.execution.rl.evaluation import load_training_config

    config = load_ppo_config("experiments/configs/pilot-futures-btcusdt-2023-12-ppo.toml")
    training = load_training_config(config.training_config)
    experiment = load_config(training.experiment_config)
    assert config.gamma == 1.0                       # undiscounted finite-horizon cost
    assert len(config.model_seeds) >= 3
    assert training.train_role == "train" and not training.include_test
    assert training.eval_roles == ("validation",)
    # Evaluation episodes are the baselines'; training pools use other sampling seeds.
    assert set(training.eval_seeds) <= set(experiment.seeds)
    assert not set(training.train_seeds) & set(training.eval_seeds)
    assert training.statistics_seed not in training.eval_seeds


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("update", "gamma", 1.5, "gamma"),
        ("update", "gamma", 0.0, "gamma"),
        ("update", "batch_size", 36, "must divide"),
        ("schedule", "total_steps", 100, "multiple"),
        ("rollout", "backend", "threads", "backend"),
        ("schedule", "selection_metric", "test_mean_shortfall_bps", "selection_metric"),
        ("network", "hidden_sizes", [], "hidden_sizes"),
        ("update", "normalize_advantage", 1, "boolean"),
    ],
)
def test_invalid_ppo_settings_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    mapping = _mapping(tmp_path)
    mapping[section][key] = value  # type: ignore[index]
    with pytest.raises(ConfigError, match=message):
        ppo_config_from_mapping(mapping)


# --- the action map and the training-side environment -----------------------------------------


def test_the_action_map_is_bounded_monotone_and_rejects_non_finite_values() -> None:
    assert policy_action_to_env(np.float32(-1.0)) == 0.0
    assert policy_action_to_env(np.array([1.0], dtype=np.float32)) == 1.0
    assert policy_action_to_env(0.0) == 0.5
    assert policy_action_to_env(-7.5) == 0.0 and policy_action_to_env(3.0) == 1.0
    grid = np.linspace(-1.5, 1.5, 301, dtype=np.float32)
    mapped = [policy_action_to_env(a) for a in grid]
    assert all(0.0 <= m <= 1.0 for m in mapped) and mapped == sorted(mapped)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            policy_action_to_env(bad)
    with pytest.raises(ValueError, match="one value"):
        policy_action_to_env(np.zeros(2))


def _reference_env(side: int = 1, **spec: int) -> tuple[PolicyActionEnv, EpisodeSpec]:
    episode_spec = worksheet_spec(side, terminal_surcharge_bps=25, **spec)
    pool = ListPool([("reference", episode_spec, worksheet_tape())])
    return PolicyActionEnv(ExecutionEnv(pool), detailed_info=True), episode_spec


@pytest.mark.parametrize("side", [1, -1])
def test_rewards_and_the_terminal_settlement_match_the_cpu_reference(side: int) -> None:
    env, spec = _reference_env(side)
    market = worksheet_tape()
    # float32, as the learner emits them; the last decision submits nothing, forcing settlement.
    raw_actions = [np.float32(-0.6), np.float32(0.1), np.float32(-1.0)]
    env.reset(seed=0)
    state = reset(spec, market)
    for index, raw in enumerate(raw_actions):
        _, reward, terminated, truncated, info = env.step(np.array([raw], dtype=np.float32))
        state, result = step(state, policy_action_to_env(raw), market, spec)
        assert info["step_result"]["reward_ticks"] == result.reward_ticks
        assert reward == reward_from_ticks(result.reward_ticks, spec, state.arrival_price_ticks,
                                           "bps")
        assert (terminated, truncated) == (result.terminated, result.truncated)
        assert terminated == (index == len(raw_actions) - 1)
    assert result.settlement_qty_ticks > 0 and state.settled_qty_ticks > 0
    replayed = iter(raw_actions)
    reference = summarize_episode(
        run_episode(spec, market, lambda _: policy_action_to_env(next(replayed)))
    )
    summary = info["episode_summary"]
    assert summary["shortfall_bps"] == pytest.approx(reference.shortfall_bps, rel=1e-12)
    assert summary["settlement_fraction"] == reference.settlement_fraction
    assert summary["filled_fraction"] == reference.filled_fraction
    assert summary["reward_sum"] == pytest.approx(-reference.shortfall_bps, rel=1e-12)
    assert summary["terminated"] and not summary["truncated"]


def test_only_a_tape_truncation_is_reported_to_the_learner_as_a_time_limit() -> None:
    spec = worksheet_spec(1, terminal_surcharge_bps=25)
    complete = worksheet_tape()
    cut = worksheet_tape(after=())           # no terminal bucket: the settlement cannot be priced
    pool = ListPool([("complete", spec, complete), ("cut", spec, cut)])
    venv = DummyVecEnv([lambda: PolicyActionEnv(ExecutionEnv(pool, cycle=True))])
    venv.reset()
    outcomes = {}
    for _ in range(2):
        done = False
        while not done:
            _, _, dones, infos = venv.step(np.array([[-1.0]], dtype=np.float32))
            done = bool(dones[0])
        summary = infos[0]["episode_summary"]
        outcomes[summary["episode_id"]] = (infos[0]["TimeLimit.truncated"], summary)
        assert "terminal_observation" in infos[0]
    assert outcomes["complete"][0] is False and outcomes["complete"][1]["terminated"]
    assert outcomes["complete"][1]["settlement_fraction"] == 1.0
    # Truncated: nothing settled, no terminal term invented, fills-only shortfall.
    assert outcomes["cut"][0] is True and outcomes["cut"][1]["truncated"]
    assert outcomes["cut"][1]["settlement_fraction"] == 0.0
    assert outcomes["cut"][1]["terminal_bps"] == 0.0


def _statistics() -> ObservationStatistics:
    n = len(OBSERVATION_FIELDS)
    return ObservationStatistics(
        fields=OBSERVATION_FIELDS, mean=tuple(0.1 * i for i in range(n)),
        std=tuple(0.0 if i == 2 else 0.5 + i for i in range(n)), n_observations=10,
        n_episodes=1, policy="twap", seed=7,
    )


def test_training_and_evaluation_see_bit_identical_normalised_observations() -> None:
    spec, market, statistics = worksheet_spec(1), worksheet_tape(), _statistics()
    env = PolicyActionEnv(ExecutionEnv(ListPool([("e", spec, market)]), statistics=statistics))
    raw = np.array([-0.5], dtype=np.float32)
    observation, _ = env.reset(seed=0)
    seen_in_training = [observation]
    while True:
        observation, _, terminated, truncated, _ = env.step(raw)
        if terminated or truncated:
            break
        seen_in_training.append(observation)

    seen_in_evaluation: list[np.ndarray] = []

    def record(vector: np.ndarray) -> float:
        seen_in_evaluation.append(vector)
        return policy_action_to_env(raw)

    run_episode(spec, market, vector_policy(record, statistics))
    assert len(seen_in_training) == len(seen_in_evaluation) == 3
    for trained, evaluated in zip(seen_in_training, seen_in_evaluation, strict=True):
        assert trained.dtype == evaluated.dtype == np.float32
        assert trained.tobytes() == evaluated.tobytes()


def test_no_future_bucket_reaches_the_policy() -> None:
    spec = worksheet_spec(1)
    base = worksheet_tape()
    raw = np.array([-0.8], dtype=np.float32)
    for changed in range(3):
        # A different price and quantity in bucket `changed` may alter fills from that decision
        # on, but the observations of decisions 0 .. changed were taken before it existed.
        other = base.with_bucket(changed, total_quantity=40_000, vwap_ticks=12_345)
        streams = []
        for market in (base, other):
            env = PolicyActionEnv(ExecutionEnv(ListPool([("e", spec, market)])))
            observation, _ = env.reset(seed=0)
            observations = [observation]
            for _ in range(changed):
                observation, *_ = env.step(raw)
                observations.append(observation)
            streams.append(np.stack(observations))
        assert streams[0].tobytes() == streams[1].tobytes()
        assert len(streams[0]) == changed + 1


def test_the_training_environment_reports_lean_info_unless_asked_for_detail() -> None:
    spec, market = worksheet_spec(1), worksheet_tape()
    lean = PolicyActionEnv(ExecutionEnv(ListPool([("e", spec, market)])))
    detailed = PolicyActionEnv(ExecutionEnv(ListPool([("e", spec, market)])), detailed_info=True)
    raw = np.array([-1.0], dtype=np.float32)
    summaries = []
    for env in (lean, detailed):
        env.reset(seed=0)
        infos = [env.step(raw)[4] for _ in range(3)]
        summaries.append(infos[-1]["episode_summary"])
        if env is lean:
            assert infos[0] == {} and set(infos[-1]) == {"episode_summary"}
        else:
            assert {"step_result", "state", "reward_ticks"} <= set(infos[0])
    assert summaries[0] == summaries[1]


# --- learning on a constructed task --------------------------------------------------------------


def _drift_task(side: int, horizon: int = 8) -> tuple[str, EpisodeSpec, object]:
    """Return an episode whose price moves 20 ticks per bucket in the agent's favour.

    With deep liquidity, equal fees and no surcharge, every tick traded later
    is cheaper, so waiting for the terminal settlement is optimal by
    arithmetic and trading early is the worst choice.
    """
    rows = [(2_000_000, 10_000 - 20 * side * (t + 1)) for t in range(horizon + 1)]
    spec = EpisodeSpec(side=side, target_qty_ticks=100_000, horizon_buckets=horizon,
                       arrival_lookback=1)
    return f"drift-{side}", spec, slice_from_rows([(2_000_000, 10_000)], rows)


def _drift_cost(policy: object) -> float:
    return float(np.mean([
        summarize_episode(run_episode(spec, market, policy)).shortfall_bps  # type: ignore[arg-type]
        for _, spec, market in (_drift_task(1), _drift_task(-1))
    ]))


def _drift_model(seed: int = 0) -> PPO:
    set_random_seed(seed)
    torch.set_num_threads(1)
    entries = [_drift_task(1), _drift_task(-1)]

    def factory() -> PolicyActionEnv:
        pool = ListPool(entries)  # type: ignore[arg-type]
        return PolicyActionEnv(ExecutionEnv(pool, selection="random", cycle=True))

    venv = DummyVecEnv([factory] * 4)
    for index, env in enumerate(venv.envs):
        env.reset(seed=seed * 10 + index)
    return PPO(
        "MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=4, gamma=1.0, learning_rate=3e-3,
        seed=None, device="cpu",
        policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}, "log_std_init": -0.5},
    )


def test_ppo_learns_an_obvious_timing_preference_on_a_constructed_tape() -> None:
    model = _drift_model()
    before = _drift_cost(checkpoint_policy(model, None))
    parameters = [p.detach().clone() for p in model.policy.parameters()]
    model.learn(8192)

    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    assert any(not torch.equal(p, q) for p, q in zip(parameters, model.policy.parameters(),
                                                     strict=True))
    after, waiting, even = (_drift_cost(checkpoint_policy(model, None)),
                            _drift_cost(zero_action), _drift_cost(twap))
    assert waiting < even < before            # the arithmetic of the task
    assert after < even - 40.0                # clearly better than an even schedule
    assert after == pytest.approx(waiting, abs=15.0)


def test_stochastic_training_actions_are_valid_for_the_simulator() -> None:
    model = _drift_model(seed=3)
    submitted: list[float] = []
    original = ExecutionEnv.step

    def recording_step(self: ExecutionEnv, action: object) -> object:
        submitted.append(float(np.asarray(action).reshape(-1)[0]))
        return original(self, action)

    ExecutionEnv.step = recording_step  # type: ignore[method-assign]
    try:
        model.learn(512)
    finally:
        ExecutionEnv.step = original  # type: ignore[method-assign]
    assert len(submitted) == 512
    assert all(0.0 <= a <= 1.0 for a in submitted)
    # The Gaussian is wide enough that both bounds are reached through the clip.
    assert min(submitted) == 0.0 and max(submitted) == 1.0


# --- the training command, checkpoints and resuming ---------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train the tiny synthetic setup once; return the configuration root and the run."""
    root = tmp_path_factory.mktemp("ppo")
    config = load_ppo_config(write_configs(root))
    return root, train(config, model_seed=0)


def test_a_short_run_writes_finite_diagnostics_and_a_validation_selection(
    trained: tuple[Path, Path]
) -> None:
    _, run = trained
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    assert [r["policy_version"] for r in rows] == [0, 1, 2, 3]
    assert [r["env_steps"] for r in rows] == [96, 192, 288, 384]
    for row in rows:
        for key in ("policy_entropy", "approx_kl", "clip_fraction", "value_loss",
                    "explained_variance", "mean_shortfall_bps", "mean_settlement_fraction",
                    "mean_reward_sum", "collect_seconds", "update_seconds", "train_seconds"):
            assert np.isfinite(row[key]), key
        assert row["episodes_finished"] > 0 and row["episodes_truncated"] == 0
    selection = json.loads((run / "selection.json").read_text())
    assert [e["policy_version"] for e in selection["evaluations"]] == [0, 2, 4]
    scores = [e["validation_mean_shortfall_bps"] for e in selection["evaluations"]]
    assert selection["selected"]["validation_mean_shortfall_bps"] == min(scores)
    assert "test role is never consulted" in selection["rule"]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["env_steps"] == 384 and summary["policy_versions"] == 4
    assert summary["seconds"]["train"] == pytest.approx(
        summary["seconds"]["collect"] + summary["seconds"]["update"]
    )
    stored = json.loads((run / "config.json").read_text())
    assert stored["ppo"]["gamma"] == 1.0 and stored["training"]["train_role"] == "train"


def test_a_reloaded_checkpoint_predicts_identically_and_carries_its_provenance(
    trained: tuple[Path, Path]
) -> None:
    _, run = trained
    checkpoint = selected_checkpoint(run)
    assert checkpoint == selected_checkpoint(checkpoint)
    first, provenance, statistics = load_checkpoint(checkpoint)
    second, _, _ = load_checkpoint(checkpoint)
    observations = np.random.default_rng(0).normal(size=(64, len(OBSERVATION_FIELDS)))
    observations = observations.astype(np.float32)
    a, _ = first.policy.predict(observations, deterministic=True)
    b, _ = second.policy.predict(observations, deterministic=True)
    assert a.tobytes() == b.tobytes()
    with torch.no_grad():
        values = [m.policy.predict_values(torch.as_tensor(observations)) for m in (first, second)]
    assert torch.equal(values[0], values[1])
    assert first.gamma == 1.0

    assert statistics is not None
    stored = json.loads((run / "statistics.json").read_text())
    assert provenance["observation_statistics"] == stored
    extra = provenance["extra"]
    assert extra["ppo_config"]["gamma"] == 1.0 and "clip" in extra["action_map"]
    assert provenance["seed"] == 0 and provenance["training_config"]["train_seeds"] == [101, 102]

    (checkpoint / "model.zip").write_bytes((checkpoint / "model.zip").read_bytes() + b"x")
    try:
        with pytest.raises(ValueError, match="sha256"):
            load_checkpoint(checkpoint)
    finally:
        data = (checkpoint / "model.zip").read_bytes()
        (checkpoint / "model.zip").write_bytes(data[:-1])


def _final_state(run: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    model = PPO.load(run / "checkpoints" / "v00004" / "model.zip", device="cpu")
    return model.policy.state_dict(), model.policy.optimizer.state_dict()


def _comparable_rows(run: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if not k.endswith("_seconds")} for r in rows]


def test_resuming_from_a_checkpoint_continues_training_bit_identically(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    _, uninterrupted = trained
    config = load_ppo_config(write_configs(tmp_path))
    partial = train(config, model_seed=0, total_steps=192)
    assert not (partial / "checkpoints" / "v00004").exists()
    resumed = train(config, model_seed=0, resume_from=partial / "checkpoints" / "v00002")
    assert resumed == partial

    weights, optimizer = _final_state(uninterrupted)
    resumed_weights, resumed_optimizer = _final_state(resumed)
    assert weights.keys() == resumed_weights.keys()
    for name, tensor in weights.items():
        assert torch.equal(tensor, resumed_weights[name]), name
    assert str(optimizer) == str(resumed_optimizer)
    assert _comparable_rows(resumed) == _comparable_rows(uninterrupted)

    # Weights alone are not a continuation: fresh environments diverge immediately.
    with pytest.raises(ConfigError, match="configuration differs"):
        train(config, model_seed=1, resume_from=partial / "checkpoints" / "v00002")


def test_a_selected_checkpoint_is_paired_with_the_baselines_in_the_report(
    trained: tuple[Path, Path]
) -> None:
    root, run = trained
    baseline_run = run_experiment(load_config(root / "experiment.toml"))
    # Every role the baseline run evaluated, never the held-out one.
    written = evaluate_checkpoint(run, baseline_run, "ppo-s0")
    assert [p.name for p in written] == [
        "ppo-s0-train-seed1.json", "ppo-s0-validation-seed1.json",
    ]
    record = json.loads(written[1].read_text())
    assert record["observation_statistics"] == json.loads((run / "statistics.json").read_text())

    selection = json.loads((run / "selection.json").read_text())["selected"]
    assert record["summary"]["mean_shortfall_bps"] == pytest.approx(
        selection["validation_mean_shortfall_bps"], rel=1e-12
    )
    report = build_report(baseline_run, reference="twap")
    paired = [p for p in report["paired"] if p["policy"] == "ppo-s0"]
    assert [p["role"] for p in paired] == ["train", "validation"]
    assert paired[1]["n_pairs"] == 6
    with pytest.raises(ValueError, match="include_test"):
        evaluate_checkpoint(run, baseline_run, "ppo-test", roles=["test"])

    # The replay tool rebuilds a learned policy's episode from its checkpoint and checks it
    # against the stored record.
    from tapeforge.execution.research import replay

    episode = "validation/s1/i2"
    assert replay.main([str(baseline_run), "--episode", episode, "--policy", "ppo-s0",
                        "--checkpoint", str(run), "--verify"]) == 0
    assert replay.main([str(baseline_run), "--episode", episode, "--policy", "ppo-s0"]) == 1
    model, _, statistics = load_checkpoint(selected_checkpoint(run))
    replayed = replay.replay_episode(baseline_run, episode, "ppo-s0",
                                     policy_fn=checkpoint_policy(model, statistics))
    assert replay.verify_against_record(replayed, baseline_run) == []


def test_a_checkpoint_is_not_evaluated_on_a_run_with_other_settings(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    root, run = trained
    text = (root / "experiment.toml").read_text()
    other = tmp_path / "experiment.toml"
    other.write_text(text.replace("terminal_surcharge_bps = 25", "terminal_surcharge_bps = 50"))
    sensitivity_run = run_experiment(load_config(other), output_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match="terminal_surcharge_bps"):
        evaluate_checkpoint(run, sensitivity_run, "ppo-s0")
    written = evaluate_checkpoint(run, sensitivity_run, "ppo-s0", strict_settings=False)
    manifest = json.loads((sensitivity_run / "manifest.json").read_text())
    provenance = manifest["evaluations"][0]["provenance"]
    assert provenance["template_keys_differing_from_training"] == ["terminal_surcharge_bps"]
    assert provenance["split_record_differs_from_training"] is False
    assert len(written) == 2

    # A latency change re-derives the admissible starts from the same days: allowed only
    # without strict settings. Other days are refused either way.
    delayed = tmp_path / "latency.toml"
    delayed.write_text(text.replace('arrival_lookback = 4', 'arrival_lookback = 4\n'
                                    'latency_buckets = 1'))
    latency_run = run_experiment(load_config(delayed), output_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match="split record differs"):
        evaluate_checkpoint(run, latency_run, "ppo-s0")
    evaluate_checkpoint(run, latency_run, "ppo-s0", strict_settings=False)
    manifest = json.loads((latency_run / "manifest.json").read_text())
    assert manifest["evaluations"][0]["provenance"]["split_record_differs_from_training"] is True
    other_days = tmp_path / "days.toml"
    other_days.write_text(text.replace('train = ["2024-01-01", "2024-01-02"]',
                                       'train = ["2024-01-01"]')
                          .replace('validation = ["2024-01-03"]', 'validation = ["2024-01-02"]'))
    period_run = run_experiment(load_config(other_days), output_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match="data digests|split record"):
        evaluate_checkpoint(run, period_run, "ppo-s0", strict_settings=False)


def test_the_command_line_trains_several_seeds_and_evaluates_their_selections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tapeforge.execution.rl.ppo import main

    config_path = write_configs(tmp_path, total_steps=192)
    assert main(["train", str(config_path)]) == 0          # every model_seeds entry
    runs = sorted((tmp_path / "training-runs").iterdir())
    assert [r.name.split("-")[2] for r in runs] == ["seed0", "seed1"]
    weights = [PPO.load(selected_checkpoint(r) / "model.zip").policy.state_dict() for r in runs]
    assert any(not torch.equal(weights[0][k], weights[1][k]) for k in weights[0])

    baseline_run = run_experiment(load_config(tmp_path / "experiment.toml"))
    arguments = ["evaluate", str(baseline_run), f"ppo-s0={runs[0]}", f"ppo-s1={runs[1]}"]
    assert main(arguments) == 0
    assert capsys.readouterr().out.count("record:") == 4
    assert main(arguments) == 1                             # names already registered
    assert "already has records" in capsys.readouterr().err
    assert main(["evaluate", str(baseline_run), "no-equals-sign"]) == 1
    assert main(["train", str(config_path), "--resume", str(runs[0])]) == 1


def test_the_subprocess_backend_resumes_bit_identically(tmp_path: Path) -> None:
    config = load_ppo_config(write_configs(tmp_path, backend="subprocess"))
    uninterrupted = train(config, model_seed=0, output_dir=tmp_path / "a")
    partial = train(config, model_seed=0, total_steps=192, output_dir=tmp_path / "b")
    resumed = train(config, model_seed=0, resume_from=partial / "checkpoints" / "v00002")
    weights, _ = _final_state(uninterrupted)
    resumed_weights, _ = _final_state(resumed)
    for name, tensor in weights.items():
        assert torch.equal(tensor, resumed_weights[name]), name
    assert _comparable_rows(resumed) == _comparable_rows(uninterrupted)


# --- optional dependency ------------------------------------------------------------------------


def test_the_trainer_is_optional_and_names_its_extra() -> None:
    block = "import sys; sys.modules['stable_baselines3'] = None; "
    env = dict(os.environ, PYTHONPATH=str(TESTS_DIR))

    def run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-c", block + code], capture_output=True,
                              text=True, check=False, env=env)

    usable = run(
        "import tapeforge.execution.rl.gym_env, tapeforge.execution.rl.evaluation; "
        "from tapeforge.execution.rl.ppo_config import load_ppo_config; print('ok')"
    )
    assert usable.returncode == 0, usable.stderr
    for module in ("ppo", "ppo_env", "ppo_eval"):
        missing = run(f"import tapeforge.execution.rl.{module}")
        assert missing.returncode != 0
        assert "pip install -e '.[train]'" in missing.stderr
