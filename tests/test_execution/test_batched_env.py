"""Tests of the batched vector environment against the CPU training environment.

The CPU side is ``PolicyActionEnv`` over ``ExecutionEnv``; the batched side
runs on CPU tensors here. These tests do not run on CUDA; simulator parity
on CUDA is covered by the gpu-marked tests in test_batched.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tapeforge.execution.research.experiment import ConfigError
from tapeforge.execution.rl.ppo_config import load_ppo_config
from test_execution.ppo_configs import write_configs
from test_execution.tapes import worksheet_spec, worksheet_tape

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from tapeforge.execution import batched, parity  # noqa: E402
from tapeforge.execution.rl.batched_env import BatchedExecutionVecEnv  # noqa: E402
from tapeforge.execution.rl.gym_env import ExecutionEnv, ListPool  # noqa: E402
from tapeforge.execution.rl.normalization import ObservationStatistics  # noqa: E402
from tapeforge.execution.rl.ppo import train  # noqa: E402
from tapeforge.execution.rl.ppo_env import PolicyActionEnv  # noqa: E402
from tapeforge.execution.rl.ppo_eval import selected_checkpoint  # noqa: E402
from tapeforge.execution.simulator import OBSERVATION_FIELDS  # noqa: E402


def _statistics() -> ObservationStatistics:
    n = len(OBSERVATION_FIELDS)
    return ObservationStatistics(
        fields=OBSERVATION_FIELDS, mean=tuple(0.05 * i for i in range(n)),
        std=tuple(0.0 if i == 2 else 0.25 + i for i in range(n)), n_observations=10,
        n_episodes=1, policy="twap", seed=7,
    )


@pytest.mark.parametrize("statistics", [None, _statistics()], ids=["raw", "normalised"])
def test_the_batched_environment_matches_the_cpu_training_environment_step_by_step(
    statistics: ObservationStatistics | None
) -> None:
    episodes = parity.random_episodes(96, 21)
    batch = batched.EpisodeBatch.from_episodes(episodes)
    vec = BatchedExecutionVecEnv(batch, len(episodes), seed=0, statistics=statistics)
    observations = vec.reset_to(range(len(episodes)))
    references = []
    for k, (spec, market) in enumerate(episodes):
        env = PolicyActionEnv(ExecutionEnv(ListPool([(f"e{k}", spec, market)]),
                                           statistics=statistics))
        first, _ = env.reset(seed=0)
        assert first.tobytes() == observations[k].tobytes()
        references.append(env)

    rng = np.random.default_rng(3)
    running = [True] * len(episodes)
    compared = ended_terminated = ended_truncated = 0
    for _ in range(max(spec.horizon_buckets for spec, _ in episodes)):
        raw = rng.uniform(-1.3, 1.3, size=(len(episodes), 1)).astype(np.float32)
        observations, rewards, dones, infos = vec.step(raw)
        for k, env in enumerate(references):
            if not running[k]:
                continue
            observation, reward, terminated, truncated, info = env.step(raw[k])
            compared += 1
            assert bool(dones[k]) == (terminated or truncated)
            assert rewards[k] == pytest.approx(np.float32(reward), rel=1e-6, abs=1e-9)
            if not dones[k]:
                assert observation.tobytes() == observations[k].tobytes()
                assert infos[k] == {}
                continue
            running[k] = False
            ended_terminated += terminated
            ended_truncated += truncated
            assert infos[k]["TimeLimit.truncated"] is truncated
            assert infos[k]["terminal_observation"].tobytes() == observation.tobytes()
            expected, got = info["episode_summary"], infos[k]["episode_summary"]
            assert got["episode_id"] == f"bank/{k}"
            assert (got["terminated"], got["truncated"]) == (terminated, truncated)
            assert got["n_steps"] == expected["n_steps"]
            for key in ("reward_sum", "shortfall_bps", "terminal_bps", "risk_bps",
                        "filled_fraction", "settlement_fraction"):
                assert got[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-12), key
    assert not any(running) and compared > 400
    assert ended_terminated > 20 and ended_truncated > 5


def test_ended_environments_restart_on_a_drawn_episode_with_reference_reset_values() -> None:
    episodes = parity.random_episodes(24, 4)
    batch = batched.EpisodeBatch.from_episodes(episodes)
    vec = BatchedExecutionVecEnv(batch, 8, seed=5)
    vec.reset()
    restarts = 0
    for _ in range(40):
        before = vec.state.episode.tolist()
        observations, _, dones, _ = vec.step(np.ones((8, 1), dtype=np.float32))
        after = vec.state.episode.tolist()
        for i in range(8):
            if not dones[i]:
                assert after[i] == before[i]
                continue
            restarts += 1
            spec, market = episodes[after[i]]
            reference = PolicyActionEnv(ExecutionEnv(ListPool([("e", spec, market)])))
            expected, _ = reference.reset(seed=0)
            assert observations[i].tobytes() == expected.tobytes()
            assert int(vec.state.step[i]) == 0 and not bool(vec.state.ended[i])
    assert restarts > 20
    # The same seed replays the same episode sequence; another seed does not.
    again, other = (BatchedExecutionVecEnv(batch, 8, seed=s) for s in (5, 6))
    first = BatchedExecutionVecEnv(batch, 8, seed=5)
    first.reset(), again.reset(), other.reset()
    assert first.state.episode.tolist() == again.state.episode.tolist()
    assert first.state.episode.tolist() != other.state.episode.tolist()


def test_invalid_use_of_the_batched_environment_raises() -> None:
    batch = batched.EpisodeBatch.from_episodes([(worksheet_spec(1), worksheet_tape())])
    vec = BatchedExecutionVecEnv(batch, 2, seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        vec.step(np.zeros((2, 1), dtype=np.float32))
    vec.reset()
    with pytest.raises(ValueError, match="finite"):
        vec.step(np.array([[0.0], [np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="one action per environment"):
        vec.step(np.zeros((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="bank row"):
        vec.reset_to([0, 1])
    with pytest.raises(NotImplementedError, match="collection_snapshot"):
        vec.env_method("snapshot")
    with pytest.raises(ValueError, match="reward_unit"):
        BatchedExecutionVecEnv(batch, 2, seed=0, reward_unit="dollars")
    assert vec.get_attr("render_mode") == [None, None]


def test_a_restored_collection_snapshot_continues_the_same_trajectory() -> None:
    episodes = parity.random_episodes(24, 9)
    batch = batched.EpisodeBatch.from_episodes(episodes)
    ids = [f"e{k}" for k in range(len(episodes))]
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, size=(60, 8, 1)).astype(np.float32)
    reference = BatchedExecutionVecEnv(batch, 8, seed=3, episode_ids=ids)
    reference.reset()
    for step in actions[:25]:
        reference.step(step)
    snapshot = reference.collection_snapshot()
    expected = [reference.step(step) for step in actions[25:]]

    # A fresh environment, other seed: only the snapshot decides how it continues.
    restored = BatchedExecutionVecEnv(batch, 8, seed=99, episode_ids=ids)
    restored.restore_collection(snapshot)
    for step, (observation, reward, done, info) in zip(actions[25:], expected, strict=True):
        got = restored.step(step)
        assert got[0].tobytes() == observation.tobytes()
        assert got[1].tobytes() == reward.tobytes()
        assert np.array_equal(got[2], done)
        assert [i.get("episode_summary") for i in got[3]] == [
            i.get("episode_summary") for i in info
        ]

    with pytest.raises(ValueError, match="n_envs"):
        BatchedExecutionVecEnv(batch, 4, seed=3, episode_ids=ids).restore_collection(snapshot)
    with pytest.raises(ValueError, match="bank"):
        BatchedExecutionVecEnv(batch, 8, seed=3).restore_collection(snapshot)


def test_the_batched_backend_resumes_bit_identically_on_cpu_tensors(tmp_path: Path) -> None:
    config = load_ppo_config(write_configs(tmp_path, backend="batched"))
    uninterrupted = train(config, model_seed=0, output_dir=tmp_path / "a")
    lines = (uninterrupted / "rollouts.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["env_steps"] for r in rows] == [96, 192, 288, 384]
    assert all(np.isfinite(r["approx_kl"]) and r["episodes_finished"] > 0 for r in rows)
    assert json.loads((uninterrupted / "summary.json").read_text())["backend"] == "batched"

    partial = train(config, model_seed=0, total_steps=192, output_dir=tmp_path / "b")
    checkpoint = partial / "checkpoints" / "v00002"
    stored = torch.load(checkpoint / "collection.pt", weights_only=False)
    assert stored["environments"] is None and stored["batched_environments"] is not None
    resumed = train(config, model_seed=0, resume_from=checkpoint)
    assert resumed == partial
    final = [PPO.load(run / "checkpoints" / "v00004" / "model.zip", device="cpu")
             for run in (uninterrupted, resumed)]
    for name, tensor in final[0].policy.state_dict().items():
        assert torch.equal(tensor, final[1].policy.state_dict()[name]), name

    def comparable(run: Path) -> list[dict[str, object]]:
        lines = (run / "rollouts.jsonl").read_text().splitlines()
        return [{k: v for k, v in json.loads(line).items() if not k.endswith("_seconds")}
                for line in lines]

    assert comparable(resumed) == comparable(uninterrupted)
    assert selected_checkpoint(resumed).name == selected_checkpoint(uninterrupted).name


def test_a_batched_checkpoint_without_a_collection_snapshot_is_not_resumed(
    tmp_path: Path
) -> None:
    config = load_ppo_config(write_configs(tmp_path, backend="batched"))
    run = train(config, model_seed=0, total_steps=192)
    checkpoint = run / "checkpoints" / "v00002"
    stored = torch.load(checkpoint / "collection.pt", weights_only=False)
    torch.save({**stored, "batched_environments": None}, checkpoint / "collection.pt")
    with pytest.raises(ConfigError, match="not resumed"):
        train(config, model_seed=0, resume_from=checkpoint)

def test_a_snapshot_from_another_device_is_refused_before_anything_changes() -> None:
    episodes = parity.random_episodes(12, 2)
    batch = batched.EpisodeBatch.from_episodes(episodes)
    ids = [f"e{k}" for k in range(len(episodes))]
    vec = BatchedExecutionVecEnv(batch, 4, seed=1, episode_ids=ids)
    vec.reset()
    snapshot = vec.collection_snapshot()
    assert snapshot["device"] == "cpu"
    vec.step(np.zeros((4, 1), dtype=np.float32))
    before = vec.collection_snapshot()
    # A CUDA generator state is 16 bytes; snapshots written before the device
    # entry existed are recognised by that size alone.
    for foreign in ({**snapshot, "device": "cuda"},
                    {k: v for k, v in snapshot.items() if k != "device"}
                    | {"generator": torch.zeros(16, dtype=torch.uint8)}):
        with pytest.raises(ValueError, match="device"):
            vec.restore_collection(foreign)
        after = vec.collection_snapshot()
        for name, tensor in before["state"].items():
            assert torch.equal(tensor, after["state"][name]), name
        assert torch.equal(before["generator"], after["generator"])


def test_cuda_generator_states_for_other_devices_are_refused(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from tapeforge.execution.rl import ppo

    states = ppo._generator_states()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ConfigError, match="2 devices but 1"):
        ppo._set_generator_states((*states[:3], [torch.zeros(16, dtype=torch.uint8)] * 2))


def _rollout(vec: BatchedExecutionVecEnv, steps: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    trace = [vec.reset()]
    for _ in range(steps):
        raw = rng.uniform(-1.3, 1.3, size=(vec.num_envs, 1)).astype(np.float32)
        observations, rewards, dones, infos = vec.step(raw)
        trace.append((observations, rewards, dones, json.dumps(infos, default=repr)))
    return trace


def _assert_compiled_equals_eager(device: str, n_envs: int) -> None:
    episodes = parity.random_episodes(64, 5)
    runs = []
    for compile_step in (False, True):
        batch = batched.EpisodeBatch.from_episodes(episodes, device)
        vec = BatchedExecutionVecEnv(batch, n_envs, seed=3, statistics=_statistics(),
                                     compile_step=compile_step)
        runs.append((_rollout(vec, 120, 11), vec.collection_snapshot()))
    (eager, eager_snapshot), (compiled, compiled_snapshot) = runs
    assert eager[0].tobytes() == compiled[0].tobytes()
    for (o1, r1, d1, i1), (o2, r2, d2, i2) in zip(eager[1:], compiled[1:], strict=True):
        assert o1.tobytes() == o2.tobytes()
        assert r1.tobytes() == r2.tobytes()
        assert d1.tobytes() == d2.tobytes()
        assert i1 == i2
    for key in ("reward_sum", "cost_ticks", "risk_ticks", "steps", "last_observation",
                "generator"):
        assert torch.equal(eager_snapshot[key], compiled_snapshot[key]), key
    for name, tensor in eager_snapshot["state"].items():
        assert torch.equal(tensor, compiled_snapshot["state"][name]), name


def test_the_compiled_transition_equals_the_eager_one_on_cpu_tensors() -> None:
    """Observations, rewards, infos, accumulators, state and generator agree bit for bit."""
    _assert_compiled_equals_eager("cpu", 16)


@pytest.mark.gpu
def test_the_compiled_transition_equals_the_eager_one_on_cuda(gpu_ready: None) -> None:
    _assert_compiled_equals_eager("cuda", 16)


def test_compile_simulator_is_an_optional_batched_only_setting(tmp_path: Path) -> None:
    from dataclasses import replace

    ppo_path = write_configs(tmp_path, backend="batched")
    config = load_ppo_config(ppo_path)
    assert config.compile_simulator is False and config.evaluation_device is None
    assert "compile_simulator" not in config.to_json()
    assert "evaluation_device" not in config.to_json()
    compiled = replace(config, compile_simulator=True, evaluation_device="cpu")
    assert compiled.to_json()["compile_simulator"] is True
    assert compiled.to_json()["evaluation_device"] == "cpu"
    with pytest.raises(ConfigError, match="batched backend only"):
        replace(config, backend="serial", compile_simulator=True)
    with pytest.raises(ConfigError, match="boolean"):
        replace(config, compile_simulator=1)  # type: ignore[arg-type]
    text = Path(ppo_path).read_text().replace("[rollout]", "[rollout]\ncompile_simulator = true")
    Path(ppo_path).write_text(text)
    assert load_ppo_config(ppo_path).compile_simulator is True


def _policy_actions(device: str, evaluation_device: str | None) -> list[float]:
    """Actions of the deterministic evaluation policy of an untrained model over fixed episodes."""
    from tapeforge.execution.rl.ppo_eval import checkpoint_policy
    from tapeforge.execution.runner import run_episode

    episodes = parity.random_episodes(8, 9)
    vec = BatchedExecutionVecEnv(batched.EpisodeBatch.from_episodes(episodes), 4, seed=0)
    model = PPO("MlpPolicy", vec, n_steps=8, batch_size=8, device=device, seed=0)
    policy = checkpoint_policy(model, _statistics(), evaluation_device)
    actions: list[float] = []
    for spec, market in episodes:
        actions.extend(run_episode(spec, market, policy).actions)
    return actions


def test_an_evaluation_copy_on_the_same_device_answers_like_the_model() -> None:
    assert _policy_actions("cpu", "cpu") == _policy_actions("cpu", None)


@pytest.mark.gpu
def test_a_cpu_evaluation_copy_of_a_cuda_policy_agrees_to_float32_rounding(
    gpu_ready: None
) -> None:
    on_cuda = _policy_actions("cuda", None)
    on_cpu = _policy_actions("cuda", "cpu")
    assert on_cpu == pytest.approx(on_cuda, abs=1e-6)
