"""Tests of the batched vector environment against the CPU training environment.

The CPU side is ``PolicyActionEnv`` over ``ExecutionEnv``; the batched side
runs on CPU tensors here. A CUDA run needs hardware and is not covered.
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
    with pytest.raises(NotImplementedError, match="exact resume"):
        vec.env_method("snapshot")
    with pytest.raises(ValueError, match="reward_unit"):
        BatchedExecutionVecEnv(batch, 2, seed=0, reward_unit="dollars")
    assert vec.get_attr("render_mode") == [None, None]


def test_the_trainer_runs_on_the_batched_backend_and_refuses_to_resume_it(tmp_path: Path) -> None:
    config = load_ppo_config(write_configs(tmp_path, backend="batched"))
    run = train(config, model_seed=0)
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    assert [r["env_steps"] for r in rows] == [96, 192, 288, 384]
    assert all(np.isfinite(r["approx_kl"]) and r["episodes_finished"] > 0 for r in rows)
    assert json.loads((run / "summary.json").read_text())["backend"] == "batched"
    checkpoint = selected_checkpoint(run)
    assert torch.load(checkpoint / "collection.pt", weights_only=False)["environments"] is None
    with pytest.raises(ConfigError, match="not resumed"):
        train(config, model_seed=0, resume_from=checkpoint)
