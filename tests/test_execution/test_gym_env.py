"""Tests for tapeforge.execution.gym_env: the Gymnasium checker, step-by-step parity with the
simulator, every baseline through the adapter, seeding and reset independence, termination and
truncation, boundary and invalid actions, delayed observations, pool exhaustion, frozen
normalisation and a random-action smoke run.

The module skips when gymnasium is not installed; ``test_optional_gymnasium``
covers the import path without it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W  # noqa: E402
from tapeforge.execution.baselines import BASELINES  # noqa: E402
from tapeforge.execution.days import synthetic_day  # noqa: E402
from tapeforge.execution.gym_env import (  # noqa: E402
    ACTION_SHAPE,
    EpisodePoolExhausted,
    ExecutionEnv,
    ListPool,
    SplitPool,
    reward_from_ticks,
)
from tapeforge.execution.normalization import observation_statistics  # noqa: E402
from tapeforge.execution.runner import (  # noqa: E402
    observation_for,
    run_episode,
    summarize_episode,
)
from tapeforge.execution.sampling import sample_episodes  # noqa: E402
from tapeforge.execution.simulator import (  # noqa: E402
    OBSERVATION_FIELDS,
    EpisodeOver,
    observe,
    reset,
    step,
)
from tapeforge.execution.spec import EpisodeSpec  # noqa: E402
from tapeforge.execution.splits import SplitDates, Splits, build_splits  # noqa: E402
from test_execution.tapes import DEEP, P0, Q_STAR, slice_from_rows, worksheet_spec, worksheet_tape  # noqa: E402

SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=6, arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02", "2024-01-03")


def _splits(template: EpisodeSpec = SPEC) -> Splits:
    days = [
        synthetic_day(1, DATES[0], 90, start_us=0),
        synthetic_day(2, DATES[1], 90, start_us=90 * W),
        synthetic_day(3, DATES[2], 70, start_us=180 * W),
    ]
    return build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],)), template)


def _pool(
    n: int = 10, template: EpisodeSpec = SPEC, role: str = "train", seed: int = 1
) -> SplitPool:
    return SplitPool(_splits(template), role, template, seed=seed, n_episodes=n)


def _worksheet_pool(**overrides: int) -> ListPool:
    spec = worksheet_spec(side=1, **overrides)
    return ListPool([("worksheet/a", spec, worksheet_tape())])


def _rollout(
    env: ExecutionEnv, policy_name: str, seed: int | None = 0, options: dict | None = None
) -> tuple[list[float], list[dict], list[np.ndarray]]:
    """Drive ``env`` with a baseline through the adapter; return rewards, infos and observations.

    The baseline's float64 action is passed unchanged: it is nudged by ulps
    so that ``floor(action * remaining)`` is the intended quantity, and a
    float32 cast can lose that last tick (see the float32 test below).
    """
    obs, info = env.reset(seed=seed, options=options)
    entry = env.pool[info["pool_index"]]
    rewards, infos, observations = [], [info], [obs]
    done = False
    while not done:
        action = BASELINES[policy_name](observation_for(env.state, entry.market, entry.spec))
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float64))
        rewards.append(reward)
        infos.append(info)
        observations.append(obs)
        done = terminated or truncated
    return rewards, infos, observations


# --- spaces and the checker ----------------------------------------------------------------


def test_spaces_are_as_documented() -> None:
    env = ExecutionEnv(_pool())
    assert env.observation_space.shape == (len(OBSERVATION_FIELDS),) == (10,)
    assert env.observation_space.dtype == np.float32 and env.action_space.dtype == np.float32
    assert env.action_space.shape == ACTION_SHAPE == (1,)
    assert float(env.action_space.low[0]) == 0.0 and float(env.action_space.high[0]) == 1.0
    assert env.observation_space.low[0] == 0.0 and env.observation_space.high[0] == 1.0
    assert env.observation_space.low[2] == -1.0 and env.observation_space.high[2] == 1.0
    assert env.simulator_version.startswith("tapeforge.execution.simulator/")


def test_passes_the_gymnasium_environment_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    env = ExecutionEnv(_pool(12), cycle=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    # The unbounded return fields make the checker warn about infinite bounds; nothing else.
    assert all("infinity" in str(w.message) for w in caught), [str(w.message) for w in caught]
    env = ExecutionEnv(_pool(12), selection="random")
    check_env(env, skip_render_check=True)


# --- parity with the simulator -------------------------------------------------------------


def test_step_by_step_parity_with_direct_simulator_calls() -> None:
    pool = _pool(4)
    env = ExecutionEnv(pool)
    obs, info = env.reset(seed=0)
    entry = pool[0]
    state = reset(entry.spec, entry.market)
    np.testing.assert_array_equal(obs, observe(state, entry.market, entry.spec).astype(np.float32))
    assert info["arrival_price_ticks"] == state.arrival_price_ticks
    rng = np.random.default_rng(5)
    done = False
    while not done:
        action = float(rng.uniform(0.0, 1.0))
        obs, reward, terminated, truncated, info = env.step(action)
        state, result = step(state, action, entry.market, entry.spec)
        assert info["step_result"]["fill_qty_ticks"] == result.fill_qty_ticks
        assert info["step_result"]["fee_ticks"] == result.fee_ticks
        assert info["state"]["cash_ticks"] == state.cash_ticks
        assert info["state"]["inventory_ticks"] == state.inventory_ticks
        assert info["reward_ticks"] == result.reward_ticks
        assert reward == result.reward_ticks * 10_000 / (
            state.arrival_price_ticks * entry.spec.target_qty_ticks
        )
        assert (terminated, truncated) == (result.terminated, result.truncated)
        done = terminated or truncated
        if not done:
            np.testing.assert_array_equal(
                obs, observe(state, entry.market, entry.spec).astype(np.float32)
            )
    assert env.state == state


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_replays_through_the_adapter_with_the_same_outcome(name: str) -> None:
    pool = _pool(6)
    env = ExecutionEnv(pool)
    for index in range(len(pool)):
        rewards, infos, _ = _rollout(env, name, seed=None, options={"episode": index})
        entry = pool[index]
        trajectory = run_episode(entry.spec, entry.market, BASELINES[name])
        summary = summarize_episode(trajectory)
        assert [i["step_result"]["fill_qty_ticks"] for i in infos[1:]] == [
            r.fill_qty_ticks for r in trajectory.results
        ]
        assert [i["step_result"]["fee_ticks"] for i in infos[1:]] == [
            r.fee_ticks for r in trajectory.results
        ]
        assert [i["reward_ticks"] for i in infos[1:]] == [
            r.reward_ticks for r in trajectory.results
        ]
        assert infos[-1]["step_result"]["terminated"] == trajectory.last_result.terminated
        assert infos[-1]["step_result"]["truncated"] == trajectory.last_result.truncated
        assert infos[-1]["state"]["cash_ticks"] == trajectory.final_state.cash_ticks
        assert infos[-1]["state"]["fees_ticks"] == summary.fees_ticks
        assert sum(rewards) == pytest.approx(
            summary.reward_sum_ticks * 10_000
            / (summary.arrival_price_ticks * summary.target_qty_ticks)
        )


def test_a_float32_action_is_floored_from_its_float32_value_not_silently_repaired() -> None:
    """The action space is float32 for the learner; an exact quantity needs the float64 action.

    ``action_for_quantity`` nudges a float64 ratio by ulps so that
    ``floor(action * remaining)`` is the requested quantity. Rounding that
    action to float32 can lose the nudge and submit one tick less. The
    adapter passes whatever value it receives to the simulator unchanged, so
    the effect is visible in ``submitted_qty_ticks`` rather than hidden.
    """
    from tapeforge.execution.simulator import action_for_quantity

    remaining, quantity = 1_642_381, 164_238
    exact = action_for_quantity(quantity, remaining)
    assert int(np.floor(exact * remaining)) == quantity
    rounded = float(np.float32(exact))
    env = ExecutionEnv(ListPool([("big", EpisodeSpec(side=1, target_qty_ticks=remaining,
                                                      horizon_buckets=3, arrival_lookback=1),
                                   worksheet_tape(totals=(DEEP * 10,) * 3))]))
    env.reset()
    _, _, _, _, info = env.step(np.array([exact], dtype=np.float64))
    assert info["step_result"]["submitted_qty_ticks"] == quantity
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([exact], dtype=np.float32))
    assert info["step_result"]["submitted_qty_ticks"] == int(np.floor(rounded * remaining))
    assert info["step_result"]["submitted_qty_ticks"] in (quantity - 1, quantity)


def test_reward_conversion_is_bps_of_the_arrival_notional_or_raw_ticks() -> None:
    spec = worksheet_spec(side=1)
    assert reward_from_ticks(-3_800_800, spec, P0, "bps") == pytest.approx(-12.669333)
    assert reward_from_ticks(-3_800_800, spec, P0, "ticks") == -3_800_800.0
    with pytest.raises(ValueError, match="reward_unit"):
        reward_from_ticks(0, spec, P0, "usd")
    env = ExecutionEnv(_worksheet_pool(), reward_unit="ticks")
    rewards, infos, _ = _rollout(env, "twap")
    # Worksheet buy table: cost = side x (notional - P_0 x fill) + fee per step.
    assert rewards == [-(200_000 + 1_000_200), -(500_000 + 1_000_500), -(100_000 + 1_000_100)]
    assert sum(rewards) == -3_800_800 and infos[-1]["step_result"]["terminated"]


# --- seeding, reset independence, selection ----------------------------------------------


def test_same_seed_replays_the_same_sequence_and_resets_are_independent() -> None:
    env = ExecutionEnv(_pool(5))
    first = [_rollout(env, "twap", seed=7 if i == 0 else None)[1] for i in range(3)]
    second = [_rollout(env, "twap", seed=7 if i == 0 else None)[1] for i in range(3)]
    assert [f[0]["episode_id"] for f in first] == ["train/s1/i0", "train/s1/i1", "train/s1/i2"]
    assert [f[0]["episode_id"] for f in first] == [s[0]["episode_id"] for s in second]
    assert [f[-1]["state"] for f in first] == [s[-1]["state"] for s in second]
    # An episode's outcome does not depend on what ran before it.
    fresh = ExecutionEnv(_pool(5))
    _, infos, _ = _rollout(fresh, "twap", seed=None, options={"episode": 2})
    assert infos[-1]["state"] == first[2][-1]["state"]
    random_env = ExecutionEnv(_pool(5), selection="random")
    ids_a = [random_env.reset(seed=3 if i == 0 else None)[1]["episode_id"] for i in range(6)]
    ids_b = [random_env.reset(seed=3 if i == 0 else None)[1]["episode_id"] for i in range(6)]
    assert ids_a == ids_b and len(set(ids_a)) > 1


def test_explicit_selection_by_index_or_identifier_and_pool_exhaustion() -> None:
    env = ExecutionEnv(_pool(3))
    assert env.reset(options={"episode": "train/s1/i2"})[1]["pool_index"] == 2
    assert env.reset(options={"episode": 1})[1]["episode_id"] == "train/s1/i1"
    with pytest.raises(KeyError):
        env.reset(options={"episode": "train/s1/i9"})
    with pytest.raises(IndexError):
        env.reset(options={"episode": 3})
    with pytest.raises(TypeError):
        env.reset(options={"episode": 1.5})
    sequential = ExecutionEnv(_pool(3))
    for _ in range(3):
        sequential.reset()
    with pytest.raises(EpisodePoolExhausted, match="all 3 episodes"):
        sequential.reset()
    assert sequential.reset(seed=0)[1]["pool_index"] == 0   # A seed restarts the pool.
    cycling = ExecutionEnv(_pool(3), cycle=True)
    assert [cycling.reset()[1]["pool_index"] for _ in range(4)] == [0, 1, 2, 0]
    with pytest.raises(ValueError, match="selection"):
        ExecutionEnv(_pool(3), selection="shuffled")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        ListPool([])


# --- termination, truncation, actions, delays ------------------------------------------


def test_termination_at_completion_and_at_the_horizon() -> None:
    env = ExecutionEnv(_worksheet_pool())
    rewards, infos, observations = _rollout(env, "immediate")
    assert len(rewards) == 2 and infos[-1]["step_result"]["terminated"]
    assert not infos[-1]["step_result"]["truncated"] and infos[-1]["remaining_qty_ticks"] == 0
    assert infos[-1]["terminal_observation_available"] and observations[-1][0] == 0.0
    with pytest.raises(EpisodeOver, match="call reset"):
        env.step(0.5)
    env.reset(seed=0)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(0.0)   # Never trade.
    assert terminated and not truncated and info["remaining_qty_ticks"] == Q_STAR
    assert info["step_result"]["terminal_price_ticks"] == 10_010
    assert info["step_result"]["terminal_component"] == 10 * Q_STAR   # Perold term, mu = 0.
    assert obs[1] == 0.0   # Time remaining fraction at the horizon.


def test_truncation_invents_no_residual_term_and_repeats_the_last_observation() -> None:
    spec = worksheet_spec(side=1)
    short = slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), (DEEP, 10_005)])   # One bucket short.
    env = ExecutionEnv(ListPool([("short", spec, short)]))
    rewards, infos, observations = _rollout(env, "twap")
    last = infos[-1]
    assert last["step_result"]["truncated"] and not last["step_result"]["terminated"]
    assert last["step_result"]["terminal_component"] == 0
    assert last["step_result"]["terminal_price_ticks"] == 0 and last["reward_ticks"] == 0
    assert last["state"]["filled_qty_ticks"] == 200_000
    assert not last["terminal_observation_available"]
    np.testing.assert_array_equal(observations[-1], observations[-2])
    with pytest.raises(EpisodeOver):
        env.step(0.0)
    # With one bucket of data delay the newest visible bucket exists and the observation is real.
    history = slice_from_rows([(DEEP, P0)] * 3, [(DEEP, 10_002), (DEEP, 10_005)])
    delayed = ExecutionEnv(ListPool([("short", worksheet_spec(side=1, data_delay_buckets=1,
                                                               arrival_lookback=2), history)]))
    _, infos, observations = _rollout(delayed, "twap")
    assert infos[-1]["step_result"]["truncated"] and infos[-1]["terminal_observation_available"]


def test_zero_and_one_actions_and_scalar_or_array_inputs() -> None:
    env = ExecutionEnv(_worksheet_pool())
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["step_result"]["submitted_qty_ticks"] == Q_STAR
    assert info["step_result"]["fill_qty_ticks"] == info["step_result"]["cap_qty_ticks"] == 200_000
    _, _, _, _, info = env.step(0)
    assert info["step_result"]["submitted_qty_ticks"] == 0 and info["reward_ticks"] == 0
    _, _, _, _, info = env.step(np.float32(0.5))
    assert info["step_result"]["submitted_qty_ticks"] == 50_000


@pytest.mark.parametrize(
    "bad", [1.0000001, -0.1, float("nan"), float("inf"), np.array([0.2, 0.3]),
            np.array([[0.5]]), "0.5", True, np.array([True])],
)
def test_invalid_actions_raise_and_are_never_clipped(bad: object) -> None:
    env = ExecutionEnv(_worksheet_pool())
    env.reset()
    with pytest.raises((ValueError, TypeError)):
        env.step(bad)
    assert env.state.step == 0   # The rejected action consumed nothing.


def test_stepping_before_reset_is_an_error() -> None:
    env = ExecutionEnv(_worksheet_pool())
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(0.5)
    with pytest.raises(RuntimeError, match="call reset"):
        _ = env.state
    assert env.current_episode_id is None


def test_delayed_observations_are_causal_through_the_adapter() -> None:
    delayed = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=6, arrival_lookback=3,
                          data_delay_buckets=1)
    pool = _pool(3, template=delayed)
    env = ExecutionEnv(pool)
    obs, info = env.reset(seed=0)
    entry = pool[0]
    assert info["newest_visible_bucket"] == -2
    state = reset(entry.spec, entry.market)
    np.testing.assert_array_equal(obs, observe(state, entry.market, entry.spec).astype(np.float32))
    # Overwriting every bucket the decision may not read leaves the observation unchanged.
    perturbed = entry.market
    for bucket in range(-1, entry.market.n_after):
        perturbed = perturbed.with_bucket(bucket, event_count=9, total_quantity=5_000_000,
                                          signed_quantity=-4_000_000, vwap_ticks=777_777)
    other = ExecutionEnv(ListPool([("p", entry.spec, perturbed)]))
    np.testing.assert_array_equal(other.reset()[0], obs)


# --- normalisation and the smoke run -----------------------------------------------------


def test_frozen_training_statistics_are_applied_and_never_fitted() -> None:
    splits = _splits()
    draws = sample_episodes(splits.record, "train", 6, 1)
    stats = observation_statistics(splits, draws, SPEC, BASELINES["twap"], "twap")
    raw = ExecutionEnv(SplitPool(splits, "validation", SPEC, seed=2, n_episodes=3))
    scaled = ExecutionEnv(SplitPool(splits, "validation", SPEC, seed=2, n_episodes=3),
                          statistics=stats)
    assert scaled.observation_space.shape == (10,) and np.isinf(scaled.observation_space.low).all()
    obs_raw, _ = raw.reset(seed=0)
    obs_scaled, _ = scaled.reset(seed=0)
    mean, std = np.array(stats.mean), np.array(stats.std)
    expected = (obs_raw.astype(np.float64) - mean) / np.where(std > 0, std, 1.0)
    np.testing.assert_allclose(obs_scaled, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
    assert scaled.statistics is stats   # Unchanged by stepping on validation data.
    for _ in range(3):
        scaled.step(0.5)
    assert scaled.statistics == stats
    bad = observation_statistics(splits, draws, SPEC, BASELINES["twap"], "twap")
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS"):
        ExecutionEnv(_pool(2), statistics=bad.__class__(
            fields=("a",), mean=(0.0,), std=(1.0,), n_observations=1, n_episodes=1,
            policy="twap", seed=1,
        ))


def test_random_action_smoke_run_is_not_a_policy_but_keeps_every_invariant() -> None:
    env = ExecutionEnv(_pool(8), selection="random")
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    episodes = 0
    for _ in range(8):
        obs, info = env.reset()
        spec_target = info["target_qty_ticks"]
        done = False
        while not done:
            action = env.action_space.sample() if rng.uniform() < 0.5 else np.float32(rng.uniform())
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (10,) and obs.dtype == np.float32 and np.isfinite(reward)
            assert 0 <= info["state"]["filled_qty_ticks"] <= spec_target
            result = info["step_result"]
            assert result["fill_qty_ticks"] <= result["submitted_qty_ticks"]
            assert not (terminated and truncated)
            done = terminated or truncated
        episodes += 1
    assert episodes == 8
