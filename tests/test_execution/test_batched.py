"""Tests of the batched tensor simulator against the CPU reference.

Every comparison is made after each transition, never on final rewards only.
On a machine without CUDA these tests exercise the tensor code path on CPU
tensors; that is not evidence about CUDA, and the CUDA test here skips.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from tapeforge.contract import INT64_MAX
from tapeforge.execution import simulator
from tapeforge.execution.spec import EpisodeSpec, check_episode_bounds
from tapeforge.execution.state import State
from test_execution.tapes import (
    DEEP,
    P0,
    legacy_spec,
    slice_from_rows,
    worksheet_spec,
    worksheet_tape,
)

torch = pytest.importorskip("torch")

from tapeforge.execution import batched, parity  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent.parent
STATE_FIELDS = [f.name for f in fields(State)]


def _actions(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _batch(*episodes: tuple[EpisodeSpec, object]) -> tuple[batched.EpisodeBatch,
                                                           batched.BatchedState]:
    batch = batched.EpisodeBatch.from_episodes(list(episodes))  # type: ignore[arg-type]
    return batch, batched.reset(batch, torch.arange(len(episodes)))


# --- randomized parity ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_scenarios_match_the_reference_after_every_transition(seed: int) -> None:
    episodes = parity.random_episodes(400, seed)
    report = parity.compare(episodes, parity.random_actions(episodes, seed + 100))
    assert report.ok, report.mismatches
    assert report.n_transitions == report.n_observations > 1_500
    # The scenario set must actually reach the cases the parity claim is about.
    for case in ("buy", "sell", "latency", "data_delay", "fees", "risk_term", "terminated",
                 "truncated", "settled", "settlement_over_cap", "legacy_residual",
                 "completed_early", "capped", "empty_fill_bucket", "near_int64_bound",
                 "no_terminal_observation"):
        assert report.coverage[case] > 0, case


def test_the_parity_driver_detects_a_changed_rounding_rule(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = parity.random_episodes(200, 5)
    actions = parity.random_actions(episodes, 6)
    monkeypatch.setattr(  # a floor where the reference takes a ceiling
        batched, "_fee", lambda notional, fee_bps: batched._floor_div(notional * fee_bps, 10_000)
    )
    report = parity.compare(episodes, actions)
    assert not report.ok
    assert any("fee" in line for line in report.mismatches)


def test_a_float32_action_is_widened_explicitly_and_floors_like_the_reference() -> None:
    spec, market = worksheet_spec(1), worksheet_tape()
    remaining = spec.target_qty_ticks
    # A fraction whose float32 and float64 values floor to different tick counts.
    wide = next(
        k / 1000 for k in range(1, 1000)
        if int(np.floor(float(np.float32(k / 1000)) * remaining))
        != int(np.floor(k / 1000 * remaining))
    )
    narrow = np.float32(wide)
    batch, state = _batch((spec, market))
    with pytest.raises(TypeError, match="widen"):
        batched.step(state, torch.tensor([narrow]), batch)      # a float32 tensor is refused
    _, widened = batched.step(state, torch.tensor([narrow]).double(), batch)
    _, reference = simulator.step(simulator.reset(spec, market), float(narrow), market, spec)
    _, other = simulator.step(simulator.reset(spec, market), wide, market, spec)
    assert int(widened.submitted_qty_ticks) == reference.submitted_qty_ticks
    assert abs(reference.submitted_qty_ticks - other.submitted_qty_ticks) == 1


# --- hand-checked cases -----------------------------------------------------------------------


def test_the_reference_fixture_settles_to_the_hand_checked_figures() -> None:
    # Reference fixture (README worked example): bucket quantities 600,000 / 400,000 /
    # 800,000 cap the fills at 60,000 / 40,000 / 80,000, so 120,000 of 300,000 ticks are
    # left at the horizon; mark 10,010 at 25 bps settles at 10,036 (buy) / 9,984 (sell).
    market = worksheet_tape(totals=(600_000, 400_000, 800_000))
    for side, price, fee, total in ((1, 10_036, 1_204_320, 7_724_720),
                                    (-1, 9_984, 1_198_080, 4_518_480)):
        spec = worksheet_spec(side, terminal_surcharge_bps=25)
        batch, state = _batch((spec, market))
        results = []
        for remaining in (300_000, 240_000, 200_000):
            action = simulator.action_for_quantity(100_000, remaining)
            state, result = batched.step(state, _actions([action]), batch)
            results.append(result)
        assert [int(r.fill_qty_ticks) for r in results] == [60_000, 40_000, 80_000]
        last = results[-1]
        assert int(last.settlement_qty_ticks) == 120_000
        assert int(last.settlement_price_ticks) == price
        assert int(last.settlement_fee_ticks) == fee
        assert not bool(last.settlement_exceeds_cap)      # the terminal bucket's cap is 200,000
        assert bool(last.terminated) and not bool(last.truncated)
        assert -sum(int(r.reward_ticks) for r in results) == total
        assert int(state.settled_qty_ticks) == 120_000 and bool(state.ended)
        assert int(batched.remaining_qty_ticks(state, batch)) == 0


def test_mixed_sides_targets_horizons_and_rules_share_one_batch() -> None:
    long_tape = slice_from_rows([(DEEP, P0)] * 6, [(DEEP, P0 + i) for i in range(9)])
    episodes = [
        (worksheet_spec(1, terminal_surcharge_bps=50), worksheet_tape()),
        (worksheet_spec(-1, target_qty_ticks=7), worksheet_tape()),
        (legacy_spec(1, residual_mu=5), worksheet_tape()),
        (EpisodeSpec(side=-1, target_qty_ticks=900_000, horizon_buckets=6, arrival_lookback=2,
                     latency_buckets=1, data_delay_buckets=1, risk_lambda=3), long_tape),
        (EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=1, arrival_lookback=1),
         worksheet_tape()),
    ]
    actions = [[0.3, 0.0, 0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
               [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0.0]]
    report = parity.compare(episodes, actions)
    assert report.ok, report.mismatches
    assert report.coverage["completed_early"] >= 1 and report.coverage["legacy_residual"] == 1


def test_empty_buckets_missing_fill_buckets_and_missing_marks_match_the_reference() -> None:
    spec = worksheet_spec(1, terminal_surcharge_bps=25)
    episodes = [
        (spec, worksheet_tape(after=())),                              # no terminal mark
        (spec, slice_from_rows([(DEEP, P0)], [(DEEP, 10_002), None])),  # tape ends mid-episode
        (spec, slice_from_rows([(DEEP, P0)], [None, (0, 0), (DEEP, 10_001), (DEEP, 10_010)])),
        (replace(spec, arrival_lookback=3),
         slice_from_rows([(DEEP, P0), None, None], [(DEEP, 10_002)] * 3 + [None, None,
                                                                             (DEEP, 10_010)])),
    ]
    report = parity.compare(episodes, [[0.5, 0.5, 0.5]] * 4)
    assert report.ok, report.mismatches
    assert report.coverage["truncated"] == 2 and report.coverage["empty_fill_bucket"] >= 1


def test_arithmetic_just_inside_the_int64_bound_is_exact() -> None:
    price, depth = 2**52, 1_000
    spec = EpisodeSpec(side=1, target_qty_ticks=1_000, horizon_buckets=1, arrival_lookback=1,
                       participation_cap_bps=10_000, fee_bps=1, terminal_surcharge_bps=0)
    market = slice_from_rows([(depth, price)], [(depth, price), (depth, price)])
    check_episode_bounds(spec, price, depth)
    cash_bound = depth * price + spec.target_qty_ticks * price
    assert INT64_MAX // 2 < cash_bound <= INT64_MAX       # within a factor of two of the limit
    report = parity.compare([(spec, market), (replace(spec, side=-1), market)], [[1.0], [0.37]])
    assert report.ok, report.mismatches
    with pytest.raises(OverflowError, match="int64"):
        batched.EpisodeBatch.from_episodes([(replace(spec, target_qty_ticks=1_100), market)])


# --- construction, validation and masking ------------------------------------------------------


def test_construction_validates_once_with_the_reference_rules() -> None:
    spec, market = worksheet_spec(1), worksheet_tape()
    with pytest.raises(ValueError, match="at least one"):
        batched.EpisodeBatch.from_episodes([])
    with pytest.raises(ValueError, match="arrival"):          # the reference reset's own error
        batched.EpisodeBatch.from_episodes([(replace(spec, arrival_lookback=5), market)])
    huge = slice_from_rows([(10, 2**53)], [(10, 2**53)] * 4)   # inside int64, not float64-exact
    with pytest.raises(ValueError, match=r"2\*\*53"):
        batched.EpisodeBatch.from_episodes([(replace(spec, target_qty_ticks=1), huge)])
    penny = slice_from_rows([(DEEP, 1)], [(DEEP, 1)] * 4)
    with pytest.raises(ValueError, match="settlement price"):
        batched.EpisodeBatch.from_episodes(
            [(worksheet_spec(-1, terminal_surcharge_bps=25), penny)]
        )
    with pytest.raises(ValueError, match="limit"):
        batched.EpisodeBatch.from_episodes([(spec, market)], memory_limit_bytes=64)
    batch = batched.EpisodeBatch.from_episodes([(spec, market)] * 3)
    assert batch.n_episodes == 3
    assert batch.nbytes == batched.EpisodeBatch.estimate_bytes(3, market.n_buckets)


def test_actions_are_validated_like_the_reference_unless_the_caller_opts_out() -> None:
    batch, state = _batch((worksheet_spec(1), worksheet_tape()))
    for bad in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            batched.step(state, _actions([bad]), batch)
    with pytest.raises(TypeError, match="float64"):
        batched.step(state, torch.tensor([0.5], dtype=torch.float32), batch)
    with pytest.raises(TypeError, match="shape"):
        batched.step(state, _actions([0.5, 0.5]), batch)
    new_state, result = batched.step(state, _actions([0.5]), batch, validate=False)
    assert int(result.submitted_qty_ticks) == 150_000 and int(new_state.step) == 1


def test_ended_environments_are_masked_and_reset_independently() -> None:
    spec = worksheet_spec(1, target_qty_ticks=150_000)     # below the 200,000-tick bucket cap
    batch, state = _batch((spec, worksheet_tape()), (spec, worksheet_tape()))
    state, result = batched.step(state, _actions([1.0, 0.1]), batch)
    assert result.terminated.tolist() == [True, False] and state.ended.tolist() == [True, False]
    frozen = {name: int(getattr(state, name)[0]) for name in STATE_FIELDS}

    state, result = batched.step(state, _actions([1.0, 0.1]), batch)
    assert result.active.tolist() == [False, True]
    assert {name: int(getattr(state, name)[0]) for name in STATE_FIELDS} == frozen
    for field in fields(result):
        assert not bool(getattr(result, field.name)[0]), field.name

    running = {name: int(getattr(state, name)[1]) for name in STATE_FIELDS}
    mask = torch.tensor([True, False])
    state = batched.reset_where(state, mask, torch.tensor([1, 0]), batch)
    assert state.episode.tolist() == [1, 1] and state.ended.tolist() == [False, False]
    reference = simulator.reset(spec, worksheet_tape())
    assert {name: int(getattr(state, name)[0]) for name in STATE_FIELDS} == {
        name: getattr(reference, name) for name in STATE_FIELDS
    }
    assert {name: int(getattr(state, name)[1]) for name in STATE_FIELDS} == running


def test_stepping_and_observing_never_synchronise_with_the_host(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = parity.random_episodes(32, 9)
    batch = batched.EpisodeBatch.from_episodes(episodes)
    state = batched.reset(batch, torch.arange(len(episodes)))
    actions = torch.full((len(episodes),), 0.5, dtype=torch.float64)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("host synchronisation inside a step")

    for name in ("item", "cpu", "numpy", "tolist", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    for _ in range(4):
        batched.observe(state, batch)
        state, _ = batched.step(state, actions, batch, validate=False)
    with pytest.raises(AssertionError, match="host synchronisation"):
        batched.step(state, actions, batch, validate=True)   # the documented exception


# --- devices and optional dependency ----------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_random_scenarios_match_the_reference_on_cuda() -> None:
    episodes = parity.random_episodes(2_000, 0)
    report = parity.compare(episodes, parity.random_actions(episodes, 1), device="cuda")
    assert report.ok, report.mismatches


def test_the_parity_command_reports_an_unavailable_device_as_a_skip(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "parity.json"
    assert parity.main(["--episodes", "50", "--seed", "3", "--out", str(out)]) == 0
    assert '"ok": true' in out.read_text() and "OK on cpu" in capsys.readouterr().out
    if not torch.cuda.is_available():
        assert parity.main(["--device", "cuda", "--episodes", "5"]) == 2
        assert "SKIPPED" in capsys.readouterr().out


def test_the_cpu_reference_does_not_import_the_batched_simulator() -> None:
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import tapeforge.execution, tapeforge.execution.simulator; "
        "from tapeforge.execution.research import experiment, report; "
        "assert 'tapeforge.execution.batched' not in sys.modules; print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=str(TESTS_DIR))
    usable = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=False, env=env)
    assert usable.returncode == 0, usable.stderr
    missing = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['torch'] = None; import tapeforge.execution.batched"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert missing.returncode != 0 and "pip install -e '.[torch]'" in missing.stderr
