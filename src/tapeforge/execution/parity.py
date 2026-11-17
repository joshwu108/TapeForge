"""Parity driver: the batched simulator against the CPU reference after every transition.

``compare`` feeds identical action sequences to ``simulator`` (one episode at
a time) and to ``batched`` (all episodes at once, on a chosen device) and
compares, at every step, the observation the policy would see (bit for bit),
every ``StepResult`` field and every ``State`` field (exact integers and
flags). It also checks that environments whose episode ended are left
untouched. ``random_episodes`` builds a varied scenario set: both sides,
targets from one tick to millions, short and long horizons, latency and data
delay, empty and zero-quantity buckets, binding and non-binding caps, fees,
both terminal rules, slices cut short (missing fill buckets and missing
terminal marks) and prices near the int64 accounting bound.

Command line:
    ``python -m tapeforge.execution.parity --device cuda --episodes 2000
    --out parity.json`` exits 0 when every transition matched, 1 on a
    mismatch and 2 when the device is unavailable. A run on ``cpu`` checks
    the tensor code path only; it is not evidence about CUDA.

What a pass means:
    The two implementations agree on these inputs. It does not make either
    more realistic, and replaying recorded data does not model the market's
    response to the agent.

Dependencies:
    ``torch`` (``pip install -e '.[torch]'``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from tapeforge.execution import batched, simulator
from tapeforge.execution.runner import code_revision
from tapeforge.execution.spec import (
    TERMINAL_RULE_RESIDUAL_PEROLD,
    TERMINAL_RULE_SETTLEMENT,
    EpisodeSpec,
)
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "PARITY_SCHEMA",
    "ParityReport",
    "compare",
    "random_actions",
    "random_episodes",
]

PARITY_SCHEMA: Final[str] = "tapeforge.execution.simulator-parity/1"
_MAX_REPORTED: Final[int] = 20
_STATE_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(State))
_RESULT_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(StepResult))

Episode = tuple[EpisodeSpec, MarketSlice]


@dataclass
class ParityReport:
    """Outcome of one comparison.

    Attributes:
        n_episodes: Episodes compared.
        n_transitions: Reference transitions compared field by field.
        n_observations: Observation vectors compared bit for bit.
        coverage: How many episodes exercised each case of interest.
        mismatches: Human-readable differences, at most ``_MAX_REPORTED``.
        n_mismatches: Total differences found.
    """

    n_episodes: int = 0
    n_transitions: int = 0
    n_observations: int = 0
    coverage: dict[str, int] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    n_mismatches: int = 0

    @property
    def ok(self) -> bool:
        """Return whether every compared value matched."""
        return self.n_mismatches == 0

    def note(self, message: str) -> None:
        """Record one mismatch."""
        self.n_mismatches += 1
        if len(self.mismatches) < _MAX_REPORTED:
            self.mismatches.append(message)

    def count(self, case: str, hit: bool = True) -> None:
        """Count an episode toward a coverage case."""
        self.coverage[case] = self.coverage.get(case, 0) + int(hit)


# --- scenarios ----------------------------------------------------------------------------


def _random_tape(rng: np.random.Generator, n_before: int, n_after: int, price: int,
                 depth: int) -> MarketSlice:
    """Return a slice with empty, zero-quantity and traded buckets around ``price``."""
    n = n_before + n_after
    # 0: empty bucket, 1: records without quantity, 2: traded.
    kind = rng.choice(3, size=n, p=[0.18, 0.05, 0.77])
    traded = kind == 2
    total = np.where(traded, rng.integers(1, depth + 1, size=n), 0).astype(np.int64)
    steps = rng.integers(-max(price // 400, 1), max(price // 400, 1) + 1, size=n)
    vwap = np.where(traded, np.maximum(price + np.cumsum(steps), 1), 0).astype(np.int64)
    signed = np.where(traded, rng.integers(-total, total + 1), 0).astype(np.int64)
    count = np.where(kind == 0, 0, rng.integers(1, 9, size=n)).astype(np.int64)
    return MarketSlice(event_count=count, total_quantity=total, signed_quantity=signed,
                       vwap_ticks=vwap, valid=count > 0, origin=n_before)


def _random_episode(rng: np.random.Generator, near_bound: bool) -> Episode | None:
    """Return one candidate episode, or ``None`` when the reference rejects it."""
    horizon = int(rng.integers(1, 13))
    latency, delay = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    lookback = int(rng.integers(1, 6))
    n_before = max(horizon, lookback + delay) + int(rng.integers(0, 3))
    full_after = horizon + latency + lookback
    n_after = full_after if rng.random() < 0.7 else int(rng.integers(1, full_after + 1))
    settles = rng.random() < 0.6
    if near_bound:
        # Prices near 2**43 with a full participation cap put cash within a factor of a
        # few of the int64 limit while every value stays float64-exact.
        price, depth, cap_bps = int(rng.integers(2**42, 2**43)), 100_000, 10_000
        target, fee_bps = int(rng.integers(50_000, 100_001)), int(rng.choice([0, 1, 10]))
        slippage = risk = mu = 0
    else:
        price = int(rng.choice([1_000, 37_219, 4_000_000]))
        depth = int(rng.choice([20, 5_000, 4_000_000]))
        cap_bps = int(rng.choice([0, 1, 500, 1_000, 10_000]))
        target = int(rng.choice([1, 7, 1_000, 250_000, 3_000_000]))
        fee_bps = int(rng.choice([0, 10, 17, 250]))
        slippage = int(rng.choice([0, 0, 1, 3]))
        risk = int(rng.choice([0, 0, 3]))
        mu = 0 if settles else int(rng.choice([0, 5]))
    spec = EpisodeSpec(
        side=int(rng.choice([1, -1])), target_qty_ticks=target, horizon_buckets=horizon,
        participation_cap_bps=cap_bps, slippage_ticks=slippage, fee_bps=fee_bps,
        latency_buckets=latency, data_delay_buckets=delay, risk_lambda=risk, residual_mu=mu,
        arrival_lookback=lookback,
        terminal_rule=TERMINAL_RULE_SETTLEMENT if settles else TERMINAL_RULE_RESIDUAL_PEROLD,
        terminal_surcharge_bps=int(rng.choice([0, 10, 25, 50, 2_500])) if settles else 0,
    )
    market = _random_tape(rng, n_before, n_after, price, depth)
    try:
        batched.EpisodeBatch.from_episodes([(spec, market)])
    except (ValueError, OverflowError):
        return None
    return spec, market


def random_episodes(n_episodes: int, seed: int) -> list[Episode]:
    """Return ``n_episodes`` varied episodes the reference accepts, deterministic in ``seed``.

    Roughly one in eight is a near-bound episode. Candidates the reference
    ``reset`` rejects (no arrival price, int64 bound) are redrawn.
    """
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    while len(episodes) < n_episodes:
        candidate = _random_episode(rng, near_bound=len(episodes) % 8 == 7)
        if candidate is not None:
            episodes.append(candidate)
    return episodes


def random_actions(episodes: Sequence[Episode], seed: int) -> list[list[float]]:
    """Return one action per decision of every episode, deterministic in ``seed``.

    A mix of exact 0 and 1, float64 uniforms, and float32 uniforms widened to
    float64 (what a float32 policy emits), so the floor conversion is tried
    on values of both precisions.
    """
    rng = np.random.default_rng(seed)
    actions = []
    for spec, _ in episodes:
        kinds = rng.choice(4, size=spec.horizon_buckets, p=[0.15, 0.15, 0.35, 0.35])
        uniform = rng.random(spec.horizon_buckets)
        narrow = uniform.astype(np.float32).astype(np.float64)
        actions.append([
            0.0 if kind == 0 else 1.0 if kind == 1 else float(uniform[i]) if kind == 2
            else float(narrow[i])
            for i, kind in enumerate(kinds)
        ])
    return actions


# --- comparison ----------------------------------------------------------------------------


def _to_host(record: Any) -> dict[str, np.ndarray]:
    """Return every tensor field of a batched record as a NumPy array."""
    return {f.name: getattr(record, f.name).cpu().numpy() for f in fields(record)}


def compare(
    episodes: Sequence[Episode],
    actions: Sequence[Sequence[float]],
    device: str | torch.device = "cpu",
) -> ParityReport:
    """Step the reference and the batched simulator together and compare every transition.

    ``actions[k]`` holds the fractions for episode ``k``, one per decision;
    a sequence shorter than the episode is padded with zeros.

    Raises:
        ValueError: ``episodes`` and ``actions`` differ in length, or the
            batch cannot be built (see ``EpisodeBatch.from_episodes``).
    """
    if len(episodes) != len(actions):
        raise ValueError("one action sequence per episode is required")
    report = ParityReport(n_episodes=len(episodes))
    batch = batched.EpisodeBatch.from_episodes(episodes, device)
    state = batched.reset(batch, torch.arange(len(episodes), device=batch.device))
    references = [simulator.reset(spec, market) for spec, market in episodes]
    ended = [False] * len(episodes)
    seen: list[dict[str, bool]] = [{} for _ in episodes]

    def check_state(k: int, host: dict[str, np.ndarray], when: str) -> None:
        for name in _STATE_FIELDS:
            if int(host[name][k]) != getattr(references[k], name):
                report.note(f"episode {k} {when}: state.{name} batched {int(host[name][k])} "
                            f"reference {getattr(references[k], name)}")

    host_state = _to_host(state)
    for k in range(len(episodes)):
        check_state(k, host_state, "after reset")

    for t in range(max(spec.horizon_buckets for spec, _ in episodes)):
        observed, observable = batched.observe(state, batch)
        observations, available = observed.cpu().numpy(), observable.cpu().numpy()
        step_actions = [seq[t] if t < len(seq) else 0.0 for seq in actions]
        state, result = batched.step(
            state, torch.tensor(step_actions, dtype=torch.float64, device=batch.device), batch
        )
        previous_state, host_state, host_result = host_state, _to_host(state), _to_host(result)
        for k, (spec, market) in enumerate(episodes):
            if ended[k] or simulator.is_done(references[k], spec):
                if bool(host_result["active"][k]) or any(
                    host_state[name][k] != previous_state[name][k] for name in _STATE_FIELDS
                ):
                    report.note(f"episode {k} step {t}: an ended environment was stepped")
                continue
            if not bool(host_result["active"][k]):
                report.note(f"episode {k} step {t}: a running environment was not stepped")
                continue
            expected = simulator.observe(references[k], market, spec)
            report.n_observations += 1
            if not available[k] or expected.tobytes() != observations[k].tobytes():
                report.note(f"episode {k} step {t}: observation batched "
                            f"{observations[k].tolist()} reference {expected.tolist()}")
            references[k], reference = simulator.step(references[k], step_actions[k], market, spec)
            report.n_transitions += 1
            for name in _RESULT_FIELDS:
                got, want = host_result[name][k], getattr(reference, name)
                if type(want)(got) != want:
                    report.note(f"episode {k} step {t}: result.{name} batched {got} "
                                f"reference {want}")
            check_state(k, host_state, f"after step {t}")
            ended[k] = reference.terminated or reference.truncated
            if bool(host_state["ended"][k]) != ended[k]:
                report.note(f"episode {k} step {t}: ended flag batched "
                            f"{bool(host_state['ended'][k])} reference {ended[k]}")
            flags = seen[k]
            flags["capped"] = flags.get("capped", False) or (
                reference.fill_qty_ticks == reference.cap_qty_ticks < reference.submitted_qty_ticks
            )
            flags["empty_fill_bucket"] = flags.get("empty_fill_bucket", False) or (
                reference.cap_qty_ticks == 0 and not reference.truncated
            )
            if ended[k]:
                flags["terminated"] = reference.terminated
                flags["truncated"] = reference.truncated
                flags["settled"] = reference.settlement_qty_ticks > 0
                flags["settlement_over_cap"] = reference.settlement_exceeds_cap
                flags["legacy_residual"] = (
                    reference.terminated and not spec.settles_at_horizon
                    and simulator.remaining_qty_ticks(references[k], spec) > 0
                )
                flags["completed_early"] = (
                    reference.terminated and references[k].step < spec.horizon_buckets
                )
                final = simulator.newest_visible_bucket(references[k].step, spec)
                flags["no_terminal_observation"] = not market.has_bucket(final)

    available = batched.observe(state, batch)[1].cpu().numpy()
    for k, (spec, market) in enumerate(episodes):
        newest = simulator.newest_visible_bucket(references[k].step, spec)
        if bool(available[k]) != market.has_bucket(newest):
            report.note(f"episode {k}: final observation availability differs")
        report.count("buy", spec.side == 1)
        report.count("sell", spec.side == -1)
        report.count("latency", spec.latency_buckets > 0)
        report.count("data_delay", spec.data_delay_buckets > 0)
        report.count("fees", spec.fee_bps > 0)
        report.count("risk_term", spec.risk_lambda > 0)
        report.count("near_int64_bound", int(market.vwap_ticks.max()) >= 2**42)
        for case in ("terminated", "truncated", "settled", "settlement_over_cap",
                     "legacy_residual", "completed_early", "capped", "empty_fill_bucket",
                     "no_terminal_observation"):
            report.count(case, seen[k].get(case, False))
    return report


# --- command line ------------------------------------------------------------------------------


def _device_description(device: torch.device) -> dict[str, Any]:
    """Return what identifies the hardware and software a parity run executed on."""
    description: dict[str, Any] = {
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
    }
    if device.type == "cuda":
        description["cuda_device"] = torch.cuda.get_device_name(device)
        description["cuda_runtime"] = torch.version.cuda
    return description


def main(argv: list[str] | None = None) -> int:
    """``--device D --episodes N --seed S [--out PATH]``: run a seeded parity comparison.

    Returns:
        0 when everything matched, 1 on any mismatch, 2 when the device is
        not available (nothing was compared; this is a skip, not a pass).
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.parity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args(argv)
    try:
        device = torch.device(args.device)
        torch.zeros(1, dtype=torch.float64, device=device)
    except (RuntimeError, AssertionError, TypeError) as exc:
        print(f"SKIPPED: device {args.device!r} is not available ({exc}); nothing was compared")
        return 2
    started = time.perf_counter()
    episodes = random_episodes(args.episodes, args.seed)
    report = compare(episodes, random_actions(episodes, args.seed + 1), device)
    payload = {
        "schema": PARITY_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "simulator_version": simulator.SIMULATOR_VERSION,
        "environment": _device_description(device),
        "seed": args.seed,
        "seconds": time.perf_counter() - started,
        "ok": report.ok,
        **asdict(report),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{'OK' if report.ok else 'MISMATCH'} on {device}: {report.n_episodes} episodes, "
          f"{report.n_transitions} transitions, {report.n_observations} observations, "
          f"{report.n_mismatches} mismatches")
    print(f"coverage: {report.coverage}")
    for line in report.mismatches:
        print(f"  {line}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
