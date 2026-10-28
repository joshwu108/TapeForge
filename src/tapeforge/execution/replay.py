"""Step-by-step replay of one recorded episode, for debugging fills, rewards and flags.

``replay_episode`` rebuilds the splits a run directory's manifest describes
(same data, same template), re-derives the draw named by an episode
identifier, replays the requested policy on it and returns every step with
what the policy could see, what it submitted, what filled at which price and
fee, what remained, the reward decomposition and how the episode ended.
Because draws, slices and transitions are deterministic, the replay is the
recorded episode, not an approximation of it; ``verify_against_record``
checks the replayed accounting against the stored summary.

Columns of a step row:
    ``step``, ``fill_bucket``, ``visible_bucket`` (the newest bucket the
    decision could read, ``t - 1 - data_delay``), ``visible_valid``,
    ``visible_qty_ticks``, ``mark_price_ticks``, ``mark_age``, ``action``,
    ``submitted_qty_ticks``, ``cap_qty_ticks``, ``fill_qty_ticks``,
    ``fill_price_ticks``, ``fee_ticks``, ``remaining_after``,
    ``inventory_after``, ``cash_after``, ``cost``, ``risk``, ``terminal``,
    ``reward_ticks``, ``terminated``, ``truncated``.

Not covered:
    Nothing here changes a record; the tool reads the run directory and the
    local data only (no download).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tapeforge.execution.baselines import BASELINES
from tapeforge.execution.experiment import (
    config_from_mapping,
    load_manifest,
    parse_episode_identifier,
    rebuild_splits,
)
from tapeforge.execution.runner import Trajectory, run_episode, summarize_episode
from tapeforge.execution.sampling import EpisodeDraw, draw_episode, materialize
from tapeforge.execution.simulator import newest_visible_bucket, remaining_qty_ticks
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import Splits
from tapeforge.execution.state import MarketSlice

__all__ = [
    "STEP_COLUMNS",
    "Replay",
    "format_replay",
    "replay_episode",
    "step_rows",
    "verify_against_record",
]

STEP_COLUMNS: tuple[str, ...] = (
    "step", "fill_bucket", "visible_bucket", "visible_valid", "visible_qty_ticks",
    "mark_price_ticks", "mark_age", "action", "submitted_qty_ticks", "cap_qty_ticks",
    "fill_qty_ticks", "fill_price_ticks", "fee_ticks", "remaining_after", "inventory_after",
    "cash_after", "cost", "risk", "terminal", "reward_ticks", "terminated", "truncated",
)


@dataclass(frozen=True, eq=False)
class Replay:
    """One replayed episode.

    Attributes:
        episode_id: The identifier replayed.
        policy: Policy name.
        draw: The re-derived draw.
        spec: The episode spec (side and target resolved).
        market: The slice the episode ran on.
        trajectory: The replayed trajectory.
    """

    episode_id: str
    policy: str
    draw: EpisodeDraw
    spec: EpisodeSpec
    market: MarketSlice
    trajectory: Trajectory


def step_rows(trajectory: Trajectory, market: MarketSlice) -> list[dict[str, Any]]:
    """Return one ``STEP_COLUMNS`` dict per decision of ``trajectory``."""
    spec = trajectory.spec
    rows = []
    for state_before, state_after, action, result in zip(
        trajectory.states[:-1], trajectory.states[1:], trajectory.actions, trajectory.results,
        strict=True,
    ):
        visible = newest_visible_bucket(state_before.step, spec)
        i = market.index(visible)
        rows.append(
            {
                "step": result.step,
                "fill_bucket": result.fill_bucket,
                "visible_bucket": visible,
                "visible_valid": bool(market.valid[i]),
                "visible_qty_ticks": int(market.total_quantity[i]),
                "mark_price_ticks": state_before.mark_price_ticks,
                "mark_age": state_before.mark_age,
                "action": action,
                "submitted_qty_ticks": result.submitted_qty_ticks,
                "cap_qty_ticks": result.cap_qty_ticks,
                "fill_qty_ticks": result.fill_qty_ticks,
                "fill_price_ticks": result.fill_price_ticks,
                "fee_ticks": result.fee_ticks,
                "remaining_after": remaining_qty_ticks(state_after, spec),
                "inventory_after": state_after.inventory_ticks,
                "cash_after": state_after.cash_ticks,
                "cost": result.cost_component,
                "risk": result.risk_component,
                "terminal": result.terminal_component,
                "reward_ticks": result.reward_ticks,
                "terminated": result.terminated,
                "truncated": result.truncated,
            }
        )
    return rows


def replay_episode(
    run_dir: str | Path, episode_id: str, policy: str, splits: Splits | None = None
) -> Replay:
    """Rebuild the episode ``episode_id`` of a run directory and replay ``policy`` on it.

    Args:
        run_dir: Directory written by ``experiment.run_experiment``.
        episode_id: ``<role>/s<seed>/i<index>``.
        policy: A ``BASELINES`` name.
        splits: Already rebuilt splits (saves the data load when replaying
            several episodes).

    Raises:
        ValueError: Unknown policy, a malformed identifier, an identifier
            outside the run's episode sets, or data that cannot be loaded.
    """
    if policy not in BASELINES:
        raise ValueError(f"unknown policy {policy!r}; available: {sorted(BASELINES)}")
    manifest = load_manifest(run_dir)
    role, seed, index = parse_episode_identifier(episode_id)
    listed = manifest["episodes"].get(role, {}).get(str(seed))
    if listed is None or index >= len(listed):
        raise ValueError(
            f"episode {episode_id!r} is not in the run: roles {sorted(manifest['episodes'])}, "
            f"seeds {sorted(manifest['episodes'].get(role, {}))}, "
            f"{len(listed or [])} episode(s) for that role and seed"
        )
    config = config_from_mapping(_config_mapping(manifest["config"]))
    if splits is None:
        splits = rebuild_splits(config)
    if json.loads(json.dumps(splits.record.to_json())) != manifest["split"]:
        raise ValueError("the rebuilt split record differs from the manifest's; the data changed")
    draw = draw_episode(splits.record, role, seed, index, tuple(config.rho_choices_bps))
    expected = {k: v for k, v in listed[index].items() if k != "id"}
    if draw.to_json() != expected:
        raise ValueError(f"the re-derived draw of {episode_id!r} differs from the manifest's")
    spec, market = materialize(splits, draw, config.template)
    trajectory = run_episode(spec, market, BASELINES[policy])
    return Replay(episode_id, policy, draw, spec, market, trajectory)


def _config_mapping(config_json: dict[str, Any]) -> dict[str, Any]:
    """Return the manifest's resolved config in the mapping layout ``config_from_mapping`` reads."""
    data = dict(config_json["data"])
    synthetic_days = data.pop("synthetic_days", [])
    n_buckets = data.pop("synthetic_n_buckets", 0)
    profile = data.pop("synthetic_profile", "uniform")
    mapping = {
        "schema": config_json.get("schema"),
        "name": config_json["name"],
        "label": config_json["label"],
        "description": config_json.get("description", ""),
        "data": data,
        "split": config_json["split"],
        "episodes": {
            "seeds": config_json["seeds"],
            "episodes_per_day": config_json["episodes_per_day"],
            "rho_choices_bps": config_json["rho_choices_bps"],
            "evaluate_roles": config_json["evaluate_roles"],
        },
        "policies": {"names": config_json["policies"]},
        "template": config_json["template"],
        "output_dir": config_json.get("output_dir", "runs"),
    }
    if data["source"] == "synthetic":
        mapping["data"]["synthetic"] = {
            "n_buckets": n_buckets, "profile": profile, "days": synthetic_days,
        }
    return mapping


def verify_against_record(replay: Replay, run_dir: str | Path) -> list[str]:
    """Compare the replayed episode summary with the stored record; return the mismatches.

    An empty list means the replay reproduced the record's accounting
    exactly (every integer field and the shortfall in basis points).
    """
    manifest = load_manifest(run_dir)
    role, seed, _ = parse_episode_identifier(replay.episode_id)
    entry = next(
        (r for r in manifest["runs"]
         if r["policy"] == replay.policy and r["role"] == role and r["seed"] == seed),
        None,
    )
    if entry is None:
        return [f"no record for policy {replay.policy!r}, role {role!r}, seed {seed}"]
    record = json.loads((Path(run_dir) / entry["record_file"]).read_text())
    stored = next(
        (e["summary"] for e, i in zip(record["episodes"], record["episode_ids"], strict=True)
         if i == replay.episode_id),
        None,
    )
    if stored is None:
        return [f"episode {replay.episode_id!r} is not in {entry['record_file']}"]
    replayed = summarize_episode(replay.trajectory).to_json()
    return [
        f"{key}: replayed {replayed[key]!r}, recorded {stored[key]!r}"
        for key in replayed if replayed[key] != stored.get(key)
    ]


def format_replay(replay: Replay) -> str:
    """Render the episode header, the step table and the terminal handling as text."""
    spec, trajectory, draw = replay.spec, replay.trajectory, replay.draw
    final, last = trajectory.final_state, trajectory.last_result
    summary = summarize_episode(trajectory)
    lines = [
        f"episode {replay.episode_id}  policy {replay.policy}",
        f"  date {draw.date}  segment {draw.segment}  start bucket {draw.start_bucket}  "
        f"side {'buy' if spec.side == 1 else 'sell'}  rho {draw.rho_bps} bps",
        f"  target {spec.target_qty_ticks} qty ticks  horizon {spec.horizon_buckets}  "
        f"latency {spec.latency_buckets}  data delay {spec.data_delay_buckets}  "
        f"cap {spec.participation_cap_bps} bps  fee {spec.fee_bps} bps  "
        f"lambda {spec.risk_lambda}  mu {spec.residual_mu}",
        f"  arrival price {trajectory.states[0].arrival_price_ticks} ticks "
        f"(mark age at reset {trajectory.states[0].mark_age})  slice buckets "
        f"[{-replay.market.origin}, {replay.market.n_after})",
        "",
    ]
    rows = step_rows(trajectory, replay.market)
    widths = {c: max(len(c), *(len(_cell(r[c])) for r in rows)) for c in STEP_COLUMNS}
    lines.append("  ".join(c.rjust(widths[c]) for c in STEP_COLUMNS))
    for row in rows:
        lines.append("  ".join(_cell(row[c]).rjust(widths[c]) for c in STEP_COLUMNS))
    ending = "terminated" if last.terminated else "truncated" if last.truncated else "unfinished"
    lines += [
        "",
        f"ending: {ending} after {summary.n_steps} step(s); filled {summary.filled_qty_ticks} of "
        f"{spec.target_qty_ticks} ({'completed' if summary.completed else 'residual'} "
        f"{summary.residual_qty_ticks})",
    ]
    if last.terminated and summary.residual_qty_ticks:
        lines.append(
            f"  terminal mark {last.terminal_price_ticks} ticks; residual term "
            f"{last.terminal_component} cash ticks = side {spec.side} x "
            f"({last.terminal_price_ticks} - {final.arrival_price_ticks}) x "
            f"{summary.residual_qty_ticks} + mu {spec.residual_mu} x {summary.residual_qty_ticks}"
        )
    if last.truncated:
        lines.append(
            "  the tape ended before the fill bucket or the terminal mark; no residual term, no "
            "terminal mark, shortfall of the fills only"
        )
    lines += [
        f"  cash {final.cash_ticks}  inventory {final.inventory_ticks}  fees {final.fees_ticks}  "
        f"exec notional {final.exec_notional_ticks}",
        f"  shortfall {summary.shortfall_ticks} cash ticks = {summary.shortfall_bps:.4f} bps; "
        f"reward sum {summary.reward_sum_ticks} = -(cost {summary.cost_sum_ticks} + risk "
        f"{summary.risk_sum_ticks} + terminal {summary.terminal_component_ticks})",
    ]
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Format one table cell."""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """``<run_dir> --episode <id> --policy <name> [--json] [--verify]``.

    Returns:
        0 on success (and, with ``--verify``, when the replay matches the
        record), 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.replay")
    parser.add_argument("run_dir")
    parser.add_argument("--episode", required=True, help="<role>/s<seed>/i<index>")
    parser.add_argument("--policy", required=True, choices=sorted(BASELINES))
    parser.add_argument("--json", action="store_true", help="print the step rows as JSON")
    parser.add_argument(
        "--verify", action="store_true", help="compare the replayed summary with the record"
    )
    args = parser.parse_args(argv)
    try:
        replay = replay_episode(args.run_dir, args.episode, args.policy)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {
            "episode_id": replay.episode_id,
            "policy": replay.policy,
            "draw": replay.draw.to_json(),
            "spec": asdict(replay.spec),
            "steps": step_rows(replay.trajectory, replay.market),
            "summary": summarize_episode(replay.trajectory).to_json(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_replay(replay))
    if args.verify:
        mismatches = verify_against_record(replay, args.run_dir)
        if mismatches:
            print("replay does NOT match the record:")
            for line in mismatches:
                print(f"  {line}")
            return 1
        print("replay matches the recorded episode summary exactly")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
