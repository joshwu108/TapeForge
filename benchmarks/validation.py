"""Validation replay: per-decision against lockstep, on fixed checkpoints, CPU and CUDA policies.

The trainer scores a checkpoint by replaying the fixed validation episodes
(draws ``0 .. n - 1`` of every evaluation seed) on the reference simulator.
``schedule.lockstep_validation`` replays them together and asks the policy
once per decision round. This script checks, on given checkpoint
directories and before any timing is reported, that the two agree:

* the same draws (episode identifiers, sides, targets) and the same
  per-episode record fields apart from the outcome of the actions;
* exact trajectories for a batch policy that answers each row with the
  one-row policy (``lockstep_exact``): the replay itself adds nothing;
* for the batched network forward (``lockstep``): the number of episodes
  with identical action sequences, the largest action difference, the
  difference of every summary statistic and of the selection metric;
* the checkpoint the selection rule (lowest metric) picks under each mode.

Each mode runs ``--repeats`` times per checkpoint and policy device; the
first repeat includes one-time costs (policy copy, CUDA context and kernel
selection), reported separately from the later repeats. Peak RSS and, on
CUDA, peak allocated device memory are recorded.

The policy device is where the network runs (``cpu``, or ``cuda`` with the
model loaded there); the simulator is always the CPU reference simulator.

Usage:
    python benchmarks/validation.py --checkpoints CK1 CK2 ... --devices cpu cuda \\
        --out runs/benchmarks/validation.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from harness import capture_environment, snapshot_metadata

from tapeforge.execution.rl.ppo import prepare_training
from tapeforge.execution.rl.ppo_config import load_ppo_config
from tapeforge.execution.rl.ppo_eval import (
    checkpoint_batch_policy,
    checkpoint_policy,
    load_checkpoint,
    sha256_of_file,
)
from tapeforge.execution.runner import run_policy

MODES = ("per_decision", "lockstep_exact", "lockstep")
SUMMARY_KEYS = ("n_episodes", "n_truncated", "mean_shortfall_bps", "p50_shortfall_bps",
                "p95_shortfall_bps", "completion_by_fills_rate", "mean_settlement_fraction")
# Episode-record entries fixed by the draw and the template, not by the policy's actions.
DRAW_KEYS = ("draw", "spec")


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def replay(mode: str, model: Any, statistics: Any, data: Any, device: str) -> dict[str, Any]:
    """Replay every validation seed once in ``mode``; return records, trajectories and timing."""
    experiment, training = data.experiment, data.training
    n_episodes = experiment.n_episodes("validation")
    _sync(device)
    started = time.perf_counter()
    single = checkpoint_policy(model, statistics, device)
    batch = None
    if mode == "lockstep":
        batch = checkpoint_batch_policy(model, statistics, device)
    elif mode == "lockstep_exact":
        def batch(observations: Any) -> list[float]:
            return [single(o) for o in observations]
    calls = [0]
    if batch is not None:
        inner = batch

        def batch(observations: Any) -> Any:  # counts network calls (rounds)
            calls[0] += 1
            return inner(observations)
    else:
        inner_single = single

        def single(observation: Any) -> float:
            calls[0] += 1
            return inner_single(observation)
    per_seed: dict[str, Any] = {}
    for seed in training.eval_seeds:
        record, trajectories = run_policy(
            data.splits, "validation", None if batch is not None else single, "ppo",
            experiment.template, n_episodes, seed, experiment.rho_choices_bps,
            batch_policy=batch,
        )
        per_seed[str(seed)] = {"record": record, "trajectories": trajectories}
    _sync(device)
    seconds = time.perf_counter() - started
    metric = float(np.mean([s["record"]["summary"]["mean_shortfall_bps"]
                            for s in per_seed.values()]))
    return {"per_seed": per_seed, "seconds": seconds, "policy_calls": calls[0], "metric": metric}


def compare(got: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Return how ``got`` differs from ``reference`` (same checkpoint, device and seeds)."""
    same_draws = True
    identical_episodes = 0
    episodes = 0
    max_action = 0.0
    exact_trajectories = True
    summary_diff = {k: 0.0 for k in SUMMARY_KEYS}
    for seed, ref in reference["per_seed"].items():
        new = got["per_seed"][seed]
        for a, b in zip(new["record"]["episodes"], ref["record"]["episodes"], strict=True):
            same_draws &= all(a[k] == b[k] for k in DRAW_KEYS)
        for a, b in zip(new["trajectories"], ref["trajectories"], strict=True):
            episodes += 1
            if a.actions == b.actions:
                identical_episodes += 1
            n = min(len(a.actions), len(b.actions))
            if n:
                diff = np.abs(np.subtract(a.actions[:n], b.actions[:n], dtype=np.float64))
                max_action = max(max_action, float(diff.max()))
            exact_trajectories &= (a.states == b.states and a.results == b.results
                                   and a.actions == b.actions)
        for k in SUMMARY_KEYS:
            x, y = new["record"]["summary"][k], ref["record"]["summary"][k]
            if x is not None and y is not None:
                summary_diff[k] = max(summary_diff[k], abs(float(x) - float(y)))
    return {
        "same_draws": same_draws, "episodes": episodes,
        "episodes_with_identical_actions": identical_episodes,
        "exact_trajectories": exact_trajectories, "max_abs_action_diff": max_action,
        "max_abs_summary_diff": summary_diff,
        "metric_diff_bps": got["metric"] - reference["metric"],
    }


def main(argv: list[str] | None = None) -> int:
    """Compare the modes on every checkpoint and device; exit 1 when the replay check fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-config", default="experiments/configs/smoke-synthetic-ppo.toml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default="runs/benchmarks/validation.json")
    args = parser.parse_args(argv)

    config = load_ppo_config(args.ppo_config)
    torch.set_num_threads(config.torch_threads)
    data = prepare_training(config)
    rows: list[dict[str, Any]] = []
    failed = False
    devices = [d for d in args.devices if d == "cpu" or torch.cuda.is_available()]
    skipped = sorted(set(args.devices) - set(devices))
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    reference_cpu: dict[str, dict[str, Any]] = {}
    for device in devices:
        for checkpoint in args.checkpoints:
            model, provenance, statistics = load_checkpoint(checkpoint, device=device)
            if statistics is not None and statistics.to_json() != data.statistics.to_json():
                raise ValueError(f"{checkpoint}: statistics differ from the configuration's")
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            results: dict[str, dict[str, Any]] = {}
            timings: dict[str, list[float]] = {}
            for mode in MODES:
                for _ in range(args.repeats):
                    result = replay(mode, model, statistics, data, device)
                    timings.setdefault(mode, []).append(result["seconds"])
                    results[mode] = result
            reference = results["per_decision"]
            if device == "cpu":
                reference_cpu[checkpoint] = reference
            row: dict[str, Any] = {
                "checkpoint": checkpoint,
                "model_sha256": sha256_of_file(Path(checkpoint) / "model.zip"),
                "policy_version": provenance.get("policy_version"),
                "policy_device": device,
                "metric_bps": {m: results[m]["metric"] for m in MODES},
                "policy_calls": {m: results[m]["policy_calls"] for m in MODES},
                "seconds_first": {m: timings[m][0] for m in MODES},
                "seconds_later": {m: timings[m][1:] for m in MODES},
                "vs_per_decision": {m: compare(results[m], reference) for m in MODES[1:]},
                "peak_rss_bytes": _rss_bytes(),
            }
            if device.startswith("cuda"):
                row["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                if checkpoint in reference_cpu:
                    row["per_decision_vs_cpu_per_decision"] = compare(reference,
                                                                      reference_cpu[checkpoint])
            exact = row["vs_per_decision"]["lockstep_exact"]
            close = row["vs_per_decision"]["lockstep"]
            row["check"] = {
                "lockstep_exact_identical": exact["exact_trajectories"] and exact["same_draws"],
                "lockstep_same_draws": close["same_draws"],
            }
            failed |= not all(row["check"].values())
            for m in MODES:
                metrics.setdefault((device, m), {})[checkpoint] = results[m]["metric"]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("checkpoint", "policy_device", "metric_bps",
                                                  "seconds_first", "seconds_later", "check")}),
                  flush=True)
    selection = {
        f"{device}/{mode}": min(values, key=values.__getitem__)
        for (device, mode), values in metrics.items()
    }
    document = {
        "schema": "tapeforge.benchmark.validation/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "scope": "validation replay of fixed checkpoints on the CPU reference simulator; "
                 "per policy device and mode, wall time of all evaluation seeds",
        "arguments": vars(args),
        "experiment": data.experiment.name,
        "eval_seeds": list(data.training.eval_seeds),
        "n_episodes_per_seed": data.experiment.n_episodes("validation"),
        "devices_skipped": skipped,
        "rows": rows,
        "selected_checkpoint": selection,
        "same_selection": len(set(selection.values())) == 1,
    }
    failed |= not document["same_selection"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, default=str))
    print(json.dumps({"selected_checkpoint": selection, "devices_skipped": skipped,
                      "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
