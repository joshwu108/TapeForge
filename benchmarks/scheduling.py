"""Kernel scope: the grouped short-bucket path against the existing dispatch, same workloads.

For each workload (seeded synthetic tape, resident on the device) this
prepares every configuration once, gates its output against the CPU
reference, and times ``PreparedAggregation.execute()`` -- kernel launches
into prepared buffers, no validation, no host sync -- with both timing
methods of ``harness.time_callable``. Preparation (validation, planning,
long-bucket or chunk schedule, allocation) is timed once per configuration
and reported as ``prepare_ms``; it is not part of the execute figures.

Configurations:
    auto                 the dispatch ops.aggregate uses (one-pass or two-pass)
    two_pass             two-pass forced
    grouped/t<T>/l<L>    grouped path with tile T (``auto`` = kernels.choose_grouped_tile,
                         the cost-model choice) and L lanes per program
    torch_eager          benchmarks.baselines.torch_index_add (no validation)

Every row records the workload's bucket-length distribution, the plan
(tile, group size, number of long buckets and their chunks) and the
environment. Rows are printed as JSON lines and written to ``--out``.

Usage:
    python benchmarks/scheduling.py --sizes 1000000 10000000 --out runs/benchmarks/sched.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from baselines import torch_index_add
from harness import capture_environment, snapshot_metadata, time_callable
from workload import check_features, host_columns, make_workload

from tapeforge import prepared, reference


def configurations(tiles: list[str], lanes: list[int]) -> list[dict[str, Any]]:
    """Return the prepare options of every configuration, in a fixed order."""
    configs: list[dict[str, Any]] = [{"name": "auto"}, {"name": "two_pass", "path": "two_pass"}]
    for tile in tiles:
        for lane in lanes:
            configs.append({
                "name": f"grouped/t{tile}/l{lane}", "path": "grouped",
                "block_size": None if tile == "auto" else int(tile), "group_lanes": lane,
            })
    configs.append({"name": "torch_eager"})
    return configs


def measure(config: dict[str, Any], columns: dict[str, Any], offsets: Any,
            expected: reference.Features, warmup: int, repeats: int) -> dict[str, Any]:
    """Prepare, gate and time one configuration."""
    import torch

    row: dict[str, Any] = {"config": config["name"]}
    try:
        if config["name"] == "torch_eager":
            def run() -> dict[str, Any]:
                return torch_index_add(columns, offsets)
            row["plan"] = None
            row["prepare_ms"] = 0.0
        else:
            options = {k: v for k, v in config.items() if k != "name"}
            torch.cuda.synchronize()
            started = time.perf_counter()
            ready = prepared.prepare(columns, offsets, **options)
            torch.cuda.synchronize()
            row["prepare_ms"] = (time.perf_counter() - started) * 1e3
            long_buckets = ready.long_buckets
            row["plan"] = {
                "path": ready.path, "block_size": ready.plan.block_size,
                "group_size": ready.plan.group_size, "n_chunks": ready.n_chunks,
                "n_long_buckets": long_buckets.n_long if long_buckets is not None else None,
            }
            run = ready.execute
        check_features(run(), expected)
    except AssertionError as exc:
        return row | {"status": "incorrect", "reason": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - recorded per row
        return row | {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    # Slow rows (the one-pass kernel on sparse tapes takes most of a second) get
    # fewer repeats so the sweep stays bounded; the count is recorded per row.
    torch.cuda.synchronize()
    started = time.perf_counter()
    run()
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - started) * 1e3
    repeats = max(10, min(repeats, int(3000 / max(single_ms, 1e-3))))
    warmup = max(2, min(warmup, repeats // 5))
    events = time_callable(run, warmup=warmup, repeats=repeats, method="cuda_events")
    wall = time_callable(run, warmup=warmup, repeats=repeats, synchronize=True)
    return row | {
        "status": "ok", "correct": True,
        **{f"events_{k}": v for k, v in events.items() if k != "repeats"},
        **{f"wall_{k}": v for k, v in wall.items() if k != "repeats"},
        "repeats": repeats, "warmup": warmup,
    }


def main(argv: list[str] | None = None) -> int:
    """Time every configuration on every workload; exit 1 on any incorrect or errored row."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--profiles", nargs="+", default=["sparse", "uniform", "burst"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--tiles", nargs="+", default=["auto", "16", "32", "64", "256"])
    parser.add_argument("--lanes", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", default="runs/benchmarks/scheduling.json")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable: scheduling.py times GPU kernels.", file=sys.stderr)
        return 2
    document: dict[str, Any] = {
        "schema": "tapeforge.benchmark.scheduling/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "scope": "kernel: prepared execute on resident tensors; prepare timed separately",
        "arguments": vars(args),
        "rows": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    for size in args.sizes:
        for profile in args.profiles:
            for seed in args.seeds:
                trades, offsets_host, info = make_workload(size, profile, seed)
                expected = reference.aggregate_reference(trades, offsets_host)
                columns = {k: v.cuda() for k, v in host_columns(trades).items()}
                offsets = torch.from_numpy(offsets_host).cuda()
                for config in configurations(args.tiles, args.lanes):
                    row = measure(config, columns, offsets, expected, args.warmup, args.repeats)
                    row["workload"] = info.as_dict()
                    failed |= row["status"] != "ok"
                    document["rows"].append(row)
                    print(json.dumps({k: row[k] for k in row if k != "workload"}
                                     | {"profile": profile, "n_records": info.n_records}),
                          flush=True)
                    out.write_text(json.dumps(document, indent=2))
                del columns, offsets
                torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
