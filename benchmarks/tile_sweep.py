"""Grouped-path tile sweep over many bucket-length distributions (data for the tile model).

``kernels.choose_grouped_tile`` picks the grouped tile from the bucket
lengths with a cost model. Checking or refitting that model needs measured
times on more than the three generator profiles, so this script builds
workloads from explicit bucket-length distributions, at several sizes and
seeds, and times ``PreparedAggregation.execute()`` (kernel scope: launches
into prepared buffers on resident inputs) for every power-of-two tile from
``kernels.MIN_GROUPED_TILE`` to ``kernels.MAX_GROUPED_TILE`` plus the
library's own choice. Every configuration is gated against the CPU
reference before it is timed.

Length families (``n_buckets`` follows from the record target and the mean):

    poisson/m          Poisson(m) records per bucket
    tail/m/p           Poisson(m), with a share p of buckets drawn from a
                       log-normal of median 400 records (long buckets)
    profile/<name>     benchmarks.workload.make_workload profiles

Seeds are split by the caller into a fitting set and a held-out set; the
row records the seed, so the split is made in the analysis, not here.
Records' values are uniform in small ranges well inside the int64 bound;
the timing depends on lengths, not values.

Usage:
    python benchmarks/tile_sweep.py --records 2000000 8000000 --seeds 21 22 23 \\
        --out runs/benchmarks/tile_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from harness import capture_environment, snapshot_metadata, time_callable
from workload import check_features, host_columns, make_workload

from tapeforge import contract, kernels, prepared, reference

POISSON_MEANS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
TAIL_FAMILIES = ((1.0, 0.001), (1.0, 0.01), (8.0, 0.001), (8.0, 0.01))
PROFILES = ("sparse", "uniform", "burst")


def lengths_for(family: str, records: int, rng: np.random.Generator) -> np.ndarray:
    """Return int64 bucket lengths of ``family`` summing to about ``records``."""
    kind, *params = family.split("/")
    mean = float(params[0])
    share = float(params[1]) if kind == "tail" else 0.0
    expected = mean * (1 - share) + share * 400 * np.exp(0.5)  # log-normal(sigma=1) mean
    n_buckets = max(1, int(records / expected))
    lengths = rng.poisson(mean, n_buckets)
    if share:
        long_rows = rng.random(n_buckets) < share
        lengths[long_rows] = np.maximum(
            1, rng.lognormal(np.log(400), 1.0, int(long_rows.sum())).astype(np.int64))
    return lengths.astype(np.int64)


def trades_for(lengths: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return a trade array and offsets with the given bucket lengths."""
    n = int(lengths.sum())
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = np.arange(n)
    trades["price"] = rng.integers(1, 1_000_000, n)
    trades["quantity"] = rng.integers(0, 1_000, n)
    trades["buyer_is_maker"] = rng.random(n) < 0.5
    trades["agg_trade_id"] = np.arange(n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return trades, offsets


def describe(lengths: np.ndarray, tiles: list[int]) -> dict[str, Any]:
    """Return the length statistics a tile model can use."""
    chunks = (lengths + kernels.CHUNK_SIZE - 1) // kernels.CHUNK_SIZE
    return {
        "n_buckets": int(lengths.size), "n_records": int(lengths.sum()),
        "mean_len": float(lengths.mean()), "max_len": int(lengths.max()),
        "p50": float(np.percentile(lengths, 50)), "p99": float(np.percentile(lengths, 99)),
        "empty_share": float((lengths == 0).mean()),
        "per_tile": {
            str(t): {"n_long": int((lengths > t).sum()),
                     "long_records": int(lengths[lengths > t].sum()),
                     "long_chunks": int(chunks[lengths > t].sum())}
            for t in tiles
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Sweep every tile on every workload; exit 1 when any configuration is incorrect."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, nargs="+", default=[2_000_000, 8_000_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22])
    parser.add_argument("--group-lanes", type=int, default=kernels.GROUP_LANES)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--out", default="runs/benchmarks/tile_sweep.json")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable: tile_sweep.py times GPU kernels.", file=sys.stderr)
        return 2
    tiles = []
    tile = kernels.MIN_GROUPED_TILE
    while tile <= kernels.MAX_GROUPED_TILE:
        tiles.append(tile)
        tile *= 2
    families = args.families or (
        [f"poisson/{m:g}" for m in POISSON_MEANS]
        + [f"tail/{m:g}/{p:g}" for m, p in TAIL_FAMILIES]
        + [f"profile/{p}" for p in PROFILES]
    )
    document: dict[str, Any] = {
        "schema": "tapeforge.benchmark.tile-sweep/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "scope": "kernel: prepared execute on resident tensors (grouped path); "
                 "prepare timed separately; CUDA events p50",
        "arguments": vars(args), "tiles": tiles, "rows": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    started = time.perf_counter()
    for records in args.records:
        for family in families:
            for seed in args.seeds:
                rng = np.random.default_rng([seed, records])
                if family.startswith("profile/"):
                    trades, offsets_host, _ = make_workload(records, family.split("/")[1], seed)
                    lengths = np.diff(offsets_host)
                else:
                    lengths = lengths_for(family, records, rng)
                    trades, offsets_host = trades_for(lengths, rng)
                expected = reference.aggregate_reference(trades, offsets_host)
                columns = {k: v.cuda() for k, v in host_columns(trades).items()}
                offsets = torch.from_numpy(offsets_host).cuda()
                row: dict[str, Any] = {"family": family, "records_target": records,
                                       "seed": seed, "lengths": describe(lengths, tiles),
                                       "times": {}}
                for choice in [None, *tiles]:
                    key = "auto" if choice is None else str(choice)
                    try:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        ready = prepared.prepare(columns, offsets, path="grouped",
                                                 block_size=choice,
                                                 group_lanes=args.group_lanes)
                        torch.cuda.synchronize()
                        prepare_ms = (time.perf_counter() - t0) * 1e3
                        check_features(ready.execute(), expected)
                        timing = time_callable(ready.execute, warmup=3, repeats=args.repeats,
                                               method="cuda_events")
                        row["times"][key] = {"p50_ms": timing["p50_ms"],
                                             "min_ms": timing["min_ms"],
                                             "prepare_ms": prepare_ms,
                                             "tile": ready.plan.block_size}
                        del ready
                    except AssertionError as exc:
                        row["times"][key] = {"status": "incorrect", "reason": str(exc)[:200]}
                        failed = True
                    except Exception as exc:  # noqa: BLE001 - recorded per configuration
                        row["times"][key] = {"status": "error",
                                             "reason": f"{type(exc).__name__}: {exc}"[:200]}
                        failed = True
                document["rows"].append(row)
                best = min((v["p50_ms"], k) for k, v in row["times"].items()
                           if k != "auto" and "p50_ms" in v)
                print(json.dumps({"family": family, "records": records, "seed": seed,
                                  "best": best, "auto": row["times"].get("auto")}), flush=True)
                out.write_text(json.dumps(document, indent=2))
                del columns, offsets
                torch.cuda.empty_cache()
    document["seconds"] = time.perf_counter() - started
    out.write_text(json.dumps(document, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
