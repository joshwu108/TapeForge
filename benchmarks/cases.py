"""Seeded adversarial cases: every GPU path against the plain-loop oracle.

A *case* is one generated input -- a trade array and its offsets -- built
from one seed (``--seed-base + i``). Bucket lengths are drawn from a mixture
that concentrates on the edges the kernels have to get right: empty runs,
lengths at and around powers of two, the one-pass limit
(``kernels.MAX_ONE_PASS_BUCKET`` +- 1), chunk multiples +- 1, and a few long
buckets. Each case also draws whether records lie before and after the
window, a share of zero-quantity records, and a value scale that puts the
int64 accumulation bound between 1 % and 100 % used. The oracle is
``reference.aggregate_reference_naive`` (Python integers, one bucket at a
time).

Every case runs through every requested path: ``auto`` (``ops.aggregate``,
the checked API), ``two_pass`` (prepared, forced), ``pipelined``
(``tapeforge.pipelined.aggregate_host`` with a random batch budget and slot
count) and any other ``prepared.prepare`` path name, with path options
drawn from the seed. A path the installed library does not provide is
reported ``not_available`` for the whole run. The output lists the seeds,
the counts per path and the seeds of any failing case, so a failure can be
regenerated with ``--seed-base <seed> --cases 1``.

Usage:
    python benchmarks/cases.py --cases 10000 --out runs/benchmarks/cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from harness import capture_environment, snapshot_metadata

from tapeforge import contract, kernels, reference

FEATURES = contract.FEATURE_NAMES


def make_case(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(trades, offsets, description)`` for one seed."""
    rng = np.random.default_rng(seed)
    n_buckets = int(rng.choice([1, 2, 3, 31, 64, 129, 500, 1500]))
    edges = [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
             257, 1023, 1024, 1025]
    long_edges = [kernels.MAX_ONE_PASS_BUCKET - 1, kernels.MAX_ONE_PASS_BUCKET,
                  kernels.MAX_ONE_PASS_BUCKET + 1, 2 * kernels.CHUNK_SIZE + 1,
                  5 * kernels.CHUNK_SIZE - 1, 3 * kernels.MAX_ONE_PASS_BUCKET + 17]
    kind = rng.random(n_buckets)
    lengths = np.where(
        kind < 0.35, 0,
        np.where(kind < 0.9, rng.choice(edges, n_buckets), rng.poisson(40, n_buckets)),
    ).astype(np.int64)
    n_long = int(rng.integers(0, 3)) if rng.random() < 0.3 else 0
    for index in rng.choice(n_buckets, size=min(n_long, n_buckets), replace=False):
        lengths[index] = int(rng.choice(long_edges))
    head = int(rng.integers(0, 4)) if rng.random() < 0.5 else 0
    tail = int(rng.integers(0, 4)) if rng.random() < 0.5 else 0
    n = head + int(lengths.sum()) + tail

    # Values: the bound longest * max_price * max_quantity uses 1 % to 100 % of INT64_MAX.
    longest = max(1, int(lengths.max()))
    budget = contract.INT64_MAX * float(rng.uniform(0.01, 1.0)) / longest
    max_price = max(1, int(budget ** float(rng.uniform(0.3, 0.7))))
    max_quantity = max(1, int(budget / max_price))
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = np.arange(n)
    trades["price"] = rng.integers(0, max_price, n, endpoint=True)
    trades["quantity"] = rng.integers(0, max_quantity, n, endpoint=True)
    zero_share = float(rng.choice([0.0, 0.05, 0.5, 1.0]))
    trades["quantity"][rng.random(n) < zero_share] = 0
    trades["buyer_is_maker"] = rng.random(n) < float(rng.uniform(0, 1))
    trades["agg_trade_id"] = np.arange(n)
    offsets = np.concatenate([[head], head + np.cumsum(lengths)]).astype(np.int64)
    description = {"seed": seed, "n_buckets": n_buckets, "n_records": n,
                   "max_bucket_len": int(lengths.max()), "zero_share": zero_share,
                   "head": head, "tail": tail}
    return trades, offsets, description


def _compare(got: dict[str, Any], expected: reference.Features) -> str | None:
    for name in FEATURES:
        value = got[name]
        actual = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        target = getattr(expected, name)
        if actual.dtype != target.dtype:
            return f"{name}: dtype {actual.dtype} != {target.dtype}"
        if name == "vwap":
            if not np.allclose(actual, target, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL):
                return "vwap outside tolerance"
        elif not np.array_equal(actual, target):
            return f"{name} differs"
    return None


def run_path(path: str, trades: np.ndarray, offsets: np.ndarray, rng: np.random.Generator,
             torch: Any) -> dict[str, Any]:
    """Run one path on one case; return its features and the options used."""
    from tapeforge import ops, pipelined, prepared

    if path == "pipelined":
        options = {"batch_records": int(rng.choice([1, 7, 64, 1000, 1 << 20])),
                   "slots": int(rng.integers(1, 4))}
        return {"features": pipelined.aggregate_host(trades, offsets, **options),
                "options": options}
    columns = {n: torch.from_numpy(trades[n].copy()).cuda()
               for n in ("price", "quantity", "buyer_is_maker")}
    device_offsets = torch.from_numpy(offsets).cuda()
    if path == "auto":
        return {"features": ops.aggregate(columns, device_offsets), "options": {}}
    options: dict[str, Any] = {}
    if path == "grouped":
        tiles = [None, 2, 4, 16, 64, 128, 1024]     # None: the library's own tile choice
        options = {"block_size": tiles[int(rng.integers(len(tiles)))],
                   "group_lanes": int(rng.choice([2, 64, 1024, 4096]))}
    prep = prepared.prepare(columns, device_offsets, path=path, **options)
    return {"features": prep.execute_owned(), "options": options}


def main(argv: list[str] | None = None) -> int:
    """Check every case on every path; write a summary. Exit 1 when any case fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--paths", nargs="+", default=["auto", "two_pass", "pipelined"])
    parser.add_argument("--out", default="runs/benchmarks/cases.json")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable: cases.py checks the GPU paths.", file=sys.stderr)
        return 2
    counts = {path: {"passed": 0, "failed": 0} for path in args.paths}
    unavailable: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    totals = {"records": 0, "buckets": 0, "two_pass_cases": 0, "empty_window_cases": 0}
    started = time.perf_counter()
    for i in range(args.cases):
        seed = args.seed_base + i
        trades, offsets, description = make_case(seed)
        expected = reference.aggregate_reference_naive(trades, offsets)
        totals["records"] += description["n_records"]
        totals["buckets"] += description["n_buckets"]
        totals["two_pass_cases"] += description["max_bucket_len"] > kernels.MAX_ONE_PASS_BUCKET
        totals["empty_window_cases"] += description["max_bucket_len"] == 0
        rng = np.random.default_rng(seed + 7_777_777)
        for path in args.paths:
            if path in unavailable:
                continue
            try:
                outcome = run_path(path, trades, offsets, rng, torch)
                problem = _compare(outcome["features"], expected)
            except ValueError as exc:
                if "path must be" in str(exc):
                    unavailable[path] = str(exc)
                    continue
                outcome, problem = {"options": {}}, f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                outcome, problem = {"options": {}}, f"{type(exc).__name__}: {exc}"
            if problem is None:
                counts[path]["passed"] += 1
            else:
                counts[path]["failed"] += 1
                if len(failures) < 50:
                    failures.append({**description, "path": path,
                                     "options": outcome.get("options"), "problem": problem})
    document = {
        "schema": "tapeforge.benchmark.cases/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "case_definition": "one seeded (trades, offsets) input from make_case(seed); "
                           "oracle reference.aggregate_reference_naive",
        "seeds": [args.seed_base, args.seed_base + args.cases - 1],
        "cases": args.cases,
        "totals": totals,
        "counts": {path: counts[path] for path in args.paths if path not in unavailable},
        "not_available": unavailable,
        "failures": failures,
        "seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2))
    print(json.dumps({k: document[k] for k in ("cases", "totals", "counts", "not_available",
                                                "seconds")}))
    return 1 if any(c["failed"] for c in counts.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
