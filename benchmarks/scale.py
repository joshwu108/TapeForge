"""Correctness and memory of the feature engine at tens of millions of records.

For each generator profile the parent process generates one seeded workload
of about ``--records`` records, computes the vectorised CPU reference and
writes inputs and reference features to ``--work`` as ``.npy`` files. Each
execution path then runs in a fresh child process, so the CUDA caching
allocator and the host heap carry no history from another path into its
memory figures. A child loads the inputs (memory-mapped), copies them to the
device, runs its path, compares every feature with the reference and
reports:

    correct            exact integer features, VWAP within the contract
                       tolerance, identical dtypes
    invariants         independent of the reference: the event counts sum to
                       the records in the window; total and signed quantity
                       sum exactly to the column sums; valid == non-empty
    overflow_refused   the checked API raises OverflowError once the prices
                       are scaled past the int64 accumulation bound (auto
                       path only; nothing is launched)
    peak_allocated_bytes / peak_reserved_bytes
                       torch.cuda.max_memory_allocated / _reserved after a
                       reset, covering inputs, preparation and execution
    max_rss_bytes      the child's peak resident host memory
    prepare_ms, execute_p50/p99_ms
                       one preparation (validation, plan, metadata,
                       allocation); prepared execution under CUDA events

Paths: ``auto`` (``prepared.prepare`` with automatic dispatch), ``two_pass``,
any other path name ``prepared.prepare`` accepts, and ``pipelined`` or
``pipelined_int32`` (``tapeforge.pipelined.OverlappedAggregator`` from host
memory, the latter with ``int32_transfer=True``, which
holds only bounded batches on the device). A path the installed library does
not provide is reported ``not_available``.

``--zero-fraction`` sets that share of quantities to zero (seeded) before the
reference runs, so zero-quantity records occur at scale. Before allocating,
the parent estimates host and device memory and skips a workload, with the
estimate, when it would not fit. Requested and generated record counts are
both recorded; synthetic tapes are labelled as such.

Usage:
    python benchmarks/scale.py --records 48000000 --profiles uniform burst \
        --paths auto two_pass pipelined --out runs/benchmarks/scale.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from harness import capture_environment, snapshot_metadata
from workload import make_workload

from tapeforge import contract, reference

FEATURES = contract.FEATURE_NAMES
HOST_BYTES_PER_RECORD = 120  # generation temporaries plus columns, measured locally
DEVICE_BYTES_PER_RECORD = 17  # int64 price, int64 quantity, bool maker
OUTPUT_BYTES_PER_BUCKET = 33  # three int64, one float64, one bool



# OverlappedAggregator options of each host-to-host path name.
PIPELINE_OPTIONS: dict[str, dict[str, Any]] = {
    "pipelined": {},
    "pipelined_int32": {"int32_transfer": True},
}


def estimate(n_records: int, n_buckets: int) -> dict[str, int]:
    """Conservative host and device byte estimates for one workload and one path."""
    device = (DEVICE_BYTES_PER_RECORD * n_records
              + 8 * (n_buckets + 1) * 3            # offsets, lengths and schedule temporaries
              + OUTPUT_BYTES_PER_BUCKET * n_buckets)
    host = HOST_BYTES_PER_RECORD * n_records + 3 * OUTPUT_BYTES_PER_BUCKET * n_buckets
    return {"host_bytes": host, "device_bytes": device}


def _available_host_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        return 0


def prepare_workload(work: Path, n_records: int, profile: str, seed: int,
                     zero_fraction: float) -> dict[str, Any]:
    """Generate a workload and its reference features into ``work``; return its description."""
    started = time.perf_counter()
    trades, offsets, info = make_workload(n_records, profile, seed)
    generate_seconds = time.perf_counter() - started
    if zero_fraction > 0:
        rng = np.random.default_rng(seed + 1_000_003)
        trades["quantity"][rng.random(trades.size) < zero_fraction] = 0
    work.mkdir(parents=True, exist_ok=True)
    for name in ("price", "quantity", "buyer_is_maker"):
        np.save(work / f"{name}.npy", np.ascontiguousarray(trades[name]))
    np.save(work / "offsets.npy", offsets)
    started = time.perf_counter()
    expected = reference.aggregate_reference(trades, offsets)
    reference_seconds = time.perf_counter() - started
    for name in FEATURES:
        np.save(work / f"expected_{name}.npy", getattr(expected, name))
    window = slice(int(offsets[0]), int(offsets[-1]))
    sign = np.where(trades["buyer_is_maker"][window], -1, 1)
    quantity = trades["quantity"][window].astype(np.int64)
    sums = {
        "records_in_window": int(offsets[-1] - offsets[0]),
        "quantity_sum": int(quantity.sum()),
        "signed_sum": int((quantity * sign).sum()),
        "non_empty_buckets": int((np.diff(offsets) > 0).sum()),
        "zero_quantity_records": int((quantity == 0).sum()),
    }
    (work / "sums.json").write_text(json.dumps(sums))
    return {
        "workload": info.as_dict(),
        "data_source": "synthetic (tapeforge.generate); no market data",
        "zero_fraction": zero_fraction,
        "generate_seconds": generate_seconds,
        "reference_seconds": reference_seconds,
        "invariant_sums": sums,
    }


def _run_path(work: Path, path: str, repeats: int) -> dict[str, Any]:
    """Child-process body: run one path on the workload in ``work`` and check it."""
    import torch

    columns_host = {n: np.load(work / f"{n}.npy", mmap_mode="r")
                    for n in ("price", "quantity", "buyer_is_maker")}
    offsets_host = np.load(work / "offsets.npy")
    torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {"path": path}
    if path in PIPELINE_OPTIONS:
        from tapeforge import pipelined

        columns = {n: np.ascontiguousarray(c) for n, c in columns_host.items()}
        started = time.perf_counter()
        aggregator = pipelined.OverlappedAggregator(columns, offsets_host,
                                                    **PIPELINE_OPTIONS[path])
        result["prepare_ms"] = (time.perf_counter() - started) * 1e3
        got = aggregator.run()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            aggregator.run()
            samples.append((time.perf_counter() - started) * 1e3)
        result["timing_method"] = "wall clock around run(), host memory to host memory"
        result["batches"] = aggregator.stats.n_batches
        result["pinned_bytes"] = aggregator.stats.pinned_bytes
        result["int32_transfer"] = aggregator.stats.int32_transfer
        outputs = {name: (lambda n=name: got[n]) for name in FEATURES}
    else:
        from tapeforge import ops, prepared

        columns = {n: torch.from_numpy(np.ascontiguousarray(c)).cuda()
                   for n, c in columns_host.items()}
        offsets = torch.from_numpy(offsets_host).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            prep = prepared.prepare(columns, offsets, path=path)  # type: ignore[arg-type]
        except ValueError as exc:
            if "path must be" in str(exc):
                return {"path": path, "status": "not_available", "reason": str(exc)}
            raise
        torch.cuda.synchronize()
        result["prepare_ms"] = (time.perf_counter() - started) * 1e3
        result["dispatch"] = {"path": prep.path, "block_size": prep.plan.block_size,
                              "group_size": getattr(prep.plan, "group_size", None),
                              "n_chunks": prep.n_chunks}
        prep.execute()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            prep.execute()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        result["timing_method"] = "CUDA events around prepared.execute()"
        # Peak memory is read before the overflow probe allocates its scaled copy.
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        if path == "auto":
            scaled = dict(columns, price=torch.full_like(columns["price"], contract.INT64_MAX // 2))
            try:
                ops.aggregate(scaled, offsets)
                result["overflow_refused"] = False
            except OverflowError:
                result["overflow_refused"] = True
            del scaled
        outputs = {name: (lambda n=name: prep.features[n].cpu().numpy()) for name in FEATURES}

    samples.sort()
    result["execute_p50_ms"] = samples[len(samples) // 2]
    result["execute_p99_ms"] = samples[min(len(samples) - 1, int(len(samples) * 0.99))]
    result["execute_min_ms"] = samples[0]
    result["repeats"] = repeats
    result.setdefault("peak_allocated_bytes", int(torch.cuda.max_memory_allocated()))
    result.setdefault("peak_reserved_bytes", int(torch.cuda.max_memory_reserved()))

    sums = json.loads((work / "sums.json").read_text())
    mismatches = []
    observed: dict[str, int] = {}
    for name in FEATURES:  # One feature at a time keeps host memory bounded.
        got_values = outputs[name]()
        expected = np.load(work / f"expected_{name}.npy", mmap_mode="r")
        if got_values.dtype != expected.dtype:
            mismatches.append(f"{name}: dtype {got_values.dtype} != {expected.dtype}")
        elif name == "vwap":
            close = np.isclose(got_values, expected, rtol=contract.VWAP_RTOL,
                               atol=contract.VWAP_ATOL)
            if not close.all():
                mismatches.append(f"vwap: {int((~close).sum())} buckets outside tolerance")
            result["vwap_max_abs_diff"] = float(np.abs(got_values - expected).max())
        elif not np.array_equal(got_values, expected):
            mismatches.append(f"{name}: {int((got_values != expected).sum())} buckets differ")
        if name == "event_count":
            observed["records_in_window"] = int(got_values.sum())
        elif name == "total_quantity":
            observed["quantity_sum"] = int(got_values.sum())
        elif name == "signed_quantity":
            observed["signed_sum"] = int(got_values.sum())
        elif name == "valid":
            observed["non_empty_buckets"] = int(got_values.sum())
        del got_values
    result["correct"] = not mismatches
    result["mismatches"] = mismatches
    result["invariants"] = all(observed[k] == sums[k] for k in observed)
    result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024)
    result["status"] = "ok" if result["correct"] and result["invariants"] else "incorrect"
    return result


def main(argv: list[str] | None = None) -> int:
    """Run every profile and path; write one JSON document. Exit 1 on any incorrect row."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=48_000_000)
    parser.add_argument("--profiles", nargs="+", default=["uniform", "burst", "sparse"])
    parser.add_argument("--paths", nargs="+", default=["auto", "two_pass", "pipelined"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-fraction", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--work", default="/tmp/tapeforge-scale")
    parser.add_argument("--out", default="runs/benchmarks/scale.json")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(_run_path(Path(args.work), args.child, args.repeats)))
        return 0

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable: scale.py checks the GPU paths.", file=sys.stderr)
        return 2
    free_device, total_device = torch.cuda.mem_get_info()
    document: dict[str, Any] = {
        "schema": "tapeforge.benchmark.scale/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "device_total_bytes": total_device,
        "arguments": {k: v for k, v in vars(args).items() if k != "child"},
        "workloads": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    for profile in args.profiles:
        from workload import RECORDS_PER_BUCKET

        n_buckets = max(1, round(args.records / RECORDS_PER_BUCKET[profile]))
        need = estimate(args.records, n_buckets)
        entry: dict[str, Any] = {"profile": profile, "requested_records": args.records,
                                 "estimate": need, "free_device_bytes": free_device,
                                 "available_host_bytes": _available_host_bytes()}
        host_limit = entry["available_host_bytes"]
        if need["device_bytes"] > 0.8 * free_device or (
                host_limit and need["host_bytes"] > host_limit):
            entry["status"] = "skipped_memory_estimate"
            document["workloads"].append(entry)
            out.write_text(json.dumps(document, indent=2))
            continue
        work = Path(args.work) / profile
        entry.update(prepare_workload(work, args.records, profile, args.seed, args.zero_fraction))
        entry["paths"] = []
        for path in args.paths:
            child = subprocess.run(
                [sys.executable, __file__, "--child", path, "--work", str(work),
                 "--repeats", str(args.repeats)],
                capture_output=True, text=True,
            )
            if child.returncode != 0:
                row = {"path": path, "status": "error", "returncode": child.returncode,
                       "stderr": child.stderr[-2000:]}
            else:
                row = json.loads(child.stdout.strip().splitlines()[-1])
            failed |= row.get("status") in ("incorrect", "error")
            entry["paths"].append(row)
            print(json.dumps({"profile": profile, **{k: v for k, v in row.items()
                                                     if k not in ("stderr",)}}), flush=True)
            out.write_text(json.dumps(document | {"workloads": [*document["workloads"], entry]},
                                      indent=2))
        document["workloads"].append(entry)
        out.write_text(json.dumps(document, indent=2))
        for file in work.glob("*.npy"):
            file.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
