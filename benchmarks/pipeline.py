"""Scope 2: host-to-device transfer plus features.

The benchmark starts from host-resident column tensors (the column extraction
from the structured array is done once, untimed) and times, per variant:

    h2d_ms         copy the three input columns and offsets to the GPU
    compute_ms     tapeforge.ops.aggregate on the resident copies
    d2h_ms         copy the five feature tensors back to the host
    end_to_end_ms  all three in one timed call

Each component is a separately timed callable, so the parts need not sum
exactly to ``end_to_end_ms``; the gap is launch and synchronization overhead
that only the end-to-end number sees.

Variants (``variants`` in the config):

    pageable_sync   ordinary host memory, blocking copies
    pinned_sync     page-locked host memory, non_blocking copies + synchronize
    pinned_streams  NOT IMPLEMENTED: overlapping the copy of one chunk with
                    the compute of the previous one on separate CUDA streams;
                    reported as skipped until it exists

``compute_ms`` is ``ops.aggregate`` as a caller uses it (validation,
planning, allocation, launches), on whichever kernel path the workload's
longest bucket selects; the path is recorded in the workload info.
``numpy_reference`` (CPU features on the host, no transfer) is recorded as
the end-to-end CPU baseline. Every variant is checked against the reference
before it is timed. CUDA is required; the exit status is 2 without it and 0
otherwise.

Usage:
    python benchmarks/pipeline.py --config benchmarks/configs/pipeline.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from harness import TIMING_METHOD, record, time_callable
from workload import check_features, features_dict, host_columns, make_workload

from tapeforge import ops, reference

# Variants with an implementation; any other configured variant is skipped.
IMPLEMENTED_VARIANTS = ("pageable_sync", "pinned_sync")


def _numpy_reference(trades: Any, offsets: Any) -> dict[str, Any]:
    """Return the host reference features in the dict shape ``ops.aggregate`` returns."""
    return features_dict(reference.aggregate_reference(trades, offsets))


def _variant_callables(
    trades: Any, offsets: Any, pinned: bool
) -> dict[str, Callable[[], Any]]:
    """Build the ``h2d``, ``compute``, ``d2h`` and ``end_to_end`` callables for one variant.

    Host columns and offsets are prepared (and pinned when ``pinned`` is
    set) once, untimed. One untimed transfer and aggregation provide the
    resident inputs for ``compute`` and the resident features for ``d2h``.
    With ``pinned`` the copies are ``non_blocking``; the timing harness
    synchronises around each call.
    """
    import torch

    columns = host_columns(trades, pin=pinned)
    host_offsets = torch.from_numpy(offsets)
    host_offsets = host_offsets.pin_memory() if pinned else host_offsets

    def h2d() -> tuple[dict[str, Any], Any]:
        dev = {k: v.to("cuda", non_blocking=pinned) for k, v in columns.items()}
        return dev, host_offsets.to("cuda", non_blocking=pinned)

    resident_columns, resident_offsets = h2d()
    resident_features = ops.aggregate(resident_columns, resident_offsets)

    def end_to_end() -> dict[str, Any]:
        dev, dev_offsets = h2d()
        return {k: v.cpu() for k, v in ops.aggregate(dev, dev_offsets).items()}

    return {
        "h2d": h2d,
        "compute": lambda: ops.aggregate(resident_columns, resident_offsets),
        "d2h": lambda: {k: v.cpu() for k, v in resident_features.items()},
        "end_to_end": end_to_end,
    }


def _measure_variant(
    trades: Any, offsets: Any, pinned: bool, expected: reference.Features, warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Gate one variant's end-to-end output against the reference, then time each part.

    Returns:
        ``{"status": "incorrect", "reason": ...}`` when the gate fails;
        otherwise ``status`` ``"ok"``, ``correct`` ``True`` and
        ``<part>_p50_ms`` / ``<part>_p95_ms`` for each of the four parts.
    """
    try:
        parts = _variant_callables(trades, offsets, pinned)
        check_features(parts["end_to_end"](), expected)
    except AssertionError as exc:
        return {"status": "incorrect", "reason": str(exc)[:500]}
    metrics: dict[str, Any] = {"status": "ok", "correct": True}
    for part, fn in parts.items():
        timing = time_callable(fn, warmup=warmup, repeats=repeats, synchronize=True)
        metrics[f"{part}_p50_ms"] = timing["p50_ms"]
        metrics[f"{part}_p95_ms"] = timing["p95_ms"]
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Time every configured variant on every workload and print one JSON line per result.

    Unimplemented variants are skipped with a message on stderr. Results are
    recorded unless ``--no-record`` is given.

    Returns:
        2 when CUDA is unavailable, otherwise 0.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/pipeline.toml")
    parser.add_argument("--sizes", type=int, nargs="+", help="override config record counts")
    parser.add_argument("--profiles", nargs="+", help="override config profiles")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-record", action="store_true", help="print only, skip results.json")
    args = parser.parse_args(argv)
    config = tomllib.loads(Path(args.config).read_text())

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable: the pipeline scope measures host-to-device transfer and "
              "needs an NVIDIA GPU.", file=sys.stderr)
        return 2

    warmup, repeats = config["warmup"], config["repeats"]
    for size in args.sizes or config["sizes"]:
        for profile in args.profiles or config["profiles"]:
            trades, offsets, info = make_workload(
                size, profile, args.seed, config["bucket_width_us"]
            )
            expected = reference.aggregate_reference(trades, offsets)
            results: dict[str, dict[str, Any]] = {}
            for variant in config["variants"]:
                if variant not in IMPLEMENTED_VARIANTS:
                    print(f"skip {variant}: not implemented", file=sys.stderr)
                    continue
                results[variant] = _measure_variant(
                    trades, offsets, variant.startswith("pinned"), expected, warmup, repeats
                )
            cpu = time_callable(
                partial(_numpy_reference, trades, offsets),
                warmup=warmup,
                repeats=repeats,
            )
            results["numpy_reference"] = {"status": "ok", "end_to_end_p50_ms": cpu["p50_ms"],
                                          "end_to_end_p95_ms": cpu["p95_ms"]}
            for name, metrics in results.items():
                if metrics["status"] == "ok":
                    metrics["records_per_s"] = info.n_records / (metrics["end_to_end_p50_ms"] / 1e3)
                    metrics["windows_per_s"] = info.n_buckets / (metrics["end_to_end_p50_ms"] / 1e3)
                run_config = {
                    "workload": info.as_dict(),
                    "variant": name,
                    "warmup": warmup,
                    "repeats": repeats,
                    "timing_method": TIMING_METHOD,
                }
                print(json.dumps({"name": name, **info.as_dict(), **metrics}))
                if not args.no_record:
                    record("pipeline", name, metrics, run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
