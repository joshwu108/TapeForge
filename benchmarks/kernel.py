"""Scope 1: kernel-only latency on already-resident tensors (TF-08).

Inputs are moved to the device *before* timing starts. This number answers
"how fast is the reduction", and must never be compared against a CPU
pipeline that also decompresses a ZIP (PRD section 6).

Implementations (``baselines`` in the config):

    tapeforge_triton   tapeforge.ops.aggregate on CUDA tensors, as a user
                       calls it: input validation (a pass over the columns
                       and one host sync), output allocation, kernel launch.
    tapeforge_triton_kernel_only
                       the same kernel launched into preallocated outputs,
                       validated once *outside* the timed region -- the
                       like-for-like comparison with torch_eager, which does
                       no validation either.
                       For both Triton rows, workloads with a bucket beyond
                       kernels.MAX_ONE_PASS_BUCKET are recorded as
                       "unsupported" -- the two-pass path does not exist yet.
    torch_eager        baselines.torch_index_add, eager PyTorch on the device.
    torch_compile      the same function under torch.compile; compilation
                       happens during warm-up and is not timed.
    numpy_reference    tapeforge.reference.aggregate_reference on the host
                       (CPU baseline; host-resident NumPy, no transfer). Its
                       correctness gate compares it with itself, so for this
                       row "correct" carries no information.

Every implementation must match the CPU reference before it is timed.
Without CUDA the script runs the CPU-capable implementations as a smoke test
and records nothing.

    python benchmarks/kernel.py --config benchmarks/configs/kernel_sweep.toml
    python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record   # quick run
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

from baselines import torch_index_add
from harness import TIMING_METHOD, record, time_callable
from workload import check_features, features_dict, host_columns, make_workload

from tapeforge import kernels, ops, reference


def _numpy_reference(trades: Any, offsets: Any) -> dict[str, Any]:
    return features_dict(reference.aggregate_reference(trades, offsets))


def _kernel_only(columns: dict[str, Any], offsets: Any) -> Callable[[], dict[str, Any]]:
    """Validate and allocate now; return a callable that only launches the kernel."""
    from tapeforge.validation import validate_columns

    summary = validate_columns(columns, offsets)
    block = kernels.select_block_size(summary.max_bucket_len)
    features = kernels.empty_features(summary.n_buckets, offsets.device)

    def launch() -> dict[str, Any]:
        kernels.launch_one_pass_unchecked(columns, offsets, features, block)
        return features

    return launch


def _implementations(
    names: list[str], device: str, trades: Any, offsets: Any
) -> dict[str, Callable[[], Any]]:
    import torch

    impls: dict[str, Callable[[], Any]] = {}
    columns = {k: v.to(device) for k, v in host_columns(trades).items()}
    dev_offsets = torch.from_numpy(offsets).to(device)
    for name in names:
        if name == "tapeforge_triton" and device == "cuda":
            impls[name] = partial(ops.aggregate, columns, dev_offsets)
        elif name == "tapeforge_triton_kernel_only" and device == "cuda":
            impls[name] = partial(_kernel_only, columns, dev_offsets)
        elif name == "torch_eager":
            impls[name] = partial(torch_index_add, columns, dev_offsets)
        elif name == "torch_compile":
            compiled = torch.compile(torch_index_add, dynamic=False)
            impls[name] = partial(compiled, columns, dev_offsets)
        elif name == "numpy_reference":
            impls[name] = partial(_numpy_reference, trades, offsets)
    return impls


def _measure(
    name: str,
    fn: Callable[[], Any],
    expected: reference.Features,
    warmup: int,
    repeats: int,
    synchronize: bool,
) -> dict[str, Any]:
    try:
        # kernel_only's entry is a setup step (validate + allocate, untimed)
        # that returns the bare launch; every other entry is timed directly.
        timed = fn() if name == "tapeforge_triton_kernel_only" else fn
        check_features(timed(), expected)
    except kernels.BucketTooLargeError as exc:
        return {"status": "unsupported", "reason": str(exc)}
    except AssertionError as exc:
        return {"status": "incorrect", "reason": str(exc)[:500]}
    except Exception as exc:  # noqa: BLE001 - e.g. torch.compile backend missing
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:500]}
    timings = time_callable(timed, warmup=warmup, repeats=repeats, synchronize=synchronize)
    return {"status": "ok", "correct": True, **timings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/kernel_sweep.toml")
    parser.add_argument("--sizes", type=int, nargs="+", help="override config record counts")
    parser.add_argument("--profiles", nargs="+", help="override config profiles")
    parser.add_argument("--seeds", type=int, nargs="+", help="override config seeds")
    parser.add_argument("--repeats", type=int, help="override config repeats")
    parser.add_argument("--no-record", action="store_true", help="print only, skip results.json")
    args = parser.parse_args(argv)
    config = tomllib.loads(Path(args.config).read_text())

    import torch

    on_cuda = torch.cuda.is_available()
    device = "cuda" if on_cuda else "cpu"
    should_record = on_cuda and not args.no_record
    if not on_cuda:
        print("CUDA unavailable: CPU smoke run of the CPU-capable baselines; nothing recorded.",
              file=sys.stderr)

    warmup, repeats = config["warmup"], args.repeats or config["repeats"]
    for size in args.sizes or config["sizes"]:
        for profile in args.profiles or config["profiles"]:
            for seed in args.seeds or config["seeds"]:
                trades, offsets, info = make_workload(
                    size, profile, seed, config["bucket_width_us"]
                )
                expected = reference.aggregate_reference(trades, offsets)
                impls = _implementations(config["baselines"], device, trades, offsets)
                for name in config["baselines"]:
                    if name not in impls:
                        print(f"skip {name}: needs CUDA", file=sys.stderr)
                        continue
                    host = name == "numpy_reference"
                    metrics = _measure(
                        name, impls[name], expected, warmup, repeats, on_cuda and not host
                    )
                    if metrics["status"] == "ok":
                        metrics["records_per_s"] = info.n_records / (metrics["p50_ms"] / 1e3)
                    run_config = {
                        "workload": info.as_dict(),
                        "device": "cpu" if host else device,
                        "warmup": warmup,
                        "repeats": repeats,
                        "timing_method": TIMING_METHOD,
                        "correctness": "exact ints, VWAP rtol/atol from contract, vs reference",
                    }
                    line = {"name": name, **run_config["workload"], **metrics}
                    print(json.dumps({k: v for k, v in line.items() if k != "reason"}))
                    if metrics.get("reason"):
                        print(f"  {name}: {metrics['status']}: {metrics['reason']}",
                              file=sys.stderr)
                    if should_record:
                        record("kernel", name, metrics, run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
