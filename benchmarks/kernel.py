"""Scope 1: kernel-only latency on already-resident tensors (TF-08).

Inputs are moved to the device *before* timing starts. This number answers
"how fast is the reduction", and must never be compared against a CPU
pipeline that also decompresses a ZIP (PRD section 6).

Rows (``baselines`` in the config). Every timed region is a complete Python
call, so each row includes Python dispatch and kernel-launch overhead; what
differs is which host-side work sits inside the region:

  Feature rows -- produce the five features and pass the correctness gate
  before they are timed:

    tapeforge_triton         tapeforge.ops.aggregate on CUDA tensors, as a
                             user calls it: validation (one host sync),
                             dispatch planning, output allocation, on the
                             two-pass path also chunk-metadata construction
                             (one more host sync) and scratch allocation,
                             then the kernel launch(es).
    tapeforge_prepared       tapeforge.prepared.PreparedAggregation.execute():
                             validation, planning, metadata and allocation
                             done once outside the timed region; the region
                             holds only the launch(es) into reused buffers.
                             No host sync inside.
    tapeforge_triton_kernel_only
                             the unchecked launcher(s) on the prepared buffers
                             (kernels.launch_one_pass_unchecked, or
                             launch_two_pass_unchecked). Same work as
                             tapeforge_prepared minus one method call; kept
                             as the like-for-like row against torch_eager,
                             which does no validation either.
    tapeforge_pass1_only     two-pass workloads: pass 1 alone (partials).
    tapeforge_pass2_only     two-pass workloads: pass 2 alone, re-combining
                             the partials the gate's pass 1 wrote. Both are
                             components of tapeforge_prepared and overlap
                             with it; do not add them to it.
    tapeforge_two_pass_forced
                             one-pass workloads only: the two-pass path
                             forced on the same input (prepared, launches
                             only), so both paths are measured on identical
                             data. "not_applicable" where auto dispatch is
                             already two-pass.
    torch_eager              baselines.torch_index_add, eager PyTorch on the
                             device; no validation, no host sync.
    torch_compile            the same function under torch.compile;
                             compilation happens during warm-up and is not
                             timed.
    numpy_reference          tapeforge.reference.aggregate_reference on the
                             host (CPU baseline; host-resident NumPy, no
                             transfer). Its correctness gate compares it
                             with itself, so for this row "correct" carries
                             no information.

  Stage rows -- preparation costs, timed on their own, no features, no
  correctness gate ("correct": null). They overlap with tapeforge_triton
  and must not be summed with it or with each other as if independent:

    tapeforge_stage_validate         validation.validate_columns (one host sync)
    tapeforge_stage_chunk_metadata   kernels.build_chunk_metadata (one host sync)
    tapeforge_stage_allocate         output + pass-1 scratch allocation through
                                     the caching allocator (cache hits after
                                     warm-up: the steady-state cost, not a
                                     cold cudaMalloc)
    tapeforge_stage_prepare          prepared.prepare end to end

Timing method: ``--timing wall`` (default; perf_counter with
torch.cuda.synchronize before and after each call) or ``--timing
cuda_events`` (CUDA event pairs on the current stream). See
``harness.TIMING_DESCRIPTIONS`` for what each includes. Each result row
records its method, what the timed region includes, the dispatch path, the
chunk count, the bucket-length distribution, warm-up and repeat counts, the
environment and the snapshot/commit.

Without CUDA the script runs the CPU-capable rows as a smoke test and
records nothing. No speedup is computed here; the rows are the evidence.

    python benchmarks/kernel.py --config benchmarks/configs/kernel_sweep.toml
    python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record   # quick run
    python benchmarks/kernel.py --timing cuda_events --sizes 1000000 --no-record
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from baselines import torch_index_add
from harness import TIMING_DESCRIPTIONS, TIMING_METHODS, record, time_callable
from workload import WorkloadInfo, check_features, features_dict, host_columns, make_workload

from tapeforge import kernels, ops, reference

FEATURE_ROWS = (
    "tapeforge_triton",
    "tapeforge_prepared",
    "tapeforge_triton_kernel_only",
    "tapeforge_pass1_only",
    "tapeforge_pass2_only",
    "tapeforge_two_pass_forced",
    "torch_eager",
    "torch_compile",
    "numpy_reference",
)
STAGE_ROWS = (
    "tapeforge_stage_validate",
    "tapeforge_stage_chunk_metadata",
    "tapeforge_stage_allocate",
    "tapeforge_stage_prepare",
)
ALL_ROWS = FEATURE_ROWS + STAGE_ROWS
CUDA_ONLY_ROWS = tuple(name for name in ALL_ROWS if name.startswith("tapeforge_"))


@dataclass(frozen=True)
class Row:
    """One timed implementation and what its timed region includes."""

    fn: Callable[[], Any]
    includes: str
    gate: Callable[[], Any] | None  # produces features for the correctness gate; None = stage row
    path: str | None = None
    n_chunks: int | None = None
    host: bool = False  # runs on the host; never synchronize


def _numpy_reference(trades: Any, offsets: Any) -> dict[str, Any]:
    return features_dict(reference.aggregate_reference(trades, offsets))


def _raise(reason: str) -> Callable[[], Any]:
    def fail() -> None:
        raise RuntimeError(reason)

    return fail


def _triton_rows(
    names: list[str], columns: dict[str, Any], offsets: Any, info: WorkloadInfo
) -> dict[str, Row]:
    """Rows that need the Triton launcher. Preparation happens here, untimed."""
    from tapeforge import prepared
    from tapeforge.validation import validate_columns

    ready = prepared.prepare(columns, offsets)
    rows: dict[str, Row] = {}
    two_pass = ready.path == "two_pass"
    base = {"path": ready.path, "n_chunks": ready.n_chunks}

    if "tapeforge_triton" in names:
        rows["tapeforge_triton"] = Row(
            partial(ops.aggregate, columns, offsets),
            "validation (1 host sync) + planning + allocation"
            + (" + chunk metadata (1 host sync) + scratch allocation" if two_pass else "")
            + " + launch(es)",
            gate=partial(ops.aggregate, columns, offsets),
            **base,
        )
    if "tapeforge_prepared" in names:
        rows["tapeforge_prepared"] = Row(
            ready.execute, "launch(es) into prepared buffers; no sync", gate=ready.execute, **base
        )
    if "tapeforge_triton_kernel_only" in names:
        if two_pass:
            assert ready.chunks is not None and ready.partials is not None
            launch: Callable[[], Any] = partial(
                kernels.launch_two_pass_unchecked,
                columns, offsets, ready.features, ready.chunks, ready.partials,
            )
        else:
            assert ready.plan.block_size is not None
            launch = partial(
                kernels.launch_one_pass_unchecked, columns, offsets, ready.features,
                ready.plan.block_size,
            )
        def launch_then_features(launch: Callable[[], Any] = launch) -> dict[str, Any]:
            launch()
            return ready.features

        rows["tapeforge_triton_kernel_only"] = Row(
            launch, "unchecked launcher(s) only; no sync", gate=launch_then_features, **base
        )
    if "tapeforge_pass1_only" in names and two_pass:
        # Pass 1 alone produces no features; its gate is the full execute,
        # which includes the same pass 1 (recorded in "correctness").
        rows["tapeforge_pass1_only"] = Row(
            ready.run_pass_1,
            "pass 1 launch only (partials); correctness inherited from execute",
            gate=ready.execute,
            **base,
        )
    if "tapeforge_pass2_only" in names and two_pass:

        def pass_2_then_features() -> dict[str, Any]:
            ready.run_pass_1()  # partials must exist before pass 2 is checked alone
            ready.run_pass_2()
            return ready.features

        rows["tapeforge_pass2_only"] = Row(
            ready.run_pass_2,
            "pass 2 launch only (combine partials)",
            gate=pass_2_then_features,
            **base,
        )
    if "tapeforge_two_pass_forced" in names and not two_pass:
        forced = prepared.prepare(columns, offsets, path="two_pass")
        rows["tapeforge_two_pass_forced"] = Row(
            forced.execute,
            "two-pass path forced on a one-pass workload; launches only",
            gate=forced.execute,
            path=forced.path,
            n_chunks=forced.n_chunks,
        )

    if "tapeforge_stage_validate" in names:
        rows["tapeforge_stage_validate"] = Row(
            partial(validate_columns, columns, offsets), "validate_columns (1 host sync)", None,
            **base,
        )
    if "tapeforge_stage_chunk_metadata" in names:
        rows["tapeforge_stage_chunk_metadata"] = Row(
            partial(kernels.build_chunk_metadata, offsets, kernels.CHUNK_SIZE),
            "build_chunk_metadata (1 host sync)",
            None,
            path="two_pass",
            n_chunks=info.two_pass_n_chunks,
        )
    if "tapeforge_stage_allocate" in names:

        def allocate() -> None:
            kernels.empty_features(info.n_buckets, offsets.device)
            if two_pass:
                kernels.allocate_partials(ready.n_chunks, offsets.device)

        rows["tapeforge_stage_allocate"] = Row(
            allocate,
            "zero-filled output allocation (5 memsets)"
            + (" + uninitialised scratch allocation" if two_pass else ""),
            None,
            **base,
        )
    if "tapeforge_stage_prepare" in names:
        rows["tapeforge_stage_prepare"] = Row(
            partial(prepared.prepare, columns, offsets),
            "prepare: validation + planning + metadata + allocation",
            None,
            **base,
        )
    return rows


def _rows(
    names: list[str], device: str, trades: Any, offsets: Any, info: WorkloadInfo
) -> dict[str, Row]:
    import torch

    columns = {k: v.to(device) for k, v in host_columns(trades).items()}
    dev_offsets = torch.from_numpy(offsets).to(device)
    rows: dict[str, Row] = {}
    if device == "cuda" and any(name in CUDA_ONLY_ROWS for name in names):
        try:
            rows.update(_triton_rows(names, columns, dev_offsets, info))
        except Exception as exc:  # noqa: BLE001 - preparation failed: every Triton row errors
            reason = f"{type(exc).__name__}: {exc}"[:500]
            for name in names:
                if name in CUDA_ONLY_ROWS:
                    rows[name] = Row(_raise(reason), "preparation failed", None)
    if "torch_eager" in names:
        fn = partial(torch_index_add, columns, dev_offsets)
        rows["torch_eager"] = Row(fn, "eager torch ops; no validation", gate=fn)
    if "torch_compile" in names:
        compiled = partial(torch.compile(torch_index_add, dynamic=False), columns, dev_offsets)
        rows["torch_compile"] = Row(compiled, "torch.compile'd baseline; compiled in warm-up",
                                    gate=compiled)
    if "numpy_reference" in names:
        fn = partial(_numpy_reference, trades, offsets)
        rows["numpy_reference"] = Row(fn, "NumPy reference on the host", gate=fn, host=True)
    return rows


def _measure(
    row: Row,
    expected: reference.Features,
    warmup: int,
    repeats: int,
    on_cuda: bool,
    method: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "includes": row.includes,
        "path": row.path,
        "n_chunks": row.n_chunks,
    }
    try:
        if row.gate is not None:
            check_features(row.gate(), expected)
        else:
            row.fn()  # stage rows have no features to check; make sure they run
    except AssertionError as exc:
        return {**metrics, "status": "incorrect", "reason": str(exc)[:500]}
    except Exception as exc:  # noqa: BLE001 - e.g. torch.compile backend missing
        return {**metrics, "status": "error", "reason": f"{type(exc).__name__}: {exc}"[:500]}
    use_cuda_timing = on_cuda and not row.host
    timings = time_callable(
        row.fn,
        warmup=warmup,
        repeats=repeats,
        synchronize=use_cuda_timing,
        method=method if use_cuda_timing else "wall",  # type: ignore[arg-type]
    )
    return {
        **metrics,
        "status": "ok",
        "correct": True if row.gate is not None else None,
        "timing_method": method if use_cuda_timing else "wall",
        **timings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/kernel_sweep.toml")
    parser.add_argument("--sizes", type=int, nargs="+", help="override config record counts")
    parser.add_argument("--profiles", nargs="+", help="override config profiles")
    parser.add_argument("--seeds", type=int, nargs="+", help="override config seeds")
    parser.add_argument("--repeats", type=int, help="override config repeats")
    parser.add_argument("--rows", nargs="+", choices=ALL_ROWS, help="override config baselines")
    parser.add_argument("--timing", choices=TIMING_METHODS, default="wall")
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
    names = args.rows or config["baselines"]
    bad_rows = 0
    unknown = sorted(set(names) - set(ALL_ROWS))
    if unknown:
        parser.error(f"unknown rows {unknown}; choose from {list(ALL_ROWS)}")
    for size in args.sizes or config["sizes"]:
        for profile in args.profiles or config["profiles"]:
            for seed in args.seeds or config["seeds"]:
                trades, offsets, info = make_workload(
                    size, profile, seed, config["bucket_width_us"]
                )
                expected = reference.aggregate_reference(trades, offsets)
                rows = _rows(names, device, trades, offsets, info)
                for name in names:
                    if name not in rows:
                        why = "needs CUDA" if not on_cuda else "not applicable to this workload"
                        print(f"skip {name}: {why}", file=sys.stderr)
                        if should_record and on_cuda:
                            record("kernel", name, {"status": "not_applicable", "reason": why},
                                   {"workload": info.as_dict()})
                        continue
                    row = rows[name]
                    metrics = _measure(row, expected, warmup, repeats, on_cuda, args.timing)
                    if metrics["status"] == "ok" and row.gate is not None:
                        metrics["records_per_s"] = info.n_records / (metrics["p50_ms"] / 1e3)
                    run_config = {
                        "workload": info.as_dict(),
                        "device": "cpu" if row.host else device,
                        "warmup": warmup,
                        "repeats": repeats,
                        "timing_method": metrics.get("timing_method", "wall"),
                        "timing_method_description": TIMING_DESCRIPTIONS[
                            metrics.get("timing_method", "wall")
                        ],
                        "correctness": (
                            "exact ints, VWAP rtol/atol from contract, vs reference"
                            if row.gate is not None
                            else "stage row: no features produced, not gated"
                        ),
                    }
                    line = {"name": name, **run_config["workload"], **metrics}
                    print(json.dumps({k: v for k, v in line.items() if k != "reason"}))
                    if metrics.get("reason"):
                        print(f"  {name}: {metrics['status']}: {metrics['reason']}",
                              file=sys.stderr)
                    if metrics["status"] in ("incorrect", "error"):
                        bad_rows += 1
                    if should_record:
                        record("kernel", name, metrics, run_config)
    if bad_rows:
        print(f"{bad_rows} row(s) incorrect or errored; see stderr above", file=sys.stderr)
    # Incorrect or errored rows are recorded (negative results are results) but
    # fail the run, so a verification step cannot pass on a broken kernel.
    return 1 if bad_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
