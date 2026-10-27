"""Shared benchmark plumbing: provenance capture, warm-up, percentiles and JSON records.

Every benchmark result is written through ``record()``, so no number appears
without the hardware, environment, code revision and dataset that produced
it. A performance claim in the public documentation must cite such a record.

Timing rules:
    Each implementation is warmed up before measurement, so Triton and
    ``torch.compile`` compilation never lands inside a timed region. Every
    timed region is a complete Python call: it includes Python dispatch,
    argument handling and kernel-launch overhead, not only GPU execution
    time.

Timing methods (recorded by name in every result):
    wall         ``time.perf_counter`` around the call; with
                 ``synchronize=True``, ``torch.cuda.synchronize()`` before
                 and after every call. The host's view: what a caller
                 waiting for the result experiences.
    cuda_events  ``torch.cuda.Event(enable_timing=True)`` pairs recorded
                 on the current stream around the call, elapsed time read
                 after a synchronize. The device-stream view. Because the
                 start event executes as soon as the stream reaches it,
                 an idle GPU still charges the host's launch latency to
                 this measurement; what it excludes is the host-side
                 synchronize cost and Python work after the last launch.

Scopes:
    The three scopes -- kernel-only on resident tensors, host-to-device
    transfer plus features, and the whole training loop -- are recorded as
    separate entries and never combined into one headline number. Stage
    timings of one pipeline (validation, metadata, allocation, pass 1,
    pass 2) overlap with the end-to-end rows and must not be summed as if
    they were independent.
"""

from __future__ import annotations

import json
import platform as platform_module
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RESULTS_PATH = Path(__file__).parent / "results.json"
REPO = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO / "snapshot.json"  # Written by ``tools/verify_snapshot.py pack``.

TimingMethod = Literal["wall", "cuda_events"]
TIMING_METHODS: tuple[str, ...] = ("wall", "cuda_events")


@dataclass(frozen=True)
class Environment:
    """Software versions and hardware identity, captured per run and pinned in every record.

    Attributes:
        python: Interpreter version.
        platform: ``platform.platform()`` string.
        processor: ``platform.processor()`` string.
        numpy: NumPy version.
        torch: PyTorch version, or ``None`` when torch is not installed.
        triton: Triton version, or ``None`` when triton is not installed.
        cuda: CUDA version PyTorch was built against, or ``None``.
        gpu_name: Name of CUDA device 0, or ``None`` without CUDA.
        gpu_capability: Compute capability of device 0 as ``"major.minor"``.
        git_commit: Short commit hash, or ``None`` when git is unavailable.
        git_dirty: True when the working tree has uncommitted or untracked
            files, in which case ``git_commit`` alone does not reproduce the
            measured code.
    """

    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform_module.platform)
    processor: str = field(default_factory=platform_module.processor)
    numpy: str | None = None
    torch: str | None = None
    triton: str | None = None
    cuda: str | None = None
    gpu_name: str | None = None
    gpu_capability: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None


def capture_environment() -> Environment:
    """Collect versions and hardware identity; unavailable pieces stay ``None``.

    Returns:
        An ``Environment`` whose git fields are ``None`` when ``git`` is not
        installed or the benchmarks directory is not inside a checkout.
    """
    torch_v = triton_v = cuda_v = gpu = capability = None
    import numpy

    try:
        import torch

        torch_v = torch.__version__
        cuda_v = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            capability = "{}.{}".format(*torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    try:
        import triton

        triton_v = triton.__version__
    except ImportError:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
        dirty: bool | None = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
        dirty = None
    return Environment(
        numpy=numpy.__version__,
        torch=torch_v,
        triton=triton_v,
        cuda=cuda_v,
        gpu_name=gpu,
        gpu_capability=capability,
        git_commit=commit,
        git_dirty=dirty,
    )


def snapshot_metadata() -> dict[str, Any] | None:
    """Return the packed tree's identity when running from an unpacked snapshot, else ``None``.

    A snapshot carries the full commit, dirty flag and a digest over every
    packed file, which identifies the tested tree even when ``git`` is not
    available on the GPU machine. An unreadable or non-object
    ``snapshot.json`` is treated as absent.
    """
    if not SNAPSHOT_PATH.is_file():
        return None
    try:
        loaded = json.loads(SNAPSHOT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


TIMING_DESCRIPTIONS: dict[str, str] = {
    "wall": (
        "wall clock (time.perf_counter) around each call; with synchronize=True, "
        "torch.cuda.synchronize() before and after every call; warm-up calls untimed; "
        "includes Python dispatch and launch overhead"
    ),
    "cuda_events": (
        "torch.cuda.Event(enable_timing=True) recorded on the current stream before and "
        "after each call, elapsed_time after synchronize; warm-up calls untimed; includes "
        "launch latency when the stream is idle, excludes host-side synchronize cost"
    ),
}
# The default method's description; ``pipeline.py`` records it under this name.
TIMING_METHOD = TIMING_DESCRIPTIONS["wall"]


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
    synchronize: bool = False,
    method: TimingMethod = "wall",
) -> dict[str, float]:
    """Time ``fn`` and return p50/p95/p99/min/mean latencies in milliseconds.

    Percentiles are read from the sorted samples by index
    (``samples[int(len(samples) * q)]``) without interpolation.

    Args:
        fn: Zero-argument callable; its complete execution is the timed region.
        warmup: Untimed calls before measurement.
        repeats: Timed calls; the sample count.
        synchronize: With ``method="wall"``, call ``torch.cuda.synchronize()``
            before and after every timed call.
        method: ``"wall"`` or ``"cuda_events"``. ``"cuda_events"`` requires
            CUDA and implies synchronisation, since the events are read after
            ``torch.cuda.synchronize()``.

    Returns:
        ``min_ms``, ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms`` and
        ``repeats`` (as a float).

    Raises:
        ValueError: ``method`` is not in ``TIMING_METHODS``.
        RuntimeError: ``method="cuda_events"`` without a CUDA device.
    """
    if method not in TIMING_METHODS:
        raise ValueError(f"method must be one of {TIMING_METHODS}, got {method!r}")
    if method == "cuda_events":
        samples = _cuda_event_samples(fn, warmup, repeats)
    else:
        samples = _wall_samples(fn, warmup, repeats, synchronize)
    samples.sort()
    return {
        "min_ms": samples[0],
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "p99_ms": samples[int(len(samples) * 0.99)],
        "repeats": float(repeats),
    }


def _wall_samples(
    fn: Callable[[], Any], warmup: int, repeats: int, synchronize: bool
) -> list[float]:
    """Return ``repeats`` wall-clock samples in milliseconds, optionally synchronised."""
    sync = _make_sync(synchronize)
    for _ in range(warmup):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _cuda_event_samples(fn: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    """Return ``repeats`` CUDA-event samples in milliseconds from the current stream.

    Raises:
        RuntimeError: No CUDA device is available.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("method='cuda_events' needs a CUDA device")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _make_sync(synchronize: bool) -> Callable[[], None]:
    """Return ``torch.cuda.synchronize`` when ``synchronize`` is set, otherwise a no-op."""
    if not synchronize:
        return lambda: None
    import torch

    return torch.cuda.synchronize


def record(
    scope: str,
    name: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Append one measurement to the results file with full provenance.

    Negative results (incorrect or errored rows) are recorded exactly like
    positive ones.

    Args:
        scope: One of ``"kernel"``, ``"pipeline"`` or ``"train"``, the three
            separately timed scopes.
        name: Row or variant name within the scope.
        metrics: Measured values and status.
        config: Workload and run configuration that produced ``metrics``.
        path: Results file; a file whose content is not a JSON list is
            replaced by a list holding only the new entry.

    Returns:
        The entry as written, including the environment and snapshot
        provenance and a UTC timestamp.
    """
    entry = {
        "scope": scope,
        "name": name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": asdict(capture_environment()),
        "snapshot": snapshot_metadata(),
        "config": config,
        "metrics": metrics,
    }
    existing: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text() or "[]")
        existing = loaded if isinstance(loaded, list) else []
    path.write_text(json.dumps([*existing, entry], indent=2) + "\n")
    return entry
