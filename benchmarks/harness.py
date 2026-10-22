"""Shared benchmark plumbing: provenance, warmup, percentiles, JSON records.

Every benchmark result is written through ``record()`` so no number ever
appears without the hardware, environment and dataset that produced it
(PRD section 6, reproducibility).

Timing rules enforced here (PRD section 6, fair comparisons):
  - each implementation is warmed up before measurement;
  - on CUDA, ``torch.cuda.synchronize()`` brackets every timed region;
  - the three scopes -- kernel-only on resident tensors, host-to-device plus
    features, and whole training loop -- are recorded as separate entries and
    never mixed into one headline number.
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
from typing import Any

RESULTS_PATH = Path(__file__).parent / "results.json"


@dataclass(frozen=True)
class Environment:
    """Captured once per run; pinned in every result record."""

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
    git_dirty: bool | None = None  # True: uncommitted or untracked files; commit won't reproduce


def capture_environment() -> Environment:
    """Collect versions and hardware identity. Missing pieces stay None."""
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


TIMING_METHOD = (
    "wall clock (time.perf_counter) around each call; with synchronize=True, "
    "torch.cuda.synchronize() before and after every call; warm-up calls untimed"
)


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
    synchronize: bool = False,
) -> dict[str, float]:
    """Return p50/p95/p99/min/mean latencies in milliseconds."""
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
    samples.sort()
    return {
        "min_ms": samples[0],
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "p99_ms": samples[int(len(samples) * 0.99)],
        "repeats": float(repeats),
    }


def _make_sync(synchronize: bool) -> Callable[[], None]:
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
    """Append one measurement to results.json with full provenance.

    ``scope`` is one of "kernel", "pipeline", "train" -- the three separately
    timed scopes. Negative results are recorded exactly like positive ones.
    """
    entry = {
        "scope": scope,
        "name": name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": asdict(capture_environment()),
        "config": config,
        "metrics": metrics,
    }
    existing: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text() or "[]")
        existing = loaded if isinstance(loaded, list) else []
    path.write_text(json.dumps([*existing, entry], indent=2) + "\n")
    return entry
