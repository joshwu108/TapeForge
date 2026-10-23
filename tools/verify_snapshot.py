"""Pack the working tree for a GPU machine and run the verification steps there.

One file holds both halves, so the snapshot that leaves this machine and the
checks that run on the other one cannot drift apart:

    python tools/verify_snapshot.py pack --out /tmp/tapeforge-snapshot.tgz
    python tools/verify_snapshot.py run --artifacts verification --record --zip

``pack`` archives every file Git tracks or would track (untracked but not
ignored), under a top-level ``TapeForge/`` directory. Uncommitted edits are
therefore what gets tested, and ignored material (planning notes, datasets,
virtualenvs, caches) never leaves the machine. The archive carries
``snapshot.json`` -- commit, dirty flag, a digest over every packed file --
so the run on the other side can say exactly which tree it verified.

``run`` executes the steps below in order, each as a subprocess, and writes
into the artifacts directory:

    environment.json   benchmarks/harness.capture_environment() plus the
                       nvidia-smi query and snapshot.json when present
    <step>.log         combined stdout/stderr of each step
    <step>.xml         JUnit report for each pytest step
    results.json       copy of benchmarks/results.json after the run
    manifest.json      every step with argv, status, exit code and duration

Steps that need CUDA are recorded as ``skipped`` with the reason when no
device is available; a skipped step is never reported as passed. The exit
status is non-zero if any executed step failed. The default step list runs
nothing that writes to ``benchmarks/results.json``; ``--record`` adds the
kernel sweep and the pipeline benchmark, which append to it.

Usable from a plain checkout with the GPU extras installed; nothing here
depends on the local virtualenv layout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = "TapeForge"
SNAPSHOT_FILE = "snapshot.json"
RESULTS = REPO / "benchmarks" / "results.json"

PY = sys.executable


@dataclass(frozen=True)
class Step:
    """One verification command. ``needs_cuda`` steps skip without a device."""

    name: str
    argv: tuple[str, ...]
    needs_cuda: bool = False
    writes_results: bool = False  # appends to benchmarks/results.json; only with --record
    pytest: bool = False  # gets --junitxml=<artifacts>/<name>.xml


STEPS: tuple[Step, ...] = (
    Step("lint", (PY, "-m", "ruff", "check", "src", "tests", "benchmarks", "tools")),
    Step("cpu_suite", (PY, "-m", "pytest", "-q", "-ra"), pytest=True),
    Step(
        "gpu_suite", (PY, "-m", "pytest", "-q", "-rsx", "-m", "gpu"), needs_cuda=True, pytest=True
    ),
    Step("slow_suite", (PY, "-m", "pytest", "-q", "-rsx", "-m", "slow"), pytest=True),
    Step("gpu_parity_script", (PY, "tools/check_gpu_parity.py"), needs_cuda=True),
    Step("toy_program", (PY, "test_program.py"), needs_cuda=True),
    Step(
        "kernel_quick",
        (PY, "benchmarks/kernel.py", "--sizes", "100000", "--seeds", "1", "--no-record"),
    ),
    Step("kernel_sweep", (PY, "benchmarks/kernel.py"), needs_cuda=True, writes_results=True),
    Step("pipeline", (PY, "benchmarks/pipeline.py"), needs_cuda=True, writes_results=True),
)
STEP_NAMES = tuple(step.name for step in STEPS)


@dataclass
class StepResult:
    """One row of ``manifest.json``; ``reason`` is set only for skipped steps."""

    name: str
    argv: list[str]
    status: str  # "ok" | "failed" | "skipped"
    reason: str | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    log: str | None = None
    junit: str | None = None


# --- git and environment ------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tree_files() -> list[Path]:
    """Files Git tracks or would track: tracked plus untracked-but-not-ignored."""
    listing = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if listing is None:
        raise RuntimeError(f"{REPO} is not a Git checkout; pack needs Git to apply .gitignore")
    # An unpacked snapshot carries its own snapshot.json; never pack it twice.
    names = sorted({name for name in listing.split("\0") if name and name != SNAPSHOT_FILE})
    return [REPO / name for name in names if (REPO / name).is_file()]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(files: list[Path]) -> str:
    """One hash over (relative path, content hash) of every packed file, in order."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path.read_bytes()).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def snapshot_metadata(files: list[Path]) -> dict[str, Any]:
    """The ``snapshot.json`` payload: commit, dirty flag, file count, tree digest, pack time."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "n_files": len(files),
        "tree_sha256": tree_digest(files),
        "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packed_python": platform.python_version(),
    }


def capture_environment() -> dict[str, Any]:
    """The benchmark harness's provenance record, so results and runs agree."""
    sys.path.insert(0, str(REPO / "benchmarks"))
    import harness  # noqa: PLC0415 - benchmarks/ is a script directory, not a package

    return asdict(harness.capture_environment())


def nvidia_smi() -> str | None:
    """The GPU query as CSV text, or ``None`` when the tool is absent or fails."""
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def cuda_status() -> tuple[bool, str]:
    """Same probe as tests/conftest.py, so the runner and the suite agree on 'skip'."""
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device available"
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "CUDA is available but triton is not installed (pip install -e '.[gpu]')"
    return True, ""


# --- pack -----------------------------------------------------------------------


def pack(out: Path) -> dict[str, Any]:
    """Write the tarball and return its metadata plus the archive path and sha256.

    Raises:
        RuntimeError: not a Git checkout (``tree_files``).
    """
    files = tree_files()
    meta = snapshot_metadata(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{ARCHIVE_ROOT}/{path.relative_to(REPO)}", recursive=False)
        payload = json.dumps(meta, indent=2).encode() + b"\n"
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{SNAPSHOT_FILE}")
        info.size = len(payload)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(payload))
    return {**meta, "archive": str(out), "archive_sha256": _sha256(out.read_bytes())}


# --- run ------------------------------------------------------------------------


def plan(
    only: list[str] | None, cuda: bool, cuda_reason: str, record: bool
) -> list[tuple[Step, str | None]]:
    """Resolve which steps execute and which are skipped, with the reason."""
    if only:
        unknown = sorted(set(only) - set(STEP_NAMES))
        if unknown:
            raise ValueError(f"unknown steps {unknown}; choose from {list(STEP_NAMES)}")
    chosen = [step for step in STEPS if not only or step.name in only]
    planned: list[tuple[Step, str | None]] = []
    for step in chosen:
        if step.needs_cuda and not cuda:
            planned.append((step, cuda_reason))
        elif step.writes_results and not record:
            planned.append((step, "appends to benchmarks/results.json; pass --record to run it"))
        else:
            planned.append((step, None))
    return planned


def run_step(step: Step, artifacts: Path) -> StepResult:
    """Execute one step as a subprocess, logging to ``<name>.log``; never raises on failure."""
    argv = list(step.argv)
    log = artifacts / f"{step.name}.log"
    junit = None
    if step.pytest:
        junit = artifacts / f"{step.name}.xml"
        argv.append(f"--junitxml={junit}")
    started = time.perf_counter()
    with log.open("w") as handle:
        completed = subprocess.run(argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - started
    return StepResult(
        name=step.name,
        argv=argv,
        status="ok" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        duration_s=round(duration, 3),
        log=log.name,
        junit=junit.name if junit else None,
    )


def _results_entries() -> int:
    if not RESULTS.exists():
        return 0
    loaded = json.loads(RESULTS.read_text() or "[]")
    return len(loaded) if isinstance(loaded, list) else 0


def run(artifacts: Path, only: list[str] | None, record: bool, make_zip: bool) -> int:
    """Run the planned steps, write the artifacts, return 1 if any executed step failed.

    Raises:
        ValueError: ``only`` names an unknown step.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    cuda, cuda_reason = cuda_status()
    snapshot_path = REPO / SNAPSHOT_FILE
    environment = {
        "environment": capture_environment(),
        "nvidia_smi": nvidia_smi(),
        "cuda": {"available": cuda, "reason": cuda_reason or None},
        "snapshot": json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None,
        "repo": str(REPO),
    }
    (artifacts / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entries_before = _results_entries()
    results: list[StepResult] = []
    for step, skip_reason in plan(only, cuda, cuda_reason, record):
        if skip_reason is not None:
            results.append(StepResult(step.name, list(step.argv), "skipped", reason=skip_reason))
            print(f"skip   {step.name}: {skip_reason}", flush=True)
            continue
        print(f"run    {step.name}: {' '.join(step.argv)}", flush=True)
        result = run_step(step, artifacts)
        results.append(result)
        print(f"{result.status:6} {step.name} ({result.duration_s}s, exit {result.exit_code})",
              flush=True)

    if RESULTS.exists():
        shutil.copy(RESULTS, artifacts / "results.json")
    manifest = {
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda": environment["cuda"],
        "snapshot": environment["snapshot"],
        "steps": [asdict(result) for result in results],
        "results_json_entries_added": _results_entries() - entries_before,
        "executed": [r.name for r in results if r.status != "skipped"],
        "skipped": {r.name: r.reason for r in results if r.status == "skipped"},
        "failed": [r.name for r in results if r.status == "failed"],
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if make_zip:
        bundle = shutil.make_archive(str(artifacts), "zip", root_dir=artifacts)
        print(f"artifacts: {bundle}")
    summary = (
        f"executed {len(manifest['executed'])}, skipped {len(manifest['skipped'])}, "
        f"failed {len(manifest['failed'])}"
    )
    print(f"manifest: {artifacts / 'manifest.json'} ({summary})")
    return 1 if manifest["failed"] else 0


# --- CLI --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI: ``pack --out PATH`` or ``run [--artifacts DIR] [--steps ...] [--record] [--zip]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    packer = sub.add_parser("pack", help="archive the working tree for a GPU machine")
    packer.add_argument("--out", type=Path, default=Path("/tmp/tapeforge-snapshot.tgz"))

    runner = sub.add_parser("run", help="execute the verification steps, write artifacts")
    runner.add_argument("--artifacts", type=Path, default=REPO / "verification")
    runner.add_argument("--steps", nargs="+", choices=STEP_NAMES, help="subset, in order")
    runner.add_argument(
        "--record", action="store_true", help="also run the steps that append to results.json"
    )
    runner.add_argument("--zip", action="store_true", help="bundle the artifacts directory")
    args = parser.parse_args(argv)

    if args.command == "pack":
        print(json.dumps(pack(args.out), indent=2))
        return 0
    return run(args.artifacts, args.steps, args.record, args.zip)


if __name__ == "__main__":
    raise SystemExit(main())
