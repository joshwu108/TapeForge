"""Tests for ``tools/verify_snapshot.py``: the snapshot packer and verification runner.

Both halves are exercised on the CPU: packing must ship exactly the
Git-visible tree with its provenance, and running must record skipped GPU
steps as skipped -- never as passed -- while executing the CPU steps. The
run test invokes the tool as a subprocess against a temporary artifacts
directory; the pack test reads back the tarball it writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_snapshot as vs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --- planning --------------------------------------------------------------


def test_default_plan_without_cuda_skips_every_gpu_step_with_the_reason() -> None:
    """Without CUDA every GPU step is planned as skipped with the probe's reason, CPU steps run."""
    planned = vs.plan(None, cuda=False, cuda_reason="no CUDA device available", record=False)
    assert [step.name for step, _ in planned] == list(vs.STEP_NAMES)
    skipped = {step.name: reason for step, reason in planned if reason}
    for step in vs.STEPS:
        if step.needs_cuda:
            assert skipped[step.name] == "no CUDA device available"
    assert "lint" not in skipped and "cpu_suite" not in skipped


def test_recording_steps_need_the_record_flag_even_with_cuda() -> None:
    """Steps that write results.json stay skipped until ``record`` is set, even with CUDA."""
    without = dict((s.name, r) for s, r in vs.plan(None, True, "", record=False))
    with_record = dict((s.name, r) for s, r in vs.plan(None, True, "", record=True))
    for step in vs.STEPS:
        if step.writes_results:
            assert "results.json" in (without[step.name] or "")
            assert with_record[step.name] is None
        elif step.needs_cuda:
            assert without[step.name] is None


def test_step_subset_keeps_the_canonical_order_and_rejects_unknown_names() -> None:
    """A requested subset runs in ``STEP_NAMES`` order regardless of how it was listed."""
    planned = vs.plan(["kernel_quick", "lint"], True, "", True)
    assert [step.name for step, _ in planned] == ["lint", "kernel_quick"]
    with pytest.raises(ValueError, match="unknown steps"):
        vs.plan(["nope"], True, "", True)


def test_pytest_steps_get_a_junit_report_and_the_rest_do_not() -> None:
    """The ``pytest`` flag on a step agrees with whether its argv invokes pytest."""
    for step in vs.STEPS:
        assert step.pytest == ("pytest" in step.argv), step.name


# --- pack -------------------------------------------------------------------


def test_tree_files_follow_gitignore() -> None:
    """The packed tree is the Git-visible tree: ignored directories and Markdown are absent."""
    files = {str(p.relative_to(REPO)) for p in vs.tree_files()}
    assert "README.rst" in files and "src/tapeforge/contract.py" in files
    assert "src/tapeforge/streaming.py" in files
    assert not any(name.startswith(("planning/", ".venv/", "venv/", "data/")) for name in files)
    assert not any(name.endswith(".md") for name in files)
    assert not any("__pycache__" in name for name in files)


def test_tree_digest_changes_when_content_changes(tmp_path: Path) -> None:
    """The tree digest is deterministic and sensitive to the file set."""
    files = vs.tree_files()[:3]
    before = vs.tree_digest(files)
    assert before == vs.tree_digest(files)  # Deterministic across calls.
    assert before != vs.tree_digest(files[:2])


def test_pack_writes_the_tree_and_snapshot_metadata(tmp_path: Path) -> None:
    """The tarball holds the tree under ``ARCHIVE_ROOT`` plus a snapshot.json with provenance."""
    out = tmp_path / "snap.tgz"
    meta = vs.pack(out)
    assert out.exists() and meta["archive_sha256"]
    with tarfile.open(out) as archive:
        names = archive.getnames()
        snapshot = json.loads(archive.extractfile(f"{vs.ARCHIVE_ROOT}/snapshot.json").read())
    assert f"{vs.ARCHIVE_ROOT}/README.rst" in names
    assert f"{vs.ARCHIVE_ROOT}/tools/verify_snapshot.py" in names
    assert not any("/planning/" in n or n.endswith(".md") for n in names)
    assert all(n.startswith(vs.ARCHIVE_ROOT + "/") for n in names)
    assert snapshot["tree_sha256"] == meta["tree_sha256"]
    assert snapshot["n_files"] == len(names) - 1
    assert snapshot["git_commit"] and isinstance(snapshot["git_dirty"], bool)


# --- run ----------------------------------------------------------------------


def test_run_records_executed_and_skipped_steps_in_the_manifest(tmp_path: Path) -> None:
    """The lint step runs for real; a requested GPU step is recorded as skipped, not passed."""
    artifacts = tmp_path / "verification"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "verify_snapshot.py"),
            "run",
            "--artifacts",
            str(artifacts),
            "--steps",
            "lint",
            "toy_program",
            "--zip",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((artifacts / "manifest.json").read_text())
    environment = json.loads((artifacts / "environment.json").read_text())
    by_name = {step["name"]: step for step in manifest["steps"]}

    assert by_name["lint"]["status"] == "ok" and by_name["lint"]["exit_code"] == 0
    assert (artifacts / "lint.log").exists()
    assert manifest["executed"] == ["lint"]
    if environment["cuda"]["available"]:
        assert by_name["toy_program"]["status"] in ("ok", "failed")
    else:
        assert by_name["toy_program"]["status"] == "skipped"
        assert by_name["toy_program"]["reason"] == environment["cuda"]["reason"]
        assert "toy_program" in manifest["skipped"]
    assert manifest["failed"] == []
    assert manifest["results_json_entries_added"] == 0
    assert environment["environment"]["python"]
    assert (tmp_path / "verification.zip").exists()
    assert (artifacts / "results.json").exists()


def test_run_reports_failure_when_a_step_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing step gives exit code 1 and lands in the manifest's ``failed`` list."""
    failing = vs.Step("boom", (sys.executable, "-c", "raise SystemExit(3)"))
    monkeypatch.setattr(vs, "STEPS", (failing,))
    monkeypatch.setattr(vs, "STEP_NAMES", ("boom",))
    code = vs.run(tmp_path / "art", None, record=False, make_zip=False)
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text())
    assert code == 1
    assert manifest["failed"] == ["boom"]
    assert manifest["steps"][0]["exit_code"] == 3
