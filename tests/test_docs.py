"""Public documentation syntax, examples, and measurement claims."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.rst"
RESULTS = Path(__file__).parent.parent / "benchmarks" / "results.json"


def _parse_problems(text: str) -> list[str]:
    docutils = pytest.importorskip("docutils.core")
    problems: list[str] = []

    class Collector:
        def write(self, message: str) -> None:
            problems.append(message)

    docutils.publish_doctree(
        text,
        settings_overrides={
            "report_level": 2,       # warnings and worse
            "halt_level": 5,         # collect rather than raise
            "warning_stream": Collector(),
            "file_insertion_enabled": False,
        },
    )
    return problems


def test_readme_is_valid_restructuredtext() -> None:
    problems = _parse_problems(README.read_text())
    assert not problems, "README.rst has RST errors:\n" + "".join(problems)


def test_readme_documents_every_public_feature() -> None:
    """A feature that exists but is undocumented is a documentation bug."""
    from tapeforge.contract import FEATURE_NAMES

    text = README.read_text()
    missing = [name for name in FEATURE_NAMES if f"``{name}``" not in text]
    assert not missing, f"undocumented features: {missing}"


def _recorded_gpu_kernel_results() -> list[dict[str, object]]:
    results = json.loads(RESULTS.read_text() or "[]")
    return [
        entry
        for entry in results
        if entry.get("scope") == "kernel"
        and (entry.get("environment") or {}).get("gpu_name")
        and (entry.get("metrics") or {}).get("status") == "ok"
        and (entry.get("metrics") or {}).get("correct") is True
    ]


def test_readme_publishes_no_speedup_claim_before_a_gpu_measurement_exists() -> None:
    """PRD section 7: no number is reported until our own benchmarks reproduce it.

    Guards against the resume figures drifting into the public docs ahead of
    the measurements. The kernel existing is not enough: the guard lifts only
    once benchmarks/results.json holds a correct, successful kernel-scope entry
    recorded on a GPU; failed or unsupported runs do not count.
    """
    if _recorded_gpu_kernel_results():
        pytest.skip("a GPU kernel measurement is recorded; claims must cite it")
    text = README.read_text().lower()
    for forbidden in ("2.8x", "1.5x", "38% greater", "28% less", "35% lower"):
        assert forbidden not in text, (
            f"README claims {forbidden!r} but no GPU kernel measurement is recorded"
        )


def _python_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        body: list[str] = []
        for following in lines[i + 1 :]:
            if following and not following.startswith("   "):
                break
            body.append(following[3:])
        blocks.append(textwrap.dedent("\n".join(body)))
    return blocks


def test_readme_python_examples_run() -> None:
    """Every ``.. code-block:: python`` in README.rst executes as written (on CPU)."""
    pytest.importorskip("torch")
    blocks = _python_blocks(README.read_text())
    assert blocks, "README has no Python example to check"
    for block in blocks:
        exec(compile(block, "README.rst", "exec"), {})  # noqa: S102 - our own docs
