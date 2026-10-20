"""Week 0/8: the public documentation has to be valid, not just present.

README.rst is the project's only public documentation (the repository tracks
no Markdown by policy), and PyPI-style RST fails silently in a browser rather
than loudly in a build. Two malformed tables shipped in the week 0 scaffold
and survived until week 2 because nothing checked. Now something does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.rst"


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


def test_readme_publishes_no_speedup_claim_before_the_kernel_exists() -> None:
    """PRD section 7: no number is reported until our own tests reproduce it.

    Guards against the resume figures drifting into the public docs ahead of
    the measurements. Delete this test the day a real benchmark lands, and say
    in the commit which recorded result replaced it.
    """
    from tapeforge import kernels

    if hasattr(kernels, "agg_kernel_one_pass"):
        pytest.skip("the kernel exists; speedup claims are now allowed if measured")
    text = README.read_text().lower()
    for forbidden in ("2.8x", "1.5x", "38% greater", "28% less", "35% lower"):
        assert forbidden not in text, (
            f"README claims {forbidden!r} but no GPU kernel exists yet"
        )
