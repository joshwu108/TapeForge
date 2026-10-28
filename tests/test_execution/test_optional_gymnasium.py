"""Tests that the CPU package stays usable without gymnasium and that the adapter names the
missing dependency.

Each case runs a fresh interpreter with the ``gymnasium`` import blocked, so
the outcome does not depend on whether this environment has it installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BLOCK = "import sys; sys.modules['gymnasium'] = None; "
TESTS_DIR = Path(__file__).resolve().parent.parent   # Makes the test helpers importable.


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(TESTS_DIR))
    return subprocess.run([sys.executable, "-c", BLOCK + code], capture_output=True, text=True,
                          check=False, env=env)


def test_the_execution_package_imports_and_runs_without_gymnasium() -> None:
    result = _run(
        "from tapeforge.execution import BASELINES, EpisodeSpec, run_episode; "
        "from tapeforge.execution import experiment, report, replay, datasets; "
        "from test_execution.tapes import worksheet_spec, worksheet_tape; "
        "t = run_episode(worksheet_spec(side=1), worksheet_tape(), BASELINES['twap']); "
        "print(t.final_state.cash_ticks)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "-3003800800"


def test_the_adapter_names_the_optional_dependency() -> None:
    result = _run("import tapeforge.execution.gym_env")
    assert result.returncode != 0
    assert "ImportError" in result.stderr and "pip install -e '.[rl]'" in result.stderr
