"""Tests of one-factor experiment matrices: strict files, sealed dates, runs and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tapeforge.execution.baselines import zero_action
from tapeforge.execution.research.experiment import ConfigError, load_config, load_manifest
from tapeforge.execution.research.matrix import (
    MATRIX_SCHEMA,
    Variant,
    load_matrix,
    main,
    matrix_from_mapping,
    run_matrix,
    summarize_matrix,
    variant_config,
)
from tapeforge.execution.rl.evaluation import evaluate_policy
from test_execution.ppo_configs import EXPERIMENT

MATRIX = """
schema = "tapeforge.execution.experiment-matrix/1"
name = "matrix-test"
description = "tiny synthetic matrix"
base_config = "{root}/experiment.toml"

[[variants]]
name = "surcharge-50bps"
template = {{ terminal_surcharge_bps = 50 }}

[[variants]]
name = "fee-5bps"
template = {{ fee_bps = 5 }}

[[variants]]
name = "latency-1"
template = {{ latency_buckets = 1 }}

[[variants]]
name = "period-swap"
[variants.split]
train = ["2024-01-01"]
validation = ["2024-01-02"]
"""


def _files(root: Path) -> Path:
    """Write a base whose third day is a sealed test day, and a matrix over it."""
    text = EXPERIMENT.format(root=root).replace(
        'validation = ["2024-01-03"]\ntest = []', 'validation = ["2024-01-02"]\ntest = '
        '["2024-01-03"]').replace('train = ["2024-01-01", "2024-01-02"]',
                                  'train = ["2024-01-01"]')
    (root / "experiment.toml").write_text(text)
    path = root / "matrix.toml"
    path.write_text(MATRIX.format(root=root))
    return path


def _mapping(**variant: object) -> dict[str, object]:
    return {"schema": MATRIX_SCHEMA, "name": "m", "base_config": "base.toml",
            "variants": [{"name": "v", **variant}]}


def test_a_variant_changes_exactly_one_factor() -> None:
    matrix = matrix_from_mapping(_mapping(template={"fee_bps": 5}))
    assert matrix.variants == (Variant(name="v", template=(("fee_bps", 5),)),)
    assert (matrix.variants[0].kind, matrix.variants[0].factor) == ("environment", "fee_bps")
    for bad, message in (
        ({"template": {"fee_bps": 5, "latency_buckets": 1}}, "one factor at a time"),
        ({"template": {"side": -1}}, "unknown template field"),
        ({"template": {"fee_bps": 5}, "split": {"train": ["a"], "validation": ["b"]}}, "either"),
        ({}, "either"),
        ({"split": {"train": ["2024-01-01"]}}, "train and validation"),
        ({"split": {"train": ["a"], "validation": ["b"], "test": ["c"]}}, "nothing else"),
    ):
        with pytest.raises(ConfigError, match=message):
            matrix_from_mapping(_mapping(**bad))
    with pytest.raises(ConfigError, match="distinct names"):
        matrix_from_mapping({**_mapping(template={"fee_bps": 5}),
                             "variants": [{"name": "base", "template": {"fee_bps": 5}}]})
    with pytest.raises(ConfigError, match="schema"):
        matrix_from_mapping({**_mapping(template={"fee_bps": 5}), "schema": "other"})


def test_variants_cannot_touch_sealed_test_dates_or_equal_the_base(tmp_path: Path) -> None:
    base = load_config(_files(tmp_path).parent / "experiment.toml")
    changed = variant_config(base, Variant(name="fee", template=(("fee_bps", 5),)))
    assert changed.template.fee_bps == 5 and changed.name == "ppo-test--fee"
    assert changed.split == base.split
    with pytest.raises(ConfigError, match="does not differ"):
        variant_config(base, Variant(name="same", template=(("fee_bps", 10),)))
    with pytest.raises(ConfigError, match="terminal_surcharge_bps"):
        variant_config(base, Variant(name="bad", template=(("terminal_surcharge_bps", -1),)))
    sealed = matrix_from_mapping(_mapping(
        split={"train": ["2024-01-01"], "validation": ["2024-01-03"]}
    )).variants[0]
    with pytest.raises(ConfigError, match="seals as test"):
        variant_config(base, sealed)
    unknown = matrix_from_mapping(_mapping(
        split={"train": ["2024-01-01"], "validation": ["2023-06-01"]}
    )).variants[0]
    with pytest.raises(ConfigError, match="does not list"):
        variant_config(base, unknown)


def test_a_matrix_runs_every_variant_once_and_marks_what_is_paired(tmp_path: Path) -> None:
    root = run_matrix(load_matrix(_files(tmp_path)))
    listing = json.loads((root / "matrix.json").read_text())
    assert [r["variant"] for r in listing["runs"]] == [
        "base", "surcharge-50bps", "fee-5bps", "latency-1", "period-swap",
    ]
    for run in listing["runs"]:
        manifest = load_manifest(root / run["run_dir"])
        assert manifest["roles_evaluated"] == ["train", "validation"]   # never the test role
        assert not list((root / run["run_dir"] / "records").glob("*-test-*"))
    summary = json.loads((root / "summary.json").read_text())
    rows = {(r["variant"], r["policy"], r["role"]): r for r in summary["table"]}
    assert len(rows) == 5 * 2 * 2
    assert rows[("fee-5bps", "twap", "validation")]["episodes_identical"] is True
    assert rows[("latency-1", "twap", "validation")]["episodes_identical"] is False
    assert rows[("period-swap", "twap", "validation")]["kind"] == "period"
    base, cheaper = rows[("base", "twap", "validation")], rows[("fee-5bps", "twap", "validation")]
    assert cheaper["mean_fees_bps"] < base["mean_fees_bps"]
    assert cheaper["mean_shortfall_bps"] < base["mean_shortfall_bps"]
    assert (root / "summary.rst").read_text().startswith("Experiment matrix: matrix-test")
    assert (root / "summary.csv").read_text().count("\n") == len(rows) + 1

    # A policy evaluated into a variant's run afterwards appears on the next summary.
    variant_run = root / listing["runs"][1]["run_dir"]
    evaluate_policy(variant_run, zero_action, "external")
    policies = {r["policy"] for r in summarize_matrix(root)["table"]
                if r["variant"] == "surcharge-50bps"}
    assert policies == {"twap", "immediate", "external"}


def test_the_matrix_command_runs_a_subset_and_reports_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _files(tmp_path)
    assert main(["run", str(path), "--only", "fee-5bps", "--output-dir",
                 str(tmp_path / "out")]) == 0
    assert "fee-5bps" in capsys.readouterr().out
    (matrix_dir,) = (tmp_path / "out").iterdir()
    assert main(["summarize", str(matrix_dir)]) == 0
    assert main(["run", str(path), "--only", "nope"]) == 1
    assert "not variants" in capsys.readouterr().err
    assert main(["summarize", str(tmp_path / "missing")]) == 1


def test_tracked_matrices_are_valid_against_their_base_configurations() -> None:
    paths = sorted(Path("experiments/matrices").glob("*.toml"))
    assert paths
    for path in paths:
        matrix = load_matrix(path)
        base = load_config(matrix.base_config)
        for variant in matrix.variants:
            config = variant_config(base, variant)
            assert not set(config.split.train + config.split.validation) & set(base.split.test)
