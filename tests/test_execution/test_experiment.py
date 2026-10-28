"""Tests for tapeforge.execution.experiment: strict configuration loading, identical episodes for
every policy regardless of order, reproducible run directories, the held-out rule, atomic and
collision-safe writes, and the comparison utility.

Everything runs on the synthetic configuration of ``test_execution.configs``;
the archive path is exercised with a fake network in ``test_datasets``.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from tapeforge.execution import experiment
from tapeforge.execution.experiment import (
    EXPERIMENT_MANIFEST_SCHEMA,
    ConfigError,
    compare_experiments,
    config_from_mapping,
    create_run_directory,
    episode_identifier,
    load_config,
    load_manifest,
    parse_episode_identifier,
    run_experiment,
    write_json_atomic,
)
from tapeforge.execution.simulator import SIMULATOR_ASSUMPTIONS, SIMULATOR_VERSION
from test_execution.configs import DATES, synthetic_config, synthetic_mapping

# --- configuration --------------------------------------------------------------------


def test_config_resolves_template_split_and_episode_counts() -> None:
    config = synthetic_config()
    assert config.template.horizon_buckets == 12 and config.template.fee_bps == 10
    assert config.split.train == (DATES[0], DATES[1]) and config.split.test == ()
    assert config.n_episodes("train") == 8 and config.n_episodes("validation") == 4
    payload = config.to_json()
    assert payload["schema"] == experiment.EXPERIMENT_CONFIG_SCHEMA
    assert "side" not in payload["template"] and payload["split"]["validation"] == [DATES[2]]
    json.dumps(payload)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("top", "label", "final", "label must be one of"),
        ("top", "name", "bad name!", "name must match"),
        ("top", "typo", 1, "unknown key"),
        ("episodes", "seeds", [], "distinct non-negative"),
        ("episodes", "seeds", [1, 1], "distinct non-negative"),
        ("episodes", "episodes_per_day", 0, "must be positive"),
        ("episodes", "evaluate_roles", ["test"], "never listed"),
        ("episodes", "evaluate_roles", ["train", "later"], "unknown role"),
        ("policies", "names", ["twap", "vwap"], "unknown policies"),
        ("policies", "names", ["twap", "twap"], "without duplicates"),
        ("template", "side", 1, "unknown key"),
        ("template", "horizon_buckets", 0, "template:"),
        ("template", "fill_rule", "touch", "template:"),
        ("data", "source", "csv", "data.source must be"),
        ("split", "later", ["2024-01-04"], "unknown key"),
    ],
)
def test_config_rejects_bad_keys_with_a_named_reason(
    section: str, key: str, value: object, match: str
) -> None:
    mapping = synthetic_mapping()
    if section == "top":
        mapping[key] = value
    else:
        mapping[section][key] = value
    with pytest.raises(ConfigError, match=match):
        config_from_mapping(mapping)


def test_config_requires_explicit_split_dates_and_an_evaluated_role_with_days() -> None:
    mapping = synthetic_mapping()
    del mapping["split"]["validation"]
    with pytest.raises(ConfigError, match="missing required key split.validation"):
        config_from_mapping(mapping)
    mapping = synthetic_mapping()
    mapping["split"]["validation"] = []
    with pytest.raises(ConfigError, match="split.validation is empty"):
        config_from_mapping(mapping)
    mapping = synthetic_mapping()
    mapping["split"]["train"] = [DATES[0], DATES[0]]
    with pytest.raises(ConfigError, match="listed under"):
        config_from_mapping(mapping)


def test_archive_source_needs_a_known_market_and_synthetic_source_needs_days() -> None:
    mapping = synthetic_mapping()
    mapping["data"] = {"source": "archive", "market": "synthetic", "symbol": "BTCUSDT"}
    with pytest.raises(ConfigError, match="data.market must be one of"):
        config_from_mapping(mapping)
    mapping["data"] = {"source": "synthetic", "market": "synthetic"}
    with pytest.raises(ConfigError, match="n_buckets must be positive"):
        config_from_mapping(mapping)
    mapping["data"] = {"source": "archive", "market": "futures-um", "symbol": "BTCUSDT"}
    config = config_from_mapping(mapping)
    assert config.data.allow_download is False and config.data.max_download_archives == 3


def test_load_config_reads_toml_and_names_syntax_errors(tmp_path: Path) -> None:
    path = tmp_path / "exp.toml"
    path.write_text(
        'schema = "tapeforge.execution.experiment-config/1"\nname = "t"\nlabel = "smoke"\n'
        "[data]\nsource = \"synthetic\"\nmarket = \"synthetic\"\n[data.synthetic]\n"
        "n_buckets = 100\ndays = [{ date = \"2024-01-01\", seed = 1 }, "
        "{ date = \"2024-01-02\", seed = 2 }]\n"
        "[split]\ntrain = [\"2024-01-01\"]\nvalidation = [\"2024-01-02\"]\n"
        "[episodes]\nseeds = [3]\nepisodes_per_day = 2\n[policies]\nnames = [\"twap\"]\n"
        "[template]\nhorizon_buckets = 5\narrival_lookback = 2\n"
    )
    config = load_config(path)
    assert config.seeds == (3,) and config.rho_choices_bps == (200, 500, 1000)
    assert config.evaluate_roles == ("train", "validation")
    path.write_text("name = [un终\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path)
    tomllib.loads("x = 1")  # The parser used is the standard library's.


def test_episode_identifiers_round_trip() -> None:
    assert episode_identifier("validation", 7, 12) == "validation/s7/i12"
    assert parse_episode_identifier("train/s0/i3") == ("train", 0, 3)
    for bad in ("train/7/3", "later/s1/i1", "train/s1", "train/s-1/i0"):
        with pytest.raises(ValueError, match="not of the form"):
            parse_episode_identifier(bad)


# --- running --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the synthetic experiment once for the module."""
    return run_experiment(synthetic_config(), output_dir=tmp_path_factory.mktemp("runs"),
                          command=["test", "run"])


def test_run_directory_holds_a_manifest_and_one_record_per_policy_role_and_seed(
    run_dir: Path,
) -> None:
    manifest = load_manifest(run_dir)
    assert manifest["schema"] == EXPERIMENT_MANIFEST_SCHEMA and manifest["label"] == "smoke"
    assert manifest["command"] == ["test", "run"]
    assert manifest["simulator"] == {"version": SIMULATOR_VERSION,
                                     "assumptions": SIMULATOR_ASSUMPTIONS}
    assert set(manifest["code_revision"]) == {"commit", "dirty"}
    assert manifest["roles_evaluated"] == ["train", "validation"] and not manifest["include_test"]
    assert [d["date"] for d in manifest["data"]] == list(DATES)
    assert all(d["source"] == "synthetic" and "seed" in d for d in manifest["data"])
    assert manifest["split"]["horizon_buckets"] == 12
    assert {(r["policy"], r["role"], r["seed"]) for r in manifest["runs"]} == {
        (p, role, seed)
        for p in ("twap", "immediate", "cap_aware_twap")
        for role in ("train", "validation")
        for seed in (1, 2)
    }
    for entry in manifest["runs"]:
        record = json.loads((run_dir / entry["record_file"]).read_text())
        assert record["config"]["policy"] == entry["policy"] and not record["held_out"]
        assert len(record["episode_ids"]) == entry["n_episodes"] == len(record["episodes"])
        assert record["summary"] == entry["summary"]
    assert manifest["resources"]["max_rss_bytes"] > 0 and manifest["resources"]["total_seconds"] > 0
    assert manifest["download_budget"]["archives_started"] == 0
    assert not list(run_dir.rglob("*.tmp"))


def test_every_policy_receives_the_same_episodes_in_the_manifest_order(run_dir: Path) -> None:
    manifest = load_manifest(run_dir)
    for role, by_seed in manifest["episodes"].items():
        for seed, listed in by_seed.items():
            ids = [e["id"] for e in listed]
            assert ids == [episode_identifier(role, int(seed), i) for i in range(len(ids))]
            for policy in ("twap", "immediate", "cap_aware_twap"):
                record = json.loads(
                    (run_dir / "records" / f"{policy}-{role}-seed{seed}.json").read_text()
                )
                assert record["episode_ids"] == ids
                assert [e["draw"] for e in record["episodes"]] == [
                    {k: v for k, v in e.items() if k != "id"} for e in listed
                ]
                assert [e["spec"]["target_qty_ticks"] for e in record["episodes"]] == [
                    e["spec"]["target_qty_ticks"]
                    for e in json.loads(
                        (run_dir / "records" / f"twap-{role}-seed{seed}.json").read_text()
                    )["episodes"]
                ]


def test_policy_execution_order_does_not_change_any_outcome(tmp_path: Path, run_dir: Path) -> None:
    reversed_config = synthetic_config(policies={"names": ["cap_aware_twap", "immediate", "twap"]})
    other = run_experiment(reversed_config, output_dir=tmp_path)
    differences = compare_experiments(run_dir, other)
    # Only the policy order in the configuration differs; every record is identical.
    assert differences == ["manifest key 'config' differs"]


def test_the_same_configuration_reproduces_byte_identical_outcomes(
    tmp_path: Path, run_dir: Path
) -> None:
    again = run_experiment(synthetic_config(), output_dir=tmp_path)
    assert compare_experiments(run_dir, again) == []
    changed = run_experiment(synthetic_config(episodes={
        "seeds": [3], "episodes_per_day": 4, "evaluate_roles": ["train", "validation"],
    }), output_dir=tmp_path)
    assert any("run records differ" in d for d in compare_experiments(run_dir, changed))


def test_the_test_role_is_evaluated_only_on_explicit_request(tmp_path: Path) -> None:
    config = synthetic_config(split={"train": [DATES[0]], "validation": [DATES[1]],
                                     "test": [DATES[2]]})
    plain = run_experiment(config, output_dir=tmp_path)
    assert load_manifest(plain)["roles_evaluated"] == ["train", "validation"]
    assert not list((plain / "records").glob("*-test-*"))
    held = run_experiment(config, include_test=True, output_dir=tmp_path)
    manifest = load_manifest(held)
    assert manifest["include_test"] and manifest["roles_evaluated"][-1] == "test"
    test_runs = [r for r in manifest["runs"] if r["role"] == "test"]
    assert test_runs and all(r["held_out"] for r in test_runs)
    assert all(
        json.loads((held / r["record_file"]).read_text())["held_out"] for r in test_runs
    )
    with pytest.raises(ConfigError, match="split.test lists no dates"):
        run_experiment(synthetic_config(), include_test=True, output_dir=tmp_path)


def test_incompatible_data_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    missing_seed = synthetic_mapping()
    missing_seed["data"]["synthetic"]["days"] = missing_seed["data"]["synthetic"]["days"][:2]
    with pytest.raises(ConfigError, match="no entry in data.synthetic.days"):
        run_experiment(config_from_mapping(missing_seed), output_dir=tmp_path)
    too_short = synthetic_config(template={"horizon_buckets": 200, "arrival_lookback": 4})
    with pytest.raises(ValueError, match="no admissible episode start"):
        run_experiment(too_short, output_dir=tmp_path)
    out_of_order = synthetic_config(split={"train": [DATES[2]], "validation": [DATES[0], DATES[1]]})
    with pytest.raises(ValueError, match="chronological"):
        run_experiment(out_of_order, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_run_directories_never_collide_and_json_writes_are_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = create_run_directory(tmp_path, "exp")
    second = create_run_directory(tmp_path, "exp")
    assert first != second and first.exists() and second.exists()
    assert second.name.startswith(first.name[:len("exp-") + 15])
    target = tmp_path / "nested" / "out.json"
    write_json_atomic(target, {"b": 1, "a": [1, 2]})
    assert target.read_text() == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'
    assert not (tmp_path / "nested" / "out.json.tmp").exists()

    import os

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 1}   # The old file survived.


def test_compare_reports_data_and_schema_differences(tmp_path: Path, run_dir: Path) -> None:
    other = run_experiment(synthetic_config(data={
        "source": "synthetic", "market": "synthetic",
        "synthetic": {"n_buckets": 300, "profile": "uniform",
                      "days": [{"date": d, "seed": i + 11} for i, d in enumerate(DATES)]},
    }), output_dir=tmp_path)
    differences = compare_experiments(run_dir, other)
    assert any(d.startswith("data provenance differs") for d in differences)
    manifest = load_manifest(other)
    manifest["schema"] = "tapeforge.execution.experiment/0"
    write_json_atomic(other / "manifest.json", manifest)
    with pytest.raises(ValueError, match="this code reads"):
        load_manifest(other)


TOML = """
schema = "tapeforge.execution.experiment-config/1"
name = "cli-check"
label = "smoke"

[data]
source = "synthetic"
market = "synthetic"

[data.synthetic]
n_buckets = 200
days = [{ date = "2024-01-01", seed = 1 }, { date = "2024-01-02", seed = 2 }]

[split]
train = ["2024-01-01"]
validation = ["2024-01-02"]

[episodes]
seeds = [1]
episodes_per_day = 3

[policies]
names = ["twap", "immediate"]

[template]
horizon_buckets = 10
arrival_lookback = 3
"""


def test_cli_runs_compares_and_reports_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli.toml"
    path.write_text(TOML)
    assert experiment.main(["run", str(path), "--output-dir", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "run directory:" in out and "immediate" in out and "peak RSS" in out
    assert experiment.main(["run", str(path), "--output-dir", str(tmp_path / "runs")]) == 0
    capsys.readouterr()
    first, second = sorted((tmp_path / "runs").iterdir())
    assert experiment.main(["compare", str(first), str(second)]) == 0
    assert "identical outcomes" in capsys.readouterr().out
    path.write_text(TOML.replace('names = ["twap", "immediate"]', 'names = ["twap"]'))
    assert experiment.main(["run", str(path), "--output-dir", str(tmp_path / "runs")]) == 0
    capsys.readouterr()
    third = sorted((tmp_path / "runs").iterdir())[-1]
    assert experiment.main(["compare", str(first), str(third)]) == 1
    assert "run records differ" in capsys.readouterr().out
    path.write_text(TOML.replace('label = "smoke"', 'label = "final"'))
    assert experiment.main(["run", str(path)]) == 1
    assert "label must be one of" in capsys.readouterr().err
    assert experiment.main(["run", str(path), "--include-test"]) == 1
