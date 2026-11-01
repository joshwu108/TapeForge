"""Tests for the cross-run sensitivity table and the environment checker command.

Both run the synthetic experiment of ``test_execution.configs`` under several
terminal-rule settings; the checker tests skip without gymnasium.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tapeforge.execution import sensitivity as sensitivity_module
from tapeforge.execution.experiment import run_experiment
from tapeforge.execution.report import build_report
from tapeforge.execution.sensitivity import (
    SENSITIVITY_SCHEMA,
    build_sensitivity,
    render_sensitivity_rst,
    write_sensitivity,
)
from test_execution.configs import synthetic_config, synthetic_mapping

TEMPLATE = {"horizon_buckets": 12, "arrival_lookback": 4}


def _template(**overrides: object) -> dict[str, object]:
    return {**TEMPLATE, **overrides}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the synthetic experiment at 0 and 25 bps and under the legacy rule."""
    out = tmp_path_factory.mktemp("runs")
    return {
        "s0": run_experiment(synthetic_config(
            name="s0", template=_template(terminal_rule="terminal_settlement",
                                          terminal_surcharge_bps=0),
            policies={"names": ["twap", "zero_action"]},
        ), output_dir=out),
        "s25": run_experiment(synthetic_config(
            name="s25", template=_template(terminal_rule="terminal_settlement",
                                           terminal_surcharge_bps=25),
            policies={"names": ["twap", "zero_action"]},
        ), output_dir=out),
        "legacy": run_experiment(synthetic_config(
            name="legacy", template=_template(terminal_rule="residual_perold"),
            policies={"names": ["twap", "zero_action"]},
        ), output_dir=out),
    }


def test_sensitivity_lines_up_identical_episodes_across_settings(runs: dict[str, Path]) -> None:
    result = build_sensitivity([runs["s0"], runs["s25"], runs["legacy"]])
    assert result["schema"] == SENSITIVITY_SCHEMA and result["episodes_identical"]
    assert [r["terminal_surcharge_bps"] for r in result["runs"]] == [0, 25, 0]
    assert [r["terminal_rule"] for r in result["runs"]] == [
        "terminal_settlement", "terminal_settlement", "residual_perold",
    ]
    table = result["table"]
    assert len(table) == 3 * 2 * 2   # Runs x policies x roles.
    by_key = {(t["run"], t["policy"], t["role"]): t for t in table}
    # Every row is reproduced by the per-run report on the same group.
    for (run, policy, role), row in by_key.items():
        group = next(g for g in build_report(runs[run.split("-")[0]])["per_policy"]
                     if g["policy"] == policy and g["role"] == role)
        assert row["mean_shortfall_bps"] == group["terminated"]["mean_shortfall_bps"]
        assert row["final_completion_rate"] == group["final_completion_rate"]
        assert row["mean_settlement_fraction"] == group["mean_settlement_fraction"]
        assert row["n"] == group["n_episodes"]
    # Zero action settles everything under settlement and settles nothing under the legacy rule.
    for role in ("train", "validation"):
        s0 = by_key[(runs["s0"].name, "zero_action", role)]
        s25 = by_key[(runs["s25"].name, "zero_action", role)]
        legacy = by_key[(runs["legacy"].name, "zero_action", role)]
        assert s0["mean_settlement_fraction"] == s25["mean_settlement_fraction"] == 1.0
        assert s0["final_completion_rate"] == 1.0 and s0["completion_by_fills_rate"] == 0.0
        assert legacy["mean_settlement_fraction"] == 0.0 and legacy["final_completion_rate"] == 0.0
        # The surcharge is a monotone adverse adjustment: 25 bps costs more than 0 bps.
        assert s25["mean_terminal_component_bps"] > s0["mean_terminal_component_bps"]
        # The fee avoidance of the legacy rule: no settlement fee, no fill fee for zero action.
        assert legacy["mean_settlement_fees_bps"] == 0.0 and legacy["mean_fill_fees_bps"] == 0.0
        assert s0["mean_settlement_fees_bps"] > 0.0
    text = render_sensitivity_rst(result)
    assert "identical sampled episodes" in text and "residual_perold" in text


def test_sensitivity_refuses_runs_with_different_episodes_or_geometry(
    runs: dict[str, Path], tmp_path: Path
) -> None:
    other_seed = run_experiment(synthetic_config(
        name="seed9", episodes={"seeds": [9], "episodes_per_day": 4,
                                "evaluate_roles": ["train", "validation"]},
        template=_template(terminal_rule="terminal_settlement", terminal_surcharge_bps=0),
        policies={"names": ["twap", "zero_action"]},
    ), output_dir=tmp_path)
    with pytest.raises(ValueError, match="episode identifiers differ"):
        build_sensitivity([runs["s0"], other_seed])
    other_fee = run_experiment(synthetic_config(
        name="fee", template=_template(terminal_rule="terminal_settlement",
                                       terminal_surcharge_bps=0, fee_bps=20),
        policies={"names": ["twap", "zero_action"]},
    ), output_dir=tmp_path)
    with pytest.raises(ValueError, match="fee_bps"):
        build_sensitivity([runs["s0"], other_fee])
    other_data = run_experiment(synthetic_config(
        name="data", template=_template(terminal_rule="terminal_settlement",
                                        terminal_surcharge_bps=0),
        policies={"names": ["twap", "zero_action"]},
        data={"source": "synthetic", "market": "synthetic", "synthetic": {
            "n_buckets": 300, "profile": "uniform",
            "days": [{"date": d["date"], "seed": d["seed"] + 20}
                     for d in synthetic_mapping()["data"]["synthetic"]["days"]],
        }},
    ), output_dir=tmp_path)
    with pytest.raises(ValueError, match="data provenance differs"):
        build_sensitivity([runs["s0"], other_data])
    with pytest.raises(ValueError, match="at least two"):
        build_sensitivity([runs["s0"]])


def test_sensitivity_cli_writes_json_csv_and_rst(
    runs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sens"
    assert sensitivity_module.main([str(runs["s0"]), str(runs["s25"]), "--out", str(out),
                                    "--quiet"]) == 0
    assert "rst:" in capsys.readouterr().out
    result = json.loads((out / "sensitivity.json").read_text())
    assert result["schema"] == SENSITIVITY_SCHEMA
    with (out / "sensitivity.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result["table"]) and set(rows[0]) == set(sensitivity_module.COLUMNS)
    paths = write_sensitivity(result, out)
    assert set(paths) == {"json", "csv", "rst"} and not list(out.glob("*.tmp"))
    assert sensitivity_module.main([str(runs["s0"]), str(tmp_path / "missing"),
                                    "--out", str(out)]) == 1
    assert "ERROR" in capsys.readouterr().err


# --- environment checker ------------------------------------------------------------------------


def test_env_check_passes_on_the_synthetic_configuration(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from tapeforge.execution import env_check
    from tapeforge.execution.evaluation import training_config_from_mapping as tc
    from tapeforge.execution.evaluation import training_statistics
    from tapeforge.execution.experiment import rebuild_splits
    from tapeforge.execution.normalization import save_statistics

    mapping = synthetic_mapping()
    path = tmp_path / "exp.toml"
    _write_toml(path, mapping)
    report = env_check.check_environment(path, "train", n_episodes=4, seed=3)
    assert report.n_episodes == 4 and report.n_steps >= 4
    assert report.n_terminated + report.n_truncated == 4
    assert report.simulator_version.endswith("/2")
    assert all("infinity" in w for w in report.checker_warnings)
    config = synthetic_config()
    stats = training_statistics(config, rebuild_splits(config), tc({
        "schema": "tapeforge.execution.training-config/1", "experiment_config": str(path),
        "policy_name": "p", "training": {"role": "train", "seeds": [1]},
        "normalization": {"reference_policy": "twap", "seed": 1},
        "evaluation": {"roles": ["validation"], "seeds": [1]},
    }))
    stats_path = save_statistics(stats, tmp_path / "stats.json")
    scaled = env_check.check_environment(path, "validation", n_episodes=2, seed=1,
                                         statistics_path=stats_path, reward_unit="ticks",
                                         run_gym_checker=False)
    assert scaled.statistics_file == str(stats_path) and scaled.checker_warnings == ()
    assert env_check.main([str(path), "--episodes", "2", "--no-gym-checker"]) == 0
    assert env_check.main([str(path), "--role", "test", "--episodes", "1"]) == 1
    with pytest.raises(env_check.ConfigError, match="held-out"):
        env_check.check_environment(path, "test")
    with pytest.raises(ValueError, match="positive"):
        env_check.check_environment(path, "train", n_episodes=0)


def _write_toml(path: Path, mapping: dict[str, object]) -> None:
    """Serialise the synthetic test mapping as TOML (only the shapes this mapping uses)."""
    data = mapping["data"]
    synthetic = data["synthetic"]  # type: ignore[index]
    days = ", ".join(f'{{ date = "{d["date"]}", seed = {d["seed"]} }}' for d in synthetic["days"])
    split = mapping["split"]
    episodes = mapping["episodes"]
    template = mapping["template"]
    lines = [
        f'schema = "{mapping["schema"]}"', f'name = "{mapping["name"]}"',
        f'label = "{mapping["label"]}"', "[data]", 'source = "synthetic"',
        'market = "synthetic"', "[data.synthetic]", f'n_buckets = {synthetic["n_buckets"]}',
        f'profile = "{synthetic["profile"]}"', f"days = [{days}]", "[split]",
    ]
    for role in ("train", "validation", "test"):
        lines.append(f"{role} = {json.dumps(split[role])}")  # type: ignore[index]
    lines += ["[episodes]"]
    for key, value in episodes.items():  # type: ignore[union-attr]
        lines.append(f"{key} = {json.dumps(value)}")
    lines += ["[policies]", f'names = {json.dumps(mapping["policies"]["names"])}',  # type: ignore[index]
              "[template]"]
    for key, value in template.items():  # type: ignore[union-attr]
        lines.append(f"{key} = {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n")
