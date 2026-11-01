"""Cross-run comparison of terminal-rule settings on identical sampled episodes.

``build_sensitivity`` reads several run directories that share data, split,
seeds and episode draws (the terminal rule, the surcharge or ``residual_mu``
may differ) and lines up, per policy and role, completion by fills, final
completion, the settled fraction, the shortfall, the terminal term, the
fees and the risk charge across the runs. Episode identifiers are checked to
be identical between runs, so every number in one row is about the same
episodes; a run whose episodes differ is rejected rather than compared.

What the table does and does not say:
    It compares accounting under different boundary conditions on the same
    tapes. The surcharge values are experimental settings, so the table is
    read for how the ordering and the gap between policies move with the
    setting, not for picking the value under which one policy wins. Day
    counts are carried so the usual uncertainty gate applies.

Not covered:
    No statistical test, no interval; the report module's day-cluster
    bootstrap applies within a run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tapeforge.execution.experiment import load_manifest, write_json_atomic
from tapeforge.execution.report import (
    EpisodeRow,
    episode_rows,
    group_statistics,
    load_records,
    policies_in_order,
    rst_table,
)

__all__ = [
    "SENSITIVITY_SCHEMA",
    "build_sensitivity",
    "render_sensitivity_rst",
    "write_sensitivity",
]

SENSITIVITY_SCHEMA: Final[str] = "tapeforge.execution.sensitivity/1"

# Columns of the per-(run, policy, role) table, in order.
COLUMNS: Final[tuple[str, ...]] = (
    "run", "terminal_rule", "terminal_surcharge_bps", "residual_mu", "policy", "role", "n",
    "n_days", "completion_by_fills_rate", "final_completion_rate", "mean_settlement_fraction",
    "settlement_label", "mean_shortfall_bps", "median_shortfall_bps", "mean_fill_shortfall_bps",
    "mean_terminal_component_bps", "mean_fill_fees_bps", "mean_settlement_fees_bps",
    "mean_risk_bps", "n_truncated",
)


def _setting(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the terminal-rule setting of a manifest's template."""
    template = manifest["config"]["template"]
    return {
        "terminal_rule": template["terminal_rule"],
        "terminal_surcharge_bps": template["terminal_surcharge_bps"],
        "residual_mu": template["residual_mu"],
    }


def _episode_ids(manifest: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Return ``{role: {seed: [ids]}}`` of a manifest."""
    return {
        role: {seed: [e["id"] for e in listed] for seed, listed in by_seed.items()}
        for role, by_seed in manifest["episodes"].items()
    }


def _check_comparable(first: Mapping[str, Any], other: Mapping[str, Any], name: str) -> None:
    """Raise ``ValueError`` unless ``other`` samples the same episodes from the same data."""
    digests = [d.get("sha256") or d.get("seed") for d in first["data"]]
    if digests != [d.get("sha256") or d.get("seed") for d in other["data"]]:
        raise ValueError(f"{name}: data provenance differs from the first run's")
    if first["split"] != other["split"]:
        raise ValueError(f"{name}: split record differs from the first run's")
    if _episode_ids(first) != _episode_ids(other):
        raise ValueError(f"{name}: episode identifiers differ from the first run's")
    geometry = ("horizon_buckets", "latency_buckets", "arrival_lookback", "data_delay_buckets",
                "participation_cap_bps", "fee_bps", "slippage_ticks", "fill_rule")
    a, b = first["config"]["template"], other["config"]["template"]
    different = [k for k in geometry if a[k] != b[k]]
    if different:
        raise ValueError(f"{name}: template fields {different} differ; only the terminal rule, "
                         "surcharge, residual_mu and risk_lambda may vary")


def _row(run: str, setting: Mapping[str, Any], policy: str, role: str,
         rows: Sequence[EpisodeRow]) -> dict[str, Any]:
    """Return one table row from a group's statistics."""
    stats = group_statistics(rows, min_days=10**9)
    terminated = stats["terminated"]
    return {
        "run": run, **setting, "policy": policy, "role": role, "n": stats["n_episodes"],
        "n_days": stats["n_days"],
        "completion_by_fills_rate": stats["completion_by_fills_rate"],
        "final_completion_rate": stats["final_completion_rate"],
        "mean_settlement_fraction": stats["mean_settlement_fraction"],
        "settlement_label": stats["settlement_label"],
        "mean_shortfall_bps": terminated["mean_shortfall_bps"],
        "median_shortfall_bps": terminated["median_shortfall_bps"],
        "mean_fill_shortfall_bps": terminated["mean_fill_shortfall_bps"],
        "mean_terminal_component_bps": terminated["mean_terminal_component_bps"],
        "mean_fill_fees_bps": stats["mean_fill_fees_bps"],
        "mean_settlement_fees_bps": stats["mean_settlement_fees_bps"],
        "mean_risk_bps": stats["mean_risk_bps"],
        "n_truncated": stats["n_truncated"],
    }


def build_sensitivity(run_dirs: Sequence[str | Path], include_test: bool = False) -> dict[str, Any]:
    """Assemble the comparison over ``run_dirs`` (``SENSITIVITY_SCHEMA``).

    Raises:
        ValueError: Fewer than two runs, a run whose episodes, split, data or
            fill geometry differ from the first, or an unreadable manifest.
    """
    if len(run_dirs) < 2:
        raise ValueError("a sensitivity comparison needs at least two run directories")
    manifests = [load_manifest(d) for d in run_dirs]
    names = [Path(d).name for d in run_dirs]
    for manifest, name in zip(manifests[1:], names[1:], strict=True):
        _check_comparable(manifests[0], manifest, name)
    table: list[dict[str, Any]] = []
    for run_dir, manifest, name in zip(run_dirs, manifests, names, strict=True):
        rows = episode_rows(load_records(run_dir, include_test))
        roles = [r for r in manifest["roles_evaluated"] if include_test or r != "test"]
        setting = _setting(manifest)
        for role in roles:
            for policy in policies_in_order(manifest):
                group = [r for r in rows if r.policy == policy and r.role == role]
                table.append(_row(name, setting, policy, role, group))
    return {
        "schema": SENSITIVITY_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": [
            {"run": name, "run_dir": str(d), "label": m["label"], **_setting(m),
             "simulator_version": m["simulator"]["version"], "code_revision": m["code_revision"]}
            for d, m, name in zip(run_dirs, manifests, names, strict=True)
        ],
        "episodes_identical": True,
        "include_test": include_test,
        "columns": list(COLUMNS),
        "table": table,
        "note": (
            "identical sampled episodes under different terminal-rule settings; the surcharge "
            "values are experimental settings, not calibrated impact estimates; read how the "
            "policy ordering moves with the setting rather than choosing the setting that "
            "favours one policy; smoke data support no claim of generalisation"
        ),
    }


def _table_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort rows by role and policy, settlement runs by surcharge first, the legacy rule last."""
    return (row["role"], row["policy"], row["terminal_rule"] != "terminal_settlement",
            row["terminal_surcharge_bps"], row["residual_mu"])


def render_sensitivity_rst(result: Mapping[str, Any]) -> str:
    """Render the comparison as reStructuredText."""
    title = "Terminal-rule sensitivity on identical episodes"
    lines = [title, "=" * len(title), "", result["note"] + ".", "", "Runs", "----", ""]
    lines.append(rst_table(
        ("run", "label", "terminal rule", "surcharge bps", "residual_mu", "simulator"),
        [(r["run"], r["label"], r["terminal_rule"], r["terminal_surcharge_bps"],
          r["residual_mu"], r["simulator_version"]) for r in result["runs"]],
    ))
    lines += ["", "Per policy and role", "-------------------", ""]
    lines.append(rst_table(
        ("policy", "role", "rule", "surcharge", "mu", "n", "days", "completed by fills",
         "final completion", "settled frac", "label", "mean IS bps", "median", "fills-only IS",
         "terminal term bps", "fill fees bps", "settlement fees bps", "risk bps", "truncated"),
        [(t["policy"], t["role"], t["terminal_rule"], t["terminal_surcharge_bps"],
          t["residual_mu"], t["n"], t["n_days"], t["completion_by_fills_rate"],
          t["final_completion_rate"], t["mean_settlement_fraction"], t["settlement_label"],
          t["mean_shortfall_bps"], t["median_shortfall_bps"], t["mean_fill_shortfall_bps"],
          t["mean_terminal_component_bps"], t["mean_fill_fees_bps"],
          t["mean_settlement_fees_bps"], t["mean_risk_bps"], t["n_truncated"])
         for t in sorted(result["table"], key=_table_order)],
    ))
    lines.append("")
    return "\n".join(lines)


def write_sensitivity(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write ``sensitivity.json``, ``sensitivity.csv`` and ``sensitivity.rst`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": write_json_atomic(out / "sensitivity.json", dict(result))}
    csv_path = out / "sensitivity.csv"
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in result["table"]:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    os.replace(tmp, csv_path)
    paths["csv"] = csv_path
    rst = out / "sensitivity.rst"
    tmp = rst.with_name(rst.name + ".tmp")
    tmp.write_text(render_sensitivity_rst(result))
    os.replace(tmp, rst)
    paths["rst"] = rst
    return paths


def main(argv: list[str] | None = None) -> int:
    """``<run_dir> <run_dir> [...] --out DIR [--include-test] [--quiet]``.

    Returns:
        0 on success, 1 when the runs are not comparable or a manifest is bad.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.sensitivity")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_sensitivity(args.run_dirs, args.include_test)
        paths = write_sensitivity(result, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(render_sensitivity_rst(result))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
