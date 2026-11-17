"""One-factor-at-a-time experiment matrices over a base configuration.

A matrix file names a base experiment configuration and a list of variants.
Each variant changes exactly one thing: one template field (an
environment-model change: fee, latency, data delay, terminal surcharge, risk
coefficient, ...) or the split dates (a historical-period change). ``run``
replays the base and every variant with the configured baselines, one run
directory each, and ``summarize`` lines their per-policy statistics up in
one table. Policies evaluated into the run directories afterwards (a trained
checkpoint, through ``rl.ppo evaluate``) appear in the table on the next
``summarize``.

Matrix file (``tapeforge.execution.experiment-matrix/1``)::

    schema = "tapeforge.execution.experiment-matrix/1"
    name = "pilot-one-factor"
    base_config = "experiments/configs/pilot-futures-btcusdt-2023-12.toml"

    [[variants]]
    name = "fee-5bps"
    template = { fee_bps = 5 }

    [[variants]]
    name = "period-early"
    [variants.split]
    train = ["2023-12-01", "2023-12-02"]
    validation = ["2023-12-03"]

How to read the table:
    * A template change that alters the episode geometry (horizon, latency,
      arrival lookback, data delay) changes which starts are admissible, so
      the same seed draws other episodes; such a variant is marked
      ``episodes_identical = false`` and its rows are not paired with the
      base. Other template changes replay the base's episodes exactly.
    * A period variant uses other days and is never paired.
    * The table reports what the same policies cost under other
      assumptions. A policy trained under the base settings and evaluated
      under a variant was not optimised for it; retraining per variant is a
      separate, larger sweep.
    * A variant may not move a date the base configuration seals as test
      into its train or validation roles, and the test role is never
      evaluated here.
    * Feature-set changes are a different kind of experiment and are not
      expressed in this file: the observation layout is part of the
      simulator version.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tapeforge.execution.data.datasets import DatasetError
from tapeforge.execution.data.splits import SplitDates
from tapeforge.execution.research.experiment import (
    ConfigError,
    ExperimentConfig,
    acquire_days,
    load_config,
    load_manifest,
    run_experiment,
    write_json_atomic,
)
from tapeforge.execution.research.report import (
    episode_rows,
    group_statistics,
    load_records,
    policies_in_order,
    rst_table,
)
from tapeforge.execution.spec import EpisodeSpec

__all__ = [
    "GEOMETRY_FIELDS",
    "MATRIX_SCHEMA",
    "MATRIX_SUMMARY_SCHEMA",
    "Matrix",
    "Variant",
    "load_matrix",
    "matrix_from_mapping",
    "run_matrix",
    "summarize_matrix",
    "variant_config",
]

MATRIX_SCHEMA: Final[str] = "tapeforge.execution.experiment-matrix/1"
MATRIX_SUMMARY_SCHEMA: Final[str] = "tapeforge.execution.experiment-matrix-summary/1"
BASE_NAME: Final[str] = "base"
# Template fields that decide which episode starts are admissible; changing one
# re-draws the episodes of a seed.
GEOMETRY_FIELDS: Final[frozenset[str]] = frozenset(
    {"horizon_buckets", "latency_buckets", "arrival_lookback", "data_delay_buckets"}
)
_TEMPLATE_FIELDS: Final[frozenset[str]] = frozenset(
    f.name for f in fields(EpisodeSpec) if f.name not in ("side", "target_qty_ticks")
)
COLUMNS: Final[tuple[str, ...]] = (
    "variant", "kind", "factor", "value", "episodes_identical", "policy", "role", "n", "n_days",
    "completion_by_fills_rate", "final_completion_rate", "mean_settlement_fraction",
    "mean_shortfall_bps", "median_shortfall_bps", "p95_shortfall_bps",
    "mean_terminal_component_bps", "mean_fees_bps", "mean_risk_bps", "n_truncated",
)


@dataclass(frozen=True)
class Variant:
    """One change against the base configuration.

    Attributes:
        name: Short identifier; part of the run directory name.
        template: The single template field changed, as ``{field: value}``;
            empty for a period variant.
        split: The replacement split dates of a period variant, else ``None``.
    """

    name: str
    template: tuple[tuple[str, Any], ...] = ()
    split: SplitDates | None = None

    @property
    def kind(self) -> str:
        """Return ``"environment"`` for a template change, ``"period"`` for a split change."""
        return "period" if self.split is not None else "environment"

    @property
    def factor(self) -> str:
        """Return the name of what changed."""
        return "split" if self.split is not None else self.template[0][0]

    @property
    def value(self) -> Any:
        """Return the changed value (the date ranges of a period variant)."""
        if self.split is None:
            return self.template[0][1]
        return "; ".join(
            f"{role} {dates[0]}..{dates[-1]}" for role, dates in
            (("train", self.split.train), ("validation", self.split.validation)) if dates
        )


@dataclass(frozen=True)
class Matrix:
    """A base configuration and its one-factor variants.

    Attributes:
        name: Matrix name; the output directory is
            ``<output_dir>/<name>-<utc>``.
        description: Free text copied into the summary.
        base_config: Path of the base experiment configuration.
        variants: The variants, in file order.
    """

    name: str
    description: str
    base_config: str
    variants: tuple[Variant, ...]


def matrix_from_mapping(mapping: Mapping[str, Any]) -> Matrix:
    """Build a ``Matrix`` from a parsed TOML mapping, strictly.

    Raises:
        ConfigError: Wrong schema, an unknown or missing key, duplicate or
            reserved variant names, or a variant that does not change
            exactly one template field or the split.
    """
    allowed = ("schema", "name", "description", "base_config", "variants")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) at top level: {unknown}")
    if mapping.get("schema") != MATRIX_SCHEMA:
        raise ConfigError(f"schema {mapping.get('schema')!r} is not {MATRIX_SCHEMA!r}")
    for key in ("name", "base_config", "variants"):
        if key not in mapping:
            raise ConfigError(f"missing required key {key}")
    variants: list[Variant] = []
    for entry in mapping["variants"]:
        unknown = sorted(set(entry) - {"name", "template", "split"})
        if unknown or "name" not in entry:
            raise ConfigError(f"a variant needs a name and allows template or split: {entry}")
        name = str(entry["name"])
        template, split = entry.get("template"), entry.get("split")
        if (template is None) == (split is None):
            raise ConfigError(f"variant {name!r} must change either one template field or split")
        if template is not None:
            if len(template) != 1:
                raise ConfigError(
                    f"variant {name!r} changes {sorted(template)}; one factor at a time"
                )
            (field_name, value), = template.items()
            if field_name not in _TEMPLATE_FIELDS:
                raise ConfigError(f"variant {name!r}: unknown template field {field_name!r}")
            variants.append(Variant(name=name, template=((field_name, value),)))
        else:
            unknown = sorted(set(split) - {"train", "validation"})
            if unknown or not split.get("train") or not split.get("validation"):
                raise ConfigError(
                    f"variant {name!r}: split needs train and validation dates and nothing else"
                )
            variants.append(Variant(name=name, split=SplitDates(
                train=tuple(split["train"]), validation=tuple(split["validation"]), test=(),
            )))
    names = [v.name for v in variants]
    if not variants or len(set(names)) < len(names) or BASE_NAME in names:
        raise ConfigError(f"variants need distinct names other than {BASE_NAME!r}")
    return Matrix(name=str(mapping["name"]), description=str(mapping.get("description", "")),
                  base_config=str(mapping["base_config"]), variants=tuple(variants))


def load_matrix(path: str | Path) -> Matrix:
    """Parse a matrix file.

    Raises:
        ConfigError: Invalid TOML or ``matrix_from_mapping`` fails.
        FileNotFoundError: No such file.
    """
    try:
        mapping = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return matrix_from_mapping(mapping)


def variant_config(base: ExperimentConfig, variant: Variant) -> ExperimentConfig:
    """Return ``base`` with the variant's single change applied.

    Raises:
        ConfigError: The changed template is invalid, the variant equals the
            base, a period variant uses a date the base does not list, or it
            moves a sealed test date into train or validation.
    """
    name = f"{base.name}--{variant.name}"
    if variant.split is None:
        (field_name, value), = variant.template
        if getattr(base.template, field_name) == value:
            raise ConfigError(f"variant {variant.name!r} does not differ from the base")
        try:
            template = replace(base.template, **{field_name: value})
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"variant {variant.name!r}: {exc}") from exc
        return replace(base, name=name, template=template)
    used = set(variant.split.train) | set(variant.split.validation)
    sealed = sorted(used & set(base.split.test))
    if sealed:
        raise ConfigError(
            f"variant {variant.name!r} uses {sealed}, which the base configuration seals as "
            "test dates"
        )
    known = set(base.split.train) | set(base.split.validation)
    unknown = sorted(used - known)
    if unknown:
        raise ConfigError(f"variant {variant.name!r} uses dates the base does not list: {unknown}")
    return replace(base, name=name, split=variant.split)


def run_matrix(
    matrix: Matrix, *, output_dir: str | Path | None = None, only: Sequence[str] | None = None
) -> Path:
    """Run the base and the variants (all, or those named in ``only``) and summarise them.

    The days are loaded once and shared. Returns the matrix directory, which
    holds one run directory per configuration, ``matrix.json`` (what was
    run) and the summary files.

    Raises:
        ConfigError: An invalid variant or an unknown name in ``only``.
        DatasetError: The base configuration's data is not available.
    """
    base = load_config(matrix.base_config)
    chosen = [v for v in matrix.variants if only is None or v.name in only]
    if only is not None and len(chosen) != len(set(only)):
        known = [v.name for v in matrix.variants]
        raise ConfigError(f"--only names {sorted(set(only) - set(known))} are not variants")
    configs = [(BASE_NAME, None, base)] + [(v.name, v, variant_config(base, v)) for v in chosen]
    tapes, provenance = acquire_days(base)
    preloaded = {tape.date: (tape, record) for tape, record in zip(tapes, provenance, strict=True)}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir or base.output_dir) / f"{matrix.name}-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    for name, variant, config in configs:
        run_dir = run_experiment(config, output_dir=root, preloaded=preloaded)
        runs.append({
            "variant": name,
            "kind": "base" if variant is None else variant.kind,
            "factor": None if variant is None else variant.factor,
            "value": None if variant is None else variant.value,
            "run_dir": run_dir.name,
        })
    write_json_atomic(root / "matrix.json", {
        "schema": MATRIX_SCHEMA, "name": matrix.name, "description": matrix.description,
        "base_config": matrix.base_config, "created_utc": stamp, "runs": runs,
    })
    summarize_matrix(root)
    return root


def _episode_ids(manifest: Mapping[str, Any]) -> Any:
    """Return the episode identifiers with their draws, which is what pairing depends on."""
    return {role: {seed: [(e["id"], e["position"], e["side"], e["rho_bps"]) for e in listed]
                   for seed, listed in by_seed.items()}
            for role, by_seed in manifest["episodes"].items()}


def summarize_matrix(matrix_dir: str | Path) -> dict[str, Any]:
    """Rebuild and write the summary of a matrix directory; returns it.

    Reads every run listed in ``matrix.json``, including policies evaluated
    into them after the matrix ran, and writes ``summary.json``,
    ``summary.csv`` and ``summary.rst``.

    Raises:
        FileNotFoundError: No ``matrix.json`` or a missing run.
        ValueError: A run manifest cannot be read under this version.
    """
    root = Path(matrix_dir)
    listing = json.loads((root / "matrix.json").read_text())
    base_manifest = load_manifest(root / listing["runs"][0]["run_dir"])
    base_ids = _episode_ids(base_manifest)
    table: list[dict[str, Any]] = []
    for run in listing["runs"]:
        run_dir = root / run["run_dir"]
        manifest = load_manifest(run_dir)
        identical = manifest["split"] == base_manifest["split"] and (
            _episode_ids(manifest) == base_ids
        )
        rows = episode_rows(load_records(run_dir))
        for role in [r for r in manifest["roles_evaluated"] if r != "test"]:
            for policy in policies_in_order(manifest):
                stats = group_statistics(
                    [r for r in rows if r.policy == policy and r.role == role], min_days=10**9
                )
                terminated = stats["terminated"]
                table.append({
                    "variant": run["variant"], "kind": run["kind"], "factor": run["factor"],
                    "value": run["value"], "episodes_identical": identical, "policy": policy,
                    "role": role, "n": stats["n_episodes"], "n_days": stats["n_days"],
                    "completion_by_fills_rate": stats["completion_by_fills_rate"],
                    "final_completion_rate": stats["final_completion_rate"],
                    "mean_settlement_fraction": stats["mean_settlement_fraction"],
                    "mean_shortfall_bps": terminated["mean_shortfall_bps"],
                    "median_shortfall_bps": terminated["median_shortfall_bps"],
                    "p95_shortfall_bps": terminated["p95_shortfall_bps"],
                    "mean_terminal_component_bps": terminated["mean_terminal_component_bps"],
                    "mean_fees_bps": stats["mean_fees_bps"],
                    "mean_risk_bps": stats["mean_risk_bps"],
                    "n_truncated": stats["n_truncated"],
                })
    summary = {
        "schema": MATRIX_SUMMARY_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "matrix": listing,
        "label": base_manifest["label"],
        "columns": list(COLUMNS),
        "table": table,
        "note": (
            "one factor changed per variant against the base; episodes_identical says whether "
            "a variant replays the base's episodes (unpaired otherwise); settings are labelled "
            "experiment values, not calibrated estimates; the held-out test role is never read"
        ),
    }
    write_json_atomic(root / "summary.json", summary)
    tmp = root / "summary.rst.tmp"
    tmp.write_text(render_matrix_rst(summary))
    os.replace(tmp, root / "summary.rst")
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(table)
    os.replace(tmp, root / "summary.csv")
    return summary


def render_matrix_rst(summary: Mapping[str, Any]) -> str:
    """Render a matrix summary as reStructuredText, one table per role."""
    matrix = summary["matrix"]
    title = f"Experiment matrix: {matrix['name']}"
    lines = [title, "=" * len(title), "", f"Base configuration ``{matrix['base_config']}`` "
             f"(label ``{summary['label']}``). {summary['note'].capitalize()}.", ""]
    roles = sorted({row["role"] for row in summary["table"]})
    for role in roles:
        lines += [role.capitalize(), "-" * len(role), ""]
        lines.append(rst_table(
            ("variant", "factor", "value", "same episodes", "policy", "n", "by fills",
             "settled frac", "mean IS bps", "median", "p95", "fees bps", "risk bps"),
            [(r["variant"], r["factor"], r["value"], r["episodes_identical"], r["policy"],
              r["n"], r["completion_by_fills_rate"], r["mean_settlement_fraction"],
              r["mean_shortfall_bps"], r["median_shortfall_bps"], r["p95_shortfall_bps"],
              r["mean_fees_bps"], r["mean_risk_bps"])
             for r in summary["table"] if r["role"] == role],
        ))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``run <matrix.toml> [--only NAME ...] [--output-dir DIR]`` or ``summarize <matrix_dir>``.

    Returns:
        0 on success; 1 on a configuration or data failure, with the reason
        on stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.research.matrix")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the base and its one-factor variants")
    run.add_argument("matrix", help="matrix file (TOML)")
    run.add_argument("--only", action="append", default=None, help="run this variant only")
    run.add_argument("--output-dir", default=None)
    summarize = sub.add_parser("summarize", help="rebuild the summary of a matrix directory")
    summarize.add_argument("matrix_dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            root = run_matrix(load_matrix(args.matrix), output_dir=args.output_dir, only=args.only)
        else:
            root = Path(args.matrix_dir)
            summarize_matrix(root)
    except (ConfigError, DatasetError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"matrix directory: {root}")
    print((root / "summary.rst").read_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
