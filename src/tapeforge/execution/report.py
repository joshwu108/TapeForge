"""Baseline evaluation report over an experiment's run records.

``build_report`` reads a run directory written by ``experiment`` and
produces per-policy, per-day and paired statistics; ``write_report`` exports
them as JSON (``REPORT_SCHEMA``), CSV tables and a readable RST page.

Outcome classes (never mixed in one statistic):
    * ``completed``: terminated with no residual; shortfall is fills plus
      fees.
    * ``incomplete``: terminated at the horizon with residual inventory;
      shortfall adds the residual term (Perold opportunity cost at the
      terminal mark plus ``residual_mu`` per residual tick) under the
      configured ``residual_mu``.
    * ``truncated``: the tape ran out; the record carries the shortfall of
      the fills alone, which is a different quantity, so truncated episodes
      are counted and described separately and excluded from the shortfall
      statistics and from pairing.

    The "terminated" statistics pool completed and incomplete episodes, which
    share one definition (implementation shortfall including the residual
    term); the completed-only statistics are shown beside them so the
    residual term's share is visible.

Paired differences:
    Every policy of a run replays the same episode identifiers, so a policy
    is compared with the reference on identical episodes: the difference of
    their shortfalls per episode, in ticks and basis points, over the pairs
    where both terminated. Pairs with a truncation on either side are
    counted as excluded.

Uncertainty:
    Episodes of one day are not independent (they overlap in time and share
    the day's regime), so intervals treat the calendar day as the sampling
    unit: a day-cluster bootstrap resamples days with replacement and pools
    their episodes. It is reported only when a group holds at least
    ``MIN_DAYS_FOR_UNCERTAINTY`` distinct days; a one-day smoke run reports
    ``insufficient_days`` rather than a misleading interval.

Held-out period:
    Records tagged ``held_out`` (the test role) are left out unless
    ``include_test`` is set, so the report cannot be used to tune on the
    final period by accident.

Not covered:
    No plotting, no training, no economic interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from tapeforge.execution.experiment import (
    EXPERIMENT_MANIFEST_SCHEMA,
    load_manifest,
    write_json_atomic,
)
from tapeforge.execution.runner import RUN_RECORD_SCHEMA, nearest_rank_percentile

__all__ = [
    "MIN_DAYS_FOR_UNCERTAINTY",
    "REPORT_SCHEMA",
    "EpisodeRow",
    "build_report",
    "day_cluster_bootstrap",
    "episode_rows",
    "group_statistics",
    "load_records",
    "paired_differences",
    "write_report",
]

REPORT_SCHEMA: Final[str] = "tapeforge.execution.report/1"
MIN_DAYS_FOR_UNCERTAINTY: Final[int] = 5   # Distinct days needed before an interval is shown.
BOOTSTRAP_RESAMPLES: Final[int] = 2_000
BOOTSTRAP_SEED: Final[int] = 0
REPORT_SUBDIR: Final[str] = "report"

OUTCOMES: Final[tuple[str, ...]] = ("completed", "incomplete", "truncated")

# Columns of episodes.csv, in order.
EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "policy", "role", "held_out", "episode_id", "seed", "index", "date", "side", "rho_bps",
    "target_qty_ticks", "arrival_price_ticks", "outcome", "n_steps", "filled_qty_ticks",
    "residual_qty_ticks", "fees_ticks", "shortfall_ticks", "shortfall_bps",
    "fill_shortfall_ticks", "terminal_component_ticks", "terminal_price_ticks",
    "n_capped_steps", "risk_sum_ticks", "reward_sum_ticks",
)


@dataclass(frozen=True)
class EpisodeRow:
    """One episode of one policy, flattened from a run record.

    Attributes:
        policy: Policy name.
        role: Split role.
        held_out: Whether the role is the test role.
        episode_id: ``experiment.episode_identifier`` string.
        seed: Sampling seed.
        index: Draw index.
        date: Calendar day of the start bucket.
        side: ``+1`` buy, ``-1`` sell.
        rho_bps: Target fraction drawn.
        target_qty_ticks: Target in quantity ticks.
        arrival_price_ticks: Arrival price.
        outcome: One of ``OUTCOMES``.
        n_steps: Decisions taken.
        filled_qty_ticks: Filled quantity.
        residual_qty_ticks: Unfilled quantity.
        fees_ticks: Fees paid, cash ticks.
        shortfall_ticks: Shortfall as the record defines it for the outcome.
        shortfall_bps: The same in basis points of the arrival notional.
        fill_shortfall_ticks: Shortfall of the fills alone (no residual term).
        terminal_component_ticks: Residual term charged, cash ticks.
        terminal_price_ticks: Terminal mark used, 0 when none.
        n_capped_steps: Fills cut by the participation cap.
        risk_sum_ticks: Sum of the risk component.
        reward_sum_ticks: Sum of the per-step rewards.
    """

    policy: str
    role: str
    held_out: bool
    episode_id: str
    seed: int
    index: int
    date: str
    side: int
    rho_bps: int
    target_qty_ticks: int
    arrival_price_ticks: int
    outcome: str
    n_steps: int
    filled_qty_ticks: int
    residual_qty_ticks: int
    fees_ticks: int
    shortfall_ticks: int
    shortfall_bps: float
    fill_shortfall_ticks: int
    terminal_component_ticks: int
    terminal_price_ticks: int
    n_capped_steps: int
    risk_sum_ticks: int
    reward_sum_ticks: int

    @property
    def terminated(self) -> bool:
        """Return whether the episode ended by completion or at the horizon."""
        return self.outcome != "truncated"


# --- loading ------------------------------------------------------------------------------


def load_records(run_dir: str | Path, include_test: bool = False) -> list[dict[str, Any]]:
    """Return the run records listed in the manifest, skipping held-out ones unless asked.

    Raises:
        ValueError: A record's schema is not ``RUN_RECORD_SCHEMA``.
    """
    manifest = load_manifest(run_dir)
    records = []
    for entry in manifest["runs"]:
        if entry["held_out"] and not include_test:
            continue
        record = json.loads((Path(run_dir) / entry["record_file"]).read_text())
        if record.get("schema") != RUN_RECORD_SCHEMA:
            raise ValueError(
                f"{entry['record_file']} has schema {record.get('schema')!r}; expected "
                f"{RUN_RECORD_SCHEMA!r}"
            )
        records.append(record)
    return records


def _outcome(summary: Mapping[str, Any]) -> str:
    """Classify a record's episode summary."""
    if summary["truncated"]:
        return "truncated"
    return "completed" if summary["completed"] else "incomplete"


def episode_rows(records: Iterable[Mapping[str, Any]]) -> list[EpisodeRow]:
    """Flatten run records into one ``EpisodeRow`` per episode and policy."""
    rows: list[EpisodeRow] = []
    for record in records:
        config = record["config"]
        ids = record.get("episode_ids") or [
            f"{config['role']}/s{config['seed']}/i{e['draw']['index']}" for e in record["episodes"]
        ]
        for episode_id, episode in zip(ids, record["episodes"], strict=True):
            draw, summary = episode["draw"], episode["summary"]
            rows.append(
                EpisodeRow(
                    policy=config["policy"],
                    role=config["role"],
                    held_out=bool(record.get("held_out", config["role"] == "test")),
                    episode_id=episode_id,
                    seed=int(draw["seed"]),
                    index=int(draw["index"]),
                    date=str(draw["date"]),
                    side=int(draw["side"]),
                    rho_bps=int(draw["rho_bps"]),
                    target_qty_ticks=int(summary["target_qty_ticks"]),
                    arrival_price_ticks=int(summary["arrival_price_ticks"]),
                    outcome=_outcome(summary),
                    n_steps=int(summary["n_steps"]),
                    filled_qty_ticks=int(summary["filled_qty_ticks"]),
                    residual_qty_ticks=int(summary["residual_qty_ticks"]),
                    fees_ticks=int(summary["fees_ticks"]),
                    shortfall_ticks=int(summary["shortfall_ticks"]),
                    shortfall_bps=float(summary["shortfall_bps"]),
                    fill_shortfall_ticks=int(summary["shortfall_ticks"])
                    - int(summary["terminal_component_ticks"]),
                    terminal_component_ticks=int(summary["terminal_component_ticks"]),
                    terminal_price_ticks=int(summary["terminal_price_ticks"]),
                    n_capped_steps=int(summary["n_capped_steps"]),
                    risk_sum_ticks=int(summary["risk_sum_ticks"]),
                    reward_sum_ticks=int(summary["reward_sum_ticks"]),
                )
            )
    return rows


# --- statistics -----------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for no values."""
    return sum(values) / len(values) if values else None


def _bps(ticks: int, row: EpisodeRow) -> float:
    """Return ``ticks`` as basis points of the row's arrival notional."""
    return ticks * 10_000 / (row.arrival_price_ticks * row.target_qty_ticks)


def day_cluster_bootstrap(
    values_by_day: Mapping[str, Sequence[float]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    min_days: int = MIN_DAYS_FOR_UNCERTAINTY,
) -> dict[str, Any]:
    """Return a 95 % day-cluster bootstrap interval for the pooled mean, or why there is none.

    Days are resampled with replacement ``n_resamples`` times; each resample
    pools the episodes of the chosen days and takes their mean. The interval
    is the 2.5th and 97.5th percentiles of those means.

    Returns:
        ``{"status": "ok", "n_days", "n_episodes", "ci95_low", "ci95_high",
        "n_resamples", "seed", "method"}`` when at least ``min_days`` days
        hold values, else ``{"status": "insufficient_days", "n_days",
        "min_days"}``.
    """
    days = {day: list(values) for day, values in values_by_day.items() if values}
    if len(days) < min_days:
        return {"status": "insufficient_days", "n_days": len(days), "min_days": min_days}
    names = sorted(days)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        picked = rng.integers(len(names), size=len(names))
        pooled = np.concatenate([np.asarray(days[names[j]], dtype=np.float64) for j in picked])
        means[i] = pooled.mean()
    return {
        "status": "ok",
        "method": "day-cluster bootstrap of the pooled mean, percentile interval",
        "n_days": len(names),
        "n_episodes": sum(len(v) for v in days.values()),
        "n_resamples": n_resamples,
        "seed": seed,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def group_statistics(
    rows: Sequence[EpisodeRow], min_days: int = MIN_DAYS_FOR_UNCERTAINTY
) -> dict[str, Any]:
    """Return the statistics of one group of rows (one policy and role, or one day of them).

    See the module docstring for which rows enter which figure. Every value
    is reproducible from ``episodes.csv`` with a plain loop; the percentiles
    are nearest-rank.
    """
    terminated = [r for r in rows if r.terminated]
    completed = [r for r in rows if r.outcome == "completed"]
    truncated = [r for r in rows if r.outcome == "truncated"]
    bps = [r.shortfall_bps for r in terminated]
    out: dict[str, Any] = {
        "n_episodes": len(rows),
        "n_completed": len(completed),
        "n_incomplete": sum(1 for r in rows if r.outcome == "incomplete"),
        "n_truncated": len(truncated),
        "completion_rate": len(completed) / len(rows) if rows else None,
        "truncation_rate": len(truncated) / len(rows) if rows else None,
        "n_days": len({r.date for r in rows}),
        "mean_steps": _mean([r.n_steps for r in rows]),
        "mean_capped_steps": _mean([r.n_capped_steps for r in rows]),
        "mean_residual_fraction": _mean(
            [r.residual_qty_ticks / r.target_qty_ticks for r in rows]
        ),
        "total_fees_ticks": sum(r.fees_ticks for r in rows),
        "mean_fees_bps": _mean([_bps(r.fees_ticks, r) for r in rows]),
        # Terminated episodes: one shortfall definition (residual term included).
        "terminated": {
            "n": len(terminated),
            "total_shortfall_ticks": sum(r.shortfall_ticks for r in terminated),
            "mean_shortfall_ticks": _mean([r.shortfall_ticks for r in terminated]),
            "mean_shortfall_bps": _mean(bps),
            "median_shortfall_bps": nearest_rank_percentile(bps, 50) if bps else None,
            "p95_shortfall_bps": nearest_rank_percentile(bps, 95) if bps else None,
            "p99_shortfall_bps": nearest_rank_percentile(bps, 99) if bps else None,
            "max_shortfall_bps": max(bps) if bps else None,
            "mean_terminal_component_bps": _mean(
                [_bps(r.terminal_component_ticks, r) for r in terminated]
            ),
            "mean_reward_bps": _mean([_bps(r.reward_sum_ticks, r) for r in terminated]),
            "mean_shortfall_bps_uncertainty": day_cluster_bootstrap(
                _by_day(terminated, lambda r: r.shortfall_bps), min_days=min_days
            ),
        },
        "completed": {
            "n": len(completed),
            "mean_shortfall_bps": _mean([r.shortfall_bps for r in completed]),
            "median_shortfall_bps": (
                nearest_rank_percentile([r.shortfall_bps for r in completed], 50)
                if completed else None
            ),
        },
        # Truncated episodes: fills-only shortfall, a different quantity.
        "truncated": {
            "n": len(truncated),
            "mean_fill_shortfall_bps": _mean([r.shortfall_bps for r in truncated]),
            "mean_filled_fraction": _mean(
                [r.filled_qty_ticks / r.target_qty_ticks for r in truncated]
            ),
        },
    }
    return out


def _by_day(rows: Iterable[EpisodeRow], value: Any) -> dict[str, list[float]]:
    """Group ``value(row)`` by calendar day."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.date].append(float(value(row)))
    return dict(grouped)


def paired_differences(
    rows: Sequence[EpisodeRow], reference: str, min_days: int = MIN_DAYS_FOR_UNCERTAINTY
) -> list[dict[str, Any]]:
    """Return per-(policy, role) differences against ``reference`` on identical episode ids.

    A pair enters only when both episodes terminated. ``mean_diff_bps > 0``
    means the policy cost more than the reference.

    Raises:
        ValueError: ``reference`` is not among the policies in ``rows``.
    """
    policies = sorted({r.policy for r in rows})
    if reference not in policies:
        raise ValueError(f"reference policy {reference!r} is not in the run; found {policies}")
    by_key: dict[tuple[str, str, str], EpisodeRow] = {
        (r.policy, r.role, r.episode_id): r for r in rows
    }
    roles = sorted({r.role for r in rows})
    out: list[dict[str, Any]] = []
    for policy in policies:
        if policy == reference:
            continue
        for role in roles:
            ids = sorted(k[2] for k in by_key if k[0] == reference and k[1] == role)
            diffs_bps: list[float] = []
            diffs_ticks: list[int] = []
            pairs: list[tuple[EpisodeRow, EpisodeRow]] = []
            n_missing = n_excluded = 0
            for episode_id in ids:
                other = by_key.get((policy, role, episode_id))
                base = by_key[(reference, role, episode_id)]
                if other is None:
                    n_missing += 1
                    continue
                if not (other.terminated and base.terminated):
                    n_excluded += 1
                    continue
                pairs.append((other, base))
                diffs_ticks.append(other.shortfall_ticks - base.shortfall_ticks)
                diffs_bps.append(other.shortfall_bps - base.shortfall_bps)
            by_day: dict[str, list[float]] = defaultdict(list)
            for (other, _), diff in zip(pairs, diffs_bps, strict=True):
                by_day[other.date].append(diff)
            out.append(
                {
                    "policy": policy,
                    "reference": reference,
                    "role": role,
                    "n_pairs": len(pairs),
                    "n_excluded_truncated": n_excluded,
                    "n_missing": n_missing,
                    "mean_diff_bps": _mean(diffs_bps),
                    "median_diff_bps": (
                        nearest_rank_percentile(diffs_bps, 50) if diffs_bps else None
                    ),
                    "total_diff_ticks": sum(diffs_ticks),
                    "fraction_policy_cheaper": (
                        sum(1 for d in diffs_ticks if d < 0) / len(diffs_ticks)
                        if diffs_ticks else None
                    ),
                    "fraction_equal": (
                        sum(1 for d in diffs_ticks if d == 0) / len(diffs_ticks)
                        if diffs_ticks else None
                    ),
                    "n_both_completed": sum(
                        1 for o, b in pairs if o.outcome == "completed" and b.outcome == "completed"
                    ),
                    "mean_diff_bps_uncertainty": day_cluster_bootstrap(by_day, min_days=min_days),
                }
            )
    return out


# --- the report -------------------------------------------------------------------------


def build_report(
    run_dir: str | Path,
    reference: str | None = None,
    include_test: bool = False,
    min_days: int = MIN_DAYS_FOR_UNCERTAINTY,
) -> dict[str, Any]:
    """Assemble the report dictionary (``REPORT_SCHEMA``) for a run directory.

    Args:
        run_dir: Directory written by ``experiment.run_experiment``.
        reference: Policy the paired differences are taken against; the
            first configured policy when ``None``.
        include_test: Include held-out records.
        min_days: Distinct days required before an interval is reported.

    Raises:
        ValueError: Unknown reference policy, or a manifest or record with an
            unexpected schema.
    """
    manifest = load_manifest(run_dir)
    records = load_records(run_dir, include_test)
    rows = episode_rows(records)
    policies = list(manifest["config"]["policies"])
    reference = policies[0] if reference is None else reference
    roles = [r for r in manifest["roles_evaluated"] if include_test or r != "test"]

    per_policy = [
        {"policy": p, "role": role, "held_out": role == "test",
         **group_statistics([r for r in rows if r.policy == p and r.role == role], min_days)}
        for role in roles for p in policies
    ]
    per_day = []
    for role in roles:
        for p in policies:
            for day in sorted({r.date for r in rows if r.role == role}):
                day_rows = [r for r in rows if r.policy == p and r.role == role and r.date == day]
                per_day.append(
                    {"policy": p, "role": role, "date": day,
                     **group_statistics(day_rows, min_days=10**9)}
                )
    template = manifest["config"]["template"]
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "experiment": {
            "name": manifest["name"],
            "label": manifest["label"],
            "description": manifest["description"],
            "created_utc": manifest["created_utc"],
            "code_revision": manifest["code_revision"],
            "simulator_version": manifest["simulator"]["version"],
            "manifest_schema": EXPERIMENT_MANIFEST_SCHEMA,
            "data": [
                {k: d.get(k) for k in ("source", "date", "sha256", "n_bytes", "seed", "market")}
                for d in manifest["data"]
            ],
            "seeds": manifest["config"]["seeds"],
            "episodes_per_day": manifest["config"]["episodes_per_day"],
            "residual_mu": template["residual_mu"],
            "risk_lambda": template["risk_lambda"],
            "fee_bps": template["fee_bps"],
            "participation_cap_bps": template["participation_cap_bps"],
            "horizon_buckets": template["horizon_buckets"],
        },
        "include_test": include_test,
        "roles": roles,
        "reference_policy": reference,
        "min_days_for_uncertainty": min_days,
        "definitions": {
            "shortfall_terminated": (
                "implementation shortfall in cash ticks: side * (exec notional - arrival "
                "price * filled) + fees + residual term; the residual term is side * "
                "(terminal mark - arrival) * residual + residual_mu * residual, charged only "
                "when the horizon ended with residual inventory"
            ),
            "shortfall_truncated": (
                "fills-only shortfall (no residual term, no terminal mark); a different "
                "quantity, reported separately and excluded from pairing"
            ),
            "shortfall_bps": "shortfall / (arrival price * target) * 10_000",
            "paired_difference": "policy shortfall minus reference shortfall on the same episode",
            "uncertainty": (
                "95 % day-cluster bootstrap percentile interval of the pooled mean; absent "
                f"below {min_days} distinct days"
            ),
        },
        "per_policy": per_policy,
        "per_day": per_day,
        "paired": paired_differences(rows, reference, min_days),
        "n_episode_rows": len(rows),
    }


# --- export ------------------------------------------------------------------------------


def _write_csv_atomic(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    """Write ``rows`` as CSV with the given columns through a temporary file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    os.replace(tmp, path)
    return path


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into ``a.b`` keys for CSV columns."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def _fmt(value: Any) -> str:
    """Format a report value for the RST tables."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def rst_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render an RST simple table; the first column must not be empty in any row."""
    cells = [[_fmt(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    rule = "  ".join("=" * w for w in widths)
    lines = [rule, "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)), rule]
    for row in cells:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    lines.append(rule)
    return "\n".join(line.rstrip() for line in lines)


def _uncertainty_text(u: Mapping[str, Any]) -> str:
    """Render an uncertainty entry as a table cell."""
    if u.get("status") == "ok":
        return f"[{u['ci95_low']:.3f}, {u['ci95_high']:.3f}] over {u['n_days']} days"
    return f"insufficient: {u.get('n_days')} day(s) < {u.get('min_days')}"


def render_rst(report: Mapping[str, Any]) -> str:
    """Render the report as reStructuredText (the shared result format of the repository)."""
    exp = report["experiment"]
    title = f"Baseline report: {exp['name']}"
    lines = [title, "=" * len(title), ""]
    if exp["label"] == "smoke":
        lines += [
            "**Smoke run.** This configuration demonstrates that the pipeline runs end to "
            "end. Its figures support no claim about any policy's economic merit.", "",
        ]
    rev = exp["code_revision"]
    lines += [
        f"Created {exp['created_utc']}; commit {rev.get('commit')} (dirty: {rev.get('dirty')}); "
        f"simulator {exp['simulator_version']}; fee {exp['fee_bps']} bps, participation cap "
        f"{exp['participation_cap_bps']} bps, horizon {exp['horizon_buckets']} buckets, "
        f"residual_mu {exp['residual_mu']}, risk_lambda {exp['risk_lambda']}; seeds "
        f"{exp['seeds']}, {exp['episodes_per_day']} episodes per day per seed.",
        "",
        "Held-out test records included: " + ("yes" if report["include_test"] else "no") + ".",
        "",
        "Data", "----", "",
        rst_table(
            ("date", "source", "market", "sha256 / seed", "bytes"),
            [
                (d["date"], d["source"], d.get("market"),
                 (d.get("sha256") or "")[:16] or d.get("seed"), d.get("n_bytes"))
                for d in exp["data"]
            ],
        ),
        "",
        "Definitions", "-----------", "",
    ]
    for key, text in report["definitions"].items():
        lines.append(f"* ``{key}``: {text}")
    if exp["residual_mu"] == 0:
        lines += [
            "",
            "``residual_mu`` is 0: an episode that ends with residual inventory pays only the "
            "Perold opportunity cost, which averages near zero, and no completion penalty. "
            "Mean shortfalls of policies with different completion rates are therefore not "
            "comparable on their own; read them next to the completion rate, the residual "
            "fraction and the completed-only mean. This is the documented inaction incentive "
            "of the residual rule, not a defect of the report.",
        ]
    lines += ["", "Per policy", "----------", ""]
    lines.append(
        rst_table(
            ("policy", "role", "n", "completed", "incomplete", "truncated", "mean IS bps",
             "median", "p95", "p99", "max", "completed-only mean", "truncated fills-only mean",
             "mean steps", "mean capped", "residual frac", "95 % CI of mean"),
            [
                (g["policy"], g["role"], g["n_episodes"], g["n_completed"], g["n_incomplete"],
                 g["n_truncated"], g["terminated"]["mean_shortfall_bps"],
                 g["terminated"]["median_shortfall_bps"], g["terminated"]["p95_shortfall_bps"],
                 g["terminated"]["p99_shortfall_bps"], g["terminated"]["max_shortfall_bps"],
                 g["completed"]["mean_shortfall_bps"], g["truncated"]["mean_fill_shortfall_bps"],
                 g["mean_steps"], g["mean_capped_steps"], g["mean_residual_fraction"],
                 _uncertainty_text(g["terminated"]["mean_shortfall_bps_uncertainty"]))
                for g in report["per_policy"]
            ],
        )
    )
    lines += ["", "Per day", "-------", ""]
    lines.append(
        rst_table(
            ("policy", "role", "date", "n", "completed", "truncated", "mean IS bps", "median",
             "total IS ticks", "total fees ticks"),
            [
                (g["policy"], g["role"], g["date"], g["n_episodes"], g["n_completed"],
                 g["n_truncated"], g["terminated"]["mean_shortfall_bps"],
                 g["terminated"]["median_shortfall_bps"],
                 g["terminated"]["total_shortfall_ticks"], g["total_fees_ticks"])
                for g in report["per_day"]
            ],
        )
    )
    lines += ["", f"Paired differences against ``{report['reference_policy']}``",
              "-" * (len(report["reference_policy"]) + 32), ""]
    if report["paired"]:
        lines.append(
            rst_table(
                ("policy", "role", "pairs", "excluded (truncated)", "mean diff bps",
                 "median diff", "total diff ticks", "policy cheaper", "equal", "95 % CI of mean"),
                [
                    (p["policy"], p["role"], p["n_pairs"], p["n_excluded_truncated"],
                     p["mean_diff_bps"], p["median_diff_bps"], p["total_diff_ticks"],
                     p["fraction_policy_cheaper"], p["fraction_equal"],
                     _uncertainty_text(p["mean_diff_bps_uncertainty"]))
                    for p in report["paired"]
                ],
            )
        )
    else:
        lines.append("Only one policy was run; nothing to pair.")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any], out_dir: str | Path, rows: Sequence[EpisodeRow] = ()
) -> dict[str, Path]:
    """Write ``report.json``, the CSV tables and ``report.rst`` into ``out_dir``.

    Returns:
        The written paths by name.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": write_json_atomic(out / "report.json", dict(report))}
    policy_rows = [_flatten(g) for g in report["per_policy"]]
    day_rows = [_flatten(g) for g in report["per_day"]]
    paired_rows = [_flatten(p) for p in report["paired"]]
    if policy_rows:
        paths["policies"] = _write_csv_atomic(
            out / "policies.csv", list(policy_rows[0]), policy_rows
        )
    if day_rows:
        paths["days"] = _write_csv_atomic(out / "days.csv", list(day_rows[0]), day_rows)
    if paired_rows:
        paths["paired"] = _write_csv_atomic(out / "paired.csv", list(paired_rows[0]), paired_rows)
    if rows:
        paths["episodes"] = _write_csv_atomic(
            out / "episodes.csv", EPISODE_COLUMNS, [asdict(r) for r in rows]
        )
    rst = out / "report.rst"
    tmp = rst.with_name(rst.name + ".tmp")
    tmp.write_text(render_rst(report))
    os.replace(tmp, rst)
    paths["rst"] = rst
    return paths


def main(argv: list[str] | None = None) -> int:
    """``<run_dir> [--reference P] [--include-test] [--out DIR] [--min-days N]``.

    Writes the report under ``<run_dir>/report`` by default and prints the
    RST page.

    Returns:
        0 on success, 1 on a bad run directory or reference.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.report")
    parser.add_argument("run_dir")
    parser.add_argument("--reference", default=None, help="policy to pair against")
    parser.add_argument("--include-test", action="store_true", help="include held-out records")
    parser.add_argument("--out", default=None, help="output directory (default <run_dir>/report)")
    parser.add_argument("--min-days", type=int, default=MIN_DAYS_FOR_UNCERTAINTY)
    parser.add_argument("--quiet", action="store_true", help="do not print the RST page")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.run_dir, args.reference, args.include_test, args.min_days)
        rows = episode_rows(load_records(args.run_dir, args.include_test))
        paths = write_report(report, args.out or Path(args.run_dir) / REPORT_SUBDIR, rows)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(render_rst(report))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
