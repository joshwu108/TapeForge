"""Configuration-driven baseline experiments: one TOML file in, one run directory out.

``run_experiment`` turns an ``ExperimentConfig`` into a run directory holding
one run record per ``(policy, role, seed)`` plus ``manifest.json`` with
everything needed to reproduce and audit the run: the resolved configuration,
the sha256 and size of every archive read (or the generator parameters of
every synthetic day), the code revision and dirty flag, the simulator version
and its assumptions, the split record, and the identifier of every episode
each policy was given.

Reproducibility contract:
    Episode draws are a pure function of ``(seed, role, index)``
    (``sampling``), so every policy of a run replays exactly the same
    episodes, in the same order, from freshly cut slices; policies share no
    state and their execution order cannot change any outcome. Two runs of
    the same configuration on the same data produce byte-identical records
    except for timestamps, timings and the code-revision fields
    (``compare_experiments`` checks this).

Held-out period:
    The test role is evaluated only when the caller passes
    ``include_test=True`` and the configuration lists test dates; its
    records are written beside the others but tagged ``held_out`` in the
    manifest. No configuration carries default dates; a ``"smoke"`` label
    marks a demonstration whose results support no economic claim.

Episode identifiers:
    ``episode_identifier(role, seed, index)`` returns ``"<role>/s<seed>/i<index>"``.
    The identifier plus the manifest rebuilds the episode exactly
    (``replay``).

Not covered:
    Reporting (``report``) and step-level inspection (``replay``) read the
    directory this module writes; nothing here trains a policy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from tapeforge.execution.baselines import BASELINES
from tapeforge.execution.datasets import (
    DEFAULT_MAX_DOWNLOAD_ARCHIVES,
    DEFAULT_MAX_DOWNLOAD_BYTES,
    ArchiveRecord,
    ArchiveSpec,
    DatasetError,
    DownloadBudget,
    ensure_archives,
)
from tapeforge.execution.days import DayTape, day_from_archive, synthetic_day
from tapeforge.execution.runner import code_revision, run_policy
from tapeforge.execution.sampling import sample_episodes
from tapeforge.execution.simulator import SIMULATOR_ASSUMPTIONS, SIMULATOR_VERSION
from tapeforge.execution.spec import ACCEPTED_TARGET_FRACTIONS_BPS, EpisodeSpec
from tapeforge.execution.splits import ROLES, SplitDates, Splits, build_splits
from tapeforge.ingest import MARKET_BASE_URLS

__all__ = [
    "EXPERIMENT_CONFIG_SCHEMA",
    "EXPERIMENT_MANIFEST_SCHEMA",
    "LABELS",
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "SyntheticDayConfig",
    "acquire_days",
    "compare_experiments",
    "config_from_mapping",
    "create_run_directory",
    "episode_identifier",
    "load_config",
    "load_manifest",
    "parse_episode_identifier",
    "rebuild_splits",
    "run_experiment",
    "write_json_atomic",
]

EXPERIMENT_CONFIG_SCHEMA: Final[str] = "tapeforge.execution.experiment-config/1"
EXPERIMENT_MANIFEST_SCHEMA: Final[str] = "tapeforge.execution.experiment/1"

# A smoke run demonstrates that the pipeline works end to end; a research run
# is the configuration whose numbers may be interpreted. Both need explicit
# split dates; the label only changes how the output is described.
LABELS: Final[tuple[str, ...]] = ("smoke", "research")
SOURCES: Final[tuple[str, ...]] = ("archive", "synthetic")
RECORDS_SUBDIR: Final[str] = "records"
MANIFEST_NAME: Final[str] = "manifest.json"

# Record fields that legitimately differ between two runs of the same inputs.
VOLATILE_RECORD_KEYS: Final[frozenset[str]] = frozenset({"created_utc", "code_revision"})

_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^(train|validation|test)/s(\d+)/i(\d+)$")
_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent; the message names the key."""


# --- configuration --------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDayConfig:
    """One synthetic day: the generator seed behind a calendar label.

    Attributes:
        date: Calendar label, ``YYYY-MM-DD``; the day starts at its UTC
            midnight.
        seed: ``generate.generate_trades`` seed.
    """

    date: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    """Where the days come from.

    Attributes:
        source: ``"archive"`` (downloaded daily aggTrades) or
            ``"synthetic"`` (seeded generator days, for demonstrations and
            tests).
        market: Archive market, one of ``ingest.MARKET_BASE_URLS``; a run
            reads one market only. ``"synthetic"`` for synthetic days.
        symbol: Exchange symbol.
        data_dir: Local cache root (ignored by Git).
        allow_download: Whether missing archives may be fetched.
        max_download_archives: Archive budget of one run.
        max_download_bytes: Byte budget of one run, every attempt included.
        synthetic_n_buckets: Buckets per synthetic day.
        synthetic_profile: Generator profile of synthetic days.
        synthetic_days: Seed per synthetic day.
    """

    source: str
    market: str
    symbol: str
    data_dir: str = "data"
    allow_download: bool = False
    max_download_archives: int = DEFAULT_MAX_DOWNLOAD_ARCHIVES
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES
    synthetic_n_buckets: int = 0
    synthetic_profile: str = "uniform"
    synthetic_days: tuple[SyntheticDayConfig, ...] = ()

    def __post_init__(self) -> None:
        """Validate the source and market combination.

        Raises:
            ConfigError: Unknown source or market, an archive source with a
                synthetic market (or the reverse), or synthetic days without a
                positive bucket count.
        """
        if self.source not in SOURCES:
            raise ConfigError(f"data.source must be one of {SOURCES}, got {self.source!r}")
        if self.source == "archive":
            if self.market not in MARKET_BASE_URLS:
                raise ConfigError(
                    f"data.market must be one of {sorted(MARKET_BASE_URLS)} for an archive "
                    f"source, got {self.market!r}"
                )
            if self.synthetic_days:
                raise ConfigError("data.synthetic days are not allowed with an archive source")
        else:
            if self.market != "synthetic":
                raise ConfigError(
                    f"data.market must be 'synthetic' for a synthetic source, got {self.market!r}"
                )
            if self.synthetic_n_buckets < 1:
                raise ConfigError("data.synthetic.n_buckets must be positive")
            if not self.synthetic_days:
                raise ConfigError("data.synthetic.days must list at least one day")
        if self.max_download_archives < 0 or self.max_download_bytes < 0:
            raise ConfigError("download limits must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """The resolved configuration of one experiment.

    Attributes:
        name: Run name; the run directory is ``<output_dir>/<name>-<utc>``.
        label: ``"smoke"`` or ``"research"``.
        description: Free text copied into the manifest.
        data: ``DataConfig``.
        split: Explicit role assignment of every day.
        seeds: Sampling seeds; each seed yields an independent episode set.
        episodes_per_day: Draws per role per seed are
            ``episodes_per_day * number of days in the role``.
        rho_choices_bps: Target fractions the draws choose from.
        evaluate_roles: Roles replayed by default (never ``"test"``).
        policies: Baseline names from ``baselines.BASELINES``, in order.
        template: Spec template; side and target come from each draw.
        output_dir: Root of run directories (ignored by Git).
    """

    name: str
    label: str
    description: str
    data: DataConfig
    split: SplitDates
    seeds: tuple[int, ...]
    episodes_per_day: int
    rho_choices_bps: tuple[int, ...]
    evaluate_roles: tuple[str, ...]
    policies: tuple[str, ...]
    template: EpisodeSpec
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        """Validate every field that the lower layers do not.

        Raises:
            ConfigError: A bad name or label, no seeds, a non-positive
                episode count, unknown or duplicate policies, an unknown or
                test role in ``evaluate_roles``, a role without dates among
                the evaluated roles, or an empty ``rho_choices_bps``.
        """
        if not _NAME.match(self.name):
            raise ConfigError(f"name must match {_NAME.pattern}, got {self.name!r}")
        if self.label not in LABELS:
            raise ConfigError(f"label must be one of {LABELS}, got {self.label!r}")
        if not self.seeds or any(s < 0 for s in self.seeds) or len(set(self.seeds)) < len(
            self.seeds
        ):
            raise ConfigError(
                "episodes.seeds must be a non-empty list of distinct non-negative ints"
            )
        if self.episodes_per_day < 1:
            raise ConfigError("episodes.episodes_per_day must be positive")
        if not self.rho_choices_bps:
            raise ConfigError("episodes.rho_choices_bps must not be empty")
        if not self.policies or len(set(self.policies)) < len(self.policies):
            raise ConfigError("policies.names must be a non-empty list without duplicates")
        unknown = [p for p in self.policies if p not in BASELINES]
        if unknown:
            raise ConfigError(f"unknown policies {unknown}; available: {sorted(BASELINES)}")
        for role in self.evaluate_roles:
            if role not in ROLES:
                raise ConfigError(f"unknown role {role!r} in episodes.evaluate_roles")
            if role == "test":
                raise ConfigError(
                    "the test role is never listed in episodes.evaluate_roles; pass "
                    "--include-test (include_test=True) to evaluate the held-out period"
                )
            if not getattr(self.split, role):
                raise ConfigError(f"evaluate_roles names {role!r} but split.{role} is empty")
        if not self.evaluate_roles:
            raise ConfigError("episodes.evaluate_roles must name at least one role")

    def n_episodes(self, role: str) -> int:
        """Return the draws per seed for ``role``: ``episodes_per_day`` times its day count."""
        return self.episodes_per_day * len(getattr(self.split, role))

    def to_json(self) -> dict[str, Any]:
        """Return the resolved configuration as plain values."""
        payload = asdict(self)
        payload["template"] = {
            f.name: getattr(self.template, f.name)
            for f in fields(self.template)
            if f.name not in ("side", "target_qty_ticks")
        }
        payload["split"] = {role: list(getattr(self.split, role)) for role in ROLES}
        payload["data"]["synthetic_days"] = [asdict(d) for d in self.data.synthetic_days]
        payload["schema"] = EXPERIMENT_CONFIG_SCHEMA
        return payload


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    """Return ``mapping[key]`` or raise ``ConfigError`` naming ``section.key``."""
    if key not in mapping:
        raise ConfigError(f"missing required key {section}.{key}")
    return mapping[key]


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str], section: str) -> None:
    """Raise ``ConfigError`` when ``mapping`` holds a key outside ``allowed``."""
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {unknown}; allowed: {list(allowed)}")


def _int_list(values: Any, key: str) -> tuple[int, ...]:
    """Coerce a list of integers, rejecting bools and floats (``ConfigError``)."""
    if not isinstance(values, list):
        raise ConfigError(f"{key} must be a list, got {type(values).__name__}")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{key} must hold integers, got {v!r}")
        out.append(int(v))
    return tuple(out)


def _str_list(values: Any, key: str) -> tuple[str, ...]:
    """Coerce a list of strings (``ConfigError``)."""
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(values)


def config_from_mapping(mapping: Mapping[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from a parsed TOML mapping, strictly.

    Every section and key is checked; an unknown key is an error rather than
    a silently ignored typo. Template fields are passed to ``EpisodeSpec``,
    which validates their domains.

    Raises:
        ConfigError: A missing, unknown or invalid key; the message names it.
    """
    _reject_unknown(
        mapping,
        ("schema", "name", "label", "description", "data", "split", "episodes", "policies",
         "template", "output_dir"),
        "top level",
    )
    schema = mapping.get("schema", EXPERIMENT_CONFIG_SCHEMA)
    if schema != EXPERIMENT_CONFIG_SCHEMA:
        raise ConfigError(f"schema {schema!r} is not {EXPERIMENT_CONFIG_SCHEMA!r}")

    data_map = _require(mapping, "data", "top level")
    _reject_unknown(
        data_map,
        ("source", "market", "symbol", "data_dir", "allow_download", "max_download_archives",
         "max_download_bytes", "synthetic"),
        "data",
    )
    synthetic = data_map.get("synthetic", {})
    _reject_unknown(synthetic, ("n_buckets", "profile", "days"), "data.synthetic")
    synthetic_days = tuple(
        SyntheticDayConfig(str(_require(d, "date", "data.synthetic.days")),
                           int(_require(d, "seed", "data.synthetic.days")))
        for d in synthetic.get("days", [])
    )
    data = DataConfig(
        source=str(_require(data_map, "source", "data")),
        market=str(_require(data_map, "market", "data")),
        symbol=str(data_map.get("symbol", "synthetic")),
        data_dir=str(data_map.get("data_dir", "data")),
        allow_download=bool(data_map.get("allow_download", False)),
        max_download_archives=int(data_map.get("max_download_archives",
                                               DEFAULT_MAX_DOWNLOAD_ARCHIVES)),
        max_download_bytes=int(data_map.get("max_download_bytes", DEFAULT_MAX_DOWNLOAD_BYTES)),
        synthetic_n_buckets=int(synthetic.get("n_buckets", 0)),
        synthetic_profile=str(synthetic.get("profile", "uniform")),
        synthetic_days=synthetic_days,
    )

    split_map = _require(mapping, "split", "top level")
    _reject_unknown(split_map, ROLES, "split")
    try:
        split = SplitDates(
            train=_str_list(_require(split_map, "train", "split"), "split.train"),
            validation=_str_list(_require(split_map, "validation", "split"), "split.validation"),
            test=_str_list(split_map.get("test", []), "split.test"),
        )
    except ValueError as exc:
        raise ConfigError(f"split: {exc}") from exc

    episodes = _require(mapping, "episodes", "top level")
    _reject_unknown(
        episodes, ("seeds", "episodes_per_day", "rho_choices_bps", "evaluate_roles"), "episodes"
    )
    policies = _require(mapping, "policies", "top level")
    _reject_unknown(policies, ("names",), "policies")

    template_map = dict(mapping.get("template", {}))
    spec_fields = [
        f.name for f in fields(EpisodeSpec) if f.name not in ("side", "target_qty_ticks")
    ]
    _reject_unknown(template_map, spec_fields, "template")
    try:
        template = EpisodeSpec(side=1, target_qty_ticks=1, **template_map)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"template: {exc}") from exc

    return ExperimentConfig(
        name=str(_require(mapping, "name", "top level")),
        label=str(_require(mapping, "label", "top level")),
        description=str(mapping.get("description", "")),
        data=data,
        split=split,
        seeds=_int_list(_require(episodes, "seeds", "episodes"), "episodes.seeds"),
        episodes_per_day=int(_require(episodes, "episodes_per_day", "episodes")),
        rho_choices_bps=_int_list(
            episodes.get("rho_choices_bps", list(ACCEPTED_TARGET_FRACTIONS_BPS)),
            "episodes.rho_choices_bps",
        ),
        evaluate_roles=_str_list(
            episodes.get("evaluate_roles", ["train", "validation"]), "episodes.evaluate_roles"
        ),
        policies=_str_list(_require(policies, "names", "policies"), "policies.names"),
        template=template,
        output_dir=str(mapping.get("output_dir", "runs")),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a TOML configuration file.

    Raises:
        ConfigError: The file is not valid TOML or fails ``config_from_mapping``.
        FileNotFoundError: No such file.
    """
    try:
        mapping = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return config_from_mapping(mapping)


# --- episodes and data ------------------------------------------------------------------


def episode_identifier(role: str, seed: int, index: int) -> str:
    """Return the identifier ``"<role>/s<seed>/i<index>"`` of one draw."""
    return f"{role}/s{seed}/i{index}"


def parse_episode_identifier(identifier: str) -> tuple[str, int, int]:
    """Return ``(role, seed, index)`` of an identifier.

    Raises:
        ValueError: The string is not of the form ``<role>/s<seed>/i<index>``.
    """
    match = _IDENTIFIER.match(identifier)
    if match is None:
        raise ValueError(
            f"episode identifier {identifier!r} is not of the form <role>/s<seed>/i<index>"
        )
    return match.group(1), int(match.group(2)), int(match.group(3))


def _all_dates(split: SplitDates) -> tuple[str, ...]:
    """Return every listed date in chronological (sorted) order."""
    return tuple(sorted(d for role in ROLES for d in getattr(split, role)))


def acquire_days(
    config: ExperimentConfig, budget: DownloadBudget | None = None
) -> tuple[tuple[DayTape, ...], tuple[dict[str, Any], ...]]:
    """Load every day the split lists and return the tapes with their provenance.

    Archive days go through ``datasets.ensure_archives`` (one market, shared
    budget, verified checksums) and ``days.day_from_archive``; synthetic
    days through ``days.synthetic_day``. A split date without a synthetic
    seed is an error.

    Returns:
        The tapes in chronological order and one provenance dict per day:
        an ``ArchiveRecord`` plus the ingest provenance for archives, or the
        generator parameters for synthetic days.

    Raises:
        ConfigError: A synthetic date without a seed.
        DatasetError: An archive could not be acquired or verified.
        ValueError: The archive does not parse as the day it claims.
    """
    dates = _all_dates(config.split)
    data = config.data
    tapes: list[DayTape] = []
    provenance: list[dict[str, Any]] = []
    if data.source == "synthetic":
        seeds = {d.date: d.seed for d in data.synthetic_days}
        missing = [d for d in dates if d not in seeds]
        if missing:
            raise ConfigError(f"split dates {missing} have no entry in data.synthetic.days")
        for day in dates:
            tapes.append(
                synthetic_day(
                    seed=seeds[day], day=day, n_buckets=data.synthetic_n_buckets,
                    profile=data.synthetic_profile,  # type: ignore[arg-type]
                )
            )
            provenance.append(
                {
                    "source": "synthetic",
                    "date": day,
                    "seed": seeds[day],
                    "n_buckets": data.synthetic_n_buckets,
                    "profile": data.synthetic_profile,
                    "generator": "tapeforge.generate.generate_trades",
                }
            )
        return tuple(tapes), tuple(provenance)

    specs = [ArchiveSpec(data.market, data.symbol, d) for d in dates]
    budget = (
        DownloadBudget(data.max_download_archives, data.max_download_bytes)
        if budget is None else budget
    )
    records: tuple[ArchiveRecord, ...] = ensure_archives(
        specs, data.data_dir, budget, allow_download=data.allow_download
    )
    for record in records:
        tape, ingest_provenance = day_from_archive(record.path, record.date)
        if ingest_provenance.sha256 != record.sha256:  # pragma: no cover - same bytes
            raise DatasetError(f"{record.path} changed between verification and parsing")
        tapes.append(tape)
        provenance.append(
            {
                "source": "archive",
                **record.to_json(),
                "ingest": asdict(ingest_provenance),
            }
        )
    return tuple(tapes), tuple(provenance)


def rebuild_splits(config: ExperimentConfig, allow_download: bool = False) -> Splits:
    """Load the days a configuration names and build its splits without downloading.

    Used by the replay and report tools to rebuild the exact episodes a
    manifest describes; ``allow_download`` is forced off unless requested.
    """
    data = config.data
    if data.source == "archive" and not allow_download:
        config = replace(config, data=replace(data, allow_download=False))
    tapes, _ = acquire_days(config)
    return build_splits(tapes, config.split, config.template)


# --- run directory and atomic writes -----------------------------------------------------


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Write ``payload`` as indented, key-sorted JSON through a temporary file and a rename.

    A reader never sees a half-written file: the content lands in
    ``<name>.tmp`` in the same directory and ``os.replace`` makes it visible
    in one step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def create_run_directory(root: str | Path, name: str) -> Path:
    """Create ``<root>/<name>-<utc>[-<k>]`` and return it.

    The directory is created with an exclusive ``mkdir``, so two runs started
    in the same second get different directories rather than sharing one.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{name}-{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base.with_name(f"{base.name}-{suffix}")


def _max_rss_bytes() -> int:
    """Return the process's peak resident set size in bytes (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _environment() -> dict[str, Any]:
    """Return the interpreter and platform identity recorded in the manifest."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


# --- the experiment -----------------------------------------------------------------------


@dataclass(frozen=True)
class _RunEntry:
    """One ``(policy, role, seed)`` replay as listed in the manifest."""

    policy: str
    role: str
    seed: int
    held_out: bool
    n_episodes: int
    record_file: str
    seconds: float
    summary: dict[str, Any] = field(default_factory=dict)


def run_experiment(
    config: ExperimentConfig,
    *,
    include_test: bool = False,
    output_dir: str | Path | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Run every policy over every evaluated role and seed and write the run directory.

    Args:
        config: The resolved configuration.
        include_test: Also evaluate the test role; requires test dates. The
            records are tagged ``held_out``.
        output_dir: Overrides ``config.output_dir``.
        command: The command line to record, if any.

    Returns:
        The run directory, holding ``manifest.json`` and ``records/``.

    Raises:
        ConfigError: ``include_test`` without test dates.
        DatasetError: Data could not be acquired or verified.
        ValueError: The days do not form valid splits, or a role has no
            admissible start under the template.
    """
    if include_test and not config.split.test:
        raise ConfigError("include_test was requested but split.test lists no dates")
    started = time.perf_counter()
    created = datetime.now(UTC).isoformat(timespec="seconds")
    roles = tuple(config.evaluate_roles) + (("test",) if include_test else ())

    load_started = time.perf_counter()
    budget = DownloadBudget(config.data.max_download_archives, config.data.max_download_bytes)
    tapes, provenance = acquire_days(config, budget)
    splits = build_splits(tapes, config.split, config.template)
    load_seconds = time.perf_counter() - load_started

    # Every policy sees exactly these draws: sampled once per (role, seed)
    # here for the manifest, and re-derived identically inside run_policy.
    episodes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in roles:
        n = config.n_episodes(role)
        if splits.record.n_starts(role) == 0:
            raise ValueError(
                f"role {role!r} has no admissible episode start: its days are too short for "
                f"horizon {config.template.horizon_buckets}, lookback "
                f"{config.template.arrival_lookback} and embargo {splits.record.embargo_buckets}"
            )
        episodes[role] = {}
        for seed in config.seeds:
            draws = sample_episodes(splits.record, role, n, seed, config.rho_choices_bps)
            episodes[role][str(seed)] = [
                {"id": episode_identifier(role, seed, d.index), **d.to_json()} for d in draws
            ]

    run_dir = create_run_directory(output_dir or config.output_dir, config.name)
    records_dir = run_dir / RECORDS_SUBDIR
    runs: list[_RunEntry] = []
    for role in roles:
        for seed in config.seeds:
            for policy_name in config.policies:
                policy_started = time.perf_counter()
                record, _ = run_policy(
                    splits, role, BASELINES[policy_name], policy_name, config.template,
                    config.n_episodes(role), seed, config.rho_choices_bps,
                )
                seconds = time.perf_counter() - policy_started
                record["episode_ids"] = [
                    episode_identifier(role, seed, e["draw"]["index"]) for e in record["episodes"]
                ]
                record["held_out"] = role == "test"
                file_name = f"{policy_name}-{role}-seed{seed}.json"
                write_json_atomic(records_dir / file_name, record)
                runs.append(
                    _RunEntry(
                        policy=policy_name, role=role, seed=seed, held_out=role == "test",
                        n_episodes=config.n_episodes(role),
                        record_file=f"{RECORDS_SUBDIR}/{file_name}", seconds=seconds,
                        summary=record["summary"],
                    )
                )

    manifest = {
        "schema": EXPERIMENT_MANIFEST_SCHEMA,
        "name": config.name,
        "label": config.label,
        "description": config.description,
        "created_utc": created,
        "command": list(command) if command is not None else None,
        "code_revision": code_revision(),
        "environment": _environment(),
        "simulator": {"version": SIMULATOR_VERSION, "assumptions": SIMULATOR_ASSUMPTIONS},
        "config": config.to_json(),
        "include_test": include_test,
        "roles_evaluated": list(roles),
        "data": list(provenance),
        "download_budget": budget.to_json(),
        "split": splits.record.to_json(),
        "episodes": episodes,
        "runs": [asdict(r) for r in runs],
        "resources": {
            "data_load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started,
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    write_json_atomic(run_dir / MANIFEST_NAME, manifest)
    return run_dir


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Read and validate ``manifest.json`` of a run directory.

    Raises:
        FileNotFoundError: No manifest.
        ValueError: The manifest's schema is not ``EXPERIMENT_MANIFEST_SCHEMA``.
    """
    path = Path(run_dir) / MANIFEST_NAME
    manifest: dict[str, Any] = json.loads(path.read_text())
    if manifest.get("schema") != EXPERIMENT_MANIFEST_SCHEMA:
        raise ValueError(
            f"{path} has schema {manifest.get('schema')!r}; this code reads "
            f"{EXPERIMENT_MANIFEST_SCHEMA!r}"
        )
    return manifest


def _stable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a record without its volatile fields."""
    return {k: v for k, v in record.items() if k not in VOLATILE_RECORD_KEYS}


def compare_experiments(first: str | Path, second: str | Path) -> list[str]:
    """Return the differences between two run directories; an empty list means equal outcomes.

    Compared: configuration, data checksums, split record, episode
    identifiers and every run record minus its volatile fields. Timings,
    timestamps, resource usage, commands and code revisions are ignored.

    Raises:
        FileNotFoundError: A manifest or record is missing.
        ValueError: A manifest has an unexpected schema.
    """
    a, b = load_manifest(first), load_manifest(second)
    differences: list[str] = []
    for key in ("config", "split", "episodes", "roles_evaluated", "simulator"):
        if a.get(key) != b.get(key):
            differences.append(f"manifest key {key!r} differs")
    digests_a = [d.get("sha256") or d.get("seed") for d in a.get("data", [])]
    digests_b = [d.get("sha256") or d.get("seed") for d in b.get("data", [])]
    if digests_a != digests_b:
        differences.append(f"data provenance differs: {digests_a} vs {digests_b}")
    files_a = {r["record_file"] for r in a.get("runs", [])}
    files_b = {r["record_file"] for r in b.get("runs", [])}
    if files_a != files_b:
        differences.append(f"run records differ: {sorted(files_a ^ files_b)}")
    for name in sorted(files_a & files_b):
        ra = _stable_record(json.loads((Path(first) / name).read_text()))
        rb = _stable_record(json.loads((Path(second) / name).read_text()))
        if ra != rb:
            differences.append(f"{name}: outcomes differ")
    return differences


# --- command line ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``run <config.toml> [--include-test] [--output-dir DIR]`` or ``compare A B``.

    Returns:
        0 on success; 1 on a configuration, data or comparison failure, with
        the reason on stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="replay every configured policy and write a run directory")
    run.add_argument("config", help="TOML experiment configuration")
    run.add_argument(
        "--include-test", action="store_true",
        help="also evaluate the held-out test dates (never done by default)",
    )
    run.add_argument("--output-dir", default=None, help="overrides output_dir of the config")
    compare = sub.add_parser("compare", help="compare the outcomes of two run directories")
    compare.add_argument("first")
    compare.add_argument("second")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            config = load_config(args.config)
            run_dir = run_experiment(
                config, include_test=args.include_test, output_dir=args.output_dir,
                command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            )
            manifest = load_manifest(run_dir)
            print(f"run directory: {run_dir}")
            for entry in manifest["runs"]:
                summary = entry["summary"]
                tag = " (held out)" if entry["held_out"] else ""
                print(
                    f"  {entry['policy']:16s} {entry['role']:10s} seed {entry['seed']:<4d}"
                    f" n={summary['n_episodes']:<4d} completion {summary['completion_rate']:.3f}"
                    f" mean IS {summary['mean_shortfall_bps']:.3f} bps"
                    f" p50 {summary['p50_shortfall_bps']:.3f}  {entry['seconds']:.2f}s{tag}"
                )
            resources = manifest["resources"]
            print(
                f"data load {resources['data_load_seconds']:.2f}s, total "
                f"{resources['total_seconds']:.2f}s, peak RSS "
                f"{resources['max_rss_bytes'] / 2**20:.0f} MiB"
            )
            return 0
        differences = compare_experiments(args.first, args.second)
        if differences:
            print("run directories differ:")
            for line in differences:
                print(f"  {line}")
            return 1
        print("identical outcomes (volatile fields ignored)")
        return 0
    except (ConfigError, DatasetError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
