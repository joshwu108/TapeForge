"""Observation scaling statistics computed from the training split only.

``observation_statistics`` replays a reference policy over sampled *train*
episodes, stacks the observation vectors the policy saw
(``OBSERVATION_FIELDS``) and keeps their per-field mean and population
standard deviation. It refuses draws from any other role, so validation and
test data cannot reach the statistics by construction; the test suite
additionally checks that perturbing validation and test days leaves them
unchanged.

``normalize`` applies ``(x - mean) / std`` with a unit divisor where the
standard deviation is zero (a constant field, such as ``side`` on a buy-only
set, is centred and left unscaled).

Invariants:
    The statistics are stored next to the split record in run records
    (``ObservationStatistics.to_json``) and can be frozen to a JSON file
    (``save_statistics`` / ``load_statistics``) tagged with
    ``STATISTICS_SCHEMA`` and the simulator version, so a trainer and a later
    evaluation load exactly the statistics the training run used. The
    baselines do not use them, since they read exact quantities rather than
    the scaled vector. The observations depend on the reference policy's
    actions through the remaining fraction, so the policy name is recorded
    with the result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tapeforge.execution.runner import Policy, run_episode
from tapeforge.execution.sampling import EpisodeDraw, materialize
from tapeforge.execution.simulator import OBSERVATION_FIELDS, SIMULATOR_VERSION
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import Splits

__all__ = [
    "STATISTICS_SCHEMA",
    "ObservationStatistics",
    "load_statistics",
    "normalize",
    "observation_statistics",
    "save_statistics",
    "statistics_from_json",
]

# Schema tag of a frozen statistics file. The file also names the simulator
# version whose observation layout the statistics describe.
STATISTICS_SCHEMA: Final[str] = "tapeforge.execution.observation-statistics/1"


@dataclass(frozen=True)
class ObservationStatistics:
    """Per-field mean and population standard deviation of training observations.

    Attributes:
        fields: Observation field names, ``OBSERVATION_FIELDS``.
        mean: Per-field mean, aligned with ``fields``.
        std: Per-field population standard deviation, aligned with
            ``fields``; zero for a constant field.
        n_observations: Number of observation vectors stacked.
        n_episodes: Number of training episodes replayed.
        policy: Name of the reference policy that produced the observations.
        seed: Seed of the training draws.
    """

    fields: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    n_observations: int
    n_episodes: int
    policy: str
    seed: int

    def __post_init__(self) -> None:
        """Check that the three per-field tuples agree in length.

        Raises:
            ValueError: ``fields``, ``mean`` and ``std`` differ in length.
        """
        if not len(self.fields) == len(self.mean) == len(self.std):
            raise ValueError("fields, mean and std must have the same length")

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict of built-in types tagged with the schema and simulator version.

        Stored verbatim in run records and written by ``save_statistics``.
        """
        return {
            "schema": STATISTICS_SCHEMA,
            "simulator_version": SIMULATOR_VERSION,
            **asdict(self),
        }


def statistics_from_json(payload: Mapping[str, Any]) -> ObservationStatistics:
    """Rebuild ``ObservationStatistics`` from ``to_json`` output, strictly.

    Raises:
        ValueError: Wrong or missing schema, a simulator version other than
            the running one (the observation layout may differ), fields that
            are not exactly ``OBSERVATION_FIELDS``, lengths that disagree, a
            non-finite mean or standard deviation, a negative standard
            deviation, or non-positive counts.
    """
    if payload.get("schema") != STATISTICS_SCHEMA:
        raise ValueError(
            f"statistics schema {payload.get('schema')!r} is not {STATISTICS_SCHEMA!r}"
        )
    if payload.get("simulator_version") != SIMULATOR_VERSION:
        raise ValueError(
            f"statistics were fitted under simulator {payload.get('simulator_version')!r}; "
            f"this code observes under {SIMULATOR_VERSION!r} and does not reuse them"
        )
    required = ("fields", "mean", "std", "n_observations", "n_episodes", "policy", "seed")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"statistics file is missing {missing}")
    fields = tuple(str(f) for f in payload["fields"])
    if fields != OBSERVATION_FIELDS:
        raise ValueError(
            f"statistics fields {fields} are not OBSERVATION_FIELDS {OBSERVATION_FIELDS}"
        )
    mean = tuple(float(v) for v in payload["mean"])
    std = tuple(float(v) for v in payload["std"])
    if len(mean) != len(fields) or len(std) != len(fields):
        raise ValueError("mean and std must have one value per field")
    if not all(math.isfinite(v) for v in mean + std):
        raise ValueError("mean and std must be finite")
    if any(v < 0 for v in std):
        raise ValueError("std must be non-negative")
    n_observations, n_episodes = int(payload["n_observations"]), int(payload["n_episodes"])
    if n_observations < 1 or n_episodes < 1:
        raise ValueError("n_observations and n_episodes must be positive")
    return ObservationStatistics(
        fields=fields, mean=mean, std=std, n_observations=n_observations,
        n_episodes=n_episodes, policy=str(payload["policy"]), seed=int(payload["seed"]),
    )


def save_statistics(statistics: ObservationStatistics, path: str | Path) -> Path:
    """Write ``statistics.to_json()`` as indented, key-sorted JSON through a temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(statistics.to_json(), indent=2, sort_keys=True) + "\n")
    tmp.replace(target)
    return target


def load_statistics(path: str | Path) -> ObservationStatistics:
    """Read a file written by ``save_statistics`` (``statistics_from_json`` validates it).

    Raises:
        FileNotFoundError: No such file.
        ValueError: The file is not JSON or fails validation.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return statistics_from_json(payload)


def observation_statistics(
    splits: Splits,
    draws: Sequence[EpisodeDraw],
    template: EpisodeSpec,
    policy: Policy,
    policy_name: str,
) -> ObservationStatistics:
    """Compute the mean and std of every observation a reference policy sees on train draws.

    Args:
        splits: Split tapes to cut episodes from.
        draws: Training draws; every ``role`` must be ``"train"``.
        template: Spec template passed to ``materialize``.
        policy: Reference policy replayed on each episode.
        policy_name: Name recorded on the result.

    Raises:
        ValueError: No draws, a draw whose role is not ``"train"``, or a set
            of episodes that yields no observation.
    """
    if not draws:
        raise ValueError("observation statistics need at least one training draw")
    for draw in draws:
        if draw.role != "train":
            raise ValueError(
                f"draw {draw.index} is from role {draw.role!r}; statistics are computed "
                "from the training split only"
            )
    rows: list[np.ndarray] = []
    for draw in draws:
        spec, market = materialize(splits, draw, template)
        rows.extend(run_episode(spec, market, policy).observations)
    if not rows:
        raise ValueError("the training episodes produced no observation")  # pragma: no cover
    matrix = np.vstack(rows)
    return ObservationStatistics(
        fields=OBSERVATION_FIELDS,
        mean=tuple(float(v) for v in matrix.mean(axis=0)),
        std=tuple(float(v) for v in matrix.std(axis=0)),
        n_observations=int(matrix.shape[0]),
        n_episodes=len(draws),
        policy=policy_name,
        seed=draws[0].seed,
    )


def normalize(vector: np.ndarray, statistics: ObservationStatistics) -> np.ndarray:
    """Return ``(vector - mean) / std`` per field, dividing by 1 where ``std == 0``.

    Raises:
        ValueError: The vector's length differs from the statistics' field
            count.
    """
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (len(statistics.fields),):
        raise ValueError(
            f"vector has shape {values.shape}, statistics cover {len(statistics.fields)} fields"
        )
    mean = np.array(statistics.mean, dtype=np.float64)
    std = np.array(statistics.std, dtype=np.float64)
    return (values - mean) / np.where(std > 0, std, 1.0)
