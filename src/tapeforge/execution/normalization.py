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
    (``ObservationStatistics.to_json``). The baselines do not use them, since
    they read exact quantities rather than the scaled vector. The
    observations depend on the reference policy's actions through the
    remaining fraction, so the policy name is recorded with the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tapeforge.execution.runner import Policy, run_episode
from tapeforge.execution.sampling import EpisodeDraw, materialize
from tapeforge.execution.simulator import OBSERVATION_FIELDS
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import Splits

__all__ = [
    "ObservationStatistics",
    "normalize",
    "observation_statistics",
]


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
        """Return a plain dict of built-in types, stored verbatim in run records."""
        return asdict(self)


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
