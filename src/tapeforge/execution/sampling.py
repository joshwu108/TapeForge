"""Seeded sampling of episodes inside one split role.

A draw is a pure function of ``(seed, role, index)``: its generator is
``numpy.random.default_rng(SeedSequence(seed, spawn_key=(role_number,
index)))``, so draw ``index`` is the same whether 10 or 10,000 episodes are
sampled, draws never depend on each other, and a run record that lists
``(seed, index)`` per episode is enough to rebuild any episode exactly.

Draw order:
    Per draw: the start position (uniform over the role's admissible starts
    in the ``SplitRecord``), the side (buy or sell with equal probability)
    and ``rho_bps`` (uniform over the given target fractions, in basis
    points). ``target_qty_ticks`` is sized when the episode is cut
    (``Splits.episode``), from the reference volume before the start; the
    draw itself carries no market data.

Not covered:
    This module chooses nothing about the split rules and runs no policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tapeforge.execution.spec import (
    ACCEPTED_TARGET_FRACTIONS_BPS,
    SIDE_BUY,
    SIDE_SELL,
    EpisodeSpec,
)
from tapeforge.execution.splits import ROLES, SplitRecord, Splits
from tapeforge.execution.state import MarketSlice

__all__ = [
    "EpisodeDraw",
    "draw_episode",
    "materialize",
    "sample_episodes",
]


@dataclass(frozen=True)
class EpisodeDraw:
    """One sampled episode: where it starts and what it trades, without features.

    Attributes:
        seed: Base seed of the sampling run.
        role: One of ``ROLES``.
        index: Draw index within the seed and role.
        position: Index into the role's admissible starts
            (``SplitRecord.start_at``).
        segment: Resolved segment index.
        start_bucket: Resolved segment-local start bucket.
        date: Calendar day holding the start bucket.
        side: ``SIDE_BUY`` or ``SIDE_SELL``.
        rho_bps: Target fraction of the reference volume, in basis points.
    """

    seed: int
    role: str
    index: int
    position: int
    segment: int
    start_bucket: int
    date: str
    side: int
    rho_bps: int

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict of built-in types; ``(seed, role, index)`` rebuilds the draw."""
        return asdict(self)


def _generator(seed: int, role: str, index: int) -> np.random.Generator:
    """Return the generator for draw ``index`` of ``seed`` in ``role``.

    Raises:
        ValueError: Unknown role, or a negative seed or index.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    if seed < 0 or index < 0:
        raise ValueError(f"seed and index must be non-negative, got {seed} and {index}")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ROLES.index(role), index)))


def draw_episode(
    record: SplitRecord,
    role: str,
    seed: int,
    index: int,
    rho_choices_bps: tuple[int, ...] = ACCEPTED_TARGET_FRACTIONS_BPS,
) -> EpisodeDraw:
    """Return the ``index``-th draw of ``seed`` in ``role``.

    See the module docstring for the order in which the fields are drawn.

    Raises:
        ValueError: Unknown role, negative seed or index, empty
            ``rho_choices_bps``, or a role with no admissible start.
    """
    if not rho_choices_bps:
        raise ValueError("rho_choices_bps must not be empty")
    # Arguments are checked before the record is consulted, so a bad role is
    # reported the same way whether or not the record knows it.
    rng = _generator(seed, role, index)
    n_starts = record.n_starts(role)
    if n_starts == 0:
        raise ValueError(f"role {role!r} has no admissible episode start to sample from")
    position = int(rng.integers(n_starts))
    side = SIDE_BUY if int(rng.integers(2)) == 0 else SIDE_SELL
    rho_bps = int(rho_choices_bps[int(rng.integers(len(rho_choices_bps)))])
    segment, start = record.start_at(role, position)
    return EpisodeDraw(
        seed=seed,
        role=role,
        index=index,
        position=position,
        segment=segment,
        start_bucket=start,
        date=record.segments[segment].date_at(start),
        side=side,
        rho_bps=rho_bps,
    )


def sample_episodes(
    record: SplitRecord,
    role: str,
    n_episodes: int,
    seed: int,
    rho_choices_bps: tuple[int, ...] = ACCEPTED_TARGET_FRACTIONS_BPS,
) -> tuple[EpisodeDraw, ...]:
    """Return draws ``0 .. n_episodes - 1`` of ``seed`` in ``role``, each independent.

    Raises:
        ValueError: Negative ``n_episodes``, or anything ``draw_episode``
            rejects.
    """
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be non-negative, got {n_episodes}")
    return tuple(
        draw_episode(record, role, seed, index, rho_choices_bps) for index in range(n_episodes)
    )


def materialize(
    splits: Splits, draw: EpisodeDraw, template: EpisodeSpec
) -> tuple[EpisodeSpec, MarketSlice]:
    """Cut the episode a draw names: ``Splits.episode`` with the draw's side and ``rho_bps``."""
    return splits.episode(draw.role, draw.position, draw.side, draw.rho_bps, template)
