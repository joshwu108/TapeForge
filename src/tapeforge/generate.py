"""Seeded synthetic trade generation (TF-01).

Deterministic fixtures: the same seed always reproduces byte-identical
records. Covers the shapes that break naive implementations -- empty
windows, duplicate timestamps, and large bursts.

CLI:
    python -m tapeforge.generate --seed 7 --seconds 10 --profile burst
"""

from __future__ import annotations

import argparse
from typing import Final, Literal

import numpy as np

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

Profile = Literal["sparse", "uniform", "burst"]

# DECISION (student): the three published stress distributions of PRD section 7.
# Values are mean records per bucket; "burst" is heavy-tailed on purpose.
PROFILE_RATES: Final[dict[str, float]] = {
    "sparse": 0.5,
    "uniform": 50.0,
    "burst": 50.0,
}
BURST_PARETO_SHAPE: Final[float] = 1.5


def generate_trades(
    seed: int,
    n_buckets: int,
    profile: Profile = "uniform",
    start_us: int = 0,
    width_us: int = DEFAULT_BUCKET_WIDTH_US,
) -> np.ndarray:
    """Return a timestamp-sorted structured array of ``TRADE_DTYPE`` records.

    Deterministic in ``seed`` alone, given identical other arguments.
    """
    raise NotImplementedError("week 1: implement after tests/test_generate.py is RED")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write a reproducible sample to data/ as .npy."""
    parser = argparse.ArgumentParser(prog="python -m tapeforge.generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=int, default=10, help="number of 1s buckets")
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_RATES), default="uniform"
    )
    parser.add_argument("--out", default="data/sample_trades.npy")
    args = parser.parse_args(argv)
    del args  # week 1: the parsed args drive generate_trades() and the .npy write
    raise NotImplementedError("week 1: implement after tests/test_generate.py is RED")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
