"""Seeded synthetic trade generation for tests, fixtures and benchmarks.

``generate_trades`` is deterministic: the same seed and arguments always
reproduce byte-identical ``TRADE_DTYPE`` records (int64 microsecond
timestamps, int64 price and quantity ticks, the raw ``buyer_is_maker`` flag,
increasing ids). The profiles cover the shapes that break naive
implementations: empty windows, duplicate timestamps and large bursts.
Prices follow a random walk around 100.00, so synthetic tapes carry no
information about real market behaviour.

Profiles:
    ``sparse``   Mean 0.5 records per bucket, so most buckets are empty.
                 Exposes implementations that assume every bucket has data,
                 and any division by a zero record count.
    ``uniform``  Mean 50 records per bucket. The easy case, used as the
                 baseline when comparing against padded implementations.
    ``burst``    Mean 50, but a small fraction of buckets are Pareto-amplified
                 into thousands of records. This distribution makes padding
                 to the longest bucket wasteful and forces the two-pass
                 reduction path on the GPU.

Data contract:
    Event timestamps are quantized to ``EVENT_TIME_TICK_US`` (1 ms) inside
    each bucket, because an aggTrades row already aggregates fills that share
    a millisecond. Duplicate timestamps are therefore the norm rather than a
    rare edge case, which matches the real archive.

CLI:
    python -m tapeforge.generate --seed 7 --seconds 10 --profile burst
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Literal

import numpy as np

from tapeforge.contract import (
    DEFAULT_BUCKET_WIDTH_US,
    PRICE_SCALE,
    QUANTITY_SCALE,
    TRADE_DTYPE,
)

Profile = Literal["sparse", "uniform", "burst"]
PROFILE_RATES: Final[dict[str, float]] = {  # Poisson mean records per bucket.
    "sparse": 0.5,
    "uniform": 50.0,
    "burst": 50.0,
}

# Burst shape: the fraction of buckets that erupt, and how strongly.
BURST_FRACTION: Final[float] = 0.02       # Probability that a bucket erupts.
BURST_PARETO_SHAPE: Final[float] = 1.5    # Pareto tail index of the multiplier.
BURST_MIN_MULTIPLIER: Final[int] = 20     # Smallest count multiplier of an erupting bucket.

# Timestamp resolution within a bucket: 1 ms, matching aggTrades aggregation.
EVENT_TIME_TICK_US: Final[int] = 1_000

# Price random walk, in price ticks: starts at 100.00 and moves by one cent per step.
START_PRICE_TICKS: Final[int] = 100 * PRICE_SCALE
PRICE_STEP_TICKS: Final[int] = PRICE_SCALE // 100  # One cent, in price ticks.

# Quantity: lognormal around 1.0 unit, floored at one tick so it is positive.
QUANTITY_LOG_MEAN: Final[float] = 0.0
QUANTITY_LOG_SIGMA: Final[float] = 1.0

AGG_TRADE_ID_START: Final[int] = 1  # First ``agg_trade_id``; ids increase by one.


def _bucket_counts(
    rng: np.random.Generator, n_buckets: int, profile: Profile
) -> np.ndarray:
    """Draw the records per bucket as int64 of shape ``(n_buckets,)``."""
    counts = rng.poisson(PROFILE_RATES[profile], size=n_buckets).astype(np.int64)
    if profile != "burst":
        return counts
    erupting = rng.random(n_buckets) < BURST_FRACTION
    multipliers = BURST_MIN_MULTIPLIER * (
        1.0 + rng.pareto(BURST_PARETO_SHAPE, size=n_buckets)
    )
    return np.where(erupting, counts * multipliers.astype(np.int64), counts)


def generate_trades(
    seed: int,
    n_buckets: int,
    profile: Profile = "uniform",
    start_us: int = 0,
    width_us: int = DEFAULT_BUCKET_WIDTH_US,
) -> np.ndarray:
    """Return a timestamp-sorted structured array of ``TRADE_DTYPE`` records.

    Output is a deterministic function of ``seed`` given identical other
    arguments.

    Args:
        seed: Seed for ``numpy.random.default_rng``.
        n_buckets: Number of buckets to populate; must be positive.
        profile: One of ``PROFILE_RATES``.
        start_us: Timestamp of the first bucket edge, in microseconds.
        width_us: Bucket width in microseconds; must be a positive multiple
            of ``EVENT_TIME_TICK_US``.

    Raises:
        ValueError: ``n_buckets`` or ``width_us`` is not positive,
            ``width_us`` is not a multiple of the event tick, or ``profile``
            is unknown. Failing fast avoids returning a degenerate array.
    """
    if profile not in PROFILE_RATES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {tuple(PROFILE_RATES)}")
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")
    if width_us <= 0:
        raise ValueError(f"width_us must be positive, got {width_us}")
    if width_us % EVENT_TIME_TICK_US:
        raise ValueError(
            f"width_us={width_us} must be a multiple of the {EVENT_TIME_TICK_US}us event tick"
        )

    rng = np.random.default_rng(seed)
    counts = _bucket_counts(rng, n_buckets, profile)
    n = int(counts.sum())

    trades = np.zeros(n, dtype=TRADE_DTYPE)
    if n == 0:
        return trades

    # Each record gets its bucket's start plus a quantized offset inside the bucket.
    bucket_index = np.repeat(np.arange(n_buckets, dtype=np.int64), counts)
    ticks_per_bucket = width_us // EVENT_TIME_TICK_US
    offset_ticks = rng.integers(0, ticks_per_bucket, size=n, dtype=np.int64)
    timestamps = start_us + bucket_index * width_us + offset_ticks * EVENT_TIME_TICK_US

    # Records are sorted within buckets. bucket_index is already non-decreasing,
    # so a stable sort on the timestamp alone leaves every record in its bucket.
    order = np.argsort(timestamps, kind="stable")

    walk = rng.integers(-1, 2, size=n, dtype=np.int64).cumsum() * PRICE_STEP_TICKS
    prices = START_PRICE_TICKS + walk
    quantities = np.maximum(
        1,
        (
            rng.lognormal(QUANTITY_LOG_MEAN, QUANTITY_LOG_SIGMA, size=n) * QUANTITY_SCALE
        ).astype(np.int64),
    )

    trades["timestamp"] = timestamps[order]
    trades["price"] = prices[order]
    trades["quantity"] = quantities[order]
    trades["buyer_is_maker"] = rng.random(n) < 0.5
    trades["agg_trade_id"] = np.arange(AGG_TRADE_ID_START, AGG_TRADE_ID_START + n, dtype=np.int64)
    return trades


def main(argv: list[str] | None = None) -> int:
    """Write a reproducible sample tape to disk as ``.npy`` and print its shape.

    Returns:
        Process exit status; 0 on success.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=int, default=10, help="number of 1s buckets")
    parser.add_argument("--profile", choices=tuple(PROFILE_RATES), default="uniform")
    parser.add_argument("--out", default="data/sample_trades.npy")
    args = parser.parse_args(argv)

    trades = generate_trades(seed=args.seed, n_buckets=args.seconds, profile=args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, trades)

    occupied = np.unique(trades["timestamp"] // DEFAULT_BUCKET_WIDTH_US).size
    print(f"wrote {out} ({trades.size} records, {args.profile}, seed {args.seed})")
    print(f"  buckets: {args.seconds} requested, {occupied} non-empty")
    if trades.size:
        _, per_bucket = np.unique(
            trades["timestamp"] // DEFAULT_BUCKET_WIDTH_US, return_counts=True
        )
        print(
            f"  records per non-empty bucket: max {per_bucket.max()},"
            f" median {int(np.median(per_bucket))}"
        )
        print(f"  price range: {trades['price'].min() / PRICE_SCALE:.2f}"
              f" .. {trades['price'].max() / PRICE_SCALE:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
