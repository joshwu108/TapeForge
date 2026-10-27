"""Volatility-regime classifier over TapeForge features.

The model is a small temporal CNN whose purpose is to measure how feature-path
throughput affects training throughput; it is not a return predictor. This
module currently holds only the chronological split record and the label
generator interface.

Not covered:
    The network definition and the training loop are not implemented.
    ``make_labels`` raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Chronological train/validation/test boundaries, in bucket indices.

    Each interval ends at its boundary, exclusive, and begins where the
    previous one ends: train is ``[0, train_end)``, validation is
    ``[train_end, val_end)`` and test is ``[val_end, test_end)``.

    Attributes:
        train_end: First bucket index after the training interval.
        val_end: First bucket index after the validation interval.
        test_end: First bucket index after the test interval.
    """

    train_end: int
    val_end: int
    test_end: int


def make_labels(vwap, horizon: int, n_regimes: int = 3):  # noqa: ANN001, ANN201
    """Label each bucket with its forward realized-volatility regime.

    Labels use only prices strictly after the bucket, and the regime
    thresholds are quantiles computed on the training interval alone, so no
    information leaks backward in time or across split boundaries.

    Args:
        vwap: Per-bucket VWAP series, in price units.
        horizon: Number of forward buckets over which realized volatility is
            measured.
        n_regimes: Number of quantile-defined regimes.

    Raises:
        NotImplementedError: Always; label generation is not implemented.
    """
    raise NotImplementedError("Volatility-regime label generation is not implemented")
