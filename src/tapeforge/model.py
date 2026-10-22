"""Small volatility-regime classifier over TapeForge features

A deliberately small temporal CNN. It exists to answer one systems
question -- does a faster feature path change training throughput? -- not
to predict returns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Chronological interval boundaries, in bucket indices."""

    train_end: int
    val_end: int
    test_end: int


def make_labels(vwap, horizon: int, n_regimes: int = 3):  # noqa: ANN001, ANN201
    """Label each bucket's forward realized-volatility regime.

    Uses only prices strictly after the bucket, and only train-interval
    quantiles for the regime thresholds.
    """
    raise NotImplementedError("week 7: implement after tests/test_model.py is RED")
