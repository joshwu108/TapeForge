"""Week 5: the PyTorch-facing API (TF-07)."""

from __future__ import annotations

import pytest

from tapeforge import ops

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="week 5: delete this marker to begin the RED phase",
)


def test_cpu_inputs_fall_back_to_the_reference() -> None:
    torch = pytest.importorskip("torch")
    trades = {"timestamp": torch.zeros(1, dtype=torch.int64)}
    ops.aggregate(trades, torch.tensor([0, 1], dtype=torch.int64))


def test_returns_documented_keys_shapes_and_dtypes() -> None:
    ops.aggregate({}, None)
