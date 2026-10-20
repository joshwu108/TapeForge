"""PyTorch-facing API (TF-07): typed tensors with documented shapes.

Public surface:

    features = tapeforge.aggregate(trades, offsets)

Returns a dict of tensors, each of shape ``(n_buckets,)``:

    event_count     int64
    total_quantity  int64   quantity ticks
    signed_quantity int64   quantity ticks, signed
    vwap            float64 undefined where valid is False
    valid           bool

Dispatch: CUDA inputs with a working Triton build go to the student's
kernels in ``tapeforge.kernels``; everything else falls back to the NumPy
reference so the API works on a CPU-only machine.

Gradients (PRD section 5): aggregating historical trades does not need a
backward pass through the trade inputs, so this is registered as a plain
custom operator. A custom *autograd* operator gets implemented, tested and
claimed only if a real need appears -- STUDENT DECISION, left open.
``torch.library.triton_op`` integration is the week-5 task.
"""

from __future__ import annotations

from typing import Any

__all__ = ["aggregate"]


def aggregate(trades: Any, offsets: Any) -> dict[str, Any]:
    """Compute per-bucket features, dispatching GPU vs CPU reference."""
    raise NotImplementedError("week 5: implement after tests/test_ops.py is RED")
