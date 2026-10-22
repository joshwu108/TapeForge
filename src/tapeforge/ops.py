"""PyTorch-facing API (TF-07): typed tensors with documented shapes.

Public surface:

    features = tapeforge.ops.aggregate(trades, offsets)

Supported inputs (offsets must be the same kind as trades):

    trades                                      offsets               path
    ------------------------------------------  --------------------  ---------------
    NumPy structured array of TRADE_DTYPE       NumPy int64 array     CPU reference
    mapping of column -> CPU torch tensors      CPU int64 tensor      CPU reference
    mapping of column -> CUDA torch tensors     CUDA int64 tensor     Triton one-pass

Column requirements for the tensor path are in ``tapeforge.validation``.
Other devices (MPS, meta, ...) and mixed devices are rejected.

Returns a dict of tensors on the input device (CPU for NumPy input), each of
shape ``(n_buckets,)``:

    event_count     int64
    total_quantity  int64   quantity ticks
    signed_quantity int64   quantity ticks, signed
    vwap            float64 price units; 0.0 where total_quantity is 0
    valid           bool    False for empty buckets

Dispatch never falls back across devices. CUDA input without Triton raises
instead of being copied to the CPU, and a bucket too long for the one-pass
kernel raises ``kernels.BucketTooLargeError`` instead of being truncated or
silently computed elsewhere. Both paths check the int64 accumulation bound
(``contract.check_exact_accumulation_bound``) first.

Requires torch (``pip install -e '.[torch]'``). ``tapeforge.reference`` is
the torch-free NumPy API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from tapeforge import contract, reference

if TYPE_CHECKING:
    import torch

__all__ = ["aggregate"]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "tapeforge.ops needs torch: pip install -e '.[torch]'. "
            "For NumPy-only use, call tapeforge.reference.aggregate_reference."
        ) from exc
    return torch


def _structured_to_columns(trades: np.ndarray) -> dict[str, torch.Tensor]:
    torch = _import_torch()
    # Structured fields are strided views; copy each to a contiguous column.
    return {
        name: torch.from_numpy(np.ascontiguousarray(trades[name]))
        for name in ("price", "quantity", "buyer_is_maker")
    }


def _columns_to_structured(columns: Mapping[str, torch.Tensor], n_records: int) -> np.ndarray:
    out = np.zeros(n_records, dtype=contract.TRADE_DTYPE)
    for name in ("price", "quantity", "buyer_is_maker"):
        out[name] = columns[name].detach().numpy()
    return out


def _reference_features(trades: np.ndarray, offsets: np.ndarray) -> dict[str, torch.Tensor]:
    torch = _import_torch()
    features = reference.aggregate_reference(trades, offsets)
    return {
        name: torch.from_numpy(np.ascontiguousarray(getattr(features, name)))
        for name in contract.FEATURE_NAMES
    }


def _aggregate_numpy(trades: np.ndarray, offsets: Any) -> dict[str, torch.Tensor]:
    from tapeforge.validation import validate_columns

    torch = _import_torch()
    if trades.dtype != contract.TRADE_DTYPE:
        raise ValueError(f"trades must have TRADE_DTYPE, got {trades.dtype}")
    if not isinstance(offsets, np.ndarray):
        raise TypeError(
            f"NumPy trades need NumPy offsets, got {type(offsets).__name__}; "
            "pass both as NumPy or both as torch tensors"
        )
    if offsets.dtype != np.int64:
        raise ValueError(f"offsets must be int64, got {offsets.dtype}")
    offsets = np.ascontiguousarray(offsets)
    validate_columns(_structured_to_columns(trades), torch.from_numpy(offsets))
    return _reference_features(trades, offsets)


def _aggregate_tensors(
    trades: Mapping[str, torch.Tensor], offsets: Any
) -> dict[str, torch.Tensor]:
    from tapeforge.validation import validate_columns

    torch = _import_torch()
    if not isinstance(offsets, torch.Tensor):
        raise TypeError(
            f"tensor trades need tensor offsets, got {type(offsets).__name__}; "
            "pass both as NumPy or both as torch tensors"
        )
    if offsets.device.type == "cuda":
        from tapeforge import kernels

        return kernels.launch_aggregate(trades, offsets)
    summary = validate_columns(trades, offsets)
    return _reference_features(
        _columns_to_structured(trades, summary.n_records), offsets.detach().numpy()
    )


def aggregate(trades: Any, offsets: Any) -> dict[str, torch.Tensor]:
    """Compute per-bucket features, dispatching CUDA to Triton, CPU to the reference.

    Raises:
        TypeError: unsupported input types, or trades/offsets of different kinds.
        ValueError: invalid columns or offsets (see ``tapeforge.validation``),
            or an unsupported device.
        OverflowError: int64 bucket sums could wrap around.
        kernels.BucketTooLargeError: CUDA input with a bucket beyond the
            one-pass limit.
        RuntimeError: CUDA input but Triton is not installed.
    """
    if isinstance(trades, np.ndarray):
        return _aggregate_numpy(trades, offsets)
    if isinstance(trades, Mapping):
        return _aggregate_tensors(trades, offsets)
    raise TypeError(
        "trades must be a TRADE_DTYPE structured array or a mapping of column tensors, "
        f"got {type(trades).__name__}"
    )
