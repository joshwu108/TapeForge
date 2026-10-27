"""PyTorch-facing aggregation API with typed tensors and documented shapes.

``aggregate`` is the public entry point for torch callers. It dispatches on
the kind and device of its inputs and never falls back across devices: CUDA
input without Triton raises instead of being copied to the CPU. Repeated
execution on fixed inputs belongs in ``tapeforge.prepared``, which validates
once; ``tapeforge.reference`` is the torch-free NumPy API.

Data contract:
    ``offsets`` must be the same kind as ``trades``:

        trades                                      offsets               path
        ------------------------------------------  --------------------  ---------------
        NumPy structured array of TRADE_DTYPE       NumPy int64 array     CPU reference
        mapping of column -> CPU torch tensors      CPU int64 tensor      CPU reference
        mapping of column -> CUDA torch tensors     CUDA int64 tensor     Triton kernels

    Column requirements for the tensor path are in ``tapeforge.validation``.
    Other devices (MPS, meta, ...) and mixed devices are rejected.

    The result is a dict of tensors on the input device (CPU for NumPy input),
    each of shape ``(n_buckets,)``:

        event_count     int64
        total_quantity  int64   quantity ticks
        signed_quantity int64   quantity ticks, taker-signed
        vwap            float64 price units; 0.0 where total_quantity is 0
        valid           bool    False for empty buckets

Invariants:
    On CUDA the launcher takes the one-pass kernel when every bucket fits one
    tile (``kernels.MAX_ONE_PASS_BUCKET`` records) and the two-pass reduction
    otherwise; see ``tapeforge.kernels``. No bucket length is rejected or
    truncated here. Both devices check the int64 accumulation bound
    (``contract.check_exact_accumulation_bound``) before computing anything.

Not covered:
    This module requires torch (``pip install -e '.[torch]'``) and imports
    it lazily so the package still imports without it.
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
    """Import torch on demand, with an actionable error when it is missing."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "tapeforge.ops needs torch: pip install -e '.[torch]'. "
            "For NumPy-only use, call tapeforge.reference.aggregate_reference."
        ) from exc
    return torch


def _structured_to_columns(trades: np.ndarray) -> dict[str, torch.Tensor]:
    """Return the three kernel columns of a ``TRADE_DTYPE`` array as CPU tensors."""
    torch = _import_torch()
    # Structured fields are strided views; each is copied to a contiguous column
    # because validation rejects non-contiguous tensors.
    return {
        name: torch.from_numpy(np.ascontiguousarray(trades[name]))
        for name in ("price", "quantity", "buyer_is_maker")
    }


def _columns_to_structured(columns: Mapping[str, torch.Tensor], n_records: int) -> np.ndarray:
    """Pack CPU column tensors into a ``TRADE_DTYPE`` array for the reference path."""
    out = np.zeros(n_records, dtype=contract.TRADE_DTYPE)
    for name in ("price", "quantity", "buyer_is_maker"):
        out[name] = columns[name].detach().numpy()
    return out


def _reference_features(trades: np.ndarray, offsets: np.ndarray) -> dict[str, torch.Tensor]:
    """Run the NumPy reference and wrap each feature array as a CPU tensor."""
    torch = _import_torch()
    features = reference.aggregate_reference(trades, offsets)
    return {
        name: torch.from_numpy(np.ascontiguousarray(getattr(features, name)))
        for name in contract.FEATURE_NAMES
    }


def _aggregate_numpy(trades: np.ndarray, offsets: Any) -> dict[str, torch.Tensor]:
    """Validate NumPy input through the shared tensor checks and run the reference."""
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
    """Dispatch tensor input by device: CUDA to the kernels, CPU to the reference."""
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
    """Compute per-bucket features, dispatching CUDA to Triton and CPU to the reference.

    Args:
        trades: A ``TRADE_DTYPE`` structured array, or a mapping of column
            name to 1-D contiguous tensor on one device.
        offsets: Bucket boundaries of length ``n_buckets + 1``, of the same
            kind and on the same device as ``trades``.

    Returns:
        A dict with one tensor per name in ``contract.FEATURE_NAMES``, each of
        shape ``(n_buckets,)`` on the input device.

    Raises:
        TypeError: An input has an unsupported type, or ``trades`` and
            ``offsets`` are of different kinds.
        ValueError: A column or ``offsets`` fails validation (see
            ``tapeforge.validation``), or the device is unsupported.
        OverflowError: An int64 bucket sum could wrap around.
        RuntimeError: The input is on CUDA but Triton is not installed.
    """
    if isinstance(trades, np.ndarray):
        return _aggregate_numpy(trades, offsets)
    if isinstance(trades, Mapping):
        return _aggregate_tensors(trades, offsets)
    raise TypeError(
        "trades must be a TRADE_DTYPE structured array or a mapping of column tensors, "
        f"got {type(trades).__name__}"
    )
