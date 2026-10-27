"""Input validation shared by ``ops.aggregate`` and ``kernels.launch_aggregate``.

``validate_columns`` checks layout, bounds and the int64 accumulation bound
for the tensor path in a single host sync, and returns the facts the
launcher needs to plan dispatch.

Data contract:
    ``columns`` maps column name to a 1-D, contiguous tensor on the same
    device as ``offsets``:

        price           int64  price ticks (price * PRICE_SCALE), non-negative
        quantity        int64  quantity ticks (quantity * QUANTITY_SCALE), non-negative
        buyer_is_maker  bool   raw exchange flag; True means sell-initiated

    Other columns (``timestamp``, ``agg_trade_id``) may be present and are
    ignored. ``offsets`` is a 1-D contiguous int64 tensor of length
    ``n_buckets + 1``, non-decreasing, with ``0 <= offsets[0]`` and
    ``offsets[-1] <= n_records``.

Invariants:
    Contiguity is required rather than silently fixed. The kernels index raw
    pointers with stride 1, so a strided view would read the wrong records,
    and a hidden ``.contiguous()`` copy would distort transfer and kernel
    timings.

    The non-negativity and int64 checks scan whole columns, including records
    outside ``[offsets[0], offsets[-1])``. This is conservative (a window can
    be rejected because of data it never reads) but never unsafe.

Not covered:
    This module imports torch at the top. Callers import it lazily so the
    package still imports on a machine without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from tapeforge import contract

REQUIRED_COLUMNS: dict[str, torch.dtype] = {
    "price": torch.int64,
    "quantity": torch.int64,
    "buyer_is_maker": torch.bool,
}

SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cpu", "cuda")


@dataclass(frozen=True)
class InputSummary:
    """Facts about a validated input, gathered with a single device sync.

    Attributes:
        n_records: Length of every column.
        n_buckets: ``offsets.numel() - 1``.
        max_bucket_len: Records in the longest bucket; decides the kernel path.
        longest_bucket: Index of the first bucket with ``max_bucket_len`` records.
    """

    n_records: int
    n_buckets: int
    max_bucket_len: int
    longest_bucket: int


def _check_offsets_layout(offsets: torch.Tensor) -> None:
    """Check type, device, shape, dtype and contiguity of ``offsets`` without a sync."""
    if not isinstance(offsets, torch.Tensor):
        raise TypeError(f"offsets must be a torch.Tensor, got {type(offsets).__name__}")
    if offsets.device.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"unsupported device {offsets.device}; expected one of {SUPPORTED_DEVICE_TYPES}"
        )
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(
            f"offsets must be 1-D with at least 2 entries, got shape {tuple(offsets.shape)}"
        )
    if offsets.dtype != torch.int64:
        raise ValueError(f"offsets must be int64, got {offsets.dtype}")
    if not offsets.is_contiguous():
        raise ValueError("offsets must be contiguous; call .contiguous() before aggregating")


def _check_column(name: str, column: object, dtype: torch.dtype, device: torch.device) -> None:
    """Check that one column is a 1-D contiguous tensor of ``dtype`` on ``device``."""
    if not isinstance(column, torch.Tensor):
        raise TypeError(f"column {name!r} must be a torch.Tensor, got {type(column).__name__}")
    if column.ndim != 1:
        raise ValueError(f"column {name!r} must be 1-D, got shape {tuple(column.shape)}")
    if column.dtype != dtype:
        raise ValueError(f"column {name!r} must be {dtype}, got {column.dtype}")
    if column.device != device:
        raise ValueError(
            f"column {name!r} is on {column.device} but offsets are on {device}; "
            "all inputs must share one device"
        )
    if not column.is_contiguous():
        raise ValueError(
            f"column {name!r} must be contiguous; call .contiguous() before aggregating"
        )


def validate_columns(
    columns: Mapping[str, torch.Tensor], offsets: torch.Tensor
) -> InputSummary:
    """Validate layout, bounds and the exact-accumulation bound of a tensor input.

    Layout checks run without touching device memory. The value checks
    (bucket lengths, offset range, price and quantity extremes) are reduced on
    the device and transferred in one host sync.

    Args:
        columns: Mapping of column name to tensor; see the module docstring.
        offsets: Bucket boundaries of length ``n_buckets + 1``.

    Returns:
        An ``InputSummary`` describing the validated input.

    Raises:
        TypeError: A column or ``offsets`` is not a tensor.
        ValueError: A required column is missing; a dtype, shape or device is
            wrong; an input is non-contiguous; ``offsets`` decrease or lie
            outside the record range; or a price or quantity is negative.
        OverflowError: An int64 bucket sum could wrap around
            (``contract.check_exact_accumulation_bound``).
    """
    if not isinstance(columns, Mapping):
        raise TypeError(f"trades must be a mapping of column tensors, got {type(columns).__name__}")
    _check_offsets_layout(offsets)
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"trades is missing columns {missing}; required: {list(REQUIRED_COLUMNS)}")
    for name, dtype in REQUIRED_COLUMNS.items():
        _check_column(name, columns[name], dtype, offsets.device)

    n_records = columns["price"].numel()
    sizes = {name: columns[name].numel() for name in REQUIRED_COLUMNS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"columns must have equal length, got {sizes}")

    price, quantity = columns["price"], columns["quantity"]
    lengths = offsets[1:] - offsets[:-1]
    zero = offsets.new_zeros(())
    stats = torch.stack(
        [
            lengths.min(),
            lengths.max(),
            lengths.argmax(),
            offsets[0],
            offsets[-1],
            price.min() if n_records else zero,
            price.max() if n_records else zero,
            quantity.min() if n_records else zero,
            quantity.max() if n_records else zero,
        ]
    )
    (
        min_len, max_len, longest, first, last, min_price, max_price, min_qty, max_qty
    ) = (int(v) for v in stats.tolist())  # The single host sync of validation.

    if min_len < 0:
        raise ValueError("offsets must be non-decreasing; came from a different trade array?")
    if first < 0 or last > n_records:
        raise ValueError(
            f"offsets span [{first}, {last}] but there are {n_records} records"
        )
    if min_price < 0 or min_qty < 0:
        raise ValueError(
            f"price and quantity must be non-negative ticks, got min price {min_price}, "
            f"min quantity {min_qty}"
        )
    contract.check_exact_accumulation_bound(max_len, max_price, max_qty)
    return InputSummary(
        n_records=n_records,
        n_buckets=offsets.numel() - 1,
        max_bucket_len=max_len,
        longest_bucket=longest,
    )
