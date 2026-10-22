"""Pure-PyTorch baseline for the kernel benchmark: searchsorted + index_add_.

Same inputs and outputs as ``tapeforge.ops.aggregate`` on tensors, built only
from stock PyTorch ops, with no host sync, so it runs eagerly or under
``torch.compile`` on any device. Every record is mapped to its bucket with
``torch.searchsorted`` over ``offsets``; records outside
``[offsets[0], offsets[-1])`` get zero weight. Sums use int64
``index_add_`` (exact), and VWAP is one float64 division, in the same order
as the reference.

This is a fair baseline, not a straw man: it reads each input once and
writes each output once, like the fused kernel, but goes through several
separate PyTorch kernels and a searchsorted.
"""

from __future__ import annotations

from typing import Any

from tapeforge import contract


def torch_index_add(columns: dict[str, Any], offsets: Any) -> dict[str, Any]:
    import torch

    price, quantity = columns["price"], columns["quantity"]
    n_records, n_buckets = quantity.numel(), offsets.numel() - 1
    record = torch.arange(n_records, device=quantity.device, dtype=torch.int64)
    bucket = torch.searchsorted(offsets, record, right=True) - 1
    inside = (bucket >= 0) & (bucket < n_buckets)
    bucket = bucket.clamp(0, max(n_buckets - 1, 0))

    weight = torch.where(inside, quantity, torch.zeros_like(quantity))
    sign = torch.where(
        columns["buyer_is_maker"],
        contract.SIGN_SELL_INITIATED,
        contract.SIGN_BUY_INITIATED,
    )
    zeros = torch.zeros(n_buckets, dtype=torch.int64, device=quantity.device)
    total = zeros.index_add(0, bucket, weight)
    signed = zeros.index_add(0, bucket, weight * sign)
    numerator = zeros.index_add(0, bucket, weight * price)

    lengths = offsets[1:] - offsets[:-1]
    has_quantity = total > 0
    vwap = torch.where(
        has_quantity,
        numerator.to(torch.float64) / total.clamp(min=1).to(torch.float64) / contract.PRICE_SCALE,
        torch.zeros((), dtype=torch.float64, device=quantity.device),
    )
    return {
        "event_count": lengths,
        "total_quantity": total,
        "signed_quantity": signed,
        "vwap": vwap,
        "valid": lengths > 0,
    }
