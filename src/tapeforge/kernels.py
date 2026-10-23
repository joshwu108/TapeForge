"""Triton GPU kernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from tapeforge.contract import PRICE_SCALE, SIGN_BUY_INITIATED, SIGN_SELL_INITIATED

if TYPE_CHECKING:
    import torch

TRITON_AVAILABLE: bool
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


MAX_ONE_PASS_BUCKET: Final[int] = 4096
MIN_BLOCK_SIZE: Final[int] = 16
CHUNK_SIZE = 1024


class BucketTooLargeError(ValueError):
    """A bucket has more records than the one-pass kernel can reduce."""


def select_block_size(max_bucket_len: int, block_size: int | None = None) -> int:
    """Return the tile size for the one-pass kernel, or raise error
    Raises:
        ValueError: on an invalid explicit ``block_size``.
        BucketTooLargeError: if the longest bucket does not fit in one tile.
    """
    if block_size is not None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")
        if block_size > MAX_ONE_PASS_BUCKET:
            raise ValueError(
                f"block_size {block_size} exceeds the one-pass limit {MAX_ONE_PASS_BUCKET}"
            )
        limit = block_size
    else:
        limit = MAX_ONE_PASS_BUCKET
    if max_bucket_len > limit:
        raise BucketTooLargeError(
            f"a bucket has {max_bucket_len} records but the one-pass kernel handles at most "
            f"{limit}; the two-pass reduction for oversized buckets is not implemented"
        )
    if block_size is not None:
        return block_size
    return max(MIN_BLOCK_SIZE, 1 << max(0, max_bucket_len - 1).bit_length())


def _num_warps(block_size: int) -> int:
    # Keep each thread's slice of the tile small as the
    # tile grows (4 warps up to 1024 lanes, 16 warps at 4096).
    return min(16, max(4, block_size // 256))

def build_chunk_metadata(offsets, chunk_size):
    import torch
    # shift array to find chunk lengths
    lengths = offsets[1:] - offsets[:-1]
    chunks_per_bucket = (lengths + chunk_size - 1) // chunk_size

    chunk_offsets = torch.cat([offsets.new_zeros(1), chunks_per_bucket.cumsum(dim=0)])
    n_chunks = int(chunk_offsets[-1].item())
    bucket_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=offsets.device), chunks_per_bucket, output_size = n_chunks)

    # chunk number with original bucket
    local_chunk = (torch.arange(n_chunks, device=offsets.device) - chunk_offsets[bucket_ids])
    chunk_starts = offsets[bucket_ids] + local_chunk * chunk_size
    chunk_ends = torch.minimum(chunk_starts + chunk_size, offsets[bucket_ids + 1])
    return chunk_starts, chunk_ends, chunk_offsets

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def segment_sums(  # type: ignore[no-untyped-def]
        values_ptr, offsets_ptr, output_ptr, BLOCK_SIZE: tl.constexpr
    ):
        # Load Segment Boundaries
        pid = tl.program_id(axis=0)
        start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)

        # load values
        indices = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(values_ptr + indices, mask=indices < end, other=0)

        sums = tl.sum(values, axis=0)
        tl.store(output_ptr + pid, sums)

    @triton.jit  # type: ignore[untyped-decorator]
    def aggregate_features(  # type: ignore[no-untyped-def]
        price_ptr,
        quantity_ptr,
        buyer_is_maker_ptr,
        offsets_ptr,
        total_quantity_ptr,
        event_count_ptr,
        signed_quantity_ptr,
        vwap_ptr,
        valid_ptr,
        BLOCK_SIZE: tl.constexpr,
        PRICE_SCALE: tl.constexpr,
        SIGN_BUY: tl.constexpr,
        SIGN_SELL: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)
        length = end - start

        # the launcher guarantees length <= BLOCK_SIZE, so one tile covers the bucket
        indices = start + tl.arange(0, BLOCK_SIZE)
        in_bucket = indices < end

        # calculate quantity sums (int64, exact)
        quantities = tl.load(quantity_ptr + indices, mask=in_bucket, other=0)
        total_quantity = tl.sum(quantities, axis=0)

        # signed quantity: buyer_is_maker means the seller was the aggressor
        buyer_is_maker = tl.load(buyer_is_maker_ptr + indices, mask=in_bucket, other=False)
        signed = tl.where(buyer_is_maker, SIGN_SELL, SIGN_BUY)
        signed_quantity = tl.sum(quantities * signed, axis=0)

        # calculate aggregate numerator sums (int64, exact within the contract bound)
        prices = tl.load(price_ptr + indices, mask=in_bucket, other=0)
        numerator_sum = tl.sum(prices * quantities, axis=0)

        # calculate vwap
        has_quantity = total_quantity > 0
        denominator = tl.where(has_quantity, total_quantity, 1).to(tl.float64)
        vwap = numerator_sum.to(tl.float64) / denominator / PRICE_SCALE
        vwap = tl.where(has_quantity, vwap, 0.0)

        # store calculated outputs
        tl.store(event_count_ptr + pid, length)
        tl.store(total_quantity_ptr + pid, total_quantity)
        tl.store(signed_quantity_ptr + pid, signed_quantity)
        tl.store(vwap_ptr + pid, vwap)
        tl.store(valid_ptr + pid, length > 0)

    @triton.jit
    def aggregate_partials(price_ptr,
            quantity_ptr,
            maker_ptr,
            chunk_start_ptr,
            chunk_end_ptr,
            partial_total_ptr,
            partial_signed_ptr,
            partial_numerator_ptr,
            BLOCK_SIZE: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        start, end = tl.load(chunk_start_ptr + chunk), tl.load(chunk_end_ptr + chunk)
        indices = start + tl.arange(0, BLOCK_SIZE)
        mask = indices < end

        quantities = tl.load(quantity_ptr + indices, mask=mask, other = 0)
        prices = tl.load(price_ptr + indices, mask=mask, other = 0)
        makers = tl.load(maker_ptr + indices, mask=mask, other = False)

        total = tl.sum(quantities, axis = 0)
        signed = tl.sum(quantities * tl.where(makers, -1, 1), axis=0)
        numerator = tl.sum(prices * quantities, axis = 0)

        tl.store(partial_total_ptr + chunk, total)
        tl.store(partial_signed_ptr + chunk, signed)
        tl.store(partial_numerator_ptr + chunk, numerator)

    @triton.jit
    def combine_partials(
        partial_total_ptr, 
        partial_signed_ptr,
        partial_numerator_ptr,
        chunk_offsets_ptr,
        offsets_ptr,
        total_quantity_ptr,
        event_count_ptr,
        signed_quantity_ptr,
        vwap_ptr,
        valid_ptr,
        BLOCK_SIZE: tl.constexpr,
        PRICE_SCALE: tl.constexpr
    ):
        bucket = tl.program_id(0)
        first_chunk = tl.load(chunk_offsets_ptr + bucket)
        last_chunk = tl.load(chunk_offsets_ptr + bucket + 1)

        # initialize accumulators using tl.full()
        lanes = tl.arange(0, BLOCK_SIZE)
        totals = tl.full((BLOCK_SIZE,), 0, tl.int64)
        signeds = tl.full((BLOCK_SIZE,), 0, tl.int64)
        numerators = tl.full((BLOCK_SIZE,), 0, tl.int64)

        for base in range(first_chunk, last_chunk, BLOCK_SIZE):
            indices = base + lanes
            mask = indices < last_chunk
            totals += tl.load(partial_total_ptr + indices, mask = mask, other = 0)
            signeds += tl.load(partial_signed_ptr + indices, mask=mask, other=0)
            numerators += tl.load(partial_numerator_ptr + indices, mask=mask, other = 0)

        total = tl.sum(totals, axis = 0)
        signed = tl.sum(signeds, axis = 0)
        numerator = tl.sum(numerators, axis = 0)

        start = tl.load(offsets_ptr + bucket)
        end = tl.load(offsets_ptr + bucket + 1)
        count = end - start

        has_quantity = total > 0
        denominator = tl.where(has_quantity, total, 1).to(tl.float64)
        vwap = numerator.to(tl.float64) / denominator / PRICE_SCALE
        vwap = tl.where(has_quantity, vwap, 0.0)

        tl.store(total_quantity_ptr + bucket, total)
        tl.store(signed_quantity_ptr + bucket, signed)
        tl.store(event_count_ptr + bucket, count)
        tl.store(vwap_ptr + bucket, vwap)
        tl.store(valid_ptr + bucket, count > 0)



def launch_aggregate(
    trades: Mapping[str, torch.Tensor],
    offsets: torch.Tensor,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Run the one-pass kernel on CUDA tensors and return the five features.

    Input format and validation: ``tapeforge.validation.validate_columns``.

    Raises:
        RuntimeError: Triton is not installed.
        TypeError: offsets is not a tensor, or a column is not a tensor.
        ValueError: inputs are not on CUDA or fail validation.
        BucketTooLargeError: a bucket exceeds the one-pass limit.
        OverflowError: int64 sums could wrap around.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed; the GPU path needs Linux, an NVIDIA GPU and "
            "pip install -e '.[gpu]'. Use tapeforge.ops.aggregate with CPU tensors instead."
        )
    import torch

    from tapeforge.validation import validate_columns

    if not isinstance(offsets, torch.Tensor):
        raise TypeError(f"offsets must be a torch.Tensor, got {type(offsets).__name__}")
    if offsets.device.type != "cuda":
        raise ValueError("launch_aggregate needs CUDA tensors; use ops.aggregate for CPU inputs")
    summary = validate_columns(trades, offsets)

    use_one_pass = (block_size is not None or summary.max_bucket_len <= MAX_ONE_PASS_BUCKET)
    block = None
    if use_one_pass:
        block = select_block_size(summary.max_bucket_len, block_size)

    with torch.cuda.device(offsets.device):
        features = empty_features(summary.n_buckets, offsets.device)
        if summary.max_bucket_len == 0:
            return features

        if use_one_pass:
            launch_one_pass_unchecked(trades, offsets, features, block)
            return features

        starts, ends, chunk_offsets = build_chunk_metadata(offsets, CHUNK_SIZE)
        n_chunks =starts.numel()
        partial_total = torch.empty(n_chunks, dtype=torch.int64, device=offsets.device)
        partial_signed = torch.empty_like(partial_total)
        partial_numerator = torch.empty_like(partial_total)

        aggregate_partials[(n_chunks,)](
            trades["price"],
            trades["quantity"],
            trades["buyer_is_maker"],
            starts,
            ends,
            partial_total,
            partial_signed,
            partial_numerator,
            BLOCK_SIZE=CHUNK_SIZE,
            num_warps=_num_warps(CHUNK_SIZE),
        )

        combine_partials[(summary.n_buckets,)](
            partial_total,
            partial_signed,
            partial_numerator,
            chunk_offsets,
            offsets,
            features["total_quantity"],
            features["event_count"],
            features["signed_quantity"],
            features["vwap"],
            features["valid"],
            BLOCK_SIZE=128,
            PRICE_SCALE=PRICE_SCALE,
            num_warps=4,
        )
        return features


def empty_features(n_buckets: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Zero-filled output tensors: the correct answer for empty buckets."""
    import torch

    return {
        "event_count": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "total_quantity": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "signed_quantity": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "vwap": torch.zeros(n_buckets, dtype=torch.float64, device=device),
        "valid": torch.zeros(n_buckets, dtype=torch.bool, device=device),
    }


def launch_one_pass_unchecked(
    trades: Mapping[str, torch.Tensor],
    offsets: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    block_size: int,
) -> None:
    """Launch the kernel into preallocated outputs. NO validation.

    Only for callers that have already run ``validate_columns`` and
    ``select_block_size`` on these exact tensors -- the kernel-only benchmark,
    which must not time validation. Invalid input here reads out of bounds.
    """
    import torch

    with torch.cuda.device(offsets.device):
        aggregate_features[(offsets.numel() - 1,)](
            trades["price"],
            trades["quantity"],
            trades["buyer_is_maker"],
            offsets,
            features["total_quantity"],
            features["event_count"],
            features["signed_quantity"],
            features["vwap"],
            features["valid"],
            BLOCK_SIZE=block_size,
            PRICE_SCALE=PRICE_SCALE,
            SIGN_BUY=SIGN_BUY_INITIATED,
            SIGN_SELL=SIGN_SELL_INITIATED,
            num_warps=_num_warps(block_size),
        )