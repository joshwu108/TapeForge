"""Triton GPU kernels and launchers for per-bucket trade aggregation on CUDA.

Two kernel paths produce the five features of the data contract. The
one-pass path (``aggregate_features``) runs one program per bucket and
reduces the whole bucket in a single tile of ``block_size`` records, so it
requires every bucket to fit ``MAX_ONE_PASS_BUCKET``. The two-pass path
splits each bucket into ``CHUNK_SIZE``-record chunks, reduces each chunk to
int64 partials (``aggregate_partials``, one program per chunk) and then
combines each bucket's partials (``combine_partials``, one program per
bucket). ``plan_dispatch`` chooses between them; ``launch_aggregate`` is
the checked entry point behind ``ops.aggregate``.

Data contract:
    Inputs are the validated columns of ``tapeforge.validation``: 1-D
    contiguous CUDA tensors, int64 ``price`` and ``quantity`` ticks, bool
    ``buyer_is_maker``, and int64 ``offsets`` of length ``n_buckets + 1``.
    Outputs are the tensors of ``empty_features``.

Invariants:
    All sums are int64 and exact within
    ``contract.check_exact_accumulation_bound``. The sign convention is the
    contract's: ``buyer_is_maker`` selects ``SIGN_SELL_INITIATED``. VWAP is
    computed once per bucket as ``numerator / quantity / PRICE_SCALE`` in
    float64, never on partials, so both paths and the CPU reference agree
    within ``contract.VWAP_RTOL`` / ``contract.VWAP_ATOL``. Empty buckets
    yield count 0, VWAP 0.0 and ``valid == False``.

Not covered:
    The ``*_unchecked`` launchers perform no validation and exist for
    ``tapeforge.prepared`` and the kernel-only benchmark. The module imports
    without Triton (``TRITON_AVAILABLE`` is then ``False``); the launchers
    raise ``RuntimeError`` in that case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

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

MAX_ONE_PASS_BUCKET: Final[int] = 4096  # Largest bucket (records) the one-pass tile handles.
MIN_BLOCK_SIZE: Final[int] = 16         # Smallest one-pass tile chosen automatically.
CHUNK_SIZE: Final[int] = 1024           # Records per pass-1 chunk on the two-pass path.

# Partials per pass-2 tile. A bucket with more chunks than this is reduced by
# looping over tiles inside its program (``combine_partials``).
COMBINE_BLOCK_SIZE: Final[int] = 128

DispatchPath = Literal["one_pass", "two_pass"]
PathChoice = Literal["auto", "one_pass", "two_pass"]


class BucketTooLargeError(ValueError):
    """A bucket has more records than the one-pass tile it was asked to use."""


@dataclass(frozen=True)
class DispatchPlan:
    """Kernel path chosen for a validated input, with the tile or chunk it uses.

    Attributes:
        path: ``"one_pass"`` or ``"two_pass"``.
        block_size: One-pass tile in records, a power of two; ``None`` on the
            two-pass path.
        chunk_size: Pass-1 chunk in records; ``None`` on the one-pass path.
    """

    path: DispatchPath
    block_size: int | None
    chunk_size: int | None


@dataclass(frozen=True)
class ChunkMetadata:
    """Device-resident pass-1 schedule for the two-pass path.

    Derived from ``offsets`` only; changing the offsets invalidates it.

    Attributes:
        chunk_starts: int64 first record index of each chunk.
        chunk_ends: int64 one-past-last record index of each chunk.
        chunk_offsets: int64 of length ``n_buckets + 1``; bucket ``b`` owns
            chunks ``[chunk_offsets[b], chunk_offsets[b + 1])``.
        n_chunks: Total chunks, as a host integer; it sizes the pass-1 grid.
        chunk_size: Maximum records per chunk; the pass-1 tile.
    """

    chunk_starts: torch.Tensor
    chunk_ends: torch.Tensor
    chunk_offsets: torch.Tensor
    n_chunks: int
    chunk_size: int


def select_block_size(max_bucket_len: int, block_size: int | None = None) -> int:
    """Return the one-pass tile for a longest bucket of ``max_bucket_len`` records.

    Without an explicit ``block_size`` the tile is the smallest power of two
    that holds the longest bucket, but at least ``MIN_BLOCK_SIZE``.

    Args:
        max_bucket_len: Records in the longest bucket.
        block_size: Explicit tile; must be a positive power of two no larger
            than ``MAX_ONE_PASS_BUCKET``.

    Raises:
        ValueError: ``block_size`` is not a positive power of two, or exceeds
            ``MAX_ONE_PASS_BUCKET``.
        BucketTooLargeError: The longest bucket does not fit in one tile.
            Without an explicit ``block_size`` the launcher never calls this
            for such inputs; it dispatches them to the two-pass path instead.
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
            f"a bucket has {max_bucket_len} records but a one-pass tile holds at most "
            f"{limit}; buckets beyond {MAX_ONE_PASS_BUCKET} records take the two-pass path "
            "when no explicit block_size is given"
        )
    if block_size is not None:
        return block_size
    return max(MIN_BLOCK_SIZE, 1 << max(0, max_bucket_len - 1).bit_length())


def plan_dispatch(
    max_bucket_len: int, block_size: int | None = None, path: PathChoice = "auto"
) -> DispatchPlan:
    """Decide the kernel path for a validated input.

    ``path="auto"`` (used by ``launch_aggregate``) takes one-pass when the
    longest bucket fits ``MAX_ONE_PASS_BUCKET`` and two-pass otherwise. An
    explicit ``block_size`` implies ``path="one_pass"``. ``path="two_pass"``
    forces the two-pass path on inputs that would fit one tile, so the
    benchmark can measure both paths on the same workload.

    Args:
        max_bucket_len: Records in the longest bucket.
        block_size: Explicit one-pass tile, or ``None`` to choose one.
        path: ``"auto"``, ``"one_pass"`` or ``"two_pass"``.

    Raises:
        ValueError: ``path`` is unknown, ``block_size`` is given with
            ``path="two_pass"``, or ``block_size`` is invalid.
        BucketTooLargeError: The one-pass path was requested (explicitly or
            via ``block_size``) for a bucket longer than its tile.
    """
    if path not in ("auto", "one_pass", "two_pass"):
        raise ValueError(f"path must be 'auto', 'one_pass' or 'two_pass', got {path!r}")
    if path == "two_pass":
        if block_size is not None:
            raise ValueError("block_size applies to the one-pass path only")
        return DispatchPlan(path="two_pass", block_size=None, chunk_size=CHUNK_SIZE)
    if path == "one_pass" or block_size is not None or max_bucket_len <= MAX_ONE_PASS_BUCKET:
        return DispatchPlan(
            path="one_pass",
            block_size=select_block_size(max_bucket_len, block_size),
            chunk_size=None,
        )
    return DispatchPlan(path="two_pass", block_size=None, chunk_size=CHUNK_SIZE)


def _num_warps(block_size: int) -> int:
    """Return the warps per program for a tile of ``block_size`` lanes.

    Each thread's slice of the tile stays small as the tile grows: 4 warps up
    to 1024 lanes, 16 warps at 4096.
    """
    return min(16, max(4, block_size // 256))


def build_chunk_metadata(offsets: torch.Tensor, chunk_size: int) -> ChunkMetadata:
    """Split every bucket into ``chunk_size``-record chunks, numbered bucket by bucket.

    Runs on whichever device ``offsets`` lives on (CPU included, so the
    schedule is testable without a GPU). An empty bucket gets no chunk; a
    bucket of ``n`` records gets ``ceil(n / chunk_size)`` chunks, the last
    one ragged. Every chunk therefore holds at least one record, which lets
    pass 1 write every partial slot and pass 2 read them without
    zero-filling. One host sync (``n_chunks``) is unavoidable because it
    sizes the pass-1 grid.

    Args:
        offsets: Validated bucket boundaries (int64, non-decreasing, 1-D).
        chunk_size: Maximum records per chunk; must be positive.

    Raises:
        ValueError: ``chunk_size`` is not positive.
    """
    import torch

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # Bucket lengths come from adjacent offsets; chunks per bucket is a ceiling division.
    lengths = offsets[1:] - offsets[:-1]
    chunks_per_bucket = (lengths + chunk_size - 1) // chunk_size

    chunk_offsets = torch.cat([offsets.new_zeros(1), chunks_per_bucket.cumsum(dim=0)])
    n_chunks = int(chunk_offsets[-1].item())
    bucket_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=offsets.device),
        chunks_per_bucket,
        output_size=n_chunks,
    )

    # Each chunk's position within its bucket gives its record range; the last
    # chunk of a bucket is clipped to the bucket's end.
    local_chunk = torch.arange(n_chunks, device=offsets.device) - chunk_offsets[bucket_ids]
    chunk_starts = offsets[bucket_ids] + local_chunk * chunk_size
    chunk_ends = torch.minimum(chunk_starts + chunk_size, offsets[bucket_ids + 1])
    return ChunkMetadata(
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_offsets=chunk_offsets,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
    )


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def segment_sums(  # type: ignore[no-untyped-def]
        values_ptr, offsets_ptr, output_ptr, BLOCK_SIZE: tl.constexpr
    ):
        # Generic segmented sum: one program per segment, one tile per segment.
        # The caller guarantees every segment fits BLOCK_SIZE records.
        pid = tl.program_id(axis=0)
        start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)

        # Lanes past the segment end load zero, so they do not affect the sum.
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
        # One-pass kernel: one program per bucket, all five features in one tile.
        pid = tl.program_id(axis=0)
        start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)
        length = end - start

        # The launcher guarantees length <= BLOCK_SIZE, so one tile covers the
        # bucket; lanes past the end are masked and load zero.
        indices = start + tl.arange(0, BLOCK_SIZE)
        in_bucket = indices < end

        # Quantity sum in int64, exact within the contract bound.
        quantities = tl.load(quantity_ptr + indices, mask=in_bucket, other=0)
        total_quantity = tl.sum(quantities, axis=0)

        # Taker-signed quantity: buyer_is_maker means the seller was the aggressor.
        buyer_is_maker = tl.load(buyer_is_maker_ptr + indices, mask=in_bucket, other=False)
        signed = tl.where(buyer_is_maker, SIGN_SELL, SIGN_BUY)
        signed_quantity = tl.sum(quantities * signed, axis=0)

        # VWAP numerator sum(price * quantity) in int64, exact within the contract bound.
        prices = tl.load(price_ptr + indices, mask=in_bucket, other=0)
        numerator_sum = tl.sum(prices * quantities, axis=0)

        # The single floating-point step, in the same order as the CPU reference.
        # A unit denominator avoids a division by zero; the result is then
        # replaced by 0.0 for buckets without quantity.
        has_quantity = total_quantity > 0
        denominator = tl.where(has_quantity, total_quantity, 1).to(tl.float64)
        vwap = numerator_sum.to(tl.float64) / denominator / PRICE_SCALE
        vwap = tl.where(has_quantity, vwap, 0.0)

        # Every bucket is written, empty ones included.
        tl.store(event_count_ptr + pid, length)
        tl.store(total_quantity_ptr + pid, total_quantity)
        tl.store(signed_quantity_ptr + pid, signed_quantity)
        tl.store(vwap_ptr + pid, vwap)
        tl.store(valid_ptr + pid, length > 0)

    @triton.jit  # type: ignore[untyped-decorator]
    def aggregate_partials(  # type: ignore[no-untyped-def]
        price_ptr,
        quantity_ptr,
        maker_ptr,
        chunk_start_ptr,
        chunk_end_ptr,
        partial_total_ptr,
        partial_signed_ptr,
        partial_numerator_ptr,
        BLOCK_SIZE: tl.constexpr,
        SIGN_BUY: tl.constexpr,
        SIGN_SELL: tl.constexpr,
    ):
        # Pass 1: one program per chunk; the launcher guarantees
        # chunk_end - chunk_start <= BLOCK_SIZE. Integer partials only, so no
        # rounding happens before pass 2 combines them.
        chunk = tl.program_id(0)
        start, end = tl.load(chunk_start_ptr + chunk), tl.load(chunk_end_ptr + chunk)
        indices = start + tl.arange(0, BLOCK_SIZE)
        mask = indices < end

        quantities = tl.load(quantity_ptr + indices, mask=mask, other=0)
        prices = tl.load(price_ptr + indices, mask=mask, other=0)
        makers = tl.load(maker_ptr + indices, mask=mask, other=False)

        total = tl.sum(quantities, axis=0)
        # Same sign convention as the one-pass kernel, from the contract.
        signed = tl.sum(quantities * tl.where(makers, SIGN_SELL, SIGN_BUY), axis=0)
        numerator = tl.sum(prices * quantities, axis=0)

        tl.store(partial_total_ptr + chunk, total)
        tl.store(partial_signed_ptr + chunk, signed)
        tl.store(partial_numerator_ptr + chunk, numerator)

    @triton.jit  # type: ignore[untyped-decorator]
    def combine_partials(  # type: ignore[no-untyped-def]
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
        PRICE_SCALE: tl.constexpr,
    ):
        # Pass 2: one program per bucket. Reduces the bucket's contiguous slice
        # of partials in BLOCK_SIZE tiles (several iterations for a bucket with
        # more than BLOCK_SIZE chunks), then divides once.
        bucket = tl.program_id(0)
        first_chunk = tl.load(chunk_offsets_ptr + bucket)
        last_chunk = tl.load(chunk_offsets_ptr + bucket + 1)

        # Per-lane int64 accumulators; the cross-lane reduction happens once
        # after the loop.
        lanes = tl.arange(0, BLOCK_SIZE)
        totals = tl.full((BLOCK_SIZE,), 0, tl.int64)
        signeds = tl.full((BLOCK_SIZE,), 0, tl.int64)
        numerators = tl.full((BLOCK_SIZE,), 0, tl.int64)

        for base in range(first_chunk, last_chunk, BLOCK_SIZE):
            indices = base + lanes
            mask = indices < last_chunk
            totals += tl.load(partial_total_ptr + indices, mask=mask, other=0)
            signeds += tl.load(partial_signed_ptr + indices, mask=mask, other=0)
            numerators += tl.load(partial_numerator_ptr + indices, mask=mask, other=0)

        total = tl.sum(totals, axis=0)
        signed = tl.sum(signeds, axis=0)
        numerator = tl.sum(numerators, axis=0)

        # The event count comes from the bucket offsets, not from the chunks, so
        # an empty bucket (no chunks) still reports zero.
        start = tl.load(offsets_ptr + bucket)
        end = tl.load(offsets_ptr + bucket + 1)
        count = end - start

        # The single floating-point step, in the same order as the CPU reference.
        has_quantity = total > 0
        denominator = tl.where(has_quantity, total, 1).to(tl.float64)
        vwap = numerator.to(tl.float64) / denominator / PRICE_SCALE
        vwap = tl.where(has_quantity, vwap, 0.0)

        tl.store(total_quantity_ptr + bucket, total)
        tl.store(signed_quantity_ptr + bucket, signed)
        tl.store(event_count_ptr + bucket, count)
        tl.store(vwap_ptr + bucket, vwap)
        tl.store(valid_ptr + bucket, count > 0)


def _require_triton() -> None:
    """Raise ``RuntimeError`` with installation guidance when Triton is unavailable."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed; the GPU path needs Linux, an NVIDIA GPU and "
            "pip install -e '.[gpu]'. Use tapeforge.ops.aggregate with CPU tensors instead."
        )


def launch_aggregate(
    trades: Mapping[str, torch.Tensor],
    offsets: torch.Tensor,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Validate CUDA inputs, pick a path, allocate outputs, launch and return the features.

    This is the checked public entry point behind ``ops.aggregate`` for CUDA
    input. Input format and validation follow
    ``tapeforge.validation.validate_columns``; path choice follows
    ``plan_dispatch`` (one-pass up to ``MAX_ONE_PASS_BUCKET`` records per
    bucket, two-pass beyond; ``block_size`` forces one-pass). Output tensors
    are fresh on every call and nothing is cached between calls. For
    repeated execution on fixed inputs use ``tapeforge.prepared``.

    Args:
        trades: Mapping of column name to CUDA tensor.
        offsets: CUDA int64 bucket boundaries of length ``n_buckets + 1``.
        block_size: Explicit one-pass tile, or ``None`` for automatic choice.

    Returns:
        A dict with one tensor per name in ``contract.FEATURE_NAMES``, each of
        shape ``(n_buckets,)`` on the input device.

    Raises:
        RuntimeError: Triton is not installed.
        TypeError: ``offsets`` or a column is not a tensor.
        ValueError: The inputs are not on CUDA or fail validation.
        BucketTooLargeError: An explicit ``block_size`` is smaller than the
            longest bucket.
        OverflowError: An int64 sum could wrap around.
    """
    _require_triton()
    import torch

    from tapeforge.validation import validate_columns

    if not isinstance(offsets, torch.Tensor):
        raise TypeError(f"offsets must be a torch.Tensor, got {type(offsets).__name__}")
    if offsets.device.type != "cuda":
        raise ValueError("launch_aggregate needs CUDA tensors; use ops.aggregate for CPU inputs")
    summary = validate_columns(trades, offsets)
    plan = plan_dispatch(summary.max_bucket_len, block_size)

    with torch.cuda.device(offsets.device):
        features = empty_features(summary.n_buckets, offsets.device)
        if summary.max_bucket_len == 0:
            # Every bucket is empty, so the zero-filled outputs are the answer.
            return features
        if plan.path == "one_pass":
            assert plan.block_size is not None
            launch_one_pass_unchecked(trades, offsets, features, plan.block_size)
            return features
        assert plan.chunk_size is not None
        chunks = build_chunk_metadata(offsets, plan.chunk_size)
        partials = allocate_partials(chunks.n_chunks, offsets.device)
        launch_two_pass_unchecked(trades, offsets, features, chunks, partials)
        return features


def empty_features(n_buckets: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate zero-filled output tensors, which are the correct answer for empty buckets.

    Returns:
        A dict keyed by ``contract.FEATURE_NAMES`` with the contract dtypes
        (int64, int64, int64, float64, bool), each of shape ``(n_buckets,)``.
    """
    import torch

    return {
        "event_count": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "total_quantity": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "signed_quantity": torch.zeros(n_buckets, dtype=torch.int64, device=device),
        "vwap": torch.zeros(n_buckets, dtype=torch.float64, device=device),
        "valid": torch.zeros(n_buckets, dtype=torch.bool, device=device),
    }


PARTIAL_NAMES: Final[tuple[str, ...]] = ("total", "signed", "numerator")  # Pass-1 scratch keys.


def allocate_partials(n_chunks: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate pass-1 scratch: three int64 arrays of length ``n_chunks``.

    The buffers are uninitialised on purpose. Every chunk holds at least one
    record (``build_chunk_metadata``), so pass 1 writes every slot before
    pass 2 reads it, and pass 2 only reads slots inside a bucket's chunk
    range.

    Returns:
        A dict keyed by ``PARTIAL_NAMES`` (structure of arrays).
    """
    import torch

    return {
        name: torch.empty(n_chunks, dtype=torch.int64, device=device) for name in PARTIAL_NAMES
    }


def launch_one_pass_unchecked(
    trades: Mapping[str, torch.Tensor],
    offsets: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    block_size: int,
) -> None:
    """Launch the one-pass kernel into preallocated outputs without validation.

    Only for callers that have already run ``validate_columns`` and
    ``plan_dispatch`` on these exact tensors: ``tapeforge.prepared`` and the
    kernel-only benchmark, which must not time validation. Invalid input
    reads out of bounds. Launches on the current stream of the offsets
    device.

    Args:
        trades: Validated CUDA columns.
        offsets: Validated CUDA bucket boundaries.
        features: Output tensors from ``empty_features``.
        block_size: Tile from ``plan_dispatch``; every bucket must fit it.
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


def launch_pass_1_unchecked(
    trades: Mapping[str, torch.Tensor],
    chunks: ChunkMetadata,
    partials: Mapping[str, torch.Tensor],
) -> None:
    """Launch pass 1 only, writing int64 partial sums for every chunk, without validation.

    A zero-chunk schedule launches nothing.

    Args:
        trades: Validated CUDA columns.
        chunks: Schedule from ``build_chunk_metadata`` on the validated
            offsets of ``trades``.
        partials: Scratch from ``allocate_partials`` with ``chunks.n_chunks``
            slots.
    """
    import torch

    if chunks.n_chunks == 0:
        return
    with torch.cuda.device(chunks.chunk_starts.device):
        aggregate_partials[(chunks.n_chunks,)](
            trades["price"],
            trades["quantity"],
            trades["buyer_is_maker"],
            chunks.chunk_starts,
            chunks.chunk_ends,
            partials["total"],
            partials["signed"],
            partials["numerator"],
            BLOCK_SIZE=chunks.chunk_size,
            SIGN_BUY=SIGN_BUY_INITIATED,
            SIGN_SELL=SIGN_SELL_INITIATED,
            num_warps=_num_warps(chunks.chunk_size),
        )


def launch_pass_2_unchecked(
    offsets: torch.Tensor,
    chunks: ChunkMetadata,
    partials: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
) -> None:
    """Launch pass 2 only, combining each bucket's partials into the five features.

    Performs no validation. Reads the partials pass 1 wrote for this same
    ``chunks`` schedule. Idempotent, so it can be re-run for timing without
    re-running pass 1. Writes every bucket, empty ones included (count 0,
    VWAP 0.0, invalid).

    Args:
        offsets: Validated CUDA bucket boundaries.
        chunks: The schedule pass 1 ran with.
        partials: The scratch pass 1 wrote.
        features: Output tensors from ``empty_features``.
    """
    import torch

    with torch.cuda.device(offsets.device):
        combine_partials[(offsets.numel() - 1,)](
            partials["total"],
            partials["signed"],
            partials["numerator"],
            chunks.chunk_offsets,
            offsets,
            features["total_quantity"],
            features["event_count"],
            features["signed_quantity"],
            features["vwap"],
            features["valid"],
            BLOCK_SIZE=COMBINE_BLOCK_SIZE,
            PRICE_SCALE=PRICE_SCALE,
            num_warps=4,
        )


def launch_two_pass_unchecked(
    trades: Mapping[str, torch.Tensor],
    offsets: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    chunks: ChunkMetadata,
    partials: Mapping[str, torch.Tensor],
) -> None:
    """Launch both passes in order into preallocated outputs and scratch, without validation."""
    launch_pass_1_unchecked(trades, chunks, partials)
    launch_pass_2_unchecked(offsets, chunks, partials, features)
