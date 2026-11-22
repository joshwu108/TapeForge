"""Host-to-device aggregation in bounded batches with overlapped copies and compute.

``aggregate_host`` turns host-resident trade columns into host-resident
features through the GPU without moving the whole input at once. The
buckets are cut into consecutive batches of at most ``batch_records``
records (a bucket longer than that forms a batch alone), and each batch
passes through three CUDA streams:

    copy_in   pinned staging -> device inputs     (host-to-device engine)
    compute   the existing Triton kernels          (SMs)
    copy_out  device outputs -> pinned staging     (device-to-host engine)

With ``slots`` sets of buffers, batch ``i`` uses slot ``i % slots``, so the
copy of batch ``i + 1`` can run while batch ``i`` is computed and batch
``i - 1`` is copied back. Dependencies are CUDA events, never
``synchronize`` calls:

    copy_in(i)   waits compute(i - slots)   device inputs of the slot are free
    compute(i)   waits copy_in(i)           inputs have arrived
                 waits copy_out(i - slots)  device outputs of the slot are free
    copy_out(i)  waits compute(i)           outputs are written
    host         waits copy_out(i - slots)  before reading that slot's pinned
                                            outputs and refilling its pinned inputs

The host reads results only after the event of the copy that produced them,
so a returned array is never read while an asynchronous copy still writes
it. Each slot owns its device and pinned buffers; no buffer is shared by two
batches in flight.

Semantics:
    The per-bucket features equal ``ops.aggregate`` on the whole input:
    batches split the input only at bucket boundaries, every bucket is
    reduced by the same one-pass or two-pass kernel the checked API would
    choose for a batch of that longest bucket, and VWAP is divided once per
    bucket. Validation runs once on the whole input on the host (layout,
    monotone offsets, non-negative values, the global int64 accumulation
    bound, which bounds every batch); batches are then launched through the
    unchecked launchers. Batch plans and two-pass chunk schedules are
    computed on the host, so the loop has no device-to-host synchronisation
    other than the event waits listed above.

Memory:
    Per slot, pinned and device inputs of ``17 * capacity_records`` bytes
    (int64 price and quantity, bool maker flag) plus offsets, and pinned and
    device outputs of ``33 * capacity_buckets`` bytes, where the capacities
    are the largest batch. Peak memory is independent of the input length
    except through the result arrays, which are ordinary host arrays of
    ``n_buckets`` entries.

Not covered:
    The grouped short-bucket path; sparse tapes therefore pay the one-pass
    kernel's one program per bucket. The function needs CUDA and Triton and
    raises otherwise.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tapeforge import contract, kernels

if TYPE_CHECKING:
    import torch

__all__ = [
    "BatchPlan",
    "OverlappedAggregator",
    "PipelineStats",
    "aggregate_host",
    "chunk_schedule",
    "overlap_ms",
    "plan_batches",
    "timeline_intervals",
]

INPUT_COLUMNS: tuple[tuple[str, Any], ...] = (
    ("price", np.int64),
    ("quantity", np.int64),
    ("buyer_is_maker", np.bool_),
)
OUTPUT_DTYPES: dict[str, Any] = {
    "event_count": np.int64,
    "total_quantity": np.int64,
    "signed_quantity": np.int64,
    "vwap": np.float64,
    "valid": np.bool_,
}


@dataclass(frozen=True)
class BatchPlan:
    """One batch: buckets ``[first, last)``, records ``[record_start, record_end)``.

    Attributes:
        first: First bucket of the batch.
        last: One past the last bucket.
        record_start: ``offsets[first]``.
        record_end: ``offsets[last]``.
        max_bucket_len: Longest bucket in the batch; picks the kernel path.
        n_chunks: Pass-1 chunks of the batch on the two-pass path, else 0.
    """

    first: int
    last: int
    record_start: int
    record_end: int
    max_bucket_len: int
    n_chunks: int

    @property
    def n_buckets(self) -> int:
        """Return the number of buckets in the batch."""
        return self.last - self.first

    @property
    def n_records(self) -> int:
        """Return the number of records in the batch."""
        return self.record_end - self.record_start

    @property
    def two_pass(self) -> bool:
        """Return whether the batch's longest bucket needs the two-pass path."""
        return self.max_bucket_len > kernels.MAX_ONE_PASS_BUCKET


def plan_batches(offsets: np.ndarray, batch_records: int) -> list[BatchPlan]:
    """Cut the buckets into consecutive batches of at most ``batch_records`` records.

    Greedy: each batch takes as many whole buckets as fit; a bucket longer
    than ``batch_records`` forms a batch alone, and runs of empty buckets
    join the batch they follow. The batches cover every bucket exactly once
    and in order.

    Args:
        offsets: Validated int64 bucket boundaries of length ``n_buckets + 1``.
        batch_records: Record budget per batch; must be positive.

    Raises:
        ValueError: ``batch_records`` is not positive.
    """
    if batch_records <= 0:
        raise ValueError(f"batch_records must be positive, got {batch_records}")
    offsets = np.asarray(offsets, dtype=np.int64)
    n_buckets = offsets.size - 1
    lengths = np.diff(offsets)
    plans = []
    first = 0
    while first < n_buckets:
        limit = offsets[first] + batch_records
        # The last edge at or below the limit closes the batch; at least one bucket.
        last = int(np.searchsorted(offsets, limit, side="right")) - 1
        last = min(max(last, first + 1), n_buckets)
        batch_lengths = lengths[first:last]
        longest = int(batch_lengths.max())
        n_chunks = (
            int((-(-batch_lengths // kernels.CHUNK_SIZE)).sum())
            if longest > kernels.MAX_ONE_PASS_BUCKET
            else 0
        )
        plans.append(
            BatchPlan(first, last, int(offsets[first]), int(offsets[last]), longest, n_chunks)
        )
        first = last
    return plans


def chunk_schedule(local_offsets: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    """Return ``kernels.build_chunk_metadata`` for ``local_offsets``, computed with NumPy.

    Host-side twin of the device builder, so the pipeline never reads a chunk
    count back from the device. Same layout: an empty bucket has no chunk,
    a bucket of ``n`` records has ``ceil(n / chunk_size)``, the last ragged.

    Returns:
        ``chunk_starts``, ``chunk_ends`` and ``chunk_offsets`` as int64 arrays.
    """
    lengths = np.diff(local_offsets)
    per_bucket = -(-lengths // chunk_size)
    chunk_offsets = np.concatenate([[0], np.cumsum(per_bucket)]).astype(np.int64)
    owner = np.repeat(np.arange(lengths.size, dtype=np.int64), per_bucket)
    local = np.arange(int(chunk_offsets[-1]), dtype=np.int64) - chunk_offsets[owner]
    starts = local_offsets[owner] + local * chunk_size
    ends = np.minimum(starts + chunk_size, local_offsets[owner + 1])
    return {"chunk_starts": starts, "chunk_ends": ends, "chunk_offsets": chunk_offsets}


@dataclass
class PipelineStats:
    """What one ``aggregate_host`` call did, for benchmarks and traces.

    Attributes:
        n_batches: Batches launched.
        slots: Buffer sets in rotation.
        capacity_records: Records per slot buffer.
        capacity_buckets: Buckets per slot buffer.
        pinned_bytes: Page-locked host bytes allocated for staging.
        device_bytes: Device bytes allocated for slot buffers and scratch.
        setup_seconds: Validation, planning and allocation (host wall clock).
        run_seconds: The batch loop through the last result copy.
        timeline: With ``trace=True``, one entry per batch with the start and
            end of its ``copy_in``, ``compute`` and ``copy_out`` in
            milliseconds from the first recorded event (CUDA events, so the
            device's own clock); otherwise empty.
    """

    n_batches: int = 0
    slots: int = 0
    capacity_records: int = 0
    capacity_buckets: int = 0
    pinned_bytes: int = 0
    device_bytes: int = 0
    setup_seconds: float = 0.0
    run_seconds: float = 0.0
    timeline: list[dict[str, Any]] = field(default_factory=list)


def _columns(trades: Any) -> dict[str, np.ndarray]:
    """Return the three kernel columns as contiguous NumPy arrays (no copy when possible)."""
    if isinstance(trades, np.ndarray) and trades.dtype.names:
        # Field views are strided; copy() packs them (also for a single record,
        # whose view ascontiguousarray would return unchanged).
        return {name: trades[name].copy() for name, _ in INPUT_COLUMNS}
    if isinstance(trades, Mapping):
        columns = {name: np.ascontiguousarray(trades[name]) for name, _ in INPUT_COLUMNS}
        return {name: c if c.strides == (c.itemsize,) else c.copy() for name, c in columns.items()}
    raise TypeError("trades must be a TRADE_DTYPE array or a mapping of NumPy columns")


def _validate_host(columns: dict[str, np.ndarray], offsets: np.ndarray) -> int:
    """Validate the whole host input once and return its longest bucket.

    Raises:
        ValueError: A dtype or shape is wrong, offsets decrease or leave the
            record range, or a price or quantity is negative.
        OverflowError: The int64 accumulation bound is exceeded.
    """
    for name, dtype in INPUT_COLUMNS:
        column = columns[name]
        if column.dtype != dtype or column.ndim != 1:
            raise ValueError(f"column {name!r} must be 1-D {np.dtype(dtype)}, got "
                             f"{column.dtype} {column.shape}")
    n = columns["price"].size
    if any(columns[name].size != n for name, _ in INPUT_COLUMNS):
        raise ValueError("columns must have equal length")
    if offsets.dtype != np.int64 or offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("offsets must be 1-D int64 with at least two entries")
    lengths = np.diff(offsets)
    if lengths.size and lengths.min() < 0:
        raise ValueError("offsets must be non-decreasing")
    if offsets[0] < 0 or offsets[-1] > n:
        raise ValueError(f"offsets span [{offsets[0]}, {offsets[-1]}] but there are {n} records")
    window = slice(int(offsets[0]), int(offsets[-1]))
    price, quantity = columns["price"][window], columns["quantity"][window]
    max_price = int(price.max()) if price.size else 0
    max_quantity = int(quantity.max()) if quantity.size else 0
    if price.size and (int(price.min()) < 0 or int(quantity.min()) < 0):
        raise ValueError("price and quantity must be non-negative ticks")
    longest = int(lengths.max()) if lengths.size else 0
    contract.check_exact_accumulation_bound(longest, max_price, max_quantity)
    return longest


def _meta_length(n_buckets: int, n_chunks: int) -> int:
    """Return the int64 slots for local offsets and, with ``n_chunks >= 0``, a chunk schedule.

    Layout: offsets (``n_buckets + 1``), then chunk starts and ends
    (``n_chunks`` each) and chunk offsets (``n_buckets + 1``).
    """
    if n_chunks < 0:
        return n_buckets + 1
    return 2 * (n_buckets + 1) + 2 * n_chunks


class _Slot:
    """One set of pinned staging, device buffers and events; used by one batch at a time."""

    def __init__(self, torch: Any, records: int, buckets: int, chunks: int, trace: bool) -> None:
        def pinned(n: int, dtype: Any) -> torch.Tensor:
            return torch.empty(n, dtype=dtype, pin_memory=True)

        def device(n: int, dtype: Any) -> torch.Tensor:
            return torch.empty(n, dtype=dtype, device="cuda")

        dtypes = {"price": torch.int64, "quantity": torch.int64, "buyer_is_maker": torch.bool}
        self.host_in = {name: pinned(records, dtype) for name, dtype in dtypes.items()}
        self.dev_in = {name: device(records, dtype) for name, dtype in dtypes.items()}
        # Offsets plus, on two-pass batches, the chunk schedule share one int64 buffer.
        self.meta_len = _meta_length(buckets, chunks)
        self.host_meta = pinned(self.meta_len, torch.int64)
        self.dev_meta = device(self.meta_len, torch.int64)
        out_dtypes = {
            "event_count": torch.int64, "total_quantity": torch.int64,
            "signed_quantity": torch.int64, "vwap": torch.float64, "valid": torch.bool,
        }
        self.dev_out = {name: device(buckets, dtype) for name, dtype in out_dtypes.items()}
        self.host_out = {name: pinned(buckets, dtype) for name, dtype in out_dtypes.items()}
        self.partials = kernels.allocate_partials(max(chunks, 1), torch.device("cuda"))
        self.copied_in = torch.cuda.Event()
        self.computed = torch.cuda.Event()
        self.copied_out = torch.cuda.Event()
        self.batch: BatchPlan | None = None  # The batch whose results the slot holds.
        self.marks: list[Any] = []
        self.trace = trace
        self.pinned_bytes = sum(t.numel() * t.element_size() for t in
                                [*self.host_in.values(), *self.host_out.values(), self.host_meta])
        self.device_bytes = sum(t.numel() * t.element_size() for t in
                                [*self.dev_in.values(), *self.dev_out.values(), self.dev_meta,
                                 *self.partials.values()])

    def mark(self, torch: Any) -> Any:
        """Record a timing event on the current stream when tracing."""
        if not self.trace:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event


class OverlappedAggregator:
    """Validated input, batch plan, slot buffers and streams for repeated runs.

    The constructor is the setup (host validation, planning, pinned and
    device allocation); ``run`` is the batch loop. The input arrays are
    borrowed and must not change between runs. One object runs one call at a
    time: its slots are reused by every ``run``.

    Args:
        trades: A ``contract.TRADE_DTYPE`` array or a mapping with NumPy
            ``price``, ``quantity`` and ``buyer_is_maker`` columns.
        offsets: int64 NumPy bucket boundaries of length ``n_buckets + 1``.
        batch_records: Record budget per batch.
        slots: Buffer sets in rotation, at least 1; 1 serialises the batches.
        trace: Record CUDA timing events per stage into ``stats.timeline``.

    Raises:
        RuntimeError: CUDA or Triton is unavailable.
        ValueError: The input fails validation or ``slots`` < 1.
        OverflowError: The int64 accumulation bound is exceeded.
    """

    def __init__(
        self,
        trades: Any,
        offsets: np.ndarray,
        *,
        batch_records: int = 1 << 20,
        slots: int = 2,
        trace: bool = False,
    ) -> None:
        kernels._require_triton()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("OverlappedAggregator needs a CUDA device")
        if slots < 1:
            raise ValueError(f"slots must be at least 1, got {slots}")
        started = time.perf_counter()
        self._torch = torch
        self.stats = PipelineStats()
        columns = _columns(trades)
        self.offsets = np.ascontiguousarray(offsets)
        _validate_host(columns, self.offsets)
        self.plans = plan_batches(self.offsets, batch_records)
        capacity_records = max(plan.n_records for plan in self.plans)
        capacity_buckets = max(plan.n_buckets for plan in self.plans)
        capacity_chunks = max(plan.n_chunks for plan in self.plans)
        self._host_columns = {name: torch.from_numpy(column) for name, column in columns.items()}
        self._ring: list[_Slot] = []
        if capacity_records:
            self._ring = [_Slot(torch, capacity_records, capacity_buckets, capacity_chunks, trace)
                          for _ in range(min(slots, len(self.plans)))]
        self._streams = tuple(torch.cuda.Stream() for _ in range(3))  # type: ignore[no-untyped-call]
        stats = self.stats
        stats.n_batches, stats.slots = len(self.plans), len(self._ring)
        stats.capacity_records, stats.capacity_buckets = capacity_records, capacity_buckets
        stats.pinned_bytes = sum(slot.pinned_bytes for slot in self._ring)
        stats.device_bytes = sum(slot.device_bytes for slot in self._ring)
        torch.cuda.synchronize()  # Buffers allocated on the default stream are ready.
        stats.setup_seconds = time.perf_counter() - started

    def run(self) -> dict[str, np.ndarray]:
        """Aggregate every batch and return fresh host feature arrays.

        Returns:
            A dict keyed by ``contract.FEATURE_NAMES`` of NumPy arrays of
            ``n_buckets`` entries with the contract dtypes.
        """
        n_buckets = self.offsets.size - 1
        results = {name: np.zeros(n_buckets, dtype=dtype) for name, dtype in OUTPUT_DTYPES.items()}
        if not self._ring:
            return results  # Every bucket is empty: the zero arrays are the answer.
        self.stats.timeline = []
        started = time.perf_counter()
        ring, plans = self._ring, self.plans
        copy_in, compute, copy_out = self._streams
        for index, plan in enumerate(plans):
            slot = ring[index % len(ring)]
            _drain(slot, results, self.stats)
            _stage(slot, plan, self._host_columns, self.offsets)
            _launch(self._torch, slot, plan, copy_in, compute, copy_out)
        for slot in ring:
            _drain(slot, results, self.stats)
        self.stats.timeline.sort(key=lambda entry: entry["first_bucket"])
        self.stats.run_seconds = time.perf_counter() - started
        return results


def aggregate_host(
    trades: Any,
    offsets: np.ndarray,
    *,
    batch_records: int = 1 << 20,
    slots: int = 2,
    trace: bool = False,
    stats: PipelineStats | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate host trades on the GPU in overlapped batches and return host features.

    One-shot form of ``OverlappedAggregator``: setup and one run. Arguments
    and errors are the class's; ``stats``, when given, receives its stats.

    Returns:
        A dict keyed by ``contract.FEATURE_NAMES`` of NumPy arrays of
        ``n_buckets`` entries with the contract dtypes.
    """
    aggregator = OverlappedAggregator(
        trades, offsets, batch_records=batch_records, slots=slots, trace=trace
    )
    results = aggregator.run()
    if stats is not None:
        vars(stats).update(vars(aggregator.stats))
    return results


def _drain(slot: _Slot, results: dict[str, np.ndarray], stats: PipelineStats) -> None:
    """Wait for the slot's last copy back, move its results out and free the slot."""
    plan = slot.batch
    if plan is None:
        return
    slot.copied_out.synchronize()
    n = plan.n_buckets
    for name, host in slot.host_out.items():
        results[name][plan.first:plan.last] = host[:n].numpy()
    if slot.trace:
        marks = slot.marks
        stats.timeline.append({"first_bucket": plan.first, "marks": marks})
    slot.batch = None


def _stage(slot: _Slot, plan: BatchPlan, host: dict[str, Any], offsets: np.ndarray) -> None:
    """Copy one batch's columns, local offsets and chunk schedule into the slot's pinned buffers.

    The slot is free: ``_drain`` waited for its last copy back, which the
    previous copy in on this slot precedes.
    """
    records = slice(plan.record_start, plan.record_end)
    for name, buffer in slot.host_in.items():
        buffer[: plan.n_records].copy_(host[name][records])
    local = offsets[plan.first: plan.last + 1] - plan.record_start
    meta = slot.host_meta.numpy()
    meta[: local.size] = local
    if plan.two_pass:
        schedule = chunk_schedule(local, kernels.CHUNK_SIZE)
        position = local.size
        for name in ("chunk_starts", "chunk_ends", "chunk_offsets"):
            values = schedule[name]
            meta[position: position + values.size] = values
            position += values.size


def _launch(torch: Any, slot: _Slot, plan: BatchPlan, copy_in: Any, compute: Any,
            copy_out: Any) -> None:
    """Enqueue copy in, compute and copy back of one batch on the three streams."""
    n_records, n_buckets = plan.n_records, plan.n_buckets
    meta_used = _meta_length(n_buckets, plan.n_chunks if plan.two_pass else -1)
    marks = []
    with torch.cuda.stream(copy_in):
        copy_in.wait_event(slot.computed)  # Device inputs of this slot are free.
        marks.append(slot.mark(torch))
        for name, device in slot.dev_in.items():
            device[:n_records].copy_(slot.host_in[name][:n_records], non_blocking=True)
        slot.dev_meta[:meta_used].copy_(slot.host_meta[:meta_used], non_blocking=True)
        marks.append(slot.mark(torch))
        slot.copied_in.record()
    with torch.cuda.stream(compute):
        compute.wait_event(slot.copied_in)
        compute.wait_event(slot.copied_out)  # Device outputs of this slot are free.
        marks.append(slot.mark(torch))
        trades = {name: device[:n_records] for name, device in slot.dev_in.items()}
        local_offsets = slot.dev_meta[: n_buckets + 1]
        outputs = {name: device[:n_buckets] for name, device in slot.dev_out.items()}
        if plan.two_pass:
            base, n = n_buckets + 1, plan.n_chunks
            chunks = kernels.ChunkMetadata(
                chunk_starts=slot.dev_meta[base: base + n],
                chunk_ends=slot.dev_meta[base + n: base + 2 * n],
                chunk_offsets=slot.dev_meta[base + 2 * n: base + 2 * n + n_buckets + 1],
                n_chunks=n,
                chunk_size=kernels.CHUNK_SIZE,
            )
            kernels.launch_two_pass_unchecked(trades, local_offsets, outputs, chunks,
                                              slot.partials)
        elif plan.max_bucket_len > 0:
            kernels.launch_one_pass_unchecked(
                trades, local_offsets, outputs, kernels.select_block_size(plan.max_bucket_len)
            )
        else:
            for tensor in outputs.values():
                tensor.zero_()
        marks.append(slot.mark(torch))
        slot.computed.record()
    with torch.cuda.stream(copy_out):
        copy_out.wait_event(slot.computed)
        marks.append(slot.mark(torch))
        for name, device in slot.dev_out.items():
            slot.host_out[name][:n_buckets].copy_(device[:n_buckets], non_blocking=True)
        marks.append(slot.mark(torch))
        slot.copied_out.record()
    slot.batch = plan
    slot.marks = marks


def timeline_intervals(stats: PipelineStats) -> list[dict[str, float]]:
    """Convert traced events into per-batch stage intervals in ms from the first event.

    Call after ``aggregate_host(..., trace=True)`` returned (every event has
    completed by then).
    """
    entries = [entry for entry in stats.timeline if entry.get("marks")]
    if not entries:
        return []
    origin = entries[0]["marks"][0]
    rows = []
    for entry in entries:
        times = [origin.elapsed_time(mark) for mark in entry["marks"]]
        rows.append({
            "first_bucket": float(entry["first_bucket"]),
            "copy_in_start": times[0], "copy_in_end": times[1],
            "compute_start": times[2], "compute_end": times[3],
            "copy_out_start": times[4], "copy_out_end": times[5],
        })
    return rows


def overlap_ms(rows: list[dict[str, float]], a: str, b: str) -> float:
    """Total time in ms during which some ``a`` interval and some ``b`` interval both run.

    ``a`` and ``b`` are stage names (``copy_in``, ``compute``, ``copy_out``).
    Intervals of one stage do not overlap each other (one stream each), so
    the pairwise intersections are disjoint and can be summed.
    """
    total = 0.0
    for left in rows:
        for right in rows:
            start = max(left[f"{a}_start"], right[f"{b}_start"])
            end = min(left[f"{a}_end"], right[f"{b}_end"])
            total += max(0.0, end - start)
    return total
