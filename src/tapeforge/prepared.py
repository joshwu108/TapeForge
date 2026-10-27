"""Prepared CUDA aggregation: validate and plan once, launch many times.

``ops.aggregate`` re-validates its input, rebuilds the chunk schedule and
allocates fresh outputs on every call. That is correct for a public API and
the wrong cost model for a training loop or a benchmark that aggregates the
same resident tensors thousands of times. This module separates the two:

    prepared = prepare(columns, offsets)      # validate, plan, allocate: once
    features = prepared.execute()             # kernel launches only
    features = prepared.execute()             # again, same buffers

Invariants:
    Inputs are borrowed, not copied. A ``PreparedAggregation`` holds
    references to the column and offset tensors it was prepared with, so
    they stay allocated for its lifetime; the caller must not resize,
    re-layout or mutate them. The chunk schedule and the output allocation
    derive from the offsets at ``prepare`` time, and the overflow bound was
    checked against the prices and quantities at ``prepare`` time. Changed
    data silently makes the prepared state stale (wrong schedule, or sums
    that can wrap); the caller must prepare again. Nothing is cached and
    nothing is keyed on tensor identity: every ``prepare`` validates from
    scratch.

    Outputs of ``execute`` are borrowed too: the same five tensors are
    returned on every call and overwritten by the next ``execute``. Callers
    that keep a result across calls use ``execute_owned``, which clones them.

Device and streams:
    All launches go to the current CUDA stream of the inputs' device at
    ``execute`` time, like any PyTorch op. Results are ready on that stream;
    a caller consuming them on another stream must synchronize.

Concurrency:
    One prepared object owns one set of output and scratch buffers, so
    concurrent ``execute`` calls from several threads or streams race on
    them. Prepare one object per concurrent user.

Synchronization:
    ``prepare`` performs the launcher's host syncs (validation statistics,
    and the chunk count on the two-pass path). ``execute`` performs none.

Path forcing:
    ``path="two_pass"`` lets the benchmark measure both paths on one
    workload; results are identical by contract. ``block_size`` forces the
    one-pass path with an explicit capacity check, exactly as
    ``kernels.launch_aggregate`` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tapeforge import kernels
from tapeforge.kernels import ChunkMetadata, DispatchPlan, PathChoice

if TYPE_CHECKING:
    import torch

    from tapeforge.validation import InputSummary

__all__ = ["PreparedAggregation", "prepare"]

COLUMNS: tuple[str, ...] = ("price", "quantity", "buyer_is_maker")  # Columns the kernels read.


@dataclass(frozen=True, eq=False)
class PreparedAggregation:
    """Validated inputs, dispatch plan and preallocated buffers for one workload.

    Instances are produced by ``prepare``. The dataclass is frozen: no field
    is rebound after construction, and ``execute`` writes into the tensors the
    fields reference. ``eq=False`` gives identity semantics, since tensor
    fields have no meaningful value equality or hash.

    Attributes:
        trades: The borrowed kernel columns (``price``, ``quantity``,
            ``buyer_is_maker``); extra input columns are not retained.
        offsets: The borrowed bucket boundaries, length ``n_buckets + 1``.
        summary: Validation facts from ``validation.validate_columns``.
        plan: The chosen path and tile or chunk size.
        features: The five output tensors, each of shape ``(n_buckets,)``.
            Owned by this object and overwritten by every ``execute``.
        chunks: Pass-1 schedule on the two-pass path; ``None`` on one-pass.
        partials: Pass-1 scratch on the two-pass path; ``None`` on one-pass.
    """

    trades: Mapping[str, torch.Tensor]
    offsets: torch.Tensor
    summary: InputSummary
    plan: DispatchPlan
    features: dict[str, torch.Tensor]
    chunks: ChunkMetadata | None
    partials: dict[str, torch.Tensor] | None

    @property
    def path(self) -> str:
        """Return ``"one_pass"`` or ``"two_pass"``."""
        return self.plan.path

    @property
    def n_chunks(self) -> int:
        """Return the number of pass-1 programs on the two-pass path; 0 on one-pass."""
        return self.chunks.n_chunks if self.chunks is not None else 0

    def execute(self) -> dict[str, torch.Tensor]:
        """Launch the planned kernels and return the borrowed output tensors.

        The returned dict holds the same tensor objects on every call; the
        next ``execute`` overwrites them. Performs no validation and no host
        sync.
        """
        if self.summary.max_bucket_len == 0:
            # Every bucket is empty, so the zero-filled outputs are already the
            # answer, and nothing has written to them since allocation.
            return self.features
        if self.plan.path == "one_pass":
            assert self.plan.block_size is not None
            kernels.launch_one_pass_unchecked(
                self.trades, self.offsets, self.features, self.plan.block_size
            )
        else:
            assert self.chunks is not None and self.partials is not None
            kernels.launch_two_pass_unchecked(
                self.trades, self.offsets, self.features, self.chunks, self.partials
            )
        return self.features

    def execute_owned(self) -> dict[str, torch.Tensor]:
        """Run ``execute`` and return independent copies that survive the next call."""
        return {name: tensor.clone() for name, tensor in self.execute().items()}

    def run_pass_1(self) -> None:
        """Write the int64 partials of the two-pass path, for instrumentation.

        Raises:
            ValueError: The object was prepared for the one-pass path.
        """
        if self.chunks is None or self.partials is None:
            raise ValueError("run_pass_1 applies to the two-pass path only")
        kernels.launch_pass_1_unchecked(self.trades, self.chunks, self.partials)

    def run_pass_2(self) -> None:
        """Combine the partials written by the last ``run_pass_1``, for instrumentation.

        Idempotent given unchanged partials, so it can be timed in isolation.
        Calling it before any pass 1 reads uninitialised scratch.

        Raises:
            ValueError: The object was prepared for the one-pass path.
        """
        if self.chunks is None or self.partials is None:
            raise ValueError("run_pass_2 applies to the two-pass path only")
        kernels.launch_pass_2_unchecked(self.offsets, self.chunks, self.partials, self.features)


def prepare(
    trades: Mapping[str, Any],
    offsets: Any,
    *,
    block_size: int | None = None,
    path: PathChoice = "auto",
) -> PreparedAggregation:
    """Validate once and preallocate everything ``execute`` needs.

    The input contract and errors are those of ``kernels.launch_aggregate``
    (``tapeforge.validation.validate_columns``).

    Args:
        trades: Mapping of column name to CUDA tensor.
        offsets: CUDA int64 bucket boundaries of length ``n_buckets + 1``.
        block_size: Explicit one-pass tile; implies ``path="one_pass"``.
        path: ``"auto"``, ``"one_pass"`` or ``"two_pass"``.

    Raises:
        ValueError: ``path`` is invalid, ``block_size`` is combined with
            ``path="two_pass"``, or the tensors are not on CUDA.
        BucketTooLargeError: The one-pass path was forced for a bucket longer
            than its tile.
    """
    kernels._require_triton()
    import torch

    from tapeforge.validation import validate_columns

    if not isinstance(offsets, torch.Tensor):
        raise TypeError(f"offsets must be a torch.Tensor, got {type(offsets).__name__}")
    if offsets.device.type != "cuda":
        raise ValueError("prepare needs CUDA tensors; use ops.aggregate for CPU inputs")
    summary = validate_columns(trades, offsets)
    plan = kernels.plan_dispatch(summary.max_bucket_len, block_size, path)

    # Hold only the columns the kernels read, so extra columns are not kept alive.
    held = {name: trades[name] for name in COLUMNS}
    with torch.cuda.device(offsets.device):
        features = kernels.empty_features(summary.n_buckets, offsets.device)
        chunks = partials = None
        if plan.path == "two_pass":
            assert plan.chunk_size is not None
            chunks = kernels.build_chunk_metadata(offsets, plan.chunk_size)
            partials = kernels.allocate_partials(chunks.n_chunks, offsets.device)
    return PreparedAggregation(
        trades=held,
        offsets=offsets,
        summary=summary,
        plan=plan,
        features=features,
        chunks=chunks,
        partials=partials,
    )
