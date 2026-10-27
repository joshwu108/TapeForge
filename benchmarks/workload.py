"""Benchmark workloads and the correctness gate every timed implementation passes.

A workload is a seeded synthetic trade array plus its offsets, sized by
record count. Nothing is timed until its output has matched the CPU
reference (exact integers, VWAP within the contract tolerance), and the
outcome of that check is recorded alongside the timing.

``WorkloadInfo`` records the bucket-length distribution because that
distribution, not the record count, decides which kernel path runs and how
much of each one-pass tile is masked: the one-pass tile follows the longest
bucket, and the two-pass chunk count is the sum over buckets of
``ceil(len / CHUNK_SIZE)``.

Not covered:
    Timing and result recording; those live in ``harness.py``. Row
    definitions live in ``kernel.py`` and ``pipeline.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tapeforge import contract, generate, kernels, reference, windows

# Approximate records per bucket for each generator profile, used only to pick
# a bucket count that lands near the requested record count. The actual record
# count is always recorded.
RECORDS_PER_BUCKET = {"sparse": 0.5, "uniform": 50.0, "burst": 100.0}


@dataclass(frozen=True)
class WorkloadInfo:
    """Description of one benchmark workload and the kernel dispatch it implies.

    Attributes:
        profile: Generator profile name.
        seed: Generator seed.
        requested_records: Record count asked for by the sweep.
        n_records: Record count actually generated.
        n_buckets: Number of buckets; ``len(offsets) - 1``.
        bucket_width_us: Bucket width in microseconds.
        max_bucket_len: Records in the longest bucket.
        buckets_over_one_pass_limit: Buckets longer than ``kernels.MAX_ONE_PASS_BUCKET``.
        n_empty_buckets: Buckets holding no records.
        bucket_len_p50: 50th percentile of bucket length, truncated to int.
        bucket_len_p90: 90th percentile of bucket length, truncated to int.
        bucket_len_p99: 99th percentile of bucket length, truncated to int.
        dispatch_path: ``kernels.plan_dispatch`` result for the longest bucket.
        one_pass_block_size: Tile the one-pass path would use; ``None`` when
            the longest bucket exceeds the one-pass limit.
        two_pass_chunk_size: ``kernels.CHUNK_SIZE``.
        two_pass_n_chunks: Pass-1 programs the two-pass path would launch.
    """

    profile: str
    seed: int
    requested_records: int
    n_records: int
    n_buckets: int
    bucket_width_us: int
    max_bucket_len: int
    buckets_over_one_pass_limit: int
    n_empty_buckets: int
    bucket_len_p50: int
    bucket_len_p90: int
    bucket_len_p99: int
    dispatch_path: str
    one_pass_block_size: int | None
    two_pass_chunk_size: int
    two_pass_n_chunks: int

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict for the ``config.workload`` field of a result record."""
        return asdict(self)


def describe_offsets(offsets: np.ndarray) -> dict[str, Any]:
    """Compute the bucket-length distribution and dispatch facts for ``offsets``.

    Args:
        offsets: Bucket boundaries of length ``n_buckets + 1``.

    Returns:
        The ``WorkloadInfo`` fields from ``max_bucket_len`` onward. Percentiles
        are truncated to ``int``; an empty ``offsets`` yields zeros.
    """
    lengths = np.diff(offsets)
    longest = int(lengths.max()) if lengths.size else 0
    plan = kernels.plan_dispatch(longest)
    chunk = kernels.CHUNK_SIZE
    return {
        "max_bucket_len": longest,
        "buckets_over_one_pass_limit": int((lengths > kernels.MAX_ONE_PASS_BUCKET).sum()),
        "n_empty_buckets": int((lengths == 0).sum()),
        "bucket_len_p50": int(np.percentile(lengths, 50)) if lengths.size else 0,
        "bucket_len_p90": int(np.percentile(lengths, 90)) if lengths.size else 0,
        "bucket_len_p99": int(np.percentile(lengths, 99)) if lengths.size else 0,
        "dispatch_path": plan.path,
        "one_pass_block_size": (
            kernels.select_block_size(longest) if longest <= kernels.MAX_ONE_PASS_BUCKET else None
        ),
        "two_pass_chunk_size": chunk,
        "two_pass_n_chunks": int((-(-lengths // chunk)).sum()),
    }


def make_workload(
    n_records: int, profile: str, seed: int, width_us: int = contract.DEFAULT_BUCKET_WIDTH_US
) -> tuple[np.ndarray, np.ndarray, WorkloadInfo]:
    """Generate seeded trades, their offsets and the workload description for ``n_records``.

    The bucket count is derived from the profile's mean rate, so the actual
    record count differs from the request; both are recorded in the returned
    ``WorkloadInfo``.

    Args:
        n_records: Requested record count; at least one bucket is generated.
        profile: Generator profile; a key of ``RECORDS_PER_BUCKET``.
        seed: Generator seed.
        width_us: Bucket width in microseconds.

    Returns:
        ``(trades, offsets, info)``.

    Raises:
        KeyError: ``profile`` is not a key of ``RECORDS_PER_BUCKET``.
        ValueError: ``generate_trades`` rejected its arguments.
    """
    n_buckets = max(1, round(n_records / RECORDS_PER_BUCKET[profile]))
    trades = generate.generate_trades(
        seed=seed, n_buckets=n_buckets, profile=profile, width_us=width_us  # type: ignore[arg-type]
    )
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, width_us)
    info = WorkloadInfo(
        profile=profile,
        seed=seed,
        requested_records=n_records,
        n_records=int(trades.size),
        n_buckets=n_buckets,
        bucket_width_us=width_us,
        **describe_offsets(offsets),
    )
    return trades, offsets, info


def host_columns(trades: np.ndarray, pin: bool = False) -> dict[str, Any]:
    """Return contiguous CPU tensors for the three kernel inputs, pinned when ``pin`` is set."""
    import torch

    columns = {
        name: torch.from_numpy(np.ascontiguousarray(trades[name]))
        for name in ("price", "quantity", "buyer_is_maker")
    }
    return {k: v.pin_memory() for k, v in columns.items()} if pin else columns


def check_features(got: dict[str, Any], expected: reference.Features) -> None:
    """Assert that ``got`` matches ``expected`` under the contract tolerances.

    Integer features and dtypes must match exactly; ``vwap`` is compared with
    ``contract.VWAP_RTOL`` and ``contract.VWAP_ATOL``. Device tensors are
    moved to the host for the comparison.

    Raises:
        AssertionError: A dtype or value differs from the reference.
    """
    for name in contract.FEATURE_NAMES:
        value = got[name]
        actual = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        target = getattr(expected, name)
        if actual.dtype != target.dtype:
            raise AssertionError(f"{name}: dtype {actual.dtype} != {target.dtype}")
        if name == "vwap":
            np.testing.assert_allclose(
                actual, target, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL, err_msg=name
            )
        else:
            np.testing.assert_array_equal(actual, target, err_msg=name)


def features_dict(features: reference.Features) -> dict[str, np.ndarray]:
    """Return the reference ``Features`` in the dict shape ``ops.aggregate`` returns."""
    return {name: getattr(features, name) for name in contract.FEATURE_NAMES}
