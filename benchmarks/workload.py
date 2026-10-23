"""Benchmark workloads and the correctness gate every timed implementation passes.

A workload is a seeded synthetic trade array plus its offsets, sized by
record count. Nothing is timed until its output has matched the CPU
reference (exact integers, VWAP within the contract tolerance), and the
outcome of that check is recorded alongside the timing.

``WorkloadInfo`` records the bucket-length distribution, because that
distribution, not the record count, decides which kernel path runs and how
much of each one-pass tile is masked: the one-pass tile follows the longest
bucket, and the two-pass chunk count follows the sum of ceil(len / CHUNK).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tapeforge import contract, generate, kernels, reference, windows

# Approximate generator means, used only to pick a bucket count that lands
# near the requested record count. The actual count is always recorded.
RECORDS_PER_BUCKET = {"sparse": 0.5, "uniform": 50.0, "burst": 100.0}


@dataclass(frozen=True)
class WorkloadInfo:
    """Everything a result record needs to name the workload and its dispatch."""

    profile: str
    seed: int
    requested_records: int
    n_records: int
    n_buckets: int
    bucket_width_us: int
    max_bucket_len: int
    buckets_over_one_pass_limit: int
    # Bucket-length distribution and the dispatch it implies.
    n_empty_buckets: int
    bucket_len_p50: int
    bucket_len_p90: int
    bucket_len_p99: int
    dispatch_path: str  # kernels.plan_dispatch on the longest bucket
    one_pass_block_size: int | None  # tile the one-pass path would use, if it applies
    two_pass_chunk_size: int
    two_pass_n_chunks: int  # pass-1 programs the two-pass path would launch

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for the ``config.workload`` field of a result record."""
        return asdict(self)


def describe_offsets(offsets: np.ndarray) -> dict[str, Any]:
    """Distribution and dispatch facts for a given offsets array."""
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
    """Seeded trades, their offsets and the workload description for about ``n_records``.

    The bucket count is chosen from the profile's mean rate, so the actual
    record count differs from the request and is recorded in ``info``.

    Raises:
        KeyError: unknown profile.
        ValueError: whatever ``generate_trades`` rejects.
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
    """Contiguous CPU tensors for the three kernel inputs, optionally pinned."""
    import torch

    columns = {
        name: torch.from_numpy(np.ascontiguousarray(trades[name]))
        for name in ("price", "quantity", "buyer_is_maker")
    }
    return {k: v.pin_memory() for k, v in columns.items()} if pin else columns


def check_features(got: dict[str, Any], expected: reference.Features) -> None:
    """Raise AssertionError unless ``got`` matches the reference per the contract."""
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
    """The reference ``Features`` as the dict shape ``ops.aggregate`` returns."""
    return {name: getattr(features, name) for name in contract.FEATURE_NAMES}
