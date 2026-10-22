"""Benchmark workloads and the correctness gate every timed implementation passes.

A workload is a seeded synthetic trade array plus its offsets, sized by
record count. Nothing is timed until its output has matched the CPU
reference (exact integers, VWAP within the contract tolerance), and the
outcome of that check is recorded alongside the timing.
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
    profile: str
    seed: int
    requested_records: int
    n_records: int
    n_buckets: int
    bucket_width_us: int
    max_bucket_len: int
    buckets_over_one_pass_limit: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_workload(
    n_records: int, profile: str, seed: int, width_us: int = contract.DEFAULT_BUCKET_WIDTH_US
) -> tuple[np.ndarray, np.ndarray, WorkloadInfo]:
    n_buckets = max(1, round(n_records / RECORDS_PER_BUCKET[profile]))
    trades = generate.generate_trades(
        seed=seed, n_buckets=n_buckets, profile=profile, width_us=width_us  # type: ignore[arg-type]
    )
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, width_us)
    lengths = np.diff(offsets)
    info = WorkloadInfo(
        profile=profile,
        seed=seed,
        requested_records=n_records,
        n_records=int(trades.size),
        n_buckets=n_buckets,
        bucket_width_us=width_us,
        max_bucket_len=int(lengths.max()),
        buckets_over_one_pass_limit=int((lengths > kernels.MAX_ONE_PASS_BUCKET).sum()),
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
    return {name: getattr(features, name) for name in contract.FEATURE_NAMES}
