"""The GPU engine must agree with the CPU reference on both kernel paths.

Every test drives the real code path -- ``ops.aggregate`` on CUDA tensors or
``kernels.launch_aggregate`` -- and compares against
``reference.aggregate_reference_naive``, the plain-loop oracle, so the GPU is
never checked against code that shares its vectorization tricks.

Skips only for genuine hardware/software absence (no torch, no CUDA device,
no Triton), via the ``gpu_ready`` fixture. Nothing is skipped or expected to
fail for missing functionality: buckets beyond the one-pass tile dispatch to
the two-pass path and are held to the same parity rule. Each test that
depends on a particular path asserts the path through ``kernels.plan_dispatch``
on the fixture's longest bucket, so a large fixture cannot silently take the
other path. The focused two-pass suite is test_two_pass.py.

Parity rule: counts and quantity sums exactly; VWAP within
contract.VWAP_RTOL / VWAP_ATOL; identical dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_gpu_parity, cuda_inputs

from tapeforge import contract, generate, kernels, ops, windows

pytestmark = pytest.mark.gpu

W = contract.DEFAULT_BUCKET_WIDTH_US
P = contract.PRICE_SCALE
Q = contract.QUANTITY_SCALE
LIMIT = kernels.MAX_ONE_PASS_BUCKET

# Helpers shared with test_two_pass.py live in conftest.py.
_cuda = cuda_inputs
_assert_parity = assert_gpu_parity


def _workload(profile: str, seed: int, n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile=profile)
    return trades, windows.build_offsets(trades["timestamp"], 0, n_buckets, W)


def _rows(rows: list[tuple[int, float, float, bool]]) -> np.ndarray:
    out = np.zeros(len(rows), dtype=contract.TRADE_DTYPE)
    for i, (second, price, qty, bim) in enumerate(rows):
        out[i] = (second * contract.US_PER_SECOND, round(price * P), round(qty * Q), bim, i)
    return out


def _one_bucket(n: int) -> tuple[np.ndarray, np.ndarray]:
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 100 * P + np.arange(n) % 97
    trades["quantity"] = 1 + np.arange(n) % 1009
    trades["buyer_is_maker"] = np.arange(n) % 3 == 0
    return trades, np.array([0, n], dtype=np.int64)


# --- parity across profiles and seeds -------------------------------------


@pytest.mark.parametrize("profile", ["sparse", "uniform"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu_reference(gpu_ready: None, profile: str, seed: int) -> None:
    trades, offsets = _workload(profile, seed, 1000)
    _assert_parity(ops.aggregate(*_cuda(trades, offsets)), trades, offsets)


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_gpu_matches_cpu_reference_on_burst_within_one_pass_limit(
    gpu_ready: None, seed: int
) -> None:
    trades, offsets = _workload("burst", seed, 200)
    longest = int(np.diff(offsets).max())
    assert 1024 < longest <= LIMIT, f"workload no longer exercises large tiles ({longest})"
    _assert_parity(ops.aggregate(*_cuda(trades, offsets)), trades, offsets)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu_reference_on_full_burst(gpu_ready: None, seed: int) -> None:
    """Burst workloads whose busiest bucket exceeds the one-pass tile: two-pass parity.

    The same workloads the automatic dispatch used to reject; now an
    independent CPU/GPU parity check through the public API, with the path
    asserted so the fixture is known to exercise the two-pass kernels.
    """
    trades, offsets = _workload("burst", seed, 1000)
    longest = int(np.diff(offsets).max())
    assert longest > LIMIT, f"workload no longer has an oversized bucket ({longest})"
    assert kernels.plan_dispatch(longest).path == "two_pass"
    _assert_parity(ops.aggregate(*_cuda(trades, offsets)), trades, offsets)


def test_oversized_bucket_takes_the_two_pass_path(gpu_ready: None) -> None:
    """One bucket several tiles long, with a ragged final tile."""
    trades, offsets = _one_bucket(3 * LIMIT + 17)
    assert kernels.plan_dispatch(3 * LIMIT + 17).path == "two_pass"
    _assert_parity(ops.aggregate(*_cuda(trades, offsets)), trades, offsets)


# --- one-pass limit boundaries --------------------------------------------


def test_bucket_of_exactly_the_one_pass_limit_is_exact(gpu_ready: None) -> None:
    trades, offsets = _one_bucket(LIMIT)
    assert kernels.plan_dispatch(LIMIT).path == "one_pass"
    _assert_parity(ops.aggregate(*_cuda(trades, offsets)), trades, offsets)


def test_bucket_one_past_the_limit_is_not_truncated(gpu_ready: None) -> None:
    """LIMIT + 1 records: the first length that cannot fit a one-pass tile.

    Dispatches to the two-pass path (LIMIT // CHUNK_SIZE full chunks plus a
    one-record ragged chunk) and must match the oracle record for record; a
    truncating or re-tiling implementation would drop the extra record.
    """
    trades, offsets = _one_bucket(LIMIT + 1)
    assert kernels.plan_dispatch(LIMIT + 1).path == "two_pass"
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["event_count"].item() == LIMIT + 1
    _assert_parity(got, trades, offsets)


def test_explicit_block_size_boundary(gpu_ready: None) -> None:
    trades, offsets = _one_bucket(1024)
    _assert_parity(kernels.launch_aggregate(*_cuda(trades, offsets), block_size=1024),
                   trades, offsets)
    trades, offsets = _one_bucket(1025)
    with pytest.raises(kernels.BucketTooLargeError, match="at most 1024"):
        kernels.launch_aggregate(*_cuda(trades, offsets), block_size=1024)


# --- edge cases -----------------------------------------------------------


def test_toy_example(gpu_ready: None) -> None:
    trades = _rows([(1, 100.0, 2.0, False), (1, 102.0, 1.0, True)])
    offsets = np.array([0, 2, 2], dtype=np.int64)
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["event_count"].tolist() == [2, 0]
    assert got["total_quantity"].tolist() == [3 * Q, 0]
    assert got["signed_quantity"].tolist() == [1 * Q, 0]
    assert got["valid"].tolist() == [True, False]
    assert got["vwap"].tolist() == pytest.approx([(100 * 2 + 102) / 3, 0.0],
                                                 rel=contract.VWAP_RTOL)


def test_all_empty_buckets_do_not_crash(gpu_ready: None) -> None:
    """Records exist but all fall outside the window: the kernel runs on empty slices."""
    trades = _rows([(0, 100.0, 1.0, False), (50, 100.0, 1.0, True)])
    offsets = windows.build_offsets(trades["timestamp"], 10 * contract.US_PER_SECOND, 5, W)
    assert offsets.tolist() == [1] * 6
    got = ops.aggregate(*_cuda(trades, offsets))
    assert not got["valid"].any()
    _assert_parity(got, trades, offsets)


def test_no_records_at_all(gpu_ready: None) -> None:
    empty = np.zeros(0, dtype=contract.TRADE_DTYPE)
    offsets = np.zeros(4, dtype=np.int64)
    _assert_parity(ops.aggregate(*_cuda(empty, offsets)), empty, offsets)


def test_zero_quantity_rows(gpu_ready: None) -> None:
    trades = _rows([(0, 100.0, 0.0, False), (0, 105.0, 0.0, True), (1, 101.0, 2.0, True),
                    (1, 900.0, 0.0, False)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 2, W)
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["vwap"].tolist() == pytest.approx([0.0, 101.0], rel=contract.VWAP_RTOL)
    _assert_parity(got, trades, offsets)


def test_sign_convention(gpu_ready: None) -> None:
    trades = _rows([(0, 100.0, 1.0, True), (0, 100.0, 2.0, True), (1, 100.0, 1.5, False),
                    (2, 100.0, 4.0, False), (2, 100.0, 1.25, True)])
    offsets = windows.build_offsets(trades["timestamp"], 0, 3, W)
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["signed_quantity"].tolist() == [-3 * Q, round(1.5 * Q), round(2.75 * Q)]
    _assert_parity(got, trades, offsets)


def test_window_boundaries_exclude_records_before_and_after(gpu_ready: None) -> None:
    trades = np.zeros(5, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = [0, W, 2 * W - 1, 2 * W, 3 * W]
    trades["price"] = [999 * P, 100 * P, 200 * P, 300 * P, 999 * P]
    trades["quantity"] = Q
    offsets = windows.build_offsets(trades["timestamp"], W, 2, W)
    assert offsets.tolist() == [1, 3, 4]
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["event_count"].tolist() == [2, 1]
    _assert_parity(got, trades, offsets)


def test_sums_just_inside_the_int64_bound_are_exact(gpu_ready: None) -> None:
    n = 64
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 2**30
    trades["quantity"] = contract.INT64_MAX // (n * 2**30)
    trades["buyer_is_maker"] = np.arange(n) % 2 == 0
    offsets = np.array([0, n], dtype=np.int64)
    got = ops.aggregate(*_cuda(trades, offsets))
    assert got["total_quantity"].item() == int(trades["quantity"].sum())
    _assert_parity(got, trades, offsets)


# --- validation on CUDA ---------------------------------------------------


def test_offsets_past_the_records_are_rejected_before_launch(gpu_ready: None) -> None:
    trades = _rows([(0, 100.0, 1.0, False)])
    columns, _ = _cuda(trades, np.array([0, 1], dtype=np.int64))
    import torch

    with pytest.raises(ValueError, match="but there are 1 records"):
        ops.aggregate(columns, torch.tensor([0, 50], dtype=torch.int64, device="cuda"))


def test_strided_cuda_column_is_rejected(gpu_ready: None) -> None:
    trades = _rows([(0, 100.0, 1.0, False)] * 4)
    columns, offsets = _cuda(trades, np.array([0, 2], dtype=np.int64))
    columns = {k: v[::2] for k, v in columns.items()}
    with pytest.raises(ValueError, match="contiguous"):
        ops.aggregate(columns, offsets)


def test_mixed_cpu_and_cuda_inputs_are_rejected(gpu_ready: None) -> None:
    trades = _rows([(0, 100.0, 1.0, False)])
    columns, offsets = _cuda(trades, np.array([0, 1], dtype=np.int64))
    columns["price"] = columns["price"].cpu()
    with pytest.raises(ValueError, match="all inputs must share one device"):
        ops.aggregate(columns, offsets)
