"""``tapeforge.prepared``: validate once, launch many times.

The prepared path must produce exactly what the checked public API
produces, on both kernel paths, while reusing its buffers. The borrowed /
owned distinction and the pass-level entry points are pinned here because
the benchmark and any future training loop rely on them.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_gpu_parity, cuda_inputs

from tapeforge import contract, generate, kernels, ops, prepared, windows

W = contract.DEFAULT_BUCKET_WIDTH_US
LIMIT = kernels.MAX_ONE_PASS_BUCKET
CHUNK = kernels.CHUNK_SIZE


def _workload(profile: str, seed: int, n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    trades = generate.generate_trades(seed=seed, n_buckets=n_buckets, profile=profile)
    return trades, windows.build_offsets(trades["timestamp"], 0, n_buckets, W)


def _one_bucket(n: int) -> tuple[np.ndarray, np.ndarray]:
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = 100 * contract.PRICE_SCALE + np.arange(n) % 97
    trades["quantity"] = 1 + np.arange(n) % 1009
    trades["buyer_is_maker"] = np.arange(n) % 3 == 0
    return trades, np.array([0, n], dtype=np.int64)


# --- without a GPU -----------------------------------------------------------


@pytest.mark.skipif(kernels.TRITON_AVAILABLE, reason="triton is installed on this machine")
def test_prepare_without_triton_raises_a_clear_error() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        prepared.prepare({}, torch.zeros(2, dtype=torch.int64))


def test_prepared_aggregation_is_frozen() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(prepared.PreparedAggregation)
    assert prepared.PreparedAggregation.__dataclass_params__.frozen  # type: ignore[attr-defined]


# --- on CUDA -----------------------------------------------------------------

gpu = pytest.mark.gpu


@gpu
@pytest.mark.parametrize(
    ("profile", "n_buckets", "path"),
    [("uniform", 1000, "one_pass"), ("sparse", 1000, "one_pass"), ("burst", 1000, "two_pass")],
)
def test_prepared_matches_public_api_and_oracle(
    gpu_ready: None, profile: str, n_buckets: int, path: str
) -> None:
    import torch

    trades, offsets = _workload(profile, 1, n_buckets)
    columns, dev_offsets = cuda_inputs(trades, offsets)
    assert kernels.plan_dispatch(int(np.diff(offsets).max())).path == path
    ready = prepared.prepare(columns, dev_offsets)
    assert ready.path == path
    got = ready.execute()
    assert_gpu_parity(got, trades, offsets)
    public = ops.aggregate(columns, dev_offsets)
    for name in contract.FEATURE_NAMES:
        assert torch.equal(got[name], public[name]), name


@gpu
def test_execute_returns_the_same_borrowed_tensors_every_call(gpu_ready: None) -> None:
    trades, offsets = _one_bucket(3 * LIMIT + 17)
    ready = prepared.prepare(*cuda_inputs(trades, offsets))
    first = ready.execute()
    second = ready.execute()
    for name in contract.FEATURE_NAMES:
        assert first[name] is second[name], f"{name}: execute must reuse its output buffer"
        assert first[name] is ready.features[name]
    assert_gpu_parity(second, trades, offsets)


@gpu
def test_execute_owned_returns_independent_copies(gpu_ready: None) -> None:
    import torch

    trades, offsets = _one_bucket(LIMIT + 1)
    ready = prepared.prepare(*cuda_inputs(trades, offsets))
    owned = ready.execute_owned()
    for name in contract.FEATURE_NAMES:
        assert owned[name] is not ready.features[name]
        assert torch.equal(owned[name], ready.features[name])
    # Overwriting the borrowed buffer must leave the owned copy untouched.
    ready.features["event_count"].fill_(-1)
    assert owned["event_count"].item() == LIMIT + 1
    assert_gpu_parity(owned, trades, offsets)


@gpu
def test_forced_two_pass_on_a_short_bucket_matches_the_oracle(gpu_ready: None) -> None:
    """Both paths can be run on the same input; results are identical by contract."""
    import torch

    trades, offsets = _workload("uniform", 2, 500)
    assert int(np.diff(offsets).max()) <= LIMIT
    columns, dev_offsets = cuda_inputs(trades, offsets)
    forced = prepared.prepare(columns, dev_offsets, path="two_pass")
    assert forced.path == "two_pass"
    assert forced.n_chunks == int(sum(-(-n // CHUNK) for n in np.diff(offsets)))
    auto = prepared.prepare(columns, dev_offsets)
    assert auto.path == "one_pass" and auto.n_chunks == 0
    got_forced, got_auto = forced.execute(), auto.execute()
    assert_gpu_parity(got_forced, trades, offsets)
    for name in contract.FEATURE_NAMES:
        assert torch.equal(got_forced[name], got_auto[name]), name


@gpu
def test_forced_one_pass_keeps_the_explicit_capacity_check(gpu_ready: None) -> None:
    columns, dev_offsets = cuda_inputs(*_one_bucket(LIMIT + 1))
    with pytest.raises(kernels.BucketTooLargeError, match="at most 4096"):
        prepared.prepare(columns, dev_offsets, path="one_pass")
    with pytest.raises(kernels.BucketTooLargeError, match="at most 1024"):
        prepared.prepare(columns, dev_offsets, block_size=1024)


@gpu
def test_block_size_with_forced_two_pass_is_rejected(gpu_ready: None) -> None:
    columns, dev_offsets = cuda_inputs(*_one_bucket(10))
    with pytest.raises(ValueError, match="one-pass path only"):
        prepared.prepare(columns, dev_offsets, block_size=16, path="two_pass")


@gpu
def test_pass_1_then_pass_2_reproduces_execute(gpu_ready: None) -> None:
    import torch

    trades, offsets = _workload("burst", 3, 1000)
    ready = prepared.prepare(*cuda_inputs(trades, offsets))
    assert ready.path == "two_pass"
    expected = ready.execute_owned()
    for tensor in ready.features.values():
        tensor.zero_()
    ready.run_pass_1()
    ready.run_pass_2()
    for name in contract.FEATURE_NAMES:
        assert torch.equal(ready.features[name], expected[name]), name
    # Pass 2 alone is idempotent on unchanged partials.
    ready.run_pass_2()
    assert_gpu_parity(ready.features, trades, offsets)


@gpu
def test_pass_entry_points_are_two_pass_only(gpu_ready: None) -> None:
    ready = prepared.prepare(*cuda_inputs(*_one_bucket(100)))
    assert ready.path == "one_pass"
    with pytest.raises(ValueError, match="two-pass path only"):
        ready.run_pass_1()
    with pytest.raises(ValueError, match="two-pass path only"):
        ready.run_pass_2()


@gpu
def test_all_empty_buckets_execute_to_zeros(gpu_ready: None) -> None:
    trades = np.zeros(2, dtype=contract.TRADE_DTYPE)
    trades["timestamp"] = [0, 50 * W]
    trades["quantity"] = 1
    offsets = windows.build_offsets(trades["timestamp"], 10 * W, 5, W)
    assert offsets.tolist() == [1] * 6
    ready = prepared.prepare(*cuda_inputs(trades, offsets))
    got = ready.execute()
    assert not got["valid"].any()
    assert_gpu_parity(got, trades, offsets)


@gpu
def test_prepare_rejects_cpu_tensors(gpu_ready: None) -> None:
    import torch

    trades, offsets = _one_bucket(10)
    columns = {name: torch.from_numpy(trades[name].copy()) for name in trades.dtype.names}
    with pytest.raises(ValueError, match="needs CUDA tensors"):
        prepared.prepare(columns, torch.from_numpy(offsets))


@gpu
def test_prepare_validates_like_the_public_api(gpu_ready: None) -> None:
    import torch

    trades, offsets = _one_bucket(4)
    columns, dev_offsets = cuda_inputs(trades, offsets)
    with pytest.raises(ValueError, match="but there are 4 records"):
        prepared.prepare(columns, torch.tensor([0, 50], dtype=torch.int64, device="cuda"))
    strided = {k: v[::2] for k, v in cuda_inputs(*_one_bucket(8))[0].items()}
    with pytest.raises(ValueError, match="contiguous"):
        prepared.prepare(strided, dev_offsets)
