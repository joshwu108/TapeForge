"""Tests for ``tapeforge.pipelined``: bounded batches with overlapped copies.

Batch planning, the host-side chunk schedule and host validation run on the
CPU everywhere; the schedule is compared with the device builder
``kernels.build_chunk_metadata`` on CPU tensors. The GPU cases compare
``aggregate_host`` with the plain-loop oracle for one to three slots, batch
budgets smaller and larger than single buckets (so two-pass batches occur),
empty and clipped windows, and check that a trace has one interval per
stage and batch in stream order.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge import contract, generate, kernels, pipelined, reference, windows

W = contract.DEFAULT_BUCKET_WIDTH_US


def _offsets(lengths: list[int], head: int = 0) -> np.ndarray:
    return np.concatenate([[head], head + np.cumsum(lengths, dtype=np.int64)]).astype(np.int64)


# --- planning ----------------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 3, 5, 8, 100])
def test_batches_cover_every_bucket_once_in_order_within_the_budget(budget: int) -> None:
    lengths = [0, 2, 3, 0, 0, 7, 1, 4, 0]
    offsets = _offsets(lengths, head=4)
    plans = pipelined.plan_batches(offsets, budget)
    assert plans[0].first == 0 and plans[-1].last == len(lengths)
    for left, right in zip(plans, plans[1:], strict=False):
        assert left.last == right.first
    for plan in plans:
        assert plan.n_buckets >= 1
        assert plan.record_start == offsets[plan.first]
        assert plan.record_end == offsets[plan.last]
        assert plan.max_bucket_len == max(lengths[plan.first:plan.last])
        # Over budget only when the batch is a single bucket longer than the budget.
        assert plan.n_records <= budget or plan.n_buckets == 1


def test_a_bucket_longer_than_the_budget_is_a_batch_alone() -> None:
    offsets = _offsets([1, 50, 1])
    plans = pipelined.plan_batches(offsets, 10)
    assert [(p.first, p.last) for p in plans] == [(0, 1), (1, 2), (2, 3)]


def test_two_pass_batches_carry_their_chunk_count() -> None:
    long = kernels.MAX_ONE_PASS_BUCKET + 5
    plans = pipelined.plan_batches(_offsets([3, long, 2]), 1 << 30)
    assert len(plans) == 1 and plans[0].two_pass
    assert plans[0].n_chunks == 1 + (-(-long // kernels.CHUNK_SIZE)) + 1


def test_batch_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        pipelined.plan_batches(_offsets([1]), 0)


@pytest.mark.parametrize("lengths", [[0, 5, 0, 17, 1], [32], [0, 0], [31, 33, 64, 0, 65]])
@pytest.mark.parametrize("chunk", [1, 4, 32])
def test_host_chunk_schedule_equals_the_device_builder(lengths: list[int], chunk: int) -> None:
    import torch

    offsets = _offsets(lengths)
    host = pipelined.chunk_schedule(offsets, chunk)
    device = kernels.build_chunk_metadata(torch.from_numpy(offsets), chunk)
    np.testing.assert_array_equal(host["chunk_starts"], device.chunk_starts.numpy())
    np.testing.assert_array_equal(host["chunk_ends"], device.chunk_ends.numpy())
    np.testing.assert_array_equal(host["chunk_offsets"], device.chunk_offsets.numpy())
    assert all(v.dtype == np.int64 for v in host.values())


def test_host_validation_rejects_bad_inputs() -> None:
    trades = generate.generate_trades(seed=1, n_buckets=10, profile="uniform")
    offsets = windows.build_offsets(trades["timestamp"], 0, 10, W)
    columns = pipelined._columns(trades)
    assert pipelined._validate_host(columns, offsets) == int(np.diff(offsets).max())
    with pytest.raises(ValueError, match="non-decreasing"):
        pipelined._validate_host(columns, offsets[::-1].copy())
    with pytest.raises(ValueError, match="span"):
        pipelined._validate_host(columns, offsets + 1)
    bad = dict(columns, price=-columns["price"])
    with pytest.raises(ValueError, match="non-negative"):
        pipelined._validate_host(bad, offsets)
    huge = dict(columns, price=np.full_like(columns["price"], 2**62))
    with pytest.raises(OverflowError):
        pipelined._validate_host(huge, offsets)
    with pytest.raises(ValueError, match="int64"):
        pipelined._validate_host(dict(columns, quantity=columns["quantity"].astype(np.int32)),
                                 offsets)


# --- GPU -----------------------------------------------------------------------


def _assert_matches_oracle(got: dict[str, np.ndarray], trades: np.ndarray,
                           offsets: np.ndarray) -> None:
    expected = reference.aggregate_reference_naive(trades, offsets)
    assert tuple(got) == contract.FEATURE_NAMES
    for name in contract.FEATURE_NAMES:
        target = getattr(expected, name)
        assert got[name].dtype == target.dtype, name
        if name == "vwap":
            np.testing.assert_allclose(got[name], target, rtol=contract.VWAP_RTOL,
                                       atol=contract.VWAP_ATOL)
        else:
            np.testing.assert_array_equal(got[name], target, err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize(("budget", "slots"), [(1 << 20, 2), (997, 1), (997, 2), (4096, 3)])
def test_overlapped_batches_match_the_oracle(
    gpu_ready: None, profile: str, budget: int, slots: int
) -> None:
    n_buckets = {"sparse": 6000, "uniform": 300, "burst": 400}[profile]
    trades = generate.generate_trades(seed=21, n_buckets=n_buckets, profile=profile)  # type: ignore[arg-type]
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, W)
    stats = pipelined.PipelineStats()
    got = pipelined.aggregate_host(trades, offsets, batch_records=budget, slots=slots,
                                   stats=stats)
    _assert_matches_oracle(got, trades, offsets)
    assert stats.n_batches == len(pipelined.plan_batches(offsets, budget))


@pytest.mark.gpu
def test_two_pass_batches_and_a_clipped_window_match_the_oracle(gpu_ready: None) -> None:
    rng = np.random.default_rng(3)
    long = 3 * kernels.MAX_ONE_PASS_BUCKET + 11
    lengths = [0, 4, long, 0, 2, kernels.CHUNK_SIZE + 1, 0]
    n = 9 + sum(lengths) + 6
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = rng.integers(1, 10**7, n)
    trades["quantity"] = rng.integers(0, 10**6, n)
    trades["buyer_is_maker"] = rng.random(n) < 0.5
    offsets = _offsets(lengths, head=9)
    for budget in (8, 5000, 1 << 22):
        got = pipelined.aggregate_host(trades, offsets, batch_records=budget, slots=2)
        _assert_matches_oracle(got, trades, offsets)


@pytest.mark.gpu
def test_an_all_empty_input_returns_zero_features(gpu_ready: None) -> None:
    trades = np.zeros(3, dtype=contract.TRADE_DTYPE)
    offsets = np.array([1, 1, 1, 1], dtype=np.int64)
    got = pipelined.aggregate_host(trades, offsets)
    _assert_matches_oracle(got, trades, offsets)


@pytest.mark.gpu
def test_a_trace_has_ordered_intervals_for_every_batch(gpu_ready: None) -> None:
    trades = generate.generate_trades(seed=2, n_buckets=2000, profile="uniform")
    offsets = windows.build_offsets(trades["timestamp"], 0, 2000, W)
    stats = pipelined.PipelineStats()
    got = pipelined.aggregate_host(trades, offsets, batch_records=10_000, slots=3, trace=True,
                                   stats=stats)
    _assert_matches_oracle(got, trades, offsets)
    rows = pipelined.timeline_intervals(stats)
    assert len(rows) == stats.n_batches > 3
    for row in rows:
        assert row["copy_in_start"] <= row["copy_in_end"] <= row["compute_start"]
        assert row["compute_start"] <= row["compute_end"] <= row["copy_out_start"]
        assert row["copy_out_start"] <= row["copy_out_end"]
    assert pipelined.overlap_ms(rows, "copy_in", "compute") >= 0.0


# --- int32 transfer ------------------------------------------------------------


def test_the_int32_transfer_bound_covers_only_the_window() -> None:
    limit = pipelined.INT32_MAX
    columns = {"price": np.array([limit + 1, limit, 5], dtype=np.int64),
               "quantity": np.array([1, limit, 0], dtype=np.int64),
               "buyer_is_maker": np.zeros(3, dtype=np.bool_)}
    assert pipelined._fits_int32(columns, np.array([1, 3], dtype=np.int64))
    assert not pipelined._fits_int32(columns, np.array([0, 3], dtype=np.int64))
    columns["quantity"][2] = limit + 1
    assert not pipelined._fits_int32(columns, np.array([1, 3], dtype=np.int64))
    assert pipelined._fits_int32(columns, np.array([2, 2], dtype=np.int64))  # empty window


@pytest.mark.gpu
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
@pytest.mark.parametrize(("budget", "slots"), [(1 << 20, 2), (997, 1), (4096, 3)])
def test_int32_transfer_matches_the_oracle(
    gpu_ready: None, profile: str, budget: int, slots: int
) -> None:
    n_buckets = {"sparse": 6000, "uniform": 300, "burst": 400}[profile]
    trades = generate.generate_trades(seed=27, n_buckets=n_buckets, profile=profile)  # type: ignore[arg-type]
    offsets = windows.build_offsets(trades["timestamp"], 0, n_buckets, W)
    stats = pipelined.PipelineStats()
    got = pipelined.aggregate_host(trades, offsets, batch_records=budget, slots=slots,
                                   int32_transfer=True, stats=stats)
    _assert_matches_oracle(got, trades, offsets)
    assert stats.int32_transfer and stats.bytes_in_per_record == 9


@pytest.mark.gpu
def test_int32_transfer_is_exact_at_the_bound_and_falls_back_past_it(gpu_ready: None) -> None:
    limit = pipelined.INT32_MAX
    lengths = [0, 3, kernels.CHUNK_SIZE + 1, 1, 0, 2 * kernels.MAX_ONE_PASS_BUCKET + 5]
    n = 2 + sum(lengths)
    rng = np.random.default_rng(8)
    trades = np.zeros(n, dtype=contract.TRADE_DTYPE)
    trades["price"] = rng.integers(limit - 1000, limit, n, endpoint=True)
    trades["quantity"] = rng.integers(0, 3, n)
    trades["price"][5] = limit
    trades["buyer_is_maker"] = rng.random(n) < 0.5
    offsets = _offsets(lengths, head=2)
    for budget in (7, 5000, 1 << 22):
        stats = pipelined.PipelineStats()
        got = pipelined.aggregate_host(trades, offsets, batch_records=budget,
                                       int32_transfer=True, stats=stats)
        _assert_matches_oracle(got, trades, offsets)
        assert stats.int32_transfer
    trades["price"][6] = limit + 1
    stats = pipelined.PipelineStats()
    got = pipelined.aggregate_host(trades, offsets, int32_transfer=True, stats=stats)
    _assert_matches_oracle(got, trades, offsets)
    assert not stats.int32_transfer and stats.bytes_in_per_record == 17
