"""Quick GPU parity check for both kernel paths.

Run with: python tools/check_gpu_parity.py (requires CUDA and Triton).

For each profile and seed, a 1,000-bucket workload is aggregated on CUDA
through the checked API (``kernels.launch_aggregate``) and through the
prepared path (``prepared.prepare(...).execute()``), and both are compared
with the NumPy reference: integers exactly, VWAP within the contract
tolerance. Burst workloads carry buckets beyond the one-pass tile and
exercise the two-pass kernels; sparse and uniform exercise the one-pass
kernel. The dispatch path and pass-1 chunk count are printed so the log
shows which kernels ran. The pytest GPU suite is the thorough check; this
script is the one-screen summary for a verification log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; run this script in a GPU runtime.")

from tapeforge import contract, generate, prepared, reference, windows  # noqa: E402
from tapeforge.kernels import launch_aggregate, plan_dispatch  # noqa: E402

N_BUCKETS = 1_000


def _assert_matches(got: dict[str, torch.Tensor], expected: reference.Features) -> None:
    for name in contract.FEATURE_NAMES:
        actual = got[name].cpu().numpy()
        target = getattr(expected, name)
        assert actual.dtype == target.dtype, (name, actual.dtype, target.dtype)
        if name == "vwap":
            np.testing.assert_allclose(
                actual, target, rtol=contract.VWAP_RTOL, atol=contract.VWAP_ATOL
            )
        else:
            np.testing.assert_array_equal(actual, target)


def check(profile: str, seed: int) -> None:
    trades = generate.generate_trades(seed=seed, n_buckets=N_BUCKETS, profile=profile)
    offsets = windows.build_offsets(
        trades["timestamp"], 0, N_BUCKETS, contract.DEFAULT_BUCKET_WIDTH_US
    )
    expected = reference.aggregate_reference(trades, offsets)
    columns = {
        name: torch.as_tensor(trades[name].copy(), device="cuda")
        for name in ("price", "quantity", "buyer_is_maker")
    }
    gpu_offsets = torch.as_tensor(offsets, device="cuda")
    longest = int(np.diff(offsets).max())
    plan = plan_dispatch(longest)

    _assert_matches(launch_aggregate(columns, gpu_offsets), expected)
    ready = prepared.prepare(columns, gpu_offsets)
    assert ready.path == plan.path
    _assert_matches(ready.execute(), expected)
    _assert_matches(ready.execute(), expected)  # buffer reuse

    detail = (
        f"block_size {plan.block_size}"
        if plan.path == "one_pass"
        else f"chunk_size {plan.chunk_size}, {ready.n_chunks} chunks"
    )
    print(
        f"PASS {profile=} {seed=}: checked API and prepared path match CPU "
        f"(longest bucket {longest}, {plan.path}, {detail})"
    )


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name())
    for profile in ("sparse", "uniform", "burst"):
        for seed in (1, 2, 3):
            check(profile, seed)
