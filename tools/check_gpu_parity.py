"""Run with: python tools/check_gpu_parity.py (requires CUDA and Triton)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; run this script in a GPU runtime.")

from tapeforge import contract, generate, reference, windows  # noqa: E402
from tapeforge.kernels import BucketTooLargeError, launch_aggregate  # noqa: E402


def check(profile, seed):
    trades = generate.generate_trades(seed=seed, n_buckets=100, profile=profile)
    offsets = windows.build_offsets(
        trades["timestamp"], 0, 100, contract.DEFAULT_BUCKET_WIDTH_US
    )
    expected = reference.aggregate_reference(trades, offsets)
    columns = {
        name: torch.as_tensor(trades[name].copy(), device="cuda")
        for name in ("price", "quantity", "buyer_is_maker")
    }
    gpu_offsets = torch.as_tensor(offsets, device="cuda")
    maximum = int(np.diff(offsets).max())
    if maximum > 1024:
        try:
            launch_aggregate(columns, gpu_offsets, block_size=1024)
        except BucketTooLargeError:
            print(f"PASS {profile=} {seed=}: oversized bucket rejected ({maximum})")
            return
        raise AssertionError("Oversized bucket was silently accepted")

    got = launch_aggregate(columns, gpu_offsets, block_size=1024)
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
    print(f"PASS {profile=} {seed=}: all five outputs match CPU")


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name())
    for profile in ("sparse", "uniform", "burst"):
        for seed in (1, 2, 3):
            check(profile, seed)
