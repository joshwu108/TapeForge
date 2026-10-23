"""Toy end-to-end check of the CUDA launcher on the hand-calculated example.

Two records in bucket 0 (buy-initiated 100.00 x 2, sell-initiated 102.00 x 1)
and an empty bucket 1: counts 2 / 0, total 3 / 0, signed +1 / 0, VWAP
100.666... / 0.0. Needs a CUDA device and Triton; ``tools/verify_snapshot.py``
runs it as the ``toy_program`` step and the test suites assert the same
numbers on the CPU.
"""

import torch

from tapeforge.kernels import launch_aggregate

trades = {
    "price": torch.tensor([10000, 10200], dtype=torch.int64, device="cuda"),
    "quantity": torch.tensor([200000, 100000], dtype=torch.int64, device="cuda"),
    "buyer_is_maker": torch.tensor([False, True], device="cuda"),
}
offsets = torch.tensor([0, 2, 2], dtype=torch.int64, device="cuda")

got = launch_aggregate(trades, offsets)

expected = {
    "event_count": [2, 0],
    "total_quantity": [300000, 0],
    "signed_quantity": [100000, 0],
    "vwap": [(100 * 2 + 102) / 3, 0.0],
    "valid": [True, False],
}

for name, values in expected.items():
    target = torch.tensor(values, dtype=got[name].dtype, device="cuda")
    torch.testing.assert_close(got[name], target, rtol=1e-9, atol=1e-12)

print("Toy example passed")