"""Scope 1: kernel-only latency on already-resident GPU tensors (TF-08).

Inputs are moved to the device *before* timing starts. This number answers
"how fast is the reduction", and must never be compared against a CPU
pipeline that also decompresses a ZIP (PRD section 6).

Baselines: eager PyTorch (scatter/segment-reduce composition) and, if
practical, the same under ``torch.compile``.

    python benchmarks/kernel.py --config benchmarks/configs/kernel_sweep.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harness import record, time_callable  # noqa: F401  (used once implemented)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/kernel_sweep.toml")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    # TODO(week 4): for each size x profile in the config, time
    #   (a) tapeforge.ops.aggregate on resident CUDA tensors,
    #   (b) the eager PyTorch baseline, (c) torch.compile if available,
    # then record(scope="kernel", ...). The kernel itself is student-written;
    # this file only measures whatever kernels.py provides.
    raise NotImplementedError(f"week 4: implement the sweep over {config['sizes']}")


if __name__ == "__main__":
    raise SystemExit(main())
