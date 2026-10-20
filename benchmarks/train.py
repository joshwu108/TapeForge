"""Scope 3: the whole training loop (TF-10, ML impact).

Reports training examples/s, GPU wait time and peak VRAM, with the feature
path swapped between the reference and the GPU engine. This is the figure
that shows whether a kernel speedup actually moves training throughput --
keep it even when it shows that it does not.

    python benchmarks/train.py --config benchmarks/configs/train.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harness import record  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/train.toml")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    # TODO(week 7): train the small model twice -- reference features vs GPU
    # features -- on the identical chronological split, and record examples/s,
    # torch.cuda.max_memory_allocated(), and profiler GPU-idle share.
    raise NotImplementedError(f"week 7: implement using {config}")


if __name__ == "__main__":
    raise SystemExit(main())
