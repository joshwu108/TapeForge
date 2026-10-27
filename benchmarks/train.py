"""Scope 3: whole-training-loop throughput. Not implemented; the entry point raises.

The benchmark is intended to report training examples/s, GPU wait time and
peak VRAM, with the feature path swapped between the reference and the GPU
engine. That figure shows whether a kernel speedup moves training
throughput, and is reported even where it shows that it does not. The
implementation is blocked on the model and label specification
(``tapeforge.model`` is a stub).

Usage:
    python benchmarks/train.py --config benchmarks/configs/train.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harness import record  # noqa: F401


def main() -> int:
    """Parse the config, then raise ``NotImplementedError``.

    Raises:
        NotImplementedError: Always; the benchmark body does not exist yet.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/train.toml")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    # Pending implementation: train the small model twice (reference features
    # and GPU features) on the identical chronological split, and record
    # examples/s, torch.cuda.max_memory_allocated() and the profiler's
    # GPU-idle share. Blocked on the model/label specification (model.py).
    raise NotImplementedError(
        f"training throughput benchmark is not implemented; config was {config}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
