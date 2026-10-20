"""Scope 2: host-to-device transfer plus the whole feature pipeline (TF-09).

Reports records/s, windows/s, and the split between time spent copying and
time spent computing -- the breakdown that tells you whether pinned memory
and CUDA streams are worth keeping.

    python benchmarks/pipeline.py --config benchmarks/configs/pipeline.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harness import record, time_callable  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/pipeline.toml")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    # TODO(week 6): time parse -> index -> H2D -> features end to end, with and
    # without pinned memory / streams, and record both. Keep the optimization
    # only if the end-to-end measurement improves (student's call).
    raise NotImplementedError(f"week 6: implement using {config}")


if __name__ == "__main__":
    raise SystemExit(main())
