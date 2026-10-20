"""Triton GPU kernels -- STUDENT-OWNED CODE. DO NOT DELEGATE TO AN AGENT.

PRD section 12.1/12.4 reserve every line of GPU kernel code, and every
thread/block, segmented-reduction, fusion and memory-layout decision, for
the student. An agent may explain Triton, review kernels on request, and
build the parity/benchmark harness around them -- it must not write or
modify them.

What belongs here (weeks 3-4), for the student to implement:

  1. ``agg_kernel_one_pass`` -- one program per bucket. Each program reads
     its ``[offsets[b], offsets[b + 1])`` slice in BLOCK_SIZE strides and
     accumulates count, total quantity, signed quantity, and the two VWAP
     inputs (sum of price*quantity, sum of quantity) in registers, writing
     only the final per-bucket values. One load of each input, one store of
     each output: the whole point of the fused kernel.

  2. A two-pass path for oversized buckets (TF-06): partial block
     reductions into a scratch buffer, then a final reduction over the
     partials, so a single program never walks an unbounded bucket.

  3. A launch wrapper that picks the one-pass or two-pass path from the
     measured bucket-length distribution, with the threshold chosen from
     profiling evidence, not guessed.

Open design questions for the student to decide and record in
``planning/decisions.md`` before implementing:

  - one program per bucket vs. one program per fixed tile of records with
    atomics into bucket slots;
  - whether VWAP is finished inside the kernel or as a cheap division in
    ``ops.py`` (affects the float tolerance story);
  - int64 vs float32/float64 accumulation for price*quantity, and the
    resulting overflow bound.

The harness this module must satisfy already exists: tests/test_gpu_parity.py
and benchmarks/kernel.py. Both skip cleanly while this module is unimplemented.
"""

from __future__ import annotations

TRITON_AVAILABLE: bool
try:  # pragma: no cover - import probe
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only machine (weeks 1-2)
    TRITON_AVAILABLE = False

# TODO(student, week 3): implement after running the official Triton
# vector-add tutorial. Leave this NotImplementedError in place until then --
# the parity tests and benchmarks detect it and skip.
