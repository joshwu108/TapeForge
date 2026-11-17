============
Contributing
============

TapeForge is a small offline research tool. This guide says where things
live, what each part is responsible for, and what a change must show before
it is merged. ``README.rst`` is the user documentation; read its
"Development guidelines" first.

Set up and check
================

.. code-block:: console

   $ python3.12 -m venv .venv && source .venv/bin/activate
   $ python -m pip install -e '.[dev,torch,train]'
   $ python -m ruff check src tests benchmarks tools
   $ python -m mypy
   $ python -m pytest -q

The suite needs no downloads and no GPU. Tests that need CUDA, an optional
package or a downloaded archive skip with the reason; a skip is never a pass.

Component responsibilities
==========================

.. list-table::
   :header-rows: 1
   :widths: 30 38 32

   * - Component
     - Owns
     - Must not
   * - ``contract``, ``validation``, ``reference``, ``streaming``
     - Units, bounds and the CPU feature reference
     - Depend on torch
   * - ``kernels``, ``prepared``, ``ops``
     - GPU feature kernels and their public API
     - Change without the kernel maintainer's approval (``kernels.py``)
   * - ``execution.spec``, ``state``, ``simulator``
     - The transition rules: fills, fees, settlement, rewards, episode ends
     - Import anything from ``data``, ``research`` or ``rl``
   * - ``execution.batched``, ``execution.parity``
     - The same rules as device tensors, and the comparison with the reference
     - Define a rule the reference does not have
   * - ``execution.data``
     - Archives, day tapes, episode windows, splits, seeded sampling
     - Run policies or read the test role implicitly
   * - ``execution.research``
     - Configured runs, reports, replay, sensitivity tables, matrices
     - Train anything
   * - ``execution.rl``
     - The Gymnasium adapter, frozen statistics, evaluation and provenance,
       the optional PPO trainer
     - Reimplement accounting; fit statistics outside the training role

Rules for a change
==================

* **Financial semantics are versioned.** A change to a fill, fee, rounding,
  settlement, reward, observation or termination rule needs the maintainer's
  explicit approval, a new ``SIMULATOR_VERSION`` and updated oracle tests.
  Old artifacts are refused, never reinterpreted.
* **The batched simulator follows the reference.** Change the reference
  first, then the tensor code, and keep ``execution.parity`` passing after
  every transition. Do not add a host synchronisation (``.item()``,
  ``.cpu()``, a Python branch on a tensor) to ``batched.step`` or
  ``batched.observe``; a test fails if one appears.
* **Configuration is strict.** Unknown keys are errors and nothing that
  changes a result may come from a library default. New settings go into the
  file format with validation and a test.
* **Optional dependencies stay optional.** The CPU reference imports NumPy
  only. ``torch``, ``gymnasium`` and ``stable-baselines3`` are imported by
  the modules that need them, with an error naming the extra; subprocess
  tests check both directions.
* **Held-out data stays held out.** The test role is evaluated only with an
  explicit flag and the maintainer's authorization. Do not tune, select or
  debug on it, and do not move test dates into another role.
* **Say what was verified.** Keep "implemented", "CPU-tested",
  "CUDA-tested", "experiment executed" and "conclusion supported" apart in
  code comments, commit messages and documentation. Performance statements
  cite a recorded measurement with its hardware and revision.
* **Tests come with the change.** Accounting changes need exact-integer
  assertions; learning code is tested on constructed tasks, never on
  historical profitability.

Comments and documentation
==========================

Write for the next maintainer: units, invariants, causality, rounding and
the reason for a non-obvious choice. Do not narrate the development process,
restate the next line, or label work by course weeks or milestones. Public
documentation is reStructuredText (``README.rst`` and this file); Markdown,
planning notes, datasets, run directories and checkpoints are ignored by Git
and must not be committed.

Commits
=======

Use ``<type>: <description>`` with one of ``feat``, ``fix``, ``refactor``,
``docs``, ``test``, ``chore``, ``perf``, ``ci``. Commit one coherent, tested
change at a time, stage explicit paths, and describe machine-assisted work
as such.
