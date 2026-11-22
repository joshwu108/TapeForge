=========
TapeForge
=========

TapeForge is an offline research platform for optimal trade execution. It
turns historical market events into features, replays execution policies in
a deterministic CPU simulator, and records comparable experiments. A Triton
backend accelerates feature aggregation; a Gymnasium adapter exposes the
reference simulator to reinforcement-learning libraries; an optional PPO
trainer and a batched tensor implementation of the simulator build on it.

The research objective is to measure whether GPU feature computation and
batched GPU simulation reduce training time while preserving numerical and
temporal semantics. The batched simulator matches the CPU reference on every
compared transition on CPU tensors and on one CUDA device (a Tesla T4). On
that device no end-to-end training speedup was measured: at eight
environments the batched CUDA backend was the slowest of the three backends.
Compiling its vector step and validating with a CPU copy of the policy
reduced that run from 139.8 s to 65.7 s, still slower than serial CPU
(53.9 s; ``benchmarks/evidence/t4-optimisation-2026-11-21.json``).

Architecture
============

.. code-block:: text

   Historical aggregate trades / synthetic events
       -> validated integer-scaled data
       -> packed window offsets
       -> NumPy reference / Triton feature aggregation
       -> bucket tapes and chronological splits
       -> sampled execution episodes
       -> baseline policy, or PPO through the Gymnasium adapter
       -> CPU simulator: fills, fees, inventory, rewards
          (or the batched tensor simulator, held to the same transitions)
       -> trajectories, checkpoints, paired reports, reproducible run records

The CPU references define numerical behavior. The GPU feature engine uses
one-pass reductions for short buckets and two-pass integer partial sums for
large buckets. A separate streaming CPU aggregator checks the same feature
semantics across ordered input chunks.

The execution simulator currently uses a synthetic, volume-capped bucket
VWAP fill model and, in its primary configuration, a synthetic terminal
settlement of whatever is unfilled at the horizon. It is not an exchange
simulator or a live trading system. There is no order routing, passive-order
queue model, market impact, or market response to the policy's actions, and
neither the fills nor the settlement represent real execution.

Status
======

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Component
     - Current state
   * - Data and features
     - Aggregate-trade ingestion, checksum-verified acquisition, synthetic
       generation, fixed-width offsets, independent CPU references, and
       ordered-chunk streaming aggregation implemented and CPU-tested.
   * - GPU feature engine
     - One-pass and two-pass kernels, dispatch, and prepared execution
       implemented. The 82-test GPU suite passed on a Tesla T4 at commit
       ``c5d0f80`` (PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0).
       This verification is specific to that revision's GPU-facing code.
   * - CPU execution research
     - Reference simulator (version 2: terminal settlement as the primary
       rule, the legacy residual rule named explicitly), day tapes,
       chronological splits, episode sampling, train-only normalization with
       versioned statistics files, three baselines plus a zero-action
       diagnostic, experiment CLI, reporting, replay and cross-run
       sensitivity tables implemented and CPU-tested.
   * - RL interface
     - Optional Gymnasium CPU adapter tested against direct simulator calls
       and the baseline policies; environment checker command; fixed
       evaluation manifests; external-policy evaluation writing
       baseline-comparable records; checkpoint provenance schema.
   * - PPO trainer
     - Optional (``train`` extra): Stable-Baselines3 PPO with a small MLP,
       strict configuration, train-only frozen normalisation, per-rollout
       diagnostics, checkpoints with provenance, validation-based checkpoint
       selection and evaluation into baseline run directories. Implemented
       and CPU-tested, including bit-identical resume on the serial and
       subprocess backends and on the batched backend over CPU tensors.
       Trained on CUDA (batched backend, one T4) in the validation pilot.
       No SAC or recurrent policy.
   * - Batched simulator
     - ``execution.batched`` applies the reference transitions to many
       episodes as device tensors, with a per-transition parity driver and a
       vector environment for the trainer. Implemented and tested on CPU
       tensors. **CUDA-tested on one Tesla T4** in two runs at different
       revisions: all gpu-marked tests passed with none skipped, and 5,000
       parity episodes (26,696 transitions) matched with no mismatch
       (``benchmarks/evidence/cuda-t4-2026-11-19.json``). Other GPUs,
       drivers and PyTorch versions are not covered.
   * - Performance evaluation
     - Kernel, stage, transfer (including overlapped pinned streams), scale
       and seeded adversarial-case benchmarks implemented; a
       training-pipeline benchmark compares serial, subprocess and batched
       environments stage by stage, with the compiled step and CPU-policy
       validation as variants. Synthetic runs on one T4 host are summarised
       in ``benchmarks/evidence/``; none shows a GPU training speedup over
       serial CPU. Kernel sweeps are not adopted into
       ``benchmarks/results.json``; kernel tuning decisions remain pending.
   * - Robustness
     - One-factor experiment matrices (fee, latency, data delay, terminal
       surcharge, risk coefficient, period) implemented and run locally for
       the baselines and for policies trained under the base settings.
       Retraining per variant, quote-aware fills and fidelity experiments
       remain unimplemented.

CI checks CPU behavior; it has no CUDA runner. Hardware verification must be
repeated after GPU-facing changes. T4 verification is summarised in
``benchmarks/evidence/cuda-t4-2026-11-19.json``; no kernel row from it is
recorded in ``benchmarks/results.json``. **No speedup figures are
published.** Local three-day futures smoke experiments demonstrated the CPU
pipeline under the terminal-settlement and legacy rules; they do not
establish policy superiority, out-of-sample skill, or a research value of
the terminal surcharge.

A pilot was executed locally on CPU (see "Pilot experiment"): three PPO
model seeds of 960,000 steps each, checkpoints selected on validation. It is
a **null result**. On the six validation days the mean implementation
shortfall of the three seeds (11.34, 11.40 and 11.43 bps) is
indistinguishable from the ``immediate`` baseline (11.37 bps; the paired
day-cluster intervals include zero or are marginally unfavourable) and from
``cap_aware_twap`` (11.26 bps). All of them are far below ``twap``
(16.10 bps), which the participation cap keeps from completing. In this fill
model a completing policy pays the fee plus zero-mean price drift, so little
is left to learn; and a one-factor sensitivity run shows that the ranking of
the baselines depends on the terminal surcharge setting (at 0 bps the
policies that wait are cheapest). These figures come from one instrument and
one month under labelled settings. They are not out-of-sample evidence, and
the sealed test days have not been evaluated.

The validation pilot was repeated with PPO trained by the batched backend on
one Tesla T4, with the same settings, frozen statistics, selection rule and
the same 300 validation episodes; the runtime configuration contained no
test dates. It is a second **null result**, not a confirmation of skill:
the selected checkpoints score 11.33, 11.41 and 11.42 bps against 11.37 for
``immediate`` (paired differences -0.03, +0.05 and +0.05 bps; only the
third interval excludes zero, unfavourably) and are indistinguishable from
``cap_aware_twap``. The batched backend draws training episodes in another
order, so its seeds are not replicas of the CPU seeds. Moving training to a
GPU does not change the conclusion. Record:
``benchmarks/evidence/pilot-cuda-validation-2026-11-19.json``.

Install
=======

Requires Python 3.11 or 3.12. CPU research runs on macOS or Linux. The Triton
backend requires Linux, an NVIDIA CUDA GPU, and a compatible PyTorch/Triton
installation.

.. code-block:: console

   $ python3.12 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install -e '.[dev,torch,rl]'      # CPU development and Gymnasium
   $ python -m pip install -e '.[dev,torch,train]'   # plus the PPO trainer
   $ python -m pip install -e '.[dev,gpu,train]'     # Linux CUDA development

NumPy is the only required runtime dependency. CPU references, streaming,
execution simulation, and baseline experiments do not require PyTorch.
``ops.aggregate`` and ``execution.batched`` require the ``torch`` extra; the
environment adapter requires ``rl``; the trainer requires ``train``
(Gymnasium, Stable-Baselines3 2.9 and PyTorch). Each optional module names
its extra when the dependency is missing.

Run a baseline experiment
=========================

Start with synthetic data; no downloads or GPU are needed:

.. code-block:: console

   $ python -m tapeforge.execution.research.experiment run experiments/configs/smoke-synthetic.toml

The command creates ``runs/<name>-<utc>/``. Use the printed directory in
subsequent commands:

.. code-block:: console

   $ python -m tapeforge.execution.research.report runs/<run-dir>
   $ python -m tapeforge.execution.research.replay runs/<run-dir> \
       --episode validation/s1/i0 --policy twap --verify
   $ python -m tapeforge.execution.research.experiment compare runs/<run-a> runs/<run-b>
   $ python -m tapeforge.execution.rl.env_check experiments/configs/smoke-synthetic.toml

The report writes JSON, CSV, and reStructuredText under ``<run-dir>/report/``.
Replay prints the visible market bucket, action, submitted quantity, fill
cap, fill price, fees, settled quantity and price, remaining inventory,
reward components, and terminal flags; ``--verify`` checks the replay
against its recorded outcome. The environment checker runs Gymnasium's
``check_env``, drives random actions through the adapter and compares every
step with a direct simulator call; it needs the ``rl`` extra.

For historical data, the futures smoke configurations use BTCUSDT aggregate
trades from 2023-12-01 through 2023-12-03: two training days, one validation
day, and no test period. They allow at most three downloads totaling 500 MiB.
They are demonstration configurations, not a research evaluation protocol.
One configuration per terminal-rule setting keeps the sampled episodes
identical across settings so the sensitivity tool can line them up:

.. code-block:: console

   $ python -m tapeforge.execution.data.datasets fetch --market futures-um \
       --symbol BTCUSDT --date 2023-12-01 --date 2023-12-02 --date 2023-12-03
   $ for s in 0 10 25 50; do python -m tapeforge.execution.research.experiment run \
       experiments/configs/smoke-futures-btcusdt-2023-12-settlement-${s}bps.toml; done
   $ python -m tapeforge.execution.research.experiment run \
       experiments/configs/smoke-futures-btcusdt-2023-12-legacy-perold.toml
   $ python -m tapeforge.execution.research.sensitivity runs/<run-0bps> runs/<run-10bps> \
       runs/<run-25bps> runs/<run-50bps> runs/<run-legacy> --out runs/<sensitivity-dir>

``smoke-futures-btcusdt-2023-12.toml`` is the 25 bps settlement configuration
with downloads enabled. The surcharge values are experimental settings for a
sensitivity comparison, not calibrated market-impact estimates; 25 bps is the
initial smoke setting, not a chosen research coefficient.

Archives live under ``data/<market>/aggTrades/`` and are checked against the
source's published SHA-256 checksum. Verified cached archives can be reused
without network access. Spot and futures are separate markets; a run names
one market and symbol. Check source licensing before redistributing data.

Pilot experiment
================

``experiments/configs/pilot-futures-btcusdt-2023-12.toml`` fixes one
protocol: Binance USD-M futures BTCUSDT, 23 consecutive UTC days, assigned
chronologically to 11 training days (2023-12-01 to 11), 6 validation days
(12 to 17) and 6 test days (18 to 23). The simulator settings are the
accepted first configuration: 10 bps fee, no latency or data delay, risk
term off, terminal settlement at 25 bps. The 25 bps surcharge is a labelled
initial setting, not a calibrated market-impact estimate. The date selection
is a proposal with plain limits: one instrument, one venue, one contiguous
month.

.. code-block:: console

   $ python -m tapeforge.execution.data.datasets fetch --market futures-um \
       --symbol BTCUSDT --max-archives 23 --max-bytes 1073741824 \
       $(for d in $(seq -w 1 23); do printf -- '--date 2023-12-%s ' "$d"; done)
   $ python -m tapeforge.execution.research.experiment run \
       experiments/configs/pilot-futures-btcusdt-2023-12.toml
   $ python -m tapeforge.execution.research.report runs/<pilot-run> --reference immediate
   $ python -m tapeforge.execution.rl.evaluation manifest runs/<pilot-run>

The 23 archives total about 335 MiB and are verified against the published
SHA-256 checksums; the configuration itself never downloads. The run replays
``twap``, ``immediate``, ``cap_aware_twap`` and the ``zero_action``
diagnostic on identical episodes of the training and validation days. The
manifest records the data checksums, seeds, split record, simulator version
and settings; ``evaluation-manifest.json`` freezes the evaluation episode
set and its digest. The label ``pilot`` marks output as a fixed small-sample
protocol: its figures are not out-of-sample evidence.

The test days are sealed. No tool reads them unless ``--include-test`` is
passed, and passing it requires a frozen protocol and the maintainer's
authorization. They have not been evaluated.

Experiment contract
===================

The TOML configuration (schema 2) specifies data source, dates, split roles,
seeds, episodes per day, target fractions, policies, and simulator settings.
Unknown keys are rejected, and ``template.terminal_rule`` must be named:
``"terminal_settlement"`` with an explicit ``terminal_surcharge_bps``, or
the legacy ``"residual_perold"``. Schema 1 configurations, run records and
manifests, which predate the terminal rule, are refused with a migration
hint rather than reinterpreted. Research configurations need explicit dates;
copying the smoke configuration does not create a held-out experiment.

* Each policy receives the same seeded episode draws with fresh state.
* Splits are chronological, with embargo and episode-window purge. History,
  fills, and terminal marks must remain within the selected split segment.
* Targets are sized from volume before the episode, not future volume.
* Observation normalization is fitted on training episodes only and frozen
  for evaluation. The adapter never fits statistics automatically.
* Test-role evaluation requires ``--include-test``. Keep that period out of
  debugging and parameter selection.

A run directory contains ``manifest.json`` and
``records/<policy>-<role>-seed<n>.json``. The manifest records resolved
configuration, data checksums, revision and dirty state, simulator version
and assumptions, split metadata, episode IDs, timings, and peak memory.
Result files are written atomically; repeated runs get distinct directories.
Externally evaluated policies add records and manifest entries to the same
directory (see "Evaluating a policy").

Reports separate four outcomes: completed (ordinary fills reached the
target), settled (the target was reached only through the terminal
settlement), incomplete (legacy rule, residual left) and truncated (the tape
ran out). They include per-policy and per-day shortfall split into the
fills-only part and the terminal term, ordinary and settlement fees, the
risk charge, completion by fills next to final completion, the settled
fraction with a ``none`` / ``light`` / ``heavy`` settlement label, tail
costs, and paired differences on identical episode IDs. Truncated episodes
have fills-only shortfall and are not pooled with terminated episodes.
Day-cluster bootstrap intervals require at least five days; smaller samples
report insufficient data for uncertainty estimation. The zero-action policy
is labelled a diagnostic.

**Completion must be reported alongside cost.** Under terminal settlement
every terminated episode reaches the target by construction, so a 100 %
final completion rate says nothing by itself; read it next to completion by
fills, the settled fraction and the settlement cost. Under the legacy rule
with ``residual_mu = 0``, unfilled quantity incurs only the terminal
price-movement charge and avoids execution fees, so a policy that leaves
more unfilled can have lower reported shortfall; that is not evidence of
better execution. Surcharge and risk settings must be chosen before
optimizing a learned policy.

Execution model
===============

An episode attempts to buy or sell a target quantity within a fixed horizon.
``EpisodeSpec`` defines the rules, ``MarketSlice`` supplies bucket data,
``State`` tracks progress, and ``StepResult`` records one transition.
``run_episode`` coordinates a policy callable and the simulator;
``run_policy`` samples and evaluates multiple episodes without training.

The supplied smoke configurations use:

* 300 one-second decisions; target fractions of 2%, 5%, or 10% of preceding
  300-bucket volume, with the configured target cap and a one-tick floor.
* Actions in ``[0, 1]``: fraction of remaining quantity to submit, floored
  to quantity ticks.
* Fills at tick-rounded execution-bucket VWAP, capped at 10% of that bucket's
  quantity. Empty or zero-quantity buckets produce no fills.
* A 10-basis-point fee, rounded up in cash ticks; zero slippage and latency.
* Observations at decision ``t`` use buckets through ``t - 1 - data_delay``.
* An arrival price from the last valid visible bucket in the lookback.
* Terminal settlement of the quantity still unfilled at the horizon, at the
  terminal mark moved 25 basis points against the side (the smoke setting),
  with the same fee; an optional inventory-risk term.

Quantities use 1e-5 units and prices use 0.01 units. Cash ticks are the
product of price and quantity ticks. Accounting uses integers; bucket VWAP
is rounded half-up to price ticks using its integer numerator. These scales
must represent the chosen instrument exactly; ingestion rejects excess
precision instead of truncating it.

For a fill, execution cost is
``side * (notional - arrival_price * filled_quantity) + fee``.
Reward is the negative sum of execution cost, inventory risk, and the
terminal term. Completion or horizon exhaustion terminates an episode.
Missing data truncates it; the simulator does not invent a terminal mark,
a settlement or a residual charge for a truncated episode.

Terminal rules
--------------

The terminal mark is the VWAP of the first valid bucket at or after bucket
``horizon + latency``, searched over ``arrival_lookback`` buckets. Two rules
decide what happens to quantity still unfilled when the horizon ends; a
specification names one explicitly.

``terminal_settlement`` (primary research configuration). The residual is
settled in one synthetic fill at the terminal mark moved against the side
by ``terminal_surcharge_bps``: a buy pays the mark times
``(10000 + surcharge) / 10000`` rounded up to a price tick, a sell receives
the mark times ``(10000 - surcharge) / 10000`` rounded down. The settlement
pays the configured fee with the usual ceiling and updates inventory, cash,
the settled quantity, the settlement notional and the settlement fee, which
are kept apart from the ordinary fills. No opportunity-cost term is charged
in addition. The task therefore always ends with the target reached unless
the tape runs out. The settlement ignores the participation cap of the
terminal bucket and the record flags when it exceeds it; it is a boundary
condition of the task, not executable depth, impact or guaranteed liquidity.
``residual_mu`` must be zero under this rule, and ordinary
``slippage_ticks`` applies to fills only. Early completion settles nothing.

``residual_perold`` (legacy, explicit). The residual stays unfilled and is
charged ``side * (terminal_mark - arrival) * residual + residual_mu *
residual``. With ``residual_mu = 0`` the residual pays no fee and a
favourable price move rewards inaction. Simulator version 1 used this rule
for every episode; naming it reproduces those transitions.

Worked example on the reference fixture (arrival 10,000 ticks, fee 10 bps,
terminal mark 10,010, 120,000 of 300,000 ticks left after fills whose own
shortfall is 2,200,400 cash ticks):

.. list-table::
   :header-rows: 1
   :widths: 10 14 16 14 18 18

   * - Side
     - Surcharge
     - Settlement price
     - Settlement fee
     - Settlement cost
     - Total shortfall
   * - buy
     - 0 bps
     - 10,010
     - 1,201,200
     - 2,401,200
     - 4,601,600
   * - buy
     - 25 bps
     - 10,036
     - 1,204,320
     - 5,524,320
     - 7,724,720
   * - sell
     - 25 bps
     - 9,984
     - 1,198,080
     - 3,118,080
     - 4,518,480

Settlement cost is ``side * (price * 120,000 - 10,000 * 120,000) + fee``;
the sell row adds the sell-side fill shortfall of 1,400,400. Under the legacy
rule with ``residual_mu = 0`` the same buy residual costs 1,200,000 and no
fee. A zero-action buy on a terminal mark of 9,990 settles at 0 bps for
minus 3,000 cash ticks: favourable market movement can make delaying
execution beneficial, and the rule does not pretend otherwise; what it
removes is the fee avoidance. These cases are pinned by the oracle tests.

Under both rules the risk term charges ``risk_lambda`` per tick unfilled
after each step's ordinary fill, before any settlement, and the per-step
rewards of a terminated episode sum to minus the implementation shortfall
minus the risk charge.

The baselines are:

* ``twap``: divide remaining quantity by remaining decisions.
* ``immediate``: request all remaining quantity at every decision; fills
  remain capped.
* ``cap_aware_twap``: increase TWAP's submission if recent liquidity suggests
  the remaining decisions may not absorb the outstanding quantity. This is
  a heuristic, not a forecast.
* ``zero_action``: never trade. A diagnostic, not a strategy: it isolates
  the cost of the terminal rule alone and may win an episode when the
  market moves favourably.

Gymnasium interface
===================

``ExecutionEnv`` wraps the CPU simulator without duplicating accounting.
``SplitPool`` supplies reproducible sampled episodes; ``ListPool`` accepts
explicit episode entries.

* Observation: float32 vector of shape ``(10,)``; remaining/time fractions,
  side, causal mark information, and recent bucket features.
* Action: ``Box(0, 1, shape=(1,), dtype=float32)``. Invalid or non-finite
  actions raise rather than being silently clipped.
* Reward: basis points of arrival notional by default. ``info`` retains the
  exact integer reward and its components; raw-tick rewards are optional.
  The horizon step's reward includes the settlement cost, and its
  ``step_result`` carries the settled quantity, price, notional, fee and
  cap-override flag. The settlement price is read from the bucket after the
  horizon and never enters an observation the agent acted on.
* ``terminated`` and ``truncated`` follow the reference simulator. No
  wrapper time limit changes the horizon.
* Frozen training statistics can be supplied explicitly or loaded from a
  versioned file. Sequential pools support fixed evaluation sets; random
  selection supports training.

A CPU smoke example:

.. code-block:: python

   from tapeforge.execution import EpisodeSpec, make_episode, twap, run_episode
   from tapeforge import generate, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=40, profile="uniform")
   offsets = windows.build_offsets(trades["timestamp"], 0, 40, DEFAULT_BUCKET_WIDTH_US)
   template = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=10, arrival_lookback=5)
   spec, market = make_episode(trades, offsets, 15, side=1, rho_bps=500, template=template)
   trajectory = run_episode(spec, market, twap)
   assert trajectory.last_result.terminated
   assert not trajectory.last_result.truncated

To use the same episode with Gymnasium after installing ``.[rl]``:

.. code-block:: text

   from tapeforge.execution.rl.gym_env import ExecutionEnv, ListPool

   env = ExecutionEnv(ListPool([("example", spec, market)]))
   observation, info = env.reset(seed=7)
   observation, reward, terminated, truncated, info = env.step([0.1])

``snapshot`` and ``restore`` carry an environment's running episode, exact
state and selection generator to another process, which is what lets the
trainer resume collection exactly. The PPO trainer below builds on this
environment; SAC and recurrent policies are not included.

Evaluating a policy
===================

``tapeforge.execution.rl.evaluation`` holds what a trainer needs around the
environment, and nothing of the trainer itself (``rl.ppo`` is the optional
trainer that uses it).

* ``load_training_config`` reads a strict TOML description
  (``tapeforge.execution.training-config/1``): the experiment configuration,
  the training role (always ``train``) and seeds, the reference baseline and
  seed for the frozen statistics, and the evaluation roles and seeds. The
  test role is refused unless ``include_test`` is set.
* ``training_statistics`` fits the statistics on training draws only;
  ``save_statistics`` and ``load_statistics`` freeze them to a JSON file
  tagged with its schema and the simulator version, and refuse files from
  another version.
* ``evaluation_manifest`` lists the fixed evaluation episodes of a run
  directory (the identifiers every baseline replayed) with the split record,
  data digests and a content digest; ``load_evaluation_manifest`` rejects an
  edited file.
* ``vector_policy`` adapts a callable over the learner's float32 observation
  vector to the runner's policy interface. Its output is validated, never
  clipped.
* ``evaluate_policy`` replays any policy on a run's episodes and writes
  records with the baseline schema into the run directory, so the report
  pairs it against the baselines on identical episode IDs.
* ``checkpoint_provenance`` returns the model-agnostic record a checkpoint
  must carry (configuration and its hash, split and data hashes, statistics,
  seed, action and reward conventions, simulator version, evaluation
  digest); ``validate_checkpoint_provenance`` checks it before an
  evaluation.

.. code-block:: python

   import tempfile

   import numpy as np

   from tapeforge.execution import evaluate_policy, run_experiment, vector_policy
   from tapeforge.execution.research.experiment import load_config, rebuild_splits

   config = load_config("experiments/configs/smoke-synthetic.toml")
   run_dir = run_experiment(config, output_dir=tempfile.mkdtemp())
   splits = rebuild_splits(config)
   policy = vector_policy(lambda vector: np.float32(0.1))  # a stand-in for a learned policy
   written = evaluate_policy(run_dir, policy, "constant-0.1", splits=splits)
   assert len(written) == 2  # one record per evaluated role and seed

Training a policy (PPO)
=======================

The trainer is optional (``pip install -e '.[train]'``) and thin:
Stable-Baselines3's PPO with separate MLP actor and critic, connected to
``ExecutionEnv``. Three strict TOML files describe a run, each rejecting
unknown keys:

* the experiment configuration (data, split, simulator settings);
* a training configuration (``training-config/1``): training pools,
  the frozen-statistics reference, the evaluation role and seeds;
* a PPO configuration (``ppo-config/1``): network, optimiser, rollout,
  update, schedule and model seeds. Every value is read from the file;
  nothing falls back to a library default.

.. code-block:: console

   $ python -m tapeforge.execution.rl.ppo train experiments/configs/smoke-synthetic-ppo.toml
   $ python -m tapeforge.execution.rl.ppo train \
       experiments/configs/pilot-futures-btcusdt-2023-12-ppo.toml \
       --evaluation-manifest runs/<pilot-run>/evaluation-manifest.json
   $ python -m tapeforge.execution.rl.ppo train <ppo.toml> --seed 0 \
       --resume runs/<training-run>/checkpoints/v00100 --total-steps 1920000
   $ python -m tapeforge.execution.rl.ppo evaluate runs/<pilot-run> \
       ppo-s0=runs/<training-run-seed0> ppo-s1=runs/<training-run-seed1>
   $ python -m tapeforge.execution.research.report runs/<pilot-run> --reference immediate
   $ python -m tapeforge.execution.research.replay runs/<pilot-run> \
       --episode validation/s1/i0 --policy ppo-s0 \
       --checkpoint runs/<training-run-seed0> --verify

The first command trains every ``model_seeds`` entry on synthetic data in a
few seconds and needs no download. ``evaluate`` takes a checkpoint directory
or a training run (it then uses the checkpoint selected on validation) and
writes records into the baseline run directory, so the report pairs each
model seed with the baselines on identical episode identifiers. Replay with
``--checkpoint`` steps a trained policy through one episode and, with
``--verify``, checks the result against the record ``evaluate`` wrote under
that policy name.

Contracts:

* **Objective.** ``gamma = 1.0``: the per-step rewards of a terminated
  episode sum to minus the implementation shortfall in basis points of the
  arrival notional (plus the risk term), a finite-horizon undiscounted cost.
  A smaller ``gamma`` discounts the terminal settlement relative to earlier
  fills and is a different objective; it must be written in the file to be
  used.
* **Actions.** The policy acts in ``[-1, 1]``; the environment receives
  ``(clip(a, -1, 1) + 1) / 2`` computed in float64. The rollout buffer keeps
  the unclipped Gaussian sample and its log-probability, so the update
  evaluates the distribution the action was drawn from. Evaluation applies
  the same map to the policy mean.
* **Observations.** Statistics are fitted once on training draws replayed by
  a named baseline, frozen, stored in every checkpoint and applied
  identically in training and evaluation. There is no running normaliser.
* **Episode ends.** ``terminated`` (target reached or horizon, settlement
  included) ends the return. ``truncated`` (the tape ran out) is passed to
  the learner as a time-limit truncation and bootstrapped; episodes cut by
  the split builder never truncate.
* **Seeds.** Model seeds (initialisation, action noise, minibatch order,
  per-environment episode selection) are separate from the sampling seeds of
  the training pools and of the fixed evaluation episodes.
* **Checkpoint selection.** Lowest validation mean shortfall over the
  checkpoints saved on the declared schedule; the earliest wins a tie. The
  test role is never consulted.

A training run directory holds ``config.json``, ``statistics.json``,
``rollouts.jsonl`` (per rollout: environment steps, wall time, the policy
version that collected it, episode reward, shortfall, settled fraction, and
the entropy, approximate KL, clip fraction, value loss and explained
variance of the update), ``evaluations.jsonl``, ``selection.json``,
``summary.json`` and ``checkpoints/v<k>/`` with ``model.zip`` (weights and
optimiser state), ``provenance.json`` (configuration, split and data hashes,
statistics, simulator version) and ``collection.pt`` (the collection
position and generator states).

Resuming: ``--resume`` continues a run from a checkpoint bit-identically to
the uninterrupted run on the ``serial`` and ``subprocess`` backends and on
the ``batched`` backend over CPU tensors, with the same library versions,
thread count and configuration (tested for all three). The batched backend
stores one snapshot of the whole vector (simulator state, accumulators,
pending observations, selection generator); CUDA generator states are stored
when CUDA is in use. On CUDA a resumed run continues the same schedule from
the same simulator and generator state, but the update's floating-point
reductions are not guaranteed deterministic, so it is not claimed to be
bit-identical there. Batched checkpoints written before the snapshot existed
can be evaluated but not resumed. Loading ``model.zip`` alone restores a
policy for evaluation; it is not a continuation of training.
``collection.pt`` is a pickle: load only checkpoints you wrote.

Batched simulator
=================

``tapeforge.execution.batched`` implements the reference transitions for
many independent episodes as int64 tensors on one device. Episode constants
(arrival price, terminal mark, settlement price, the int64 bound check) are
computed once by the reference functions when an ``EpisodeBatch`` is built;
``step`` and ``observe`` are tensor code that never reads a value back to
the host, except for the optional action-range check
(``step(..., validate=True)``).

.. code-block:: console

   $ python -m tapeforge.execution.parity --device cpu --episodes 5000 --out parity-cpu.json
   $ python -m tapeforge.execution.parity --device cuda --episodes 5000 --out parity-cuda.json

The parity driver feeds identical action sequences to both implementations
and compares the observation, every step-record field and every state field
after each transition, over seeded scenarios covering both sides, mixed
targets and horizons, latency and data delay, empty buckets, binding caps,
fees, both terminal rules, missing fill buckets and terminal marks, and
prices near the int64 accounting bound. It exits 2 when the device is
unavailable.

Limits: prices, bucket quantities and targets below 2**53 (so the float64
action conversion and observations equal the reference's); actions must be
float64 tensors; a settling episode whose settlement price would not be
positive is rejected at construction; stepping an ended environment is a
masked no-op; memory is 41 bytes per episode bucket.
``rollout.backend = "batched"`` trains through it. Because the trainer's
rollout buffer is host memory, that backend still copies actions,
observations, rewards and done flags between host and device every step.

Three optional PPO settings change how, not what, the trainer computes; all
are left out of the stored configuration at their defaults, so existing runs
and resume checks are unaffected:

* ``rollout.compile_simulator = true`` (batched backend only) compiles the
  device part of each vector step (action map, transition, rewards,
  accumulators, observations, episode summary, masked reset) with
  ``torch.compile(fullgraph=True, dynamic=False)``. Episode draws are taken
  before the compiled function in the eager order, so the selection
  generator is unchanged; a test checks that compiled and eager steps agree
  bit for bit on CPU tensors and on CUDA. The first step pays the compilation.
* ``schedule.evaluation_device = "cpu"`` scores checkpoints on validation
  with a copy of the policy on that device, for example during CUDA training.
  The reference simulator asks the policy for one action at a time, so on
  CUDA each decision is a small device round trip. Episodes, statistics,
  metric and selection rule are unchanged; the copy's float32 outputs may
  differ from the CUDA policy's in the last bits, so a selection score is not
  bit-identical to one computed on CUDA.
* ``schedule.lockstep_validation = true`` (any backend) replays the
  validation episodes together on the reference simulator and asks the policy
  once per decision round, for every episode still running, instead of once
  per decision (``runner.run_episodes_lockstep``). A profile of the
  per-decision replay showed about 85% of validation time inside the single
  observation policy call. Episodes, statistics, simulator, metric and
  selection rule are unchanged, and a batch policy that answers each
  observation as it would alone reproduces the per-decision trajectories
  exactly (tested); a network forward over several rows may round
  differently from a one-row forward, so a selection score is not guaranteed
  bit-identical. On the synthetic smoke benchmark on one laptop CPU
  (96,000 samples, five validation passes of 50 episodes, one run per
  variant, not a T4 measurement) validation fell from 4.5 to 1.1 s on the
  serial backend and from 4.7 to 1.0 s on the batched CPU backend, each
  variant pair selected the same checkpoint, and the selected scores differed
  by at most 5e-9 bps. The fixed-checkpoint replay of ``ppo_eval`` still
  uses the per-decision path.

**Verification status.** On CPU tensors the batched simulator matches the
reference on every compared transition. On CUDA it was executed on one Tesla
T4 (PyTorch 2.14.1, CUDA 13.0) in two runs at different source revisions:
the gpu-marked suite passed with nothing skipped, and ``--device cuda
--episodes 5000`` compared 26,696 transitions with no mismatch. The record,
with revisions and content hashes, is
``benchmarks/evidence/cuda-t4-2026-11-19.json``. Repeat both checks on any
other device or software stack and after any change to the batched
simulator; CPU-tensor agreement is not a substitute. Parity is agreement
between two implementations of the same synthetic rules: replaying recorded
market data does not model the market's response to the agent.

Robustness matrices
===================

.. code-block:: console

   $ python -m tapeforge.execution.research.matrix run \
       experiments/matrices/smoke-synthetic-one-factor.toml
   $ python -m tapeforge.execution.research.matrix run experiments/matrices/pilot-one-factor.toml
   $ python -m tapeforge.execution.rl.ppo evaluate runs/<matrix-dir>/<variant-run> \
       ppo-s0=runs/<training-run> --allow-setting-mismatch
   $ python -m tapeforge.execution.research.matrix summarize runs/<matrix-dir>

A matrix file names a base configuration and variants that each change
exactly one template field (fee, latency, data delay, terminal surcharge,
risk coefficient) or the split dates (a period). The pilot matrix keeps the
0 / 10 / 25 / 50 bps surcharge settings. The summary states per variant
whether it replays the base's episodes: a change of latency, data delay,
horizon or lookback alters the admissible starts, so its rows are unpaired.
A variant cannot use a date the base seals as test. A policy trained under
the base settings and evaluated under a variant
(``--allow-setting-mismatch``, recorded with the evaluation) was not
optimised for that variant; retraining per variant is a larger sweep that
has not been run. Changes to the observation features are a different kind
of experiment and are not expressed in a matrix.

Feature engine
==============

Each bucket is a half-open slice ``[offsets[b], offsets[b + 1])`` into sorted
records. Repeated offsets represent empty buckets. The feature outputs are:

* ``event_count``: aggregate-trade record count, int64.
* ``total_quantity``: quantity sum in integer ticks, int64.
* ``signed_quantity``: buy-initiated minus sell-initiated quantity, int64.
  ``buyer_is_maker=True`` denotes a sell-initiated trade.
* ``vwap``: quantity-weighted price in ordinary price units, float64.
* ``valid``: whether the bucket contains records, bool.

VWAP is zero when total quantity is zero. A bucket containing zero-quantity
records still has ``valid=True``. Integer results are exact within the
validated overflow bound; float VWAP comparisons use the contract tolerance.
These are trade features, not order-book imbalance features.

.. code-block:: python

   from tapeforge import generate, ops, windows
   from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US

   trades = generate.generate_trades(seed=7, n_buckets=100, profile="burst")
   offsets = windows.build_offsets(trades["timestamp"], 0, 100, DEFAULT_BUCKET_WIDTH_US)
   features = ops.aggregate(trades, offsets)
   assert features["event_count"].sum().item() == len(trades)

``ops.aggregate`` accepts a structured NumPy trade array with NumPy int64
offsets, or a mapping of contiguous column tensors with offsets on the same
device. CPU inputs use the reference; CUDA inputs use Triton. There is no
implicit cross-device fallback. ``reference.aggregate_reference`` is the
NumPy-only API.

The one-pass capacity is 4,096 records per bucket. Automatic dispatch uses
two-pass beyond that; an explicitly requested one-pass block size remains a
hard capacity limit. Prices and quantities are int64 and maker flags bool.
Validation checks shape, dtype, device, contiguity, offsets, non-negative
values, and conservative overflow bounds.

For unchanged CUDA inputs, ``prepared.prepare(columns, offsets)`` validates,
plans, and allocates once. ``execute()`` reuses output/scratch buffers;
``execute_owned()`` returns independent copies. Inputs are borrowed and must
not be mutated. One prepared object cannot be executed concurrently, and
cross-stream consumers must synchronize. There is no global cache.

For trades in host memory, ``pipelined.aggregate_host(trades, offsets,
batch_records=..., slots=...)`` (or ``OverlappedAggregator``, which separates
setup from ``run()``) returns the same features while overlapping transfers
with computation. It cuts the buckets into batches of whole buckets of at
most ``batch_records`` records (a longer bucket forms a batch alone), copies
each batch through its slot's pinned buffers on a copy-in stream, reduces it
with the one-pass or two-pass kernel on a compute stream and copies the
result back on a copy-out stream; CUDA events order the three, so batch
``N + 1`` is copied in while batch ``N`` is reduced. Each of the ``slots``
buffer sets is reused only after the host has waited for the copy that last
read or wrote it, and the host reads results only after their copy-out
event. Validation runs once on the host, and two-pass chunk schedules are
built with NumPy, so the loop reads nothing back from the device to plan.
``trace=True`` records a CUDA-event interval per stage and batch;
``pipelined.overlap_ms`` measures how long two stages ran concurrently. The
pinned and device memory held is bounded by ``slots`` times one batch.

``StreamingAggregator.push`` accepts timestamp-ordered chunks and emits
closed buckets; ``finalize`` emits remaining buckets. It preserves batch
feature semantics, rejects decreasing timestamps, and does not deduplicate
IDs or buffer late arrivals. It is an offline replay/reference component;
no live collector is provided.

Checks and GPU verification
===========================

.. code-block:: console

   $ python -m ruff check src tests benchmarks tools
   $ python -m pytest -q
   $ python -m pytest -q --cov=tapeforge --cov-report=term-missing
   $ python -m pytest -q tests/test_execution/test_gym_env.py tests/test_execution/test_end_to_end.py
   $ python -m mypy

CUDA tests skip with a capability reason on CPU machines. Tests using
optional downloaded archives name missing files when skipped. The default
suite includes slow tests. README Python examples execute in the docs tests.

To verify the current working tree on a CUDA machine:

.. code-block:: console

   $ python tools/verify_snapshot.py pack --out /tmp/tapeforge-snapshot.tgz

Transfer the archive, then run on that machine:

.. code-block:: console

   $ tar xzf tapeforge-snapshot.tgz
   $ cd TapeForge
   $ python -m pip install -e '.[dev,train]'   # use the runtime's CUDA torch and Triton
   $ python tools/verify_snapshot.py run --artifacts verification --record --zip
   $ python tools/verify_snapshot.py run --artifacts verification \
       --steps gpu_suite simulator_parity training_benchmark_quick --zip

The snapshot includes tracked and eligible untracked files, excluding ignored
files, and records a content digest. The second ``run`` line is the bounded
subset for the batched simulator: the GPU test suite, the CUDA parity
comparison and a quick training benchmark. The runner saves step logs, environment,
JUnit XML, status manifest, and a results copy. Download ``verification.zip``
and recorded benchmark results before disposing of the remote environment.

Benchmarks
==========

.. code-block:: console

   $ python benchmarks/kernel.py --sizes 100000 --seeds 1 --no-record
   $ python benchmarks/kernel.py --timing cuda_events --no-record
   $ python benchmarks/kernel.py
   $ python benchmarks/pipeline.py
   $ python benchmarks/training.py --config benchmarks/configs/training_smoke.toml --no-record
   $ python benchmarks/training.py --config benchmarks/configs/training_pilot.toml \
       --out runs/benchmarks/training-pilot.json
   $ python benchmarks/training.py --config benchmarks/configs/training_smoke.toml \
       --device cuda --no-record

``kernel.py`` compares the checked API, prepared execution, individual passes,
forced two-pass execution, eager/compiled PyTorch, and the CPU reference.
Stage rows separate validation, metadata, allocation, and preparation; these
overlap with end-to-end rows and must not be added to them.
``pipeline.py`` measures host/device transfers and compute using pageable or
pinned memory, whole-input synchronous variants, ``batched_sync`` (the
pipelined batches with one slot, so nothing overlaps) and ``pinned_streams``
(``pipelined.OverlappedAggregator`` with ``slots`` buffer sets). The
overlapped rows also report setup time, the bytes of pinned and device memory
held, and, from one traced run, the time copy-in, compute and copy-out each
spent running concurrently with another stage.

``scale.py`` checks one synthetic workload per profile at a requested record
count (48 million by default; requested and generated counts are both
recorded): it estimates host and device memory first and skips a workload
that would not fit, then runs every path in a fresh process against the CPU
reference -- exact integer features, VWAP within tolerance, independent sum
invariants, the overflow refusal -- and records peak allocated and reserved
device memory, peak host RSS, preparation time and execution percentiles.
``cases.py`` generates seeded adversarial inputs (a *case* is one
``(trades, offsets)`` pair from one seed; lengths concentrate on tile,
one-pass-limit and chunk boundaries, with empty runs, clipped windows, zero
quantities and values using up to the whole int64 bound) and checks every
path against the plain-loop oracle, listing the seeds of any failure.

``training.py`` backend entries take options after ``+``:
``batched+compile`` sets ``rollout.compile_simulator``, ``+cpu_eval``
sets ``schedule.evaluation_device = "cpu"`` and ``+lockstep`` (any backend,
for example ``serial+lockstep``) sets ``schedule.lockstep_validation``; with ``core_compiled = true``
the batched-core section also times the compiled vector step.

``training.py`` (needs the ``train`` extra) compares serial, subprocess and
batched environments under one PPO configuration: same simulator settings,
episode pools, policy, rollout size, sample budget and evaluation protocol.
Its sections are recorded separately and must not be summed: per-episode
feature preparation from precomputed tapes against on-demand aggregation;
environment stepping alone; the batched simulator alone across batch sizes;
policy inference and the PPO update; and complete training runs in fresh
processes with data loading, setup, collection, optimisation, evaluation and
checkpointing timed apart, memory, CPU time over wall time, environment
steps per second (one PPO sample per step) and the training time at which a
predeclared validation threshold was first met, or that it never was. The
batched rows are timed only after the parity comparison passes on the
benchmark device, and an unavailable device yields ``skipped`` rows. Known
differences between backends are stored with every record: the batched
backend draws the same episode distribution in a different sequence and
pays per-step host/device copies. One synthetic run on a T4 host (96,000
samples, eight environments, one repeat) is summarised in
``benchmarks/evidence/cuda-t4-2026-11-19.json``: 56.2 s end to end for
serial CPU, 72.4 s for subprocess CPU and 129.4 s for batched CUDA, whose
collection took 66.8 s against 15.8 s serial. The device-only batched core
costs about 3 to 4 ms per vector step at every batch size from 8 to 4,096,
so at eight environments a near-constant per-step device cost dominates.
That is not a speedup, and larger batches were not trained.

A second synthetic run on a T4 host (same experiment, configuration, seed and
sample budget; ``benchmarks/evidence/t4-optimisation-2026-11-21.json``)
measured the optional settings. Compiling the vector step brought the
device-only batched core from 2.95 ms to 0.20 ms per vector step at eight
environments (3.06 ms to 0.21 ms at 4,096), and environment stepping with
host copies from 4.41 ms to 0.57 ms. End to end, batched CUDA took 139.8 s
eager, 82.2 s compiled, 122.2 s with CPU-policy validation and 65.7 s with
both (collection 70.3 s to 24.5 s, evaluation 55.9 s to 27.4 s); serial CPU
took 53.9 s. Every batched variant selected the same checkpoint, and the
CPU-copy validation score differed from the CUDA one by 2.8e-9 bps. One
repeat per variant; differences of a few seconds are within run-to-run
variation.

Transfer scope, same evidence file: for 10 million synthetic records
(uniform profile) in host memory, ``pinned_streams`` (1,048,576-record
batches, three slots) returned features in 30.2 ms against 35.2 ms for the
best whole-input variant (``pinned_sync``) and 46.7 ms for the same batches
without overlap; the event trace shows 4.1 of 4.4 ms of compute running
concurrently with copy-in, so the remaining time is the 27.6 ms host-to-device
copy. Setup (pinned allocation and planning, about 150 ms) is excluded and
reported separately; at one million records the input is one batch and
nothing overlaps. ``scale.py`` checked 48.0, 53.3 and 48.0 million records
(uniform, burst, sparse with 96 million buckets) on every path: exact
integer features, VWAP within tolerance, invariants and the overflow
refusal held. ``torch.compile`` of the ``index_add`` baseline traced one
graph with no breaks yet ran 3.8 to 4.8 times slower than eager on that GPU
(1 and 10 million records). The cause, from a per-kernel profile: inductor
lowers int64 ``index_add`` to ``index_put_(accumulate=True)``, a radix sort
and sorted reduction (15.8 ms for one 10M-record call against 0.75 ms
eager), so the eager baseline is the one compared against.

A third T4 session (``benchmarks/evidence/t4-optimisation-2026-11-22.json``,
synthetic data and synthetic checkpoints) checked the optional settings
above. Lockstep validation replayed five fixed checkpoints (50 episodes
each): with a batch policy that calls the scalar policy per row, the
trajectories equal the per-decision replay exactly; with the batched forward,
actions differ by at most 6e-8 and the validation metric by at most 3.2e-9
bps, and the same checkpoint is selected on CPU and CUDA policies. A
validation pass took 0.9 to 1.3 s instead of 7.0 to 8.0 s (CPU policy) or
12.5 to 13.5 s (CUDA policy), two timed passes per checkpoint. End to end (96,000 samples, one repeat),
serial CPU took 59.7 s without lockstep and 30.9 s with it (evaluation 32.8
to 4.5 s); batched CUDA with the compiled step and CPU-policy validation
took 78.0 s and 42.4 s. Each backend selected the same checkpoint with and
without lockstep. A collection profile at eight environments puts the
compiled batched rollout at 2.20 ms per vector step against 1.63 ms serial,
with the device environment step alone (0.96 ms) no faster than the CPU one
(0.87 ms), so collection is not faster on this GPU at that batch size.
``int32_transfer=True`` (bounds checked per window, widened on the device,
bit-identical) cut 10 million uniform records from 30.0 to 16.8 ms in
``pinned_streams`` and one million from 4.6 to 2.8 ms; at 100,000 records
it saved under 0.1 ms. It costs device memory for the int64 widening
buffers (77 against 53 MB at 10 million records).

Correctness is checked before timing. Warm-up excludes compilation. Results
record wall-clock or CUDA-event timing, workload distribution, setup included
in each region, hardware, software, revision, dirty state, and snapshot hash
when available. Preparation costs must be reported separately when comparing
repeated prepared execution with a checked API. Record tuning sweeps in
``benchmarks/results.json`` before publishing performance claims.

Repository layout
=================

.. code-block:: text

   src/tapeforge/
     contract.py, validation.py       units, bounds, input checks
     ingest.py, generate.py           historical and synthetic events
     windows.py                      packed bucket offsets
     reference.py, streaming.py      batch and incremental CPU features
     kernels.py, prepared.py, ops.py  GPU kernels and public feature API
     pipelined.py                    batched host-to-device aggregation on overlapping streams
     execution/
       spec.py, state.py             execution rules and data structures
       simulator.py                  fills, accounting and state transitions
       batched.py, parity.py         the same transitions as device tensors; parity driver
       baselines.py, runner.py        reference policies and episode replay
       data/
         datasets.py                 archive acquisition and checksums
         days.py, episodes.py        bucket tapes and episode construction
         splits.py, sampling.py      chronological partitions and seeded tasks
       research/
         experiment.py               configured runs and artifact manifests
         report.py, replay.py        result summaries and episode inspection
         sensitivity.py              terminal-rule comparisons
         matrix.py                   one-factor robustness matrices
       rl/
         normalization.py            training statistics and versioned files
         evaluation.py               fixed evaluation sets and provenance
         gym_env.py, env_check.py     optional Gymnasium interface and checker
         ppo_config.py               strict trainer configuration
         ppo_env.py, ppo.py          action map, training environments, PPO training
         ppo_eval.py                 checkpoint loading and evaluation
         batched_env.py              vector environment over the batched simulator
   experiments/configs/              smoke and pilot experiment, training and PPO configs
   experiments/matrices/             one-factor robustness matrices
   benchmarks/                       kernel, transfer and training-pipeline timings
   tests/                            CPU/GPU parity, accounting, leakage, tooling
   tools/                            parity summary and snapshot verification
   data/, runs/                      local datasets and outputs; Git-ignored

Execution modules are grouped by the workflow they support: ``data`` prepares
inputs, ``research`` runs and inspects experiments, and ``rl`` connects trainers
to the simulator. The core simulation files stay in ``execution/``.
``CONTRIBUTING.rst`` lists each component's responsibilities.
Public imports such as ``from tapeforge.execution import EpisodeSpec, run_policy``
are unchanged. Module-specific imports and CLI commands use the subpackages,
for example ``python -m tapeforge.execution.research.experiment``. Update local
scripts using the former flat module paths; artifact schemas are unchanged.

Development guidelines
======================

Write comments, docstrings, errors, and public documentation for developers
maintaining the code. Explain behavior, units, invariants, algorithm choices,
and non-obvious constraints. Do not label source comments or public docs by
curriculum weeks, lesson milestones, or classroom instructions. Describe
unfinished work by the missing capability and acceptance criteria. Keep
personal learning schedules and exercises under ignored ``planning/``.
These conventions apply to human and AI-generated changes alike.

Distinguish intended behavior from implemented and verified behavior. Report
CPU tests, interpreter checks, and actual CUDA execution separately. Support
performance claims with reproducible measurements and environment metadata.
Preserve exact reductions within the data contract and verify correctness
against an independent reference before benchmarking.

Documentation policy
====================

``README.rst`` is the public documentation and ``CONTRIBUTING.rst`` the
contributor guide. Markdown, private design notes,
and learning plans remain local and ignored. Shared results use JSON, CSV,
or reStructuredText. Downloaded market data and run artifacts are not committed.

License
=======

MIT.
