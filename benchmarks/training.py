"""Scope 3: the training pipeline, stage by stage and end to end.

One configuration (``benchmarks/configs/training_*.toml``) names a PPO
configuration and the backends to compare. The simulator rules, episode
pools, policy architecture, rollout size, sample budget and evaluation
protocol are those of the PPO configuration for every backend; only how the
environments are stepped changes:

    serial      one process steps the environments in turn (CPU reference)
    subprocess  one worker process per environment (CPU reference)
    batched     the tensor simulator on ``device`` (``execution.batched``)

Sections, each recorded as separate entries that must not be summed:

    features     Building one episode's market slice from precomputed day
                 tapes against aggregating its window from raw trades on
                 demand. The one-time cost of precomputing a day is reported
                 beside the per-episode costs.
    env_step     Environment stepping alone through the trainer's vector
                 interface, with pre-drawn actions and no policy.
    batched_core The tensor simulator alone at several batch sizes, actions
                 drawn on the device and nothing copied to the host. This is
                 simulator throughput, not training throughput.
    policy       One policy forward for a batch of observations, and one
                 PPO update on a full rollout.
    end_to_end   A complete training run per backend in a fresh process:
                 data loading, statistics, environment and model setup,
                 collection, optimisation, evaluation and checkpointing
                 timed apart; environment steps per second (one PPO sample
                 per step); peak memory; CPU time over wall time; and the
                 training time at which the predeclared validation threshold
                 was first met, or that it never was.

Unavoidable differences between backends, recorded with every result:

    * The batched backend draws episodes from the union of the pools with a
      device generator, so its episode sequence differs from the CPU
      backends' for the same seed; the distribution is the same.
    * The trainer's rollout buffer is host memory, so the batched backend
      copies actions to the device and observations, rewards and done flags
      back on every step. ``batched_core`` shows the simulator without them.
    * Runs with different episode sequences reach different policies, so
      time-to-threshold compares stochastic outcomes, one seed each unless
      more are configured.

Correctness is checked before timing: the on-demand slice must equal the
precomputed one exactly, and the batched simulator must pass the parity
comparison on the benchmark device, otherwise its rows are recorded as
``incorrect`` and not timed. A device that is not available produces
``skipped`` rows; nothing is estimated in its place.

Usage:
    python benchmarks/training.py --config benchmarks/configs/training_smoke.toml --no-record
    python benchmarks/training.py --config benchmarks/configs/training_pilot.toml \\
        --out runs/benchmarks/training-pilot.json
    python benchmarks/training.py --config benchmarks/configs/training_smoke.toml \\
        --device cuda --no-record

Without ``--no-record`` or ``--out`` the entries are appended to
``benchmarks/results.json``. Needs the ``train`` extra.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from harness import RESULTS_PATH, record
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from tapeforge import generate, ingest, windows
from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US
from tapeforge.execution import batched, parity
from tapeforge.execution.data.datasets import ArchiveSpec
from tapeforge.execution.data.days import BUCKETS_PER_DAY, utc_day_start_us
from tapeforge.execution.data.episodes import build_market_slice
from tapeforge.execution.data.sampling import materialize, sample_episodes
from tapeforge.execution.rl.ppo import TrainingData, prepare_training, train
from tapeforge.execution.rl.ppo_config import PPOConfig, load_ppo_config
from tapeforge.execution.rl.ppo_env import make_vec_env

SCOPE = "train"
PARITY_EPISODES = 1_000
UNAVOIDABLE_DIFFERENCES = (
    "batched: episodes drawn from the union of the pools with a device generator; same "
    "distribution, different sequence",
    "batched: per-step host/device copies because the trainer's rollout buffer is host memory",
    "time-to-threshold compares stochastic training outcomes, one seed per backend",
)


def _max_rss_bytes(who: int) -> int:
    """Return the peak resident set size in bytes (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(who).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _cpu_seconds() -> float:
    """Return user plus system CPU time of this process and its finished children."""
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def _device_available(device: str) -> str | None:
    """Return ``None`` when float64 tensors can be created on ``device``, else the reason."""
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
    except (RuntimeError, AssertionError, TypeError) as exc:
        return str(exc).splitlines()[0]
    return None


def _synchronize(device: str) -> None:
    """Wait for queued CUDA work so a wall-clock reading covers it."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# --- features ------------------------------------------------------------------------------


def _first_training_day_trades(data: TrainingData) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the raw trades, bucket offsets and date of the first training day."""
    experiment = data.experiment
    day = experiment.split.train[0]
    if experiment.data.source == "synthetic":
        seed = {d.date: d.seed for d in experiment.data.synthetic_days}[day]
        n_buckets = experiment.data.synthetic_n_buckets
        trades = generate.generate_trades(
            seed=seed, n_buckets=n_buckets, start_us=utc_day_start_us(day),
            profile=experiment.data.synthetic_profile,  # type: ignore[arg-type]
        )
    else:
        spec = ArchiveSpec(experiment.data.market, experiment.data.symbol, day)
        trades, _ = ingest.load_agg_trades(spec.local_path(experiment.data.data_dir))
        n_buckets = BUCKETS_PER_DAY
    offsets = windows.build_offsets(
        trades["timestamp"], utc_day_start_us(day), n_buckets, DEFAULT_BUCKET_WIDTH_US
    )
    return trades, offsets, day


def measure_features(data: TrainingData, n_episodes: int) -> dict[str, Any]:
    """Time per-episode slice construction from precomputed tapes and on demand from trades.

    Both paths build the slices of the same ``n_episodes`` training draws
    that start on the first training day and far enough from its end; the
    on-demand slices must equal the precomputed ones exactly before either
    is timed.
    """
    started = time.perf_counter()
    trades, offsets, day = _first_training_day_trades(data)
    load_seconds = time.perf_counter() - started
    record_ = data.splits.record
    n_before, n_after = record_.n_before, record_.n_after
    n_buckets = offsets.size - 1
    draws = [
        d for d in sample_episodes(record_, "train", 20 * n_episodes, 12345,
                                   data.experiment.rho_choices_bps)
        if d.date == day and d.segment == record_.segments_for("train")[0].index
        and d.start_bucket + n_after <= n_buckets
    ][:n_episodes]
    if not draws:
        return {"status": "skipped", "reason": f"no training draw starts on {day}"}

    def precomputed(draw: Any) -> Any:
        return materialize(data.splits, draw, data.experiment.template)[1]

    def on_demand(draw: Any) -> Any:
        first = draw.start_bucket - n_before
        window = offsets[first : draw.start_bucket + n_after + 1]
        return build_market_slice(trades, window, n_before, n_before, n_after)

    for draw in draws:
        a, b = precomputed(draw), on_demand(draw)
        for name in ("event_count", "total_quantity", "signed_quantity", "vwap_ticks", "valid"):
            if not np.array_equal(getattr(a, name), getattr(b, name)):
                return {"status": "incorrect", "reason": f"{name} differs for draw {draw.index}"}

    timings = {}
    for name, build in (("precomputed", precomputed), ("on_demand", on_demand)):
        started = time.perf_counter()
        for draw in draws:
            build(draw)
        timings[name] = (time.perf_counter() - started) / len(draws)
    started = time.perf_counter()
    build_market_slice(trades, offsets, 0, 0, n_buckets)      # the whole day, as a tape build does
    precompute_seconds = time.perf_counter() - started
    return {
        "status": "ok",
        "day": day,
        "n_episodes": len(draws),
        "n_trades": int(trades.size),
        "slice_buckets": n_before + n_after,
        "precomputed_ms_per_episode": timings["precomputed"] * 1e3,
        "on_demand_ms_per_episode": timings["on_demand"] * 1e3,
        "precompute_day_tape_seconds": precompute_seconds,
        "load_and_index_trades_seconds": load_seconds,
        "all_days_load_seconds": data.data_seconds,
        "note": (
            "on_demand aggregates only the episode's window of buckets with the NumPy "
            "reference; the trades must already be parsed and indexed "
            "(load_and_index_trades_seconds). precomputed cuts the window from a day tape "
            "that costs precompute_day_tape_seconds once per day"
        ),
    }


# --- stepping, policy and update ----------------------------------------------------------------


def _variant(config: PPOConfig, backend: str, device: str) -> PPOConfig:
    """Return ``config`` with the backend replaced; the batched backend also takes the device.

    A backend entry may carry options after ``+``: ``batched+compile`` sets
    ``compile_simulator``, ``+cpu_eval`` sets ``evaluation_device = "cpu"``.
    Everything else in the PPO configuration stays as written.
    """
    name, *options = backend.split("+")
    unknown = set(options) - {"compile", "cpu_eval"}
    if unknown or (options and name != "batched"):
        raise ValueError(f"unknown backend variant {backend!r}")
    return replace(
        config, backend=name, device=device if name == "batched" else "cpu",
        compile_simulator="compile" in options,
        evaluation_device="cpu" if "cpu_eval" in options else None,
    )


def measure_env_stepping(config: PPOConfig, data: TrainingData, steps: int) -> dict[str, Any]:
    """Time ``steps`` vector steps of pre-drawn actions with no policy in the loop."""
    started = time.perf_counter()
    vec_env = make_vec_env(config, data.experiment, data.training, data.splits, data.statistics, 0)
    setup = time.perf_counter() - started
    try:
        vec_env.reset()
        actions = np.random.default_rng(0).uniform(-1, 1, size=(steps, config.n_envs, 1))
        actions = actions.astype(np.float32)
        for index in range(min(50, steps)):          # warm-up, not timed
            vec_env.step(actions[index])
        _synchronize(config.device)
        started = time.perf_counter()
        for index in range(steps):
            vec_env.step(actions[index])
        _synchronize(config.device)
        elapsed = time.perf_counter() - started
    finally:
        vec_env.close()
    return {
        "status": "ok",
        "n_envs": config.n_envs,
        "vector_steps": steps,
        "setup_seconds": setup,
        "seconds": elapsed,
        "env_steps_per_second": steps * config.n_envs / elapsed,
        "microseconds_per_vector_step": elapsed / steps * 1e6,
    }


def measure_batched_core(
    data: TrainingData, config: PPOConfig, device: str, batch_sizes: list[int], steps: int,
    compile_rows: bool = False,
) -> list[dict[str, Any]]:
    """Time the tensor simulator alone: device-drawn actions, automatic resets, no host copies."""
    from tapeforge.execution.rl.gym_env import SplitPool

    pool = SplitPool(data.splits, "train", data.experiment.template, data.training.train_seeds[0],
                     min(config.pool_episodes, 512), data.experiment.rho_choices_bps)
    episodes = [(pool[i].spec, pool[i].market) for i in range(len(pool))]
    started = time.perf_counter()
    batch = batched.EpisodeBatch.from_episodes(episodes, device)
    build_seconds = time.perf_counter() - started
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    def vector_step(state: batched.BatchedState, actions: torch.Tensor,
                    draws: torch.Tensor) -> batched.BatchedState:
        batched.observe(state, batch)
        state, result = batched.step(state, actions, batch, validate=False)
        return batched.reset_where(state, result.terminated | result.truncated, draws, batch)

    # One compiled function per batch size (static shapes), traced during warm-up.
    compiled_step = torch.compile(vector_step, fullgraph=True, dynamic=False)
    rows = []
    for size in batch_sizes:
        def draw(size: int = size) -> torch.Tensor:
            return torch.randint(batch.n_episodes, (size,), generator=generator, device=device)

        def run(n: int, state: batched.BatchedState, size: int = size,
                core: Any = vector_step) -> batched.BatchedState:
            for _ in range(n):
                actions = torch.rand(size, dtype=torch.float64, generator=generator,
                                     device=device)
                state = core(state, actions, draw())
            return state

        for compiled in (False, True) if compile_rows else (False,):
            core = compiled_step if compiled else vector_step
            started = time.perf_counter()
            state = run(20, batched.reset(batch, draw()), core=core)  # warm-up, not timed
            _synchronize(device)
            warmup_seconds = time.perf_counter() - started
            started = time.perf_counter()
            run(steps, state, core=core)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            rows.append({
                "status": "ok", "batch_size": size, "vector_steps": steps, "seconds": elapsed,
                "compiled": compiled, "warmup_seconds_including_compile": warmup_seconds,
                "env_steps_per_second": steps * size / elapsed,
                "microseconds_per_vector_step": elapsed / steps * 1e6,
                "bank_episodes": batch.n_episodes, "bank_bytes": batch.nbytes,
                "bank_build_seconds": build_seconds,
                "includes": "observe + step + masked reset per vector step; no host transfer"
                + ("; torch.compile(fullgraph, static shapes), compiled during warm-up"
                   if compiled else ""),
            })
    return rows


def measure_policy(config: PPOConfig, data: TrainingData) -> dict[str, Any]:
    """Time one policy forward at the collection batch size and one PPO update on a rollout."""
    set_random_seed(0)
    torch.set_num_threads(config.torch_threads)
    vec_env = make_vec_env(replace(config, backend="serial", device="cpu"), data.experiment,
                           data.training, data.splits, data.statistics, 0)
    try:
        model = PPO(
            "MlpPolicy", vec_env, learning_rate=config.learning_rate, n_steps=config.n_steps,
            batch_size=config.batch_size, n_epochs=config.n_epochs, gamma=config.gamma,
            gae_lambda=config.gae_lambda, clip_range=config.clip_range,
            ent_coef=config.entropy_coef, vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm, device=config.device, seed=None,
            policy_kwargs={"net_arch": {"pi": list(config.hidden_sizes),
                                        "vf": list(config.hidden_sizes)},
                           "log_std_init": config.log_std_init},
        )
        model.learn(total_timesteps=config.rollout_steps)    # fills the buffer; one update
        # The update flattens the buffer to [steps * envs, fields]; take one collection batch.
        stored = model.rollout_buffer.observations.reshape(-1, model.observation_space.shape[0])
        observations = torch.as_tensor(stored[: config.n_envs], device=model.device)
        repeats = 2_000
        with torch.no_grad():
            for _ in range(100):
                model.policy(observations)
            _synchronize(config.device)
            started = time.perf_counter()
            for _ in range(repeats):
                model.policy(observations)
            _synchronize(config.device)
            forward = (time.perf_counter() - started) / repeats
        started = time.perf_counter()
        model.train()
        _synchronize(config.device)
        update = time.perf_counter() - started
    finally:
        vec_env.close()
    return {
        "status": "ok",
        "policy_device": config.device,
        "batch": config.n_envs,
        "forward_microseconds": forward * 1e6,
        "update_seconds_per_rollout": update,
        "update_minibatches": config.n_epochs * config.rollout_steps // config.batch_size,
        "samples_per_update_second": config.rollout_steps * config.n_epochs / update,
    }


# --- end to end -----------------------------------------------------------------------------


def time_to_threshold(run_dir: Path, metric: str, maximum: float) -> dict[str, Any]:
    """Return when a run's validation metric first met the threshold, or that it never did."""
    rows = [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text().splitlines()]
    overhead = 0.0
    for row in rows:
        overhead += row["evaluate_seconds"]
        if row[metric] <= maximum:
            return {
                "reached": True,
                "policy_version": row["policy_version"],
                "env_steps": row["env_steps"],
                "train_seconds": row["train_seconds"],
                "train_plus_evaluation_seconds": row["train_seconds"] + overhead,
                "value": row[metric],
            }
    return {"reached": False, "best_value": min(r[metric] for r in rows),
            "evaluations": len(rows), "env_steps": rows[-1]["env_steps"]}


def run_end_to_end(config: PPOConfig, benchmark: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Train one backend in this process and return its timing, memory and threshold record."""
    wall_started, cpu_started = time.perf_counter(), _cpu_seconds()
    data = prepare_training(config)
    run_dir = train(config, int(benchmark["model_seed"]), data=data, output_dir=out_dir,
                    total_steps=int(benchmark["total_steps"]))
    wall = time.perf_counter() - wall_started
    summary = json.loads((run_dir / "summary.json").read_text())
    seconds = summary["seconds"]
    threshold = benchmark["threshold"]
    metrics: dict[str, Any] = {
        "status": "ok",
        "backend": config.backend,
        "device": config.device,
        "compile_simulator": config.compile_simulator,
        "evaluation_device": config.evaluation_device or config.device,
        "n_envs": config.n_envs,
        "env_steps": summary["env_steps"],
        "seconds": {k: seconds[k] for k in ("data_load", "statistics", "setup", "collect",
                                            "update", "evaluate", "checkpoint", "train")},
        "process_wall_seconds": wall,
        "env_steps_per_collect_second": summary["env_steps_per_collect_second"],
        "env_steps_per_train_second": summary["env_steps_per_train_second"],
        "samples_per_train_second": summary["env_steps_per_train_second"],
        "cpu_seconds_over_wall": (_cpu_seconds() - cpu_started) / wall,
        "max_rss_bytes_main": _max_rss_bytes(resource.RUSAGE_SELF),
        "max_rss_bytes_largest_child": _max_rss_bytes(resource.RUSAGE_CHILDREN),
        "threshold": {**threshold, **time_to_threshold(run_dir, threshold["metric"],
                                                        float(threshold["max"]))},
        "selected": summary["selected"],
        "run_dir": str(run_dir),
    }
    if torch.device(config.device).type == "cuda":
        metrics["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    else:
        metrics["gpu_utilization"] = "not measured (no CUDA device in this run)"
    return metrics


# --- orchestration ---------------------------------------------------------------------------


def _gate_batched(device: str) -> dict[str, Any] | None:
    """Return ``None`` when the batched simulator may be timed on ``device``, else the row."""
    reason = _device_available(device)
    if reason is not None:
        return {"status": "skipped", "reason": f"device {device!r} unavailable: {reason}"}
    episodes = parity.random_episodes(PARITY_EPISODES, 0)
    report = parity.compare(episodes, parity.random_actions(episodes, 1), device)
    if not report.ok:
        return {"status": "incorrect", "reason": report.mismatches[:3],
                "n_mismatches": report.n_mismatches}
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the configured sections and record or print one entry per measurement.

    Returns:
        0 when every requested row was measured or explicitly skipped; 1 when
        a correctness gate failed or a child run crashed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-record", action="store_true", help="print only")
    parser.add_argument("--out", default=None, help="record to this file instead of results.json")
    parser.add_argument("--device", default=None,
                        help="override the config's device for the batched backend (e.g. cuda)")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    benchmark = tomllib.loads(Path(args.config).read_text())
    base = load_ppo_config(benchmark["ppo_config"])
    if args.device is not None:
        benchmark["device"] = args.device
    device = str(benchmark["device"])
    run_root = Path(benchmark.get("output_dir", "runs/benchmarks")) / benchmark["name"]

    if args.child is not None:      # one end-to-end variant, isolated for memory accounting
        metrics = run_end_to_end(_variant(base, args.child, device), benchmark, run_root)
        Path(args.child_out).write_text(json.dumps(metrics))
        return 0

    entries: list[tuple[str, dict[str, Any]]] = []
    failed = False
    gate = (_gate_batched(device)
            if any(b.startswith("batched") for b in benchmark["backends"]) else None)
    data = prepare_training(base)
    torch.set_num_threads(base.torch_threads)
    sections = benchmark["sections"]
    if "features" in sections:
        entries.append(("features", measure_features(data, int(benchmark["feature_episodes"]))))
    if "env_step" in sections:
        for backend in benchmark["backends"]:
            if backend.startswith("batched") and gate is not None:
                row = gate
            else:
                row = measure_env_stepping(_variant(base, backend, device), data,
                                           int(benchmark["env_step_vector_steps"]))
            entries.append((f"env_step/{backend}", row))
    if "batched_core" in sections:
        if gate is not None:
            entries.append(("batched_core", gate))
        else:
            for row in measure_batched_core(data, base, device,
                                            [int(b) for b in benchmark["core_batch_sizes"]],
                                            int(benchmark["core_vector_steps"]),
                                            bool(benchmark.get("core_compiled", False))):
                prefix = "batched_core_compiled" if row["compiled"] else "batched_core"
                entries.append((f"{prefix}/b{row['batch_size']}", row))
    if "policy" in sections:
        entries.append(("policy", measure_policy(base, data)))
    if "end_to_end" in sections:
        for backend in benchmark["backends"]:
            if backend.startswith("batched") and gate is not None:
                entries.append((f"end_to_end/{backend}", gate))
                continue
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "metrics.json"
                child = subprocess.run(
                    [sys.executable, __file__, "--config", args.config, "--device", device,
                     "--child", backend, "--child-out", str(out)], check=False,
                )
                if child.returncode != 0 or not out.exists():
                    entries.append((f"end_to_end/{backend}",
                                    {"status": "error", "returncode": child.returncode}))
                    failed = True
                else:
                    entries.append((f"end_to_end/{backend}", json.loads(out.read_text())))

    run_config = {
        "benchmark": benchmark,
        "ppo": base.to_json(),
        "experiment": data.experiment.name,
        "unavoidable_differences": list(UNAVOIDABLE_DIFFERENCES),
        "timing": "time.perf_counter around each region; CUDA synchronised at region ends",
    }
    path = Path(args.out) if args.out else RESULTS_PATH
    if args.out:
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, metrics in entries:
        failed = failed or metrics.get("status") == "incorrect"
        if not args.no_record:
            record(SCOPE, name, metrics, run_config, path=path)
        print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
