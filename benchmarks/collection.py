"""Where PPO collection time goes, per backend: policy, environment, buffer, transfers.

``model.collect_rollouts`` (stable-baselines3) is the trainer's collection
loop. For each backend variant (``training.py`` syntax, e.g.
``batched+compile``) this builds the trainer's environments and model
(``ppo.build_model``) for one seed and then:

1. collects one warm-up rollout (compilation, CUDA context, allocator),
   timed separately;
2. collects ``--rollouts`` rollouts uninstrumented: the collection time;
3. collects ``--rollouts`` more with each component wrapped in a timer
   that synchronises the device at both ends -- ``obs_as_tensor`` (host to
   device), the policy forward (``policy.forward``), the environment's
   ``step_async`` and ``step_wait``, ``predict_values`` (time-limit
   bootstraps and the final value) and ``rollout_buffer.add`` (which
   copies values and log-probabilities to the host). The remainder
   (``other``) is the loop itself: the action copy to the host, clipping,
   callbacks and per-environment bookkeeping;
4. on CUDA, profiles ``--profile-steps`` vector steps with ``torch.profiler``
   and counts kernel launches, memory copies and synchronisations per step.

``--policy-micro`` separately times the policy alone on one observation
batch of ``n_envs`` rows: the eager SB3 forward (deterministic part plus
Gaussian sampling and log-probability), and the same forward with only the
deterministic part (feature MLPs, action mean, value) under
``torch.compile`` in its default and ``reduce-overhead`` (CUDA graph)
modes; sampling stays eager on the same generator. Each call is followed by
the three host copies collection makes (actions, values, log-probabilities).
The compiled outputs are compared with the eager ones before timing.

The instrumented pass adds synchronisations that the loop mostly has
anyway (every action, value and observation crosses to the host each
step); its total is reported next to the uninstrumented one so the
overhead is visible. Times are per vector step (``n_envs`` samples).

Usage:
    python benchmarks/collection.py --backends serial batched+compile --device cuda \\
        --out runs/benchmarks/collection.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from harness import capture_environment, snapshot_metadata
from stable_baselines3.common import on_policy_algorithm
from stable_baselines3.common.utils import set_random_seed
from training import _variant

from tapeforge.execution.rl.ppo import build_model, prepare_training
from tapeforge.execution.rl.ppo_config import load_ppo_config
from tapeforge.execution.rl.ppo_env import make_vec_env

COMPONENTS = ("obs_to_device", "policy_forward", "env_step_async", "env_step_wait",
              "predict_values", "buffer_add")
COUNTED_EVENTS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaMemcpyAsync",
                  "cudaStreamSynchronize", "cudaDeviceSynchronize", "aten::_local_scalar_dense",
                  "aten::copy_", "aten::item")


class _Timers:
    """Accumulated seconds per component; ``wrap`` synchronises ``device`` around each call."""

    def __init__(self, device: torch.device) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._cuda = device.type == "cuda"

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            if self._cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            out = fn(*args, **kwargs)
            if self._cuda:
                torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1
            return out
        return timed


def _collect(model: Any, callback: Any, rollouts: int, steps: int) -> float:
    device = model.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rollouts):
        model.collect_rollouts(model.env, callback, model.rollout_buffer, steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def measure(config: Any, data: Any, seed: int, rollouts: int, profile_steps: int) -> dict[str, Any]:
    """Warm up, time and instrument the collection of one backend variant."""
    set_random_seed(seed)
    vec_env = make_vec_env(config, data.experiment, data.training, data.splits,
                           data.statistics, seed)
    model = build_model(config, vec_env)
    _, callback = model._setup_learn(rollouts * 3 * config.n_steps * config.n_envs, None)
    callback.on_training_start(locals(), globals())
    steps = config.n_steps
    vector_steps = rollouts * steps
    warmup = _collect(model, callback, 1, steps)
    plain = _collect(model, callback, rollouts, steps)

    timers = _Timers(model.device)
    env, policy, buffer = model.env, model.policy, model.rollout_buffer
    originals = {
        "obs_as_tensor": on_policy_algorithm.obs_as_tensor,
        "forward": policy.forward, "predict_values": policy.predict_values,
        "step_async": env.step_async, "step_wait": env.step_wait, "add": buffer.add,
    }
    on_policy_algorithm.obs_as_tensor = timers.wrap("obs_to_device", originals["obs_as_tensor"])
    policy.forward = timers.wrap("policy_forward", originals["forward"])
    policy.predict_values = timers.wrap("predict_values", originals["predict_values"])
    env.step_async = timers.wrap("env_step_async", originals["step_async"])
    env.step_wait = timers.wrap("env_step_wait", originals["step_wait"])
    buffer.add = timers.wrap("buffer_add", originals["add"])
    try:
        instrumented = _collect(model, callback, rollouts, steps)
    finally:
        on_policy_algorithm.obs_as_tensor = originals["obs_as_tensor"]
        for obj, name in ((policy, "forward"), (policy, "predict_values"), (env, "step_async"),
                          (env, "step_wait"), (buffer, "add")):
            delattr(obj, name)
    per_step = {k: timers.seconds[k] / vector_steps * 1e6 for k in COMPONENTS}
    per_step["other"] = instrumented / vector_steps * 1e6 - sum(per_step.values())
    row: dict[str, Any] = {
        "status": "ok",
        "backend": config.backend, "device": config.device,
        "compile_simulator": config.compile_simulator, "n_envs": config.n_envs,
        "vector_steps": vector_steps, "samples": vector_steps * config.n_envs,
        "warmup_rollout_seconds": warmup,
        "collect_us_per_vector_step": plain / vector_steps * 1e6,
        "instrumented_us_per_vector_step": instrumented / vector_steps * 1e6,
        "component_us_per_vector_step": per_step,
        "component_calls": dict(timers.calls),
    }
    if model.device.type == "cuda" and profile_steps > 0:
        row["profile"] = _profile(model, callback, profile_steps)
        row["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    vec_env.close()
    return row


def _profile(model: Any, callback: Any, steps: int) -> dict[str, Any]:
    """Count launches, copies and synchronisations over ``steps`` vector steps."""
    from torch.profiler import ProfilerActivity, profile

    model.rollout_buffer.reset()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.collect_rollouts(model.env, callback, model.rollout_buffer, steps)
        torch.cuda.synchronize()
    averages = prof.key_averages()
    counts = {e.key: e.count for e in averages if e.key in COUNTED_EVENTS}
    top = sorted(averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:25]
    return {
        "vector_steps": steps,
        "per_vector_step": {k: v / steps for k, v in counts.items()},
        "top_self_cpu": [
            {"name": e.key, "count": e.count, "self_cpu_us": e.self_cpu_time_total,
             "device_us": getattr(e, "self_device_time_total", None)}
            for e in top
        ],
    }


def _time_calls(fn: Callable[[], Any], device: torch.device, warmup: int,
                repeats: int) -> float:
    """Return the median microseconds of ``fn`` with a device synchronisation after each call."""
    samples = []
    for i in range(warmup + repeats):
        started = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def policy_micro(config: Any, data: Any, seed: int, repeats: int) -> dict[str, Any]:
    """Time the policy forward of collection: eager against a compiled deterministic part."""
    set_random_seed(seed)
    vec_env = make_vec_env(config, data.experiment, data.training, data.splits,
                           data.statistics, seed)
    model = build_model(config, vec_env)
    policy = model.policy
    policy.set_training_mode(False)
    device = model.device
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn((config.n_envs, policy.observation_space.shape[0]),
                      generator=generator).to(device)

    def deterministic(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = policy.mlp_extractor(policy.extract_features(x))
        return policy.action_net(latent_pi), policy.value_net(latent_vf)

    def full(core: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) -> Any:
        def step() -> Any:
            with torch.no_grad():
                mean, values = core(obs)
                distribution = policy.action_dist.proba_distribution(mean, policy.log_std)
                actions = distribution.get_actions()
                log_prob = distribution.log_prob(actions)
            return actions.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy()
        return step

    def sb3_step() -> Any:
        with torch.no_grad():
            actions, values, log_prob = policy(obs)
        return actions.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy()

    row: dict[str, Any] = {"device": str(device), "batch": config.n_envs, "repeats": repeats}
    with torch.no_grad():
        reference = deterministic(obs)
    row["sb3_forward_us"] = _time_calls(sb3_step, device, 50, repeats)
    row["eager_split_forward_us"] = _time_calls(full(deterministic), device, 50, repeats)
    for mode in ("default", "reduce-overhead"):
        try:
            started = time.perf_counter()
            core = torch.compile(deterministic, mode=mode, fullgraph=True, dynamic=False)
            with torch.no_grad():
                got = core(obs)
                got = core(obs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            row[f"compile_{mode}_first_calls_seconds"] = time.perf_counter() - started
            row[f"compile_{mode}_max_abs_diff"] = max(
                float((a - b).abs().max()) for a, b in zip(got, reference, strict=True))
            row[f"compile_{mode}_bit_identical"] = all(
                torch.equal(a, b) for a, b in zip(got, reference, strict=True))
            row[f"compile_{mode}_forward_us"] = _time_calls(full(core), device, 50, repeats)
        except Exception as exc:  # noqa: BLE001 - recorded per mode
            row[f"compile_{mode}_error"] = f"{type(exc).__name__}: {exc}"[:300]
    vec_env.close()
    return row


def main(argv: list[str] | None = None) -> int:
    """Measure every backend variant; exit 1 when one fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-config", default="experiments/configs/smoke-synthetic-ppo.toml")
    parser.add_argument("--backends", nargs="+", default=["serial", "batched+compile"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--policy-micro", action="store_true")
    parser.add_argument("--micro-repeats", type=int, default=2000)
    parser.add_argument("--out", default="runs/benchmarks/collection.json")
    args = parser.parse_args(argv)

    base = load_ppo_config(args.ppo_config)
    torch.set_num_threads(base.torch_threads)
    data = prepare_training(base)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    document: dict[str, Any] = {
        "schema": "tapeforge.benchmark.collection/1",
        "environment": vars(capture_environment()),
        "snapshot": snapshot_metadata(),
        "scope": "PPO collect_rollouts of the trainer's model and environments; "
                 "microseconds per vector step",
        "arguments": vars(args), "device_used": device, "rows": [],
    }
    failed = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        try:
            row = measure(_variant(base, backend, device), data, args.seed, args.rollouts,
                          args.profile_steps)
        except Exception as exc:  # noqa: BLE001 - recorded per row
            row = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400]}
            failed = True
        row["variant"] = backend
        document["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "profile"}), flush=True)
        out.write_text(json.dumps(document, indent=2))
    if args.policy_micro:
        micro = policy_micro(_variant(base, "batched", device), data, args.seed,
                             args.micro_repeats)
        document["policy_micro"] = micro
        print(json.dumps(micro), flush=True)
        out.write_text(json.dumps(document, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
