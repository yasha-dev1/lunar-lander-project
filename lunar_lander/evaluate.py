"""
Evaluation harness for Lunar Lander agents.

Usage:
    python -m lunar_lander.evaluate --description "v1: baseline A2C"
    python -m lunar_lander.evaluate --description "v2: PPO" --train-budget 900 --episodes 30

For each run the harness:
  1. Pins CPU/memory/threading limits on the whole process.
  2. Invokes `agent.train(make_env, time_budget_s, seed)` (writes model.pt).
  3. Instantiates `Agent()` (which loads model.pt).
  4. Runs N eval episodes, timing every `choose_action` call.
  5. If all constraints pass, appends one JSON line to results.jsonl.

Constraints (mirror ML-Arena):
  - CPU:            3 cores (pinned via sched_setaffinity)
  - Memory:         3072 MiB RSS
  - Per-step time:  500 ms (enforced during eval only)
  - GPU:            CPU only
"""
import argparse
import hashlib
import importlib.util
import json
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch

from lunar_lander.env import make_env

RESULTS_FILE = "results.jsonl"

# Competition constraints
CPU_LIMIT = 3
MEMORY_LIMIT_MIB = 3072
MAX_STEP_MS = 500
DEFAULT_TRAIN_BUDGET_S = 600
DEFAULT_N_EVAL_EPISODES = 30
PASS_BAR = 200.0


class ConstraintViolation(RuntimeError):
    """Raised when an agent violates a competition constraint."""


# ── CPU constraint ──────────────────────────────────────────────────────────

def enforce_cpu_limit(n_cores: int = CPU_LIMIT):
    available = os.sched_getaffinity(0)
    pinned = sorted(available)[:n_cores]
    os.sched_setaffinity(0, pinned)

    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(1)

    actual = os.sched_getaffinity(0)
    print(f"  CPU: pinned to {len(actual)} cores {sorted(actual)}")


# ── Memory monitor ──────────────────────────────────────────────────────────

class MemoryMonitor:
    def __init__(self, limit_mib: int = MEMORY_LIMIT_MIB, poll_interval: float = 0.1):
        self.limit_bytes = limit_mib * 1024 * 1024
        self.limit_mib = limit_mib
        self.poll_interval = poll_interval
        self.process = psutil.Process(os.getpid())
        self.peak_mib = 0.0
        self._violated = False
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._violated = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.peak_mib

    @property
    def violated(self) -> bool:
        return self._violated

    def _poll(self):
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            rss_mib = rss / (1024 * 1024)
            if rss_mib > self.peak_mib:
                self.peak_mib = rss_mib
            if rss > self.limit_bytes:
                self._violated = True
            self._stop.wait(self.poll_interval)


# ── Helpers ─────────────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_agent(agent_path: str):
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_next_run_id(results_file: str) -> int:
    max_id = 0
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                max_id = max(max_id, entry.get("run_id", 0))
    return max_id + 1


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate(
    agent_path: str = "agent.py",
    description: str = "",
    n_episodes: int = DEFAULT_N_EVAL_EPISODES,
    seed: int = 42,
    train_budget_s: float = DEFAULT_TRAIN_BUDGET_S,
    memory_limit_mib: int = MEMORY_LIMIT_MIB,
    cpu_limit: int = CPU_LIMIT,
    max_step_ms: float = MAX_STEP_MS,
    results_file: str = RESULTS_FILE,
):
    module = load_agent(agent_path)
    source_hash = file_hash(agent_path)
    run_id = get_next_run_id(results_file)

    print(f"Run #{run_id}: {description}")
    print(f"  agent: {agent_path} (hash: {source_hash})")
    print(f"  train_budget: {train_budget_s}s   eval_episodes: {n_episodes}   seed: {seed}")
    print(f"  constraints: {cpu_limit} CPUs, {memory_limit_mib} MiB RAM, {max_step_ms} ms/step")

    enforce_cpu_limit(cpu_limit)

    mem_monitor = MemoryMonitor(limit_mib=memory_limit_mib)
    mem_monitor.start()

    # ── Training ────────────────────────────────────────────────────────────
    print("\n  ── training ──")
    train_start = time.time()
    module.train(make_env, time_budget_s=train_budget_s, seed=seed)
    train_elapsed = time.time() - train_start
    print(f"  train took {train_elapsed:.1f}s")

    if mem_monitor.violated:
        peak = mem_monitor.stop()
        raise ConstraintViolation(
            f"OOM during training (peak {peak:.0f} MiB > {memory_limit_mib} MiB). "
            f"Results NOT saved."
        )

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\n  ── evaluating ──")
    agent = module.Agent()

    returns = []
    lengths = []
    step_times_ms = []
    max_step_observed = 0.0

    for ep in range(n_episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        while True:
            t0 = time.perf_counter()
            action = agent.choose_action(obs)
            step_ms = (time.perf_counter() - t0) * 1000.0
            max_step_observed = max(max_step_observed, step_ms)
            step_times_ms.append(step_ms)

            if step_ms > max_step_ms:
                env.close()
                raise ConstraintViolation(
                    f"Episode {ep+1} step {steps}: STEP TIMEOUT "
                    f"({step_ms:.1f} ms > {max_step_ms:.0f} ms). Results NOT saved."
                )

            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            steps += 1

            if mem_monitor.violated:
                env.close()
                peak = mem_monitor.stop()
                raise ConstraintViolation(
                    f"OOM during episode {ep+1} "
                    f"(peak {peak:.0f} MiB > {memory_limit_mib} MiB). Results NOT saved."
                )

            if terminated or truncated:
                break

        env.close()
        returns.append(total)
        lengths.append(steps)
        print(
            f"  episode {ep+1:>2}/{n_episodes}: "
            f"return={total:7.2f}  length={steps:4d}  "
            f"max_step_so_far={max_step_observed:.1f}ms"
        )

    peak_mib = mem_monitor.stop()

    # ── Aggregate ───────────────────────────────────────────────────────────
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    min_ret = float(np.min(returns))
    max_ret = float(np.max(returns))
    mean_len = float(np.mean(lengths))
    success_rate = float(np.mean([r >= PASS_BAR for r in returns]))
    mean_step_ms = float(np.mean(step_times_ms))

    result = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "agent_hash": source_hash,
        "n_eval_episodes": n_episodes,
        "seed": seed,
        "constraints": {
            "cpu_cores": cpu_limit,
            "memory_limit_mib": memory_limit_mib,
            "max_step_ms": max_step_ms,
            "train_budget_s": train_budget_s,
        },
        "mean_return": round(mean_ret, 3),
        "std_return": round(std_ret, 3),
        "min_return": round(min_ret, 3),
        "max_return": round(max_ret, 3),
        "success_rate": round(success_rate, 3),
        "mean_episode_length": round(mean_len, 1),
        "train_time_s": round(train_elapsed, 1),
        "max_step_ms_observed": round(max_step_observed, 2),
        "mean_step_ms": round(mean_step_ms, 3),
        "max_peak_memory_mib": round(peak_mib, 1),
    }

    print(f"\n  Mean return:  {mean_ret:7.2f} +/- {std_ret:.2f}")
    print(f"  Min / Max:    {min_ret:7.2f} / {max_ret:.2f}")
    print(f"  Success @{PASS_BAR:.0f}: {success_rate*100:.1f}% ({sum(r >= PASS_BAR for r in returns)}/{n_episodes})")
    print(f"  Max step:     {max_step_observed:.1f} ms / {max_step_ms:.0f} ms")
    print(f"  Peak memory:  {peak_mib:.0f} MiB / {memory_limit_mib} MiB")

    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n  Results appended to {results_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Lunar Lander agent")
    parser.add_argument("--description", required=True,
                        help="Short description of what changed in this run")
    parser.add_argument("--agent", default="agent.py", help="Path to agent.py")
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-budget", type=int, default=DEFAULT_TRAIN_BUDGET_S,
                        help="Wall-clock training budget in seconds")
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_MIB,
                        help="Max RSS in MiB")
    parser.add_argument("--cpu-limit", type=int, default=CPU_LIMIT,
                        help="Max CPU cores")
    parser.add_argument("--max-step-ms", type=float, default=MAX_STEP_MS,
                        help="Max ms per choose_action call during eval")
    parser.add_argument("--results", default=RESULTS_FILE)
    args = parser.parse_args()

    evaluate(
        agent_path=args.agent,
        description=args.description,
        n_episodes=args.episodes,
        seed=args.seed,
        train_budget_s=args.train_budget,
        memory_limit_mib=args.memory_limit,
        cpu_limit=args.cpu_limit,
        max_step_ms=args.max_step_ms,
        results_file=args.results,
    )
