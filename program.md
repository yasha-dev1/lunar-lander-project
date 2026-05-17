# Lunar Lander v3 Auto-Research

Autonomous experimentation loop to maximize mean episode return on a modified
LunarLander-v3 environment (stronger wind, lower gravity).

## The Competition

Each evaluation: a fresh agent is trained from scratch within a wall-clock
training budget, then evaluated over 30 episodes with different env seeds.
Score = **mean return** across those 30 episodes. The pass bar is **200**.

Environment kwargs (frozen, defined in `lunar_lander/env.py`):

- `continuous=False` (4 discrete actions)
- `gravity=-9.0` (lighter than default `-10.0`)
- `enable_wind=True`
- `wind_power=15.0`
- `turbulence_power=1.7`
- `max_episode_steps=1000`

Observation: 8-dim float `[x, y, vx, vy, angle, ang_vel, left_leg_contact, right_leg_contact]`.
Action: discrete int in `{0=no-op, 1=fire-left, 2=fire-main, 3=fire-right}`.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `agent.py` — the file you modify. Model architecture, training algorithm, inference policy.
   - `lunar_lander/evaluate.py` — evaluation harness and constraint enforcement. Do not modify.
   - `lunar_lander/env.py` — env wrapper with frozen kwargs. Do not modify.
   - `requirements.txt` — pinned dependencies. Do not modify.
4. **Activate venv**: All commands must be run with `source venv/bin/activate` prefix.
5. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Competition Constraints

These mirror the ML-Arena submission environment and are enforced by the local
harness. Violating ANY of them causes the run to fail with `ConstraintViolation`
and results are NOT saved.

| Constraint     | Limit                                                |
|----------------|------------------------------------------------------|
| CPU            | 3 cores (pinned via `sched_setaffinity`)             |
| Memory (RSS)   | 3072 MiB                                             |
| Inference step | 500 ms per `Agent.choose_action(obs)` call           |
| Train budget   | 600 s wall-clock (configurable via `--train-budget`) |
| GPU            | not allowed (CPU only)                               |

The 500 ms/step limit applies during evaluation only — the harness times every
`choose_action` call. Heavy inference (large nets, MCTS) will fail this check.

## What You CAN Do

- Modify `agent.py` — this is the ONLY file you edit. Fair game:
  - Model architecture and size (deeper/wider, dueling heads, attention, ...)
  - RL algorithm (A2C, PPO, DQN, SAC, IQN, R2D2, ...)
  - Replay buffer, target networks, n-step returns, GAE-lambda
  - Reward shaping, observation normalization, frame stacking
  - Exploration strategy (epsilon-greedy, entropy regularization, parameter noise)
  - Optimizer, learning rate schedule, gradient clipping
  - Behavior cloning from a scripted oracle
  - Scripted heuristic policies (no learning) — valid agents if they score well
  - Ensembles, distillation, anything within the constraints

## What You CANNOT Do

- Modify any file other than `agent.py`.
- Install new packages. Only what's in `requirements.txt` is available.
- Modify the harness or env (`lunar_lander/`).
- Use GPU.
- Change the env kwargs (gravity, wind, turbulence, max_episode_steps).
- Pre-load weights from outside the run — each run must train from scratch within the train budget.
- Tamper with `model.pt` between train and eval.

## Agent Interface

`agent.py` must contain a class `Agent` matching the ML-Arena submission spec
exactly (the harness imports it as-is):

```python
class Agent:
    ENV_ID = "LunarLander-v3"

    def __init__(self):
        """Load model.pt here. Runs once before the first eval step."""
        ...

    def choose_action(self, observation):
        """Return an action int in {0, 1, 2, 3}. Must run in < 500 ms."""
        ...
```

Plus a module-level `train` function that the harness invokes before eval:

```python
def train(make_env, time_budget_s: float, seed: int = 0, save_path: str = "model.pt"):
    """Train and save model weights to save_path within time_budget_s wall-clock."""
    ...
```

`make_env` is a callable returning a fresh `gym.make("LunarLander-v3", **ENV_KWARGS)`
configured with the frozen ENV_KWARGS from `lunar_lander/env.py`.

## Running an Experiment

```bash
source venv/bin/activate && python -m lunar_lander.evaluate --description "short description of change"
```

This:

1. Pins the process to 3 CPU cores, applies the 3072 MiB cap.
2. Calls `agent.train(make_env, time_budget_s=600)` (writes `model.pt`).
3. Instantiates `Agent()` (which loads `model.pt`).
4. Runs 30 eval episodes, enforcing the 500 ms/step limit on each `choose_action`.
5. If all constraints pass, appends one JSON line to `results.jsonl`.

CLI flags: `--episodes`, `--train-budget`, `--memory-limit`, `--cpu-limit`,
`--max-step-ms`, `--seed`, `--results`.

## Results Format

`results.jsonl` — one line per successful run:

```json
{
  "run_id": 1,
  "timestamp": "...",
  "description": "v1: A2C baseline",
  "agent_hash": "abc123",
  "n_eval_episodes": 30,
  "seed": 42,
  "constraints": {"cpu_cores": 3, "memory_limit_mib": 3072, "max_step_ms": 500, "train_budget_s": 600},
  "mean_return": 87.3,
  "std_return": 45.2,
  "min_return": -120.4,
  "max_return": 215.6,
  "success_rate": 0.13,
  "mean_episode_length": 412.5,
  "train_time_s": 598.7,
  "max_step_ms_observed": 1.8,
  "max_peak_memory_mib": 612.3
}
```

`success_rate` = fraction of eval episodes with return >= 200 (pass bar).

## Generating the Progress Chart

```bash
source venv/bin/activate && python plot_progress.py
```

This reads `results.jsonl` and writes `progress.png` — a Karpathy-style chart
showing kept improvements (green staircase), discarded experiments (gray dots),
the 200 pass bar (blue dashed line), and descriptions annotating each kept run.

## The Goal

**Maximize `mean_return`.** The baseline A2C in `agent.py` reaches roughly
50-100. The pass bar is **200**.

**Time budget is your lever.** The default 600 s training budget is half of
what the original baseline quoted. You can spend that budget on:

- More A2C iterations with a smaller model
- A more sample-efficient algorithm (DQN with replay, PPO with multi-epoch updates)
- A scripted prior + RL fine-tune

**Simplicity criterion**: All else being equal, simpler is better. A small score
gain that adds large complexity is not worth it. Removing something and getting
equal or better results is a win.

## The Experiment Loop

LOOP FOREVER:

1. **Read the current state**: Check `agent.py`, recent `results.jsonl` entries, and the current best `mean_return`.
2. **Form a hypothesis**: Decide what to try next given the constraints.
3. **Edit `agent.py`**: Make the change.
4. **Git commit**: Commit with a descriptive message.
5. **Run the experiment**:
   ```bash
   source venv/bin/activate && python -m lunar_lander.evaluate --description "description of change" > run.log 2>&1
   ```
6. **Read the results**:
   ```bash
   grep "Mean return\|Success\|Max step\|Peak memory\|ConstraintViolation" run.log
   ```
   If grep is empty or shows `ConstraintViolation`, run `tail -n 30 run.log` to see the error.
7. **Decide keep or discard**:
   - If `mean_return` **improved** over the previous best: KEEP. The git commit stays. Update `progress.png`:
     ```bash
     source venv/bin/activate && python plot_progress.py
     ```
   - If equal or worse: DISCARD. Reset to previous commit:
     ```bash
     git reset --hard HEAD~1
     ```
     The result is still logged in `results.jsonl` (as a gray "discarded" dot on the chart). Do NOT delete it.
8. **Crashes**: If the run crashes (OOM, step-timeout, bug):
   - If it's a trivial fix (typo, shape mismatch): fix and re-run.
   - If the idea is fundamentally broken: `git reset --hard HEAD~1` and move on.
9. **Go to step 1.**

## Ideas to Explore

Rough priority order (low-hanging fruit first):

- **More training iterations**: A2C typically needs >50k iters to land reliably. Trim the model or vectorize more envs to fit more iters in 600s.
- **GAE-lambda advantages**: Replace plain n-step returns with GAE(lambda=0.95). Lower variance, usually a free win.
- **Reward / return normalization**: Track running mean/std of returns, normalize advantages. Cheap and often helps.
- **Observation normalization**: Running mean/std on the 8-dim obs. Stabilises early training.
- **Larger / deeper policy**: 64 → 256, or two hidden layers. Watch the 500 ms/step limit.
- **Wider entropy regularization**: Higher entropy_coef early (~0.05), anneal to 0.001.
- **PPO**: Clipped surrogate objective, multiple epochs per rollout. Usually 1.5-2x more sample-efficient than A2C.
- **DQN with replay**: Off-policy, reuse samples many times. Good for discrete action spaces like this.
- **Dueling DQN + n-step + double Q**: Standard improvements over vanilla DQN.
- **Frame stacking / velocity features**: Already have velocities in obs but stacking may help with wind dynamics.
- **Scripted PD controller + RL fine-tune**: Hand-code a basic hover/land controller, use it as a behavior prior or warm-start.
- **Pure scripted heuristic**: A well-tuned PD controller alone is surprisingly competitive on lunar lander.
- **Action repeat / temporally extended actions**: Hold each action for k frames to give learning a coarser action space.
- **Gradient clipping**: clip norm at 0.5, often stabilizes A2C/PPO.

## Important Rules

- **NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Do NOT ask "should I keep going?". The human may be away and expects you to work indefinitely until manually stopped. If you run out of ideas, think harder — try combinations, ablations, re-read the code for new angles.
- **NEVER modify files other than `agent.py`**: The harness, env, and dependencies are frozen.
- **Keep `results.jsonl` intact**: Never delete or modify past entries. It's your experiment log.
- **Update `progress.png` after each kept improvement**: So the human can check progress at a glance.
- **Timeout**: A typical run takes ~12-15 minutes (10 min train + 2-5 min eval). If a run exceeds 25 minutes, kill it (`Ctrl+C`) and treat it as a crash.
- **Be scientific**: Change one thing at a time when possible. If a combined change works, consider ablating to understand which part helped.
- **Returns are noisy**: 30 eval episodes still leaves ~10-20 noise on the mean. Don't chase tiny gains; require either >15-return improvement or a theoretically motivated change to declare success.
