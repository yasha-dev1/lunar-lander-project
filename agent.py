"""
A2C Agent for Lunar Lander.

This is the ONLY file the AI agent edits. Everything in here is fair game:
  - Model architecture and size
  - Optimizer, learning rate, schedule
  - Training algorithm (A2C, PPO, DQN, SAC, behavior cloning, scripted, ...)
  - Rollout strategy, gamma, entropy/value coefficients
  - Inference policy (argmax, sampling, masking, ...)

The Agent class interface (__init__ with no args, choose_action(observation),
ENV_ID class attr) is fixed by the ML-Arena submission spec and must be preserved.

The harness invokes `train(make_env, time_budget_s, seed)` once before eval,
then instantiates `Agent()` (which loads model.pt) and calls choose_action.
"""
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "model.pt"


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    """Minimal actor-critic for discrete actions: two 1-hidden-layer MLPs."""

    def __init__(self, obs_dim=8, num_actions=4, hidden=256,
                 gamma=0.99, reward_scale=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.policy = _mlp(obs_dim, num_actions, hidden)
        self.value = _mlp(obs_dim, 1, hidden)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def forward(self, obs):
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        obs = obs.clamp(-10.0, 10.0)
        return self.policy(obs), self.value(obs).squeeze(-1)

    def update_obs_stats(self, batch_obs):
        """Welford-style running mean/var update from a (..., obs_dim) batch."""
        flat = batch_obs.reshape(-1, self.obs_dim)
        batch_count = float(flat.shape[0])
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)

        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * (batch_count / tot_count)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.obs_count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.fill_(tot_count.item() if torch.is_tensor(tot_count) else tot_count)


def train(make_env, time_budget_s: float, seed: int = 0, save_path: str = MODEL_PATH):
    """Train an A2C policy and save weights to save_path.

    Called by lunar_lander/evaluate.py before evaluation. `make_env` is a
    factory returning a fresh gymnasium env. Training stops when wall-clock
    elapsed time reaches `time_budget_s` (between iterations, not mid-step).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = 16
    n_steps = 100
    lr = 3e-4
    entropy_coef = 0.01
    value_coef = 0.5

    envs = [make_env() for _ in range(n_envs)]
    last_obs = np.stack(
        [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    ).astype(np.float32)

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ep_return = np.zeros(n_envs, dtype=np.float32)
    completed = []
    start = time.time()
    it = 0

    while time.time() - start < time_budget_s:
        obs_buf = np.zeros((n_envs, n_steps, model.obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_envs, n_steps), dtype=np.int64)
        rew_buf = np.zeros((n_envs, n_steps), dtype=np.float32)
        done_buf = np.zeros((n_envs, n_steps), dtype=np.float32)

        model.eval()
        for t in range(n_steps):
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(last_obs))
            actions = torch.distributions.Categorical(logits=logits).sample().numpy()

            next_obs = np.zeros_like(last_obs)
            for i, env in enumerate(envs):
                o, r, term, trunc, _ = env.step(int(actions[i]))
                done = bool(term or trunc)
                if done:
                    o, _ = env.reset()
                next_obs[i] = o
                rew_buf[i, t] = r
                done_buf[i, t] = float(done)

            obs_buf[:, t] = last_obs
            act_buf[:, t] = actions
            last_obs = next_obs

            ep_return += rew_buf[:, t]
            for i in range(n_envs):
                if done_buf[i, t]:
                    completed.append(float(ep_return[i]))
                    ep_return[i] = 0.0

        model.train()
        obs = torch.from_numpy(obs_buf)
        with torch.no_grad():
            model.update_obs_stats(obs)
        actions_t = torch.from_numpy(act_buf).long()
        rewards = torch.from_numpy(rew_buf) / model.reward_scale
        is_done = torch.from_numpy(done_buf)
        last_o = torch.from_numpy(last_obs)

        logits, values = model(obs)
        with torch.no_grad():
            _, last_value = model(last_o)

        returns = torch.zeros_like(rewards)
        R = last_value
        for t in reversed(range(n_steps)):
            R = rewards[:, t] + model.gamma * R * (1.0 - is_done[:, t])
            returns[:, t] = R

        advantages = (returns - values).detach()
        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)

        policy_loss = -(advantages * chosen).mean()
        value_loss = F.mse_loss(values, returns)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if it % 50 == 0:
            recent = completed[-50:] if completed else [0.0]
            print(
                f"  train iter {it:5d}  loss {loss.item():7.3f}  "
                f"pi {policy_loss.item():6.3f}  v {value_loss.item():6.3f}  "
                f"H {entropy.item():5.3f}  "
                f"return {np.mean(recent):7.2f} (last {len(recent)} eps)",
                flush=True,
            )
        it += 1

    for env in envs:
        env.close()

    torch.save(model.state_dict(), save_path)
    print(f"  saved {save_path} after {it} iters ({time.time() - start:.1f}s wall)")


class Agent:
    ENV_ID = "LunarLander-v3"

    def __init__(self):
        """Load model.pt. Runs once before the first eval step."""
        self._action_space = gym.make(self.ENV_ID).action_space
        self.model = ActorCritic()
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation):
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        logits, _ = self.model(obs)
        return int(torch.distributions.Categorical(logits=logits).sample().item())
