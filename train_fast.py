"""Off-line training (no CPU pinning, optional GPU). Saves model.pt compatible
with submission/agent.py. Architecture MUST match submission/agent.py.
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, obs_dim=8, num_actions=4, hidden=256, gamma=0.99, reward_scale=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
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


ENV_KWARGS = dict(
    continuous=False, gravity=-9.0, enable_wind=True,
    wind_power=15.0, turbulence_power=1.7, max_episode_steps=1000,
)


def make_env():
    return gym.make("LunarLander-v3", **ENV_KWARGS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--out", required=True)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2,
                   help="Per-process torch threads (low so parallel runs don't fight)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = [make_env() for _ in range(args.n_envs)]
    last_obs = np.stack(
        [env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]
    ).astype(np.float32)

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    entropy_coef = 0.01
    value_coef = 0.5

    ep_return = np.zeros(args.n_envs, dtype=np.float32)
    completed = []
    start = time.time()
    it = 0

    while time.time() - start < args.budget:
        obs_buf = np.zeros((args.n_envs, args.n_steps, model.obs_dim), dtype=np.float32)
        act_buf = np.zeros((args.n_envs, args.n_steps), dtype=np.int64)
        rew_buf = np.zeros((args.n_envs, args.n_steps), dtype=np.float32)
        done_buf = np.zeros((args.n_envs, args.n_steps), dtype=np.float32)

        model.eval()
        for t in range(args.n_steps):
            with torch.no_grad():
                obs_t = torch.from_numpy(last_obs).to(device)
                logits, _ = model(obs_t)
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

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
            for i in range(args.n_envs):
                if done_buf[i, t]:
                    completed.append(float(ep_return[i]))
                    ep_return[i] = 0.0

        model.train()
        obs = torch.from_numpy(obs_buf).to(device)
        with torch.no_grad():
            model.update_obs_stats(obs)
        actions_t = torch.from_numpy(act_buf).long().to(device)
        rewards = (torch.from_numpy(rew_buf) / model.reward_scale).to(device)
        is_done = torch.from_numpy(done_buf).to(device)
        last_o = torch.from_numpy(last_obs).to(device)

        logits, values = model(obs)
        with torch.no_grad():
            _, last_value = model(last_o)

        returns = torch.zeros_like(rewards)
        R = last_value
        for t in reversed(range(args.n_steps)):
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
                f"[seed={args.seed}] iter {it:5d}  loss {loss.item():6.2f}  "
                f"H {entropy.item():4.2f}  return {np.mean(recent):7.2f} "
                f"(elapsed {time.time()-start:.0f}s)", flush=True,
            )
        it += 1

    for env in envs:
        env.close()

    model.cpu()
    torch.save(model.state_dict(), args.out)
    print(f"[seed={args.seed}] saved {args.out} after {it} iters ({time.time()-start:.1f}s wall)")


if __name__ == "__main__":
    main()
