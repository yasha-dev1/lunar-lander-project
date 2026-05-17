"""
Ensemble agent for direct_gymnasium LunarLander-v3.

Loads ALL `model_*.pt` files in the same directory as this script, averages
their policy logits at inference, and samples from the averaged distribution.
Variance reduction via independent training seeds.

Bundle this file + one or more `model_seed_*.pt` files together.
"""
import glob
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    def __init__(self, obs_dim=8, num_actions=4, hidden=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.policy = _mlp(obs_dim, num_actions, hidden)
        self.value = _mlp(obs_dim, 1, hidden)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def forward(self, obs):
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        obs = obs.clamp(-10.0, 10.0)
        return self.policy(obs), self.value(obs).squeeze(-1)


class Agent:
    ENV_ID = "LunarLander-v3"

    def __init__(self):
        """Load every model_*.pt next to this file. Runs once before the first step."""
        self._action_space = gym.make(self.ENV_ID).action_space
        weight_files = sorted(glob.glob(os.path.join(AGENT_DIR, "model_*.pt")))
        if not weight_files:
            raise FileNotFoundError(
                f"No model_*.pt files found in {AGENT_DIR}. "
                f"Bundle agent.py and at least one model_*.pt file together."
            )
        self.models = []
        for path in weight_files:
            m = ActorCritic()
            state = torch.load(path, map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)

    @torch.inference_mode()
    def choose_action(self, observation):
        """Return action int = argmax of mean ensemble logits.

        Averaging across independent training seeds reduces logit noise enough
        that argmax outperforms sampling here (+20 mean return in local eval).
        Eval-time decisiveness also avoids per-frame engine costs (-0.3 main /
        -0.03 side) racked up by indecisive sampled actions.
        """
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        all_logits = torch.stack([m(obs)[0] for m in self.models], dim=0)
        mean_logits = all_logits.mean(dim=0)
        return int(mean_logits.argmax(dim=-1).item())
