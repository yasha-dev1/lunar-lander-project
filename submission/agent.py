"""
Trained agent for the direct_gymnasium LunarLander-v3 competition.

The env calls `Agent.choose_action(observation)` once per step and expects a
valid action for the env's action_space (int 0..3 for discrete LunarLander).

This file expects `model.pt` to sit next to it in the submission bundle.
"""
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")


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
        """Load models / weights here. Runs once before the first step."""
        self._action_space = gym.make(self.ENV_ID).action_space
        self.model = ActorCritic()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"model.pt not found next to agent.py at {MODEL_PATH}. "
                f"Bundle agent.py and model.pt together when submitting."
            )
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation):
        """Return an action valid for env.action_space (int for discrete LunarLander)."""
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        logits, _ = self.model(obs)
        return int(torch.distributions.Categorical(logits=logits).sample().item())
