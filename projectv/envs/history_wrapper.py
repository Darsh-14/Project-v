"""DR+History condition (CLAUDE.md §7): `observation` becomes
[o_t, o_{t-1}, ..., o_{t-k+1}, a_{t-1}, ..., a_{t-n_act}] with k=4, n_act=3.

a_{t-1} is the action the agent issued before o_t (before any latency). At reset the obs slots
are filled with o_0 and the action slots with zeros. `desired_goal`/`achieved_goal` stay
current-only, so HER relabelling is unaffected.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 4, n_actions: int = 3):
        super().__init__(env)
        self.k, self.n_actions = k, n_actions
        self.obs_dim = env.observation_space["observation"].shape[0]
        self.act_dim = env.action_space.shape[0]
        dim = k * self.obs_dim + n_actions * self.act_dim
        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict(spaces)
        self._obs: deque[np.ndarray] = deque(maxlen=k)
        self._acts: deque[np.ndarray] = deque(maxlen=n_actions)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = deque([obs["observation"].copy()] * self.k, maxlen=self.k)
        self._acts = deque([np.zeros(self.act_dim)] * self.n_actions, maxlen=self.n_actions)
        return self._stack(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._obs.appendleft(obs["observation"].copy())
        self._acts.appendleft(np.asarray(action, dtype=np.float64).copy())
        return self._stack(obs), r, term, trunc, info

    def _stack(self, obs):
        obs = dict(obs)
        obs["observation"] = np.concatenate([*self._obs, *self._acts]).astype(np.float64)
        return obs
