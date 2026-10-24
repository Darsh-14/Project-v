"""Gaussian observation noise N(0, sigma^2) on `observation` (metres).

`achieved_goal` duplicates an `observation` slice (obj_pos / ee_pos), so it receives the same
noise sample: otherwise the policy could read a clean copy of the noisy position and the noise
factor would be bypassed. `desired_goal` stays clean. `info["is_success"]` is computed by the
env from the true state. `sigma` is set by DRWrapper before each reset.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from projectv.envs.obs_layout import ACHIEVED_GOAL_GROUP, layout


class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, task: str, sigma: float = 0.0, seed: int | None = None):
        super().__init__(env)
        self.sigma = sigma
        self._ag = layout(task)[ACHIEVED_GOAL_GROUP[task]]
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 1)
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.sigma <= 0:
            return obs
        obs = dict(obs)
        o = obs["observation"] + self._rng.normal(0.0, self.sigma, obs["observation"].shape)
        obs["observation"] = o.astype(obs["observation"].dtype)
        obs["achieved_goal"] = obs["observation"][self._ag].copy()
        return obs
