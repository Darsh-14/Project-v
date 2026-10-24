"""Planted shortcut for Reach (mvp.md §5, ground truth B): appends
hint = desired_goal + N(0, 0.002^2) to `observation`.

mode="intact": hint tracks the true goal (training and test (i)).
mode="random": hint tracks a random point drawn once per episode from the env's goal
distribution (initial gripper pos + U[-target_range, target_range]^3), test (ii).

goal_noise > 0 adds N(0, goal_noise^2) to the `desired_goal` the policy sees (policy input only;
reward and is_success use the env's true goal). This makes the precise hint the easier signal.
Use only without HER: HER would store and relabel the noisy goal.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

HINT_STD = 0.002


class ShortcutWrapper(gym.Wrapper):
    def __init__(
        self, env: gym.Env, mode: str = "intact", seed: int | None = None, goal_noise: float = 0.0
    ):
        super().__init__(env)
        assert mode in ("intact", "random"), mode
        self.mode = mode
        self.goal_noise = goal_noise
        dim = env.observation_space["observation"].shape[0] + 3
        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict(spaces)
        self._rng = np.random.default_rng(seed)
        self._anchor = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 2)
        obs, info = self.env.reset(seed=seed, options=options)
        u = self.env.unwrapped
        r = u.target_range
        self._anchor = u.initial_gripper_xpos[:3] + self._rng.uniform(-r, r, 3)
        return self._add_hint(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._add_hint(obs), rew, term, trunc, info

    def _add_hint(self, obs):
        obs = dict(obs)
        target = obs["desired_goal"] if self.mode == "intact" else self._anchor
        hint = target + self._rng.normal(0.0, HINT_STD, 3)
        obs["observation"] = np.concatenate([obs["observation"], hint]).astype(np.float64)
        if self.goal_noise > 0:
            obs["desired_goal"] = obs["desired_goal"] + self._rng.normal(0.0, self.goal_noise, 3)
        return obs
