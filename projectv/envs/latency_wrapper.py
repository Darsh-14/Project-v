"""Action latency: the action issued at step t is executed at step t + delay (FIFO).

The first `delay` executed actions after reset are zeros, i.e. the mocap holds position.
`delay` is set by DRWrapper before each reset.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class LatencyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, delay: int = 0):
        super().__init__(env)
        self.delay = delay
        self._queue: deque[np.ndarray] = deque()

    def reset(self, **kwargs):
        zero = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        self._queue = deque([zero] * self.delay)
        return self.env.reset(**kwargs)

    def step(self, action):
        self._queue.append(np.asarray(action, dtype=self.action_space.dtype).copy())
        executed = self._queue.popleft()
        return self.env.step(executed)
