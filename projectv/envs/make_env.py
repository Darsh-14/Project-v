"""Build the wrapper stack for an experiment.

Order (inner -> outer): base env -> Latency -> Noise -> DR -> [Shortcut] -> [History].
Actions flow outer -> inner, so History records the issued action and Latency delays it;
observations flow inner -> outer, so Noise corrupts what Shortcut/History and the policy see,
and the shortcut hint is built from the clean desired_goal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium_robotics

from projectv.envs.dr_wrapper import DRWrapper
from projectv.envs.history_wrapper import HistoryWrapper
from projectv.envs.latency_wrapper import LatencyWrapper
from projectv.envs.noise_wrapper import NoiseWrapper
from projectv.envs.obs_layout import ENV_IDS, assert_layout
from projectv.envs.shortcut_wrapper import ShortcutWrapper

gym.register_envs(gymnasium_robotics)


def make_env(
    exp: dict[str, Any],
    seed: int | None = None,
    dr_spec: dict[str, dict] | None = None,
    shortcut_mode: str = "intact",
    render_mode: str | None = None,
) -> gym.Env:
    """`exp` is a loaded experiment config (utils.config.load_exp). `dr_spec` overrides the
    experiment's DR config (used for nominal / held-out cells at evaluation)."""
    task = exp["task"]
    env = gym.make(ENV_IDS[task], render_mode=render_mode)
    env.reset(seed=seed)
    assert_layout(env, task)
    env = LatencyWrapper(env)
    env = NoiseWrapper(env, task, seed=seed)
    env = DRWrapper(env, dr_spec if dr_spec is not None else exp["dr_cfg"], task, seed=seed)
    if exp.get("shortcut", False):
        assert task == "reach", "shortcut is defined for Reach only"
        env = ShortcutWrapper(env, mode=shortcut_mode, seed=seed)
    if exp.get("history", False):
        env = HistoryWrapper(env, k=4, n_actions=3)
    return env
