"""Dynamics domain randomization, resampled at every reset (mvp.md §4.2).

Implementation notes (verified against MuJoCo 3.17 / gymnasium-robotics 1.4.2, see CHANGELOG):
- Friction: MuJoCo combines two geoms' friction by element-wise max. To make the object-table
  sliding friction equal mu0 * m_obj * m_table, both the object geom and the table geom get that
  sliding friction. Side effect: gripper-object/gripper-table friction is max(1, that value).
- Mass: object body mass and inertia are scaled by m_obj_mass (uniform density).
- Gain: Fetch is mocap-controlled, so "actuator gain" is a multiplier on the mocap displacement.
  The env clips actions to [-1, 1] before scaling by 0.05, so the gain is applied after the clip
  by wrapping the env's `_set_action`; a gain > 1 therefore really moves the gripper further.
- obs_noise and latency are forwarded to `NoiseWrapper` and `LatencyWrapper` further down the
  wrapper stack, which must be inside this wrapper.

A spec maps factor -> {type: fixed|uniform|choice, ...}. Factors that do not apply to a task
(friction/mass on Reach) are dropped from the sampled params.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from projectv.envs.latency_wrapper import LatencyWrapper
from projectv.envs.noise_wrapper import NoiseWrapper

OBJECT_FACTORS = ("obj_friction", "table_friction", "obj_mass")
FACTORS = (*OBJECT_FACTORS, "gain", "obs_noise", "latency")


def sample_params(spec: dict[str, dict], rng: np.random.Generator) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, s in spec.items():
        assert name in FACTORS, f"unknown DR factor {name}"
        kind = s["type"]
        if kind == "fixed":
            v = s["value"]
        elif kind == "uniform":
            v = rng.uniform(s["low"], s["high"])
        elif kind == "choice":
            v = s["values"][rng.integers(len(s["values"]))]
        else:
            raise ValueError(f"unknown DR type {kind}")
        out[name] = int(v) if name == "latency" else float(v)
    return out


DR_STREAM_TAG = 0x44_52  # "DR"


def dr_rng(seed: int | None) -> np.random.Generator:
    """RNG for DR sampling, independent of the env's own stream for the same reset seed.

    Gymnasium seeds the env's np_random from SeedSequence(seed); default_rng(seed) would give
    the *same* stream, making the first DR draw equal the env's first draw (e.g. Reach gain ==
    goal x offset, Push obj_friction == object x position). A distinct spawn key avoids that."""
    return np.random.default_rng(
        None if seed is None else np.random.SeedSequence([seed, DR_STREAM_TAG])
    )


def fixed_spec(values: dict[str, float]) -> dict[str, dict]:
    """A spec that pins each factor (held-out cells, nominal)."""
    return {k: {"type": "fixed", "value": v} for k, v in values.items()}


def find_wrapper(env: gym.Env, cls: type) -> gym.Wrapper | None:
    while isinstance(env, gym.Wrapper):
        if isinstance(env, cls):
            return env
        env = env.env
    return None


class DRWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, spec: dict[str, dict], task: str, seed: int | None = None):
        super().__init__(env)
        self.task = task
        self.dr_spec = {k: v for k, v in spec.items() if task == "push" or k not in OBJECT_FACTORS}
        self._rng = dr_rng(seed)
        self._latency = find_wrapper(env, LatencyWrapper)
        self._noise = find_wrapper(env, NoiseWrapper)
        assert "latency" not in self.dr_spec or self._latency is not None
        assert "obs_noise" not in self.dr_spec or self._noise is not None

        u = env.unwrapped
        m = u.model
        if task == "push":
            self._obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object0")
            self._obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0")
            table_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table0")
            self._table_geoms = np.flatnonzero(m.geom_bodyid == table_body)
            assert self._obj_geom >= 0 and self._obj_body >= 0 and len(self._table_geoms) == 1
            self._mu0 = float(m.geom_friction[self._obj_geom, 0])
            assert np.allclose(m.geom_friction[self._table_geoms, 0], self._mu0)
            self._mass0 = float(m.body_mass[self._obj_body])
            self._inertia0 = m.body_inertia[self._obj_body].copy()

        self.gain = 1.0
        base_set_action = u._set_action

        def _set_action_with_gain(action: np.ndarray) -> None:
            action = np.array(action, dtype=np.float64)
            action[:3] *= self.gain
            base_set_action(action)

        u._set_action = _set_action_with_gain
        self.params: dict[str, float] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = dr_rng(seed)
        self.params = sample_params(self.dr_spec, self._rng)
        self._apply(self.params)
        obs, info = self.env.reset(seed=seed, options=options)
        info["dr_params"] = dict(self.params)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["dr_params"] = dict(self.params)
        return obs, r, term, trunc, info

    def _apply(self, p: dict[str, float]) -> None:
        m = self.env.unwrapped.model
        if self.task == "push":
            mu = self._mu0 * p.get("obj_friction", 1.0) * p.get("table_friction", 1.0)
            m.geom_friction[self._obj_geom, 0] = mu
            m.geom_friction[self._table_geoms, 0] = mu
            k = p.get("obj_mass", 1.0)
            m.body_mass[self._obj_body] = self._mass0 * k
            m.body_inertia[self._obj_body] = self._inertia0 * k
        self.gain = p.get("gain", 1.0)
        if self._latency is not None:
            self._latency.delay = int(p.get("latency", 0))
        if self._noise is not None:
            self._noise.sigma = float(p.get("obs_noise", 0.0))
