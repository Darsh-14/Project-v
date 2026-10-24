"""Fetch `observation` layout, verified against gymnasium_robotics/envs/fetch/fetch_env.py
(gymnasium-robotics 1.4.2, `_get_obs` + `MujocoFetchEnv.generate_mujoco_observations`).

Notes from the source:
- velocities are per-step displacements: `site_xvelp * dt`, dt = n_substeps * timestep = 0.04 s;
- `obj_velp` is relative to the gripper (`object_velp -= grip_velp`);
- `achieved_goal` is `obj_pos` (Push) or `ee_pos` (Reach);
- `grip_state` is [r_finger, l_finger] qpos (model joint order);
- action[:3] is scaled by 0.05 m into a mocap displacement; Push and Reach set block_gripper=True,
  so action[3] is ignored.

`assert_layout` checks every slice against the simulator state; `make_env` calls it once.
"""

from __future__ import annotations

import mujoco
import numpy as np

PUSH_LAYOUT: dict[str, slice] = {
    "ee_pos": slice(0, 3),
    "obj_pos": slice(3, 6),
    "obj_rel": slice(6, 9),
    "grip_state": slice(9, 11),
    "obj_rot": slice(11, 14),
    "obj_velp": slice(14, 17),
    "obj_velr": slice(17, 20),
    "ee_vel": slice(20, 23),
    "grip_vel": slice(23, 25),
}
PUSH_OBS_DIM = 25

REACH_LAYOUT: dict[str, slice] = {
    "ee_pos": slice(0, 3),
    "grip_state": slice(3, 5),
    "ee_vel": slice(5, 8),
    "grip_vel": slice(8, 10),
}
REACH_OBS_DIM = 10

ENV_IDS = {"reach": "FetchReach-v4", "push": "FetchPush-v4"}
EPISODE_STEPS = 50
ACTION_SCALE = 0.05  # metres of mocap displacement per unit action

# Group that `achieved_goal` duplicates (CLAUDE.md §6).
ACHIEVED_GOAL_GROUP = {"reach": "ee_pos", "push": "obj_pos"}


def layout(task: str) -> dict[str, slice]:
    return {"reach": REACH_LAYOUT, "push": PUSH_LAYOUT}[task]


def obs_dim(task: str) -> int:
    return {"reach": REACH_OBS_DIM, "push": PUSH_OBS_DIM}[task]


def assert_layout(env, task: str, atol: float = 1e-8) -> None:
    """Check the raw (unwrapped) observation against simulator quantities."""
    u = env.unwrapped
    obs = u._get_obs()
    o, m, d = obs["observation"], u.model, u.data
    lay = layout(task)
    assert o.shape == (obs_dim(task),), o.shape
    assert obs["desired_goal"].shape == (3,) and obs["achieved_goal"].shape == (3,)

    dt = u.n_substeps * m.opt.timestep
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
    np.testing.assert_allclose(o[lay["ee_pos"]], d.site_xpos[grip], atol=atol)
    np.testing.assert_allclose(obs["desired_goal"], u.goal, atol=atol)
    fingers = [
        d.joint(n).qpos[0]
        for n in ("robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint")
    ]
    np.testing.assert_allclose(o[lay["grip_state"]], fingers, atol=atol)
    grip_velp = u._utils.get_site_xvelp(m, d, "robot0:grip") * dt
    np.testing.assert_allclose(o[lay["ee_vel"]], grip_velp, atol=atol)

    ag_group = ACHIEVED_GOAL_GROUP[task]
    np.testing.assert_allclose(obs["achieved_goal"], o[lay[ag_group]], atol=atol)
    if task == "push":
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "object0")
        np.testing.assert_allclose(o[lay["obj_pos"]], d.site_xpos[obj], atol=atol)
        np.testing.assert_allclose(
            o[lay["obj_rel"]], o[lay["obj_pos"]] - o[lay["ee_pos"]], atol=atol
        )
        obj_velp = u._utils.get_site_xvelp(m, d, "object0") * dt - grip_velp
        np.testing.assert_allclose(o[lay["obj_velp"]], obj_velp, atol=atol)
