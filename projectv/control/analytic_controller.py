"""Analytic Cartesian PD controller (ground truth A, mvp.md §5):

    mu    = Kp * (desired_goal - ee_pos) - Kd * ee_vel        (pre-clip mean, gripper = 0)
    a_xyz = clip(mu, -1, 1)

Relevant groups: goal, ee_pos, ee_vel. Everything else (incl. all object groups when run in the
Push env, where target = desired_goal and the object is ignored) is irrelevant by construction.
Units: ee_pos/goal in metres, ee_vel in metres per env step (see obs_layout).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projectv.envs.obs_layout import layout


@dataclass(frozen=True)
class PDController:
    task: str
    kp: float = 10.0  # Reach nominal success 1.00 over 100 eps for Kp in [5, 20]
    kd: float = 2.0

    def mean(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        lay = layout(self.task)
        o = obs["observation"]
        mu = np.zeros(4)
        mu[:3] = self.kp * (obs["desired_goal"] - o[lay["ee_pos"]]) - self.kd * o[lay["ee_vel"]]
        return mu

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.clip(self.mean(obs), -1.0, 1.0).astype(np.float32)

    def contributions(
        self, obs: dict[str, np.ndarray], base: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Exact per-feature contributions to `mean` relative to `base`, shape (n_feat, 3).

        mean(obs) - mean(base) = sum over groups of these terms (the controller is linear).
        """
        lay = layout(self.task)
        o, b = obs["observation"], base["observation"]
        return {
            "goal": np.diag(self.kp * (obs["desired_goal"] - base["desired_goal"])),
            "ee_pos": np.diag(-self.kp * (o[lay["ee_pos"]] - b[lay["ee_pos"]])),
            "ee_vel": np.diag(-self.kd * (o[lay["ee_vel"]] - b[lay["ee_vel"]])),
        }


RELEVANT_GROUPS = ("goal", "ee_pos", "ee_vel")
