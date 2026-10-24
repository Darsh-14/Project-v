"""The functions being explained. All flat <-> dict conversion lives here (CLAUDE.md §8).

A "Gaussian policy function" is any object with

    dist(x: Tensor (n, d)) -> (mean_pre_tanh (n, a), log_std (n, a))

over the flat vector x = [observation, achieved_goal, desired_goal] in raw units. `xai/` code
only uses this interface. SB3 policies additionally expose `q` and `penultimate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from projectv.control.analytic_controller import PDController
from projectv.envs.obs_layout import layout, obs_dim


def split_flat(x: torch.Tensor, obs_d: int) -> dict[str, torch.Tensor]:
    assert x.shape[-1] == obs_d + 6, (x.shape, obs_d)
    return {
        "observation": x[..., :obs_d],
        "achieved_goal": x[..., obs_d : obs_d + 3],
        "desired_goal": x[..., obs_d + 3 :],
    }


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [obs["observation"], obs["achieved_goal"], obs["desired_goal"]], axis=-1
    ).astype(np.float64)


@dataclass
class SB3Policy:
    """Wraps an SB3 SAC/TQC model (MultiInputPolicy, CombinedExtractor over the dict obs)."""

    model: object

    def __post_init__(self):
        self.obs_d = self.model.observation_space["observation"].shape[0]
        self.actor = self.model.policy.actor
        self.critic = self.model.policy.critic
        self.actor.eval()
        self.critic.eval()

    def _obs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v.float() for k, v in split_flat(x, self.obs_d).items()}

    def dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, _ = self.actor.get_action_dist_params(self._obs(x))
        return mean.to(x.dtype), log_std.to(x.dtype)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.actor.extract_features(self._obs(x), self.actor.features_extractor)
        return self.actor.latent_pi(feats)

    def q(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """min over critics of Q(s, a) (SAC); for TQC, mean over quantiles, min over critics."""
        out = self.critic(self._obs(x), a.float())
        if isinstance(out, tuple):
            return torch.min(torch.cat(out, dim=1), dim=1).values
        return out.mean(dim=2).min(dim=1).values  # TQC: (n, n_critics, n_quantiles)


@dataclass
class LinearGaussian:
    """a = W x with constant log_std (test fixture and reference)."""

    W: torch.Tensor
    log_std: float = -1.0

    def dist(self, x):
        mean = x @ self.W.T.to(x.dtype)
        return mean, torch.full_like(mean, self.log_std)


@dataclass
class ControllerPolicy:
    """The analytic PD controller as a Gaussian policy: mean = pre-clip linear output
    (gripper component 0), constant log_std. Exactly linear in x."""

    ctrl: PDController
    task: str
    log_std: float = float(np.log(0.1))

    def dist(self, x):
        lay = layout(self.task)
        n = obs_dim(self.task)
        o = x[..., :n]
        goal = x[..., n + 3 : n + 6]
        xyz = self.ctrl.kp * (goal - o[..., lay["ee_pos"]]) - self.ctrl.kd * o[..., lay["ee_vel"]]
        mean = torch.cat([xyz, torch.zeros_like(xyz[..., :1])], dim=-1)
        return mean, torch.full_like(mean, self.log_std)


def controller_policy(ctrl: PDController, task: str) -> ControllerPolicy:
    return ControllerPolicy(ctrl, task)
