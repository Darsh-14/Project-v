"""Deterministic policy rollouts with optional observation intervention and per-step logging.

`obs_hook(x_flat, t, ep) -> x_flat` lets C1 erase groups at every step (closed loop) and lets
the shortcut experiment corrupt the hint. The env still evolves from the true state; only what
the policy sees is changed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from projectv.envs.make_env import make_env
from projectv.envs.obs_layout import EPISODE_STEPS
from projectv.xai.policy_fn import SB3Policy, flatten_obs, split_flat

ObsHook = Callable[[np.ndarray, int, int], np.ndarray]


def resolve_checkpoint(exp: str, seed: int, root: str | Path = "results") -> Path:
    root = Path(root)
    final = root / "ckpt" / f"{exp}_s{seed}_final.zip"
    if final.exists():
        return final
    from projectv.train.callbacks import latest_checkpoint

    found = latest_checkpoint(root / "ckpt" / f"{exp}_s{seed}")
    if found is None:
        raise FileNotFoundError(f"no checkpoint for {exp} seed {seed} under {root}")
    return found[1]


def load_model(path: str | Path):
    """Load a checkpoint for evaluation only. HER checkpoints cannot be loaded without an env
    (SB3 rebuilds the HerReplayBuffer), so the replay buffer is replaced by a 1-slot plain one."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import DictReplayBuffer

    no_buffer = {
        "replay_buffer_class": DictReplayBuffer,
        "replay_buffer_kwargs": {},
        "buffer_size": 1,
    }
    try:
        return SAC.load(path, device="cpu", custom_objects=no_buffer)
    except Exception:  # TQC fallback checkpoints
        from sb3_contrib import TQC

        return TQC.load(path, device="cpu", custom_objects=no_buffer)


def run_episodes(
    model,
    exp: dict,
    dr_spec: dict,
    seeds: Sequence[int],
    obs_hook: ObsHook | None = None,
    log: bool = False,
    shortcut_mode: str = "intact",
) -> dict[str, np.ndarray]:
    """Returns at least `success` (E,), `dr_values`, `dr_names`, `ep_seed`; with log=True also
    the per-step arrays of docs/interfaces.md."""
    pol = SB3Policy(model)
    env = make_env(exp, seed=int(seeds[0]), dr_spec=dr_spec, shortcut_mode=shortcut_mode)
    rec: dict[str, list] = {
        k: []
        for k in ("obs_flat", "act_mean", "log_std", "action", "q_min", "penult", "is_success")
    }
    success, dr_values, dr_names = [], [], None
    for ep, sd in enumerate(seeds):
        obs, info = env.reset(seed=int(sd))
        names = sorted(info["dr_params"])
        dr_names = dr_names or names
        dr_values.append([info["dr_params"][k] for k in dr_names])
        ep_rec: dict[str, list] = {k: [] for k in rec}
        for t in range(EPISODE_STEPS):
            x = flatten_obs(obs)
            if obs_hook is not None:
                x = obs_hook(x, t, ep)
            xt = torch.as_tensor(x[None], dtype=torch.float64)
            with torch.no_grad():
                mean, log_std = pol.dist(xt)
                a = torch.tanh(mean)
                if log:
                    ep_rec["q_min"].append(float(pol.q(xt, a)[0]))
                    ep_rec["penult"].append(pol.penultimate(xt)[0].numpy())
            action = a[0].numpy().astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            ep_rec["is_success"].append(bool(info["is_success"]))
            if log:
                ep_rec["obs_flat"].append(x)
                ep_rec["act_mean"].append(mean[0].numpy())
                ep_rec["log_std"].append(log_std[0].numpy())
                ep_rec["action"].append(action)
            if term or trunc:
                break
        success.append(ep_rec["is_success"][-1])
        for k in rec:
            if ep_rec[k]:
                rec[k].append(ep_rec[k])
    out = {
        "success": np.array(success, dtype=bool),
        "dr_values": np.array(dr_values, dtype=np.float32),
        "dr_names": np.array(dr_names),
        "ep_seed": np.array(seeds, dtype=np.int64),
        "is_success": np.array(rec["is_success"], dtype=bool),
    }
    if log:
        for k, dt in (
            ("obs_flat", np.float32),
            ("act_mean", np.float32),
            ("log_std", np.float32),
            ("action", np.float32),
            ("q_min", np.float32),
            ("penult", np.float32),
        ):
            out[k] = np.array(rec[k], dtype=dt)
    return out


__all__ = ["resolve_checkpoint", "load_model", "run_episodes", "split_flat"]
