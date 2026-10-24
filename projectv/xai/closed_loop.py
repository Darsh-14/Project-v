"""Closed- and open-loop group importance for C1 (mvp.md §5), plus ranking metrics.

Closed-loop erasure: at EVERY step of the episode the erased features are replaced by the
corresponding values of one pool row drawn uniformly at random (marginal erasure, M=1 per step,
since a single action must be issued). All groups erased together take their values from the
same row, so jointly erased groups keep their joint pool distribution. The environment evolves
from the true state; only what the policy sees changes (eval/rollout.py obs_hook).

A *runner* is any callable `runner(seeds, obs_hook) -> success (E,) bool`. `sb3_runner` wraps
eval/rollout.run_episodes for SB3 models; `dist_runner` rolls out any Gaussian policy function
(policy_fn interface) with a deterministic squashed mean, so analytic policies go through the
same erasure code path as learned ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau

from projectv.envs.make_env import make_env
from projectv.envs.obs_layout import EPISODE_STEPS
from projectv.xai.attributions import DT, gaussian_kl
from projectv.xai.policy_fn import flatten_obs

ObsHook = Callable[[np.ndarray, int, int], np.ndarray]
Runner = Callable[[Sequence[int], ObsHook | None], np.ndarray]


# --- erasure + rollouts -------------------------------------------------------------------------
def erasure_hook(idx: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> ObsHook:
    """Replace features `idx` with those of a freshly drawn pool row at every call."""
    idx = np.asarray(idx, dtype=int)

    def hook(x: np.ndarray, t: int, ep: int) -> np.ndarray:
        y = np.array(x, dtype=np.float64, copy=True)
        y[idx] = pool[rng.integers(len(pool)), idx]
        return y

    return hook


def sb3_runner(model, exp: dict, dr_spec: dict, shortcut_mode: str = "intact") -> Runner:
    from projectv.eval.rollout import run_episodes

    def run(seeds: Sequence[int], hook: ObsHook | None) -> np.ndarray:
        rec = run_episodes(model, exp, dr_spec, seeds, obs_hook=hook, shortcut_mode=shortcut_mode)
        return rec["success"]

    return run


def dist_runner(
    pol, exp: dict, dr_spec: dict, squash: Callable[[np.ndarray], np.ndarray] = np.tanh
) -> Runner:
    """Deterministic rollouts of a policy_fn-style policy: action = squash(mean)."""

    def run(seeds: Sequence[int], hook: ObsHook | None) -> np.ndarray:
        env = make_env(exp, seed=int(seeds[0]), dr_spec=dr_spec)
        succ = []
        for ep, sd in enumerate(seeds):
            obs, _ = env.reset(seed=int(sd))
            info: dict = {"is_success": False}
            for t in range(EPISODE_STEPS):
                x = flatten_obs(obs)
                if hook is not None:
                    x = hook(x, t, ep)
                with torch.no_grad():
                    mean, _ = pol.dist(torch.as_tensor(x[None], dtype=DT))
                action = squash(mean[0].numpy()).astype(np.float32)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            succ.append(bool(info["is_success"]))
        return np.array(succ, dtype=bool)

    return run


def closed_loop_success(
    runner: Runner, seeds: Sequence[int], idx: np.ndarray, pool: np.ndarray, rng_seed: int
) -> np.ndarray:
    """Per-episode success with features `idx` erased at every step. Using the same `rng_seed`
    for every erasure unit gives common random numbers (same pool-row sequence)."""
    if len(idx) == 0:
        return runner(seeds, None)
    return runner(seeds, erasure_hook(idx, pool, np.random.default_rng(rng_seed)))


# --- open loop ----------------------------------------------------------------------------------
def open_loop_importance(
    pol,
    states: np.ndarray,
    units: dict[str, np.ndarray],
    pool: np.ndarray,
    rng: np.random.Generator,
    M: int = 8,
) -> np.ndarray:
    """KL(pi(.|s) || pi(.|E_u(s))) per state and unit, marginal erasure averaged over M pool
    rows. The same M rows per state are used for every unit. Returns (n_states, n_units)."""
    S = np.asarray(states, dtype=np.float64)
    n = len(S)
    rows = pool[rng.integers(len(pool), size=n * M)]
    with torch.no_grad():
        m0, l0 = pol.dist(torch.as_tensor(S, dtype=DT))
        m0, l0 = m0.repeat_interleave(M, 0), l0.repeat_interleave(M, 0)
        out = np.empty((n, len(units)))
        for j, idx in enumerate(units.values()):
            X = np.repeat(S, M, axis=0)
            X[:, idx] = rows[:, idx]
            m, ls = pol.dist(torch.as_tensor(X, dtype=DT))
            out[:, j] = gaussian_kl(m0, l0, m, ls).numpy().reshape(n, M).mean(1)
    return out


# --- ranking metrics ----------------------------------------------------------------------------
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, ties broken by position (stable)."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b between two score vectors over the same items (NaN if either is constant)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def precision_at_k_sets(scores: np.ndarray, reference: np.ndarray, k: int) -> float:
    """|top-k(scores) & top-k(reference)| / k (NaN if the reference is constant)."""
    if np.ptp(np.asarray(reference, dtype=np.float64)) == 0:
        return float("nan")
    return len(set(top_k(scores, k)) & set(top_k(reference, k))) / k


def deletion_pauc(nominal: float, success_k: np.ndarray) -> float:
    """Normalised area of the success drop over k = 1..K: mean_k (nominal - success_k) / nominal.
    1 = the first group already destroys all success; 0 = no drop. NaN if nominal == 0."""
    if nominal <= 0:
        return float("nan")
    return float(np.mean(nominal - np.asarray(success_k, dtype=np.float64)) / nominal)
