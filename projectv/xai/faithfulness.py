"""C1: faithfulness of attribution methods (mvp.md §5).

    python -m projectv.xai.faithfulness --task reach_controller
    python -m projectv.xai.faithfulness --task push_controller   # Push env, target = goal
    python -m projectv.xai.faithfulness --task shortcut          # reach_shortcut, seeds 0-2
    python -m projectv.xai.faithfulness --task push --exp push_dr

Ground truth A (analytic PD controller). The controller is linear in its inputs (pre-clip mean),
so the analytic group importance is sum_{i in g} |df/dx_i * (x_i - base_i)|, with f the
CLAUDE.md scalar target and base the pool mean. Metrics per state: precision@k (k = number of
relevant groups), Spearman rho vs analytic group importance, attribution share on irrelevant
groups, runtime (ms). Summary: mean with 95% percentile-bootstrap CI over states.
Outputs: c1_{task}_per_state.csv, c1_{task}_summary.csv, run_meta_{task}.json.

Ground truth B (planted shortcut, `--task shortcut`). Per seed: success over the same episodes
with the hint intact vs replaced by a random point; per method, the share assigned to `hint` and
`goal` on states sampled from the intact rollouts. Outputs: c1_shortcut_summary.csv,
c1_shortcut_per_state.csv, c1_shortcut_success.csv, run_meta_shortcut.json.

Learned Push policies (`--task push --exp push_dr`). Per seed: open-loop importance (mean KL
under marginal erasure over pool states), closed-loop importance (success drop with the group
erased at every step, xai/closed_loop.py), method rankings (mean group shares over nominal
states), Kendall tau and precision@k vs the closed-loop ranking, deletion curves and pAUC.
Per-seed metrics are averaged over seeds; CIs are two-level bootstraps (seeds, then
states/episodes within seed). The deletion sets are fixed by each seed's point-estimate
ranking (the bootstrap resamples episodes and seeds, not the rankings). Outputs:
c1_{exp}_summary.csv, c1_{exp}_deletion.csv, c1_{exp}_importance.csv, run_meta_{exp}.json.

Learned-policy tasks keep per-seed partial results in {out}/partial/ and skip them on rerun when
their settings match.

Seeds: the calib pool is step-log cell 0 (seeds eval_episode_seed(s, 0, ep)) or, if that log is
missing, fresh nominal rollouts with the same seeds. Attribution states and closed-loop episodes
use C1_SEED_OFFSET + eval_episode_seed(s, purpose, ep) so they never coincide with pool or
held-out episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projectv.analysis.stats import bootstrap_ci, two_level_bootstrap_ci
from projectv.control.analytic_controller import RELEVANT_GROUPS, PDController
from projectv.envs.dr_wrapper import fixed_spec
from projectv.envs.make_env import make_env
from projectv.utils.io import write_run_meta
from projectv.utils.seeding import eval_episode_seed, set_global_seed
from projectv.xai.attributions import ALL_METHODS, attribute, grad_x_delta, group_shares
from projectv.xai.closed_loop import (
    closed_loop_success,
    deletion_pauc,
    kendall_tau,
    open_loop_importance,
    precision_at_k_sets,
    sb3_runner,
    top_k,
)
from projectv.xai.groups import REDUNDANT_CLUSTER, make_groups
from projectv.xai.policy_fn import SB3Policy, controller_policy, flatten_obs

CALIB_CELL, EVAL_CELL = 0, 1  # cell ids used only to separate seeds of the two splits
C1_SEED_OFFSET = 5_000_000
STATES_PURPOSE, EPISODES_PURPOSE = 1, 2
OPEN_LOOP = "open_loop"


# ================================================================================================
# Ground truth A: analytic controller
# ================================================================================================
def rollout_states(task: str, ctrl: PDController, n_eps: int, cell: int, seed: int) -> np.ndarray:
    env = make_env({"task": task, "dr_cfg": {}}, seed=seed, dr_spec=fixed_spec({}))
    states = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=eval_episode_seed(seed, cell, ep))
        for _ in range(50):
            states.append(flatten_obs(obs))
            obs, *_ = env.step(ctrl.act(obs))
    return np.array(states)


def _denoise(scores: dict[str, float], rel_tol: float = 1e-9) -> dict[str, float]:
    """Zero out floating-point noise (|v| < rel_tol * total) so exact-zero ties in the ground
    truth are not ranked by numerical noise (affects least-squares methods such as KernelSHAP)."""
    tot = sum(abs(v) for v in scores.values())
    return {g: (v if abs(v) >= rel_tol * tot else 0.0) for g, v in scores.items()}


def precision_at_k(scores: dict[str, float], relevant: set[str]) -> float:
    k = len(relevant)
    top = sorted(scores, key=lambda g: scores[g], reverse=True)[:k]
    return len(set(top) & relevant) / k


def controller_c1(task: str, n_pool_eps: int, n_states: int, seed: int, methods) -> pd.DataFrame:
    env_task = task.replace("_controller", "")
    ctrl = PDController(env_task)
    pol = controller_policy(ctrl, env_task)
    groups = make_groups(env_task)
    pool = rollout_states(env_task, ctrl, n_pool_eps, CALIB_CELL, seed)
    rng = np.random.default_rng(seed)
    eval_states = rollout_states(env_task, ctrl, max(1, n_states // 50) + 1, EVAL_CELL, seed)
    eval_states = eval_states[rng.choice(len(eval_states), n_states, replace=False)]
    base = pool.mean(axis=0)
    relevant = set(RELEVANT_GROUPS)
    rows = []
    for i, s in enumerate(eval_states):
        a = grad_x_delta(pol, s, base)  # exact for the linear controller
        truth = {g: float(np.abs(a[idx]).sum()) for g, idx in groups.items()}
        for m in methods:
            t = time.perf_counter()
            scores = attribute(m, pol, s, groups, pool, np.random.default_rng(seed * 100_003 + i))
            ms = (time.perf_counter() - t) * 1e3
            shares = group_shares(scores)
            names = list(groups)
            clean = _denoise(scores)
            rho = spearmanr([clean[g] for g in names], [truth[g] for g in names]).statistic
            rows.append(
                {
                    "task": task,
                    "state": i,
                    "method": m,
                    "precision_at_k": precision_at_k(scores, relevant),
                    "spearman": float(rho),
                    "irrelevant_share": sum(v for g, v in shares.items() if g not in relevant),
                    "runtime_ms": ms,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (task, m), d in df.groupby(["task", "method"], sort=False):
        row = {"task": task, "method": m, "n_states": len(d)}
        for col in ("precision_at_k", "spearman", "irrelevant_share", "runtime_ms"):
            v = d[col].to_numpy()
            fn = np.nanmedian if col == "runtime_ms" else np.nanmean
            p, lo, hi = bootstrap_ci(v, stat=fn)
            row.update({col: p, f"{col}_lo": lo, f"{col}_hi": hi})
        out.append(row)
    return pd.DataFrame(out)


# ================================================================================================
# Shared pieces for learned policies
# ================================================================================================
def c1_seeds(train_seed: int, purpose: int, n: int) -> list[int]:
    return [C1_SEED_OFFSET + eval_episode_seed(train_seed, purpose, ep) for ep in range(n)]


def load_pool(model, exp: dict, seed: int, root: str | Path, n_pool_eps: int) -> np.ndarray:
    """Nominal calib pool: obs_flat of step-log cell 0 if present, else fresh nominal rollouts
    with the same episode seeds. Returns (N, d) float64."""
    from projectv.eval.cells import NOMINAL
    from projectv.eval.rollout import run_episodes
    from projectv.eval.steplog import load_steplog, steplog_path

    path = steplog_path(root, exp["name"], seed, 0)
    if path.exists():
        obs = load_steplog(path)["obs_flat"]
    else:
        seeds = [eval_episode_seed(seed, 0, ep) for ep in range(n_pool_eps)]
        obs = run_episodes(model, exp, NOMINAL.dr_spec(), seeds, log=True)["obs_flat"]
    return obs.reshape(-1, obs.shape[-1]).astype(np.float64)


def pool_source(exp: dict, seed: int, root: str | Path, n_pool_eps: int) -> str:
    from projectv.eval.steplog import steplog_path

    path = steplog_path(root, exp["name"], seed, 0)
    return f"steplog:{path.as_posix()}" if path.exists() else f"fresh:{n_pool_eps}"


def sample_states(rec: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    S = rec["obs_flat"].reshape(-1, rec["obs_flat"].shape[-1]).astype(np.float64)
    return S[rng.choice(len(S), n, replace=n > len(S))]


def attribute_states(
    pol, states: np.ndarray, groups: dict[str, np.ndarray], pool: np.ndarray, methods, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Group shares (n_states, n_methods, n_groups) and runtime_ms (n_states, n_methods)."""
    names = list(groups)
    shares = np.zeros((len(states), len(methods), len(names)))
    runtime = np.zeros((len(states), len(methods)))
    for i, s in enumerate(states):
        for j, m in enumerate(methods):
            rng = np.random.default_rng([seed, i, j])
            t = time.perf_counter()
            scores = attribute(m, pol, s, groups, pool, rng)
            runtime[i, j] = (time.perf_counter() - t) * 1e3
            sh = group_shares(scores)
            shares[i, j] = [sh[g] for g in names]
    return shares, runtime


def cached_stage(
    path: Path, key: dict, compute: Callable[[], dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Load `path` if it was written with the same settings `key`, else compute and save."""
    key_s = json.dumps(key, sort_keys=True, default=str)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["cfg_key"]) == key_s:
                return {k: z[k] for k in z.files if k != "cfg_key"}
    rec = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, cfg_key=np.array(key_s), **rec)
    tmp.replace(path)
    return rec


def load_her_model(path: str | Path, exp: dict):
    """Load a SAC/TQC + HER checkpoint for inference. SB3 refuses to rebuild a HerReplayBuffer
    without an env (eval/rollout.load_model fails on such checkpoints), so a throwaway env is
    attached and the buffer is shrunk to avoid allocating 1M transitions."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda: make_env(exp, seed=0)])
    kw = dict(env=venv, device="cpu", custom_objects={"buffer_size": 1_000})
    try:
        return SAC.load(path, **kw)
    except Exception:  # TQC fallback checkpoints
        from sb3_contrib import TQC

        return TQC.load(path, **kw)


def _model_loader(exp: dict, seed: int, root: str | Path) -> Callable[[], object]:
    cache: dict[str, object] = {}

    def get():
        if "m" not in cache:
            from projectv.eval.rollout import resolve_checkpoint

            cache["m"] = load_her_model(resolve_checkpoint(exp["name"], seed, root), exp)
        return cache["m"]

    return get


def _lazy(fn: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    cache: dict[str, np.ndarray] = {}

    def get():
        if "v" not in cache:
            cache["v"] = fn()
        return cache["v"]

    return get


def _per_seed_str(vals) -> str:
    return ";".join(f"{v:.4g}" for v in np.atleast_1d(vals))


def _ci_cols(row: dict, name: str, pt, lo, hi) -> None:
    row.update({name: float(pt), f"{name}_lo": float(lo), f"{name}_hi": float(hi)})


# ================================================================================================
# Ground truth B: planted shortcut
# ================================================================================================
def shortcut_seed(get_model, exp: dict, seed: int, cfg: dict, partial: Path) -> dict:
    from projectv.eval.cells import NOMINAL
    from projectv.eval.rollout import run_episodes

    groups = make_groups("reach", shortcut=True)
    names = list(groups)
    key = {
        "stage": "shortcut",
        **{k: cfg[k] for k in ("episodes", "n_states", "methods", "n_pool_eps")},
        "pool": pool_source(exp, seed, cfg["root"], cfg["n_pool_eps"]),
    }

    def compute():
        model = get_model()
        pool = load_pool(model, exp, seed, cfg["root"], cfg["n_pool_eps"])
        seeds = c1_seeds(seed, EPISODES_PURPOSE, cfg["episodes"])
        dr = NOMINAL.dr_spec()
        intact = run_episodes(model, exp, dr, seeds, log=True, shortcut_mode="intact")
        rand = run_episodes(model, exp, dr, seeds, shortcut_mode="random")
        states = sample_states(intact, cfg["n_states"], np.random.default_rng([seed, 1]))
        shares, runtime = attribute_states(
            SB3Policy(model), states, groups, pool, cfg["methods"], seed
        )
        return {
            "hint_share": shares[:, :, names.index("hint")],
            "goal_share": shares[:, :, names.index("goal")],
            "runtime_ms": runtime,
            "succ": np.stack([intact["success"], rand["success"]]).astype(np.float64),
            "ep_seed": np.array(seeds, dtype=np.int64),
        }

    return cached_stage(partial / f"shortcut_s{seed}.npz", key, compute)


def shortcut_summary(
    per_seed: dict[int, dict], methods: list[str], n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = list(per_seed)
    clusters = [
        {
            "h": d["hint_share"],
            "g": d["goal_share"],
            "det": (d["hint_share"] > d["goal_share"]).astype(np.float64),
            "rt": d["runtime_ms"],
            "succ": d["succ"],
        }
        for d in per_seed.values()
    ]
    axes = {"succ": 1}

    def pooled(key):
        return lambda cs: np.concatenate([c[key] for c in cs]).mean(0)

    def drop(cs):
        s = np.concatenate([c["succ"] for c in cs], axis=1)
        return s[0].mean() - s[1].mean()

    def rt(cs):
        return np.median(np.concatenate([c["rt"] for c in cs]), axis=0)

    ci = {
        "hint_share": two_level_bootstrap_ci(pooled("h"), clusters, axes, n_boot),
        "goal_share": two_level_bootstrap_ci(pooled("g"), clusters, axes, n_boot),
        "detected_rate": two_level_bootstrap_ci(pooled("det"), clusters, axes, n_boot),
        "runtime_ms": two_level_bootstrap_ci(rt, clusters, axes, n_boot),
    }
    d_pt, d_lo, d_hi = two_level_bootstrap_ci(drop, clusters, axes, n_boot)
    s_int = np.concatenate([c["succ"][0] for c in clusters]).mean()
    s_rnd = np.concatenate([c["succ"][1] for c in clusters]).mean()
    drop_seed = [drop([c]) for c in clusters]
    rows = []
    for j, m in enumerate(methods):
        row = {"method": m, "n_seeds": len(seeds), "n_states": sum(len(c["h"]) for c in clusters)}
        for name, (pt, lo, hi) in ci.items():
            _ci_cols(row, name, pt[j], lo[j], hi[j])
        row["hint_share_per_seed"] = _per_seed_str([c["h"][:, j].mean() for c in clusters])
        row.update({"success_intact": s_int, "success_random": s_rnd})
        _ci_cols(row, "success_drop", d_pt, d_lo, d_hi)
        row["success_drop_per_seed"] = _per_seed_str(drop_seed)
        rows.append(row)
    per_state, succ = [], []
    for sd, d in per_seed.items():
        for i in range(len(d["hint_share"])):
            for j, m in enumerate(methods):
                h, g = d["hint_share"][i, j], d["goal_share"][i, j]
                per_state.append(
                    {
                        "seed": sd,
                        "state": i,
                        "method": m,
                        "hint_share": h,
                        "goal_share": g,
                        "detected": bool(h > g),
                        "runtime_ms": d["runtime_ms"][i, j],
                    }
                )
        for mi, mode in enumerate(("intact", "random")):
            for e, es in enumerate(d["ep_seed"]):
                succ.append(
                    {"seed": sd, "mode": mode, "episode": e, "ep_seed": int(es),
                     "success": bool(d["succ"][mi, e])}
                )  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(per_state), pd.DataFrame(succ)


# ================================================================================================
# Learned Push policies
# ================================================================================================
def push_units(groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every group plus the redundant cluster erased jointly (last unit)."""
    return {**groups, "cluster": np.concatenate([groups[g] for g in REDUNDANT_CLUSTER])}


def push_seed(get_model, exp: dict, seed: int, cfg: dict, partial: Path) -> dict:
    from projectv.eval.cells import NOMINAL
    from projectv.eval.rollout import run_episodes

    groups = make_groups(exp["task"], history=exp.get("history", False))
    names = list(groups)
    units = push_units(groups)
    name = exp["name"]
    pool_key = pool_source(exp, seed, cfg["root"], cfg["n_pool_eps"])
    get_pool = _lazy(lambda: load_pool(get_model(), exp, seed, cfg["root"], cfg["n_pool_eps"]))

    # --- attribution shares on nominal states ---
    attr_key = {"stage": "attr", "pool": pool_key, "n_states": cfg["n_states"]}
    attr_key["methods"] = cfg["methods"]

    def attr():
        model = get_model()
        n_eps = max(2, -(-cfg["n_states"] // 10))
        rec = run_episodes(
            model, exp, NOMINAL.dr_spec(), c1_seeds(seed, STATES_PURPOSE, n_eps), log=True
        )
        states = sample_states(rec, cfg["n_states"], np.random.default_rng([seed, 1]))
        shares, runtime = attribute_states(
            SB3Policy(model), states, groups, get_pool(), cfg["methods"], seed
        )
        return {"shares": shares, "runtime_ms": runtime}

    a = cached_stage(partial / f"{name}_s{seed}_attr.npz", attr_key, attr)

    # --- open-loop importance ---
    open_key = {"stage": "open", "pool": pool_key, "n_open": cfg["n_open"], "M": cfg["M"]}

    def open_loop():
        pool = get_pool()
        rng = np.random.default_rng([seed, 2])
        S = pool[rng.choice(len(pool), min(cfg["n_open"], len(pool)), replace=False)]
        kl = open_loop_importance(SB3Policy(get_model()), S, units, pool, rng, cfg["M"])
        return {"kl": kl}

    o = cached_stage(partial / f"{name}_s{seed}_open.npz", open_key, open_loop)

    # --- closed loop: per-unit erasure and deletion curves ---
    rankers = list(cfg["methods"]) + [OPEN_LOOP]
    scores = np.vstack([a["shares"].mean(0), o["kl"].mean(0)[: len(names)]])
    k_max = min(cfg["k_max"], len(names))
    orders = np.array([[names[i] for i in top_k(sc, k_max)] for sc in scores])
    closed_key = {
        "stage": "closed",
        "pool": pool_key,
        "episodes": cfg["episodes"],
        "orders": orders.tolist(),
        "units": list(units),
    }

    def closed():
        pool = get_pool()
        runner = sb3_runner(get_model(), exp, NOMINAL.dr_spec())
        seeds = c1_seeds(seed, EPISODES_PURPOSE, cfg["episodes"])
        rs = seed * 1_000 + 7  # same pool-row sequence for every erasure (common random numbers)
        nominal = runner(seeds, None)
        unit = np.array(
            [closed_loop_success(runner, seeds, idx, pool, rs) for idx in units.values()]
        )
        memo: dict[frozenset, np.ndarray] = {}
        dele = np.zeros((len(rankers), k_max, len(seeds)))
        for r in range(len(rankers)):
            for k in range(1, k_max + 1):
                s = frozenset(orders[r, :k])
                if s not in memo:
                    idx = np.sort(np.concatenate([groups[g] for g in s]))
                    memo[s] = closed_loop_success(runner, seeds, idx, pool, rs)
                dele[r, k - 1] = memo[s]
        return {
            "nominal": nominal.astype(np.float64),
            "unit": unit.astype(np.float64),
            "deletion": dele,
            "ep_seed": np.array(seeds, dtype=np.int64),
        }

    c = cached_stage(partial / f"{name}_s{seed}_closed.npz", closed_key, closed)
    return {
        "names": names,
        "units": list(units),
        "rankers": rankers,
        "orders": orders,
        **a,
        **o,
        **c,
    }


def push_summary(
    per_seed: dict[int, dict], prec_k: int, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = next(iter(per_seed.values()))
    names, units, rankers = first["names"], first["units"], first["rankers"]
    n_g, n_m, k_max = len(names), len(rankers) - 1, first["deletion"].shape[1]
    n_u = len(units)
    clusters = []
    for d in per_seed.values():
        succ = np.vstack(
            [d["nominal"][None], d["unit"], d["deletion"].reshape(-1, d["unit"].shape[1])]
        )
        clusters.append({"shares": d["shares"], "rt": d["runtime_ms"], "kl": d["kl"], "succ": succ})
    axes = {"succ": 1}

    def parts(c):
        s = c["succ"].mean(1)
        nom, unit = s[0], s[1 : 1 + n_u]
        dele = s[1 + n_u :].reshape(len(rankers), k_max)
        closed_drop = nom - unit
        sc = np.vstack([c["shares"].mean(0), c["kl"].mean(0)[None, :n_g]])
        return nom, closed_drop, dele, sc

    def per_cluster(c):
        nom, cd, dele, sc = parts(c)
        tau = [kendall_tau(sc[r], cd[:n_g]) for r in range(len(rankers))]
        prec = [precision_at_k_sets(sc[r], cd[:n_g], prec_k) for r in range(len(rankers))]
        pauc = [deletion_pauc(nom, dele[r]) for r in range(len(rankers))]
        return np.array([tau, prec, pauc])  # (3, n_rankers)

    def metrics(cs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean([per_cluster(c) for c in cs], axis=0)

    m_pt, m_lo, m_hi = two_level_bootstrap_ci(metrics, clusters, axes, n_boot)

    def rt(cs):
        return np.median(np.concatenate([c["rt"] for c in cs]), axis=0)

    r_pt, r_lo, r_hi = two_level_bootstrap_ci(rt, clusters, axes, n_boot)
    seed_vals = np.array([per_cluster(c) for c in clusters])  # (n_seeds, 3, n_rankers)
    nominal = float(np.mean([parts(c)[0] for c in clusters]))
    rows = []
    for r, m in enumerate(rankers):
        row = {"method": m, "n_seeds": len(clusters), "nominal_success": nominal, "k": prec_k}
        for i, col in enumerate(("tau_closed", "precision_at_k", "pauc")):
            _ci_cols(row, col, m_pt[i, r], m_lo[i, r], m_hi[i, r])
            row[f"{col}_per_seed"] = _per_seed_str(seed_vals[:, i, r])
        if r < n_m:
            _ci_cols(row, "runtime_ms", r_pt[r], r_lo[r], r_hi[r])
        else:  # open-loop importance is a reference ranking, not a per-state method
            _ci_cols(row, "runtime_ms", np.nan, np.nan, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows)

    def imp(cs):
        op = np.mean([c["kl"].mean(0) for c in cs], axis=0)
        cl = np.mean([parts(c)[1] for c in cs], axis=0)
        return np.vstack([op, cl])

    i_pt, i_lo, i_hi = two_level_bootstrap_ci(imp, clusters, axes, n_boot)
    seed_imp = np.array([imp([c]) for c in clusters])
    importance = pd.DataFrame(
        [
            {
                "group": u,
                "members": "+".join(REDUNDANT_CLUSTER) if u == "cluster" else u,
                "open_loop": i_pt[0, j],
                "open_loop_lo": i_lo[0, j],
                "open_loop_hi": i_hi[0, j],
                "closed_loop": i_pt[1, j],
                "closed_loop_lo": i_lo[1, j],
                "closed_loop_hi": i_hi[1, j],
                "open_loop_per_seed": _per_seed_str(seed_imp[:, 0, j]),
                "closed_loop_per_seed": _per_seed_str(seed_imp[:, 1, j]),
                "nominal_success": nominal,
            }
            for j, u in enumerate(units)
        ]
    )
    dele_rows = []
    for sd, d in per_seed.items():
        nom = float(d["nominal"].mean())
        for r, m in enumerate(rankers):
            dele_rows.append(
                {"method": m, "k": 0, "seed": sd, "success": nom, "erased": "", "n_episodes":
                 len(d["nominal"])}
            )  # fmt: skip
            for k in range(1, k_max + 1):
                dele_rows.append(
                    {
                        "method": m,
                        "k": k,
                        "seed": sd,
                        "success": float(d["deletion"][r, k - 1].mean()),
                        "erased": "+".join(d["orders"][r, :k]),
                        "n_episodes": len(d["nominal"]),
                    }
                )
    return summary, pd.DataFrame(dele_rows), importance


# ================================================================================================
# CLI
# ================================================================================================
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", required=True, choices=["reach_controller", "push_controller", "shortcut", "push"]
    )
    p.add_argument("--exp", default=None, help="experiment (default push_dr / reach_shortcut)")
    p.add_argument("--seed", type=int, default=0, help="controller tasks: rollout seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="training seeds")
    p.add_argument("--root", default="results", help="holds ckpt/ and steplogs/")
    p.add_argument("--n-pool-eps", type=int, default=40)
    p.add_argument("--n-states", type=int, default=200)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--n-open-states", type=int, default=2000)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--prec-k", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--out", default="results/c1")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",")
    show = pd.option_context("display.width", 200, "display.max_columns", 20)

    if args.task.endswith("_controller"):
        df = controller_c1(args.task, args.n_pool_eps, args.n_states, args.seed, methods)
        df.to_csv(out / f"c1_{args.task}_per_state.csv", index=False)
        summ = summarize(df)
        summ.to_csv(out / f"c1_{args.task}_summary.csv", index=False)
        write_run_meta(
            out, vars(args), sys.argv, t0, filename=f"run_meta_{args.task}.json", task=args.task
        )
        with show:
            print(summ[["method", "precision_at_k", "spearman", "irrelevant_share", "runtime_ms"]])
        return

    from projectv.utils.config import load_exp

    exp_name = args.exp or ("reach_shortcut" if args.task == "shortcut" else "push_dr")
    exp = load_exp(f"configs/exp/{exp_name}.yaml")
    cfg = {
        "root": args.root,
        "n_pool_eps": args.n_pool_eps,
        "n_states": args.n_states,
        "episodes": args.episodes,
        "n_open": args.n_open_states,
        "M": args.M,
        "k_max": args.k_max,
        "methods": methods,
    }
    partial = out / "partial"
    per_seed = {}
    for sd in args.seeds:
        get_model = _model_loader(exp, sd, args.root)
        if args.task == "shortcut":
            assert exp.get("shortcut", False), f"{exp_name} has no shortcut hint"
            per_seed[sd] = shortcut_seed(get_model, exp, sd, cfg, partial)
        else:
            per_seed[sd] = push_seed(get_model, exp, sd, cfg, partial)
        print(f"[c1 {args.task}] seed {sd} done ({time.time() - t0:.0f}s)", flush=True)

    if args.task == "shortcut":
        summ, per_state, succ = shortcut_summary(per_seed, methods, args.n_boot)
        summ.to_csv(out / "c1_shortcut_summary.csv", index=False)
        per_state.to_csv(out / "c1_shortcut_per_state.csv", index=False)
        succ.to_csv(out / "c1_shortcut_success.csv", index=False)
        tag = "shortcut"
        cols = ["method", "hint_share", "goal_share", "detected_rate", "success_drop"]
    else:
        summ, dele, imp = push_summary(per_seed, args.prec_k, args.n_boot)
        summ.to_csv(out / f"c1_{exp_name}_summary.csv", index=False)
        dele.to_csv(out / f"c1_{exp_name}_deletion.csv", index=False)
        imp.to_csv(out / f"c1_{exp_name}_importance.csv", index=False)
        tag = exp_name
        cols = ["method", "tau_closed", "precision_at_k", "pauc", "runtime_ms"]
    write_run_meta(
        out, vars(args), sys.argv, t0, filename=f"run_meta_{tag}.json", task=args.task, exp=exp_name
    )
    with show:
        print(summ[cols])


if __name__ == "__main__":
    main()
