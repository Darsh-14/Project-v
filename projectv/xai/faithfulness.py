"""C1: faithfulness of attribution methods (mvp.md §5).

Implemented so far: ground truth A, the analytic PD controller.

    python -m projectv.xai.faithfulness --task reach_controller
    python -m projectv.xai.faithfulness --task push_controller   # Push env, target = goal

The controller is linear in its inputs (pre-clip mean), so the analytic group importance is
sum_{i in g} |df/dx_i * (x_i - base_i)|, with f the CLAUDE.md scalar target and base the pool
mean. Metrics per state: precision@k (k = number of relevant groups), Spearman rho vs analytic
group importance, attribution share on irrelevant groups, runtime (ms). Summary: mean with 95%
percentile-bootstrap CI over states.

Outputs: {out}/c1_{task}_per_state.csv, {out}/c1_{task}_summary.csv, {out}/run_meta.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projectv.analysis.stats import bootstrap_ci
from projectv.control.analytic_controller import RELEVANT_GROUPS, PDController
from projectv.envs.dr_wrapper import fixed_spec
from projectv.envs.make_env import make_env
from projectv.utils.io import write_run_meta
from projectv.utils.seeding import eval_episode_seed, set_global_seed
from projectv.xai.attributions import ALL_METHODS, attribute, grad_x_delta, group_shares
from projectv.xai.groups import make_groups
from projectv.xai.policy_fn import controller_policy, flatten_obs

CALIB_CELL, EVAL_CELL = 0, 1  # cell ids used only to separate seeds of the two splits


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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["reach_controller", "push_controller"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pool-eps", type=int, default=40)
    p.add_argument("--n-states", type=int, default=200)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--out", default="results/c1")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",")
    df = controller_c1(args.task, args.n_pool_eps, args.n_states, args.seed, methods)
    df.to_csv(out / f"c1_{args.task}_per_state.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / f"c1_{args.task}_summary.csv", index=False)
    write_run_meta(out, vars(args), sys.argv, t0, task=args.task)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ[["method", "precision_at_k", "spearman", "irrelevant_share", "runtime_ms"]])


if __name__ == "__main__":
    main()
