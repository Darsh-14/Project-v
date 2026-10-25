"""C4 implicit system identification: linear (ridge) probes (mvp.md §8).

Targets (per episode, from `dr_values` / `dr_names`):
- `log_friction`: log obj_friction + log table_friction (the object-table sliding friction is
  mu0 * obj_friction * table_friction, see envs/dr_wrapper.py);
- `log_mass`: log obj_mass;
- `gain`: action-scale multiplier.

Feature sets:
- (a) `penult`: trained actor penultimate activations (`latent_pi`);
- (b) `obs_single`: raw flat state, current slot only (+ achieved/desired goal);
- (c) `obs_stacked`: raw flat state incl. history slots and past actions (DR+Hist only);
- (d) `penult_random`: penultimate activations of a randomly initialised actor of the same
  architecture (built from the model's policy class, fixed torch seed).

Protocol: ridge on standardised features, 5-fold GroupKFold over episodes; alpha chosen by an
inner 3-fold GroupKFold over the training folds' episodes. R^2 is computed on the pooled
out-of-fold predictions. CIs: per seed, bootstrap over episodes; across seeds, hierarchical
bootstrap (resample seeds, then episodes within each) of the mean per-seed R^2.

Claim rule (mvp.md §8): implicit sysID is claimed for (exp, target) only if (a) > (b) and
(a) > (d) with non-overlapping CIs, i.e. lo(a) > hi(b) and lo(a) > hi(d), across seeds.

States come from training-distribution episodes (`--source train`, default): fresh rollouts under
the train DR spec (configs/dr/train.yaml for every condition, so the NoDR policy is probed on the
same parameter distribution), cached under `{out}/probe_rollouts/`. `--source steplogs` uses the
held-out step logs instead (all cells pooled).
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projectv.eval.steplog import load_steplog, save_steplog
from projectv.utils.config import load_exp, load_yaml
from projectv.utils.io import write_json, write_run_meta
from projectv.utils.seeding import set_global_seed
from projectv.xai.groups import make_groups, slot_of

TARGETS = ("log_friction", "log_mass", "gain")
FEATURES = ("penult", "obs_single", "obs_stacked", "penult_random")
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
EXPS = {
    "push": ("push_nodr", "push_dr", "push_drhist"),
    "reach": ("reach_nodr", "reach_dr", "reach_drhist"),  # Gate G2 track
}
TRAIN_DR = "configs/dr/train.yaml"
PROBE_SEED_OFFSET = 1_000_000  # episode seeds disjoint from eval seeds (CLAUDE.md §7)


# --- targets / features ---------------------------------------------------------------------
def episode_targets(dr_values: np.ndarray, dr_names) -> dict[str, np.ndarray]:
    names = [str(n) for n in dr_names]
    v = np.asarray(dr_values, dtype=np.float64)

    def col(n: str) -> np.ndarray:
        return v[:, names.index(n)] if n in names else np.ones(len(v))

    return {
        "log_friction": np.log(col("obj_friction")) + np.log(col("table_friction")),
        "log_mass": np.log(col("obj_mass")),
        "gain": col("gain"),
    }


def current_slot_idx(groups: dict[str, np.ndarray]) -> np.ndarray:
    """Flat indices of the current slot (no history slots, no past actions)."""
    keep = [idx for g, idx in groups.items() if slot_of(g) == 0 and not g.startswith("act@")]
    return np.sort(np.concatenate(keep))


def random_actor_penultimate(model, seed: int = 12345) -> Callable[[np.ndarray], np.ndarray]:
    """Penultimate activations of a fresh, untrained actor with the model's architecture."""
    from projectv.xai.policy_fn import SB3Policy

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = model.policy_class(
            model.observation_space, model.action_space, model.lr_schedule, **model.policy_kwargs
        )
    shim = types.SimpleNamespace(policy=policy, observation_space=model.observation_space)
    pol = SB3Policy(shim)

    def fn(X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xt = torch.as_tensor(X[i : i + 4096], dtype=torch.float64)
                out.append(pol.penultimate(xt).numpy())
        return np.concatenate(out).astype(np.float32)

    return fn


# --- probe ----------------------------------------------------------------------------------
def episode_folds(episode: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) per fold; every episode lies entirely in one test fold."""
    episode = np.asarray(episode)
    return list(GroupKFold(n_splits=n_splits).split(episode[:, None], groups=episode))


def probe_oof(
    X: np.ndarray,
    y: np.ndarray,
    episode: np.ndarray,
    n_splits: int = 5,
    alphas: tuple[float, ...] = ALPHAS,
    inner_splits: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold ridge predictions with GroupKFold over episodes; alpha per outer fold from
    an inner GroupKFold. Returns (oof predictions, chosen alphas)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred = np.full(len(y), np.nan)
    chosen = []
    for tr, te in episode_folds(episode, n_splits):
        gs = GridSearchCV(
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": list(alphas)},
            cv=GroupKFold(n_splits=inner_splits),
            scoring="r2",
        )
        gs.fit(X[tr], y[tr], groups=episode[tr])
        pred[te] = gs.predict(X[te])
        chosen.append(float(gs.best_params_["ridge__alpha"]))
    return pred, chosen


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else np.nan


def _episode_index(episode: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(episode, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]


def bootstrap_r2(
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """runs: per seed (y, oof_pred, episode). Point = mean per-seed R^2. CI: resample seeds,
    then episodes within each chosen seed (one run -> plain episode bootstrap)."""
    rng = np.random.default_rng(seed)
    eps = [_episode_index(e) for _, _, e in runs]
    point = float(np.mean([r2(y, p) for y, p, _ in runs]))
    boots = []
    for _ in range(n_boot):
        vals = []
        for k in rng.integers(len(runs), size=len(runs)):
            y, p, _ = runs[k]
            pick = rng.integers(len(eps[k]), size=len(eps[k]))
            idx = np.concatenate([eps[k][i] for i in pick])
            vals.append(r2(y[idx], p[idx]))
        boots.append(np.nanmean(vals))
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def sysid_claim(a: tuple[float, float], b: tuple[float, float], d: tuple[float, float]) -> bool:
    """(lo, hi) CIs of R^2 for features (a), (b), (d). Claim iff a beats b and d with
    non-overlapping CIs."""
    return bool(a[0] > b[1] and a[0] > d[1])


# --- data -----------------------------------------------------------------------------------
def probe_data(
    rec: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    history: bool,
    random_penult: Callable[[np.ndarray], np.ndarray] | None,
    max_states: int | None = None,
    seed: int = 0,
) -> dict:
    """Flatten an (E, T, .) rollout record into per-state features, targets and episode ids."""
    obs = np.asarray(rec["obs_flat"], dtype=np.float64)
    E, T, d = obs.shape
    X_obs = obs.reshape(E * T, d)
    episode = np.repeat(np.arange(E), T)
    tgt = {
        k: np.repeat(v, T) for k, v in episode_targets(rec["dr_values"], rec["dr_names"]).items()
    }
    feats = {
        "penult": np.asarray(rec["penult"]).reshape(E * T, -1),
        "obs_single": X_obs[:, current_slot_idx(groups)],
    }
    if history:
        feats["obs_stacked"] = X_obs
    sel = np.arange(E * T)
    if max_states is not None and E * T > max_states:
        sel = np.sort(np.random.default_rng(seed).choice(E * T, max_states, replace=False))
    if random_penult is not None:
        feats["penult_random"] = random_penult(X_obs[sel])
    return {
        "features": {k: (v if k == "penult_random" else v[sel]) for k, v in feats.items()},
        "targets": {k: v[sel] for k, v in tgt.items()},
        "episode": episode[sel],
    }


def rollout_record(
    model, exp_cfg: dict, seed: int, n_eps: int, cache: Path | None, dr_path: str = TRAIN_DR
) -> dict[str, np.ndarray]:
    """Train-DR rollouts with per-step logging, cached as a step-log npz."""
    if cache is not None and cache.exists():
        return load_steplog(cache)
    from projectv.eval.rollout import run_episodes

    seeds = [PROBE_SEED_OFFSET + 10_000 * seed + ep for ep in range(n_eps)]
    rec = run_episodes(model, exp_cfg, load_yaml(dr_path), seeds, log=True)
    if cache is not None:
        save_steplog(cache, rec, cell_id=-1)
    return rec


def steplog_record(root: Path, exp: str, seed: int) -> dict[str, np.ndarray] | None:
    d = root / "steplogs" / f"{exp}_s{seed}"
    files = [f for f in sorted(d.glob("cell*.npz")) if not f.name.endswith(".tmp.npz")]
    if not files:
        return None
    logs = [load_steplog(f) for f in files]
    keys = ("obs_flat", "penult", "dr_values")
    out = {k: np.concatenate([lg[k] for lg in logs]) for k in keys}
    out["dr_names"] = logs[0]["dr_names"]
    return out


# --- driver ---------------------------------------------------------------------------------
def probe_table(
    exp: str,
    per_seed: dict[int, dict],
    n_splits: int = 5,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """per_seed[seed] = probe_data(...) output. One row per (target, features, scope)."""
    rows = []
    for target in TARGETS:
        cis: dict[str, tuple[float, float]] = {}
        for feat in FEATURES:
            runs, alphas = {}, {}
            for s, data in per_seed.items():
                if feat not in data["features"]:
                    continue
                y = data["targets"][target]
                if np.var(y) == 0:
                    continue
                pred, al = probe_oof(data["features"][feat], y, data["episode"], n_splits)
                runs[s] = (y, pred, data["episode"])
                alphas[s] = al
            if not runs:
                continue
            for s, run in runs.items():
                p, lo, hi = bootstrap_r2([run], n_boot, seed=s)
                rows.append(_prow(exp, target, feat, f"seed{s}", p, lo, hi, run, alphas[s]))
            p, lo, hi = bootstrap_r2(list(runs.values()), n_boot, seed=0)
            allrun = (
                np.concatenate([r[0] for r in runs.values()]),
                None,
                np.concatenate([r[2] + 100_000 * i for i, r in enumerate(runs.values())]),
            )
            al = [a for v in alphas.values() for a in v]
            row = _prow(exp, target, feat, "all", p, lo, hi, allrun, al)
            row["n_seeds"] = len(runs)
            rows.append(row)
            cis[feat] = (lo, hi)
        claim = (
            sysid_claim(cis["penult"], cis["obs_single"], cis["penult_random"])
            if all(f in cis for f in ("penult", "obs_single", "penult_random"))
            else False
        )
        a_gt_c = (
            bool(cis["penult"][0] > cis["obs_stacked"][1])
            if "penult" in cis and "obs_stacked" in cis
            else None
        )
        for r in rows:
            if r["exp"] == exp and r["target"] == target:
                r["claim_sysid"] = claim
                r["penult_beats_stacked"] = a_gt_c
    return pd.DataFrame(rows)


def _prow(exp, target, feat, scope, p, lo, hi, run, alphas) -> dict:
    return {
        "exp": exp,
        "target": target,
        "features": feat,
        "scope": scope,
        "r2": p,
        "r2_lo": lo,
        "r2_hi": hi,
        "n_episodes": len(np.unique(run[2])),
        "n_states": len(run[0]),
        "alpha_median": float(np.median(alphas)),
    }


def run_probes(
    root: Path,
    out: Path,
    exps: list[str],
    seeds: list[int],
    source: str = "train",
    n_eps: int = 200,
    max_states: int | None = 10_000,
    load_model_fn: Callable[[str, int], object] | None = None,
    exp_cfgs: dict[str, dict] | None = None,
    n_splits: int = 5,
    n_boot: int = 2000,
) -> pd.DataFrame:
    if load_model_fn is None:
        from projectv.eval.rollout import load_model, resolve_checkpoint

        def load_model_fn(e, s):
            return load_model(resolve_checkpoint(e, s, root))

    tables = []
    for exp in exps:
        cfg = (exp_cfgs or {}).get(exp) or load_exp(f"configs/exp/{exp}.yaml")
        hist = bool(cfg.get("history", False))
        groups = make_groups(cfg["task"], history=hist, shortcut=bool(cfg.get("shortcut", False)))
        per_seed = {}
        for seed in seeds:
            try:
                model = load_model_fn(exp, seed)
            except FileNotFoundError as e:
                print(f"[probes] skip {exp} s{seed}: {e}")
                continue
            if source == "train":
                cache = out / "probe_rollouts" / f"{exp}_s{seed}_n{n_eps}.npz"
                rec = rollout_record(model, cfg, seed, n_eps, cache)
            else:
                rec = steplog_record(root, exp, seed)
                if rec is None:
                    print(f"[probes] skip {exp} s{seed}: no step logs")
                    continue
            per_seed[seed] = probe_data(
                rec, groups, hist, random_actor_penultimate(model), max_states, seed
            )
            print(f"[probes] {exp} s{seed}: {len(per_seed[seed]['episode'])} states")
        if per_seed:
            tables.append(probe_table(exp, per_seed, n_splits, n_boot))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(EXPS))
    p.add_argument("--exps", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--root", default="results")
    p.add_argument("--out", default=None, help="default {root}/c4")
    p.add_argument("--source", choices=["train", "steplogs"], default="train")
    p.add_argument("--n-eps", type=int, default=200)
    p.add_argument("--max-states", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    set_global_seed(args.seed)
    root = Path(args.root)
    out = Path(args.out or root / "c4")
    out.mkdir(parents=True, exist_ok=True)
    if args.source == "steplogs":
        from projectv.eval.cells import assert_frozen

        assert_frozen()
    df = run_probes(
        root,
        out,
        args.exps or list(EXPS[args.task]),
        args.seeds,
        args.source,
        args.n_eps,
        args.max_states,
    )
    df.to_csv(out / "probes.csv", index=False)
    meta = write_run_meta(out, vars(args), sys.argv, t0, task=args.task, analysis="probes")
    write_json(out / "run_meta_probes.json", meta)
    if len(df):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df[df.scope == "all"].drop(columns=["scope"]))


if __name__ == "__main__":
    main()
