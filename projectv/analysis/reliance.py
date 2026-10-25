"""C4 reliance profile and temporal reliance (mvp.md §8).

For every Push policy (push_nodr / push_dr / push_drhist x seeds):
- group shares (IG and GWIE-C) on `n_states` nominal states from step log cell 0;
- attribution entropy of the share vector (per state, averaged over states);
- for DR+Hist both per-slot groups (`level=slot`) and base groups summed over slots
  (`level=base`; past actions act@t-1..3 sum into `act`);
- temporal reliance for DR+Hist: share on slots t, t-1, t-2, t-3 and on past actions, nominal vs
  held-out cells (pooled, and per held-out factor).

Erasure / baseline pool: cell 0 episodes are split in half by a seeded permutation; one half is
the marginal pool (CLAUDE.md §8), the other half supplies the explained nominal states. Held-out
states are explained against the same nominal pool. Across seeds: mean, percentile bootstrap CI
over seeds and every per-seed value (n=3: the CI is coarse, read the per-seed values).

Per-policy, per-(method, scope) raw group scores are cached under `{out}/cache/` so reruns skip
completed units.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from projectv.analysis.stats import seed_summary
from projectv.eval.cells import HELDOUT_PATH, load_cells
from projectv.eval.steplog import load_steplog, steplog_path
from projectv.utils.config import load_exp
from projectv.utils.io import write_json, write_run_meta
from projectv.utils.seeding import set_global_seed
from projectv.xai.attributions import integrated_gradients_batch, to_groups
from projectv.xai.groups import base_name, make_groups, slot_of

METHODS = ("ig", "gwie_c")
TEMPORAL_BUCKETS = ("t", "t-1", "t-2", "t-3", "act")
EXPS = {
    "push": ("push_nodr", "push_dr", "push_drhist"),
    "reach": ("reach_nodr", "reach_dr", "reach_drhist"),  # Gate G2 track
}


# --- pure helpers ---------------------------------------------------------------------------
def shares_from_scores(scores: np.ndarray) -> np.ndarray:
    """(n, G) unsigned group scores -> (n, G) shares summing to 1 (rows of zeros stay zero)."""
    s = np.abs(np.asarray(scores, dtype=np.float64))
    tot = s.sum(axis=-1, keepdims=True)
    return np.divide(s, tot, out=np.zeros_like(s), where=tot > 0)


def entropy(shares: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each share row, 0 log 0 = 0."""
    p = np.asarray(shares, dtype=np.float64)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


def normalized_entropy(shares: np.ndarray) -> np.ndarray:
    """Entropy / log(G), in [0, 1]; comparable across different group counts."""
    G = np.asarray(shares).shape[-1]
    return entropy(shares) / np.log(G) if G > 1 else np.zeros(np.asarray(shares).shape[:-1])


def aggregate_base(shares: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum per-slot group shares over slots: 'obj_pos@t-2' -> 'obj_pos', 'act@t-1' -> 'act'."""
    base = list(dict.fromkeys(base_name(g) for g in names))
    M = np.zeros((len(names), len(base)))
    for i, g in enumerate(names):
        M[i, base.index(base_name(g))] = 1.0
    return np.asarray(shares) @ M, base


def temporal_buckets(shares: np.ndarray, names: list[str]) -> np.ndarray:
    """(n, G) shares -> (n, 5) shares on [t, t-1, t-2, t-3, act]. Current-only groups
    (`goal`, and `achieved_goal` merged into `obj_pos`) count as slot t."""
    M = np.zeros((len(names), len(TEMPORAL_BUCKETS)))
    for i, g in enumerate(names):
        M[i, 4 if base_name(g) == "act" else slot_of(g)] = 1.0
    return np.asarray(shares) @ M


def split_episodes(n_eps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic half/half split of episode indices -> (pool_eps, explain_eps)."""
    perm = np.random.default_rng(seed).permutation(n_eps)
    h = n_eps // 2
    return np.sort(perm[:h]), np.sort(perm[h:])


def sample_states(obs_flat: np.ndarray, eps: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n states (without replacement when possible) from episodes `eps` of an (E, T, d) log."""
    X = np.asarray(obs_flat, dtype=np.float64)[eps].reshape(-1, obs_flat.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=n > len(X))
    return X[idx]


def attribute_states(
    pol,
    S: np.ndarray,
    groups: dict[str, np.ndarray],
    pool: np.ndarray,
    method: str,
    seed: int,
    M: int = 8,
) -> np.ndarray:
    """(n, G) unsigned group scores, columns in `groups` order."""
    names = list(groups)
    if method == "ig":
        A = integrated_gradients_batch(pol, S, pool.mean(axis=0), steps=16)
        return np.array([[to_groups(a, groups)[g] for g in names] for a in A])
    if method == "gwie_c":
        from projectv.xai.gwie_c import gwie_c

        rng = np.random.default_rng(seed)
        out = []
        for s in S:
            gains = gwie_c(pol, s, groups, pool, rng, M).gains
            out.append([abs(gains[g]) for g in names])
        return np.array(out)
    raise ValueError(f"unknown method {method}")


def cached_scores(path: Path | None, fn: Callable[[], np.ndarray], names: list[str]) -> np.ndarray:
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert list(z["names"]) == names, f"stale cache {path}"
            return z["scores"]
    scores = fn()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, scores=scores, names=np.array(names))
        tmp.replace(path)
    return scores


# --- per policy -----------------------------------------------------------------------------
def policy_reliance(
    pol_loader: Callable[[], object],
    logs: dict[int, dict[str, np.ndarray]],
    groups: dict[str, np.ndarray],
    methods: tuple[str, ...],
    n_states: int,
    n_states_cell: int,
    seed: int,
    cache_dir: Path | None,
    temporal: bool,
) -> dict[tuple[str, int], np.ndarray]:
    """Raw group scores per (method, cell_id). Cell 0 always; held-out cells only if
    `temporal`. `pol_loader` is only called if some unit is not cached."""
    names = list(groups)
    nominal = logs[0]
    pool_eps, expl_eps = split_episodes(len(nominal["obs_flat"]), seed)
    obs = np.asarray(nominal["obs_flat"], dtype=np.float64)
    pool = obs[pool_eps].reshape(-1, obs.shape[-1])
    pol_box: list[object] = []

    def pol():
        if not pol_box:
            pol_box.append(pol_loader())
        return pol_box[0]

    out = {}
    cells = sorted(logs) if temporal else [0]
    for method in methods:
        for cid in cells:
            if cid == 0:
                S = sample_states(nominal["obs_flat"], expl_eps, n_states, seed)
            else:
                log = logs[cid]
                S = sample_states(
                    log["obs_flat"], np.arange(len(log["obs_flat"])), n_states_cell, seed + cid
                )
            path = (
                None if cache_dir is None else cache_dir / f"{method}_cell{cid:02d}_n{len(S)}.npz"
            )
            out[(method, cid)] = cached_scores(
                path,
                lambda S=S, m=method, c=cid: attribute_states(pol(), S, groups, pool, m, seed + c),
                names,
            )
    return out


# --- tables ---------------------------------------------------------------------------------
def reliance_rows(
    exp: str, per_seed: dict[int, dict[tuple[str, int], np.ndarray]], names: list[str]
) -> list[dict]:
    """Shares and entropies on nominal states (cell 0), slot level and base level."""
    rows = []
    has_slots = any("@t-" in g for g in names)
    methods = sorted({m for d in per_seed.values() for (m, c) in d if c == 0})
    for method in methods:
        levels = {}
        for s, d in per_seed.items():
            sh = shares_from_scores(d[(method, 0)])
            base, bnames = aggregate_base(sh, names)
            levels.setdefault("base", {})[s] = (base, bnames)
            if has_slots:
                levels.setdefault("slot", {})[s] = (sh, names)
        for level, by_seed in levels.items():
            gnames = next(iter(by_seed.values()))[1]
            for j, g in enumerate(gnames):
                ps = {s: float(v[0][:, j].mean()) for s, v in by_seed.items()}
                rows.append(_row(exp, method, level, "share", g, ps))
            for q, fn in (("entropy", entropy), ("entropy_norm", normalized_entropy)):
                ps = {s: float(fn(v[0]).mean()) for s, v in by_seed.items()}
                rows.append(_row(exp, method, level, q, "", ps, n_groups=len(gnames)))
    return rows


def temporal_rows(
    exp: str,
    per_seed: dict[int, dict[tuple[str, int], np.ndarray]],
    names: list[str],
    factor_of: dict[int, str],
) -> list[dict]:
    rows = []
    methods = sorted({m for d in per_seed.values() for (m, _) in d})
    for method in methods:
        scopes: dict[str, dict[int, np.ndarray]] = {}
        for s, d in per_seed.items():
            by_cell = {
                c: temporal_buckets(shares_from_scores(v), names)
                for (m, c), v in d.items()
                if m == method
            }
            scopes.setdefault("nominal", {})[s] = by_cell[0]
            held = [c for c in by_cell if c != 0]
            if held:
                scopes.setdefault("heldout", {})[s] = np.concatenate([by_cell[c] for c in held])
                for f in sorted({factor_of.get(c, "unknown") for c in held}):
                    cs = [c for c in held if factor_of.get(c, "unknown") == f]
                    stack = np.concatenate([by_cell[c] for c in cs])
                    scopes.setdefault(f"factor:{f}", {})[s] = stack
        for scope, by_seed in scopes.items():
            for j, b in enumerate(TEMPORAL_BUCKETS):
                ps = {s: float(v[:, j].mean()) for s, v in by_seed.items()}
                r = _row(exp, method, "temporal", "share", b, ps)
                r["scope"] = scope
                rows.append(r)
    return rows


def _row(exp, method, level, quantity, group, per_seed, **extra) -> dict:
    return {
        "exp": exp,
        "method": method,
        "level": level,
        "quantity": quantity,
        "group": group,
        **extra,
        **seed_summary(per_seed),
    }


# --- driver ---------------------------------------------------------------------------------
def load_logs(root: Path, exp: str, seed: int, with_heldout: bool) -> dict[int, dict] | None:
    d = steplog_path(root, exp, seed, 0).parent
    if not (d / "cell00.npz").exists():
        return None
    files = sorted(d.glob("cell*.npz")) if with_heldout else [d / "cell00.npz"]
    return {int(f.stem[4:]): load_steplog(f) for f in files if not f.name.endswith(".tmp.npz")}


def run_reliance(
    root: Path,
    out: Path,
    exps: list[str],
    seeds: list[int],
    load_policy: Callable[[str, int], object],
    factor_of: dict[int, str],
    methods: tuple[str, ...] = METHODS,
    n_states: int = 300,
    n_states_cell: int = 100,
    base_seed: int = 0,
    exp_cfgs: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel, tmp = [], []
    for exp in exps:
        cfg = (exp_cfgs or {}).get(exp) or load_exp(f"configs/exp/{exp}.yaml")
        hist = bool(cfg.get("history", False))
        groups = make_groups(cfg["task"], history=hist, shortcut=bool(cfg.get("shortcut", False)))
        names = list(groups)
        per_seed = {}
        for seed in seeds:
            logs = load_logs(root, exp, seed, with_heldout=hist)
            if logs is None:
                print(f"[reliance] skip {exp} s{seed}: no step log cell00")
                continue
            per_seed[seed] = policy_reliance(
                lambda e=exp, s=seed: load_policy(e, s),
                logs,
                groups,
                methods,
                n_states,
                n_states_cell,
                10_000 * seed + base_seed,
                out / "cache" / f"{exp}_s{seed}",
                temporal=hist,
            )
            print(f"[reliance] {exp} s{seed} done ({len(logs)} cells)")
        if not per_seed:
            continue
        rel += reliance_rows(exp, per_seed, names)
        if hist:
            tmp += temporal_rows(exp, per_seed, names, factor_of)
    return pd.DataFrame(rel), pd.DataFrame(tmp)


def sb3_loader(root: Path) -> Callable[[str, int], object]:
    def load(exp: str, seed: int):
        from projectv.eval.rollout import load_model, resolve_checkpoint
        from projectv.xai.policy_fn import SB3Policy

        return SB3Policy(load_model(resolve_checkpoint(exp, seed, root)))

    return load


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(EXPS))
    p.add_argument("--exps", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--root", default="results")
    p.add_argument("--out", default=None, help="default {root}/c4")
    p.add_argument("--cells", default=HELDOUT_PATH)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--n-states", type=int, default=300)
    p.add_argument("--n-states-cell", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    set_global_seed(args.seed)
    root = Path(args.root)
    out = Path(args.out or root / "c4")
    out.mkdir(parents=True, exist_ok=True)
    exps = args.exps or list(EXPS[args.task])
    factor_of = {}
    if Path(args.cells).exists():
        factor_of = {c.id: c.factor for c in load_cells(args.cells, check_hash=True)}
    else:
        print(f"[reliance] {args.cells} missing: held-out factors reported as 'unknown'")
    rel, tmp = run_reliance(
        root,
        out,
        exps,
        args.seeds,
        sb3_loader(root),
        factor_of,
        tuple(args.methods.split(",")),
        args.n_states,
        args.n_states_cell,
        args.seed,
    )
    rel.to_csv(out / "reliance.csv", index=False)
    tmp.to_csv(out / "temporal.csv", index=False)
    meta = write_run_meta(out, vars(args), sys.argv, t0, task=args.task, analysis="reliance")
    write_json(out / "run_meta_reliance.json", meta)
    print("note: n=3 seeds -> bootstrap CIs over seeds are coarse; per-seed values in seed{n}.")
    if len(rel):
        ent = rel[rel.quantity == "entropy_norm"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(ent.drop(columns=["quantity", "group"]))
    if len(tmp):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(tmp[tmp.scope.isin(["nominal", "heldout"])])


if __name__ == "__main__":
    main()
