"""C3 pipeline (mvp.md §7): ASM vs 5 baselines for early failure prediction on held-out dynamics.

    python -m projectv.monitor.asm --exp push_dr [--seeds 0 1 2] [--root results]

Protocol (per exp, pooled over training seeds = clusters):
1. Load the per-step logs `{root}/steplogs/{exp}_s{seed}/cell{id:02d}.npz` of every cell in the
   frozen held-out grid (the hash is asserted first). Episodes are split calib/test by
   `splits.calib_mask` within each (seed, cell).
2. Per policy, nominal statistics come from calib-split nominal (cell 0) episodes only: the IG
   baseline (mean state), the ASM share model, and the obs/penultimate Mahalanobis models.
3. Raw per-step scores m (E, 50) for every signal; ASM shares and ensemble scores are cached per
   (policy, cell) in `{out}/cache/` and reused on rerun when their fingerprint matches.
4. Per signal, EWMA alpha is chosen on the calib split (all cells) to maximise AUROC of S(25).
   Thresholds for TPR@5%FPR = 95th percentile of S(T) of calib successful episodes.
5. Metrics on the test split only (nominal + held-out cells): AUROC, AUPRC, TPR@5%FPR, median
   detection step of failed episodes (first t < T with e_t > threshold, among detected ones),
   each with a 95% cluster-bootstrap CI (seeds, then episodes). Per-seed AUROC columns. Per-factor
   AUROC uses the test episodes of that factor's cells plus the nominal test episodes.

Outputs in {out} (default {root}/c3): c3_{exp}_metrics.csv, c3_{exp}_per_factor.csv,
c3_{exp}_alpha.json, c3_{exp}_episode_scores.csv (per-step e_t for the GIF alarm overlay),
run_meta.json (+ a copy c3_{exp}_run_meta.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from projectv.eval import rollout
from projectv.eval.cells import HELDOUT_PATH, Cell, assert_frozen, load_cells
from projectv.eval.steplog import load_steplog, steplog_path
from projectv.monitor import asm, baselines
from projectv.monitor.metrics import (
    auprc,
    auroc,
    cluster_bootstrap_ci,
    detection_steps,
    ewma,
    prefix_score,
    select_alpha,
)
from projectv.monitor.splits import CALIB, TEST, calib_mask
from projectv.utils.config import file_hash, load_exp
from projectv.utils.io import write_json, write_run_meta
from projectv.xai.policy_fn import SB3Policy

PREFIXES = (10, 25)
ALPHA_T = 25
FPR = 0.05


# --- loading ------------------------------------------------------------------------------------
def load_policy_logs(
    root: Path, exp: str, seed: int, cells: list[Cell]
) -> tuple[list[tuple[Cell, dict]], list[int]]:
    """Step logs of every cell present; cell 0 is required. Returns (logs, missing cell ids)."""
    logs, missing = [], []
    for c in cells:
        p = steplog_path(root, exp, seed, c.id)
        if not p.exists():
            assert c.id != 0, f"nominal step log missing: {p}"
            missing.append(c.id)
            continue
        log = load_steplog(p)
        assert int(log["cell_id"]) == c.id, (p, int(log["cell_id"]))
        logs.append((c, log))
    return logs, missing


def check_steplog_meta(root: Path, exp: str, seed: int, heldout_hash: str) -> None:
    meta = steplog_path(root, exp, seed, 0).parent / "meta.json"
    if meta.exists():
        rec = json.loads(meta.read_text(encoding="utf-8")).get("heldout_hash")
        assert rec in (None, heldout_hash), f"{meta}: step logs were made with another grid"


def load_actor_critic(path: str | Path):
    """Load a SAC/TQC(+HER) checkpoint for inference only (shared loader in eval.rollout)."""
    from projectv.eval.rollout import load_model

    return load_model(path)


# --- cache --------------------------------------------------------------------------------------
def _fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(
            np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else str(p).encode()
        )
        h.update(b"|")
    return h.hexdigest()


def cached(path: Path, fingerprint: str, ep_seed: np.ndarray, compute: Callable[[], np.ndarray]):
    """Load `value` from path if fingerprint and episode seeds match, else compute and save."""
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) == fingerprint and np.array_equal(z["ep_seed"], ep_seed):
                return z["value"]
    value = np.asarray(compute())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, value=value, fingerprint=np.array(fingerprint), ep_seed=ep_seed)
    tmp.replace(path)
    return value


# --- per-policy raw scores ----------------------------------------------------------------------
def policy_scores(
    exp_cfg: dict,
    exp: str,
    seed: int,
    logs: list[tuple[Cell, dict]],
    pol,
    ckpt_hash: str,
    ens_pols: list | None,
    ens_hash: str,
    cache_dir: Path,
    ig_steps: int = 16,
    batch: int = 512,
) -> list[dict[str, np.ndarray]]:
    """Raw per-step scores {signal: (E, T)} for every cell of one policy."""
    assert logs[0][0].id == 0
    nom = logs[0][1]
    cal0 = calib_mask(nom["ep_seed"])
    d = nom["obs_flat"].shape[-1]
    base = nom["obs_flat"][cal0].reshape(-1, d).astype(np.float64).mean(axis=0)
    groups = asm.monitor_groups(
        exp_cfg["task"], bool(exp_cfg.get("history")), bool(exp_cfg.get("shortcut"))
    )
    fp_asm = _fingerprint("asm-ig", ckpt_hash, base, ig_steps, list(groups))
    fp_ens = _fingerprint("ensemble", ens_hash)

    def shares(log: dict) -> np.ndarray:
        X = log["obs_flat"]
        z = asm.attribution_shares(pol, X.reshape(-1, d), base, groups, ig_steps, batch)
        return z.reshape(*X.shape[:2], len(groups))

    zs = [
        cached(
            cache_dir / f"{exp}_s{seed}" / f"cell{c.id:02d}_asm.npz",
            fp_asm,
            log["ep_seed"],
            lambda log=log: shares(log),
        )
        for c, log in logs
    ]
    m_asm = asm.fit_nominal(zs[0][cal0].reshape(-1, zs[0].shape[-1]))
    m_obs = baselines.fit_feature_model(nom["obs_flat"], cal0)
    m_pen = baselines.fit_feature_model(nom["penult"], cal0)
    out = []
    for (c, log), z in zip(logs, zs, strict=True):
        s = {
            "asm": asm.asm_scores(z, m_asm),
            "obs_ood": baselines.obs_ood(log, m_obs),
            "act_std": baselines.act_std(log),
            "q_drop": baselines.q_drop(log),
            "act_ood": baselines.act_ood(log, m_pen),
        }
        if ens_pols is not None:
            s["ensemble"] = cached(
                cache_dir / f"{exp}_s{seed}" / f"cell{c.id:02d}_ensemble.npz",
                fp_ens,
                log["ep_seed"],
                lambda log=log: baselines.ensemble_std(ens_pols, log["obs_flat"]),
            )
        out.append(s)
    return out


# --- evaluation ---------------------------------------------------------------------------------
def _ci(metric, scores, labels, clusters, n_boot, seed, ignore_nan=False):
    if len(labels) == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")
    return cluster_bootstrap_ci(
        metric, scores, labels, clusters, n_boot=n_boot, seed=seed, ignore_nan=ignore_nan
    )


def _median_detection(det: np.ndarray, labels: np.ndarray) -> float:
    d = det[np.asarray(labels, bool)]
    d = d[~np.isnan(d)]
    return float(np.median(d)) if d.size else float("nan")


def evaluate_signal(
    m: np.ndarray, tab: pd.DataFrame, n_boot: int, boot_seed: int
) -> tuple[list[dict], list[dict], dict, dict]:
    """m: (N, steps) raw scores aligned with tab (columns seed, cell_id, factor, calib, failed).
    Returns (metric rows, per-factor rows, alpha info, per-episode arrays)."""
    calib = tab["calib"].to_numpy(bool)
    failed = tab["failed"].to_numpy(bool)
    test = ~calib
    seeds = tab["seed"].to_numpy()
    factor = tab["factor"].to_numpy()
    alpha, cal_auc = select_alpha(m, failed, calib, T=ALPHA_T)
    e = ewma(m, alpha)
    rows, pf_rows, ep = [], [], {"alpha": alpha, "e": e}
    y_t, cl_t = failed[test], seeds[test]
    for T in PREFIXES:
        S = prefix_score(e, T)
        neg = S[calib & ~failed]
        thr = float(np.quantile(neg, 1.0 - FPR)) if neg.size else float("nan")
        det = detection_steps(e[:, :T], thr)
        ep[f"S{T}"], ep[f"thr_T{T}"] = S, thr
        s_t = S[test]
        alarm = s_t > thr

        def tpr(s, y, thr=thr):
            return float(np.mean(s[y.astype(bool)] > thr))

        row = {
            "signal": None,
            "T": T,
            "alpha": alpha,
            "n_test": int(test.sum()),
            "n_fail_test": int(y_t.sum()),
            "threshold": thr,
            "fpr_test": float(alarm[~y_t].mean()) if (~y_t).any() else float("nan"),
        }
        for name, fn, sc, kw in (
            ("auroc", auroc, s_t, {}),
            ("auprc", auprc, s_t, {}),
            ("tpr_at_5fpr", tpr, s_t, {}),
            ("det_step_median", _median_detection, det[test], {"ignore_nan": True}),
        ):
            pt, lo, hi = _ci(fn, sc, y_t, cl_t, n_boot, boot_seed, **kw)
            row.update({name: pt, f"{name}_lo": lo, f"{name}_hi": hi})
        for sd in np.unique(seeds):
            k = test & (seeds == sd)
            yk = failed[k]
            row[f"auroc_s{sd}"] = auroc(S[k], yk) if np.unique(yk).size == 2 else float("nan")
        rows.append(row)
        for f in sorted(set(factor[test]) - {"nominal"}):
            k = test & ((factor == f) | (tab["cell_id"].to_numpy() == 0))
            pt, lo, hi = _ci(auroc, S[k], failed[k], seeds[k], n_boot, boot_seed)
            pf_rows.append(
                {
                    "signal": None,
                    "T": T,
                    "factor": f,
                    "n_test": int(k.sum()),
                    "n_fail_test": int(failed[k].sum()),
                    "auroc": pt,
                    "auroc_lo": lo,
                    "auroc_hi": hi,
                }
            )
    info = {"alpha": alpha, "calib_auroc_T25": {str(a): v for a, v in cal_auc.items()}}
    return rows, pf_rows, info, ep


# --- CLI ----------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default=None, help="default {root}/c3")
    ap.add_argument("--cells", default=HELDOUT_PATH)
    ap.add_argument("--hash-path", default=None, help="default {root}/heldout_hash.txt")
    ap.add_argument("--no-hash-check", action="store_true", help="tests only")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-wandb", action="store_true", help="accepted; C3 never logs to W&B")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    t0 = time.time()
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "c3"
    hash_path = Path(a.hash_path) if a.hash_path else root / "heldout_hash.txt"
    if not a.no_hash_check:
        assert_frozen(a.cells, hash_path)
    cells = load_cells(a.cells, check_hash=False)
    heldout_hash = file_hash(a.cells)
    exp_cfg = load_exp(f"configs/exp/{a.exp}.yaml")

    ckpts = {s: rollout.resolve_checkpoint(a.exp, s, root) for s in a.seeds}
    ckpt_hash = {s: file_hash(p) for s, p in ckpts.items()}
    pols = {s: SB3Policy(load_actor_critic(p)) for s, p in ckpts.items()}
    ens_pols = [pols[s] for s in a.seeds] if len(a.seeds) >= 2 else None
    if ens_pols is None:
        warnings.warn("ensemble baseline needs >= 2 seeds; skipped", stacklevel=1)
    ens_hash = _fingerprint(*[ckpt_hash[s] for s in a.seeds])

    tabs, scores, missing = [], {}, {}
    for s in a.seeds:
        if not a.no_hash_check:
            check_steplog_meta(root, a.exp, s, heldout_hash)
        logs, missing[s] = load_policy_logs(root, a.exp, s, cells)
        per_cell = policy_scores(
            exp_cfg,
            a.exp,
            s,
            logs,
            pols[s],
            ckpt_hash[s],
            ens_pols,
            ens_hash,
            out / "cache",
            a.ig_steps,
            a.batch,
        )
        for (c, log), sc in zip(logs, per_cell, strict=True):
            E = len(log["ep_seed"])
            tabs.append(
                pd.DataFrame(
                    {
                        "seed": s,
                        "cell_id": c.id,
                        "cell_name": c.name,
                        "factor": c.factor,
                        "episode": np.arange(E),
                        "ep_seed": log["ep_seed"],
                        "calib": calib_mask(log["ep_seed"]),
                        "failed": ~log["success"].astype(bool),
                    }
                )
            )
            for k, v in sc.items():
                scores.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    tab = pd.concat(tabs, ignore_index=True)
    signals = [k for k in baselines.SIGNALS if k in scores]

    rows, pf_rows, alpha_info, ep_frames = [], [], {}, []
    for sig in signals:
        m = np.concatenate(scores[sig])
        r, pf, info, ep = evaluate_signal(m, tab, a.n_boot, a.seed)
        for x in (*r, *pf):
            x["signal"] = sig
        rows += r
        pf_rows += pf
        alpha_info[sig] = info
        epf = tab[["seed", "cell_id", "factor", "ep_seed", "failed"]].copy()
        epf.insert(0, "signal", sig)
        epf.insert(4, "split", np.where(tab["calib"], CALIB, TEST))
        epf["alpha"] = ep["alpha"]
        for T in PREFIXES:
            epf[f"thr_T{T}"] = ep[f"thr_T{T}"]
            epf[f"S{T}"] = ep[f"S{T}"]
            epf[f"alarm_T{T}"] = ep[f"S{T}"] > ep[f"thr_T{T}"]
        epf["threshold"] = ep["thr_T25"]  # alarm level for the GIF overlay (docs/interfaces.md)
        e = pd.DataFrame(ep["e"], columns=[f"e_{t}" for t in range(ep["e"].shape[1])])
        ep_frames.append(pd.concat([epf, e], axis=1))

    out.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame(rows)
    met.insert(0, "exp", a.exp)
    met.to_csv(out / f"c3_{a.exp}_metrics.csv", index=False)
    pf = pd.DataFrame(pf_rows)
    pf.insert(0, "exp", a.exp)
    pf.to_csv(out / f"c3_{a.exp}_per_factor.csv", index=False)
    write_json(
        out / f"c3_{a.exp}_alpha.json",
        {"exp": a.exp, "selected_on": "calib", "T": ALPHA_T, "signals": alpha_info},
    )
    epdf = pd.concat(ep_frames, ignore_index=True)
    epdf.insert(0, "exp", a.exp)
    epdf.to_csv(out / f"c3_{a.exp}_episode_scores.csv", index=False, float_format="%.6g")
    meta = write_run_meta(
        out,
        {"exp": exp_cfg, "seeds": a.seeds, "n_boot": a.n_boot, "ig_steps": a.ig_steps},
        argv,
        t0,
        exp=a.exp,
        seeds=a.seeds,
        heldout_hash=heldout_hash,
        hash_checked=not a.no_hash_check,
        checkpoints={s: str(p) for s, p in ckpts.items()},
        missing_cells={s: v for s, v in missing.items() if v},
        signals=signals,
        groups=list(
            asm.monitor_groups(
                exp_cfg["task"], bool(exp_cfg.get("history")), bool(exp_cfg.get("shortcut"))
            )
        ),
    )
    write_json(out / f"c3_{a.exp}_run_meta.json", meta)
    show = met[["signal", "T", "alpha", "auroc", "auroc_lo", "auroc_hi", "tpr_at_5fpr"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
