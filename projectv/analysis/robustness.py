"""Robustness table (mvp.md §8 "robustness link", A7).

From `results/eval/{exp}_s{seed}.csv` (columns cell_id, cell_name, factor, episode, ep_seed,
success): per condition, a score matrix (seeds x cells) of per-cell success rates for each
split (`nominal` = cell 0, `heldout` = every other cell, `factor:<f>` = the held-out cells of
one factor). Aggregate = mean over seeds and cells, with a 95% stratified bootstrap CI from
`rliable` (2,000 resamples; runs = seeds resampled within each cell, strata = cells), the
per-seed values (mean over that seed's cells), and the probability of improvement
(`rliable.metrics.probability_of_improvement`) DR vs NoDR and DR+Hist vs DR with its stratified
independent-bootstrap CI.

rliable 1.2.0's `library` module imports `arch`; arch 7.x fails to import under pandas 3 and
arch 8 renamed `random_state`, which rliable passes. If `rliable.library` cannot be used, the
identical resampling scheme is run by hand (`ci_backend` column says which was used).
`probability_of_improvement` (rliable.metrics) does not need arch and is always rliable's.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rliable import metrics as rly_metrics

from projectv.utils.io import write_run_meta

CONDITIONS = ("push_nodr", "push_dr", "push_drhist")
POI_PAIRS = (("push_dr", "push_nodr"), ("push_drhist", "push_dr"))
N_BOOT = 2000
IQM_NOTE = (
    "note: n=3 seeds per condition -> IQM over runs is degenerate (it trims to ~1 run), so the "
    "aggregate is the mean; per-seed values are reported alongside the bootstrap CI."
)


def load_eval(root: Path, exps: tuple[str, ...] = CONDITIONS) -> pd.DataFrame:
    frames = []
    for exp in exps:
        for f in sorted((root / "eval").glob(f"{exp}_s*.csv")):
            seed = f.stem[len(exp) + 2 :]
            if not seed.isdigit():
                continue
            d = pd.read_csv(f)
            d["exp"], d["seed"] = exp, int(seed)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_cells(df: pd.DataFrame) -> dict[str, list[int]]:
    cells = df.drop_duplicates("cell_id").set_index("cell_id")["factor"].to_dict()
    held = sorted(c for c in cells if c != 0)
    out = {"nominal": [0] if 0 in cells else [], "heldout": held}
    for f in sorted({cells[c] for c in held}):
        out[f"factor:{f}"] = [c for c in held if cells[c] == f]
    return {k: v for k, v in out.items() if v}


def score_matrix(df: pd.DataFrame, exp: str, cells: list[int]) -> tuple[np.ndarray, list[int]]:
    """(seeds x cells) per-cell success rate. Only seeds that have every cell are kept."""
    d = df[(df.exp == exp) & df.cell_id.isin(cells)]
    tab = d.groupby(["seed", "cell_id"])["success"].mean().unstack("cell_id")
    tab = tab.reindex(columns=cells)
    incomplete = [int(s) for s in tab.index[tab.isna().any(axis=1)]]
    if incomplete:
        warnings.warn(
            f"{exp}: seeds {incomplete} lack some cells of this split and are excluded from it "
            "(stratified bootstrap needs complete rows); n_seeds in the output reflects this",
            stacklevel=2,
        )
    tab = tab.dropna(axis=0, how="any")
    return tab.to_numpy(dtype=np.float64), [int(s) for s in tab.index]


def _mean(x: np.ndarray) -> np.ndarray:
    return np.array([np.mean(x)])


@functools.cache
def _rliable_library():
    try:
        from rliable import library

        # arch may import but reject rliable's arguments, so try one tiny call
        library.get_interval_estimates({"_": np.ones((2, 2))}, _mean, reps=2)
        return library
    except Exception:  # noqa: BLE001 - any arch/pandas incompatibility -> fallback
        return None


def stratified_ci(
    scores: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, use_rliable: bool = True
) -> tuple[float, float, float, str]:
    """Mean over (runs x tasks) with a stratified (runs within task) percentile bootstrap CI."""
    point = float(np.mean(scores))
    lib = _rliable_library() if use_rliable else None
    if lib is not None:
        np.random.seed(seed)  # rliable's StratifiedBootstrap draws from the global RNG
        _, ci = lib.get_interval_estimates({"x": scores}, _mean, reps=n_boot)
        return point, float(ci["x"][0, 0]), float(ci["x"][1, 0]), "rliable"
    rng = np.random.default_rng(seed)
    n, m = scores.shape
    idx = rng.integers(n, size=(n_boot, n, m))
    boots = scores[idx, np.arange(m)].mean(axis=(1, 2))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi), "manual"


def poi_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, use_rliable: bool = True
) -> tuple[float, float, float, str]:
    """P(X > Y) (rliable.metrics.probability_of_improvement) with a stratified independent
    bootstrap CI (runs of X and of Y resampled independently within each task)."""
    poi = rly_metrics.probability_of_improvement
    point = float(poi(x, y))
    lib = _rliable_library() if use_rliable else None
    if lib is not None:
        np.random.seed(seed)

        def fn(a, b):
            return np.array([poi(a, b)])

        _, ci = lib.get_interval_estimates({"p": [x, y]}, fn, reps=n_boot)
        return point, float(ci["p"][0, 0]), float(ci["p"][1, 0]), "rliable"
    rng = np.random.default_rng(seed)
    m = x.shape[1]
    boots = []
    for _ in range(n_boot):
        xb = x[rng.integers(len(x), size=x.shape), np.arange(m)]
        yb = y[rng.integers(len(y), size=y.shape), np.arange(m)]
        boots.append(poi(xb, yb))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi), "manual"


def robustness_tables(
    df: pd.DataFrame,
    conditions: tuple[str, ...] = CONDITIONS,
    pairs: tuple[tuple[str, str], ...] = POI_PAIRS,
    n_boot: int = N_BOOT,
    seed: int = 0,
    use_rliable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prow = [], []
    present = [c for c in conditions if c in set(df.exp)]
    for split, cells in split_cells(df).items():
        mats = {}
        for exp in present:
            S, seeds = score_matrix(df, exp, cells)
            if not len(S):
                continue
            mats[exp] = (S, seeds)
            p, lo, hi, backend = stratified_ci(S, n_boot, seed, use_rliable)
            row = {
                "exp": exp,
                "split": split,
                "n_cells": len(cells),
                "n_seeds": len(seeds),
                "episodes_per_cell_min": int(
                    df[(df.exp == exp) & df.cell_id.isin(cells)]
                    .groupby(["seed", "cell_id"])
                    .size()
                    .min()
                ),
                "success": p,
                "success_lo": lo,
                "success_hi": hi,
                "ci_backend": backend,
            }
            row.update({f"seed{s}": float(S[i].mean()) for i, s in enumerate(seeds)})
            rows.append(row)
        for a, b in pairs:
            if a in mats and b in mats:
                p, lo, hi, backend = poi_ci(mats[a][0], mats[b][0], n_boot, seed, use_rliable)
                prow.append(
                    {
                        "x": a,
                        "y": b,
                        "split": split,
                        "n_cells": len(cells),
                        "n_seeds_x": len(mats[a][1]),
                        "n_seeds_y": len(mats[b][1]),
                        "poi": p,
                        "poi_lo": lo,
                        "poi_hi": hi,
                        "ci_backend": backend,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prow)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--out", default=None, help="default {root}/robustness")
    p.add_argument("--exps", nargs="*", default=list(CONDITIONS))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    root = Path(args.root)
    out = Path(args.out or root / "robustness")
    df = load_eval(root, tuple(args.exps))
    if df.empty:
        raise SystemExit(f"no eval CSVs under {root / 'eval'}")
    # POI pairs follow the condition names: DR vs NoDR, DR+Hist vs DR, for any task prefix
    task = args.exps[0].split("_")[0]
    pairs = ((f"{task}_dr", f"{task}_nodr"), (f"{task}_drhist", f"{task}_dr"))
    rob, poi = robustness_tables(df, tuple(args.exps), pairs, args.n_boot, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    rob.to_csv(out / "robustness.csv", index=False)
    poi.to_csv(out / "poi.csv", index=False)
    write_run_meta(out, vars(args), sys.argv, t0, analysis="robustness", note=IQM_NOTE)
    print(IQM_NOTE)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rob[rob.split.isin(["nominal", "heldout"])])
        print(poi)


if __name__ == "__main__":
    main()
