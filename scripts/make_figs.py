"""Generate every results figure from files in `results/` (mvp.md A8).

    python scripts/make_figs.py                       # results/ -> results/figs/
    python scripts/make_figs.py --results R --out R/figs --no-gwie

Each figure is skipped with a message when its inputs are missing. Colours: one fixed,
colour-blind-validated categorical palette; every method / monitor / condition always gets the
same colour across figures (assigned by a fixed order, never by rank).

Figures:
- c1_controller_bars.png      precision@k and Spearman per method, analytic controller
- c1_push_dr_deletion.png     closed-loop deletion curves (success vs k erased groups)
- c3_{exp}_roc.png            ROC curves of ASM and baselines from episode scores (test split)
- c4_reliance.png             group-share stacked bars per condition
- c4_temporal.png             share on history slots (DR+Hist)
- robustness.png              nominal / held-out success per condition, per-seed dots
- c1_shortcut_shares.png      shortcut v2: hint vs goal share per method (method colours)
Reach track (Gate G2 insurance):
- robustness_reach.png        success per split (nominal, held-out, per factor) per condition
- c3_reach_dr_roc.png         ROC per signal for reach_dr (needs the git-ignored
                              results/c3/c3_reach_dr_episode_scores.csv; skipped if missing)
- c4_reach_reliance.png       base-level group-share stacked bars per condition
- c4_reach_temporal.png       DR+Hist shares on t, t-1..t-3 and past actions, per scope
- gwie_c_erasure_order.png    GWIE-C erasure order for 3 example states (needs checkpoints and
                              step logs; computed with projectv.xai.gwie_c)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette (dataviz skill, validated light-mode order).
PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
OTHER = "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METHODS = (
    "grad",
    "grad_x_delta",
    "ig",
    "smoothgrad",
    "occlusion",
    "kernelshap_grouped",
    "gwie_c",
    "gwie_orig_port",
)
METHOD_COLORS = dict(zip(METHODS, PALETTE, strict=False))
MONITORS = ("asm", "obs_ood", "act_std", "q_drop", "ensemble", "act_ood")
MONITOR_COLORS = dict(zip(MONITORS, PALETTE, strict=False))
CONDITIONS = ("push_nodr", "push_dr", "push_drhist")
CONDITION_LABELS = {"push_nodr": "NoDR", "push_dr": "DR", "push_drhist": "DR+Hist"}
CONDITION_COLORS = dict(zip(CONDITIONS, PALETTE, strict=False))
# Reach track (Gate G2): the same condition keeps the same colour across tasks.
REACH_CONDITIONS = ("reach_nodr", "reach_dr", "reach_drhist")
_REACH_TO_PUSH = dict(zip(REACH_CONDITIONS, CONDITIONS, strict=True))
CONDITION_LABELS.update({r: CONDITION_LABELS[p] for r, p in _REACH_TO_PUSH.items()})
CONDITION_COLORS.update({r: CONDITION_COLORS[p] for r, p in _REACH_TO_PUSH.items()})
SPLIT_COLORS = {"nominal": PALETTE[0], "heldout": PALETTE[1]}

Log = Callable[[str], None]


def color_for(name: str, table: dict[str, str]) -> str:
    key = _norm(name)
    for k, c in table.items():
        if _norm(k) == key:
            return c
    return OTHER


def _norm(s: object) -> str:
    s = str(s).lower()
    aliases = {
        "nodr": "push_nodr",
        "dr": "push_dr",
        "dr+hist": "push_drhist",
        "drhist": "push_drhist",
    }
    s = aliases.get(s.replace(" ", ""), s)
    return re.sub(r"[^a-z0-9]", "", s)


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path, log: Log) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)


def _yerr(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if f"{col}_lo" in df.columns and f"{col}_hi" in df.columns:
        v = df[col].to_numpy(float)
        return np.vstack([v - df[f"{col}_lo"].to_numpy(float), df[f"{col}_hi"].to_numpy(float) - v])
    return None


def _skip(name: str, why: str, log: Log) -> None:
    log(f"skip {name}: {why}")


# ---------------------------------------------------------------------------- C1


def fig_c1_controller(root: Path, out: Path, log: Log) -> None:
    name = "c1_controller_bars.png"
    paths = {t: root / "c1" / f"c1_{t}_controller_summary.csv" for t in ("reach", "push")}
    dfs = {t: d for t, p in paths.items() if (d := _read(p)) is not None}
    if not dfs:
        return _skip(name, "no results/c1/c1_*_controller_summary.csv", log)
    metrics = [("precision_at_k", "precision@k"), ("spearman", "Spearman ρ vs analytic")]
    fig, axes = plt.subplots(
        len(metrics), len(dfs), figsize=(4.2 * len(dfs), 5.6), squeeze=False, sharey="row"
    )
    for j, (task, df) in enumerate(dfs.items()):
        x = np.arange(len(df))
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            colors = [color_for(m, METHOD_COLORS) for m in df["method"]]
            ax.bar(x, df[col], color=colors, width=0.7, edgecolor="white", linewidth=1)
            err = _yerr(df, col)
            if err is not None:
                ax.errorbar(x, df[col], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x, df["method"], rotation=45, ha="right")
            _style(ax)
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=9)
            if i == 0:
                ax.set_title(f"{task.capitalize()} env, analytic controller", fontsize=10)
    fig.suptitle(
        "C1: attribution faithfulness vs analytic ground truth (mean, 95% CI)", fontsize=11
    )
    fig.tight_layout()
    _save(fig, out / name, log)


def fig_c1_deletion(root: Path, out: Path, log: Log) -> None:
    name = "c1_push_dr_deletion.png"
    df = _read(root / "c1" / "c1_push_dr_deletion.csv")
    if df is None:
        return _skip(name, "no results/c1/c1_push_dr_deletion.csv", log)
    kcol = _col(df, ("k", "n_erased", "n_groups"))
    vcol = _col(df, ("success", "success_rate", "value", "score"))
    if "method" not in df.columns or kcol is None or vcol is None:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, d in df.groupby("method", sort=False):
        agg = d.groupby(kcol)[vcol].mean().sort_index()
        c = color_for(m, METHOD_COLORS)
        ax.plot(agg.index, agg.values, marker="o", ms=4, lw=2, color=c, label=str(m))
        lo, hi = f"{vcol}_lo", f"{vcol}_hi"
        if lo in d.columns and hi in d.columns:
            band = d.groupby(kcol)[[lo, hi]].mean().sort_index()
            ax.fill_between(band.index, band[lo], band[hi], color=c, alpha=0.12, lw=0)
    ax.set_xlabel("groups erased (top-k by method's ranking), every step", fontsize=9)
    ax.set_ylabel(vcol, fontsize=9)
    ax.set_title("C1 push_dr: closed-loop deletion curves", fontsize=10)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out / name, log)


# ---------------------------------------------------------------------------- C3


def fig_c3_roc(root: Path, out: Path, log: Log, exp: str = "push_dr") -> None:
    from sklearn.metrics import roc_curve

    name = f"c3_{exp}_roc.png"
    df = _read(root / "c3" / f"c3_{exp}_episode_scores.csv")
    if df is None:
        return _skip(name, f"no results/c3/c3_{exp}_episode_scores.csv", log)
    mcol = _col(df, ("monitor", "method", "signal"))
    if mcol is None:
        return _skip(name, "no monitor column", log)
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"]
    failed = _failed(df)
    if failed is None:
        return _skip(name, "no failed/success column", log)
    tcols = [(T, c) for T in (10, 25) if (c := _col(df, (f"score_T{T}", f"S{T}", f"T{T}")))]
    if not tcols:
        return _skip(name, "no score_T10/score_T25 columns", log)
    fig, axes = plt.subplots(1, len(tcols), figsize=(4.4 * len(tcols), 4.2), squeeze=False)
    for ax, (T, scol) in zip(axes[0], tcols, strict=False):
        ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=1)
        for m in _ordered(df[mcol].unique(), MONITORS):
            sel = (df[mcol] == m).to_numpy()
            y, s = failed[sel], df.loc[sel, scol].to_numpy(float)
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            lw = 2.5 if _norm(m) == "asm" else 1.5
            ax.plot(fpr, tpr, color=color_for(m, MONITOR_COLORS), lw=lw, label=str(m))
        ax.set_xlabel("false-positive rate", fontsize=9)
        ax.set_ylabel("true-positive rate", fontsize=9)
        ax.set_title(f"first {T} of 50 steps", fontsize=10)
        ax.set_aspect("equal")
        _style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"C3 {exp}: failure prediction on held-out dynamics (test split)", fontsize=11)
    fig.tight_layout()
    _save(fig, out / name, log)


def _failed(df: pd.DataFrame) -> np.ndarray | None:
    if "failed" in df.columns:
        return df["failed"].astype(bool).to_numpy().astype(int)
    if "success" in df.columns:
        return (~df["success"].astype(bool)).to_numpy().astype(int)
    return None


def _ordered(values, order: tuple[str, ...]) -> list:
    known = [v for o in order for v in values if _norm(v) == _norm(o)]
    return known + [v for v in values if v not in known]


# ---------------------------------------------------------------------------- C4


def fig_c4_reliance(root: Path, out: Path, log: Log, max_groups: int = 7) -> None:
    name = "c4_reliance.png"
    df = _read(root / "c4" / "reliance.csv")
    if df is None:
        return _skip(name, "no results/c4/reliance.csv", log)
    ccol = _col(df, ("condition", "exp", "policy"))
    vcol = _col(df, ("share", "mean_share"))
    if ccol is None or vcol is None or "group" not in df.columns:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    title = "C4: what each training condition relies on"
    _reliance_bars(df, ccol, vcol, out / name, log, title, max_groups)


def fig_c4_reach_reliance(root: Path, out: Path, log: Log) -> None:
    """Reach track: base-level group shares (`level == base`, `quantity == share`)."""
    name = "c4_reach_reliance.png"
    df = _read(root / "c4_reach" / "reliance.csv")
    if df is None:
        return _skip(name, "no results/c4_reach/reliance.csv", log)
    ccol = _col(df, ("exp", "condition", "policy"))
    if ccol is None or "mean" not in df.columns or "group" not in df.columns:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    if "level" in df.columns:
        df = df[df["level"].astype(str) == "base"]
    if "quantity" in df.columns:
        df = df[df["quantity"].astype(str) == "share"]
    if not len(df):
        return _skip(name, "no base-level share rows", log)
    title = "C4 Reach: group attribution shares per training condition (nominal states)"
    _reliance_bars(df, ccol, "mean", out / name, log, title)


def _reliance_bars(
    df: pd.DataFrame, ccol: str, vcol: str, path: Path, log: Log, title: str, max_groups: int = 7
) -> None:
    mcol = _col(df, ("method",))
    methods = list(df[mcol].unique()) if mcol else [None]
    # Fixed group order + colours: largest mean share overall first; the tail folds into
    # "other" so no hue is generated past the palette.
    mean_share = df.groupby("group")[vcol].mean().sort_values(ascending=False)
    top = list(mean_share.index[:max_groups])
    gcolors = {g: PALETTE[i] for i, g in enumerate(top)}
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 4.2), squeeze=False)
    for ax, m in zip(axes[0], methods, strict=False):
        d = df if m is None else df[df[mcol] == m]
        conds = _ordered(list(d[ccol].unique()), CONDITIONS + REACH_CONDITIONS)
        piv = d.groupby([ccol, "group"])[vcol].mean().unstack(fill_value=0.0).reindex(conds)
        piv["other"] = piv.drop(columns=[g for g in top if g in piv.columns]).sum(axis=1)
        bottom = np.zeros(len(conds))
        x = np.arange(len(conds))
        for g in [*top, "other"]:
            if g not in piv.columns or (g == "other" and not piv[g].abs().sum()):
                continue
            v = piv[g].to_numpy(float)
            ax.bar(
                x,
                v,
                bottom=bottom,
                color=gcolors.get(g, OTHER),
                width=0.6,
                edgecolor="white",
                linewidth=2,
                label=g,
            )
            bottom += v
        ax.set_xticks(x, [CONDITION_LABELS.get(c, c) for c in conds])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("attribution share (nominal states)", fontsize=9)
        ax.set_title(str(m) if m is not None else "", fontsize=10)
        _style(ax)
    h, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        h[::-1],
        labels[::-1],
        fontsize=7,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path, log)


def fig_c4_temporal(root: Path, out: Path, log: Log) -> None:
    name = "c4_temporal.png"
    df = _read(root / "c4" / "temporal.csv")
    if df is None:
        return _skip(name, "no results/c4/temporal.csv", log)
    scol = _col(df, ("slot", "time_slot", "lag"))
    vcol = _col(df, ("share", "mean_share"))
    if scol is None or vcol is None:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    hcol = _col(df, ("regime", "split", "cells", "cell_set"))
    hues = list(df[hcol].unique()) if hcol else [None]
    slots = list(dict.fromkeys(df[scol].astype(str)))
    x = np.arange(len(slots))
    w = 0.8 / len(hues)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, h in enumerate(hues):
        d = df if h is None else df[df[hcol] == h]
        d = d.assign(_s=d[scol].astype(str)).groupby("_s").mean(numeric_only=True).reindex(slots)
        color = SPLIT_COLORS.get(_norm(h), PALETTE[i % len(PALETTE)]) if h else PALETTE[0]
        xs = x + (i - (len(hues) - 1) / 2) * w
        ax.bar(xs, d[vcol], width=w * 0.92, color=color, label=str(h) if h else None)
        err = _yerr(d, vcol)
        if err is not None:
            ax.errorbar(xs, d[vcol], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.set_xticks(x, slots)
    ax.set_ylabel("attribution share", fontsize=9)
    ax.set_title("C4 DR+Hist: reliance on history slots and past actions", fontsize=10)
    _style(ax)
    if hcol:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out / name, log)


# ---------------------------------------------------------------------------- robustness


def fig_robustness(root: Path, out: Path, log: Log) -> None:
    name = "robustness.png"
    df = _read(root / "robustness" / "robustness.csv")
    if df is None:
        return _skip(name, "no results/robustness/robustness.csv", log)
    ccol = _col(df, ("exp", "condition", "policy"))
    vcol = _col(df, ("success", "success_mean", "success_rate"))
    hcol = _col(df, ("split", "cells", "cell_set", "regime", "eval_set"))
    if ccol is None or vcol is None:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    seed_cols = [c for c in df.columns if re.fullmatch(r"(success_)?s(eed)?_?\d+", c)]
    conds = _ordered(list(df[ccol].unique()), CONDITIONS)
    hues = list(df[hcol].unique()) if hcol else [None]
    x = np.arange(len(conds))
    w = 0.8 / len(hues)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, h in enumerate(hues):
        d = (df if h is None else df[df[hcol] == h]).set_index(ccol).reindex(conds)
        xs = x + (i - (len(hues) - 1) / 2) * w
        color = SPLIT_COLORS.get(_norm(h), PALETTE[i % len(PALETTE)]) if h else PALETTE[0]
        ax.bar(xs, d[vcol], width=w * 0.92, color=color, label=str(h) if h else None)
        err = _yerr(d, vcol)
        if err is not None:
            ax.errorbar(xs, d[vcol], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for sc in seed_cols:
            ax.scatter(xs, d[sc], s=14, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xticks(x, [CONDITION_LABELS.get(c, c) for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate", fontsize=9)
    note = " (dots: per seed)" if seed_cols else ""
    ax.set_title(f"Success on nominal vs held-out dynamics, mean and 95% CI{note}", fontsize=10)
    _style(ax)
    if hcol:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out / name, log)


# ---------------------------------------------------------------------------- Reach track (G2)


def fig_robustness_reach(root: Path, out: Path, log: Log) -> None:
    """Success per split (nominal, held-out, per held-out factor), one bar per condition in its
    fixed colour, 95% CI whiskers and per-seed dots."""
    name = "robustness_reach.png"
    df = _read(root / "robustness_reach" / "robustness.csv")
    if df is None:
        return _skip(name, "no results/robustness_reach/robustness.csv", log)
    ccol = _col(df, ("exp", "condition", "policy"))
    vcol = _col(df, ("success", "success_mean", "success_rate"))
    hcol = _col(df, ("split", "cells", "cell_set", "regime", "eval_set"))
    if ccol is None or vcol is None or hcol is None:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    seed_cols = [c for c in df.columns if re.fullmatch(r"(success_)?s(eed)?_?\d+", c)]
    splits = list(dict.fromkeys(df[hcol].astype(str)))
    conds = _ordered(list(df[ccol].unique()), REACH_CONDITIONS)
    x = np.arange(len(splits))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(splits) + 1.5), 3.8))
    for i, c in enumerate(conds):
        d = df[df[ccol] == c].assign(_s=lambda t: t[hcol].astype(str)).set_index("_s")
        d = d.reindex(splits)
        xs = x + (i - (len(conds) - 1) / 2) * w
        label = CONDITION_LABELS.get(c, str(c))
        ax.bar(xs, d[vcol], width=w * 0.92, color=color_for(c, CONDITION_COLORS), label=label)
        err = _yerr(d, vcol)
        if err is not None:
            ax.errorbar(xs, d[vcol], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for sc in seed_cols:
            ax.scatter(xs, d[sc], s=12, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xticks(x, [s.replace("factor:", "") for s in splits])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate", fontsize=9)
    note = ", dots: per seed" if seed_cols else ""
    ax.set_title(
        f"FetchReach: nominal vs held-out dynamics (sim-to-sim), mean and 95% CI{note}",
        fontsize=10,
    )
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out / name, log)


def fig_c4_reach_temporal(root: Path, out: Path, log: Log) -> None:
    """DR+Hist (Reach): share on the current step, the 3 history slots and past actions, per
    method, for each scope (nominal, held-out, per factor)."""
    name = "c4_reach_temporal.png"
    df = _read(root / "c4_reach" / "temporal.csv")
    if df is None:
        return _skip(name, "no results/c4_reach/temporal.csv", log)
    scol = _col(df, ("scope", "split", "regime"))
    if scol is None or "group" not in df.columns or "mean" not in df.columns:
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    if "quantity" in df.columns:
        df = df[df["quantity"].astype(str) == "share"]
    mcol = _col(df, ("method",))
    methods = list(df[mcol].unique()) if mcol else [None]
    slots = list(dict.fromkeys(df["group"].astype(str)))
    scopes = list(dict.fromkeys(df[scol].astype(str)))
    extra = iter(PALETTE[2:])
    scolors = {s: SPLIT_COLORS.get(s) or next(extra, OTHER) for s in scopes}
    x = np.arange(len(slots))
    w = 0.8 / len(scopes)
    fig, axes = plt.subplots(
        1, len(methods), figsize=(5.2 * len(methods), 3.8), squeeze=False, sharey=True
    )
    for ax, m in zip(axes[0], methods, strict=False):
        dm = df if m is None else df[df[mcol] == m]
        for i, s in enumerate(scopes):
            d = dm[dm[scol].astype(str) == s].assign(_g=lambda t: t["group"].astype(str))
            d = d.set_index("_g").reindex(slots)
            xs = x + (i - (len(scopes) - 1) / 2) * w
            ax.bar(xs, d["mean"], width=w * 0.92, color=scolors[s], label=s.replace("factor:", ""))
            err = _yerr(d, "mean")
            if err is not None:
                ax.errorbar(xs, d["mean"], yerr=err, fmt="none", ecolor=INK, elinewidth=0.8)
        ax.set_xticks(x, slots)
        ax.set_title(str(m) if m is not None else "", fontsize=10)
        _style(ax)
    axes[0, 0].set_ylabel("attribution share", fontsize=9)
    axes[0, -1].legend(fontsize=7, frameon=False, title="scope", title_fontsize=7)
    fig.suptitle("C4 Reach DR+Hist: reliance on history slots and past actions", fontsize=11)
    fig.tight_layout()
    _save(fig, out / name, log)


def _per_seed(v: object) -> list[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [float(x) for x in str(v).split(";") if x.strip()]


def fig_c1_shortcut(root: Path, out: Path, log: Log) -> None:
    """Shortcut v2 (ground truth B): share on the planted hint vs on the goal per method, in the
    method's fixed colour (hint solid, goal hatched), 95% CI and per-seed hint dots. The measured
    success drop with a random hint (true reliance) is printed from the same file."""
    name = "c1_shortcut_shares.png"
    df = _read(root / "c1" / "c1_shortcut_summary.csv")
    if df is None:
        return _skip(name, "no results/c1/c1_shortcut_summary.csv", log)
    if not {"method", "hint_share", "goal_share"} <= set(df.columns):
        return _skip(name, f"unexpected columns {list(df.columns)}", log)
    df = df.set_index("method").reindex(_ordered(list(df["method"]), METHODS)).reset_index()
    x = np.arange(len(df))
    w = 0.38
    colors = [color_for(m, METHOD_COLORS) for m in df["method"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for off, col, hatch, alpha in (
        (-w / 2, "hint_share", None, 1.0),
        (w / 2, "goal_share", "///", 0.35),
    ):
        ax.bar(
            x + off,
            df[col],
            width=w * 0.95,
            color=colors,
            alpha=alpha,
            hatch=hatch,
            edgecolor=colors if hatch else "white",
            linewidth=1,
        )
        err = _yerr(df, col)
        if err is not None:
            ax.errorbar(x + off, df[col], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    if "hint_share_per_seed" in df.columns:
        for xi, v in zip(x, df["hint_share_per_seed"], strict=True):
            ys = _per_seed(v)
            ax.scatter([xi - w / 2] * len(ys), ys, s=12, color="white", edgecolor=INK, zorder=3)
    ax.set_xticks(x, df["method"], rotation=40, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("attribution share", fontsize=9)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=OTHER, label="hint (planted shortcut)"),
            Patch(facecolor=OTHER, alpha=0.35, hatch="///", edgecolor=OTHER, label="goal"),
        ],
        fontsize=8,
        frameon=False,
    )
    sub = ""
    if {"success_drop", "success_drop_lo", "success_drop_hi"} <= set(df.columns):
        r = df.iloc[0]
        sub = (
            f"\nmeasured reliance: success drop with a random hint "
            f"{r['success_drop']:.3f} [{r['success_drop_lo']:.3f}, {r['success_drop_hi']:.3f}]"
        )
    ax.set_title(f"C1 shortcut v2: hint vs goal share per method (mean, 95% CI){sub}", fontsize=9)
    _style(ax)
    fig.tight_layout()
    _save(fig, out / name, log)


# ---------------------------------------------------------------------------- GWIE-C


def load_policy(path: Path, exp: dict):
    """Shared with make_gifs (HER-safe checkpoint loading)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_gifs import load_policy as _load

    return _load(path, exp)


def gwie_examples(root: Path, log: Log, seed: int = 0, pool_size: int = 2000) -> list[dict]:
    """Up to 3 example states with the policy, groups and pool needed by gwie_c:
    (1) a nominal success state of push_dr, (2) a state from a held-out push_dr failure,
    (3) a reach_shortcut state with a random hint. Each is skipped if inputs are missing."""
    from projectv.eval.rollout import resolve_checkpoint
    from projectv.eval.steplog import load_steplog
    from projectv.utils.config import load_exp
    from projectv.xai.groups import make_groups
    from projectv.xai.policy_fn import SB3Policy

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    steps = root / "steplogs" / f"push_dr_s{seed}"
    try:
        ckpt = resolve_checkpoint("push_dr", seed, root)
    except FileNotFoundError:
        ckpt = None
    nominal = steps / "cell00.npz"
    if ckpt is None or not nominal.exists():
        log(f"  gwie: skip push_dr examples (checkpoint or {nominal} missing)")
    else:
        pol = SB3Policy(load_policy(ckpt, load_exp("configs/exp/push_dr.yaml")))
        nom = load_steplog(nominal)
        flat = nom["obs_flat"].reshape(-1, nom["obs_flat"].shape[-1]).astype(np.float64)
        pool = flat[rng.choice(len(flat), min(pool_size, len(flat)), replace=False)]
        groups = make_groups("push")
        ok = np.flatnonzero(nom["success"])
        if len(ok):
            out.append(
                dict(
                    title=f"push_dr s{seed}: nominal success (ep {ok[0]}, t=10)",
                    pol=pol,
                    s=nom["obs_flat"][ok[0], 10],
                    groups=groups,
                    pool=pool,
                )
            )
        for f in sorted(steps.glob("cell*.npz")):
            if f.name == "cell00.npz":
                continue
            log_ = load_steplog(f)
            bad = np.flatnonzero(~log_["success"])
            if len(bad):
                out.append(
                    dict(
                        title=f"push_dr s{seed}: held-out failure ({f.stem}, ep {bad[0]}, t=10)",
                        pol=pol,
                        s=log_["obs_flat"][bad[0], 10],
                        groups=groups,
                        pool=pool,
                    )
                )
                break
    try:
        sc_ckpt = resolve_checkpoint("reach_shortcut", seed, root)
    except FileNotFoundError:
        sc_ckpt = None
    if sc_ckpt is None:
        log("  gwie: skip shortcut example (no reach_shortcut checkpoint)")
    else:
        from projectv.eval.cells import NOMINAL
        from projectv.eval.rollout import run_episodes
        from projectv.utils.seeding import eval_episode_seed

        exp = load_exp("configs/exp/reach_shortcut.yaml")
        model = load_policy(sc_ckpt, exp)

        spec = NOMINAL.dr_spec()
        seeds = [eval_episode_seed(seed, 0, ep) for ep in range(20)]
        pool_rec = run_episodes(model, exp, spec, seeds, log=True, shortcut_mode="intact")
        flat = pool_rec["obs_flat"].reshape(-1, pool_rec["obs_flat"].shape[-1]).astype(np.float64)
        rand = run_episodes(model, exp, spec, seeds[:1], log=True, shortcut_mode="random")
        out.append(
            dict(
                title=f"reach_shortcut s{seed}: random hint (t=10)",
                pol=SB3Policy(model),
                s=rand["obs_flat"][0, 10],
                groups=make_groups("reach", shortcut=True),
                pool=flat,
            )
        )
    return out


def fig_gwie_erasure(root: Path, out: Path, log: Log, seed: int = 0) -> None:
    name = "gwie_c_erasure_order.png"
    examples = gwie_examples(root, log, seed)
    if not examples:
        return _skip(name, "no checkpoints/step logs for the example states", log)
    from projectv.xai.gwie_c import gwie_c

    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 4.4), squeeze=False)
    for ax, ex in zip(axes[0], examples, strict=False):
        res = gwie_c(ex["pol"], ex["s"], ex["groups"], ex["pool"], np.random.default_rng(seed))
        draw_erasure(ax, res.order, res.shares, res.divergence_trace)
        ax.set_title(ex["title"], fontsize=8)
    fig.suptitle("GWIE-C: erasure order (top = erased first), share of KL gain", fontsize=11)
    fig.tight_layout()
    _save(fig, out / name, log)


def draw_erasure(ax: plt.Axes, order: list[str], shares: dict[str, float], trace: list[float]):
    y = np.arange(len(order))[::-1]
    vals = [shares[g] for g in order]
    ax.barh(y, vals, color=PALETTE[6], height=0.7)
    ax.set_yticks(y, [f"{i + 1}. {g}" for i, g in enumerate(order)], fontsize=7)
    for yi, v, kl in zip(y, vals, trace[1:], strict=False):
        ax.text(v, yi, f" {v:.2f} (cum. KL {kl:.3g})", va="center", fontsize=6, color=INK2)
    ax.set_xlabel("share of marginal KL gain", fontsize=8)
    ax.set_xlim(0, max(max(vals, default=0) * 1.6, 0.1))
    _style(ax)
    ax.xaxis.grid(True, color=GRID)
    ax.yaxis.grid(False)


# ---------------------------------------------------------------------------- main

FIGS = {
    "c1_controller": fig_c1_controller,
    "c1_deletion": fig_c1_deletion,
    "c3_roc": lambda r, o, log: [fig_c3_roc(r, o, log, e) for e in CONDITIONS],
    "c4_reliance": fig_c4_reliance,
    "c4_temporal": fig_c4_temporal,
    "robustness": fig_robustness,
    "c1_shortcut": fig_c1_shortcut,
    "c3_roc_reach": lambda r, o, log: fig_c3_roc(r, o, log, "reach_dr"),
    "c4_reach_reliance": fig_c4_reach_reliance,
    "c4_reach_temporal": fig_c4_reach_temporal,
    "robustness_reach": fig_robustness_reach,
    "gwie": fig_gwie_erasure,
}


def make_all(root: Path, out: Path, skip: set[str] = frozenset(), log: Log = print) -> None:
    for key, fn in FIGS.items():
        if key in skip:
            _skip(key, "disabled", log)
            continue
        try:
            fn(root, out, log)
        except Exception as e:  # one broken input must not stop the other figures
            log(f"FAILED {key}: {type(e).__name__}: {e}")
            plt.close("all")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="default: {results}/figs")
    p.add_argument("--no-gwie", action="store_true", help="skip the GWIE-C figure (slow)")
    args = p.parse_args(argv)
    root = Path(args.results)
    make_all(
        root, Path(args.out) if args.out else root / "figs", {"gwie"} if args.no_gwie else set()
    )


if __name__ == "__main__":
    sys.exit(main())
