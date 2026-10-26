"""Generate every results table from files in `results/` (CLAUDE.md §2.3, mvp.md A8).

    python scripts/make_tables.py                      # results/ -> results/tables/, fills docs
    python scripts/make_tables.py --results R --out R/tables --docs README.md report/report.md

Reads the result files listed in docs/interfaces.md ("Result files consumed by make_tables").
For each table it writes `results/tables/{name}.md`. A missing input gives a table that says
TBD. Numbers are formatted `mean [lo, hi]` (3 decimals) wherever `{metric}_lo` / `{metric}_hi`
columns exist. `results/tables/numbers.json` holds the mvp.md §14 headline numbers (X, Y, Z, W)
or null.

Finally, the content between `<!-- BEGIN:name -->` and `<!-- END:name -->` in each doc is
replaced by the generated block `name` (idempotent; unknown names are left untouched). A marker
pair on one line is filled inline (single-line blocks such as `num_X`).

Numbers are never typed by hand: everything shown comes from the CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

C3_EXPS = ("push_dr", "push_drhist", "push_nodr")
C3_BASELINES = ("obs_ood", "act_std", "q_drop", "ensemble", "act_ood")
RUNTIME_COLS = ("runtime_ms", "runtime_ms_lo", "runtime_ms_hi")
TBD = "TBD"

# Column-name aliases for files whose exact schema is owned by other modules.
EXP_COLS = ("exp", "condition", "policy")
SPLIT_COLS = ("split", "cells", "cell_set", "regime", "eval_set")
SUCCESS_COLS = ("success", "success_mean", "success_rate")
MONITOR_COLS = ("monitor", "method", "signal")
T_COLS = ("T", "prefix", "t_prefix")


# ---------------------------------------------------------------------------- formatting


def fmt_num(v: object) -> str:
    """3 decimals for floats, integers as-is, `n/a` for NaN/None."""
    if v is None:
        return "n/a"
    if isinstance(v, bool | np.bool_):
        return str(bool(v))
    if isinstance(v, int | np.integer):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "n/a"
    s = f"{f:.3f}"
    return "0.000" if s == "-0.000" else s


def fmt_ci(v: object, lo: object, hi: object) -> str:
    return f"{fmt_num(v)} [{fmt_num(lo)}, {fmt_num(hi)}]"


def ci_frame(df: pd.DataFrame, drop: Iterable[str] = ()) -> pd.DataFrame:
    """Collapse `{m}, {m}_lo, {m}_hi` into one `mean [lo, hi]` string column; format the rest."""
    drop = set(drop)
    out: dict[str, list[str]] = {}
    cols = [c for c in df.columns if c not in drop]
    for c in cols:
        if c.endswith(("_lo", "_hi")) and c[:-3] in df.columns:
            continue
        if f"{c}_lo" in df.columns and f"{c}_hi" in df.columns:
            out[c] = [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(df[c], df[f"{c}_lo"], df[f"{c}_hi"], strict=False)
            ]
        elif pd.api.types.is_integer_dtype(df[c]):
            out[c] = [str(int(v)) for v in df[c]]
        elif pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            out[c] = [fmt_num(float(v)) for v in df[c]]
        elif c.endswith("_per_seed"):  # "0.2938;0.225;0.2959" -> "0.294 / 0.225 / 0.296"
            out[c] = [fmt_seed_list(v) for v in df[c]]
        else:
            out[c] = ["" if pd.isna(v) else str(v) for v in df[c]]
    return pd.DataFrame(out, index=df.index)


def fmt_seed_list(v: object) -> str:
    """A `;`-separated per-seed list, each value with 3 decimals."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return " / ".join(fmt_num(x.strip()) for x in str(v).split(";") if x.strip())


def to_markdown(df: pd.DataFrame) -> str:
    """Plain GitHub markdown (no tabulate dependency)."""

    def cell(x: object) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def tbd_block(missing: Iterable[Path], root: Path) -> str:
    names = ", ".join(f"`{_rel(p, root)}`" for p in missing)
    return f"| status |\n|---|\n| {TBD} (missing: {names}) |"


def _rel(p: Path, root: Path) -> str:
    try:
        return (Path(root.name) / p.relative_to(root)).as_posix()
    except ValueError:
        return p.as_posix()


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


# ---------------------------------------------------------------------------- tables


def _generic(root: Path, rel: str, drop: Iterable[str] = ()) -> str:
    p = root / rel
    df = _read(p)
    return tbd_block([p], root) if df is None else to_markdown(ci_frame(df, drop))


def table_c1_controller(root: Path) -> str:
    paths = [root / "c1" / f"c1_{t}_controller_summary.csv" for t in ("reach", "push")]
    dfs = [d for d in map(_read, paths) if d is not None]
    if not dfs:
        return tbd_block(paths, root)
    df = pd.concat(dfs, ignore_index=True)
    return to_markdown(ci_frame(df, drop=RUNTIME_COLS))


def table_c2_runtime(root: Path) -> str:
    """Runtime per state (ms) for every method, one column per C1 setting (median [CI])."""
    paths = [
        root / "c1" / "c1_reach_controller_summary.csv",
        root / "c1" / "c1_push_controller_summary.csv",
        root / "c1" / "c1_shortcut_summary.csv",
        root / "c1" / "c1_push_dr_summary.csv",
    ]
    cols: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for p in paths:
        df = _read(p)
        if df is None or "runtime_ms" not in df.columns or "method" not in df.columns:
            continue
        setting = p.stem.removeprefix("c1_").removesuffix("_summary")
        if "task" in df.columns:
            df = df.copy()
            df["_setting"] = df["task"].astype(str)
        else:
            df = df.assign(_setting=setting)
        has_ci = {"runtime_ms_lo", "runtime_ms_hi"} <= set(df.columns)
        for _, r in df.iterrows():
            m = str(r["method"])
            if m not in order:
                order.append(m)
            val = (
                fmt_ci(r["runtime_ms"], r["runtime_ms_lo"], r["runtime_ms_hi"])
                if has_ci
                else fmt_num(float(r["runtime_ms"]))
            )
            cols.setdefault(str(r["_setting"]), {})[m] = val
    if not cols:
        return tbd_block(paths, root)
    out = pd.DataFrame({"method": order})
    for setting, vals in cols.items():
        out[f"{setting} (ms)"] = [vals.get(m, "n/a") for m in order]
    return to_markdown(out)


def table_c3(root: Path, exp: str, per_factor: bool) -> str:
    suffix = "per_factor" if per_factor else "metrics"
    p = root / "c3" / f"c3_{exp}_{suffix}.csv"
    df = _read(p)
    if df is None:
        return tbd_block([p], root)
    keys = [c for c in (_first_col(df, T_COLS), _first_col(df, ("factor",))) if c]
    if keys:
        df = df.sort_values(keys, kind="stable")
    return to_markdown(ci_frame(df))


def table_c3_metrics(root: Path, exp: str) -> str:
    """C3 metrics table; adds `n_seeds` (count of non-empty `auroc_s{n}` columns) when the file
    has per-seed AUROCs but no `n_seeds` column."""
    p = root / "c3" / f"c3_{exp}_metrics.csv"
    df = _read(p)
    if df is None:
        return tbd_block([p], root)
    seed_cols = _seed_cols(df, "auroc")
    if seed_cols and "n_seeds" not in df.columns:
        df = df.copy()
        df["n_seeds"] = df[seed_cols].notna().sum(axis=1).astype(int)
    tcol = _first_col(df, T_COLS)
    if tcol:
        df = df.sort_values([tcol], kind="stable")
    return to_markdown(ci_frame(df))


def _seed_cols(df: pd.DataFrame, metric: str | None = None) -> list[str]:
    """Per-seed columns: `seed{n}` / `s{n}` (metric None) or `{metric}_s{n}`."""
    pat = rf"{re.escape(metric)}_s\d+" if metric else r"s(eed)?_?\d+"
    return [c for c in df.columns if re.fullmatch(pat, c)]


def _seeds_txt(row: pd.Series, seed_cols: list[str]) -> str:
    vals = [fmt_num(row[c]) for c in seed_cols if c in row.index and not pd.isna(row[c])]
    return " / ".join(vals) if vals else "n/a"


def _ci_seeds_cell(row: pd.Series, col: str, seed_cols: list[str]) -> str:
    s = fmt_ci(row[col], row.get(f"{col}_lo"), row.get(f"{col}_hi"))
    if seed_cols:
        s += f"; seeds {_seeds_txt(row, seed_cols)}"
    return s


# ---------------------------------------------------------------------------- Reach track

REACH_CONDS = ("reach_nodr", "reach_dr", "reach_drhist")
REACH_LABELS = {"reach_nodr": "NoDR", "reach_dr": "DR", "reach_drhist": "DR+Hist"}
PUSH_CONDS = ("push_nodr", "push_dr", "push_drhist")
PUSH_LABELS = {"push_nodr": "NoDR", "push_dr": "DR", "push_drhist": "DR+Hist"}
# track -> (conditions, labels, robustness dir, C4 dir, env name)
TRACKS = {
    "reach": (REACH_CONDS, REACH_LABELS, "robustness_reach", "c4_reach", "FetchReach"),
    "push": (PUSH_CONDS, PUSH_LABELS, "robustness", "c4", "FetchPush"),
}


def _ordered(values: Iterable[str], order: Iterable[str]) -> list[str]:
    vals = list(dict.fromkeys(values))
    known = [o for o in order if o in vals]
    return known + [v for v in vals if v not in known]


def table_robustness_reach(root: Path) -> str:
    """Success per (condition, split) with CI and per-seed values, then probability of
    improvement (P(x > y), rliable) per split."""
    rp = root / "robustness_reach" / "robustness.csv"
    pp = root / "robustness_reach" / "poi.csv"
    rob, poi = _read(rp), _read(pp)
    if rob is None:
        return tbd_block([rp], root)
    seed_cols = _seed_cols(rob)
    out = rob.copy()
    if seed_cols:
        out["per_seed"] = [_seeds_txt(r, seed_cols) for _, r in rob.iterrows()]
        out = out.drop(columns=seed_cols)
    parts = [to_markdown(ci_frame(out, drop=("ci_backend",)))]
    parts.append("Probability of improvement P(x > y) across seeds and cells (95% CI):")
    parts.append(tbd_block([pp], root) if poi is None else to_markdown(ci_frame(poi)))
    return "\n\n".join(parts)


def _long_pivot(
    df: pd.DataFrame, row_keys: list[str], col_key: str, col_order: Iterable[str], label=None
) -> pd.DataFrame:
    """Pivot a long `mean, mean_lo, mean_hi, seed{n}, n_seeds` frame: one row per `row_keys`,
    one column per `col_key` value holding `mean [lo, hi]; seeds a / b / c`. Column headers
    carry n (from `n_seeds`)."""
    seed_cols = _seed_cols(df)
    cols = _ordered(df[col_key].astype(str), col_order)
    rows: dict[tuple, dict[str, str]] = {}
    ns: dict[str, set[int]] = {c: set() for c in cols}
    for _, r in df.iterrows():
        key = tuple("" if pd.isna(r[k]) else str(r[k]) for k in row_keys)
        c = str(r[col_key])
        rows.setdefault(key, {})[c] = _ci_seeds_cell(r, "mean", seed_cols)
        if "n_seeds" in r.index and not pd.isna(r["n_seeds"]):
            ns[c].add(int(r["n_seeds"]))
    label = label or (lambda c: c)
    head = {c: f"{label(c)} (n={'/'.join(map(str, sorted(ns[c]))) or '?'})" for c in cols}
    data = [
        dict(zip(row_keys, k, strict=True)) | {head[c]: v.get(c, "n/a") for c in cols}
        for k, v in rows.items()
    ]
    return pd.DataFrame(data, columns=[*row_keys, *head.values()])


def table_c4_reach_reliance(root: Path, level: str = "base") -> str:
    p = root / "c4_reach" / "reliance.csv"
    df = _read(p)
    if df is None:
        return tbd_block([p], root)
    ecol = _first_col(df, EXP_COLS)
    if "level" in df.columns:
        df = df[df["level"].astype(str) == level]
    if not len(df) or ecol is None or "mean" not in df.columns:
        return tbd_block([p], root)
    if "quantity" in df.columns:  # non-share rows (entropy, ...) are labelled by quantity
        df = df.assign(
            group=[
                str(g) if q == "share" else str(q)
                for g, q in zip(df["group"].fillna(""), df["quantity"].astype(str), strict=True)
            ]
        )
    piv = _long_pivot(df, ["method", "group"], ecol, REACH_CONDS, REACH_LABELS.get)
    return to_markdown(piv)


def table_c4_reach_temporal(root: Path) -> str:
    p = root / "c4_reach" / "temporal.csv"
    df = _read(p)
    if df is None or "mean" not in df.columns or "group" not in df.columns:
        return tbd_block([p], root)
    if "quantity" in df.columns:
        df = df[df["quantity"].astype(str) == "share"]
    ecol = _first_col(df, EXP_COLS)
    scol = _first_col(df, ("scope", *SPLIT_COLS))
    keys = [c for c in (ecol, "method", scol) if c]
    piv = _long_pivot(df, keys, "group", ("t", "t-1", "t-2", "t-3", "act"))
    return to_markdown(piv)


def table_c4_reach_probes(root: Path) -> str:
    """One row per (condition, target, feature set): pooled R² [CI] (scope `all`), per-seed R²
    (scopes `seed{n}`), n_seeds and the claim-rule columns."""
    p = root / "c4_reach" / "probes.csv"
    df = _read(p)
    if df is None or "r2" not in df.columns:
        return tbd_block([p], root)
    ecol = _first_col(df, EXP_COLS) or "exp"
    keys = [ecol, "target", "features"]
    rows = []
    for k, d in df.groupby(keys, sort=False):
        scope = d["scope"].astype(str) if "scope" in d.columns else pd.Series("all", d.index)
        pooled = d[scope == "all"]
        r = pooled.iloc[0] if len(pooled) else d.iloc[0]
        seeds = d[scope.str.fullmatch(r"seed\d+")].sort_values("scope")
        row = dict(zip(keys, k, strict=True))
        row["r2 (pooled)"] = fmt_ci(r["r2"], r.get("r2_lo"), r.get("r2_hi"))
        row["r2 per seed"] = " / ".join(fmt_num(v) for v in seeds["r2"]) or "n/a"
        n = r.get("n_seeds")
        row["n_seeds"] = str(int(n)) if n is not None and not pd.isna(n) else str(len(seeds))
        for c in ("claim_sysid", "penult_beats_stacked"):
            if c in d.columns:
                row[c] = "n/a" if pd.isna(r[c]) else str(r[c])
        rows.append(row)
    return to_markdown(pd.DataFrame(rows)) + "\n\n" + probes_claim_block(root)


def _bool(v: object) -> bool | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v).strip().lower() in {"true", "1", "1.0", "yes"}


def probes_claim_block(root: Path, c4_dir: str = "c4_reach") -> str:
    """Claim-rule outcome per (condition, target), from `claim_sysid` in probes.csv."""
    p = root / c4_dir / "probes.csv"
    df = _read(p)
    if df is None or "claim_sysid" not in df.columns:
        return f"- Claim rule outcome (`claim_sysid`): {TBD} (missing: `{_rel(p, root)}`)."
    ecol = _first_col(df, EXP_COLS) or "exp"
    outcomes = []
    for (e, t), d in df.groupby([ecol, "target"], sort=False):
        vals = [b for b in map(_bool, d["claim_sysid"]) if b is not None]
        outcomes.append((f"{e}/{t}", any(vals) if vals else None))
    txt = ", ".join(f"{k}: **{'n/a' if v is None else v}**" for k, v in outcomes)
    lines = [f"- Claim rule outcome (`claim_sysid`, {_rel(p, root)}): {txt}."]
    claimed = [k for k, v in outcomes if v]
    lines.append(
        "- Implicit system identification is claimed for: " + ", ".join(claimed) + "."
        if claimed
        else "- No implicit system identification is claimed for any (condition, target)."
    )
    meta = root / c4_dir / "run_meta_probes.json"
    if meta.exists():
        sha = json.loads(meta.read_text(encoding="utf-8")).get("git_sha", "")
        if sha:
            lines.append(f"- Probe run code version (`run_meta_probes.json` git SHA): `{sha[:7]}`.")
    return "\n".join(lines)


def _c3_best(df: pd.DataFrame, T: int, metric: str = "auroc"):
    mcol, tcol = _first_col(df, MONITOR_COLS), _first_col(df, T_COLS)
    if mcol is None or tcol is None or metric not in df.columns:
        return None, None
    d = df[pd.to_numeric(df[tcol], errors="coerce") == T]
    asm = d[d[mcol].map(_norm) == "asm"]
    base = d[d[mcol].map(_norm).isin({_norm(b) for b in C3_BASELINES})]
    base = base[pd.to_numeric(base[metric], errors="coerce").notna()]
    best = base.loc[pd.to_numeric(base[metric]).idxmax()] if len(base) else None
    return (asm.iloc[0] if len(asm) else None), best


def _verdict(a: pd.Series, b: pd.Series, col: str = "auroc") -> str:
    try:
        alo, ahi, blo, bhi = (
            float(x) for x in (a[f"{col}_lo"], a[f"{col}_hi"], b[f"{col}_lo"], b[f"{col}_hi"])
        )
    except (KeyError, TypeError, ValueError):
        return "no CI available"
    if alo > bhi:
        return "ASM above the best baseline (CIs disjoint)"
    if ahi < blo:
        return "ASM below the best baseline (CIs disjoint)"
    return "CIs overlap"


def reach_summary_block(root: Path, T: int = 10, track: str = "reach") -> str:
    """Generated one-line-per-result summary of a track (every number from files)."""
    conds, labels, rob_dir, c4_dir, env = TRACKS[track]
    lines = []
    rob = _read(root / rob_dir / "robustness.csv")
    if rob is None:
        lines.append(f"- Held-out-dynamics (sim-to-sim) success on {env}: {TBD}.")
    else:
        parts, ns = [], set()
        for c in conds:
            v = _robustness_value(rob, {_norm(c)}, c, "")
            parts.append(f"{labels[c]} **{_pct(v)}**")
            if v and "n_seeds" in v:
                ns.add(v["n_seeds"])
        n_txt = "n=" + "/".join(map(str, sorted(ns))) + " seeds" if ns else "n=TBD seeds"
        lines.append(
            f"- Held-out-dynamics (sim-to-sim) success on {env}, mean over "
            f"{n_txt} [95% bootstrap CI]: " + ", ".join(parts) + "."
        )
    poi = _read(root / rob_dir / "poi.csv")
    if poi is not None and {"x", "y", "split", "poi"} <= set(poi.columns):
        d = poi[poi["split"].astype(str) == "heldout"]
        if len(d):
            txt = ", ".join(
                f"P({labels.get(r['x'], r['x'])} > {labels.get(r['y'], r['y'])}) "
                f"{fmt_ci(r['poi'], r.get('poi_lo'), r.get('poi_hi'))}"
                for _, r in d.iterrows()
            )
            lines.append(f"- Probability of improvement on the held-out cells: {txt}.")
    for c in conds:
        df = _read(root / "c3" / f"c3_{c}_metrics.csv")
        label = f"- C3 {c}, AUROC from the first {T} of 50 steps (test split):"
        if df is None:
            lines.append(f"{label} {TBD}.")
            continue
        a, b = _c3_best(df, T)
        if a is None or b is None:
            lines.append(f"{label} {TBD}.")
            continue
        mcol = _first_col(df, MONITOR_COLS)
        lines.append(
            f"{label} ASM {fmt_ci(a['auroc'], a.get('auroc_lo'), a.get('auroc_hi'))} vs best "
            f"baseline ({b[mcol]}) {fmt_ci(b['auroc'], b.get('auroc_lo'), b.get('auroc_hi'))}: "
            f"{_verdict(a, b)}."
        )
    first = probes_claim_block(root, c4_dir).splitlines()[0]
    lines.append(first.replace("- Claim rule outcome", "- C4 probes, claim rule outcome", 1))
    return "\n".join(lines)


def headline_numbers(root: Path) -> dict[str, dict | None]:
    """mvp.md §14 numbers: X/Y = held-out success NoDR/DR (push), Z = ASM AUROC from the first
    10 steps on push_dr, W = best of the 5 baselines at the same prefix. Null when missing."""
    out: dict[str, dict | None] = {"X": None, "Y": None, "Z": None, "W": None}
    rob_p = root / "robustness" / "robustness.csv"
    rob = _read(rob_p)
    if rob is not None:
        for key, names, desc in (
            ("X", {"pushnodr", "nodr"}, "held-out success, push_nodr"),
            ("Y", {"pushdr", "dr"}, "held-out success, push_dr"),
        ):
            out[key] = _robustness_value(rob, names, desc, _rel(rob_p, root))
    c3_p = root / "c3" / "c3_push_dr_metrics.csv"
    c3 = _read(c3_p)
    if c3 is not None:
        mcol, tcol = _first_col(c3, MONITOR_COLS), _first_col(c3, T_COLS)
        acol = _first_col(c3, ("auroc",))
        if mcol and tcol and acol:
            d = c3[pd.to_numeric(c3[tcol], errors="coerce") == 10]
            src = _rel(c3_p, root)
            asm = d[d[mcol].map(_norm) == "asm"]
            if len(asm):
                out["Z"] = _val(asm.iloc[0], acol, src, "ASM AUROC, T=10, push_dr (test)")
            base = d[d[mcol].map(_norm).isin({_norm(b) for b in C3_BASELINES})]
            if len(base):
                best = base.loc[pd.to_numeric(base[acol]).idxmax()]
                w = _val(best, acol, src, "best baseline AUROC, T=10, push_dr (test)")
                w["monitor"] = str(best[mcol])
                out["W"] = w
    return out


def _val(row: pd.Series, col: str, src: str, desc: str) -> dict:
    d = {"value": float(row[col]), "lo": None, "hi": None, "source": src, "description": desc}
    if "n_seeds" in row.index and not pd.isna(row["n_seeds"]):
        d["n_seeds"] = int(row["n_seeds"])
    for k in ("lo", "hi"):
        if f"{col}_{k}" in row.index and not pd.isna(row[f"{col}_{k}"]):
            d[k] = float(row[f"{col}_{k}"])
    seeds = {c: row[c] for c in row.index if re.fullmatch(r"seed\d+", str(c))}
    d["per_seed"] = {int(c[4:]): float(v) for c, v in seeds.items() if not pd.isna(v)}
    return d


def _robustness_value(rob: pd.DataFrame, names: set[str], desc: str, src: str) -> dict | None:
    ecol = _first_col(rob, EXP_COLS)
    if ecol is None:
        return None
    rows = rob[rob[ecol].map(_norm).isin(names)]
    scol = _first_col(rob, SPLIT_COLS)
    vcol = _first_col(rob, SUCCESS_COLS)
    if scol is not None and vcol is not None:
        rows = rows[rows[scol].map(_norm).str.startswith("heldout")]
    elif "heldout" in rob.columns:  # wide format: one column per split
        vcol = "heldout"
    if vcol is None or not len(rows):
        return None
    return _val(rows.iloc[0], vcol, src, desc)


def _pct(d: dict | None) -> str:
    if d is None:
        return TBD
    s = f"{100 * d['value']:.1f}%"
    if d.get("n_seeds") == 1:  # a bootstrap over one seed has zero width; do not print it as a CI
        s += " (n=1, no seed CI)"
    elif d["lo"] is not None and d["hi"] is not None:
        s += f" [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    if len(d.get("per_seed", {})) > 1:  # CLAUDE.md §7: always print per-seed values
        s += (
            " (" + ", ".join(f"s{k} {100 * v:.1f}%" for k, v in sorted(d["per_seed"].items())) + ")"
        )
    return s


def _auc(d: dict | None) -> str:
    if d is None:
        return TBD
    s = fmt_num(d["value"])
    if d["lo"] is not None and d["hi"] is not None:
        s += f" [{fmt_num(d['lo'])}, {fmt_num(d['hi'])}]"
    return s


def headline_block(nums: dict[str, dict | None]) -> str:
    """Neutral wording: the sentence does not claim a direction the numbers might not show."""
    w = nums.get("W")
    wname = f" ({w['monitor']})" if w else ""
    ns = sorted({v["n_seeds"] for v in (nums.get("X"), nums.get("Y")) if v and "n_seeds" in v})
    n_txt = "n=" + "/".join(map(str, ns)) + " seeds" if ns else "n=TBD seeds"
    return (
        f"- Held-out-dynamics (sim-to-sim) success on FetchPush, mean over {n_txt} "
        f"[95% bootstrap CI]: NoDR **{_pct(nums.get('X'))}**, DR **{_pct(nums.get('Y'))}**.\n"
        f"- Failure prediction from the first 10 of 50 steps (push_dr, test split): ASM AUROC "
        f"**{_auc(nums.get('Z'))}** vs best baseline{wname} **{_auc(w)}**."
    )


def build_tables(root: Path) -> dict[str, str]:
    t: dict[str, str] = {
        "c1_controller": table_c1_controller(root),
        "c1_shortcut": _generic(root, "c1/c1_shortcut_summary.csv", RUNTIME_COLS),
        "c1_push_dr": _generic(root, "c1/c1_push_dr_summary.csv", RUNTIME_COLS),
        "c1_push_dr_deletion": _generic(root, "c1/c1_push_dr_deletion.csv"),
        "c2_runtime": table_c2_runtime(root),
    }
    for exp in C3_EXPS:
        t[f"c3_{exp}"] = table_c3(root, exp, per_factor=False)
        t[f"c3_{exp}_per_factor"] = table_c3(root, exp, per_factor=True)
    for exp in REACH_CONDS:
        t[f"c3_{exp}"] = table_c3_metrics(root, exp)
        t[f"c3_{exp}_per_factor"] = table_c3(root, exp, per_factor=True)
    t["robustness_reach"] = table_robustness_reach(root)
    t["c4_reach_reliance"] = table_c4_reach_reliance(root, "base")
    t["c4_reach_reliance_slots"] = table_c4_reach_reliance(root, "slot")
    t["c4_reach_temporal"] = table_c4_reach_temporal(root)
    t["c4_reach_probes"] = table_c4_reach_probes(root)
    t["c1_shortcut_her"] = _generic(root, "c1/c1_shortcut_her_summary.csv", RUNTIME_COLS)
    t["c4_reliance"] = _generic(root, "c4/reliance.csv")
    t["c4_temporal"] = _generic(root, "c4/temporal.csv")
    t["c4_probes"] = _generic(root, "c4/probes.csv")
    t["robustness"] = _generic(root, "robustness/robustness.csv")
    t["pilot_fps"] = _generic(root, "pilot/fps.csv")
    return t


# ---------------------------------------------------------------------------- doc markers

_MARKER = re.compile(
    r"(<!-- BEGIN:(?P<name>[\w.-]+) -->)(?P<body>.*?)(<!-- END:(?P=name) -->)", re.S
)


def fill_markers(text: str, blocks: dict[str, str]) -> str:
    """Replace the body of every known marker pair; idempotent."""

    def repl(m: re.Match) -> str:
        name = m.group("name")
        if name not in blocks:
            return m.group(0)
        block = blocks[name].strip("\n")
        inline = "\n" not in m.group("body") and "\n" not in block
        body = block if inline else f"\n{block}\n"
        return f"{m.group(1)}{body}{m.group(4)}"

    return _MARKER.sub(repl, text)


def fill_doc(path: Path, blocks: dict[str, str]) -> bool:
    """Returns True if the file changed."""
    old = path.read_text(encoding="utf-8")
    new = fill_markers(old, blocks)
    if new != old:
        path.write_text(new, encoding="utf-8", newline="\n")
    return new != old


# ---------------------------------------------------------------------------- main


def make_all(root: Path, out: Path, docs: Iterable[Path], log: Callable[[str], None] = print):
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(root)
    for name, md in tables.items():
        (out / f"{name}.md").write_text(md + "\n", encoding="utf-8", newline="\n")
        log(f"{'TBD ' if TBD in md and md.startswith('| status') else 'ok  '} {name}")
    nums = headline_numbers(root)
    (out / "numbers.json").write_text(json.dumps(nums, indent=2) + "\n", encoding="utf-8")
    blocks = dict(tables)
    blocks["headline"] = headline_block(nums)
    blocks["reach_summary"] = reach_summary_block(root)
    blocks["push_summary"] = reach_summary_block(root, track="push")
    blocks["num_X"], blocks["num_Y"] = _pct(nums["X"]), _pct(nums["Y"])
    blocks["num_Z"], blocks["num_W"] = _auc(nums["Z"]), _auc(nums["W"])
    for doc in docs:
        if doc.exists():
            changed = fill_doc(doc, blocks)
            log(f"{'filled' if changed else 'unchanged'} {doc}")
        else:
            log(f"skip (missing) {doc}")
    return tables, nums


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="default: {results}/tables")
    p.add_argument("--docs", nargs="*", default=["README.md", "report/report.md"])
    args = p.parse_args(argv)
    root = Path(args.results)
    out = Path(args.out) if args.out else root / "tables"
    make_all(root, out, [Path(d) for d in args.docs])


if __name__ == "__main__":
    sys.exit(main())
