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
        else:
            out[c] = ["" if pd.isna(v) else str(v) for v in df[c]]
    return pd.DataFrame(out, index=df.index)


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
    if d["lo"] is not None and d["hi"] is not None:
        s += f" [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
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
