"""Calibrate and freeze the held-out grid (mvp.md §4.2, docs/interfaces.md). Run ONCE.

    python scripts/calibrate_heldout.py --policy results/ckpt/push_dr_s0/step_N.zip \
        --exp configs/exp/push_dr.yaml --candidates configs/dr/heldout_candidates.yaml \
        --episodes 50 --out configs/dr/heldout.yaml

One-factor cells: for each candidate value, evaluate the source policy (others at nominal). If
success is outside [0.2, 0.8], move the value along the factor axis between the train-range
boundary and the candidate (bisection: log scale for multipliers, linear for obs noise, integer
steps for latency), at most `--max-evals` evaluations per cell. If no value reaches the band, the
closest one is kept and flagged in the log; no cell is dropped.

Combined cells: `n_cells` draws (seed from the candidates file, 20261024); each factor is sampled
from the calibrated held-out values outside the train range (see `sample_combined`). They are
evaluated and logged but never re-tuned.

Every evaluation goes to `results/calibration/calibration_log.csv`. Writes `--out` with
`frozen: true` and `--hash-out` (default results/heldout_hash.txt). Refuses to run when the hash
file exists unless `--force-refreeze`.

Calibration episodes use reset seeds `CALIB_SEED_BASE + ep`, disjoint from the sweep's
`eval_episode_seed` range, and the same episodes for every evaluated value (common random numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from projectv.eval.cells import NOMINAL, Cell, write_cells
from projectv.eval.rollout import load_model, run_episodes
from projectv.utils.config import file_hash, load_exp, load_yaml
from projectv.utils.io import write_run_meta

BAND = (0.2, 0.8)
CALIB_SEED_BASE = 5_000_000  # > any eval_episode_seed (10_000 * seed + 1_000 * cell + ep)
FACTOR_GROUP = {
    "obj_friction": "friction",
    "table_friction": "friction",
    "obj_mass": "mass",
    "gain": "gain",
    "obs_noise": "noise",
    "latency": "latency",
}
FACTOR_SCALE = {
    "obj_friction": "log",
    "table_friction": "log",
    "obj_mass": "log",
    "gain": "log",
    "obs_noise": "linear",
    "latency": "int",
}
LOG_COLUMNS = (
    "phase",
    "cell",
    "factor",
    "candidate",
    "eval_idx",
    "value",
    "params",
    "success",
    "n_episodes",
    "in_band",
    "selected",
    "flag",
)


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested with a fake evaluator)


def round_value(x: float, scale: str) -> float:
    """Readable values: integers for latency, 3 significant digits otherwise."""
    if scale == "int":
        return int(round(x))
    if x == 0:
        return 0.0
    return float(f"{x:.3g}")


def midpoint(near: float, far: float, scale: str) -> float | None:
    """Point strictly between `near` (train boundary side) and `far`; None if none exists."""
    if scale == "int":
        lo, hi = sorted((int(near), int(far)))
        if hi - lo <= 1:
            return None
        return int(math.floor((near + far) / 2)) if far > near else int(math.ceil((near + far) / 2))
    if scale == "log":
        assert near > 0 and far > 0, (near, far)
        m = math.sqrt(near * far)
    else:
        m = 0.5 * (near + far)
    m = round_value(m, scale)
    lo, hi = sorted((near, far))
    return m if lo < m < hi else None


def band_distance(s: float, band: tuple[float, float] = BAND) -> float:
    return max(band[0] - s, 0.0, s - band[1])


@dataclass
class CalibResult:
    value: float
    success: float
    in_band: bool
    history: list[tuple[float, float]] = field(default_factory=list)


def calibrate_value(
    evaluate: Callable[[float], float],
    boundary: float,
    candidate: float,
    scale: str,
    band: tuple[float, float] = BAND,
    max_evals: int = 6,
) -> CalibResult:
    """Search [boundary, candidate] for a value with success in `band`.

    Assumes success decreases with distance from the train range. The candidate is evaluated
    first; too hard (< band) moves toward the boundary, too easy (> band) moves toward the
    candidate. The boundary itself (in the train range) is never returned. If the band is not
    reached within `max_evals`, the evaluated value closest to the band is returned with
    `in_band=False`."""
    near, far = boundary, candidate
    x: float | None = round_value(candidate, scale)
    history: list[tuple[float, float]] = []
    while x is not None and len(history) < max_evals:
        s = float(evaluate(x))
        history.append((x, s))
        if band[0] <= s <= band[1]:
            return CalibResult(x, s, True, history)
        if s < band[0]:
            far = x
        else:
            near = x
        x = midpoint(near, far, scale)
        if x is not None and any(x == v for v, _ in history):
            x = None
    best = min(history, key=lambda vs: (band_distance(vs[1], band), abs(vs[1] - 0.5)))
    return CalibResult(best[0], best[1], False, history)


def train_range(spec: dict) -> tuple[float, float]:
    if spec["type"] == "uniform":
        return float(spec["low"]), float(spec["high"])
    if spec["type"] == "choice":
        return float(min(spec["values"])), float(max(spec["values"]))
    return float(spec["value"]), float(spec["value"])


def boundary_for(value: float, rng: tuple[float, float]) -> float:
    lo, hi = rng
    if value < lo:
        return lo
    if value > hi:
        return hi
    raise ValueError(f"held-out value {value} lies inside the train range {rng}")


def sample_combined(
    calibrated: dict[str, list[float]],
    ranges: dict[str, tuple[float, float]],
    n_cells: int,
    seed: int,
) -> list[dict[str, float]]:
    """Each combined cell sets every factor outside its train range. Per factor, the calibrated
    held-out values on each side of the train range span an interval [min, max] (a point if the
    side has one value); a side is picked uniformly among the sides that have values, then a
    value uniformly within it (log-uniform for multipliers, integers for latency)."""
    rs = np.random.default_rng(seed)
    out = []
    for _ in range(n_cells):
        params: dict[str, float] = {}
        for f in sorted(calibrated):
            lo, hi = ranges[f]
            sides = [
                [v for v in calibrated[f] if v < lo],
                [v for v in calibrated[f] if v > hi],
            ]
            sides = [s for s in sides if s]
            side = sides[int(rs.integers(len(sides)))]
            a, b = min(side), max(side)
            scale = FACTOR_SCALE[f]
            if scale == "int":
                v: float = int(rs.integers(int(a), int(b) + 1))
            elif scale == "log":
                v = round_value(math.exp(rs.uniform(math.log(a), math.log(b))), scale)
            else:
                v = round_value(rs.uniform(a, b), scale)
            params[f] = v
        out.append(params)
    return out


def cell_name(factor: str, value: float) -> str:
    return f"{factor}={value}"


# ---------------------------------------------------------------------------------------------
# Script


class LogWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()

    def write(self, **row) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(
                {k: row.get(k, "") for k in LOG_COLUMNS}
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate and freeze the held-out grid")
    p.add_argument("--policy", required=True, help="source push_dr checkpoint (.zip)")
    p.add_argument("--exp", default="configs/exp/push_dr.yaml")
    p.add_argument("--candidates", default="configs/dr/heldout_candidates.yaml")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-evals", type=int, default=6)
    p.add_argument("--band", type=float, nargs=2, default=list(BAND))
    p.add_argument("--out", default="configs/dr/heldout.yaml")
    p.add_argument("--hash-out", default="results/heldout_hash.txt")
    p.add_argument("--log-out", default="results/calibration/calibration_log.csv")
    p.add_argument("--force-refreeze", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="global seed (rollouts are seeded per ep)")
    p.add_argument("--no-wandb", action="store_true", help="accepted for CLI uniformity")
    return p.parse_args(argv)


def refuse_or_warn(hash_out: Path, force: bool) -> None:
    if not hash_out.exists():
        return
    if not force:
        raise SystemExit(
            f"REFUSING: {hash_out} exists, the held-out grid is frozen (CLAUDE.md §2.4). "
            "Pass --force-refreeze only if the grid really must change."
        )
    bar = "!" * 88
    print(
        f"{bar}\n!! --force-refreeze: overwriting the frozen held-out grid.\n"
        "!! EVERY downstream result (held-out sweeps, step logs, C3, C4, robustness tables) must\n"
        "!! be rerun, and the refreeze must be noted in CHANGELOG.md.\n"
        f"{bar}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    band = (float(args.band[0]), float(args.band[1]))
    out, hash_out = Path(args.out), Path(args.hash_out)
    refuse_or_warn(hash_out, args.force_refreeze)

    exp = load_exp(args.exp)
    cand = load_yaml(args.candidates)
    ranges = {f: train_range(s) for f, s in exp["dr_cfg"].items()}
    model = load_model(args.policy)
    ep_seeds = [CALIB_SEED_BASE + ep for ep in range(args.episodes)]
    log = LogWriter(Path(args.log_out))

    def evaluate(params: dict[str, float]) -> float:
        rec = run_episodes(model, exp, Cell(-1, "calib", "calib", params).dr_spec(), ep_seeds)
        return float(np.mean(rec["success"]))

    nominal_success = evaluate({})
    log.write(
        phase="nominal",
        cell="nominal",
        factor="nominal",
        eval_idx=0,
        params="{}",
        success=nominal_success,
        n_episodes=args.episodes,
        in_band=band[0] <= nominal_success <= band[1],
        selected=True,
    )
    print(f"[calib] nominal success {nominal_success:.3f}", flush=True)

    cells: list[Cell] = [NOMINAL]
    calibrated: dict[str, list[float]] = {}
    summary = []
    for factor, values in cand["one_factor"].items():
        assert factor in FACTOR_SCALE, f"unknown factor {factor}"
        scale = FACTOR_SCALE[factor]
        for candidate in values:
            boundary = boundary_for(float(candidate), ranges[factor])
            res = calibrate_value(
                lambda v, f=factor: evaluate({f: v}),
                boundary,
                candidate,
                scale,
                band,
                args.max_evals,
            )
            flag = "" if res.in_band else "no_value_in_band_kept_closest"
            for i, (v, s) in enumerate(res.history):
                log.write(
                    phase="one_factor",
                    cell=cell_name(factor, res.value),
                    factor=factor,
                    candidate=candidate,
                    eval_idx=i,
                    value=v,
                    params=json.dumps({factor: v}),
                    success=s,
                    n_episodes=args.episodes,
                    in_band=band[0] <= s <= band[1],
                    selected=v == res.value,
                    flag=flag if v == res.value else "",
                )
            name = cell_name(factor, res.value)
            if any(c.name == name for c in cells):
                name = f"{name}#{len(cells)}"  # two candidates converged to the same value
            cells.append(Cell(len(cells), name, FACTOR_GROUP[factor], {factor: res.value}))
            calibrated.setdefault(factor, []).append(res.value)
            summary.append(dict(cell=name, candidate=candidate, success=res.success, flag=flag))
            print(
                f"[calib] {factor} candidate {candidate} -> {res.value}: success "
                f"{res.success:.3f} after {len(res.history)} evals {flag}",
                flush=True,
            )

    comb = cand.get("combined", {})
    for i, params in enumerate(
        sample_combined(calibrated, ranges, int(comb.get("n_cells", 0)), int(comb["seed"]))
        if comb
        else []
    ):
        s = evaluate(params)
        name = f"combined_{i:02d}"
        log.write(
            phase="combined",
            cell=name,
            factor="combined",
            eval_idx=0,
            params=json.dumps(params),
            success=s,
            n_episodes=args.episodes,
            in_band=band[0] <= s <= band[1],
            selected=True,
            flag="" if band[0] <= s <= band[1] else "out_of_band_not_retuned",
        )
        cells.append(Cell(len(cells), name, "combined", params))
        print(f"[calib] {name} {params}: success {s:.3f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    write_cells(
        cells,
        out,
        frozen=True,
        source_policy=Path(args.policy).as_posix(),
        calibration=dict(
            episodes=args.episodes,
            band=list(band),
            max_evals=args.max_evals,
            candidates=Path(args.candidates).as_posix(),
            candidates_hash=file_hash(args.candidates),
            log=Path(args.log_out).as_posix(),
        ),
    )
    hash_out.parent.mkdir(parents=True, exist_ok=True)
    digest = file_hash(out)
    hash_out.write_text(digest + "\n", encoding="utf-8")
    write_run_meta(
        Path(args.log_out).parent,
        {"exp": exp, "candidates": cand},
        ["calibrate_heldout", *argv],
        t_start,
        heldout_hash=digest,
        nominal_success=nominal_success,
        one_factor=summary,
    )
    print(f"[calib] wrote {out} ({len(cells)} cells), hash {digest} -> {hash_out}", flush=True)


if __name__ == "__main__":
    main()
