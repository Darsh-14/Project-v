"""Held-out sweep: success per (policy, cell) and per-step logs (mvp.md §4.3, docs/interfaces.md).

    python -m projectv.eval.heldout_sweep --exp configs/exp/push_dr.yaml --seeds 0 1 2 \
        --episodes 100 [--cells configs/dr/heldout.yaml] [--root results] \
        [--steplog auto|all|none] [--nominal-episodes 200] [--only-cells 0,3]

Outputs
- `{root}/eval/{exp}_s{seed}.csv`: one row per episode (`cell_id, cell_name, factor, episode,
  ep_seed, success`), appended cell by cell.
- `{root}/steplogs/{exp}_s{seed}/cell{id:02d}.npz` + `meta.json` for the logged cells.
- `{root}/eval/run_meta.json`.

Cell 0 (nominal) always gets `max(--episodes, --nominal-episodes)` episodes: it is the calib pool
for C3. Resumable: a (seed, cell) is skipped when its CSV rows are complete and, if it must be
logged, its step log exists. The step log is written before the CSV rows, so complete rows imply
a complete log.

The held-out hash is asserted against `{root}/heldout_hash.txt` (or `--hash-file`). `--dry-cells`
skips that check; it exists for tests and calibration only and marks every output as dry.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from projectv.eval.cells import Cell, assert_frozen, load_cells
from projectv.eval.rollout import load_model, resolve_checkpoint, run_episodes
from projectv.eval.steplog import save_steplog, steplog_path
from projectv.utils.config import config_hash, file_hash, load_exp, text_hash
from projectv.utils.io import git_sha, write_json, write_run_meta
from projectv.utils.seeding import eval_episode_seed

CSV_COLUMNS = ("cell_id", "cell_name", "factor", "episode", "ep_seed", "success")
LOG_ALL_EXPS = ("push_dr", "push_drhist", "push_nodr")  # C3 runs on all three


def steplog_cell_ids(exp_name: str, cells: Sequence[Cell], mode: str) -> set[int]:
    """Cells whose per-step logs are saved. `auto`: all cells for the Push conditions (C3/C4),
    only the nominal calib pool (cell 0) for every other experiment."""
    if mode == "none":
        return set()
    if mode == "all" or exp_name in LOG_ALL_EXPS:
        return {c.id for c in cells}
    assert mode == "auto", mode
    return {0}


def n_episodes_for(cell: Cell, episodes: int, nominal_episodes: int) -> int:
    return max(episodes, nominal_episodes) if cell.id == 0 else episodes


def csv_path(root: str | Path, exp: str, seed: int) -> Path:
    return Path(root) / "eval" / f"{exp}_s{seed}.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: Iterable[dict], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = mode == "w" or not path.exists()
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def completed_cells(
    root: str | Path,
    exp: str,
    seed: int,
    cells: Sequence[Cell],
    n_eps: dict[int, int],
    log_ids: set[int],
) -> set[int]:
    """Cells whose CSV rows are complete and whose step log (if required) exists. Rows of
    incomplete cells are dropped from the CSV so the cell can be rerun cleanly."""
    path = csv_path(root, exp, seed)
    rows = read_rows(path)
    counts: dict[int, int] = {}
    for r in rows:
        counts[int(r["cell_id"])] = counts.get(int(r["cell_id"]), 0) + 1
    done = set()
    for c in cells:
        if counts.get(c.id, 0) != n_eps[c.id]:
            continue
        if c.id in log_ids and not steplog_path(root, exp, seed, c.id).exists():
            continue
        done.add(c.id)
    stale = {cid for cid in counts if cid not in done and cid in n_eps}
    if stale:
        tmp = path.with_suffix(".csv.tmp")
        _write_rows(tmp, [r for r in rows if int(r["cell_id"]) not in stale], "w")
        tmp.replace(path)
    return done


def _check_steplog_meta(meta_path: Path, meta: dict) -> None:
    """Refuse to mix step logs from different held-out grids or checkpoints in one directory."""
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    # checkpoint identity by content hash (paths differ between Kaggle and local machines)
    keys = (
        ("heldout_hash", "checkpoint_sha256")
        if "checkpoint_sha256" in old
        else ("heldout_hash", "checkpoint")
    )
    for k in keys:
        if old.get(k) != meta[k]:
            raise RuntimeError(
                f"{meta_path} was written with {k}={old.get(k)!r}, now {meta[k]!r}; "
                "move the old results away before rerunning"
            )


def sweep_seed(
    exp: dict,
    seed: int,
    cells: Sequence[Cell],
    root: str | Path,
    episodes: int,
    nominal_episodes: int,
    steplog_mode: str,
    heldout_hash: str,
    model=None,
) -> dict[int, float]:
    """Evaluate one training seed on every cell. Returns success rate per newly evaluated cell."""
    name = exp["name"]
    ckpt = resolve_checkpoint(name, seed, root)
    log_ids = steplog_cell_ids(name, cells, steplog_mode)
    n_eps = {c.id: n_episodes_for(c, episodes, nominal_episodes) for c in cells}
    done = completed_cells(root, name, seed, cells, n_eps, log_ids)
    todo = [c for c in cells if c.id not in done]
    for c in cells:
        if c.id in done:
            print(f"[sweep] {name} s{seed} cell {c.id:02d} done, skipping", flush=True)
    meta = {
        "exp": name,
        "seed": seed,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": file_hash(ckpt),
        "config_hash": config_hash(exp),
        "heldout_hash": heldout_hash,
        "git_sha": git_sha(),
    }
    log_dir = steplog_path(root, name, seed, 0).parent
    if log_ids:
        # checked even when every cell is done: stale logs from another checkpoint must not pass
        _check_steplog_meta(log_dir / "meta.json", meta)
    if not todo:
        return {}
    if log_ids:
        write_json(log_dir / "meta.json", meta)

    model = model if model is not None else load_model(ckpt)
    rates: dict[int, float] = {}
    for c in todo:
        t0 = time.time()
        seeds = [eval_episode_seed(seed, c.id, ep) for ep in range(n_eps[c.id])]
        log = c.id in log_ids
        rec = run_episodes(model, exp, c.dr_spec(), seeds, log=log)
        if log:
            save_steplog(steplog_path(root, name, seed, c.id), rec, c.id)
        rows = [
            dict(
                cell_id=c.id,
                cell_name=c.name,
                factor=c.factor,
                episode=ep,
                ep_seed=int(sd),
                success=int(s),
            )
            for ep, (sd, s) in enumerate(zip(rec["ep_seed"], rec["success"], strict=True))
        ]
        _write_rows(csv_path(root, name, seed), rows, "a")
        rates[c.id] = float(np.mean(rec["success"]))
        print(
            f"[sweep] {name} s{seed} cell {c.id:02d} {c.name}: success "
            f"{rates[c.id]:.3f} over {len(seeds)} eps ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seed", type=int, default=None, help="single seed (overrides --seeds)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--nominal-episodes", type=int, default=200)
    p.add_argument("--cells", default="configs/dr/heldout.yaml")
    p.add_argument("--hash-file", default=None, help="default: {root}/heldout_hash.txt")
    p.add_argument(
        "--dry-cells",
        default=None,
        help="cells YAML loaded WITHOUT the freeze/hash check (tests and calibration only)",
    )
    p.add_argument("--root", default="results", help="results root (checkpoints and outputs)")
    p.add_argument("--out", default=None, help="alias of --root")
    p.add_argument("--steplog", choices=["auto", "all", "none"], default="auto")
    p.add_argument("--only-cells", default=None, help="comma-separated cell ids, e.g. 0,3")
    p.add_argument("--no-wandb", action="store_true", help="accepted for CLI uniformity")
    return p.parse_args(argv)


def load_sweep_cells(args: argparse.Namespace, root: Path) -> tuple[list[Cell], str]:
    if args.dry_cells:
        print(
            f"[sweep] WARNING: --dry-cells {args.dry_cells}: held-out hash NOT checked; "
            "outputs are not valid C3/C4/robustness results",
            flush=True,
        )
        return load_cells(args.dry_cells, check_hash=False), "dry:" + text_hash(args.dry_cells)
    hash_file = Path(args.hash_file) if args.hash_file else root / "heldout_hash.txt"
    assert hash_file.exists(), f"{hash_file} missing: freeze the grid first (calibrate_heldout.py)"
    assert_frozen(args.cells, hash_file)
    return load_cells(args.cells, check_hash=False), text_hash(args.cells)


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    root = Path(args.out or args.root)
    exp = load_exp(args.exp)
    cells, heldout_hash = load_sweep_cells(args, root)
    if args.only_cells:
        keep = {int(x) for x in args.only_cells.split(",")}
        unknown = keep - {c.id for c in cells}
        assert not unknown, f"unknown cell ids {sorted(unknown)}"
        cells = [c for c in cells if c.id in keep]
    seeds = [args.seed] if args.seed is not None else args.seeds

    summary: dict[str, dict[int, float]] = {}
    for seed in seeds:
        summary[str(seed)] = sweep_seed(
            exp,
            seed,
            cells,
            root,
            args.episodes,
            args.nominal_episodes,
            args.steplog,
            heldout_hash,
        )
    write_run_meta(
        root / "eval",
        exp,
        ["heldout_sweep", *argv],
        t_start,
        exp=exp["name"],
        seeds=seeds,
        heldout_hash=heldout_hash,
        cells=[c.id for c in cells],
        new_success=summary,
    )


if __name__ == "__main__":
    main()
