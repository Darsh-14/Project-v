import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from projectv.eval import heldout_sweep
from projectv.eval.cells import Cell, load_cells, write_cells
from projectv.eval.steplog import load_steplog, steplog_path
from projectv.utils.config import text_hash
from projectv.utils.seeding import eval_episode_seed
from projectv.xai.groups import flat_dim

REPO = Path(__file__).resolve().parents[1]


def _load_calib():
    spec = importlib.util.spec_from_file_location(
        "calibrate_heldout", REPO / "scripts" / "calibrate_heldout.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


calib = _load_calib()


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def sweep_root(tiny_model, tmp_path):
    root = tmp_path / "results"
    (root / "ckpt").mkdir(parents=True)
    tiny_model("push_dr").save(root / "ckpt" / "push_dr_s0_final.zip")
    cells = [Cell(0, "nominal", "nominal", {}), Cell(1, "obj_mass=3.0", "mass", {"obj_mass": 3.0})]
    cells_yaml = tmp_path / "cells.yaml"
    write_cells(cells, cells_yaml, frozen=True, source_policy="test")
    return root, cells_yaml


def _sweep_args(root, cells_yaml, *extra):
    return [
        "--exp",
        "configs/exp/push_dr.yaml",
        "--seeds",
        "0",
        "--episodes",
        "2",
        "--nominal-episodes",
        "3",
        "--root",
        str(root),
        "--dry-cells",
        str(cells_yaml),
        *extra,
    ]


# --- sweep ----------------------------------------------------------------------------------


def test_sweep_csv_and_steplog_contract(sweep_root):
    root, cells_yaml = sweep_root
    heldout_sweep.main(_sweep_args(root, cells_yaml))
    rows = _rows(root / "eval" / "push_dr_s0.csv")
    assert tuple(rows[0]) == heldout_sweep.CSV_COLUMNS
    by_cell = {cid: [r for r in rows if int(r["cell_id"]) == cid] for cid in (0, 1)}
    assert len(by_cell[0]) == 3 and len(by_cell[1]) == 2  # nominal calib pool is larger
    assert [int(r["ep_seed"]) for r in by_cell[1]] == [eval_episode_seed(0, 1, e) for e in (0, 1)]
    assert by_cell[1][0]["factor"] == "mass" and by_cell[1][0]["cell_name"] == "obj_mass=3.0"

    for cid, n in ((0, 3), (1, 2)):  # push_dr: every cell logged under --steplog auto
        log = load_steplog(steplog_path(root, "push_dr", 0, cid))
        assert int(log["cell_id"]) == cid
        assert log["obs_flat"].shape == (n, 50, flat_dim("push"))
        assert log["obs_flat"].dtype == np.float32 and log["is_success"].dtype == bool
        assert log["success"].shape == (n,) and log["ep_seed"].dtype == np.int64
        succ = [bool(int(r["success"])) for r in by_cell[cid]]
        np.testing.assert_array_equal(log["success"], succ)
    names = list(log["dr_names"])
    assert log["dr_values"][0, names.index("obj_mass")] == np.float32(3.0)
    assert (root / "steplogs" / "push_dr_s0" / "meta.json").exists()
    assert len(list((root / "eval").glob("run_meta_push_dr_s*.json"))) == 1


def test_sweep_resume_skips_completed_cells(sweep_root, monkeypatch):
    root, cells_yaml = sweep_root
    heldout_sweep.main(_sweep_args(root, cells_yaml, "--only-cells", "0"))
    calls = []
    real = heldout_sweep.run_episodes

    def counting(model, exp, dr_spec, seeds, **kw):
        calls.append(list(seeds))
        return real(model, exp, dr_spec, seeds, **kw)

    monkeypatch.setattr(heldout_sweep, "run_episodes", counting)
    heldout_sweep.main(_sweep_args(root, cells_yaml))
    assert calls == [[eval_episode_seed(0, 1, e) for e in (0, 1)]]  # only cell 1 ran
    heldout_sweep.main(_sweep_args(root, cells_yaml))
    assert len(calls) == 1  # everything done
    rows = _rows(root / "eval" / "push_dr_s0.csv")
    assert len(rows) == 5

    # a missing step log makes the cell incomplete: its rows are dropped and it is rerun
    steplog_path(root, "push_dr", 0, 1).unlink()
    heldout_sweep.main(_sweep_args(root, cells_yaml))
    assert len(calls) == 2
    assert len(_rows(root / "eval" / "push_dr_s0.csv")) == 5


def test_steplog_policy():
    cells = [Cell(i, f"c{i}", "mass", {}) for i in range(4)]
    assert heldout_sweep.steplog_cell_ids("push_dr", cells, "auto") == {0, 1, 2, 3}
    assert heldout_sweep.steplog_cell_ids("push_drhist", cells, "auto") == {0, 1, 2, 3}
    # C3 runs on push_nodr too (review bug 1): all Push conditions log every cell
    assert heldout_sweep.steplog_cell_ids("push_nodr", cells, "auto") == {0, 1, 2, 3}
    assert heldout_sweep.steplog_cell_ids("reach_dr", cells, "auto") == {0}
    assert heldout_sweep.steplog_cell_ids("reach_dr", cells, "all") == {0, 1, 2, 3}
    assert heldout_sweep.steplog_cell_ids("push_dr", cells, "none") == set()


def test_sweep_asserts_heldout_hash(sweep_root):
    root, cells_yaml = sweep_root
    args = ["--exp", "configs/exp/push_dr.yaml", "--seeds", "0", "--episodes", "1"]
    args += ["--nominal-episodes", "1", "--root", str(root), "--cells", str(cells_yaml)]
    with pytest.raises(AssertionError):  # no hash file
        heldout_sweep.main(args)
    (root / "heldout_hash.txt").write_text("0" * 64)
    with pytest.raises(AssertionError):  # wrong hash
        heldout_sweep.main(args)
    (root / "heldout_hash.txt").write_text(text_hash(cells_yaml))
    heldout_sweep.main(args + ["--only-cells", "1", "--steplog", "none"])
    assert len(_rows(root / "eval" / "push_dr_s0.csv")) == 1


# --- calibration helper -----------------------------------------------------------------------


def _fake_success(boundary, scale):
    """Monotone: 1.0 at the train boundary, decreasing with distance."""

    def f(v):
        d = abs(math.log(v / boundary)) if scale == "log" else abs(v - boundary)
        return float(np.exp(-3.0 * d)) if scale == "log" else float(np.exp(-d / 0.004))

    return f


@pytest.mark.parametrize(
    "boundary,candidate,scale",
    [(0.5, 0.05, "log"), (1.5, 20.0, "log"), (0.005, 0.03, "linear")],
)
def test_bisection_converges_to_band(boundary, candidate, scale):
    f = _fake_success(boundary, scale)
    assert f(candidate) < 0.2  # candidate too hard: the search must move toward the boundary
    res = calib.calibrate_value(f, boundary, candidate, scale, max_evals=6)
    assert res.in_band and 0.2 <= res.success <= 0.8
    assert len(res.history) <= 6 and res.history[0][0] == candidate
    lo, hi = sorted((boundary, candidate))
    assert lo < res.value <= hi  # stays outside the train range, between boundary and candidate


def test_bisection_integer_latency():
    succ = {2: 0.9, 3: 0.5, 4: 0.1, 5: 0.0, 6: 0.0}
    res = calib.calibrate_value(lambda v: succ[v], 1, 6, "int")
    assert res.in_band and res.value == 3
    assert all(isinstance(v, int) for v, _ in res.history)


def test_bisection_flags_unreachable_band():
    res = calib.calibrate_value(lambda v: 0.0 if v > 2 else 1.0, 1, 3, "int")
    assert not res.in_band and res.value in (2, 3)  # closest kept, not dropped
    res = calib.calibrate_value(lambda v: 0.95, 0.5, 0.2, "log")  # too easy at the candidate
    assert not res.in_band and res.value == 0.2 and len(res.history) == 1
    calls = []
    res = calib.calibrate_value(lambda v: calls.append(v) or 0.0, 0.5, 0.01, "log", max_evals=6)
    assert len(calls) == 6 and not res.in_band


def test_sample_combined_outside_train_range():
    ranges = {"obj_mass": (0.5, 2.0), "latency": (0.0, 1.0), "obs_noise": (0.0, 0.005)}
    calibrated = {"obj_mass": [0.3, 3.0, 4.0], "latency": [2, 3], "obs_noise": [0.01]}
    a = calib.sample_combined(calibrated, ranges, 20, 20261024)
    assert a == calib.sample_combined(calibrated, ranges, 20, 20261024)
    for p in a:
        assert set(p) == set(calibrated)
        assert p["obj_mass"] == 0.3 or 3.0 <= p["obj_mass"] <= 4.0
        assert p["latency"] in (2, 3) and p["obs_noise"] == 0.01


# --- calibration script -----------------------------------------------------------------------


def _calib_args(tmp_path, *extra):
    pol = tmp_path / "pol.zip"
    cand = tmp_path / "cand.yaml"
    cand.write_text(
        yaml.safe_dump(
            {
                "one_factor": {"obj_mass": [3.0], "latency": [3]},
                "combined": {"n_cells": 2, "seed": 1},
            },
            sort_keys=False,
        ),
    )
    return [
        "--policy",
        str(pol),
        "--exp",
        "configs/exp/push_dr.yaml",
        "--candidates",
        str(cand),
        "--episodes",
        "1",
        "--max-evals",
        "2",
        "--out",
        str(tmp_path / "heldout.yaml"),
        "--hash-out",
        str(tmp_path / "heldout_hash.txt"),
        "--log-out",
        str(tmp_path / "calibration" / "calibration_log.csv"),
        *extra,
    ]


def test_calibration_writes_frozen_yaml_and_hash(tiny_model, tmp_path):
    tiny_model("push_dr").save(tmp_path / "pol.zip")
    calib.main(_calib_args(tmp_path))
    out, hash_out = tmp_path / "heldout.yaml", tmp_path / "heldout_hash.txt"
    doc = yaml.safe_load(out.read_text())
    assert doc["frozen"] is True and doc["source_policy"].endswith("pol.zip")
    assert hash_out.read_text().strip() == text_hash(out)
    cells = load_cells(out, check_hash=False)
    assert [c.id for c in cells] == [0, 1, 2, 3, 4]
    assert [c.factor for c in cells] == ["nominal", "mass", "latency", "combined", "combined"]
    assert 2.0 < cells[1].params["obj_mass"] <= 3.0 and cells[2].params["latency"] in (2, 3)
    assert set(cells[3].params) == {"obj_mass", "latency"}
    log = _rows(tmp_path / "calibration" / "calibration_log.csv")
    assert {r["phase"] for r in log} == {"nominal", "one_factor", "combined"}
    one = [r for r in log if r["phase"] == "one_factor"]
    for cand in ("3.0", "3"):
        sel = [r for r in one if r["candidate"] == cand and r["selected"] == "True"]
        assert len(sel) >= 1
        assert sel[0]["in_band"] == "True" or sel[0]["flag"] == "no_value_in_band_kept_closest"

    # frozen: refuse to recalibrate, unless forced
    with pytest.raises(SystemExit, match="REFUSING"):
        calib.main(_calib_args(tmp_path))
    calib.main(_calib_args(tmp_path, "--force-refreeze"))
    assert hash_out.read_text().strip() == text_hash(out)


def test_steplog_meta_rejects_other_checkpoint(tmp_path):
    import json

    import pytest as _pt

    meta_path = tmp_path / "meta.json"
    old = {"heldout_hash": "h", "checkpoint": "/kaggle/a.zip", "checkpoint_sha256": "aaa"}
    meta_path.write_text(json.dumps(old))
    # same content hash, different machine path: accepted
    heldout_sweep._check_steplog_meta(meta_path, {**old, "checkpoint": "D:/b.zip"})
    with _pt.raises(RuntimeError):
        heldout_sweep._check_steplog_meta(meta_path, {**old, "checkpoint_sha256": "bbb"})


def test_grid_hash_ignores_line_endings(tmp_path):
    from projectv.eval.cells import Cell, write_cells
    from projectv.utils.config import text_hash

    p = tmp_path / "grid.yaml"
    write_cells([Cell(0, "nominal", "nominal", {})], p, frozen=True)
    assert b"\r" not in p.read_bytes()  # always written with LF
    crlf = tmp_path / "grid_crlf.yaml"
    crlf.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert text_hash(p) == text_hash(crlf)  # Windows checkout == Linux clone
