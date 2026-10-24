"""C1 on learned policies and the planted shortcut (mvp.md §5). Small counts, tiny models."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from projectv.analysis.stats import two_level_bootstrap_ci
from projectv.control.analytic_controller import PDController
from projectv.eval.cells import NOMINAL
from projectv.eval.rollout import run_episodes
from projectv.utils.config import load_exp
from projectv.utils.seeding import eval_episode_seed
from projectv.xai import faithfulness as fa
from projectv.xai.closed_loop import (
    closed_loop_success,
    deletion_pauc,
    dist_runner,
    erasure_hook,
    kendall_tau,
    open_loop_importance,
    precision_at_k_sets,
    top_k,
)
from projectv.xai.groups import make_groups
from projectv.xai.policy_fn import LinearGaussian, controller_policy


# --- pure helpers -------------------------------------------------------------------------------
def test_kendall_tau_known_rankings():
    a = np.array([4.0, 3.0, 2.0, 1.0])
    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert kendall_tau(a, -a) == pytest.approx(-1.0)
    # one adjacent swap among 4 items: (5 concordant - 1 discordant) / 6
    assert kendall_tau(a, np.array([4.0, 3.0, 1.0, 2.0])) == pytest.approx(4 / 6)
    assert np.isnan(kendall_tau(a, np.zeros(4)))  # constant reference ranking is undefined


def test_top_k_and_precision():
    s = np.array([0.1, 0.5, 0.3, 0.5])
    assert list(top_k(s, 2)) == [1, 3]  # ties broken by group order
    assert precision_at_k_sets(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), 1) == 0.0
    assert precision_at_k_sets(np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 3.0]), 2) == 1.0


def test_push_summary_on_synthetic_seed_results():
    names = ["a", "b", "c", "d"]
    n_eps = 10

    def succ(p):  # episodes with mean success p
        return (np.arange(n_eps) < round(p * n_eps)).astype(np.float64)

    d = {
        "names": names,
        "units": names + ["cluster"],
        "rankers": ["m1", "open_loop"],
        "orders": np.array([["a", "b"], ["d", "c"]]),
        "shares": np.tile([[[0.4, 0.3, 0.2, 0.1]]], (5, 1, 1)),
        "runtime_ms": np.ones((5, 1)),
        "kl": np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (7, 1)),  # reversed ranking
        "nominal": succ(1.0),
        "unit": np.array([succ(p) for p in (0.2, 0.4, 0.6, 0.8, 0.0)]),
        "deletion": np.array([[succ(0.2), succ(0.0)], [succ(0.8), succ(0.6)]]),
    }
    summ, dele, imp = fa.push_summary({0: d, 1: d}, prec_k=1, n_boot=50)
    s = summ.set_index("method")
    assert s.loc["m1", "tau_closed"] == pytest.approx(1.0)
    assert s.loc["open_loop", "tau_closed"] == pytest.approx(-1.0)
    assert s.loc["m1", "precision_at_k"] == 1.0 and s.loc["open_loop", "precision_at_k"] == 0.0
    assert s.loc["m1", "pauc"] == pytest.approx(0.9)  # drops 0.8, 1.0
    assert s.loc["open_loop", "pauc"] == pytest.approx(0.3)  # drops 0.2, 0.4
    assert np.isnan(s.loc["open_loop", "runtime_ms"])
    i = imp.set_index("group")
    np.testing.assert_allclose(i["closed_loop"], [0.8, 0.6, 0.4, 0.2, 1.0])
    np.testing.assert_allclose(i["open_loop"], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert len(dele) == 2 * 2 * 3  # seeds x rankers x (k = 0..2)


def test_deletion_pauc_synthetic_curve():
    # drops 0.2, 0.4, 0.6, 0.8, 0.8 -> mean 0.56, normalised by nominal 0.8 -> 0.7
    assert deletion_pauc(0.8, np.array([0.6, 0.4, 0.2, 0.0, 0.0])) == pytest.approx(0.7)
    assert deletion_pauc(0.8, np.full(5, 0.8)) == pytest.approx(0.0)
    assert np.isnan(deletion_pauc(0.0, np.zeros(5)))


def test_two_level_bootstrap_ci():
    rng = np.random.default_rng(0)
    clusters = [{"x": rng.normal(m, 1.0, size=(200, 2))} for m in (0.0, 1.0, 2.0)]

    def stat(cs):
        return np.mean([c["x"].mean(0) for c in cs], axis=0)

    pt, lo, hi = two_level_bootstrap_ci(stat, clusters, n_boot=300)
    assert pt.shape == (2,) and np.all(lo <= pt) and np.all(pt <= hi)
    assert np.all(hi - lo > 0.5)  # seed-level spread dominates
    const = [{"x": np.ones((5, 1))}] * 3
    pt, lo, hi = two_level_bootstrap_ci(stat, const, n_boot=50)
    assert pt[0] == lo[0] == hi[0] == 1.0
    # axis selection: episodes on axis 1 stay paired across rows
    paired = [{"s": np.vstack([np.arange(10.0), np.arange(10.0)])}]
    pt, lo, hi = two_level_bootstrap_ci(
        lambda cs: cs[0]["s"][0].mean() - cs[0]["s"][1].mean(), paired, axes={"s": 1}, n_boot=50
    )
    assert pt == lo == hi == 0.0


def test_open_loop_importance_ignored_group_is_zero():
    groups = make_groups("push")
    d = 31
    W = np.zeros((4, d))
    W[0, groups["goal"]] = 1.0
    pol = LinearGaussian(torch.as_tensor(W))
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(100, d))
    kl = open_loop_importance(pol, pool[:20], groups, pool, np.random.default_rng(1), M=4)
    names = list(groups)
    assert kl.shape == (20, len(groups))
    assert np.all(kl[:, names.index("obj_pos")] == 0.0)
    assert np.all(kl[:, names.index("goal")] > 0.0)


# --- closed loop --------------------------------------------------------------------------------
def test_closed_loop_erasure_changes_policy_input_every_step(tiny_model):
    model = tiny_model("push_dr")
    exp = load_exp("configs/exp/push_dr.yaml")
    idx = make_groups("push")["obj_pos"]
    pool = np.random.default_rng(0).normal(size=(30, 31))
    inner = erasure_hook(idx, pool, np.random.default_rng(1))
    calls = []

    def hook(x, t, ep):
        y = inner(x, t, ep)
        calls.append((x.copy(), y.copy()))
        return y

    rec = run_episodes(model, exp, NOMINAL.dr_spec(), seeds=[3], obs_hook=hook, log=True)
    assert len(calls) == 50
    rest = np.setdiff1d(np.arange(31), idx)
    for t, (x, y) in enumerate(calls):
        assert np.all(y[idx] != x[idx])  # erased at every step
        np.testing.assert_array_equal(y[rest], x[rest])  # only the group changes
        assert np.any(np.all(np.isclose(pool[:, idx], y[idx]), axis=1))  # values from the pool
        np.testing.assert_allclose(rec["obs_flat"][0, t], y, rtol=1e-6, atol=1e-6)  # policy saw y


def test_ignored_group_gives_zero_closed_loop_drop():
    """Analytic controller (ignores grip_state) through the same erasure code path."""
    exp = {"task": "reach", "dr_cfg": {}}
    pol = controller_policy(PDController("reach"), "reach")
    runner = dist_runner(pol, exp, NOMINAL.dr_spec(), squash=lambda m: np.clip(m, -1, 1))
    seeds = [eval_episode_seed(0, 0, ep) for ep in range(6)]
    pool = fa.rollout_states("reach", PDController("reach"), 2, 0, 0)
    groups = make_groups("reach")
    nominal = runner(seeds, None)
    ignored = closed_loop_success(runner, seeds, groups["grip_state"], pool, rng_seed=0)
    used = closed_loop_success(runner, seeds, groups["goal"], pool, rng_seed=0)
    assert nominal.mean() == 1.0
    np.testing.assert_array_equal(ignored, nominal)
    assert used.mean() < nominal.mean()


# --- end-to-end schemas -------------------------------------------------------------------------
def _save(model, root, exp):
    (root / "ckpt").mkdir(parents=True, exist_ok=True)
    model.save(root / "ckpt" / f"{exp}_s0_final.zip")


def test_push_task_outputs_and_resume(tiny_model, tmp_path):
    _save(tiny_model("push_dr"), tmp_path, "push_dr")
    out = tmp_path / "c1"
    argv = [
        "--task", "push", "--exp", "push_dr", "--seeds", "0", "--root", str(tmp_path),
        "--out", str(out), "--episodes", "2", "--n-states", "3", "--n-open-states", "20",
        "--n-pool-eps", "2", "--n-boot", "20", "--methods", "grad,occlusion,gwie_c",
        "--no-wandb",
    ]  # fmt: skip
    fa.main(argv)
    summ = pd.read_csv(out / "c1_push_dr_summary.csv")
    assert list(summ["method"]) == ["grad", "occlusion", "gwie_c", "open_loop"]
    for c in ("tau_closed", "precision_at_k", "pauc", "runtime_ms"):
        assert {c, f"{c}_lo", f"{c}_hi"} <= set(summ.columns)
    dele = pd.read_csv(out / "c1_push_dr_deletion.csv")
    assert {"method", "k", "seed", "success"} <= set(dele.columns)
    assert sorted(dele["k"].unique()) == [0, 1, 2, 3, 4, 5]
    imp = pd.read_csv(out / "c1_push_dr_importance.csv")
    assert set(imp["group"]) == set(make_groups("push")) | {"cluster"}
    assert {"open_loop", "open_loop_lo", "closed_loop", "closed_loop_hi"} <= set(imp.columns)
    assert np.all(imp["open_loop"] >= 0)
    meta = json.loads((out / "run_meta_push_dr.json").read_text())
    assert meta["task"] == "push"
    # resume: partial files are reused, not recomputed
    partial = sorted((out / "partial").glob("push_dr_s0_*.npz"))
    assert len(partial) == 3
    mtimes = [p.stat().st_mtime_ns for p in partial]
    fa.main(argv)
    assert [p.stat().st_mtime_ns for p in partial] == mtimes
    pd.testing.assert_frame_equal(pd.read_csv(out / "c1_push_dr_summary.csv"), summ)


def test_shortcut_task_outputs(tiny_model, tmp_path):
    _save(tiny_model("reach_shortcut"), tmp_path, "reach_shortcut")
    out = tmp_path / "c1"
    fa.main(
        ["--task", "shortcut", "--seeds", "0", "--root", str(tmp_path), "--out", str(out),
         "--episodes", "3", "--n-states", "4", "--n-pool-eps", "2", "--n-boot", "20",
         "--methods", "grad,ig,kernelshap_grouped"]
    )  # fmt: skip
    summ = pd.read_csv(out / "c1_shortcut_summary.csv")
    assert list(summ["method"]) == ["grad", "ig", "kernelshap_grouped"]
    for c in ("hint_share", "goal_share", "detected_rate", "success_drop"):
        assert {c, f"{c}_lo", f"{c}_hi"} <= set(summ.columns)
    assert summ[["hint_share", "goal_share"]].sum(axis=1).le(1 + 1e-9).all()
    per = pd.read_csv(out / "c1_shortcut_per_state.csv")
    assert len(per) == 4 * 3 and set(per["seed"]) == {0}
    succ = pd.read_csv(out / "c1_shortcut_success.csv")
    assert set(succ["mode"]) == {"intact", "random"} and len(succ) == 6
    assert (out / "run_meta_shortcut.json").exists()


def test_controller_task_run_meta_name(tmp_path):
    fa.main(
        ["--task", "reach_controller", "--n-pool-eps", "1", "--n-states", "2",
         "--methods", "grad", "--out", str(tmp_path)]
    )  # fmt: skip
    assert (tmp_path / "run_meta_reach_controller.json").exists()
    assert not (tmp_path / "run_meta.json").exists()
