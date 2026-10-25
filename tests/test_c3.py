"""C3 pipeline: splits, baselines, ASM signal, alpha selection, end-to-end smoke run."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from projectv.eval.steplog import save_steplog, steplog_path
from projectv.monitor import asm as asm_mod
from projectv.monitor import baselines as bl
from projectv.monitor.asm import fit_nominal, mahalanobis
from projectv.monitor.metrics import ALPHAS, auroc, ewma, prefix_score, select_alpha
from projectv.monitor.splits import CALIB, TEST, assign_splits, calib_mask
from projectv.utils.config import text_hash
from projectv.utils.seeding import eval_episode_seed
from projectv.xai.groups import flat_dim, make_groups
from projectv.xai.policy_fn import LinearGaussian

T = 50


# --- splits -------------------------------------------------------------------------------------
def test_split_deterministic_disjoint_balanced():
    seeds = np.array([eval_episode_seed(1, 3, ep) for ep in range(100)])
    a = assign_splits(seeds)
    b = assign_splits(seeds[::-1].copy())[::-1]  # order of episodes must not matter
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) == {CALIB, TEST}
    assert (a == CALIB).sum() == 50 and (a == TEST).sum() == 50
    cal = set(seeds[a == CALIB].tolist())
    tst = set(seeds[a == TEST].tolist())
    assert cal.isdisjoint(tst) and len(cal | tst) == 100


def test_split_is_50_50_within_each_cell():
    cells = np.repeat([0, 1, 2], [200, 100, 100])
    seeds = np.concatenate(
        [
            [eval_episode_seed(0, c, ep) for ep in range(n)]
            for c, n in ((0, 200), (1, 100), (2, 100))
        ]
    )
    m = calib_mask(seeds, cells)
    for c, n in ((0, 200), (1, 100), (2, 100)):
        assert m[cells == c].sum() == n // 2
    # not a trivial first-half / second-half split
    assert 0 < m[cells == 1][:50].sum() < 50


# --- synthetic step logs ------------------------------------------------------------------------
def synth_log(rng, E, d=31, H=16, shift=0.0, fail=None, cell_id=0, seed=0):
    """Failed episodes drift away from nominal after step 5 in every logged quantity."""
    fail = np.zeros(E, bool) if fail is None else np.asarray(fail, bool)
    drift = np.zeros((E, T))
    drift[:, 5:] = shift * fail[:, None]
    obs = rng.normal(0, 1, (E, T, d)) + drift[..., None]
    act_mean = rng.normal(0, 0.5, (E, T, 4))
    return {
        "obs_flat": obs.astype(np.float32),
        "act_mean": act_mean.astype(np.float32),
        "log_std": (rng.normal(-2, 0.1, (E, T, 4)) + 0.3 * drift[..., None]).astype(np.float32),
        "action": np.tanh(act_mean).astype(np.float32),
        "q_min": (rng.normal(-5, 0.5, (E, T)) - drift).astype(np.float32),
        "penult": (rng.normal(0, 1, (E, T, H)) + drift[..., None]).astype(np.float32),
        "is_success": np.repeat(~fail[:, None], T, axis=1),
        "success": ~fail,
        "dr_values": np.zeros((E, 1), np.float32),
        "dr_names": np.array(["obj_mass"]),
        "ep_seed": np.array([eval_episode_seed(seed, cell_id, ep) for ep in range(E)], np.int64),
    }


# --- baselines ----------------------------------------------------------------------------------
def test_scalar_baselines_shape_and_sign():
    rng = np.random.default_rng(0)
    log = synth_log(rng, 6)
    s = bl.act_std(log)
    assert s.shape == (6, T)
    np.testing.assert_allclose(s, np.exp(log["log_std"]).mean(-1), rtol=1e-6)
    log2 = dict(log, log_std=log["log_std"] + 1.0)
    assert np.all(bl.act_std(log2) > s)
    q = bl.q_drop(log)
    assert q.shape == (6, T)
    np.testing.assert_allclose(q, -log["q_min"])


@pytest.mark.parametrize("key,fn", [("obs_flat", bl.obs_ood), ("penult", bl.act_ood)])
def test_mahalanobis_baselines_increase_off_nominal(key, fn):
    rng = np.random.default_rng(1)
    nom = synth_log(rng, 40)
    model = bl.fit_feature_model(nom[key], np.ones(40, bool))
    shifted = synth_log(rng, 10, shift=3.0, fail=np.ones(10, bool))
    clean = synth_log(rng, 10)
    s_shift, s_clean = fn(shifted, model), fn(clean, model)
    assert s_shift.shape == (10, T)
    assert s_shift[:, 10:].mean() > 2 * s_clean[:, 10:].mean()
    assert np.all(s_clean >= 0)


def test_feature_model_fit_uses_calib_rows_only():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (20, T, 5))
    calib = np.zeros(20, bool)
    calib[:10] = True
    X_poison = X.copy()
    X_poison[~calib] += 1e3  # test rows must not influence the nominal model
    a = bl.fit_feature_model(X, calib)
    b = bl.fit_feature_model(X_poison, calib)
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.precision, b.precision)


def test_standardized_nominal_handles_mixed_scales_and_constants():
    rng = np.random.default_rng(3)
    z = np.c_[rng.normal(0, 1, 2000), rng.normal(0, 1e-3, 2000), np.full(2000, 0.7)]
    m = fit_nominal(z, standardize=True)
    # a 5-sigma move along the tiny-scale feature must be as visible as on the unit feature
    d_small = mahalanobis(m, np.array([[0.0, 5e-3, 0.7]]))[0]
    d_big = mahalanobis(m, np.array([[5.0, 0.0, 0.7]]))[0]
    assert d_small == pytest.approx(d_big, rel=0.2)
    assert np.isfinite(mahalanobis(m, z)).all()


def test_ensemble_std():
    d = 31
    rng = np.random.default_rng(4)
    W = torch.as_tensor(rng.normal(size=(4, d)))
    obs = rng.normal(size=(3, T, d)).astype(np.float32)
    same = bl.ensemble_std([LinearGaussian(W), LinearGaussian(W.clone())], obs)
    assert same.shape == (3, T)
    np.testing.assert_allclose(same, 0.0, atol=1e-9)
    W2 = W.clone()
    W2[:, 0] += 1.0
    diff = bl.ensemble_std([LinearGaussian(W), LinearGaussian(W2)], obs, batch=7)
    x = obs.astype(np.float64)
    ref = np.stack([x @ W.numpy().T, x @ W2.numpy().T]).std(0).mean(-1)
    np.testing.assert_allclose(diff, ref, rtol=1e-6)


# --- ASM signal ---------------------------------------------------------------------------------
def test_monitor_groups_push_and_history():
    g = asm_mod.monitor_groups("push", history=False)
    assert len(g) == 10 and "goal" in g and "act" not in g
    gh = asm_mod.monitor_groups("push", history=True)
    assert len(gh) == 11 and "act" in gh
    full = make_groups("push", history=True)
    assert len(gh["obj_pos"]) == sum(
        len(v) for k, v in full.items() if k.split("@")[0] == "obj_pos"
    )
    assert len(gh["act"]) == 12
    allidx = np.sort(np.concatenate(list(gh.values())))
    np.testing.assert_array_equal(allidx, np.arange(flat_dim("push", history=True)))


def test_attribution_shares_single_group_policy():
    d = flat_dim("push")
    groups = asm_mod.monitor_groups("push")
    W = torch.zeros(4, d, dtype=torch.float64)
    rng = np.random.default_rng(5)
    W[:, groups["obj_rel"]] = torch.as_tensor(rng.normal(size=(4, 3)))
    X = rng.normal(size=(37, d))
    z = asm_mod.attribution_shares(LinearGaussian(W), X, X.mean(0), groups, batch=10)
    assert z.shape == (37, 10)
    np.testing.assert_allclose(z.sum(-1), 1.0, atol=1e-9)
    j = list(groups).index("obj_rel")
    np.testing.assert_allclose(z[:, j], 1.0, atol=1e-9)


# --- alpha selection ----------------------------------------------------------------------------
def _spike_vs_drift(rng, n, kind):
    """Failures with a single spike (alpha=1 best) or a weak sustained drift (small alpha best)."""
    m = rng.normal(0, 1, (2 * n, T))
    y = np.r_[np.zeros(n), np.ones(n)].astype(bool)
    if kind == "spike":
        m[n:, 12] += 4.0
    else:
        m[n:, 3:] += 0.6
    return m, y


def test_alpha_selection_uses_calib_only():
    rng = np.random.default_rng(6)
    m_cal, y_cal = _spike_vs_drift(rng, 150, "spike")
    m_tst, y_tst = _spike_vs_drift(rng, 150, "drift")
    m = np.r_[m_cal, m_tst]
    y = np.r_[y_cal, y_tst]
    calib = np.r_[np.ones(300, bool), np.zeros(300, bool)]

    # the test split is adversarial: it prefers heavy smoothing
    test_auc = {a: auroc(prefix_score(ewma(m_tst, a), 25), y_tst) for a in ALPHAS}
    assert max(test_auc, key=test_auc.get) < 1.0
    alpha, cal_auc = select_alpha(m, y, calib, T=25)
    assert alpha == 1.0
    assert set(cal_auc) == set(ALPHAS)

    # arbitrary changes to the test split do not move the choice
    m2, y2 = m.copy(), y.copy()
    m2[~calib] = rng.normal(0, 50, m2[~calib].shape)
    y2[~calib] = rng.permutation(y2[~calib])
    assert select_alpha(m2, y2, calib, T=25) == (alpha, cal_auc)


# --- end-to-end smoke ---------------------------------------------------------------------------
def _write_world(tmp_path, tiny_model):
    from projectv.monitor.c3 import load_actor_critic as load_model

    cells = [
        {"id": 0, "name": "nominal", "factor": "nominal", "params": {}},
        {
            "id": 1,
            "name": "obj_friction=0.2",
            "factor": "friction",
            "params": {"obj_friction": 0.2},
        },
        {"id": 2, "name": "obj_mass=3.0", "factor": "mass", "params": {"obj_mass": 3.0}},
    ]
    cells_path = tmp_path / "heldout.yaml"
    cells_path.write_text(yaml.safe_dump({"frozen": True, "cells": cells}, sort_keys=False))
    root = tmp_path / "results"
    root.mkdir()
    (root / "heldout_hash.txt").write_text(text_hash(cells_path))

    model = tiny_model("push_dr")
    (root / "ckpt").mkdir()
    model.save(root / "ckpt" / "push_dr_s0_final.zip")
    other = load_model(root / "ckpt" / "push_dr_s0_final.zip")
    with torch.no_grad():
        for p in other.policy.actor.parameters():
            p.add_(0.1 * torch.randn_like(p))
    other.save(root / "ckpt" / "push_dr_s1_final.zip")

    rng = np.random.default_rng(7)
    for seed in (0, 1):
        for cid, E, ffrac in ((0, 40, 0.15), (1, 24, 0.5), (2, 24, 0.5)):
            fail = rng.random(E) < ffrac
            fail[:2] = [True, False]
            fail[-2:] = [True, False]
            # nominal failures carry no drift, so the nominal fit is not contaminated by it
            shift = 0.0 if cid == 0 else 2.5
            log = synth_log(rng, E, shift=shift, fail=fail, cell_id=cid, seed=seed)
            save_steplog(steplog_path(root, "push_dr", seed, cid), log, cid)
    return cells_path, root


def test_c3_end_to_end(tmp_path, tiny_model, monkeypatch):
    cells_path, root = _write_world(tmp_path, tiny_model)
    out = root / "c3"
    argv = [
        "--exp", "push_dr", "--seeds", "0", "1", "--root", str(root), "--cells", str(cells_path),
        "--n-boot", "40", "--no-wandb",
    ]  # fmt: skip
    asm_mod.main(argv)

    met = pd.read_csv(out / "c3_push_dr_metrics.csv")
    assert set(met["signal"]) == set(bl.SIGNALS)
    assert sorted(set(met["T"])) == [10, 25]
    assert len(met) == 2 * len(bl.SIGNALS)
    for col in ("auroc", "auprc", "tpr_at_5fpr", "det_step_median"):
        assert {col, f"{col}_lo", f"{col}_hi"} <= set(met.columns)
    assert met["auroc"].between(0, 1).all()
    assert {"auroc_s0", "auroc_s1"} <= set(met.columns)
    # failures drift strongly in obs: obs_ood must detect them on the test split
    assert met.loc[(met.signal == "obs_ood") & (met["T"] == 25), "auroc"].item() > 0.8

    pf = pd.read_csv(out / "c3_push_dr_per_factor.csv")
    assert set(pf["factor"]) == {"friction", "mass"}
    assert len(pf) == 2 * 2 * len(bl.SIGNALS)

    alpha = json.loads((out / "c3_push_dr_alpha.json").read_text())
    assert set(alpha["signals"]) == set(bl.SIGNALS)
    assert all(v["alpha"] in ALPHAS for v in alpha["signals"].values())

    ep = pd.read_csv(out / "c3_push_dr_episode_scores.csv")
    n_eps = 2 * (40 + 24 + 24)
    assert len(ep) == n_eps * len(bl.SIGNALS)
    assert {"e_0", "e_49", "S10", "S25", "thr_T25", "split", "failed"} <= set(ep.columns)
    assert set(ep["split"]) == {CALIB, TEST}
    assert (out / "run_meta.json").exists()

    cache = sorted((out / "cache").rglob("*.npz"))
    assert len(cache) == 2 * 3 * 2  # (asm, ensemble) x 3 cells x 2 seeds

    # rerun: every cached cell is skipped (no attribution or ensemble recomputation)
    def boom(*a, **k):
        raise AssertionError("cache not used")

    monkeypatch.setattr(asm_mod, "attribution_shares", boom)
    monkeypatch.setattr(bl, "ensemble_std", boom)
    asm_mod.main(argv)
    met2 = pd.read_csv(out / "c3_push_dr_metrics.csv")
    pd.testing.assert_frame_equal(met, met2)


def test_c3_refuses_changed_heldout(tmp_path, tiny_model):
    cells_path, root = _write_world(tmp_path, tiny_model)
    cells_path.write_text(cells_path.read_text() + "\n# edited after freeze\n")
    argv = ["--exp", "push_dr", "--seeds", "0", "1", "--root", str(root), "--cells",
            str(cells_path), "--n-boot", "10"]  # fmt: skip
    with pytest.raises(AssertionError, match="changed after freeze"):
        asm_mod.main(argv)
    asm_mod.main([*argv, "--no-hash-check"])  # tests only
    assert (root / "c3" / "c3_push_dr_metrics.csv").exists()


def test_c3_meta_rejects_logs_from_other_checkpoint(tmp_path):
    import json

    import pytest as _pt

    from projectv.eval.steplog import steplog_path
    from projectv.monitor.c3 import check_steplog_meta

    d = steplog_path(tmp_path, "push_dr", 0, 0).parent
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"heldout_hash": "h", "checkpoint_sha256": "aaa"}))
    check_steplog_meta(tmp_path, "push_dr", 0, "h", "aaa")
    with _pt.raises(AssertionError):
        check_steplog_meta(tmp_path, "push_dr", 0, "h", "bbb")
    with _pt.raises(AssertionError):
        check_steplog_meta(tmp_path, "push_dr", 0, "other", "aaa")
