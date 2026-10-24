"""C4 analysis (reliance, probes) and the robustness table. Synthetic data + tiny_model only."""

import numpy as np
import pandas as pd
import pytest

from projectv.analysis import probes, reliance, robustness
from projectv.eval.cells import NOMINAL, Cell
from projectv.eval.rollout import run_episodes
from projectv.eval.steplog import save_steplog, steplog_path
from projectv.utils.config import load_exp
from projectv.utils.seeding import eval_episode_seed
from projectv.xai.groups import make_groups
from projectv.xai.policy_fn import SB3Policy


# --- probes -----------------------------------------------------------------------------------
def _episodes(E=40, T=20):
    return np.repeat(np.arange(E), T)


def test_probe_recovers_planted_linear_target():
    rng = np.random.default_rng(0)
    ep = _episodes()
    X = rng.normal(size=(len(ep), 10))
    y = X @ rng.normal(size=10) + 0.05 * rng.normal(size=len(ep))
    pred, alphas = probes.probe_oof(X, y, ep)
    assert probes.r2(y, pred) > 0.9
    assert len(alphas) == 5 and set(alphas) <= set(probes.ALPHAS)


def test_probe_fails_on_noise():
    rng = np.random.default_rng(1)
    ep = _episodes()
    X = rng.normal(size=(len(ep), 10))
    y = np.repeat(rng.normal(size=40), 20)  # per-episode target unrelated to X
    pred, _ = probes.probe_oof(X, y, ep)
    assert abs(probes.r2(y, pred)) < 0.1


def test_episode_grouped_cv_never_splits_an_episode():
    ep = np.repeat(np.random.default_rng(2).permutation(23), 7)
    folds = probes.episode_folds(ep, 5)
    seen = []
    for tr, te in folds:
        assert not set(ep[tr]) & set(ep[te])
        seen.append(set(ep[te]))
    assert set().union(*seen) == set(ep) and sum(len(s) for s in seen) == 23


def test_sysid_claim_rule():
    a = (0.6, 0.8)
    assert probes.sysid_claim(a, b=(0.1, 0.3), d=(0.0, 0.2))
    assert not probes.sysid_claim(a, b=(0.5, 0.7), d=(0.0, 0.2))  # overlaps raw obs
    assert not probes.sysid_claim(a, b=(0.1, 0.3), d=(0.55, 0.65))  # overlaps random net
    assert not probes.sysid_claim((0.1, 0.2), b=(0.3, 0.4), d=(0.0, 0.05))  # a below b


def test_bootstrap_r2_ci_contains_point():
    rng = np.random.default_rng(3)
    ep = _episodes(30, 10)
    y = rng.normal(size=len(ep))
    runs = [(y, y + 0.3 * rng.normal(size=len(y)), ep) for _ in range(3)]
    p, lo, hi = probes.bootstrap_r2(runs, n_boot=200)
    assert lo <= p <= hi and 0.7 < p < 1.0


def test_episode_targets_and_current_slot():
    names = np.array(["gain", "obj_friction", "obj_mass", "table_friction"])
    v = np.array([[1.1, 0.5, 2.0, 1.2]], dtype=np.float32)
    t = probes.episode_targets(v, names)
    np.testing.assert_allclose(t["log_friction"], np.log(0.5 * 1.2), rtol=1e-6)
    np.testing.assert_allclose(t["log_mass"], np.log(2.0), rtol=1e-6)
    np.testing.assert_allclose(t["gain"], 1.1, rtol=1e-6)
    g = make_groups("push", history=True)
    idx = probes.current_slot_idx(g)
    assert len(idx) == 25 + 6 and idx.max() == 4 * 25 + 12 + 5


def test_probes_smoke_on_tiny_model(tiny_model, tmp_path):
    model = tiny_model("push_drhist")
    exp = load_exp("configs/exp/push_drhist.yaml")
    df = probes.run_probes(
        tmp_path, tmp_path / "c4", ["push_drhist"], [0], n_eps=10, max_states=400,
        load_model_fn=lambda e, s: model, exp_cfgs={"push_drhist": exp}, n_boot=50,
    )  # fmt: skip
    assert set(df.features) == set(probes.FEATURES)
    assert set(df.target) == set(probes.TARGETS)
    assert df.r2.notna().all() and "claim_sysid" in df
    assert (tmp_path / "c4" / "probe_rollouts" / "push_drhist_s0_n10.npz").exists()
    # random-init actor: same width as the trained one, different weights
    fn = probes.random_actor_penultimate(model)
    X = np.zeros((2, model.observation_space["observation"].shape[0] + 6))
    assert fn(X).shape[1] == SB3Policy(model).penultimate(__import__("torch").as_tensor(X)).shape[1]


# --- reliance ---------------------------------------------------------------------------------
def test_entropy_and_shares_on_synthetic_attributions():
    scores = np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 0.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]])
    sh = reliance.shares_from_scores(scores)
    np.testing.assert_allclose(sh.sum(1), 1.0)
    np.testing.assert_allclose(sh[2], [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(reliance.entropy(sh), [np.log(4), 0.0, np.log(2)])
    np.testing.assert_allclose(reliance.normalized_entropy(sh), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(reliance.shares_from_scores(np.zeros((1, 3))), 0.0)


def test_base_and_temporal_aggregation():
    names = ["obj_pos", "goal", "obj_pos@t-1", "obj_pos@t-3", "act@t-1", "act@t-2"]
    sh = np.array([[0.1, 0.2, 0.3, 0.15, 0.05, 0.2]])
    base, bn = reliance.aggregate_base(sh, names)
    assert bn == ["obj_pos", "goal", "act"]
    np.testing.assert_allclose(base[0], [0.55, 0.2, 0.25])
    tb = reliance.temporal_buckets(sh, names)
    np.testing.assert_allclose(tb[0], [0.3, 0.3, 0.0, 0.15, 0.25])


def test_split_episodes_is_a_partition():
    a, b = reliance.split_episodes(11, seed=4)
    assert not set(a) & set(b) and set(a) | set(b) == set(range(11))
    a2, _ = reliance.split_episodes(11, seed=4)
    np.testing.assert_array_equal(a, a2)


def test_reliance_smoke_on_tiny_model(tiny_model, tmp_path):
    model = tiny_model("push_drhist")
    exp = load_exp("configs/exp/push_drhist.yaml")
    cells = [NOMINAL, Cell(1, "obj_mass=3.0", "mass", {"obj_mass": 3.0})]
    for c in cells:
        seeds = [eval_episode_seed(0, c.id, ep) for ep in range(4)]
        rec = run_episodes(model, exp, c.dr_spec(), seeds, log=True)
        save_steplog(steplog_path(tmp_path, "push_drhist", 0, c.id), rec, c.id)
    calls = []

    def load(e, s):
        calls.append((e, s))
        return SB3Policy(model)

    kw = dict(
        methods=("ig", "gwie_c"), n_states=6, n_states_cell=4,
        exp_cfgs={"push_drhist": exp},
    )  # fmt: skip
    out = tmp_path / "c4"
    factor_of = {c.id: c.factor for c in cells}
    rel, tmp = reliance.run_reliance(
        tmp_path, out, ["push_drhist"], [0, 1], load, factor_of, **kw
    )  # seed 1 has no logs -> skipped
    assert calls == [("push_drhist", 0)]
    assert set(rel.level) == {"slot", "base"} and set(rel.method) == {"ig", "gwie_c"}
    for (m, lvl), d in rel[rel.quantity == "share"].groupby(["method", "level"]):
        assert d["mean"].sum() == pytest.approx(1.0), (m, lvl)
    ent = rel[rel.quantity == "entropy_norm"]["mean"]
    assert ((ent >= 0) & (ent <= 1)).all()
    assert set(tmp.scope) == {"nominal", "heldout", "factor:mass"}
    for _, d in tmp.groupby(["method", "scope"]):
        assert d["mean"].sum() == pytest.approx(1.0) and list(d.group) == list(
            reliance.TEMPORAL_BUCKETS
        )
    assert "seed0" in rel and (rel.n_seeds == 1).all()
    # resumable: a rerun reads the per-policy cache and never loads the policy
    calls.clear()
    rel2, _ = reliance.run_reliance(tmp_path, out, ["push_drhist"], [0], load, factor_of, **kw)
    assert calls == [] and np.allclose(rel2["mean"], rel["mean"])


# --- robustness -------------------------------------------------------------------------------
RATES = {  # (condition) -> per-cell success rate, cells 0 nominal, 1-2 friction, 3 mass
    "push_nodr": [0.9, 0.2, 0.3, 0.4],
    "push_dr": [0.9, 0.5, 0.6, 0.7],
    "push_drhist": [0.95, 0.6, 0.7, 0.8],
}
FACTORS = ["nominal", "friction", "friction", "mass"]


def _write_eval(root, n_eps=20):
    (root / "eval").mkdir(parents=True)
    for exp, rates in RATES.items():
        for seed in range(3):
            rows = []
            for cid, (r, f) in enumerate(zip(rates, FACTORS, strict=True)):
                r_s = min(1.0, r + 0.05 * (seed - 1))
                k = int(round(r_s * n_eps))
                for ep in range(n_eps):
                    rows.append(
                        dict(cell_id=cid, cell_name=f"c{cid}", factor=f, episode=ep,
                             ep_seed=eval_episode_seed(seed, cid, ep), success=ep < k)
                    )  # fmt: skip
            pd.DataFrame(rows).to_csv(root / "eval" / f"{exp}_s{seed}.csv", index=False)


def test_robustness_on_synthetic_eval(tmp_path):
    _write_eval(tmp_path)
    df = robustness.load_eval(tmp_path)
    rob, poi = robustness.robustness_tables(df, n_boot=300, use_rliable=False)
    r = rob.set_index(["exp", "split"])
    assert r.loc[("push_dr", "nominal"), "success"] == pytest.approx(0.9)
    assert r.loc[("push_dr", "heldout"), "success"] == pytest.approx(0.6)
    assert r.loc[("push_nodr", "factor:friction"), "success"] == pytest.approx(0.25)
    row = r.loc[("push_dr", "heldout")]
    assert row.success_lo <= row.success <= row.success_hi and row.n_seeds == 3
    np.testing.assert_allclose([row.seed0, row.seed1, row.seed2], [0.55, 0.6, 0.65])
    p = poi.set_index(["x", "y", "split"])
    assert p.loc[("push_dr", "push_nodr", "heldout"), "poi"] == pytest.approx(1.0)
    assert p.loc[("push_dr", "push_nodr", "nominal"), "poi"] == pytest.approx(0.5)
    assert set(p.index.get_level_values("split")) >= {"nominal", "heldout", "factor:mass"}


def test_robustness_rliable_backend_matches_manual():
    if robustness._rliable_library() is None:
        pytest.skip("rliable.library unusable here (arch/pandas incompatibility)")
    S = np.random.default_rng(0).uniform(size=(3, 6))
    a = robustness.stratified_ci(S, 2000, use_rliable=True)
    b = robustness.stratified_ci(S, 2000, use_rliable=False)
    assert a[3] == "rliable" and b[3] == "manual"
    assert a[0] == b[0] and abs(a[1] - b[1]) < 0.03 and abs(a[2] - b[2]) < 0.03


def test_robustness_cli_writes_outputs(tmp_path, capsys):
    _write_eval(tmp_path)
    robustness.main(["--root", str(tmp_path), "--n-boot", "100"])
    out = tmp_path / "robustness"
    assert (out / "robustness.csv").exists() and (out / "poi.csv").exists()
    assert (out / "run_meta.json").exists()
    assert "IQM" in capsys.readouterr().out
