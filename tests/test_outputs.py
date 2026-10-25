"""Output scripts: tables from synthetic results, TBD for missing inputs, idempotent marker
filling, figures skip gracefully when inputs are missing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mt():
    return _load("make_tables")


@pytest.fixture(scope="module")
def mf():
    return _load("make_figs")


def _ci(rows: list[dict], metric: str, half: float = 0.05) -> list[dict]:
    for r in rows:
        r[f"{metric}_lo"], r[f"{metric}_hi"] = r[metric] - half, r[metric] + half
    return rows


def synthetic_results(root: Path) -> None:
    (root / "c1").mkdir(parents=True)
    rows = [
        dict(
            task="reach_controller",
            method=m,
            n_states=200,
            precision_at_k=p,
            spearman=0.9,
            irrelevant_share=0.0,
            runtime_ms=rt,
        )
        for m, p, rt in (("ig", 1.0, 0.8), ("gwie_c", 0.6667, 4.5))
    ]
    for c in ("precision_at_k", "spearman", "irrelevant_share", "runtime_ms"):
        _ci(rows, c, 0.01)
    pd.DataFrame(rows).to_csv(root / "c1" / "c1_reach_controller_summary.csv", index=False)

    (root / "c3").mkdir()
    rows = []
    for T in (10, 25):
        for mon, auc in (("asm", 0.81), ("obs_ood", 0.7), ("act_ood", 0.74), ("q_drop", 0.6)):
            rows.append(dict(monitor=mon, T=T, auroc=auc + T / 1000, auprc=0.5, tpr_at_5fpr=0.3))
    rows = _ci(_ci(rows, "auroc"), "auprc")
    pd.DataFrame(rows).to_csv(root / "c3" / "c3_push_dr_metrics.csv", index=False)
    rng = np.random.default_rng(0)
    eps = []
    for mon in ("asm", "obs_ood"):
        for i in range(40):
            failed = i % 2 == 0
            eps.append(
                dict(
                    seed=0,
                    cell_id=1 + i % 3,
                    ep_seed=i,
                    split="test",
                    monitor=mon,
                    failed=failed,
                    score_T10=rng.normal(1.0 if failed else 0.0),
                    score_T25=rng.normal(1.5 if failed else 0.0),
                    detect_step=5,
                )
            )
    pd.DataFrame(eps).to_csv(root / "c3" / "c3_push_dr_episode_scores.csv", index=False)

    (root / "robustness").mkdir()
    rob = [
        dict(exp=e, split=s, success=v, success_s0=v, success_s1=v - 0.02, success_s2=v + 0.02)
        for e, base in (("push_nodr", 0.3), ("push_dr", 0.55), ("push_drhist", 0.6))
        for s, v in (("nominal", base + 0.3), ("heldout", base))
    ]
    pd.DataFrame(_ci(rob, "success")).to_csv(root / "robustness" / "robustness.csv", index=False)

    (root / "c4").mkdir()
    rel = [
        dict(condition=c, method=m, group=g, share=s)
        for c in ("push_nodr", "push_dr", "push_drhist")
        for m in ("ig", "gwie_c")
        for g, s in (("obj_rel", 0.5), ("goal", 0.3), ("ee_vel", 0.2))
    ]
    pd.DataFrame(rel).to_csv(root / "c4" / "reliance.csv", index=False)


DOC = """# Doc
Intro.

<!-- BEGIN:c1_controller -->
TBD
<!-- END:c1_controller -->

X is <!-- BEGIN:num_X -->TBD<!-- END:num_X --> and Z is <!-- BEGIN:num_Z -->TBD<!-- END:num_Z -->.

<!-- BEGIN:c1_shortcut -->
<!-- END:c1_shortcut -->

<!-- BEGIN:not_a_table -->
keep me
<!-- END:not_a_table -->
"""


def test_formatting(mt):
    assert mt.fmt_ci(0.12345, 0.1, 0.2) == "0.123 [0.100, 0.200]"
    assert mt.fmt_num(-1e-17) == "0.000"
    assert mt.fmt_num(np.int64(200)) == "200"
    assert mt.fmt_num(float("nan")) == "n/a"
    df = pd.DataFrame({"m": ["a"], "x": [0.5], "x_lo": [0.4], "x_hi": [0.6], "n": [3]})
    out = mt.ci_frame(df)
    assert list(out.columns) == ["m", "x", "n"]
    assert out.iloc[0].tolist() == ["a", "0.500 [0.400, 0.600]", "3"]


def test_make_tables_synthetic(mt, tmp_path):
    root = tmp_path / "results"
    synthetic_results(root)
    doc = tmp_path / "README.md"
    doc.write_text(DOC, encoding="utf-8")
    out = root / "tables"
    tables, nums = mt.make_all(root, out, [doc], log=lambda s: None)

    for name in ("c1_controller", "c1_shortcut", "c2_runtime", "c3_push_dr", "robustness"):
        assert (out / f"{name}.md").exists()
    c1 = (out / "c1_controller.md").read_text()
    assert "0.667 [0.657, 0.677]" in c1 and "runtime" not in c1
    assert "4.500 [4.490, 4.510]" in (out / "c2_runtime.md").read_text()
    assert "TBD" in (out / "c1_shortcut.md").read_text()
    assert "TBD" in (out / "c3_push_drhist.md").read_text()
    assert "TBD" not in (out / "c3_push_dr.md").read_text()

    saved = json.loads((out / "numbers.json").read_text())
    assert saved["X"]["value"] == pytest.approx(0.3)
    assert saved["Y"]["value"] == pytest.approx(0.55)
    assert saved["Z"]["value"] == pytest.approx(0.81 + 0.01)
    assert saved["W"]["value"] == pytest.approx(0.74 + 0.01) and saved["W"]["monitor"] == "act_ood"

    text = doc.read_text(encoding="utf-8")
    assert "| reach_controller | gwie_c | 200 |" in text
    assert "X is <!-- BEGIN:num_X -->30.0% [25.0, 35.0]<!-- END:num_X -->" in text
    assert "Z is <!-- BEGIN:num_Z -->0.820 [0.770, 0.870]<!-- END:num_Z -->" in text
    assert "keep me" in text  # unknown marker untouched
    assert "TBD (missing: `results/c1/c1_shortcut_summary.csv`)" in text

    # idempotent: a second run changes nothing
    mt.make_all(root, out, [doc], log=lambda s: None)
    assert doc.read_text(encoding="utf-8") == text
    assert mt.fill_markers(text, {**tables, "num_X": "30.0% [25.0, 35.0]"}).count("BEGIN:") == 5


def test_make_tables_empty(mt, tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    tables, nums = mt.make_all(root, root / "tables", [], log=lambda s: None)
    assert all("TBD" in md for md in tables.values())
    assert nums == {"X": None, "Y": None, "Z": None, "W": None}
    assert "NoDR **TBD**" in mt.headline_block(nums)


def test_make_figs_empty_dir_skips(mf, tmp_path):
    msgs: list[str] = []
    mf.make_all(tmp_path / "results", tmp_path / "figs", log=msgs.append)
    assert not any(m.startswith("FAILED") for m in msgs), msgs
    assert not list((tmp_path / "figs").glob("*.png")) if (tmp_path / "figs").exists() else True
    assert sum(m.startswith("skip") for m in msgs) >= 6


def test_make_figs_synthetic(mf, tmp_path):
    root = tmp_path / "results"
    synthetic_results(root)
    msgs: list[str] = []
    mf.make_all(root, tmp_path / "figs", skip={"gwie"}, log=msgs.append)
    assert not any(m.startswith("FAILED") for m in msgs), msgs
    made = {p.name for p in (tmp_path / "figs").glob("*.png")}
    assert {
        "c1_controller_bars.png",
        "c3_push_dr_roc.png",
        "robustness.png",
        "c4_reliance.png",
    } <= made


def test_method_colours_fixed(mf):
    assert len(set(mf.METHOD_COLORS.values())) == len(mf.METHODS)
    assert mf.color_for("gwie_c", mf.METHOD_COLORS) == mf.METHOD_COLORS["gwie_c"]
    assert mf.color_for("DR+Hist", mf.CONDITION_COLORS) == mf.CONDITION_COLORS["push_drhist"]


# ---------------------------------------------------------------------------- Reach track (G2)

REACH = ("reach_nodr", "reach_dr", "reach_drhist")


def _ci_row(v: float, half: float = 0.01, seeds: bool = True) -> dict:
    d = dict(mean=v, mean_lo=v - half, mean_hi=v + half, n_seeds=3)
    if seeds:
        d.update(seed0=v, seed1=v, seed2=v)
    return d


def synthetic_reach(root: Path) -> None:
    """Small files with the schemas of the real Reach-track outputs."""
    (root / "c3").mkdir(parents=True, exist_ok=True)
    for i, exp in enumerate(REACH):
        rows = []
        for T in (10, 25):
            for sig, auc in (("asm", 0.6 + 0.05 * i), ("obs_ood", 0.62), ("act_ood", 0.8)):
                rows.append(
                    dict(exp=exp, signal=sig, T=T, alpha=0.2, n_test=100, n_fail_test=30)
                    | dict(auroc=auc, auroc_lo=auc - 0.02, auroc_hi=auc + 0.02)
                    | dict(auroc_s0=auc - 0.01, auroc_s1=auc, auroc_s2=auc + 0.01)
                )
        pd.DataFrame(rows).to_csv(root / "c3" / f"c3_{exp}_metrics.csv", index=False)
        pf = [
            dict(exp=exp, signal="asm", T=10, factor=f, n_test=50, n_fail_test=nf, auroc=a)
            | dict(auroc_lo=a, auroc_hi=a)
            for f, nf, a in (("gain", 5, 0.7), ("latency", 0, float("nan")))
        ]
        pd.DataFrame(pf).to_csv(root / "c3" / f"c3_{exp}_per_factor.csv", index=False)
    rng = np.random.default_rng(1)
    eps = [
        dict(exp="reach_dr", signal=sig, seed=0, split="test" if i % 2 else "calib", ep_seed=i)
        | dict(failed=i % 4 < 2, S10=rng.normal(i % 4 < 2), S25=rng.normal(i % 4 < 2))
        for sig in ("asm", "act_ood")
        for i in range(40)
    ]
    pd.DataFrame(eps).to_csv(root / "c3" / "c3_reach_dr_episode_scores.csv", index=False)

    (root / "robustness_reach").mkdir()
    rob = [
        dict(exp=e, split=s, n_cells=1, n_seeds=3, ci_backend="rliable")
        | dict(success=v, success_lo=v - 0.01, success_hi=v + 0.01)
        | dict(seed0=v - 0.02, seed1=v, seed2=v + 0.02)
        for e, v in zip(REACH, (0.5, 0.6612, 0.74), strict=True)
        for s in ("nominal", "heldout", "factor:gain")
    ]
    pd.DataFrame(rob).to_csv(root / "robustness_reach" / "robustness.csv", index=False)
    poi = [
        dict(x="reach_dr", y="reach_nodr", split="heldout", poi=0.887, poi_lo=0.87, poi_hi=0.89),
        dict(x="reach_drhist", y="reach_dr", split="heldout", poi=0.748, poi_lo=0.7, poi_hi=0.8),
    ]
    pd.DataFrame(poi).to_csv(root / "robustness_reach" / "poi.csv", index=False)

    (root / "c4_reach").mkdir()
    rel = [
        dict(exp=e, method=m, level="base", quantity=q, group=g or None) | _ci_row(v)
        for e in REACH
        for m in ("ig", "gwie_c")
        for q, g, v in (("share", "ee_pos", 0.45), ("share", "goal", 0.5), ("entropy", "", 0.7))
    ]
    pd.DataFrame(rel).to_csv(root / "c4_reach" / "reliance.csv", index=False)
    tmp = [
        dict(exp="reach_drhist", method=m, level="temporal", quantity="share", group=g)
        | _ci_row(v)
        | dict(scope=sc)
        for m in ("ig", "gwie_c")
        for sc in ("nominal", "heldout")
        for g, v in (("t", 0.5444), ("t-1", 0.2), ("t-2", 0.1), ("t-3", 0.1), ("act", 0.0556))
    ]
    pd.DataFrame(tmp).to_csv(root / "c4_reach" / "temporal.csv", index=False)
    pr = [
        dict(exp=e, target="gain", features=f, scope=sc, r2_lo=-0.03, r2_hi=0.04)
        | dict(r2=0.0092 if sc == "all" else 0.01, claim_sysid=False)
        | dict(n_seeds=3.0 if sc == "all" else None)
        for e in REACH
        for f in ("penult", "obs_single", "penult_random")
        for sc in ("seed0", "seed1", "seed2", "all")
    ]
    pd.DataFrame(pr).to_csv(root / "c4_reach" / "probes.csv", index=False)

    (root / "c1").mkdir(exist_ok=True)
    her = [
        dict(method=m, n_seeds=3, n_states=600, runtime_ms=1.0)
        | dict(hint_share=h, hint_share_lo=h - 0.01, hint_share_hi=h + 0.01)
        | dict(goal_share=0.4, goal_share_lo=0.39, goal_share_hi=0.41)
        | dict(hint_share_per_seed="0.03448;0.02327;0.02112")
        | dict(success_drop=0.0, success_drop_lo=0.0, success_drop_hi=0.0)
        for m, h in (("ig", 0.039), ("gwie_c", 0.029))
    ]
    pd.DataFrame(her).to_csv(root / "c1" / "c1_shortcut_her_summary.csv", index=False)
    v2 = [
        r
        | dict(hint_share=0.42, hint_share_lo=0.41, hint_share_hi=0.43)
        | dict(success_drop=0.987, success_drop_lo=0.967)
        for r in her
    ]
    pd.DataFrame(v2).to_csv(root / "c1" / "c1_shortcut_summary.csv", index=False)


REACH_DOC = """# Report
<!-- BEGIN:reach_summary -->
TBD
<!-- END:reach_summary -->

<!-- BEGIN:robustness_reach -->
TBD
<!-- END:robustness_reach -->

<!-- BEGIN:c3_reach_dr -->
<!-- END:c3_reach_dr -->

<!-- BEGIN:c4_reach_probes -->
<!-- END:c4_reach_probes -->

<!-- BEGIN:c1_shortcut_her -->
<!-- END:c1_shortcut_her -->
"""

REACH_TABLES = (
    *(f"c3_{e}" for e in REACH),
    *(f"c3_{e}_per_factor" for e in REACH),
    "robustness_reach",
    "c4_reach_reliance",
    "c4_reach_reliance_slots",
    "c4_reach_temporal",
    "c4_reach_probes",
    "c1_shortcut_her",
)


def test_reach_tables_synthetic(mt, tmp_path):
    root = tmp_path / "results"
    synthetic_reach(root)
    doc = tmp_path / "report.md"
    doc.write_text(REACH_DOC, encoding="utf-8")
    tables, _ = mt.make_all(root, root / "tables", [doc], log=lambda s: None)
    for name in REACH_TABLES:
        assert name in tables and (root / "tables" / f"{name}.md").exists()
    for name in set(REACH_TABLES) - {"c4_reach_reliance_slots"}:  # no slot rows in synthetic
        assert not tables[name].startswith("| status"), name

    c3 = tables["c3_reach_dr"]
    assert "0.650 [0.630, 0.670]" in c3 and "| n_seeds |" in c3 and c3.count("| 3 |") == 6
    assert "n/a [n/a, n/a]" in tables["c3_reach_dr_per_factor"]  # no failures -> no AUROC

    rob = tables["robustness_reach"]
    assert "0.661 [0.651, 0.671]" in rob and "0.641 / 0.661 / 0.681" in rob
    assert "0.887 [0.870, 0.890]" in rob and "ci_backend" not in rob.split("\n\n")[0]

    rel = tables["c4_reach_reliance"]
    assert "NoDR (n=3)" in rel and "DR+Hist (n=3)" in rel and "| ig | entropy |" in rel
    assert "0.450 [0.440, 0.460]; seeds 0.450 / 0.450 / 0.450" in rel
    tmp = tables["c4_reach_temporal"]
    assert "t-3 (n=3)" in tmp and "0.544 [0.534, 0.554]" in tmp and "| heldout |" in tmp

    pr = tables["c4_reach_probes"]
    assert "| reach_dr | gain | penult | 0.009 [-0.030, 0.040] | 0.010 / 0.010 / 0.010 | 3 |" in pr
    assert "Claim rule outcome (`claim_sysid`" in pr and "reach_dr/gain: **False**" in pr
    assert "No implicit system identification is claimed" in pr

    her = tables["c1_shortcut_her"]
    assert "0.034 / 0.023 / 0.021" in her and "runtime" not in her
    assert "0.034 / 0.023 / 0.021" in tables["c1_shortcut"]  # same per-seed formatting

    text = doc.read_text(encoding="utf-8")
    assert "DR **66.1% [65.1, 67.1]**" in text and "n=3 seeds" in text
    assert "P(DR > NoDR) 0.887 [0.870, 0.890]" in text
    assert "ASM 0.650 [0.630, 0.670] vs best baseline (act_ood) 0.800 [0.780, 0.820]" in text
    assert "ASM below the best baseline (CIs disjoint)" in text
    assert "TBD" not in text
    # idempotent: a second run changes nothing
    mt.make_all(root, root / "tables", [doc], log=lambda s: None)
    assert doc.read_text(encoding="utf-8") == text


def test_reach_tables_missing_tbd(mt, tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    doc = tmp_path / "report.md"
    doc.write_text(REACH_DOC, encoding="utf-8")
    tables, _ = mt.make_all(root, root / "tables", [doc], log=lambda s: None)
    for name in REACH_TABLES:
        assert "TBD" in tables[name], name
    summary = mt.reach_summary_block(root)
    assert summary.count("TBD") >= 5 and "FetchReach: TBD" in summary
    text = doc.read_text(encoding="utf-8")
    assert "TBD (missing: `results/robustness_reach/robustness.csv`)" in text
    assert "TBD (missing: `results/c4_reach/probes.csv`)" in text
    mt.make_all(root, root / "tables", [doc], log=lambda s: None)
    assert doc.read_text(encoding="utf-8") == text

    # robustness present, POI missing: the table renders and the POI part says TBD
    synthetic_reach(tmp_path / "r2")
    (tmp_path / "r2" / "robustness_reach" / "poi.csv").unlink()
    rob = mt.table_robustness_reach(tmp_path / "r2")
    assert "0.661 [0.651, 0.671]" in rob and "TBD (missing: `r2/robustness_reach/poi.csv`)" in rob


def test_reach_figs_synthetic(mf, tmp_path):
    root = tmp_path / "results"
    synthetic_reach(root)
    msgs: list[str] = []
    mf.make_all(root, tmp_path / "figs", skip={"gwie"}, log=msgs.append)
    assert not any(m.startswith("FAILED") for m in msgs), msgs
    made = {p.name for p in (tmp_path / "figs").glob("*.png")}
    assert {
        "robustness_reach.png",
        "c3_reach_dr_roc.png",
        "c4_reach_reliance.png",
        "c4_reach_temporal.png",
        "c1_shortcut_shares.png",
    } <= made


def test_reach_roc_skips_without_episode_scores(mf, tmp_path):
    root = tmp_path / "results"
    synthetic_reach(root)
    (root / "c3" / "c3_reach_dr_episode_scores.csv").unlink()
    msgs: list[str] = []
    mf.make_all(root, tmp_path / "figs", skip={"gwie"}, log=msgs.append)
    assert not any(m.startswith("FAILED") for m in msgs), msgs
    assert any(m.startswith("skip c3_reach_dr_roc.png") for m in msgs)
    assert (tmp_path / "figs" / "robustness_reach.png").exists()
    assert mf.CONDITION_COLORS["reach_dr"] == mf.CONDITION_COLORS["push_dr"]
