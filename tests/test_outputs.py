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
