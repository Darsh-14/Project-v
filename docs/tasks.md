# Project V: task list

Mirrors `mvp.md` §10 (Day 1 to 3) and §9 (A1 to A10). Status: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked.

## Step 0: GWIE
- [x] Get DePlag repo path (found on OneDrive Desktop, not D:)
- [x] Read original GWIE, verify it empirically (docs/gwie_verification), write `docs/gwie_original.md`
- [x] `tests/test_gwie.py` then `projectv/xai/gwie_c.py` + `gwie_orig_port`

## Day 1: Build, pilot, launch
- [x] D1.1 Scaffold (CLAUDE.md §4), venv, requirements.txt, pyproject.toml, .gitignore
- [!] D1.1 GitHub repo (need name + visibility), push
- [x] D1.1 Env sanity: env IDs, `obs_layout.py` asserts, `tests/test_envs.py` (CHECKPOINT 1), `requirements.lock`
- [x] D1.2 Wrappers: dr, latency, noise, history, shortcut + `tests/test_wrappers.py`
- [x] D1.2 Analytic controller, Reach nominal success 1.00 over 100 eps
- [x] D1.3 `train.py` + `callbacks.py` (ckpt 50k, replay 200k, `--resume auto`, run_meta.json); algo configs incl. TQC
- [!] D1.3 Pilot: fps measured (zoo Push config 1.4 steps/s on CPU, infeasible); needs decision + Kaggle GPU measurement (CHECKPOINT 2)
- [ ] D1.4 Kaggle notebook + kernel push for 9 Push + 6 Reach runs (CHECKPOINT 3)
- [x] D1.5 `policy_fn`, `groups`, `attributions`, `gwie_c` + `tests/test_attr_sanity.py`
- [ ] D1.5 C1 on analytic controller, first C1 table in `results/`
- [ ] Gate G1 decision (CHECKPOINT 4)

## Day 2: Evaluate + C1/C4
- [ ] D2.1 Reach runs done, shortcut experiment, C1 shortcut rows
- [ ] D2.2 Calibrate + freeze `configs/dr/heldout.yaml`, hash to `results/heldout_hash.txt` (CHECKPOINT)
- [ ] D2.3 Held-out sweeps + per-step logs (`results/steplogs/`)
- [ ] D2.4 C1 closed-loop protocol on push_dr, deletion curves
- [ ] D2.5 C4 reliance, temporal shares, probes (CHECKPOINT: C1 + C4 tables)
- [ ] Gate G2 decision

## Day 3: C3, stats, write-up
- [ ] D3.1 `tests/test_monitor.py`, ASM + 5 baselines, calib/test split, per-factor breakdown (CHECKPOINT)
- [ ] D3.2 Robustness table, rliable profiles, probability of improvement
- [ ] D3.3 make_tables, make_figs, make_gifs
- [ ] D3.4 README + report/report.md
- [ ] D3.5 Buffer: reruns, cleanup, HF Hub upload

## Acceptance criteria
- [ ] A1 pytest passes (wrappers, attribution sanity, GWIE, monitor)
- [ ] A2 push_dr nominal success >= 0.70 for >= 2/3 seeds (or fallback documented)
- [ ] A3 Held-out grid frozen with hash; every (policy, cell) has >= 100 episodes
- [ ] A4 C1 table: 7 methods x {controller, shortcut, push_dr}: tau, precision@k, deletion pAUC, CIs
- [ ] A5 C3 table: ASM vs 5 baselines, T in {10, 25}, AUROC/AUPRC/TPR@5%FPR, CIs, per-factor
- [ ] A6 C4: reliance fig, temporal-share fig, probe R^2 table with 3 controls
- [ ] A7 Robustness table: nominal + held-out, NoDR/DR/DR+Hist, CIs + per-seed
- [ ] A8 make_tables.py + make_figs.py regenerate everything from results/
- [ ] A9 3 GIFs: nominal success, held-out failure w/ ASM alarm, shortcut failure
- [ ] A10 README + report/report.md (~4 pages)
