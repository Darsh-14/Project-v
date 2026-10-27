# Project V: task list

Mirrors `mvp.md` §10 (Day 1 to 3) and §9 (A1 to A10). Status: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked.

## Step 0: GWIE
- [x] Get DePlag repo path (found on OneDrive Desktop, not D:)
- [x] Read original GWIE, verify it empirically (docs/gwie_verification), write `docs/gwie_original.md`
- [x] `tests/test_gwie.py` then `projectv/xai/gwie_c.py` + `gwie_orig_port`

## Day 1: Build, pilot, launch
- [x] D1.1 Scaffold (CLAUDE.md §4), venv, requirements.txt, pyproject.toml, .gitignore
- [x] D1.1 GitHub repo `Darsh-14/Project-v` (public), pushed
- [x] D1.1 Env sanity: env IDs, `obs_layout.py` asserts, `tests/test_envs.py` (CHECKPOINT 1), `requirements.lock`
- [x] D1.2 Wrappers: dr, latency, noise, history, shortcut + `tests/test_wrappers.py`
- [x] D1.2 Analytic controller, Reach nominal success 1.00 over 100 eps
- [x] D1.3 `train.py` + `callbacks.py` (ckpt 50k, replay 200k, `--resume auto`, run_meta.json); algo configs incl. TQC
- [x] D1.3 Pilot: fps measured (zoo Push config 1.4 steps/s on CPU, infeasible); needs decision + Kaggle GPU measurement (CHECKPOINT 2)
- [x] D1.4 Kaggle notebook + kernels launched: Reach 6 runs (CPU), Push seeds {0,1} 6 runs (T4x2) (CHECKPOINT 3)
- [x] D1.5 `policy_fn`, `groups`, `attributions`, `gwie_c` + `tests/test_attr_sanity.py`
- [x] D1.5 C1 on analytic controller, first C1 table in `results/c1/`
- [x] Gate G1: SAC+HER learned 1/6 Push runs -> rung 1, TQC+HER (CHANGELOG 2026-10-25)

## Day 2: Evaluate + C1/C4
- [x] D2.1 Reach runs done (all 15 Reach policies), shortcut v2 experiment, C1 shortcut rows
- [x] D2.2 Calibrate + freeze `configs/dr/heldout.yaml` (on push_dr s1; s0 did not learn), hash in `results/heldout_hash.txt`
- [x] D2.3 Held-out sweeps + per-step logs (4 Push policies, 38 cells, >= 100 eps)
- [x] D2.4 C1 closed-loop protocol on push_dr s1, deletion curves
- [x] D2.5 C4 reliance, temporal shares, probes (Push + Reach)
- [x] Gate G2: Reach track computed as insurance (3 seeds). A2 was later met with push_dr s2 (2026-10-27), so Push is reported in full with 2 learned seeds per condition

## Day 3: C3, stats, write-up
- [x] D3.1 ASM + 5 baselines, calib/test split, per-factor breakdown
- [x] D3.2 Robustness table, rliable CIs, probability of improvement
- [x] D3.3 make_tables, make_figs, make_gifs
- [x] D3.4 README + report/report.md filled from results/
- [ ] D3.5 HF Hub checkpoint upload (needs the user's HF token)

## Acceptance criteria
- [x] A1 pytest passes (135 tests: wrappers, attribution sanity, GWIE, monitor, outputs, ...)
- [x] A2 push_dr nominal success >= 0.70 for 2 of 3 seeds (s1 0.850, s2 0.835 over 200 eps; s0 0.055), TQC+HER per the fallback ladder
- [x] A3 Grids frozen with hashes (Push 6e9b1e89…, Reach 860a5913…); >= 100 eps per (policy, cell)
- [x] A4 C1 tables: controller, shortcut, push_dr (learned seeds 1, 2)
- [x] A5 C3 tables: ASM vs 5 baselines, T in {10, 25}, CIs, per-factor (Push: 2 learned seeds per condition; Reach: 3)
- [x] A6 C4 reliance + temporal figs, probe tables with 3 controls (Push + Reach)
- [x] A7 Robustness tables (Push + Reach), CIs + per-seed
- [x] A8 make_tables.py + make_figs.py regenerate everything from results/
- [x] A9 3 GIFs in results/gifs/
- [x] A10 README + report/report.md (text ~4 pages plus generated tables)
