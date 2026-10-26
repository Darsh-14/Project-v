# CHANGELOG

Deviations, fallbacks and decisions, newest first.

## 2026-10-26 (Day 3)

- **Push TQC training ended** (`pv-push-tqc-r1`, 7.5 h wall-clock cap, ~759k steps per run, no
  `_final.zip`; the latest `step_*.zip` is used). Nominal eval success at the last checkpoint
  (20 episodes, `results/train_logs/push_*`): push_dr s1 0.75, push_drhist s0 0.85,
  push_nodr s0 0.90, push_dr s0 0.05 (flat for the whole run, as under SAC).
- **A2 not met; Gate G2 applies.** push_dr reaches >= 0.70 for 1 of 2 trained seeds (seed 2 was
  never trained, fallback rung 4). Rung 2 (dense-reward SAC) is not attempted: the GPU quota for
  the week is used up. The Reach track (already computed) carries C3/C4; Push results are
  reported as **partial**, n = 1 learned seed per condition, with push_dr s0 shown as a failed
  seed wherever per-seed values are printed. No Push number is extrapolated to n = 3.
- **Push held-out grid calibrated on push_dr s1, not s0** (mvp.md §10 Day 2 block 2 names s0,
  which did not learn). Frozen: `configs/dr/heldout.yaml`, hash `6e9b1e89…` in
  `results/heldout_hash.txt`, log `results/calibration/calibration_log.csv`. Nominal success of
  the source policy over 50 calibration episodes: 0.80.
  - Kept and flagged (no value in the 20–80% band, the policy is robust there): obj_friction 2.0,
    table_friction 1.8, obj_mass 3.0 and 4.0, gain 0.5 and 0.6.
  - Latency candidate 3 was too hard and bisected to 2, so the grid has two latency-2 cells with
    identical parameters (different episode seeds). Kept as calibrated; the grid is not edited.
- Calibration `run_meta` is now written per grid (`calibration_log[_reach]_run_meta.json`); the
  Reach file was renamed, its contents are unchanged.
- **Push sweeps run locally** (evaluation only, as for Reach), not on Kaggle.
- **BUG: every held-out sweep wrote the same `results/eval/run_meta.json`.** The four parallel
  Push sweeps overwrote each other and the Reach meta. Fixed: one file per sweep
  (`run_meta_{exp}_s{seeds}.json`). The Reach file was restored from git and renamed; of the Push
  metas only push_drhist s0's survived. The other three sweeps' provenance (git SHA, checkpoint
  sha256, grid hash) is still recorded in each `results/steplogs/{exp}_s{seed}/meta.json`.
- **Learned-policy rule for C1/C3/C4 on Push.** push_dr s0 (nominal success 0.05) is left out of
  C1, C3 and C4 (`--skip push_dr:0` in reliance/probes, `--seeds 1` for C1/C3). Explaining or
  monitoring a policy that fails almost every nominal episode measures nothing about the method,
  and mixing it into a one-seed-per-condition mean would make the DR row about training failure.
  It is kept in the robustness table, where a failed seed is a real outcome of training with DR.

## 2026-10-25 (Day 2)

- **BUG: DR parameters were drawn from the env's own random stream (probe leak).**
  - `DRWrapper` seeded its RNG with `default_rng(seed)`, which is the same stream Gymnasium
    uses for the env's `np_random` on a seeded reset. The first DR draw equalled the env's first
    draw: on Reach, gain was an exact function of the goal's x offset (corr 1.0); on Push,
    obj_friction tracked the object's x position.
  - Found because the first Reach probe run gave gain R² = 1.000 from a single raw observation
    *and* from a random-init network.
  - Fixed: `dr_rng` uses `SeedSequence([seed, tag])`. A regression test asserts |corr| < 0.3
    between every DR parameter and every initial-state feature over 300 seeded resets (it fails
    on the old code for both Reach and Push).
  - Impact. Probes are invalid: they use per-episode-seeded train-DR rollouts. The Reach probe
    results were deleted and rerun; Push probes will only ever run with the fix.
  - Not affected: held-out sweeps, calibration, C1, C3, robustness. Every cell there fixes the DR
    values, so no DR draws occur and existing results are reproducible bit-for-bit.
  - Training: only the first reset is seeded. After it the DR and env streams consume different
    numbers of draws per episode and desynchronise. Training runs (including the TQC runs in
    flight on Kaggle with the old code) are kept, and this is noted as a limitation.
- **Reach G2 track computed** (3 seeds per condition; `results/robustness_reach/`,
  `results/c3/c3_reach_*`, `results/c4_reach/`).
  - Held-out success: NoDR 0.505 [0.497, 0.514], DR 0.661 [0.652, 0.670], DR+Hist 0.740
    [0.732, 0.748].
  - Probability of improvement on held-out: DR > NoDR 0.887, DR+Hist > DR 0.748.
  - C3 AUROC at T=10: ASM 0.629 / 0.593 / 0.673 (NoDR / DR / DR+Hist). The best baseline is
    act_ood 0.720 / 0.818 and obs_ood 0.668 (DR+Hist), so ASM does not beat the best baseline on
    Reach DR or NoDR and roughly ties on DR+Hist. Reported as is.
  - Probes, rerun after the RNG fix: gain R² ≈ 0 for every feature set.
    - Trained penultimate activations: 0.009 / 0.070 / −0.402 (NoDR / DR / DR+Hist).
    - Raw single obs ≈ −0.03; raw stacked obs −0.034; random-init activations < 0.
    - No implicit system identification of gain is claimed on Reach. Gain is identifiable only
      as a ratio (displacement / (0.05·action)), which a linear probe cannot represent.

- **Held-out grid hash made line-ending agnostic (portability bug).** The freeze hash was a
  byte-exact sha256 of the YAML. Windows writes CRLF and git stores LF, so a Kaggle/Colab clone
  hashed `heldout_reach.yaml` to `860a59…`, not the recorded `1ecade…`, and `assert_frozen`
  would refuse every off-machine run (the Push grid would be affected the same way).
  - Grid files are now hashed with `utils.config.text_hash` (CRLF → LF before sha256) and always
    written with LF. Checkpoints keep byte-exact `file_hash`.
  - The Reach grid's *values* are unchanged. Only the recorded fingerprint is re-computed with
    the new function (`860a59…`), and the step-log `meta.json` `heldout_hash` fields of the
    reach sweeps are migrated to it. This is not an edit of the frozen grid.
- **Reach G2 track.** `configs/dr/heldout_reach.yaml` was calibrated on reach_dr s0
  (`results/calibration/calibration_log_reach.csv`). The low-gain cells (0.5, 0.6) cannot be
  brought into the 20–80% band (Reach is robust to them, success 1.00); they are kept and
  flagged, not dropped.
  - Held-out success for reach_dr spans 0.04–1.00, with failures concentrated in gain 1.8,
    latency 3 and half of the combined cells (`results/eval/reach_dr_s*.csv`).
- **TQC resume (`pv-push-tqc-r1` v3)** passed the resume guard (still running 15 min in) after
  the input path fix.

- **CORRECTION to the G1 entry below.** The SAC session P1 finished (11 h, ~933k steps).
  push_dr s1 *did* learn: 0.50 nominal success at 150k, 0.95 at 600k–934k. push_dr s0 and
  push_drhist s0/s1 stayed at 0.05 (`results/archive/sac_her_p1/train_logs/`). So SAC+HER
  learned in 1 of 6 Push runs, not 0 of 6 as stated below. The G1 rule still triggers (any Push
  run not learning → switch all to TQC), and all reported Push results use TQC for every
  condition. The SAC runs are archived evidence only.
- **Reach NoDR / DR+Hist / shortcut v2 done** (Kaggle CPU, 100k steps): every seed ends at 1.00
  nominal success (20 eval episodes).
- **Resume bug confirmed.** The `reach_dr` s0 resume test in `pv-reach-nodr-r0` silently
  retrained from scratch: attached outputs mount under `/kaggle/input/notebooks/...`, deeper
  than the copy step searched. That duplicate is discarded (the original `reach_dr` s0 is kept).
  The notebook now searches recursively, and `RESUME_REQUIRED` caught the same bug in the first
  TQC resume attempt within 2 minutes.

- **TQC+HER probe (4 h, ~258k steps per run, ~18 steps/s per run on T4×2 with 4 runs).**
  Nominal eval success / train success at the end:
  - push_dr s1: 0.15 / 0.40, rising;
  - push_drhist s0: 0.40 / 0.56, rising;
  - push_dr s0: 0.05 / 0.05, flat;
  - push_nodr s0: 0.00 / 0.04, flat.

  TQC learns Push where SAC never did (2 of 4 runs by 258k). All 4 runs are resumed unchanged
  in `pv-push-tqc-r1` (7.5 h cap, remaining GPU quota). Seeds are not replaced: swapping the
  flat seed would be cherry-picking.
- **Resume guard.** Resume sessions now assert that every run's checkpoint is present after the
  input copy (`RESUME_REQUIRED`), so a broken input path fails in minutes instead of silently
  retraining from step 0.
- **Local machine stall.** A ~3 h test run (normally ~2 min) and a stalled agent indicate the
  dev machine slept or hung between ~06:00 and 18:30 IST. Kaggle sessions were unaffected. The
  TQC resume launched ~18:50 IST, about 9 h later than it could have.

- **Reach training done** (Kaggle CPU, 100k steps). Final nominal success: reach_dr
  0.95 / 1.00 / 1.00 and reach_shortcut 1.00 / 1.00 / 1.00 for seeds 0/1/2 (20 eval episodes per
  checkpoint; `results/train_logs/`).
- **Gate G1 FAILED for SAC+HER on Push → fallback rung 1 (TQC+HER).**
  - push_nodr s0/s1 trained the full 1M steps (zoo config, T4): nominal success stayed at
    0.05–0.10 throughout, about the "object already at goal" chance level
    (`results/archive/sac_her/train_logs/`).
  - Diagnosis on the final s0 policy:
    - entropy coefficient collapsed to 1.9e-4;
    - deterministic bang-bang actions (mean |a| ≈ 0.8 on every axis);
    - closest gripper–object approach 0.115 m, so it never reaches the object;
    - the gripper is driven off the table edge.
  - The wrapped env is identical to stock FetchPush (same gripper limits under fixed action
    sequences), so this is an algorithm failure, not an env bug. This is consistent with
    rl-baselines3-zoo using TQC for Push.
  - All Push runs switch to `configs/algo/tqc_her_push.yaml` (same buffer/HER/network
    settings). The SAC session P1 (push_dr, push_drhist) could not be cancelled from the CLI and
    runs to its 11 h cap. Its curves are archived as SAC evidence and are never used as results.
- **GPU quota plan.** About 18 of ~30 weekly GPU-hours are used or committed. A 4 h TQC probe
  session trains push_dr s0, s1, push_nodr s0 and push_drhist s0 (priority: A2 needs two
  push_dr seeds), then resumes toward 1M if learning. Second seeds of NoDR/DR+Hist only if
  quota remains. n per condition will be stated.
- **G2 insurance (CPU, no GPU quota).** reach_nodr and reach_drhist × seeds 0–2 are trained so
  that C3/C4 can fall back to Reach (gain/noise/latency DR) if Push A2 fails by end of Day 2.
- **Planted shortcut not learned under HER → shortcut v2 (deviation from mvp.md §5).**
  - With reach_shortcut (SAC+HER, hint = goal + N(0, 0.002²)), replacing the hint by a random
    point leaves success at 1.00 for all 3 seeds (drop 0.0). Every method gives the hint a small
    share (0.00–0.05 vs goal 0.28–0.65; `results/c1/c1_shortcut_her_summary.csv`).
  - Cause: HER relabels ~80% of sampled transitions with achieved goals. In those, the stored
    goal-copy hint contradicts `desired_goal`, so the policy learns to ignore it. A goal-copy
    hint cannot be a shortcut under HER. This is kept as a documented finding (a valid negative
    control: true reliance ≈ 0, and attributions agree).
  - Ground truth B now uses `reach_shortcut_v2`: SAC **without HER** on `FetchReachDense-v4`,
    same hint, and the policy's `desired_goal` input gets N(0, 0.03²) noise (policy input only;
    reward and success use the true goal), so the precise hint is the easier signal. True
    reliance is still *measured* (success drop with a random hint), not assumed.
  - Outputs: `c1_shortcut_*` = v2 (primary), `c1_shortcut_her_*` = HER variant.
- **Checkpoint loading bug fixed.** `SAC.load` silently accepts a TQC zip. `load_model` now
  reads the algorithm from the checkpoint metadata (`checkpoint_algo`), with a regression test.

## 2026-10-24 (Day 1)

- **Day-2/3 modules merged** (built in parallel by agents against `docs/interfaces.md`,
  reviewed and replayed onto `main`, 115 tests pass). Key protocol decisions:
  - *C1 closed loop:* at every step the erased group takes the values of one fresh pool row
    (M=1, since one action must be issued). Jointly erased groups share that row. Comparisons
    are paired, using the same row sequence and episode seeds for every erasure unit.
    - τ, precision@k (k=3) and pAUC are computed per seed, then averaged. pAUC is the mean over
      k=1..5 of (nominal − success_k)/nominal.
    - The deletion sets are fixed per seed, so the bootstrap does not re-rank.
    - Attribution/episode seeds are offset by 5,000,000 from evaluation seeds.
  - *C3:*
    - Calib/test split by sha256(ep_seed), 50/50 within each (seed, cell).
    - All nominal statistics are fit on calib cell 0 only. obs/activation Mahalanobis use
      Ledoit-Wolf on z-scored features (the shrinkage target is not scale invariant).
    - α is chosen per signal on calib AUROC of S(25) from {0.1, 0.2, 0.3, 0.5, 1.0}.
    - The TPR@5%FPR threshold is the 95th percentile of S(T) over calib successes.
    - `ensemble` is the std across seed actors (needs ≥ 2 seeds).
  - *C4:*
    - Entropy in nats and normalised by log(#groups) (10 groups for NoDR/DR, 11 for DR+Hist).
    - Probes use train-DR rollouts for every condition (also NoDR, outside its own training
      distribution), with an episode-grouped 5-fold CV ridge and an inner α search.
    - The sysID claim needs the trained-activation CI to be above both the raw-obs and the
      random-init CIs.
  - *Robustness:* `rliable` stratified bootstrap (strata = cells). It needs `pandas<3` and
    `arch<8`, now pinned; otherwise a hand-rolled identical resampler is used and recorded in
    `ci_backend`.
  - *Outputs:* headline numbers come from `robustness.csv` (X, Y) and the C3 metrics (Z = ASM
    AUROC at T=10, W = the best of the 5 baselines at T=10, name recorded). The headline is
    phrased neutrally (no "DR raised success" unless the numbers show it).

- **Held-out sweep and calibration** (`projectv/eval/heldout_sweep.py`,
  `scripts/calibrate_heldout.py`):
  - Cell 0 gets max(episodes, 200) episodes for every experiment (the calib pool).
  - Each step log is written before its CSV rows, and a step-log folder whose `meta.json`
    names a different held-out hash or checkpoint is refused.
  - Calibration episodes use reset seeds 5,000,000 + ep (disjoint from evaluation), the same
    episodes for every candidate. Values are rounded to 3 significant figures (latency to an
    integer) and evaluated after rounding.
  - Search only between the train-range boundary and the candidate. A cell that cannot reach
    [0.2, 0.8] keeps its closest value and is flagged in `results/calibration/`. It is never
    dropped silently.
  - Combined cells put every factor outside its train range, sampled from the calibrated
    interval on a random side, and are not re-tuned. Out-of-band cells are flagged.
- **`rollout.load_model`** swaps the HER replay buffer for a 1-slot plain buffer, because SB3
  cannot rebuild a HerReplayBuffer without an env and the old fallback hid that error.
- **Episode-seed collisions (known, kept).** CLAUDE.md §7's `10_000*seed + 1_000*cell + ep`
  repeats reset seeds across training seeds once cell id ≥ 10, e.g. (seed 0, cell 10) and
  (seed 1, cell 0). These are different policies and cells, so no estimate is biased; episodes
  are just not globally unique. The formula is kept as specified.

- **Gate G1 timing.** Kaggle exposes a kernel's output only after the session ends, and the CLI
  cannot cancel a running kernel. The first Push sessions (11 h cap) therefore give their
  first learning-curve evidence when they end (~950k–1M steps), not after 50k steps. G1 is
  judged then. If SAC+HER is not learning, rung 1 (TQC+HER) costs one more session. Future
  first sessions should use a short `--max-hours` probe to get earlier visibility.

- **Push compute decision (Checkpoint 2, option A1 + fallback 4 early).** Kaggle measurements
  (`results/pilot/fps_kaggle_{gpu,cpu}.csv`):
  - zoo config on a T4: 47 steps/s for 1 run per GPU; on T4×2, ~24.5 each with 4 runs
    (~97 per session);
  - small config (256, [256]³) on Kaggle CPU: ~16 each with 3 runs;
  - zoo config on Kaggle CPU: 1.3.

  Push keeps the mvp.md §4.1 hyperparameters (SAC+HER, batch 2048, [512]³) and trains on
  Kaggle T4×2. 9 runs × 1M steps need ≥26 GPU-session hours even perfectly packed (more with
  resumes), which reaches the ~30 h budget and the weekly GPU quota. Per mvp.md §11, fallback
  step 4 is applied early: Push seeds {0, 1} (n = 2, stated explicitly in every table). Seed 2
  is launched only if quota remains, and is reported only if it completes.

- **Kaggle image** (verified by the pilot kernels): Python 3.13.15, torch 2.11, numpy 2.1.3,
  4 vCPU / 31 GB RAM, GPU sessions with 2× Tesla T4. Local dev is also Python 3.13, so the
  earlier 3.10/3.11 note no longer applies.
- **Kaggle compat shim** (`projectv/envs/compat.py`). On the Kaggle image,
  gymnasium-robotics' joint helpers mis-compare `model.jnt_type` (numpy int32) with
  `mjtJoint.mjJNT_FREE`, so `FetchPush-v4` fails at construction. A fresh `MjModel` shows
  correct joint types, so the model itself is fine. The shim hands those helpers Python-int
  joint types and is a no-op where the comparison already works.
- **Compute (Checkpoint 2).** The zoo Push config (batch 2048, [512]³, 1 update/step) runs at
  1.5 env-steps/s on one local CPU core (`results/pilot/fps.csv`), about 185 h per 1M steps.
  Option A, approved by the user: measure on a Kaggle T4 and a Kaggle CPU, then choose between
  the zoo config on GPU and batch 256 / [256]³ on CPU.
- **C1 Spearman.** Group scores below 1e-9 of the state total are set to 0 before Spearman, so
  least-squares noise (~1e-15, KernelSHAP) does not rank irrelevant groups against the exact
  zero ties of the ground truth. Applied to all methods equally.
- **Commit attribution.** At the user's request, no commit carries a Claude co-author
  trailer. The 18 earlier commits were rewritten to strip it (message-only rewrite, trees
  identical) and force-pushed with the user's approval. Commit SHAs before `39d0e83` changed.
  Kaggle kernels launched earlier pin pre-rewrite SHAs; resume sessions use the new ones.
- **GWIE Step 0.** The original GWIE was found in DePlag on the OneDrive Desktop, not on D:
  (`explainability/gwie.py`, HEAD `5ba9fcb`). As the user asked, it was verified *before*
  porting: the unmodified code ran with the deployed fine-tuned SBERT on 40 pairs
  (`docs/gwie_verification/`). Results:
  - IG is correct.
  - The search beats random erasure (0.071 vs 0.029 similarity drop at k=1) and reaches
    84–94% of a brute-force greedy oracle.
  - The paper's "monotonically non-increasing" claim is false (8.4% of erasure steps raised
    similarity), and convergence is not guaranteed (33/40 converged).

  GWIE is a counterfactual-set search, not a full attribution method, so the CLAUDE.md
  pseudocode is a non-trivial deviation. Both `gwie_orig_port` (direct port) and `gwie_c` (full
  ranking + KL gains) are implemented. The deviations table is in `docs/gwie_original.md`.
  `gwie_c` scores pivots with IG on the current erased state, as the original does, not with
  CLAUDE.md's provisional grad×delta.

- **Friction DR implementation.** MuJoCo combines contact friction as the element-wise max of
  the two geoms. Scaling only the object geom (mvp.md §4.2 literally) would make low-friction
  cells (0.2, 0.3) no-ops against the table at 1.0. Both the object geom and the (unnamed)
  table geom get sliding friction `1.0 * m_obj_friction * m_table_friction`, so the
  object–table contact uses exactly that product. Side effect: gripper–object and gripper–table
  friction become `max(1, product)`. No geom priority is used, because a priority would also
  change the nominal finger contacts (condim 4 → 3). Under nominal multipliers the physics is
  bit-identical to the stock env. `tests/test_wrappers.py` checks the physical effect: slide
  distance at 0.2 > 1.5× nominal > distance at 2.5.
- **Mass DR** scales the object's body mass *and* inertia (uniform density).
- **Gain DR** is applied after the env's [-1, 1] action clip, by wrapping the env's
  `_set_action`, so the mocap displacement is `0.05 * gain * a`. A pre-clip multiplier would
  saturate on SAC's near-±1 actions. This is a mocap step-size multiplier, not an actuator gain
  (CLAUDE.md §6).
- **Obs noise** is added to `observation`, and `achieved_goal` gets the same sample for its
  duplicated slice. Otherwise the policy could read a clean copy of the noisy position.
  `desired_goal` and `info["is_success"]` stay clean.
- **Reach DR** keeps only gain, obs_noise and latency. Friction and mass have no object to act on.
- **SB3 algo configs.** The current rl-baselines3-zoo has no SAC FetchPush entry. SAC Push
  uses the zoo's TQC FetchPush-v1 values (identical to mvp.md §4.1). The zoo's Reach config uses
  VecNormalize; it is dropped so attributions are in raw observation units (CLAUDE.md §8). The
  zoo's legacy `her.yml` also used a TimeFeatureWrapper for Push; it is not used here because the
  episode length is fixed at 50.
- **Env IDs.** `FetchReach-v4` and `FetchPush-v4` (highest registered).

- **Local Python is 3.13** (only interpreter installed). CLAUDE.md §3 asks for 3.10/3.11 to match
  Kaggle. Local venv is for development and tests only; all training runs on Kaggle with the image's
  Python. `requirements.lock` pins only project packages, not torch/numpy, so it installs on the
  Kaggle image.
