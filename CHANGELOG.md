# CHANGELOG

Deviations, fallbacks and decisions, newest first.

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
