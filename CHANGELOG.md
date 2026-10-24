# CHANGELOG

Deviations, fallbacks and decisions, newest first.

## 2026-10-24 (Day 1)

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
