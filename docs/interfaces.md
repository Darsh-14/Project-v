# Internal interfaces (contract between modules)

Fixed on Day 1 so that eval, C1, C3, C4 and the output scripts can be built in parallel. If you
change something here, update every consumer in the same commit.

## Flat state vector
`x = [observation (D), achieved_goal (3), desired_goal (3)]` in raw units, float64 for
attribution and float32 in logs. Conversion lives only in `projectv/xai/policy_fn.py`
(`flatten_obs`, `split_flat`). Groups: `projectv/xai/groups.py::make_groups(task, history,
shortcut)`.

## Cells (`configs/dr/heldout.yaml`, written ONLY by `scripts/calibrate_heldout.py`)
```yaml
frozen: true
source_policy: results/ckpt/push_dr_s0/step_N.zip   # policy used for calibration
cells:
  - {id: 0, name: nominal, factor: nominal, params: {}}
  - {id: 1, name: "obj_friction=0.2", factor: friction, params: {obj_friction: 0.2}}
  - ...                                    # one-factor-at-a-time cells
  - {id: 21, name: combined_00, factor: combined, params: {obj_friction: 0.3, obj_mass: 3.0, ...}}
```
- `params` override nominal values (`configs/dr/none.yaml`); unspecified factors are nominal.
- `factor` ∈ {nominal, friction, mass, gain, noise, latency, combined}. `table_friction` and
  `obj_friction` cells both have factor `friction`.
- A cell's DR spec = `dr_wrapper.fixed_spec({**nominal, **params})`.
- Loader: `projectv/eval/cells.py::load_cells(path) -> list[Cell]` (dataclass: id, name,
  factor, params).
- After freezing, `results/heldout_hash.txt` holds the sha256 of the YAML
  (`utils.config.file_hash`). Every consumer asserts the hash matches before computing
  C3/C4 results.

## Episode seeds
`utils.seeding.eval_episode_seed(train_seed, cell_id, ep)`. Policies are evaluated
deterministically (action = tanh(mean), i.e. SB3 `predict(deterministic=True)`).

## Per-step logs (`results/steplogs/{exp}_s{seed}/cell{id:02d}.npz`), `projectv/eval/steplog.py`
Arrays (E episodes, T = 50 steps):

| key | shape | dtype | meaning |
|---|---|---|---|
| `obs_flat` | (E, T, d) | float32 | state at step t (before the action) |
| `act_mean` | (E, T, 4) | float32 | actor pre-tanh mean |
| `log_std` | (E, T, 4) | float32 | actor log std (clamped, as SB3) |
| `action` | (E, T, 4) | float32 | action issued = tanh(act_mean) |
| `q_min` | (E, T) | float32 | min over critics of Q(s, action) (TQC: mean over quantiles, min over critics) |
| `penult` | (E, T, H) | float32 | actor `latent_pi` activations |
| `is_success` | (E, T) | bool | `info["is_success"]` after the step |
| `success` | (E,) | bool | `is_success` at the final step |
| `dr_values` | (E, F) | float32 | DR params of the episode, order `dr_names` |
| `dr_names` | (F,) | str | factor names |
| `ep_seed` | (E,) | int64 | reset seed |
| `cell_id` | () | int | |

Plus `meta.json` next to the npz files: exp, seed, checkpoint path, config hash, heldout hash,
git SHA.

Nominal calib pool: `cell00.npz` with 200 episodes (mvp.md §4.3); the other cells have 100.
Splits for C3: `projectv/monitor/splits.py` assigns episodes to `calib`/`test`
deterministically by `ep_seed` (hash-based, 50/50 within each cell). Nothing is fit on `test`.

## Success tables
`results/eval/{exp}_s{seed}.csv` with columns `cell_id, cell_name, factor, episode, ep_seed,
success`.

## Checkpoints
`results/ckpt/{exp}_s{seed}_final.zip` (final), else the latest
`results/ckpt/{exp}_s{seed}/step_N.zip`. Resolver: `projectv/eval/rollout.py::resolve_checkpoint(exp,
seed, root="results")`. Load with `train.algo_class(name).load(path, device="cpu")`.

## Result files consumed by `scripts/make_tables.py`
| contribution | file |
|---|---|
| C1 controller | `results/c1/c1_{reach,push}_controller_summary.csv` |
| C1 shortcut | `results/c1/c1_shortcut_summary.csv` |
| C1 push | `results/c1/c1_push_dr_summary.csv`, `results/c1/c1_push_dr_deletion.csv` |
| C2 runtime | the `runtime_ms` columns in the C1 summaries |
| C3 | `results/c3/c3_{exp}_metrics.csv`, `results/c3/c3_{exp}_per_factor.csv` |
| C4 | `results/c4/reliance.csv`, `results/c4/temporal.csv`, `results/c4/probes.csv` |
| Robustness | `results/robustness/robustness.csv` (built from `results/eval/*.csv`) |

Every CSV has CI columns `{metric}_lo`, `{metric}_hi` where applicable, and every directory has
a `run_meta.json` (`utils.io.write_run_meta`).

## Column schemas of result CSVs (pinned after the module merge)
- `c3_{exp}_metrics.csv`: `exp, signal, T, alpha, n_test, n_fail_test, threshold, fpr_test`,
  `auroc|auprc|tpr_at_5fpr|det_step_median` each with `_lo/_hi`, per-seed `auroc_s{n}`.
  Signals: `asm, obs_ood, act_std, q_drop, ensemble, act_ood`.
- `c3_{exp}_per_factor.csv`: `exp, signal, T, factor, auroc, auroc_lo, auroc_hi, ...`.
- `c3_{exp}_episode_scores.csv` (git-ignored, large): one row per (signal, seed, episode):
  `signal, seed, cell_id, factor, split (calib|test), ep_seed, failed, alpha, thr_T10, S10,
  alarm_T10, thr_T25, S25, alarm_T25, threshold (= thr_T25), e_0..e_49` (EWMA scores).
- ASM groups: 10 for Push; for push_drhist the slots are merged into base groups and the past
  actions form one `act` group (11) (`monitor/asm.py::monitor_groups`).
- `c1_push_dr_deletion.csv`: `method, k, seed, success, erased`. Reliance:
  `condition, method, level, group, share, ...`. Robustness: `exp, split, success[_lo,_hi],
  success_s{n}, ci_backend`. Consumers accept the aliases listed in `scripts/make_tables.py`.

## Test fixture
`tests/conftest.py::tiny_model(exp_name)` returns a SAC model with a [16] net, trained for 300
steps on that experiment's env. Modules are tested on it, never on Kaggle checkpoints.
