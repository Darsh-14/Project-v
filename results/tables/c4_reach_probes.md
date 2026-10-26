| exp | target | features | r2 (pooled) | r2 per seed | n_seeds | claim_sysid | penult_beats_stacked |
|---|---|---|---|---|---|---|---|
| reach_nodr | gain | penult | 0.009 [-0.030, 0.038] | 0.024 / 0.002 / 0.002 | 3 | False | n/a |
| reach_nodr | gain | obs_single | -0.024 [-0.058, 0.007] | 0.003 / -0.037 / -0.037 | 3 | False | n/a |
| reach_nodr | gain | penult_random | -0.178 [-0.483, -0.032] | -0.110 / -0.039 / -0.386 | 3 | False | n/a |
| reach_dr | gain | penult | 0.070 [-0.054, 0.225] | -0.041 / 0.233 / 0.019 | 3 | False | n/a |
| reach_dr | gain | obs_single | -0.029 [-0.063, 0.001] | -0.004 / -0.042 / -0.040 | 3 | False | n/a |
| reach_dr | gain | penult_random | -0.089 [-0.148, -0.039] | -0.053 / -0.096 / -0.118 | 3 | False | n/a |
| reach_drhist | gain | penult | -0.402 [-1.174, -0.031] | -0.031 / -0.212 / -0.963 | 3 | False | False |
| reach_drhist | gain | obs_single | -0.028 [-0.062, 0.002] | -0.002 / -0.042 / -0.040 | 3 | False | False |
| reach_drhist | gain | obs_stacked | -0.034 [-0.066, -0.006] | -0.017 / -0.032 / -0.051 | 3 | False | False |
| reach_drhist | gain | penult_random | -0.053 [-0.093, -0.018] | -0.027 / -0.057 / -0.077 | 3 | False | False |

- Claim rule outcome (`claim_sysid`, results/c4_reach/probes.csv): reach_nodr/gain: **False**, reach_dr/gain: **False**, reach_drhist/gain: **False**.
- No implicit system identification is claimed for any (condition, target).
- Probe run code version (`run_meta_probes.json` git SHA): `9143745`.
