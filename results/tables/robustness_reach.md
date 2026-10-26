| exp | split | n_cells | n_seeds | episodes_per_cell_min | success | per_seed |
|---|---|---|---|---|---|---|
| reach_nodr | nominal | 1 | 3 | 200 | 1.000 [1.000, 1.000] | 1.000 / 1.000 / 1.000 |
| reach_dr | nominal | 1 | 3 | 200 | 0.995 [0.985, 1.000] | 0.985 / 1.000 / 1.000 |
| reach_drhist | nominal | 1 | 3 | 200 | 1.000 [1.000, 1.000] | 1.000 / 1.000 / 1.000 |
| reach_nodr | heldout | 28 | 3 | 100 | 0.505 [0.497, 0.514] | 0.486 / 0.517 / 0.513 |
| reach_dr | heldout | 28 | 3 | 100 | 0.661 [0.652, 0.670] | 0.673 / 0.684 / 0.626 |
| reach_drhist | heldout | 28 | 3 | 100 | 0.740 [0.732, 0.748] | 0.718 / 0.739 / 0.763 |
| reach_nodr | factor:combined | 20 | 3 | 100 | 0.408 [0.397, 0.419] | 0.387 / 0.422 / 0.415 |
| reach_dr | factor:combined | 20 | 3 | 100 | 0.601 [0.592, 0.610] | 0.611 / 0.621 / 0.572 |
| reach_drhist | factor:combined | 20 | 3 | 100 | 0.642 [0.632, 0.652] | 0.619 / 0.639 / 0.668 |
| reach_nodr | factor:gain | 4 | 3 | 100 | 1.000 [1.000, 1.000] | 1.000 / 1.000 / 1.000 |
| reach_dr | factor:gain | 4 | 3 | 100 | 0.866 [0.838, 0.889] | 0.880 / 0.885 / 0.832 |
| reach_drhist | factor:gain | 4 | 3 | 100 | 1.000 [1.000, 1.000] | 1.000 / 1.000 / 1.000 |
| reach_nodr | factor:latency | 2 | 3 | 100 | 0.133 [0.098, 0.177] | 0.105 / 0.095 / 0.200 |
| reach_dr | factor:latency | 2 | 3 | 100 | 0.570 [0.473, 0.633] | 0.590 / 0.645 / 0.475 |
| reach_drhist | factor:latency | 2 | 3 | 100 | 0.948 [0.885, 1.000] | 0.885 / 0.960 / 1.000 |
| reach_nodr | factor:noise | 2 | 3 | 100 | 0.862 [0.832, 0.902] | 0.835 / 0.920 / 0.830 |
| reach_dr | factor:noise | 2 | 3 | 100 | 0.943 [0.925, 0.962] | 0.965 / 0.950 / 0.915 |
| reach_drhist | factor:noise | 2 | 3 | 100 | 0.992 [0.980, 1.000] | 0.980 / 0.995 / 1.000 |

Probability of improvement P(x > y) across seeds and cells (95% CI):

| x | y | split | n_cells | n_seeds_x | n_seeds_y | poi | ci_backend |
|---|---|---|---|---|---|---|---|
| reach_dr | reach_nodr | nominal | 1 | 3 | 3 | 0.333 [0.000, 0.500] | rliable |
| reach_drhist | reach_dr | nominal | 1 | 3 | 3 | 0.667 [0.500, 1.000] | rliable |
| reach_dr | reach_nodr | heldout | 28 | 3 | 3 | 0.887 [0.873, 0.893] | rliable |
| reach_drhist | reach_dr | heldout | 28 | 3 | 3 | 0.748 [0.708, 0.792] | rliable |
| reach_dr | reach_nodr | factor:combined | 20 | 3 | 3 | 1.000 [1.000, 1.000] | rliable |
| reach_drhist | reach_dr | factor:combined | 20 | 3 | 3 | 0.706 [0.653, 0.758] | rliable |
| reach_dr | reach_nodr | factor:gain | 4 | 3 | 3 | 0.250 [0.250, 0.250] | rliable |
| reach_drhist | reach_dr | factor:gain | 4 | 3 | 3 | 0.750 [0.750, 0.750] | rliable |
| reach_dr | reach_nodr | factor:latency | 2 | 3 | 3 | 1.000 [1.000, 1.000] | rliable |
| reach_drhist | reach_dr | factor:latency | 2 | 3 | 3 | 1.000 [1.000, 1.000] | rliable |
| reach_dr | reach_nodr | factor:noise | 2 | 3 | 3 | 0.917 [0.722, 1.000] | rliable |
| reach_drhist | reach_dr | factor:noise | 2 | 3 | 3 | 0.917 [0.750, 1.000] | rliable |
