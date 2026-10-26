| method | k | seed | success | erased | n_episodes |
|---|---|---|---|---|---|
| grad | 0 | 1 | 0.850 |  | 100 |
| grad | 1 | 1 | 0.710 | obj_velp | 100 |
| grad | 2 | 1 | 0.750 | obj_velp+ee_vel | 100 |
| grad | 3 | 1 | 0.070 | obj_velp+ee_vel+goal | 100 |
| grad | 4 | 1 | 0.020 | obj_velp+ee_vel+goal+obj_rel | 100 |
| grad | 5 | 1 | 0.040 | obj_velp+ee_vel+goal+obj_rel+obj_pos | 100 |
| grad_x_delta | 0 | 1 | 0.850 |  | 100 |
| grad_x_delta | 1 | 1 | 0.000 | obj_pos | 100 |
| grad_x_delta | 2 | 1 | 0.100 | obj_pos+goal | 100 |
| grad_x_delta | 3 | 1 | 0.050 | obj_pos+goal+obj_rel | 100 |
| grad_x_delta | 4 | 1 | 0.050 | obj_pos+goal+obj_rel+obj_rot | 100 |
| grad_x_delta | 5 | 1 | 0.040 | obj_pos+goal+obj_rel+obj_rot+ee_pos | 100 |
| ig | 0 | 1 | 0.850 |  | 100 |
| ig | 1 | 1 | 0.050 | obj_rel | 100 |
| ig | 2 | 1 | 0.020 | obj_rel+goal | 100 |
| ig | 3 | 1 | 0.050 | obj_rel+goal+obj_pos | 100 |
| ig | 4 | 1 | 0.050 | obj_rel+goal+obj_pos+ee_pos | 100 |
| ig | 5 | 1 | 0.040 | obj_rel+goal+obj_pos+ee_pos+obj_rot | 100 |
| smoothgrad | 0 | 1 | 0.850 |  | 100 |
| smoothgrad | 1 | 1 | 0.710 | obj_velp | 100 |
| smoothgrad | 2 | 1 | 0.750 | obj_velp+ee_vel | 100 |
| smoothgrad | 3 | 1 | 0.070 | obj_velp+ee_vel+goal | 100 |
| smoothgrad | 4 | 1 | 0.020 | obj_velp+ee_vel+goal+obj_rel | 100 |
| smoothgrad | 5 | 1 | 0.040 | obj_velp+ee_vel+goal+obj_rel+obj_pos | 100 |
| occlusion | 0 | 1 | 0.850 |  | 100 |
| occlusion | 1 | 1 | 0.000 | obj_pos | 100 |
| occlusion | 2 | 1 | 0.050 | obj_pos+obj_rel | 100 |
| occlusion | 3 | 1 | 0.050 | obj_pos+obj_rel+goal | 100 |
| occlusion | 4 | 1 | 0.050 | obj_pos+obj_rel+goal+ee_pos | 100 |
| occlusion | 5 | 1 | 0.050 | obj_pos+obj_rel+goal+ee_pos+ee_vel | 100 |
| kernelshap_grouped | 0 | 1 | 0.850 |  | 100 |
| kernelshap_grouped | 1 | 1 | 0.050 | obj_rel | 100 |
| kernelshap_grouped | 2 | 1 | 0.020 | obj_rel+goal | 100 |
| kernelshap_grouped | 3 | 1 | 0.050 | obj_rel+goal+obj_pos | 100 |
| kernelshap_grouped | 4 | 1 | 0.050 | obj_rel+goal+obj_pos+ee_pos | 100 |
| kernelshap_grouped | 5 | 1 | 0.040 | obj_rel+goal+obj_pos+ee_pos+obj_rot | 100 |
| gwie_c | 0 | 1 | 0.850 |  | 100 |
| gwie_c | 1 | 1 | 0.040 | goal | 100 |
| gwie_c | 2 | 1 | 0.020 | goal+obj_rel | 100 |
| gwie_c | 3 | 1 | 0.050 | goal+obj_rel+obj_pos | 100 |
| gwie_c | 4 | 1 | 0.050 | goal+obj_rel+obj_pos+ee_pos | 100 |
| gwie_c | 5 | 1 | 0.040 | goal+obj_rel+obj_pos+ee_pos+obj_rot | 100 |
| gwie_orig_port | 0 | 1 | 0.850 |  | 100 |
| gwie_orig_port | 1 | 1 | 0.050 | obj_rel | 100 |
| gwie_orig_port | 2 | 1 | 0.020 | obj_rel+goal | 100 |
| gwie_orig_port | 3 | 1 | 0.050 | obj_rel+goal+obj_pos | 100 |
| gwie_orig_port | 4 | 1 | 0.050 | obj_rel+goal+obj_pos+ee_pos | 100 |
| gwie_orig_port | 5 | 1 | 0.040 | obj_rel+goal+obj_pos+ee_pos+obj_rot | 100 |
| open_loop | 0 | 1 | 0.850 |  | 100 |
| open_loop | 1 | 1 | 0.040 | goal | 100 |
| open_loop | 2 | 1 | 0.100 | goal+obj_pos | 100 |
| open_loop | 3 | 1 | 0.050 | goal+obj_pos+obj_rel | 100 |
| open_loop | 4 | 1 | 0.050 | goal+obj_pos+obj_rel+obj_rot | 100 |
| open_loop | 5 | 1 | 0.040 | goal+obj_pos+obj_rel+obj_rot+ee_pos | 100 |
