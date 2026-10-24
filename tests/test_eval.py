import numpy as np

from projectv.eval.cells import NOMINAL, Cell
from projectv.eval.rollout import run_episodes
from projectv.eval.steplog import load_steplog, save_steplog, steplog_path
from projectv.utils.config import load_exp
from projectv.xai.groups import flat_dim


def test_run_episodes_logs_interface_shapes(tiny_model, tmp_path):
    model = tiny_model("push_dr")
    exp = load_exp("configs/exp/push_dr.yaml")
    rec = run_episodes(model, exp, NOMINAL.dr_spec(), seeds=[11, 12], log=True)
    E, T, d = 2, 50, flat_dim("push")
    assert rec["obs_flat"].shape == (E, T, d) and rec["act_mean"].shape == (E, T, 4)
    assert rec["q_min"].shape == (E, T) and rec["penult"].shape[:2] == (E, T)
    assert rec["success"].shape == (E,) and rec["is_success"].shape == (E, T)
    np.testing.assert_allclose(rec["action"], np.tanh(rec["act_mean"]), atol=1e-6)
    p = steplog_path(tmp_path, "push_dr", 0, 3)
    save_steplog(p, rec, 3)
    back = load_steplog(p)
    assert int(back["cell_id"]) == 3 and back["obs_flat"].shape == (E, T, d)


def test_cell_params_reach_env(tiny_model):
    model = tiny_model("push_dr")
    exp = load_exp("configs/exp/push_dr.yaml")
    cell = Cell(5, "obj_mass=3.0", "mass", {"obj_mass": 3.0})
    rec = run_episodes(model, exp, cell.dr_spec(), seeds=[1])
    names = list(rec["dr_names"])
    assert rec["dr_values"][0, names.index("obj_mass")] == np.float32(3.0)
    assert rec["dr_values"][0, names.index("gain")] == np.float32(1.0)


def test_obs_hook_changes_policy_input_only(tiny_model):
    model = tiny_model("push_dr")
    exp = load_exp("configs/exp/push_dr.yaml")
    seen = []

    def hook(x, t, ep):
        seen.append(t)
        y = x.copy()
        y[-3:] = 0.0  # corrupt the goal the policy sees
        return y

    rec = run_episodes(model, exp, NOMINAL.dr_spec(), seeds=[5], obs_hook=hook, log=True)
    assert seen == list(range(50))
    assert np.all(rec["obs_flat"][0, :, -3:] == 0.0)


def test_rollouts_are_deterministic(tiny_model):
    model = tiny_model("push_dr")
    exp = load_exp("configs/exp/push_dr.yaml")
    a = run_episodes(model, exp, NOMINAL.dr_spec(), seeds=[7], log=True)
    b = run_episodes(model, exp, NOMINAL.dr_spec(), seeds=[7], log=True)
    np.testing.assert_array_equal(a["obs_flat"], b["obs_flat"])
