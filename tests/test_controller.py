import numpy as np

from projectv.control.analytic_controller import PDController
from projectv.envs.dr_wrapper import fixed_spec
from projectv.envs.make_env import make_env
from projectv.utils.seeding import eval_episode_seed


def test_reach_nominal_success_at_least_095():
    env = make_env({"task": "reach", "dr_cfg": {}}, seed=0, dr_spec=fixed_spec({}))
    ctrl = PDController("reach")
    succ = []
    for ep in range(100):
        o, _ = env.reset(seed=eval_episode_seed(0, 0, ep))
        for _ in range(50):
            o, _, _, _, info = env.step(ctrl.act(o))
        succ.append(info["is_success"])
    assert np.mean(succ) >= 0.95


def test_contributions_are_exact():
    rng = np.random.default_rng(0)
    ctrl = PDController("push")

    def rand_obs():
        return {"observation": rng.normal(size=25), "desired_goal": rng.normal(size=3)}

    obs, base = rand_obs(), rand_obs()
    total = sum(c.sum(axis=0) for c in ctrl.contributions(obs, base).values())
    np.testing.assert_allclose(total, ctrl.mean(obs)[:3] - ctrl.mean(base)[:3], atol=1e-12)
