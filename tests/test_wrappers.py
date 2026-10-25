import gymnasium as gym
import mujoco
import numpy as np
import pytest

from projectv.envs.dr_wrapper import DRWrapper, fixed_spec
from projectv.envs.latency_wrapper import LatencyWrapper
from projectv.envs.make_env import make_env
from projectv.envs.obs_layout import ENV_IDS, PUSH_LAYOUT
from projectv.utils.config import load_exp

ZERO = np.zeros(4, dtype=np.float32)


def push_env(**fixed):
    return make_env(load_exp("configs/exp/push_dr.yaml"), seed=0, dr_spec=fixed_spec(fixed))


def _ids(env):
    m = env.unwrapped.model
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object0")
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0")
    t = np.flatnonzero(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table0"))
    return m, g, b, t


# --- DR: parameters reach the model ------------------------------------------------------------
def test_dr_sets_geom_friction_and_body_mass():
    env = push_env(obj_friction=0.3, table_friction=2.0, obj_mass=3.0)
    env.reset(seed=0)
    m, g, b, t = _ids(env)
    assert m.geom_friction[g, 0] == pytest.approx(0.6)
    assert m.geom_friction[t, 0] == pytest.approx(0.6)
    assert m.body_mass[b] == pytest.approx(6.0)
    # nominal values are restored when the spec goes back to nominal
    dr = env
    while not isinstance(dr, DRWrapper):
        dr = dr.env
    dr.dr_spec = fixed_spec({"obj_friction": 1.0, "table_friction": 1.0, "obj_mass": 1.0})
    env.reset(seed=1)
    assert m.geom_friction[g, 0] == pytest.approx(1.0) and m.body_mass[b] == pytest.approx(2.0)


def test_dr_resamples_each_reset_within_train_ranges():
    exp = load_exp("configs/exp/push_dr.yaml")
    env = make_env(exp, seed=0)
    seen = []
    for i in range(20):
        _, info = env.reset(seed=100 + i)
        p = info["dr_params"]
        assert 0.5 <= p["obj_friction"] <= 1.5 and 0.5 <= p["obj_mass"] <= 2.0
        assert 0.8 <= p["gain"] <= 1.2 and 0 <= p["obs_noise"] <= 0.005
        assert p["latency"] in (0, 1)
        seen.append(p["obj_friction"])
    assert len(set(seen)) == 20
    _, _, _, _, info = env.step(ZERO)
    assert info["dr_params"]["obj_friction"] == seen[-1]


def test_reach_drops_object_factors():
    env = make_env(load_exp("configs/exp/reach_dr.yaml"), seed=0)
    _, info = env.reset(seed=0)
    assert set(info["dr_params"]) == {"gain", "obs_noise", "latency"}


# --- DR: parameters have physical effect ------------------------------------------------------
def _slide_distance(friction):
    env = push_env(obj_friction=friction)
    env.reset(seed=0)
    u = env.unwrapped
    j = u.model.joint("object0:joint")
    u.data.qvel[j.dofadr[0] : j.dofadr[0] + 2] = [0.0, 0.6]  # slide sideways along +y
    mujoco.mj_forward(u.model, u.data)
    p0 = u.data.site("object0").xpos.copy()
    for _ in range(5):
        env.step(ZERO)
    return np.linalg.norm(u.data.site("object0").xpos[:2] - p0[:2])


def test_low_friction_slides_further():
    d_low, d_nom, d_high = _slide_distance(0.2), _slide_distance(1.0), _slide_distance(2.5)
    assert d_low > 1.5 * d_nom > 0 and d_nom > d_high


def _lift_height(mass):
    env = push_env(obj_mass=mass)
    env.reset(seed=0)
    u = env.unwrapped
    _, _, b, _ = _ids(env)
    z0 = u.data.site("object0").xpos[2]
    u.data.xfrc_applied[b, 2] = 1.5 * 2.0 * 9.81  # 1.5x nominal weight, upward
    for _ in range(3):
        env.step(ZERO)
    return u.data.site("object0").xpos[2] - z0


def test_mass_changes_dynamics():
    assert _lift_height(1.0) > 0.01
    assert _lift_height(3.0) < 1e-3


def _ee_travel(gain):
    env = push_env(gain=gain)
    o, _ = env.reset(seed=0)
    for _ in range(4):
        o, *_ = env.step(np.array([1.0, 0, 0, 0], dtype=np.float32))
    return o["observation"][0] - env.unwrapped.initial_gripper_xpos[0]


def test_gain_applies_after_clip():
    t1, t15 = _ee_travel(1.0), _ee_travel(1.5)
    assert t15 > 1.3 * t1 > 0


# --- latency / noise / history / shortcut -----------------------------------------------------
class _Spy(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.executed = []

    def step(self, action):
        self.executed.append(np.asarray(action).copy())
        return self.env.step(action)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_latency_delays_exactly_k(k):
    spy = _Spy(gym.make(ENV_IDS["push"]))
    env = LatencyWrapper(spy, delay=k)
    env.reset(seed=0)
    issued = [np.full(4, 0.1 * (i + 1), dtype=np.float32) for i in range(6)]
    for a in issued:
        env.step(a)
    for t in range(6):
        expected = np.zeros(4) if t < k else issued[t - k]
        np.testing.assert_allclose(spy.executed[t], expected)


def test_dr_spec_sets_latency_on_inner_wrapper():
    env = push_env(latency=2)
    env.reset(seed=0)
    lat = env
    while not isinstance(lat, LatencyWrapper):
        lat = lat.env
    assert lat.delay == 2 and len(lat._queue) == 2


def test_noise_magnitude_and_achieved_goal_consistency():
    clean = push_env(obs_noise=0.0)
    noisy = push_env(obs_noise=0.01)
    o_c, _ = clean.reset(seed=3)
    o_n, _ = noisy.reset(seed=3)
    diffs = []
    for _ in range(40):
        a = np.array([0.3, -0.2, 0.1, 0], dtype=np.float32)
        o_c, *_ = clean.step(a)
        o_n, *_ = noisy.step(a)
        diffs.append(o_n["observation"] - o_c["observation"])
        np.testing.assert_allclose(o_n["achieved_goal"], o_n["observation"][3:6])
        np.testing.assert_allclose(o_n["desired_goal"], o_c["desired_goal"])
    assert np.std(diffs) == pytest.approx(0.01, rel=0.15)


def test_history_shapes_and_contents():
    env = make_env(load_exp("configs/exp/push_drhist.yaml"), seed=0, dr_spec=fixed_spec({}))
    assert env.observation_space["observation"].shape == (4 * 25 + 3 * 4,)
    o0, _ = env.reset(seed=0)
    s = o0["observation"]
    assert s.shape == (112,)
    for i in range(1, 4):
        np.testing.assert_allclose(s[25 * i : 25 * (i + 1)], s[:25])
    np.testing.assert_allclose(s[100:], 0)
    a1 = np.array([0.5, 0, 0, 0], dtype=np.float32)
    o1, *_ = env.step(a1)
    s1 = o1["observation"]
    np.testing.assert_allclose(s1[25:50], s[:25])
    np.testing.assert_allclose(s1[100:104], a1)
    np.testing.assert_allclose(o1["achieved_goal"], s1[PUSH_LAYOUT["obj_pos"]])
    assert o1["desired_goal"].shape == (3,)


@pytest.mark.parametrize("mode", ["intact", "random"])
def test_shortcut_hint(mode):
    exp = load_exp("configs/exp/reach_shortcut.yaml")
    env = make_env(exp, seed=0, dr_spec=fixed_spec({}), shortcut_mode=mode)
    assert env.observation_space["observation"].shape == (13,)
    errs = []
    for ep in range(20):
        o, _ = env.reset(seed=ep)
        errs.append(np.linalg.norm(o["observation"][10:] - o["desired_goal"]))
    if mode == "intact":
        assert max(errs) < 0.02
    else:
        assert np.median(errs) > 0.05


# --- HER through the full wrapper stack -------------------------------------------------------
@pytest.mark.parametrize(
    "exp_name", ["push_nodr", "push_dr", "push_drhist", "reach_dr", "reach_shortcut"]
)
def test_her_compute_reward_through_wrappers(exp_name):
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv

    exp = load_exp(f"configs/exp/{exp_name}.yaml")
    venv = DummyVecEnv([lambda: make_env(exp, seed=0)])
    o = venv.reset()
    r = venv.env_method("compute_reward", o["achieved_goal"], o["desired_goal"], [{}])[0]
    assert r.shape == (1,)
    model = SAC(
        "MultiInputPolicy",
        venv,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        learning_starts=100,
        batch_size=32,
        policy_kwargs=dict(net_arch=[16]),
        verbose=0,
        seed=0,
    )
    model.learn(160)
    assert model.replay_buffer.size() > 0


def test_shortcut_goal_noise_is_policy_input_only():
    exp = {**load_exp("configs/exp/reach_shortcut_v2.yaml")}
    env = make_env(exp, seed=0, dr_spec=fixed_spec({}))
    assert "Dense" in env.unwrapped.spec.id
    o, _ = env.reset(seed=0)
    true_goal = env.unwrapped.goal
    assert np.linalg.norm(o["desired_goal"] - true_goal) > 1e-4
    assert np.linalg.norm(o["observation"][10:] - true_goal) < 0.02
    _, r, *_ = env.step(np.zeros(4, dtype=np.float32))
    ee = env.unwrapped._get_obs()["achieved_goal"]
    assert r == pytest.approx(-np.linalg.norm(ee - true_goal), abs=1e-5)


def test_train_without_her_builds_plain_buffer():
    from stable_baselines3.common.buffers import DictReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv

    from projectv.train.train import build_model

    exp = load_exp("configs/exp/reach_shortcut_v2.yaml")
    algo = dict(exp["algo_cfg"], net_arch=[8], buffer_size=1000)
    m = build_model(algo, DummyVecEnv([lambda: make_env(exp, seed=0)]), 0, None)
    assert type(m.replay_buffer) is DictReplayBuffer


@pytest.mark.parametrize("exp_name", ["reach_dr", "push_dr"])
def test_dr_params_independent_of_initial_state(exp_name):
    """Regression: DR draws must not reuse the env's random stream for the same reset seed."""
    env = make_env(load_exp(f"configs/exp/{exp_name}.yaml"), seed=0)
    params, init = [], []
    for sd in range(300):
        o, info = env.reset(seed=1000 + sd)
        params.append([info["dr_params"][k] for k in sorted(info["dr_params"])])
        init.append(np.concatenate([o["observation"], o["desired_goal"]]))
    P, X = np.array(params), np.array(init)
    for i in range(P.shape[1]):
        if np.std(P[:, i]) == 0:
            continue
        for j in range(X.shape[1]):
            if np.std(X[:, j]) < 1e-9:
                continue
            assert abs(np.corrcoef(P[:, i], X[:, j])[0, 1]) < 0.3, (i, j)
