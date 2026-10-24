import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pytest

from projectv.envs.obs_layout import ENV_IDS, EPISODE_STEPS, assert_layout, obs_dim

gym.register_envs(gymnasium_robotics)


def test_env_ids_are_highest_registered_version():
    for task, eid in ENV_IDS.items():
        base = eid.rsplit("-v", 1)[0]
        versions = [
            int(k.rsplit("-v", 1)[1]) for k in gym.envs.registry if k.rsplit("-v", 1)[0] == base
        ]
        assert int(eid.rsplit("-v", 1)[1]) == max(versions), (task, versions)


@pytest.mark.parametrize("task", ["reach", "push"])
def test_layout_holds_over_random_rollout(task):
    env = gym.make(ENV_IDS[task])
    assert env.spec.max_episode_steps == EPISODE_STEPS
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(EPISODE_STEPS):
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        assert_layout(env, task)
        assert "is_success" in info
        if term or trunc:
            break
    assert trunc
    assert env.observation_space["observation"].shape == (obs_dim(task),)


def test_push_compute_reward_is_vectorised():
    env = gym.make(ENV_IDS["push"])
    o, _ = env.reset(seed=0)
    ag = np.stack([o["achieved_goal"], o["desired_goal"]])
    dg = np.stack([o["desired_goal"], o["desired_goal"]])
    r = env.unwrapped.compute_reward(ag, dg, [{}, {}])
    assert r.shape == (2,) and r[1] == 0.0
