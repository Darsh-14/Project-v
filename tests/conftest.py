import functools

import pytest


@functools.cache
def _tiny_model(exp_name: str):
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv

    from projectv.envs.make_env import make_env
    from projectv.utils.config import load_exp

    exp = load_exp(f"configs/exp/{exp_name}.yaml")
    model = SAC(
        "MultiInputPolicy",
        DummyVecEnv([lambda: make_env(exp, seed=0)]),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        learning_starts=100,
        batch_size=32,
        policy_kwargs=dict(net_arch=[16]),
        device="cpu",
        seed=0,
    )
    model.learn(300)
    return model


@pytest.fixture
def tiny_model():
    """tiny_model(exp_name) -> small trained SAC model (cached per session)."""
    return _tiny_model
