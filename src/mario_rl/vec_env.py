from __future__ import annotations

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from mario_rl.config import ExperimentConfig
from mario_rl.env_factory import build_env
from mario_rl.reward_api import load_reward_function


def build_vec_env(
    config: ExperimentConfig,
    reward_path: str | None,
    render_mode: str | None = None,
):
    reward_fn = load_reward_function(reward_path)

    env_fns = [
        (lambda reward_fn=reward_fn, render_mode=render_mode: build_env(config, reward_fn=reward_fn, render_mode=render_mode))
        for _ in range(config.n_envs)
    ]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    if config.frame_stack > 1:
        env = VecFrameStack(env, n_stack=config.frame_stack, channels_order="first")

    return env
