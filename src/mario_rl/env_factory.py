from __future__ import annotations

from typing import Callable

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, RecordEpisodeStatistics, ResizeObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_rl.config import ExperimentConfig
from mario_rl.reward_api import RewardFn, validate_reward_output


ACTION_SETS = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,
}


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.prev_info: dict = {}

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            self.prev_info = dict(info)
            return obs, info

        self.prev_info = {}
        return result

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            obs, base_reward, terminated, truncated, info = result
            reward = self.reward_fn(
                base_reward=float(base_reward),
                prev_info=self.prev_info,
                info=info,
                action=int(action),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            self.prev_info = dict(info)
            return obs, validate_reward_output(reward), terminated, truncated, info

        obs, base_reward, done, info = result
        reward = self.reward_fn(
            base_reward=float(base_reward),
            prev_info=self.prev_info,
            info=info,
            action=int(action),
            terminated=bool(done),
            truncated=False,
        )
        self.prev_info = dict(info)
        return obs, validate_reward_output(reward), done, info


def _apply_observation_wrappers(env: gym.Env, config: ExperimentConfig) -> gym.Env:
    if config.resize_shape:
        env = ResizeObservation(env, shape=tuple(config.resize_shape))
    if config.grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    if config.frame_stack > 1:
        env = FrameStack(env, num_stack=config.frame_stack)
    return env


def build_env(config: ExperimentConfig, reward_fn: Callable[..., float] | None = None) -> gym.Env:
    if config.action_set not in ACTION_SETS:
        raise ValueError(f"Unsupported action set: {config.action_set}")

    env = gym_super_mario_bros.make(config.env_id)
    env = JoypadSpace(env, ACTION_SETS[config.action_set])
    env = RecordEpisodeStatistics(env)
    env = _apply_observation_wrappers(env, config)

    if reward_fn is not None:
        env = RewardShapingWrapper(env, reward_fn)

    return env

