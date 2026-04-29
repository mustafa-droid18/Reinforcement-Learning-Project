"""Mario environment construction and custom Gym wrappers.

Builds a single Mario environment from an ExperimentConfig by composing:
  - GymApiCompatibilityWrapper  — bridges nes-py's old 4-return step API to the
                                   modern 5-return (obs, reward, terminated, truncated, info)
  - FrameSkipWrapper            — repeats each action for `frame_skip` game frames,
                                   making full Mario jumps learnable from a single action
  - StagnationTerminationWrapper — truncates episodes when x_pos stops improving,
                                   preventing the agent wasting rollouts on stuck loops
  - RecordEpisodeStatistics     — standard SB3 episode logging
  - GrayScaleObservation / ResizeObservation — pixel preprocessing to 84x84 greyscale
  - RewardShapingWrapper        — applies a custom reward function if provided;
                                   receives (base_reward, prev_info, info, action,
                                   terminated, truncated) and must return a finite float
"""
from __future__ import annotations

from typing import Callable

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, RecordEpisodeStatistics, ResizeObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_rl.config import ExperimentConfig
from mario_rl.reward_api import RewardFn, validate_reward_output


ACTION_SETS = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,
}


class FrameSkipWrapper(gym.Wrapper):
    """Repeat each action for `skip` game frames and sum the rewards.

    Placed before observation wrappers so that only the final frame's
    observation is returned, matching standard Mario/Atari practice.
    Each agent step now covers ~67ms of game time (4 × 1/60s) instead
    of ~17ms, making jump button-holds learnable from a single action.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StagnationTerminationWrapper(gym.Wrapper):
    """Truncate episodes that stop making horizontal progress."""

    def __init__(self, env: gym.Env, max_stagnation_steps: int):
        super().__init__(env)
        self.max_stagnation_steps = max_stagnation_steps
        self.best_x = 0.0
        self.stagnation_steps = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        info = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
        self.best_x = float(info.get("x_pos", 0.0))
        self.stagnation_steps = 0
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_x = float(info.get("x_pos", self.best_x))
        if curr_x > self.best_x + 1.0:
            self.best_x = curr_x
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        if (
            not terminated
            and not truncated
            and self.max_stagnation_steps > 0
            and self.stagnation_steps >= self.max_stagnation_steps
        ):
            truncated = True
            info = dict(info)
            info["stagnation_truncated"] = True
            info["stagnation_steps"] = self.stagnation_steps

        return obs, reward, terminated, truncated, info


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


class GymApiCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)

        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)

        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            return result

        obs, reward, done, info = result
        return obs, reward, bool(done), False, info


def _apply_observation_wrappers(env: gym.Env, config: ExperimentConfig) -> gym.Env:
    if config.resize_shape:
        env = ResizeObservation(env, shape=tuple(config.resize_shape))
    if config.grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    return env


def build_env(
    config: ExperimentConfig,
    reward_fn: Callable[..., float] | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    if config.action_set not in ACTION_SETS:
        raise ValueError(f"Unsupported action set: {config.action_set}")

    env = gym_super_mario_bros.make(
        config.env_id,
        apply_api_compatibility=True,
        disable_env_checker=True,
        render_mode=render_mode,
    )
    env = JoypadSpace(env, ACTION_SETS[config.action_set])
    env = GymApiCompatibilityWrapper(env)

    if config.frame_skip > 1:
        env = FrameSkipWrapper(env, skip=config.frame_skip)

    if config.max_stagnation_steps > 0:
        env = StagnationTerminationWrapper(env, max_stagnation_steps=config.max_stagnation_steps)

    env = RecordEpisodeStatistics(env)
    env = _apply_observation_wrappers(env, config)

    if reward_fn is not None:
        env = RewardShapingWrapper(env, reward_fn)

    return env
