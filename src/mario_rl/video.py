"""Video recording utilities for trained Mario PPO agents.

Provides a VideoRecorderCallback for recording checkpoint videos during
training, and a standalone entrypoint for recording a saved model after
training.

Usage:
    PYTHONPATH=src python -m mario_rl.video \
        --config configs/llm/llm_v1_final.json \
        --model artifacts/llm_v1_final_seed0/models/best_model.zip \
        --output artifacts/eval_videos/llm_v1_final_best_model.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_rl.config import ExperimentConfig
from mario_rl.vec_env import build_vec_env


def _capture_frame(env) -> np.ndarray:
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()

    if frame is None:
        raise RuntimeError("Environment did not return a render frame.")

    frame_array = np.asarray(frame)
    if frame_array.ndim == 4:
        frame_array = frame_array[0]

    return frame_array


def record_policy_video(
    *,
    model: Any,
    config: ExperimentConfig,
    output_path: str | Path,
    max_steps: int,
    fps: int,
) -> dict:
    import imageio.v2 as imageio
    from dataclasses import replace

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Always record a single environment regardless of training n_envs setting
    env = build_vec_env(replace(config, n_envs=1), reward_path=None, render_mode="rgb_array")
    obs = env.reset()

    total_reward = 0.0
    final_info: dict = {}
    steps = 0

    with imageio.get_writer(output_path, fps=fps) as writer:
        writer.append_data(_capture_frame(env))

        for _ in range(max_steps):
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            final_info = infos[0]
            steps += 1

            writer.append_data(_capture_frame(env))

            if bool(dones[0]):
                break

    env.close()

    return {
        "output_path": str(output_path),
        "steps": steps,
        "total_reward": total_reward,
        "x_pos": int(final_info.get("x_pos", 0)),
        "score": int(final_info.get("score", 0)),
        "flag_get": bool(final_info.get("flag_get", False)),
    }


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        video_dir: str | Path,
        record_freq: int,
        max_steps: int,
        fps: int,
    ):
        super().__init__()
        self.config = config
        self.video_dir = Path(video_dir)
        self.record_freq = record_freq
        self.max_steps = max_steps
        self.fps = fps
        self._last_recorded_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True

        if self.num_timesteps - self._last_recorded_step < self.record_freq:
            return True

        self._last_recorded_step = self.num_timesteps
        output_path = self.video_dir / f"policy_step_{self.num_timesteps}.{self.config.video_format}"
        record_policy_video(
            model=self.model,
            config=self.config,
            output_path=output_path,
            max_steps=self.max_steps,
            fps=self.fps,
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained Mario policy as a video.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--model", required=True, help="Path to the saved model zip file.")
    parser.add_argument("--output", required=True, help="Output video path, usually .gif or .mp4.")
    parser.add_argument("--max-steps", type=int, default=1200, help="Maximum environment steps to record.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    model = PPO.load(args.model)
    summary = record_policy_video(
        model=model,
        config=config,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
    )
    print(summary)


if __name__ == "__main__":
    main()
