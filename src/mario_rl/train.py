from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from mario_rl.config import ExperimentConfig
from mario_rl.video import VideoRecorderCallback
from mario_rl.vec_env import build_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO Mario agent.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    return parser.parse_args()


def _callback_freq(target_timesteps: int, n_envs: int) -> int:
    return max(target_timesteps // max(n_envs, 1), 1)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)

    set_random_seed(config.seed)

    run_dir = Path(config.log_dir) / config.experiment_name
    model_dir = run_dir / "models"
    eval_dir = run_dir / "eval"
    tb_dir = run_dir / "tensorboard"
    video_dir = run_dir / "videos"
    paths = [run_dir, model_dir, eval_dir, tb_dir]
    if config.record_videos:
        paths.append(video_dir)

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    (run_dir / "resolved_config.json").write_text(json.dumps(config.to_json_dict(), indent=2))

    train_env = build_vec_env(config, reward_path=config.train_reward_path)
    eval_env = build_vec_env(config, reward_path=None)

    callbacks = [
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(eval_dir),
            eval_freq=_callback_freq(config.eval_freq, config.n_envs),
            n_eval_episodes=config.n_eval_episodes,
            deterministic=False,
            render=False,
        )
    ]

    if config.record_videos:
        callbacks.append(
            VideoRecorderCallback(
                config=config,
                video_dir=video_dir,
                record_freq=_callback_freq(config.video_freq or config.eval_freq, config.n_envs),
                max_steps=config.video_length,
                fps=config.video_fps,
            )
        )

    callback = CallbackList(callbacks)

    model = PPO(
        policy=config.ppo.policy,
        env=train_env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        seed=config.seed,
        device=config.device,
        verbose=1,
        tensorboard_log=str(tb_dir),
    )

    model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)
    model.save(str(model_dir / "final_model"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
