from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from mario_rl.config import ExperimentConfig
from mario_rl.env_factory import build_env
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Mario agent.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--model", required=True, help="Path to the saved model zip file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    return parser.parse_args()


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, _info = result
        return obs
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return obs, reward, done, info

    obs, reward, done, info = result
    return obs, reward, bool(done), info


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    env = build_env(config, reward_fn=None)
    model = PPO.load(args.model)

    episodes = []
    for episode_idx in range(args.episodes):
        obs = reset_env(env)
        done = False
        total_reward = 0.0
        final_info = {}
        step_count = 0

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = step_env(env, action)
            total_reward += float(reward)
            final_info = info
            step_count += 1

        episodes.append(
            {
                "episode": episode_idx,
                "total_reward": total_reward,
                "steps": step_count,
                "x_pos": int(final_info.get("x_pos", 0)),
                "score": int(final_info.get("score", 0)),
                "coins": int(final_info.get("coins", 0)),
                "flag_get": bool(final_info.get("flag_get", False)),
                "time_left": int(final_info.get("time", 0)),
            }
        )

    summary = {
        "episodes": episodes,
        "mean_total_reward": sum(row["total_reward"] for row in episodes) / len(episodes),
        "mean_x_pos": sum(row["x_pos"] for row in episodes) / len(episodes),
        "mean_score": sum(row["score"] for row in episodes) / len(episodes),
        "completion_rate": sum(1 for row in episodes if row["flag_get"]) / len(episodes),
    }

    output_path = Path(config.log_dir) / config.experiment_name / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    main()
