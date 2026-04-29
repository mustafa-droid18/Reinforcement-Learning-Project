"""Converts a Stable-Baselines3 evaluations.npz file to a human-readable CSV.

EvalCallback saves per-checkpoint rewards and episode lengths as a compressed
numpy archive. This script unpacks that archive into a CSV with one row per
evaluation checkpoint, showing mean/min/max reward and episode length.

Usage:
    PYTHONPATH=src python -m mario_rl.summarize_eval \
        --npz artifacts/baseline_seed0/eval/evaluations.npz \
        --output artifacts/baseline_seed0/eval/evaluation_history.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stable-Baselines3 evaluation history.")
    parser.add_argument("--npz", required=True, help="Path to evaluations.npz.")
    parser.add_argument("--output", required=True, help="Path to write CSV summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)

    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timesteps",
                "mean_reward",
                "min_reward",
                "max_reward",
                "mean_episode_length",
                "min_episode_length",
                "max_episode_length",
            ],
        )
        writer.writeheader()

        for idx, timestep in enumerate(timesteps):
            rewards = results[idx]
            lengths = ep_lengths[idx]
            writer.writerow(
                {
                    "timesteps": int(timestep),
                    "mean_reward": float(np.mean(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "mean_episode_length": float(np.mean(lengths)),
                    "min_episode_length": int(np.min(lengths)),
                    "max_episode_length": int(np.max(lengths)),
                }
            )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
