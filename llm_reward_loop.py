"""
Automated LLM reward engineering loop for Mario PPO.

Workflow:
  1. Call Claude API to generate a reward function
  2. Validate and save it
  3. Train for 1M steps
  4. Evaluate and summarize results
  5. Feed summary back to Claude for revision
  6. Repeat up to MAX_ROUNDS times
  7. Promote the best reward function to llm_v1_final.py

Usage:
  PYTHONPATH=src python llm_reward_loop.py

Requires:
  ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import anthropic
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ROUNDS = 5
MODEL = "claude-opus-4-7"
TRAIN_CONFIG = "configs/llm_v1_test.json"
PYTHONPATH = "src"

REWARD_DIR = Path("reward_functions")
PROMPTS_DIR = Path("prompts")
PROMPTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a reinforcement learning reward engineer. Your job is to write a Python reward function that trains a PPO agent to play Super Mario Bros World 1-1.

## Environment

- Game: SuperMarioBros-1-1-v0 (NES emulator via gym-super-mario-bros)
- Goal: Mario must reach the flag at the far right end of the level (x_pos ≈ 3166)
- Action set: SIMPLE_MOVEMENT (7 discrete actions: idle, right, right+jump, right+run, right+run+jump, left, jump)
- Each agent step covers 4 game frames (frame skip = 4), so actions are held for ~67ms
- Episodes end when Mario dies, time runs out, or stagnation is detected

## Available info fields

The `info` dict contains:
- `x_pos` (int): Mario's horizontal position (0 at start, ~3166 at flag)
- `y_pos` (int): Mario's vertical position
- `score` (int): in-game score (increases by 100 for stomping enemies, 200 for coins, etc.)
- `coins` (int): coins collected
- `flag_get` (bool): True if Mario reached the flag this step
- `life` (int): lives remaining
- `status` (str): "small", "tall", or "fireball"
- `time` (int): game timer counting down from 400
- `world` (int): world number (always 1)
- `stage` (int): stage number (always 1)

`prev_info` contains the same fields from the previous step.

## Baseline

A PPO agent trained with no reward shaping (just the default environment reward) reliably reaches x_pos=315 and dies at the first enemy. It never learns to jump over obstacles.

## Function signature

You must implement exactly this function:

```python
from __future__ import annotations
import math

def compute_reward(
    *,
    base_reward: float,
    prev_info: dict,
    info: dict,
    action: int,
    terminated: bool,
    truncated: bool,
) -> float:
    ...
    return float(reward)
```

## Constraints

- Only use Python standard library (math, etc.) — no numpy, no torch
- Must return a finite float
- Do not hardcode specific x_pos coordinates (e.g. "if x_pos == 315")
- Keep reward magnitudes reasonable — avoid values above 1000 per step
- `base_reward` is the environment's native reward (roughly +1 to +15 per step for forward progress, -15 on death)

## Output format

Return ONLY a Python code block with the complete reward function. No explanation before or after — just the code block."""


def extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_reward_fn(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    namespace: dict = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Execution error: {e}"

    if "compute_reward" not in namespace:
        return False, "compute_reward function not found"

    try:
        dummy_info = {"x_pos": 100, "y_pos": 79, "score": 0, "coins": 0,
                      "flag_get": False, "life": 2, "status": "small", "time": 380}
        result = namespace["compute_reward"](
            base_reward=1.0,
            prev_info={"x_pos": 90, "y_pos": 79, "score": 0, "coins": 0,
                       "flag_get": False, "life": 2, "status": "small", "time": 381},
            info=dummy_info,
            action=1,
            terminated=False,
            truncated=False,
        )
        if not isinstance(result, (int, float)) or not np.isfinite(result):
            return False, f"compute_reward returned non-finite value: {result}"
    except Exception as e:
        return False, f"compute_reward raised exception: {e}"

    return True, "ok"


def run_training(round_num: int) -> bool:
    config = json.loads(Path(TRAIN_CONFIG).read_text())
    config["experiment_name"] = f"llm_v1_test_r{round_num}"
    tmp_config = Path(f"configs/llm_v1_test_r{round_num}.json")
    tmp_config.write_text(json.dumps(config, indent=2))

    print(f"\n[Round {round_num}] Training for 1M steps...")
    result = subprocess.run(
        [sys.executable, "-m", "mario_rl.train", "--config", str(tmp_config)],
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
    )
    return result.returncode == 0


def load_eval_results(round_num: int) -> dict | None:
    npz_path = Path(f"artifacts/llm_v1_test_r{round_num}/eval/evaluations.npz")
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    means = data["results"].mean(axis=1)
    ep_lengths = data["ep_lengths"].mean(axis=1)

    best_idx = means.argmax()
    final_idx = len(means) - 1

    early_mean = means[:4].mean() if len(means) >= 4 else means.mean()
    mid_mean = means[len(means)//4: len(means)//2].mean() if len(means) >= 8 else means.mean()
    late_mean = means[-4:].mean() if len(means) >= 4 else means.mean()

    return {
        "total_evals": len(timesteps),
        "best_reward": float(means[best_idx]),
        "best_reward_step": int(timesteps[best_idx]),
        "final_reward": float(means[final_idx]),
        "early_avg": float(early_mean),
        "mid_avg": float(mid_mean),
        "late_avg": float(late_mean),
        "avg_ep_length_final": float(ep_lengths[final_idx]),
    }


def build_feedback_message(round_num: int, reward_code: str, results: dict) -> str:
    trend = "improving" if results["late_avg"] > results["early_avg"] else "declining or flat"

    return f"""Round {round_num} training results (1M steps, 4 parallel envs, SIMPLE_MOVEMENT):

- Best mean eval reward: {results['best_reward']:.1f} at step {results['best_reward_step']:,}
- Final mean eval reward: {results['final_reward']:.1f}
- Learning trend: early avg {results['early_avg']:.1f} → mid avg {results['mid_avg']:.1f} → late avg {results['late_avg']:.1f} ({trend})
- Avg episode length at end: {results['avg_ep_length_final']:.0f} steps

Reminder: baseline PPO (no reward shaping) reaches mean reward ~252 and x_pos ~315.

The reward function used in round {round_num}:

```python
{reward_code}
```

Based on these results, revise the reward function to improve the agent's performance. Think about:
- Is the agent learning or stagnating?
- Are there reward components that might be conflicting or missing?
- Is the reward scale appropriate?

Return ONLY the revised Python code block."""


def save_prompt_log(round_num: int, messages: list, response_text: str, code: str) -> None:
    prompt_file = PROMPTS_DIR / f"llm_v1_r{round_num}_prompt.md"
    response_file = PROMPTS_DIR / f"llm_v1_r{round_num}_response.md"

    prompt_content = f"# LLM v1 Round {round_num} — Prompt\n\n"
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        prompt_content += f"## {role}\n\n{content}\n\n---\n\n"
    prompt_file.write_text(prompt_content)

    response_file.write_text(
        f"# LLM v1 Round {round_num} — Response\n\n"
        f"## Raw response\n\n{response_text}\n\n"
        f"## Extracted code\n\n```python\n{code}\n```\n"
    )


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []
    best_reward = -float("inf")
    best_round = 0
    best_code = ""
    results_log = []

    initial_message = "Generate a reward function for training a PPO agent to play Super Mario Bros World 1-1. The agent must learn to run right, jump over enemies and obstacles, and reach the flag."
    messages.append({"role": "user", "content": initial_message})

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'='*60}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        response_text = response.content[0].text
        messages.append({"role": "assistant", "content": response_text})

        code = extract_code(response_text)
        valid, reason = validate_reward_fn(code)

        if not valid:
            print(f"[Round {round_num}] Invalid reward function: {reason}")
            messages.append({"role": "user", "content": f"The reward function failed validation: {reason}. Please fix it and return only the corrected code block."})
            save_prompt_log(round_num, messages, response_text, code)
            continue

        current_path = REWARD_DIR / "llm_v1_current.py"
        archive_path = REWARD_DIR / f"llm_v1_r{round_num}.py"
        current_path.write_text(code)
        archive_path.write_text(code)
        print(f"[Round {round_num}] Reward function saved to {archive_path}")

        success = run_training(round_num)
        if not success:
            print(f"[Round {round_num}] Training failed, skipping eval.")
            save_prompt_log(round_num, messages, response_text, code)
            continue

        results = load_eval_results(round_num)
        if results is None:
            print(f"[Round {round_num}] No eval results found.")
            save_prompt_log(round_num, messages, response_text, code)
            continue

        results_log.append({"round": round_num, **results})
        print(f"[Round {round_num}] Best reward: {results['best_reward']:.1f} | Final: {results['final_reward']:.1f}")

        if results["best_reward"] > best_reward:
            best_reward = results["best_reward"]
            best_round = round_num
            best_code = code
            print(f"[Round {round_num}] New best!")

        save_prompt_log(round_num, messages, response_text, code)

        if round_num < MAX_ROUNDS:
            feedback = build_feedback_message(round_num, code, results)
            messages.append({"role": "user", "content": feedback})

    # Save best reward as llm_v1_final.py
    if best_code:
        final_path = REWARD_DIR / "llm_v1_final.py"
        final_path.write_text(best_code)
        print(f"\nBest reward from round {best_round} (reward={best_reward:.1f}) saved to {final_path}")

    # Save summary
    summary = {
        "best_round": best_round,
        "best_reward": best_reward,
        "rounds": results_log,
    }
    Path("prompts/llm_v1_loop_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nLoop complete. Summary saved to prompts/llm_v1_loop_summary.json")
    print(f"Run the final 5M training with: PYTHONPATH=src python -m mario_rl.train --config configs/llm_v1_final.json")


if __name__ == "__main__":
    main()
