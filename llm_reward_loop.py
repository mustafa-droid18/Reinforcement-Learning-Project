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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ROUNDS = 5
MODEL = "claude-opus-4-7"
TRAIN_CONFIG = "configs/stochastic_llm/llm_v1_test_r1.json"
PYTHONPATH = "src"

REWARD_DIR = Path("reward_functions/stochastic_llm")
PROMPTS_DIR = Path("prompts")
PROMPTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a reinforcement learning reward engineer. Your job is to write a Python reward function that trains a PPO agent to play Super Mario Bros World 1-1.

## Environment

- Game: SuperMarioBros-1-1-v0 (NES emulator via gym-super-mario-bros)
- Goal: Mario must reach the flag at the far right end of the level (x_pos ≈ 3166)
- Action set: SIMPLE_MOVEMENT (7 discrete actions: idle, right, right+jump, right+run, right+run+jump, left, jump)
- Each agent step covers 4 game frames (frame skip = 4), so actions are held for ~67ms
- Episodes end when Mario dies, time runs out, or stagnation is detected

## Execution model

- The reward function runs inside 4 parallel environments (n_envs=4) using SB3's DummyVecEnv
- All 4 environments import the same Python module, so module-level variables are shared across all 4 envs
- There is no reliable per-environment identifier available in the `info` dict: `world`, `stage`, and `life` are the same value across all 4 envs in normal play, and `id(info)` / `id(prev_info)` is not stable across steps. This means **module-level state keyed on any info field will be contaminated across all 4 environments**. Stateless reward computation — using only the current `prev_info` and `info` dicts — is the correct approach. Do not use module-level variables to track per-episode history.
- Episode-level stagnation is already handled externally: a StagnationTerminationWrapper terminates the episode after 200 steps with no x_pos progress
- y_pos uses screen coordinates: y=0 is the top of the screen, larger y values are lower. Mario jumping means y_pos decreases.

## Available info fields

The `info` dict contains:
- `x_pos` (int): Mario's horizontal position (0 at start, ~3166 at flag)
- `y_pos` (int): Mario's vertical position (0 = top of screen, increases downward)
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

A PPO agent trained with no reward shaping (just the default environment reward) reliably reaches x_pos=315 and dies at the first enemy.

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

## Training diagnostics provided in feedback

After each round, the feedback includes training-time diagnostics from the PPO rollout loop:
- `ep_rew_mean` (training): mean episode return under the shaped reward function during training. This is different from the eval curve, which uses the unshaped Mario reward.
- `ep_len_mean` (training): mean episode length during training rollouts.
- `entropy_loss`: negative entropy of the policy distribution. A value near 0 means the policy has collapsed to near-deterministic behaviour. Healthy values for a 7-action space are typically in the -0.5 to -1.5 range.
- `explained_variance`: how well the value function predicts returns (1.0 = perfect, 0 = no better than the mean, negative = worse than the mean).

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


def run_training(round_num: int, reward_path: Path) -> bool:
    config = json.loads(Path(TRAIN_CONFIG).read_text())
    config["experiment_name"] = f"llm_v1_test_r{round_num}"
    config["train_reward_path"] = str(reward_path)
    tmp_config = Path(f"configs/stochastic_llm/llm_v1_test_r{round_num}.json")
    tmp_config.write_text(json.dumps(config, indent=2))

    print(f"\n[Round {round_num}] Training for 1M steps...")
    result = subprocess.run(
        [sys.executable, "-m", "mario_rl.train", "--config", str(tmp_config)],
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
    )
    return result.returncode == 0


def eval_best_model(round_num: int, n_episodes: int = 20) -> dict | None:
    """Run eval on best_model.zip and return full per-episode + aggregate stats."""
    model_path = Path(f"artifacts/llm_v1_test_r{round_num}/models/best_model.zip")
    if not model_path.exists():
        return None
    try:
        from stable_baselines3 import PPO
        from mario_rl.config import ExperimentConfig
        from mario_rl.vec_env import build_vec_env

        config_path = Path(f"configs/stochastic_llm/llm_v1_test_r{round_num}.json")
        config = ExperimentConfig.from_json(str(config_path))
        env = build_vec_env(config, reward_path=None)
        model = PPO.load(str(model_path))

        episodes = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            final_info = {}
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, dones, infos = env.step(action)
                final_info = infos[0]
                done = bool(dones[0])
                steps += 1
            episodes.append({
                "x_pos":        int(final_info.get("x_pos", 0)),
                "score":        int(final_info.get("score", 0)),
                "coins":        int(final_info.get("coins", 0)),
                "flag_get":     bool(final_info.get("flag_get", False)),
                "time_left":    int(final_info.get("time", 0)),
                "ep_length":    steps,
            })
        env.close()

        best_ep = max(episodes, key=lambda e: e["x_pos"])
        n = len(episodes)
        return {
            "mean_x_pos":        sum(e["x_pos"]     for e in episodes) / n,
            "mean_score":        sum(e["score"]      for e in episodes) / n,
            "mean_coins":        sum(e["coins"]      for e in episodes) / n,
            "completion_rate":   sum(e["flag_get"]   for e in episodes) / n,
            "mean_time_left":    sum(e["time_left"]  for e in episodes) / n,
            "mean_ep_length":    sum(e["ep_length"]  for e in episodes) / n,
            "best_x_pos":        best_ep["x_pos"],
            "best_score":        best_ep["score"],
            "best_coins":        best_ep["coins"],
            "best_flag_get":     best_ep["flag_get"],
            "best_time_left":    best_ep["time_left"],
            "best_ep_length":    best_ep["ep_length"],
        }
    except Exception as e:
        print(f"[eval_best_model failed: {e}]")
        return None


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

    task_data = eval_best_model(round_num)

    return {
        "total_evals":          len(timesteps),
        "best_reward":          float(means[best_idx]),
        "best_reward_step":     int(timesteps[best_idx]),
        "final_reward":         float(means[final_idx]),
        "early_avg":            float(early_mean),
        "mid_avg":              float(mid_mean),
        "late_avg":             float(late_mean),
        "avg_ep_length_final":  float(ep_lengths[final_idx]),
        "learning_curve":       [(int(t), round(float(m), 1)) for t, m in zip(timesteps, means)],
        "task":                 task_data,
    }


def load_training_diagnostics(round_num: int) -> dict | None:
    """Read final training stats from TensorBoard logs."""
    tb_dir = Path(f"artifacts/llm_v1_test_r{round_num}/tensorboard")
    subdirs = sorted(tb_dir.glob("*/")) if tb_dir.exists() else []
    if not subdirs:
        return None
    try:
        ea = EventAccumulator(str(subdirs[0]))
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])

        def last_n(tag, n=8):
            if tag not in scalars:
                return []
            events = ea.Scalars(tag)
            return [round(e.value, 4) for e in events[-n:]]

        return {
            "ep_rew_mean":       last_n("rollout/ep_rew_mean"),
            "ep_len_mean":       last_n("rollout/ep_len_mean"),
            "entropy_loss":      last_n("train/entropy_loss"),
            "explained_variance": last_n("train/explained_variance"),
        }
    except Exception as e:
        print(f"[load_training_diagnostics r{round_num} failed: {e}]")
        return None


def build_feedback_message(round_num: int, reward_code: str, results: dict) -> str:
    t = results.get("task")

    task_section = ""
    if t:
        completion_pct = t['completion_rate'] * 100
        task_section = f"""Task-level metrics — best model evaluated over 10 episodes using unshaped/default Mario reward with the same evaluation wrappers (comparable across rounds):

  Mean across all episodes:
  - x_pos:          {t['mean_x_pos']:.0f} / 3166
  - score:          {t['mean_score']:.0f}
  - coins:          {t['mean_coins']:.1f}
  - completion:     {completion_pct:.0f}%
  - time left:      {t['mean_time_left']:.0f}
  - episode length: {t['mean_ep_length']:.0f} steps

  Best single episode:
  - x_pos:          {t['best_x_pos']} / 3166
  - score:          {t['best_score']}
  - coins:          {t['best_coins']}
  - flag reached:   {t['best_flag_get']}
  - time left:      {t['best_time_left']}
  - episode length: {t['best_ep_length']} steps
"""

    curve_lines = "\n".join(
        f"  step {ts:>9,}: {m:.1f}"
        for ts, m in results["learning_curve"]
    )

    diag = load_training_diagnostics(round_num)
    diag_section = ""
    if diag:
        def fmt(vals):
            return "  →  ".join(f"{v:.3f}" for v in vals) if vals else "n/a"
        diag_section = f"""
Training diagnostics (last 8 rollout updates, chronological order):
  ep_rew_mean (shaped):  {fmt(diag['ep_rew_mean'])}
  ep_len_mean:           {fmt(diag['ep_len_mean'])}
  entropy_loss:          {fmt(diag['entropy_loss'])}
  explained_variance:    {fmt(diag['explained_variance'])}
"""

    return f"""Round {round_num} results.

{task_section}
Eval reward learning curve (eval every 25,000 steps, evaluated on the default unshaped Mario reward — comparable across rounds):
{curve_lines}
{diag_section}
The reward function used in round {round_num}:

```python
{reward_code}
```

Revise the reward function. Return ONLY the revised Python code block."""


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

    initial_message = "Generate a reward function for training a PPO agent to play Super Mario Bros World 1-1."
    messages.append({"role": "user", "content": initial_message})

    MAX_VALIDATION_RETRIES = 3

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'='*60}")

        # Retry validation failures within the same round so a bad response
        # doesn't consume a round.
        code = ""
        valid = False
        reason = ""
        response_text = ""
        for attempt in range(MAX_VALIDATION_RETRIES):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            response_text = response.content[0].text
            messages.append({"role": "assistant", "content": response_text})

            code = extract_code(response_text)
            valid, reason = validate_reward_fn(code)
            if valid:
                break

            print(f"[Round {round_num}] Attempt {attempt + 1}/{MAX_VALIDATION_RETRIES} invalid: {reason}")
            messages.append({"role": "user", "content": f"The reward function failed validation: {reason}. Please fix it and return only the corrected code block."})

        if not valid:
            print(f"[Round {round_num}] All {MAX_VALIDATION_RETRIES} attempts failed validation. Skipping round.")
            save_prompt_log(round_num, messages, response_text, code)
            continue

        current_path = REWARD_DIR / "llm_v1_current.py"
        archive_path = REWARD_DIR / f"llm_v1_r{round_num}.py"
        current_path.write_text(code)
        archive_path.write_text(code)
        print(f"[Round {round_num}] Reward function saved to {archive_path}")

        success = run_training(round_num, archive_path)
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
        task_x = results.get("task", {}).get("mean_x_pos", 0) if results.get("task") else 0
        print(f"[Round {round_num}] mean x_pos: {task_x:.0f} | best eval reward: {results['best_reward']:.1f}")

        # Select best round by task-level metric (mean_x_pos) since it directly
        # reflects level progress in pixels, while best eval reward is a scalar
        # that conflates progress with score and time bonuses.
        if task_x > best_reward:
            best_reward = task_x
            best_round = round_num
            best_code = code
            print(f"[Round {round_num}] New best (mean x_pos = {task_x:.0f})")

        save_prompt_log(round_num, messages, response_text, code)

        if round_num < MAX_ROUNDS:
            feedback = build_feedback_message(round_num, code, results)
            messages.append({"role": "user", "content": feedback})

    # Save best reward as llm_v1_final.py (selected by mean x_pos task metric)
    if best_code:
        final_path = REWARD_DIR / "llm_v1_final.py"
        final_path.write_text(best_code)
        print(f"\nBest reward from round {best_round} (mean x_pos={best_reward:.0f}) saved to {final_path}")

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
