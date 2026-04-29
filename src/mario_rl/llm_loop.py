"""
Automated LLM reward design loop for Mario PPO.

Each iteration:
  1. Ask Claude to generate / revise a reward function
  2. Train PPO for --timesteps steps
  3. Evaluate the best checkpoint on task metrics
  4. Feed results back to Claude
  5. Repeat up to --iterations times

Usage:
  PYTHONPATH=src python -m mario_rl.llm_loop \
      --api-key sk-ant-... \
      --iterations 5 \
      --timesteps 25000
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
REWARD_DIR    = PROJECT_ROOT / "reward_functions"
CONFIG_DIR    = PROJECT_ROOT / "configs"
PROMPTS_DIR   = PROJECT_ROOT / "prompts"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SRC_DIR       = PROJECT_ROOT / "src"

SYSTEM_PROMPT = """\
You are a reward engineering assistant for a PPO agent learning to play \
Super Mario Bros (level 1-1, SIMPLE_MOVEMENT action set, pixel observations).

Your job is to write or revise a Python reward function that helps the agent \
make forward progress through the level.

The function MUST have this exact signature:
  def compute_reward(*, base_reward: float, prev_info: dict, info: dict,
                     action: int, terminated: bool, truncated: bool) -> float:

Available variables:
  base_reward  - default environment reward (small positive for moving right)
  prev_info    - dict from previous step: x_pos (0-3186), score, time, coins, flag_get
  info         - same dict for the current step
  action       - integer 0-6 (SIMPLE_MOVEMENT)
  terminated   - True if Mario died
  truncated    - True if episode hit the time limit

Rules:
  - Use ONLY the variables above and the Python standard library (math is fine)
  - Do not hardcode specific x_pos values for enemies or pipes
  - Keep per-step reward magnitudes moderate (single digits); large bonuses for flag ok
  - Return ONLY the Python code for the function — no prose, no markdown fences\
"""

INITIAL_PROMPT = """\
Baseline PPO agent results (default environment reward, 300k timesteps):
  mean reward      : 252
  mean x_pos       : 315  (level ends at x ≈ 3186)
  completion rate  : 0%
  observed behavior: Mario runs right and immediately dies to the first Goomba

Design a reward function that will help the agent survive longer and make \
forward progress. Explain each term briefly in inline comments.\
"""


def _feedback_prompt(iteration: int, timesteps: int, summary: dict, behavior: str) -> str:
    return (
        f"Iteration {iteration} results with your reward function "
        f"({timesteps:,} timesteps):\n"
        f"  mean reward     : {summary['mean_total_reward']:.1f}\n"
        f"  mean x_pos      : {summary['mean_x_pos']:.1f}  (level ends at x ≈ 3186)\n"
        f"  completion rate : {summary['completion_rate'] * 100:.1f}%\n"
        f"  observed        : {behavior}\n\n"
        "Please revise the reward function to improve these results. "
        "Explain what you changed and why in inline comments."
    )


def _infer_behavior(summary: dict) -> str:
    x  = summary["mean_x_pos"]
    cr = summary["completion_rate"]
    if cr > 0:
        return f"Mario completed the level {cr * 100:.0f}% of the time"
    if x < 400:
        return "Mario still dying very early, likely to the first Goomba"
    if x < 800:
        return f"Mario reaching x≈{x:.0f} before dying — past first obstacle but stalling"
    if x < 1500:
        return f"Mario reaching x≈{x:.0f} — moderate progress but stuck mid-level"
    return f"Mario reaching x≈{x:.0f} — good progress but not completing the level"


def _ask_claude(client: anthropic.Anthropic, messages: list[dict], model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_code(code: str) -> None:
    ast.parse(code)
    if "def compute_reward" not in code:
        raise ValueError("Response does not contain 'def compute_reward'")


def _make_config(iteration: int, timesteps: int) -> Path:
    name = f"llm_loop_v{iteration}_seed0"
    cfg = {
        "experiment_name": name,
        "env_id": "SuperMarioBros-1-1-v0",
        "action_set": "SIMPLE_MOVEMENT",
        "total_timesteps": timesteps,
        "eval_freq": 25000,
        "n_eval_episodes": 5,
        "seed": 0,
        "n_envs": 4,
        "frame_skip": 4,
        "max_stagnation_steps": 200,
        "frame_stack": 4,
        "grayscale": True,
        "resize_shape": [84, 84],
        "train_reward_path": str(REWARD_DIR / f"llm_v{iteration}.py"),
        "log_dir": str(ARTIFACTS_DIR),
        "record_videos": True,
        "video_freq": 25000,
        "video_length": 10000,
        "video_fps": 30,
        "video_format": "mp4",
        "device": "auto",
        "ppo": {
            "policy": "CnnPolicy",
            "learning_rate": 0.00025,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.03,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    }
    path = CONFIG_DIR / f"llm_loop_v{iteration}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def _train(config_path: Path) -> None:
    print(f"\n[loop] Training with {config_path.name} ...")
    _run([sys.executable, "-m", "mario_rl.train", "--config", str(config_path)])


def _evaluate(config_path: Path, experiment_name: str) -> dict:
    model_path  = ARTIFACTS_DIR / experiment_name / "models" / "best_model.zip"
    output_path = ARTIFACTS_DIR / experiment_name / "eval_summary.json"
    print(f"[loop] Evaluating best model ...")
    _run([
        sys.executable, "-m", "mario_rl.eval",
        "--config",   str(config_path),
        "--model",    str(model_path),
        "--episodes", "5",
        "--output",   str(output_path),
    ])
    return json.loads(output_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated LLM reward design loop.")
    parser.add_argument("--api-key",    default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of reward design iterations (default: 5)")
    parser.add_argument("--timesteps",  type=int, default=25_000,
                        help="Training timesteps per iteration (default: 25000)")
    parser.add_argument("--model",      default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6)")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Error: provide --api-key or set the ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=args.api_key)
    PROMPTS_DIR.mkdir(exist_ok=True)

    messages: list[dict] = []
    results:  list[dict] = []
    best_iter  = 0
    best_x_pos = 0.0

    for i in range(1, args.iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"  ITERATION {i} / {args.iterations}")
        print(f"{'=' * 60}")

        user_content = (
            INITIAL_PROMPT if i == 1
            else _feedback_prompt(
                i - 1, args.timesteps,
                results[-1]["summary"],
                _infer_behavior(results[-1]["summary"]),
            )
        )
        messages.append({"role": "user", "content": user_content})

        print("[loop] Asking Claude for reward function ...")
        try:
            response_text = _ask_claude(client, messages, args.model)
        except Exception as exc:
            print(f"[loop] Claude API error: {exc}. Stopping.")
            break
        messages.append({"role": "assistant", "content": response_text})

        (PROMPTS_DIR / f"llm_v{i}_prompt.md").write_text(user_content)
        (PROMPTS_DIR / f"llm_v{i}_response.md").write_text(response_text)
        print(f"[loop] Prompt and response saved to prompts/llm_v{i}_*.md")

        code = _extract_code(response_text)
        try:
            _validate_code(code)
        except Exception as exc:
            print(f"[loop] Invalid code from Claude ({exc}). Skipping iteration {i}.")
            continue

        reward_path = REWARD_DIR / f"llm_v{i}.py"
        reward_path.write_text(code)
        print(f"[loop] Reward function saved to reward_functions/llm_v{i}.py")

        config_path     = _make_config(i, args.timesteps)
        experiment_name = f"llm_loop_v{i}_seed0"

        try:
            _train(config_path)
        except RuntimeError as exc:
            print(f"[loop] Training failed: {exc}. Skipping iteration {i}.")
            continue

        try:
            summary = _evaluate(config_path, experiment_name)
        except RuntimeError as exc:
            print(f"[loop] Eval failed: {exc}. Skipping iteration {i}.")
            continue

        results.append({"iteration": i, "summary": summary})
        x = summary["mean_x_pos"]
        print(f"\n[loop] Iteration {i} done:")
        print(f"         mean x_pos      : {x:.1f}")
        print(f"         mean reward     : {summary['mean_total_reward']:.1f}")
        print(f"         completion rate : {summary['completion_rate'] * 100:.1f}%")

        if x > best_x_pos:
            best_x_pos = x
            best_iter  = i

    if not results:
        print("\n[loop] No successful iterations. Check errors above.")
        return

    # Save full results log
    log_path = ARTIFACTS_DIR / "llm_loop_results.json"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Run':<18} {'mean x_pos':>10} {'complete%':>10} {'mean reward':>12}")
    print(f"  {'-'*54}")
    print(f"  {'baseline':<18} {'315.0':>10} {'0.0%':>10} {'252.0':>12}")
    print(f"  {'human_heuristic':<18} {'n/a':>10} {'n/a':>10} {'374.0':>12}")
    for r in results:
        s   = r["summary"]
        tag = f"llm_v{r['iteration']}"
        print(f"  {tag:<18} {s['mean_x_pos']:>10.1f} "
              f"{s['completion_rate']*100:>9.1f}% {s['mean_total_reward']:>12.1f}")

    print(f"\n  Best LLM iteration : v{best_iter}  (mean x_pos: {best_x_pos:.1f})")
    print(f"  Best reward file   : reward_functions/llm_v{best_iter}.py")
    print(f"  All prompts saved  : prompts/")
    print(f"  Full results log   : artifacts/llm_loop_results.json")


if __name__ == "__main__":
    main()
