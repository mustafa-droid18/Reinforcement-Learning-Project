# Mario RL Reward-Shaping Starter

This repository turns the proposal into a runnable experiment scaffold for:

1. Baseline PPO with the environment reward.
2. PPO with an externally supplied reward function.
3. PPO with a revised reward function after reviewing training results.

The goal is to keep the experiment reproducible. The code separates:

- training reward
- evaluation metrics
- experiment config
- reward function versions

## Project layout

```text
configs/                 Experiment configs for baseline and shaped runs
reward_functions/        Reward function files loaded at runtime
src/mario_rl/            Training, evaluation, env factory, config loading
PLAN_NOTES.md            Proposal fixes before full experiment execution
```

## Quick start

Create a virtual environment, install the dependencies, then run:

```bash
python -m mario_rl.train --config configs/baseline.json
python -m mario_rl.eval \
  --config configs/baseline.json \
  --model artifacts/baseline_seed0/final_model.zip
```

Use the heuristic reward file as a stand-in until the LLM loop is added:

```bash
python -m mario_rl.train --config configs/llm_v1_template.json
```

## Current scope

This starter intentionally does not call an LLM yet. It gives you the fixed
training/evaluation pipeline first, because that is the part you need before
an LLM-generated reward function is meaningful.
