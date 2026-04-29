# Mario RL — LLM-Generated Reward Shaping

This repository implements an automated LLM reward loop for training a PPO agent to play Super Mario Bros (World 1-1). Three reward designs are compared: baseline (no shaping), human-crafted heuristic, and iteratively LLM-generated rewards.

The experiment is fully reproducible. The code separates:

- training reward
- evaluation metrics
- experiment config
- reward function versions

## Project layout

```text
configs/                 Experiment configs for all runs
reward_functions/        Reward function files loaded at runtime
src/mario_rl/            Training, evaluation, env factory, config loading
llm_reward_loop.py       Automated LLM reward generation and iteration loop
prompts/                 Saved LLM prompts and responses for all 5 rounds
LLM_LOOP_LOG.md          Per-round training curves and eval results
PROJECT_LOG.md           Cross-run experiment log
REPORT_REFERENCE.md      Complete methodology, results, and conclusions
```

## Quick start

Create a virtual environment, install the dependencies, then run:

```bash
# Baseline
python -m mario_rl.train --config configs/baseline.json

# Human heuristic
python -m mario_rl.train --config configs/human_heuristic_v3.json

# LLM final (R1 reward function, 5M steps)
python -m mario_rl.train --config configs/llm_v1_final.json
```

## Running the LLM reward loop

The loop calls Claude (Anthropic API) to generate and iteratively refine a reward function over N rounds. Set your API key, then:

```bash
ANTHROPIC_API_KEY=your_key PYTHONPATH=src python llm_reward_loop.py
```

Configuration (model, rounds, train config) is set via constants at the top of `llm_reward_loop.py`. Each round trains for 1M steps, evaluates the best checkpoint, and feeds results back to Claude for the next revision.

## Key results

All task evals: 20 episodes on `best_model.zip`, unshaped native reward.

| Agent | Train steps | Train eval | Stoch mean x_pos | Stoch max | Flags | Det x_pos |
|-------|------------|------------|-----------------|-----------|-------|-----------|
| Baseline | 1M | det=True | 579 | 722 | 0/20 | — |
| LLM v1 R1 (loop round 1) | 1M | det=False | 1167 | 1511 | 0/20 | — |
| LLM v1 Final (R1, 5M) | 5M | det=False | 1374 | 2130 | 0/20 | 1129 |
| LLM v1 R3 Final | 5M | det=False | 1441 | 1909 | 0/20 | 1904 |
| Human v3 (original) | 5M | det=True, n=5 | 2044 | 3161 | 3/20 | 2354 |
| Human v3 (stoch retrain) | 5M | det=False, n=10 | 1934 | 2475 | 0/20 | 1797 |
| Human v4 | 10M | det=True, n=5 | 2378 | 3161 | 9/20 | 2402 |

See `REPORT_REFERENCE.md` for full methodology and results.
