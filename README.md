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
python llm_reward_loop.py --config configs/llm_v1_test_r1.json --rounds 5
```

Each round trains for 1M steps, evaluates the best checkpoint, and feeds results back to Claude for the next revision.

## Key results

| Agent | Steps | Stoch mean x_pos | Flags |
|-------|-------|-----------------|-------|
| Baseline | 1M | 579 | 0/20 |
| LLM v1 Final | 5M | 1374 | 0/20 |
| Human v4 | 10M | 2378 | 9/20 |

See `REPORT_REFERENCE.md` for full methodology and results.
