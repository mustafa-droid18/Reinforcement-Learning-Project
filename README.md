# Mario RL Reward Shaping

This project trains a PPO agent to play **Super Mario Bros World 1-1** and compares three reward-design strategies:

1. **Baseline**: native Mario environment reward only
2. **Human heuristic**: hand-designed reward shaping
3. **LLM-generated rewards**: an automated Claude feedback loop that writes and revises reward functions

The core research question is whether an LLM can produce useful reinforcement-learning reward functions from training feedback, and how those rewards compare against both no shaping and human-designed shaping.

The full methodology, result tables, and interpretation are in `REPORT_REFERENCE.md`.

## Main Results

All task evaluations use `best_model.zip`, 20 stochastic episodes (`deterministic=False`), and the unshaped native Mario reward for comparability.

| Agent | Steps | Mean x_pos | Std | Max x_pos | Flags |
|-------|-------|------------|-----|-----------|-------|
| Baseline | 1M | 579 | 96 | 722 | 0/20 |
| LLM, det=False selection - Iteration 1 | 1M | 1167 | 242 | 1511 | 0/20 |
| LLM, det=False selection - Iteration 1 final | 5M | 1374 | 502 | 2130 | 0/20 |
| LLM, det=False selection - Iteration 3 final | 5M | 1441 | 355 | 1909 | 0/20 |
| Human heuristic v3 | 5M | 2044 | 594 | 3161 | 3/20 |
| Human heuristic v4 | 10M | 2378 | 935 | 3161 | 9/20 |

Key findings:

- LLM reward shaping substantially improves over the baseline: 579 to 1167 mean x_pos at 1M steps.
- The best human heuristic still outperforms the LLM rewards at higher compute, including successful level completions.
- The LLM loop did not improve monotonically. Round 1 had the best 1M mean; later rounds often overcorrected.
- Reward scale was highly sensitive. Larger progress bonuses caused entropy collapse and short-episode reward hacking.
- Stochastic evaluation was important for PPO because deterministic evaluation often hid useful policy behavior.

## Environment

- Game: `SuperMarioBros-1-1-v0`
- Algorithm: PPO from Stable-Baselines3
- Policy: `CnnPolicy`
- Action set: `SIMPLE_MOVEMENT`
- Observation: grayscale `84x84`, 4-frame stack
- Frame skip: 4
- Parallel envs: 4
- Episode termination: death, time-out, or 200 steps with no x-position progress

The agent observes pixels, not game objects. Rewards receive Mario metadata from the environment such as `x_pos`, `score`, `coins`, `status`, `time`, and `flag_get`.

## Repository Layout

```text
configs/                  Experiment configs
configs/human/            Human heuristic training configs
configs/llm/              LLM reward training configs
reward_functions/         Reward functions loaded during training
reward_functions/llm/     Archived LLM reward functions from each round
src/mario_rl/             Training, evaluation, env construction, video tools
prompts/                  Saved LLM prompts and responses
llm_reward_loop.py        Automated LLM reward-generation loop
PROJECT_LOG.md            Running project notes
REPORT_REFERENCE.md       Final report reference with full methodology/results
artifacts/                Models, TensorBoard logs, eval files, and videos
```

## Setup

Python 3.10 was used for the project.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, CUDA is not available. PyTorch MPS may be available, but this project defaults to `device: auto`; for this Mario CNN workload, CPU can still be competitive because environment stepping is a major bottleneck.

## Running Training

Run commands from the repository root.

```bash
# Baseline: native reward only
PYTHONPATH=src python -m mario_rl.train --config configs/baseline.json

# Human heuristic v3
PYTHONPATH=src python -m mario_rl.train --config configs/human/human_heuristic_v3.json

# Human heuristic v4
PYTHONPATH=src python -m mario_rl.train --config configs/human/human_heuristic_v4.json

# LLM final reward: iteration 1 reward trained for 5M steps
PYTHONPATH=src python -m mario_rl.train --config configs/llm/llm_v1_final.json

# LLM iteration 3 reward trained for 5M steps
PYTHONPATH=src python -m mario_rl.train --config configs/llm/llm_v1_r3_final.json
```

Training outputs are written under `artifacts/<experiment_name>_seed0/`, including:

- `models/best_model.zip`
- `models/final_model.zip`
- `eval/evaluations.npz`
- `tensorboard/`

## Running Evaluation

Run a deterministic sanity-check evaluation with:

```bash
PYTHONPATH=src python -m mario_rl.eval \
  --config configs/llm/llm_v1_final.json \
  --model artifacts/llm_v1_final_seed0/models/best_model.zip \
  --episodes 20
```

Note: `src/mario_rl/eval.py` currently uses `deterministic=True`. The report tables use 20 stochastic episodes (`deterministic=False`) collected during the final analysis workflow; see `REPORT_REFERENCE.md` for those exact numbers.

The project also includes video tooling. Example:

```bash
PYTHONPATH=src python -m mario_rl.video \
  --config configs/llm/llm_v1_final.json \
  --model artifacts/llm_v1_final_seed0/models/best_model.zip \
  --output artifacts/eval_videos/llm_v1_final_best_model.mp4
```

Existing videos:

- `artifacts/eval_videos/llm_v1_final_best_model.mp4`
- `artifacts/eval_videos/human_v3_best_model.mp4`

## LLM Reward Loop

The automated loop asks Claude to generate a reward function, trains PPO for 1M steps, evaluates the best checkpoint, and feeds the results back into the next round.

```bash
ANTHROPIC_API_KEY=your_key PYTHONPATH=src python llm_reward_loop.py
```

The loop runs 5 rounds by default. Each round:

- Saves the generated reward function to `reward_functions/llm/llm_v1_rN.py`
- Writes a config to `configs/llm/llm_v1_test_rN.json`
- Trains for 1M steps
- Evaluates 20 stochastic episodes on the best checkpoint
- Saves prompts and responses under `prompts/`

Important design constraint: reward functions must be stateless. The training uses 4 parallel environments, and module-level reward state is shared across envs, so per-episode state inside the reward module would contaminate training.

## Reward Designs

Baseline uses the environment's native reward only.

Human heuristic rewards forward progress, useful milestones, survival, and flag completion while penalizing death. This produced the strongest final result, including 3/20 completions for v3 and 9/20 completions for v4.

The LLM-generated rewards were interpretable and useful but sensitive to coefficient scale. The best LLM reward was generated in round 1 and used moderate shaping anchored to the native reward. More aggressive later designs sometimes collapsed to deterministic short trajectories.

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Policy | `CnnPolicy` |
| Learning rate | `0.00025` |
| n_steps | `128` |
| batch_size | `64` |
| n_epochs | `4` |
| gamma | `0.99` |
| gae_lambda | `0.95` |
| clip_range | `0.2` |
| ent_coef | `0.02` |
| vf_coef | `0.5` |
| max_grad_norm | `0.5` |
| n_envs | `4` |
| frame_skip | `4` |
| max_stagnation_steps | `200` |

## Report Files

- `REPORT_REFERENCE.md`: final source of truth for results, methodology, and conclusions
- `PROJECT_LOG.md`: chronological experiment notes
- `prompts/`: exact prompt/response history for the LLM reward loop

## Conclusion

LLM-generated reward shaping is a viable bootstrapping method: it doubled baseline progress at 1M steps and produced clean, stateless reward functions. However, human reward engineering remained stronger at higher compute because it encoded better domain knowledge about Mario's obstacles and long-horizon objectives. The best practical workflow is likely hybrid: use the LLM to generate reward drafts and diagnostics, then apply human judgment to stabilize scale and long-term incentives.
