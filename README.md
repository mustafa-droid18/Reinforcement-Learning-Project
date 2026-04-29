# Mario RL Reward Shaping

This project trains a PPO agent to play **Super Mario Bros World 1-1** and compares three reward-design strategies:

1. **Baseline**: native Mario environment reward only
2. **Human heuristic**: hand-designed reward shaping with milestone bonuses
3. **LLM-generated rewards**: an automated Claude feedback loop that writes and revises reward functions over 5 iterative rounds

The core research question: can an LLM produce useful reinforcement-learning reward functions from training feedback, and how do those rewards compare against no shaping and human-designed shaping?

Full methodology, result tables, and interpretation are in `REPORT_REFERENCE.md`.

---

## Main Results

All task evaluations: `best_model.zip`, 20 stochastic episodes (`deterministic=False`), unshaped native Mario reward for comparability.

**Naming**: "Stochastic LLM" = LLM loop with `det=False` checkpoint selection. "Deterministic LLM" = LLM loop with `det=True` checkpoint selection. Same distinction applies to Human Heuristic runs.

### 1M Reference (equal compute)

| Agent | Mean x_pos | Std | Min | Max |
|-------|-----------|-----|-----|-----|
| Baseline | 579 | 96 | 312 | 722 |
| Deterministic LLM - Iteration 3 | 661 | 275 | 296 | 898 |
| Stochastic LLM - Iteration 3 | 1110 | 365 | 312 | 2011 |
| Stochastic LLM - Iteration 1 | 1167 | 242 | 898 | 1511 |

### 5M Comparison

| Agent | Mean x_pos | Std | Min | Max | Flags |
|-------|-----------|-----|-----|-----|-------|
| Deterministic LLM | 618 | 332 | 312 | 1434 | 0/20 |
| Stochastic LLM - Iteration 1 | 1374 | 502 | 434 | 2130 | 0/20 |
| Stochastic LLM - Iteration 3 | 1441 | 355 | 696 | 1909 | 0/20 |
| Stochastic Human Heuristic | 1934 | 422 | 898 | 2475 | 0/20 |
| Deterministic Human Heuristic | 2044 | 594 | 1431 | 3161 | 3/20 |

### Human Heuristic at 10M (extended compute)

| Agent | Mean x_pos | Std | Min | Max | Flags |
|-------|-----------|-----|-----|-----|-------|
| Human v4 | 2378 | 935 | 594 | 3161 | 9/20 |

---

## Key Findings

**1. LLM reward shaping substantially outperforms no shaping**
At 1M steps, Stochastic LLM Iteration 1 reached mean x_pos=1167 vs baseline 579 — a +101% improvement with zero human reward engineering effort.

**2. Checkpoint selection methodology matters**
The same reward function trained with `det=True` vs `det=False` checkpoint selection produced different best checkpoints. For milestone-based rewards (human heuristic), `det=True` found a stronger checkpoint because the greedy policy reliably hits each waypoint. For continuous dx-shaping (LLM), `det=False` was more reliable because the greedy policy gets trapped at obstacles — stochastic sampling breaks through them.

**3. Human milestones explain the performance gap**
The human heuristic uses explicit milestone bonuses every 100px from x=1000–2000 (+50 each). These act as a curriculum — the agent always has a reachable local goal 100px ahead. The LLM reward functions used pure continuous shaping with no waypoints, so the greedy policy stalls once it hits a difficult obstacle (pipe at x≈1129 for Iteration 1). This also explains why the human training curve keeps improving through 5M steps while the LLM curves plateau early.

**4. The LLM loop did not improve monotonically**
Iteration 1 had the best stochastic mean at 1M (1167). Later rounds overcorrected — Iterations 2 and 4 collapsed due to aggressive dx multipliers (1.0–1.5×) producing short-episode reward hacking. Iteration 5 over-corrected to ultra-conservative shaping and locked into a fixed strategy (std=2). Iteration 3 was selected as the second final run because it had the highest single-episode max (2011), suggesting a higher ceiling with more compute.

**5. Reward coefficient sensitivity is a real failure mode**
Small changes to the dx multiplier (0.3×, 0.5×, 1.0×, 1.5×) had dramatic effects on stability. At 1.0–1.5×, entropy collapsed and the agent learned short fixed trajectories that maximised the shaped reward without level progress. Training diagnostics (entropy_loss, ep_len_mean) were effective early-warning signals.

**6. Stochastic vs deterministic eval gap**
For PPO, the stochastic policy (sampled actions) consistently outperforms the deterministic policy (greedy argmax) on this task. Stochastic LLM Iteration 1 at 5M: stoch mean=1374 vs det=1129. Human v3 original: stoch mean=2044 vs det=2354 (here det wins because milestones make the greedy policy strong). Always report both for PPO.

**7. Prompt design affects LLM loop performance**
The teammate's loop (same environment, different prompt, `det=True` checkpoint selection) achieved best mean=854 at 1M vs our 1167, and their 5M final reached only 618 vs our 1374–1441. The stateless constraint, stochastic eval methodology, and feedback structure (training diagnostics + learning curve) in our prompt design appear to be significant factors.

---

## Environment

- Game: `SuperMarioBros-1-1-v0`
- Algorithm: PPO from Stable-Baselines3
- Policy: `CnnPolicy`
- Action set: `SIMPLE_MOVEMENT` (7 actions)
- Observation: grayscale `84×84`, 4-frame stack
- Frame skip: 4
- Parallel envs: 4
- Episode termination: death, time-out, or 200 steps with no x-position progress
- Key x_pos landmarks: x=315 (first enemy), x≈700 (first pipe), x≈1200 (gap section), x=3166 (flag)

The agent observes pixels only. Reward functions receive Mario metadata: `x_pos`, `score`, `coins`, `status`, `time`, `flag_get`, `life`.

---

## Repository Layout

```text
configs/baseline.json         Baseline (no shaping)
configs/human/                Human heuristic training configs
configs/llm/                  LLM reward training configs
reward_functions/human_heuristic.py   Human reward function
reward_functions/llm/         LLM reward functions from each round + finals
src/mario_rl/                 Training, evaluation, env construction, video tools
prompts/                      Saved LLM prompts and responses for all 5 rounds
llm_reward_loop.py            Automated LLM reward-generation loop
PROJECT_LOG.md                Chronological experiment notes
REPORT_REFERENCE.md           Full methodology, result tables, and conclusions
artifacts/                    Models, TensorBoard logs, eval files, and videos
```

---

## Setup

Python 3.10 was used for this project.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, this project defaults to `device: auto`. For this Mario CNN workload, CPU is competitive because environment stepping is the main bottleneck.

---

## Running Training

```bash
# Baseline
PYTHONPATH=src python -m mario_rl.train --config configs/baseline.json

# Human heuristic v3 (5M)
PYTHONPATH=src python -m mario_rl.train --config configs/human/human_heuristic_v3.json

# Human heuristic v4 (10M)
PYTHONPATH=src python -m mario_rl.train --config configs/human/human_heuristic_v4.json

# Stochastic LLM - Iteration 1 final (5M)
PYTHONPATH=src python -m mario_rl.train --config configs/llm/llm_v1_final.json

# Stochastic LLM - Iteration 3 final (5M)
PYTHONPATH=src python -m mario_rl.train --config configs/llm/llm_v1_r3_final.json
```

Training outputs go to `artifacts/<experiment_name>_seed0/`:
- `models/best_model.zip` — best checkpoint by eval reward
- `models/final_model.zip` — last checkpoint
- `eval/evaluations.npz` — per-checkpoint eval rewards and episode lengths
- `tensorboard/` — training diagnostics

---

## Running Evaluation

```bash
PYTHONPATH=src python -m mario_rl.eval \
  --config configs/llm/llm_v1_final.json \
  --model artifacts/llm_v1_final_seed0/models/best_model.zip \
  --episodes 20
```

Note: `mario_rl.eval` uses `deterministic=True`. The result tables above use 20 stochastic episodes (`deterministic=False`) — see `REPORT_REFERENCE.md` for exact numbers.

```bash
# Record a video of the best model
PYTHONPATH=src python -m mario_rl.video \
  --config configs/llm/llm_v1_final.json \
  --model artifacts/llm_v1_final_seed0/models/best_model.zip \
  --output artifacts/eval_videos/llm_v1_final_best_model.mp4
```

Existing videos:
- `artifacts/eval_videos/llm_v1_final_best_model.mp4`
- `artifacts/eval_videos/human_v3_best_model.mp4`

---

## LLM Reward Loop

The loop asks Claude (Anthropic API) to generate a reward function, trains PPO for 1M steps, evaluates the best checkpoint, and feeds results back for the next revision.

```bash
ANTHROPIC_API_KEY=your_key PYTHONPATH=src python llm_reward_loop.py
```

Configuration (model, rounds, train config) is set via constants at the top of `llm_reward_loop.py`. Each round:
- Generates a reward function and saves it to `reward_functions/llm/llm_v1_rN.py`
- Writes a config to `configs/llm/llm_v1_test_rN.json`
- Trains for 1M steps
- Evaluates 20 stochastic episodes on the best checkpoint
- Feeds task metrics + training diagnostics (entropy, ep_len, explained_variance) back to Claude
- Saves all prompts and responses under `prompts/`

**Critical constraint**: reward functions must be stateless. Training uses 4 parallel environments that share all module-level state — per-episode state inside the reward module contaminates all 4 envs simultaneously.

---

## Reward Designs

**Baseline**: native environment reward only. Agent reliably reaches x≈594 (past first enemy) but dies at the first pipe.

**Human heuristic** (`reward_functions/human_heuristic.py`): forward progress bonus, dense milestone bonuses every 100px from x=1000–2000 (+50 each), stagnation penalty, death penalty (−100), flag bonus (+500). The milestones act as a curriculum — the agent always has a reachable local goal. This produced the strongest results: 3/20 flag completions at 5M, 9/20 at 10M.

**LLM-generated** (`reward_functions/llm/`): 5 iterative rounds, each 1M steps. Best design (Iteration 1): native reward as base (capped ±15), +0.5×dx forward, modest score/coin/status bonuses, death −25, flag +50. Interpretable, stable, and fully stateless — but no milestones, so the greedy policy stalls at obstacles.

---

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

---

## Report Files

- `REPORT_REFERENCE.md` — full methodology, result tables, and conclusions
- `PROJECT_LOG.md` — chronological experiment notes
- `prompts/` — exact prompt/response history for all 5 LLM rounds
