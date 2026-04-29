# Project Reference — RL Reward Shaping for Super Mario Bros

## Overview

We train a PPO agent to play Super Mario Bros (World 1-1) using three reward designs:
1. **Baseline** — no shaping, native environment reward only
2. **Human heuristic** — hand-designed reward function (v3), 5M steps
3. **LLM-generated** — iterative reward loop using Claude, 5 rounds × 1M steps, then 5M final run

All agents: PPO, `SIMPLE_MOVEMENT` (7 actions), frame_skip=4, n_envs=4, grayscale 84×84, frame_stack=4.
All evals: 20 stochastic episodes (`deterministic=False`) on `best_model.zip`, unshaped native reward.

---

## Environment

- **Game**: SuperMarioBros-1-1-v0 (gym-super-mario-bros)
- **Action space**: SIMPLE_MOVEMENT — 7 actions (idle, right, right+jump, right+run, right+run+jump, jump, left)
- **Observation**: grayscale 84×84, 4-frame stack
- **Frame skip**: 4 (each action held for ~67ms)
- **Episode termination**: death, time-out, or stagnation (200 steps with no x_pos progress)
- **Key positions**: x=315 (first enemy), x~700 (first pipe), x~1200 (gap section), x=3166 (flag)
- **Native reward**: ~+1 to +15 per step for forward progress, -15 on death

---

## Baseline

**Config**: 1M steps, no reward shaping, `n_eval_episodes=5`, `deterministic=True` during training eval.

**20-episode stochastic eval:**
- x_pos: mean=579, std=96, min=312, max=722
- score: mean=35, std=48
- All x_pos: [314, 594, 594, 594, 594, 594, 594, 594, 722, 594, 594, 594, 594, 312, 594, 594, 594, 722, 594, 594]

**Interpretation**: Agent reliably reaches ~594 (past first enemy area) but dies at the first pipe/gap. No reward shaping gives a weak signal for exploration.

---

## Human Heuristic — All Runs Summary

| Run | Steps | Train eval methodology | Stoch mean x_pos | Stoch max | Flags | Det mean |
|-----|-------|----------------------|-----------------|-----------|-------|---------|
| v3 (original) | 5M | det=True, n=5 | 2044 | 3161 | 3/20 | 2354 |
| v3 stochastic retrain | 5M | det=False, n=10 | 1934 | 2475 | 0/20 | 1797 |
| **v4** | **10M** | det=True, n=5 | **2378** | **3161** | **9/20** | **2402** |

Note: v3 retrain with `deterministic=False` performed worse than original v3 with `deterministic=True` — the original's best checkpoint at 4.925M was simply a stronger policy. Methodology change alone does not guarantee improvement.

---

## Human Heuristic v3

**Design** (hand-crafted):
- Forward progress: +dx per step (rightward movement bonus)
- Milestone bonuses at key x_pos thresholds
- Death penalty, flag bonus, time bonus
- 5M training steps

**Training**: Best checkpoint at step 4,925,000 (mean shaped reward=2223, mean_len=455).

**20-episode stochastic eval (best_model.zip):**
- x_pos: mean=2044, std=594, min=1431, max=3161
- score: mean=380, std=312
- steps: mean=239, std=124
- All x_pos: [3161, 2472, 2226, 1431, 1521, 3161, 1788, 1793, 1434, 2457, 1523, 1523, 1524, 2457, 3161, 1525, 1522, 1796, 2462, 1943]

**Deterministic eval**: x_pos=2354 every episode (std=0), score=500, steps=455.

**Key result**: 3/20 episodes completed the level (reached flag at x=3161). Strong, consistent performance past the gap section.

**Video**: `artifacts/eval_videos/human_v3_best_model.mp4`

---

## Human Heuristic — 1M Step Performance

No dedicated 1M-step run was evaluated with task metrics (best_model.zip is always saved at peak performance, which for v3 was step 4,925,000). The training-time shaped reward curve gives a proxy:

**Human v3 at 1M** (training eval, shaped reward):
- Highly oscillating throughout first 1M steps — best checkpoint was 610 at step 925k, but at step 1M itself only 246
- Zero-variance deterministic episodes dominate this phase (min=max at each checkpoint) → policy not yet exploring

**Human v2 at 3M** (best checkpoint step 2,525,000):
- Deterministic eval (original methodology): x_pos=1523 every episode, score=0, steps=151
- Ran 3M steps total

**Key finding: Human heuristic needed 3M+ steps to break through consistently.**
- Steps 1–1.4M: oscillating, peak shaped reward ~600–800, comparable to LLM v1 R1 at same compute
- Steps 1.4–3.4M: sporadic strong episodes but no sustained improvement
- Steps 3.4M+: consistent 1000–1700 shaped reward, multiple episodes reaching x>2000
- Steps 4.5M+: consistently reaching 1300–1700 shaped reward, flag completions emerging

**Interpretation**: At equal compute (1M steps), LLM v1 (x_pos mean=1167) likely outperforms or matches the human heuristic, which had not yet converged. The human heuristic's advantage is in long-run sample efficiency — with 5× the steps, it pulls far ahead. This suggests the human reward signal is richer but slower to exploit, while the LLM reward signal is simpler but learnable faster.

---

## LLM Reward Loop — Methodology

### Setup
- **Model**: Claude Opus 4 (`claude-opus-4-7`, Anthropic API)
- **Rounds**: 5 rounds per attempt, each 1M steps
- **Feedback per round**: task metrics (x_pos, score, steps), full eval learning curve, last-8-update training diagnostics (ep_rew_mean, ep_len_mean, entropy_loss, explained_variance from TensorBoard)
- **Eval during training**: `deterministic=False`, `n_eval_episodes=10` (stochastic checkpoint selection)
- **Task eval**: 20 stochastic episodes on best_model.zip, unshaped native reward
- **Claude sees**: its own previous reward functions + results for all prior rounds

### Key system prompt constraints
- Stateless computation only — no module-level variables (4 parallel envs share all module state, no reliable per-env identifier)
- Standard library only (no numpy/torch)
- Must return finite float
- No hardcoded specific x_pos coordinates (softened: general thresholds acceptable)

### Feedback message structure
1. Task metrics (x_pos, score, steps) from previous round's best model
2. Full shaped-reward learning curve (every 25k step checkpoint)
3. Training diagnostics: ep_rew_mean, ep_len_mean, entropy_loss, explained_variance (last 8 updates)

---

## LLM v1 — Fifth Attempt (Official)

Five previous attempts refined the methodology (stateless guardrail, TensorBoard diagnostics, stochastic eval). The fifth attempt is the final official run.

### Round Results

| Round | Design highlights | Best ckpt step | Stoch mean x_pos | Stoch max | Stoch std | Det mean |
|-------|------------------|---------------|-----------------|-----------|-----------|---------|
| R1 | br as base, +0.5×dx, -0.05 idle, score/100, coin+1, status±5, death-25, flag+50 | 750k | **1167** | 1511 | 242 | — |
| R2 | +1.0×dx, +0.5×backtrack, milestones at 500/800/…/3100 (+5 each), flag+200 | 475k | 689 | 722 | 89 | 434 |
| R3 | zero-base+0.2×br, +1.0×dx, no idle, score/50, coin+2, status+15, flag+300 | 975k | 1110 | **2011** | 365 | 434 |
| R4 | +1.5×dx, 32px block bonus (+2 per block), score/50, flag+500 | 350k | 921 | 1410 | 321 | 696 |
| R5 | br as base, +0.3×dx (most conservative), score/100, status±8/3, flag+100 | 850k | 1140 | 1142 | **2** | 1141 |

### Observations
- **Guardrail worked**: All 5 rounds produced fully stateless functions
- **R1 best mean** (1167): conservative, native br anchored — discovered in round 1
- **R3 best max** (2011): stronger shaping, zero-base design
- **R2 and R4 collapsed**: aggressive dx (1.0–1.5×) + dense bonuses → short-episode reward hacking, entropy collapse
- **R5 over-corrected**: ultra-conservative → locked-in fixed strategy (std=2), no exploration ceiling
- **Iterative loop finding**: the first round produced the best design — subsequent rounds overcorrected, showing that reward shaping is sensitive to coefficient scale
- **Deterministic collapse at x=434** (R2, R3): greedy policy traps on a specific obstacle; stochastic sampling breaks past it

### Selected for final 5M run: R1
Rationale: best stochastic mean, no collapse, proven stable design.

---

## LLM v1 Final — 5M Run

**Reward function**: `reward_functions/llm/llm_v1_final.py` (identical to R1)
**Config**: 5M steps, same hyperparameters as rounds.

**Training**: Best checkpoint at step 1,475,000 (mean shaped reward=1127.9, max=1635).

**20-episode stochastic eval (best_model.zip):**
- x_pos: mean=1374, std=502, min=434, max=2130
- score: mean=305, std=213
- steps: mean=239, std=107
- All x_pos: [1508, 2130, 1417, 2022, 1130, 1128, 1945, 2130, 722, 434, 1495, 1517, 722, 677, 898, 1419, 1512, 1128, 1521, 2023]

**Deterministic eval**: x_pos=1129 every episode (std=0), score=200, steps=143.

**Note on deterministic vs stochastic gap**: Best checkpoint selected under stochastic eval (deterministic=False). The deterministic policy hits a fixed failure point at x=1129 (pipe/gap). Stochastic sampling occasionally draws the correct action at that obstacle and pushes past to 2000+. This is expected behavior for PPO — the policy is stochastic by design.

**Video**: `artifacts/eval_videos/llm_v1_final_best_model.mp4`

---

## LLM v1 R3 Final — 5M Run

**Reward function**: `reward_functions/llm/llm_v1_r3.py`
**Config**: 5M steps, same hyperparameters as other runs.

**Training**: Best checkpoint at step 4,775,000 (mean shaped reward=1462.3, max=1814).

**20-episode stochastic eval (best_model.zip):**
- x_pos: mean=1441, std=355, min=696, max=1909
- score: mean=225, std=228
- steps: mean=192
- flags: 0/20
- All x_pos: [1759, 1429, 1672, 1670, 1435, 898, 1896, 1434, 1665, 1434, 1675, 696, 1127, 1674, 1909, 1433, 1528, 830, 898, 1767]

**Deterministic eval**: x_pos=1904 every episode (std=0).

**Key finding**: R3 at 5M edges ahead of R1 in stochastic mean (1441 vs 1374) and significantly outperforms in deterministic (1904 vs 1129). R3's greedy policy reliably reaches a deeper checkpoint. However, R3's max (1909) is lower than R1's (2130) — R1's higher variance occasionally produces exceptional episodes. Neither clears the flag.

---

## Final Comparison

### Naming Convention
- **LLM, det=False selection**: Our LLM loop runs — stochastic checkpoint selection, n=10 eval episodes
- **LLM, det=True selection**: Teammate LLM loop runs — deterministic checkpoint selection, n=5 eval episodes
- **Stochastic Human Heuristic**: Human v3 stoch retrain — `det=False` checkpoint selection
- **Deterministic Human Heuristic**: Human v3 original — `det=True` checkpoint selection
- All task evals: 20-episode stochastic (det=False), unshaped native reward

### Rationale for Iteration 1 and Iteration 3
- Iteration 1 (R1): best stochastic mean at 1M (1167 vs 1110) — most reliable
- Iteration 3 (R3): highest single-episode max at 1M (2011 vs 1511) — highest ceiling, selected to test if more compute would unlock further progress

### 1M Reference Table

| Agent | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Baseline | 579 | 96 | 312 | 722 |
| LLM, det=True selection - Iteration 3 | 661 | 275 | 296 | 898 |
| LLM, det=False selection - Iteration 1 | 1167 | 242 | 898 | 1511 |
| LLM, det=False selection - Iteration 3 | 1110 | 365 | 312 | 2011 |

Note: Teammate's Iteration 3 shown (not best round v2) for direct 1M→5M comparison consistency.

### 5M Comparison Table

| Agent | Mean | Std | Min | Max | Flags | Det x_pos |
|-------|------|-----|-----|-----|-------|-----------|
| LLM, det=True selection | 618 | 332 | 312 | 1434 | 0/20 | — |
| LLM, det=False selection - Iteration 1 | 1374 | 502 | 434 | 2130 | 0/20 | 1129 |
| LLM, det=False selection - Iteration 3 | 1441 | 355 | 696 | 1909 | 0/20 | 1904 |
| Stochastic Human Heuristic | 1934 | 422 | 898 | 2475 | 0/20 | 1797 |
| Deterministic Human Heuristic | 2044 | 594 | 1431 | 3161 | 3/20 | 2354 |

### Human v4 (10M)

| Agent | Mean | Std | Min | Max | Flags | Det x_pos |
|-------|------|-----|-----|-----|-------|-----------|
| Human v4 | 2378 | 935 | 594 | 3161 | 9/20 | 2402 |

Separate from the 5M table — not a fair compute comparison. Shown to demonstrate the human heuristic's scaling ceiling.

### Improvement over baseline
- LLM det=False selection 1M: **+101%** (579→1167)
- LLM det=False selection 5M: **+137%** (579→1374)
- Deterministic Human 5M: **+253%** (579→2044)
- Human 10M: **+311%** (579→2378)

### Sample efficiency
- LLM det=False selection converges fast — most gains by 1M, 5M adds only +18%
- Human heuristic slower to converge but scales better — milestone structure keeps providing signal through 5M and beyond
- Iteration 3 (5M) edges Iteration 1 (5M) in mean (1441 vs 1374) and det (1904 vs 1129), but Iteration 1 has higher upside per episode (max 2130 vs 1909)
- Checkpoint selection methodology interacts with reward design: milestone-based rewards benefit from det=True selection; continuous dx shaping benefits from det=False to avoid greedy traps

---

## Key Findings

1. **LLM shaping substantially outperforms no shaping**: 2× improvement at 1M steps
2. **Human reward engineering still wins at equal compute**: 48% better mean x_pos, flag completion vs none
3. **Reward sensitivity**: Small coefficient changes (0.5×dx vs 1.0×dx vs 1.5×dx) dramatically affect stability — R2/R4 collapsed, R1/R3/R5 did not
4. **Iterative loop convergence**: The loop did not progressively improve round-over-round; R1 was best. Later rounds overcorrected based on failure analysis
5. **Stochastic vs deterministic eval**: For PPO, stochastic eval is more informative — deterministic can underestimate capability by 30-50%
6. **Entropy collapse is a real failure mode**: Aggressive shaping pushed R2/R4 into near-zero entropy, identical short episodes, reward hacking without level progress

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/baseline_seed0/` | Baseline training run |
| `artifacts/human_heuristic_v3_seed0/` | Human heuristic v3, 5M steps |
| `artifacts/llm_v1_final_seed0/` | LLM final, 5M steps (R1 reward) |
| `artifacts/llm_v1_r3_final_seed0/` | LLM R3, 5M steps |
| `artifacts/eval_videos/llm_v1_final_best_model.mp4` | LLM final best model video |
| `artifacts/eval_videos/human_v3_best_model.mp4` | Human v3 best model video |
| `reward_functions/llm/llm_v1_final.py` | Final LLM reward function |
| `reward_functions/human_heuristic.py` | Human heuristic reward function |
| `llm_reward_loop.py` | Full LLM loop implementation |

---

## Conclusion

### What worked
- **LLM-generated reward shaping is effective**: The automated loop produced a functional, non-trivial reward function in round 1 that doubled the baseline x_pos at equal compute. The function was fully stateless, interpretable, and trained stably.
- **Iterative feedback loop is informative**: Even though R1 was the best round, the loop correctly identified failure modes (entropy collapse in R2/R4, over-conservatism in R5) and Claude's self-diagnosis in R5's comments ("entropy_loss ~0, ep_len ~42, eval flatlined") showed the training diagnostics were being used meaningfully.
- **Stochastic eval is more appropriate for PPO**: Switching from deterministic to stochastic evaluation (deterministic=False) revealed genuine policy capability that deterministic eval masked. The "lucky snapshot" problem (deterministic eval saving a temporarily good checkpoint) was a real methodological issue corrected in this work.
- **Guardrails work**: Making the stateless constraint explicit in the system prompt (no module-level state, no reliable per-env ID) eliminated a recurring implementation bug across all 5 rounds.

### What didn't work / limitations
- **Iterative improvement was not monotonic**: R1 was the best round. Later rounds overcorrected — oscillating between aggressive designs that collapsed and conservative designs that plateaued. The loop did not progressively improve performance.
- **LLM reward shaping does not match human reward engineering at 5M steps**: Human v3 (x_pos=2044 mean, 3/20 flag completions) substantially outperforms LLM final (x_pos=1374 mean, 0/20 flag completions). Human domain knowledge — knowing where obstacles are, understanding Mario's physics — produces a better reward signal.
- **Scale matters for human heuristic**: At 1M steps the human heuristic had not converged; by 5M it had. The LLM reward function converged faster (strong at 1M) but had a lower ceiling (did not continue improving as sharply at 5M).
- **Reward coefficient sensitivity**: Small changes to the dx multiplier (0.3×, 0.5×, 1.0×, 1.5×) had dramatic effects — R2 and R4 collapsed, R1 and R5 were stable. The LLM did not reliably find the right scale.

### Broader takeaways
- LLM reward generation is a viable bootstrapping tool: it outperforms no shaping and produces interpretable functions, but currently requires expert-designed evaluation methodology to be reliable
- The gap between LLM and human reward engineering shrinks at low compute budgets and widens at high compute — human reward functions are harder to discover but scale better
- Entropy collapse (near-zero entropy_loss) is a detectable and diagnosable failure mode when training diagnostics are included in the feedback loop
- For future work: combining LLM-generated reward with human refinement (LLM as first draft, human edits coefficients) may close the gap

---

## PPO Hyperparameters (all runs)

| Parameter | Value |
|-----------|-------|
| Policy | CnnPolicy |
| Learning rate | 0.00025 |
| n_steps | 128 |
| batch_size | 64 |
| n_epochs | 4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.02 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| n_envs | 4 |
| frame_skip | 4 |
| max_stagnation | 200 steps |
