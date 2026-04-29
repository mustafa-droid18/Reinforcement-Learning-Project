# Project Log

Experiment notes for the Mario RL reward shaping project.
Training: PPO (SB3), `SuperMarioBros-1-1-v0`, SIMPLE_MOVEMENT, frame_skip=4, n_envs=4, max_stagnation_steps=200, ent_coef=0.02, device=auto.
All task evals: 20 stochastic episodes (deterministic=False), best_model.zip, unshaped native Mario reward.

---

## Baseline

- 1M steps, no reward shaping.
- Stochastic eval: x_pos mean=579, std=96, min=312, max=722.
- Agent reliably reaches x≈594 but dies at the first pipe. Never clears the first enemy cluster reliably.

---

## Human Heuristic Development

### Iteration history

| # | What changed | Why | Outcome |
|---|---|---|---|
| 1 | Forward progress, backward penalty, score delta, flag +500, death -100, n_envs=4, n_steps=128, 1M steps | Baseline died at x=315, needed basic shaping | Agent still failed at first Goomba |
| 2 | Added sqrt reward transform | 5000:1 reward ratio swamped value function gradients | Stabilized gradient scale |
| 3 | Added FrameSkipWrapper (skip=4), stagnation penalty -0.3, StagnationTerminationWrapper (max=64) | Single-frame A press physically cannot produce a real Mario jump | Agent could now jump over obstacles |
| 4 | Raised stagnation penalty -0.3 → -2.0, removed backward penalty, SIMPLE_MOVEMENT → COMPLEX_MOVEMENT | -0.3 compressed to ≈-0.14 by sqrt transform; backward penalty blocked back-up-then-jump | Agent more willing to back up at pipes |
| 5 | Added milestones [500,1000,1500,2000,2500] at +100 each, 1M → 3M timesteps, ent_coef 0.01 → 0.03 | Agent peaked at x≈856 with no signal to go further | Milestone structure added |
| 6 | Moved first milestone 500 → 200, reduced bonus +100 → +50, revised spacing to [200,400,600,800,1000,1500,2000,2500] | x=500 milestone never fired during early training (agent dying at x=315) | Consistent early feedback signal |
| 7 | COMPLEX_MOVEMENT → SIMPLE_MOVEMENT, max_stagnation 64 → 200 | COMPLEX_MOVEMENT (12 actions) peaked at 620 vs SIMPLE_MOVEMENT 816; 64-step cutoff too tight | Became the v2 final config |
| 8 | Denser milestones every 100px from x=1000–2000, ent_coef 0.03 → 0.02, 3M → 5M steps | v2 inconsistent past x=1000 (500px gap with no milestone signal) | v3 final config |

**Key observation:** A human expert needed 8 deliberate iterations, direct observation of failure modes through videos, and domain knowledge about action sets, frame skip, and reward scale to reach the final heuristic. The LLM comparison is evaluated against this fully-iterated version.

### Human heuristic v2 results (3M steps)

- Best mean reward: 1435 at step 2,525,000.
- Agent plateaued near 816 from steps 1.05M–2.3M, then broke through to 1432–1435 in the final third.
- Best model eval: mean x_pos 1523, 0 flag completions.

### Human heuristic v3 results (5M steps, original — det=True EvalCallback)

- Config: denser milestones every 100px from x=1000–2000 (17 checkpoints), ent_coef=0.02, 5M steps.
- Best mean shaped reward: 2223 at step 4,925,000.
- Task eval (20 stochastic episodes):
  - x_pos: mean=2044, std=594, min=1431, max=3161
  - flags: 3/20
  - all x_pos: [3161, 2472, 2226, 1431, 1521, 3161, 1788, 1793, 1434, 2457, 1523, 1523, 1524, 2457, 3161, 1525, 1522, 1796, 2462, 1943]
- Deterministic eval: x_pos mean=2354, std=0.
- Note: EvalCallback used det=True — training curves not directly comparable with stochastic runs.

### Human heuristic v3 stochastic retrain (5M steps, det=False EvalCallback)

- Same config as v3 original, retrained with det=False EvalCallback methodology.
- Task eval (20 stochastic episodes):
  - x_pos: mean=1934, std=422, min=898, max=2475
  - flags: 0/20
  - all x_pos: [1787, 1794, 1671, 1665, 1654, 1664, 2469, 898, 2467, 2006, 1433, 1957, 1919, 1410, 2472, 2227, 2226, 2471, 2475, 2007]
- Deterministic eval: x_pos mean=1797, std=0.
- Note: Training with det=False eval produced a weaker best checkpoint than the original. Milestone-based rewards benefit from det=True checkpoint selection — the greedy policy reliably hits each waypoint, so the highest-reward checkpoint under det=True is often the strongest policy overall.

### Human heuristic v4 (10M steps)

- 10M steps, same config as v3.
- Task eval (20 stochastic episodes):
  - x_pos: mean=2378, std=935, min=594, max=3161
  - flags: 9/20
  - all x_pos: [2466, 2759, 3161, 3161, 3161, 2466, 594, 2466, 2469, 1228, 3161, 3161, 3161, 1959, 3161, 3161, 594, 594, 1523, 3161]
- Deterministic eval: x_pos mean=2402, std=0.
- Extra 5M steps over v3 yielded dramatic improvement in flag completion (3/20 → 9/20).

---

## LLM Reward Loop — Attempts 1–4 (archived)

Four attempts were run before the final fifth attempt. Key findings extracted:

- **Attempt 1 (aborted after 3 rounds):** Two methodology flaws identified — (1) best-round selection used shaped reward (not comparable across rounds); (2) system/feedback prompts contained subtle reward design hints. Fixed before attempt 2.

- **Attempt 2 (aborted after 2 rounds):** Fixed reproducibility bug (per-round config pointed to mutable file not archived copy). Fixed mislabeled feedback curve (eval uses unshaped reward, is comparable across rounds). Found reward hacking: Claude keyed state by `id(prev_info)` which is unstable — PPO discovered the exploit within 175k steps.

- **Attempt 3 (4 effective rounds):** Fixed id() instability by adding explicit system prompt warning. Best round: R3 at x_pos=722 (deterministic, 10 episodes). All 4 rounds used module-level state keyed by id() despite the warning — gap not fully closed. max_tokens=2048 truncated one response mid-statement.

- **Attempt 4 (5 rounds):** max_tokens bumped to 4096. Retry-on-failure fixed to not advance round counter. Best round: R1 at det x_pos=1139, stoch mean=764. R5 reached 1230 (det) / 806 (stoch mean) using milestone bonuses at 250px intervals. Training diagnostics (entropy_loss, ep_rew_mean, explained_variance) added to feedback message for attempt 5.

All artifacts archived to `archive/`.

---

## LLM Reward Loop — Fifth Attempt (official)

Fixes applied vs fourth attempt:
- `build_feedback_message()` includes last-8-update TensorBoard diagnostics: ep_rew_mean, ep_len_mean, entropy_loss, explained_variance.
- System prompt explicitly states no reliable per-env identifier exists (world/stage/life identical across all 4 envs, id() unstable) — stateless computation is the correct approach.
- EvalCallback: `deterministic=False`, `n_eval_episodes=10` — stochastic checkpoint selection.
- Task eval: 20 episodes, `deterministic=False`.

### Round results (1M steps each, 20-episode stochastic eval)

| Round | Stoch mean | Stoch std | Stoch max | Det mean | Notes |
|-------|-----------|-----------|-----------|----------|-------|
| R1    | 1167      | 242       | 1511      | —        | Best mean; br base capped ±15, 0.5×dx |
| R2    | 689       | 89        | 722       | 434      | Regression; 1.0×dx + hardcoded milestones → entropy collapse |
| R3    | 1110      | 365       | 2011      | 434      | Best single-episode max; 1.0×dx, no idle penalty |
| R4    | 921       | 321       | 1410      | 696      | Collapsed at 450k; 350k checkpoint salvaged |
| R5    | 1140      | 2         | 1142      | 1141     | Ultra-conservative; locked fixed strategy, no upside |

- All 5 rounds fully stateless — the guardrail worked.
- R2 and R4 both collapsed due to aggressive dx multipliers (1.0–1.5×) + dense bonuses → reward hacking short episodes.
- R5 overcorrected after R4 collapse → zero variance (std=2), ceiling-less fixed strategy.
- Deterministic collapse at x=434 in R2/R3 — greedy policy loops on level structure; stochastic sampling breaks past it.
- **R1 selected for final 5M run**: best stochastic mean, no collapse, moderate design.
- **R3 selected for second 5M run**: highest single-episode max (2011), suggesting higher ceiling with more compute.

### LLM v1 Final (R1 reward, 5M steps)

Config: `configs/llm/llm_v1_final.json`, reward: `reward_functions/llm/llm_v1_final.py`.
Best checkpoint: step 1,475,000 (converged early, diminishing returns past 1.5M).

Task eval (20 stochastic episodes):
- x_pos: mean=1374, std=502, min=434, max=2130
- score: mean=305, std=213
- steps: mean=239, std=107
- flags: 0/20
- all x_pos: [1508, 2130, 1417, 2022, 1130, 1128, 1945, 2130, 722, 434, 1495, 1517, 722, 677, 898, 1419, 1512, 1128, 1521, 2023]

Deterministic eval: x_pos mean=1129, std=0 — locked at pipe/gap obstacle.

Note: +18% mean over R1 at 1M (1374 vs 1167); +41% max (2130 vs 1511). Zero flag completions. Deterministic policy traps at x=1129; stochastic sampling occasionally clears it.

### LLM v1 R3 Final (R3 reward, 5M steps)

Config: `configs/llm/llm_v1_r3_final.json`, reward: `reward_functions/llm/llm_v1_r3.py`.
Best checkpoint: step 4,775,000 (kept improving through 4.75M — more sample-efficient than R1).

Task eval (20 stochastic episodes):
- x_pos: mean=1441, std=355, min=696, max=1909
- score: mean=225, std=228
- steps: mean=192
- flags: 0/20
- all x_pos: [1759, 1429, 1672, 1670, 1435, 898, 1896, 1434, 1665, 1434, 1675, 696, 1127, 1674, 1909, 1433, 1528, 830, 898, 1767]

Deterministic eval: x_pos mean=1904, std=0.

Note: R3 slightly edges R1 in stochastic mean (1441 vs 1374) and substantially outperforms in deterministic (1904 vs 1129). R3's upside ceiling is lower — max 1909 vs R1's 2130. Neither clears the flag.

---

## All Runs Complete — Final Comparison

| Agent | Steps | Stoch mean x_pos | Stoch max | Flags | Det x_pos |
|-------|-------|-----------------|-----------|-------|-----------|
| Baseline | 1M | 579 | 722 | 0/20 | — |
| LLM R1 (loop round 1) | 1M | 1167 | 1511 | 0/20 | — |
| LLM v1 Final (R1) | 5M | 1374 | 2130 | 0/20 | 1129 |
| LLM v1 R3 Final | 5M | 1441 | 1909 | 0/20 | 1904 |
| Human v3 (original, det=True trained) | 5M | 2044 | 3161 | 3/20 | 2354 |
| Human v3 (stoch retrain, det=False trained) | 5M | 1934 | 2475 | 0/20 | 1797 |
| Human v4 | 10M | 2378 | 3161 | 9/20 | 2402 |

Improvement over baseline: LLM 1M +101%, LLM 5M +137–149%, Human 5M +235–253%, Human 10M +311%.

---

## Teammate LLM Loop (different prompt, det=True checkpoint selection)

Teammate ran their own LLM reward loop (5 rounds × 1M steps) with a different prompt, plus a 5M final run using their v3 reward function. Evaluated with our standard methodology: 20-episode stochastic eval, unshaped native reward, best_model.zip.

| Agent | Steps | Stoch mean x_pos | Stoch max | Flags |
|-------|-------|-----------------|-----------|-------|
| Teammate LLM v1 | 1M | 786 | 1440 | 0/20 |
| Teammate LLM v2 | 1M | 854 | 1521 | 0/20 |
| Teammate LLM v3 | 1M | 661 | 898 | 0/20 |
| Teammate LLM v4 | 1M | 721 | 1151 | 0/20 |
| Teammate LLM v5 | 1M | 402 | 1433 | 0/20 |
| Teammate LLM v3 (5M) | 5M | 618 | 1434 | 0/20 |

Key observations:
- Best round: v2 at 1M (mean=854) — weaker than our R1 (mean=1167).
- v3 at 5M regressed vs 1M (618 vs 661) — reward function did not scale with more compute.
- v5 nearly collapsed (mean=402, most episodes stuck at x≈303) — same failure mode as our R2/R4.
- Their loop used det=True checkpoint selection; teammate's best (854) < our R1 (1167), showing prompt design and eval methodology matter significantly.

Note: teammate's `llm_loop_results.json` showed 5 episodes per iteration with zero variance — confirms they used deterministic=True for their in-loop eval. Our stochastic re-eval is the canonical comparison.

---

## Final Presentation Tables and Naming Convention

### Naming convention
- "Stochastic LLM - Iteration N": our loop (det=False checkpoint selection)
- "Deterministic LLM - Iteration N": teammate loop (det=True checkpoint selection)
- "Stochastic Human Heuristic": human v3 stoch retrain (det=False trained)
- "Deterministic Human Heuristic": human v3 original (det=True trained)

### Rationale for showing both Iteration 1 and Iteration 3 (5M finals)
- Iteration 1: best stochastic mean at 1M (1167 vs 1110) → selected for best-mean final run
- Iteration 3: highest single-episode max at 1M (2011 vs 1511) → selected to test higher-ceiling potential at 5M

### 1M Reference Table

| Agent | Mean x_pos | Std | Min | Max |
|-------|-----------|-----|-----|-----|
| Baseline | 579 | 96 | 312 | 722 |
| Deterministic LLM - Iteration 3 | 661 | 275 | 296 | 898 |
| Stochastic LLM - Iteration 1 | 1167 | 242 | 898 | 1511 |
| Stochastic LLM - Iteration 3 | 1110 | 365 | 312 | 2011 |

Note: Teammate's Iteration 3 shown (not their best round v2) for direct 1M→5M comparison consistency.

### 5M Comparison Table

| Agent | Mean x_pos | Std | Min | Max | Flags |
|-------|-----------|-----|-----|-----|-------|
| Deterministic LLM | 618 | 332 | 312 | 1434 | 0/20 |
| Stochastic LLM - Iteration 1 | 1374 | 502 | 434 | 2130 | 0/20 |
| Stochastic LLM - Iteration 3 | 1441 | 355 | 696 | 1909 | 0/20 |
| Stochastic Human Heuristic | 1934 | 422 | 898 | 2475 | 0/20 |
| Deterministic Human Heuristic | 2044 | 594 | 1431 | 3161 | 3/20 |

### Human v4 (extended compute)

| Agent | Mean x_pos | Std | Min | Max | Flags |
|-------|-----------|-----|-----|-----|-------|
| Human v4 (10M) | 2378 | 935 | 594 | 3161 | 9/20 |
