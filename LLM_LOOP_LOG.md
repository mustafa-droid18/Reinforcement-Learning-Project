# LLM Reward Loop Log — v1 Fifth Attempt (in progress)

Tracks each round of the automated LLM reward generation loop for v1.
All training: 1M steps, SIMPLE_MOVEMENT, frame_skip=4, n_envs=4, max_stagnation=200, ent_coef=0.02, device=auto.

Previous attempts archived under `archive/`. See `PROJECT_LOG.md` for cross-run summary.

Fixes applied vs fourth attempt:
- `build_feedback_message()` now includes last-8-update training diagnostics from TensorBoard:
  ep_rew_mean (shaped), ep_len_mean, entropy_loss, explained_variance
- System prompt updated with neutral technical descriptions of each diagnostic
- EvalCallback: `deterministic=False`, `n_eval_episodes=10` (stochastic checkpoint selection)
- System prompt gap fixed: now explicitly states there is **no reliable per-env identifier** in
  the info dict (world/stage/life are identical across all 4 envs, id() is unstable), and that
  **stateless reward computation is the correct approach** — do not use module-level state.

---

## Round 1 — complete

**Reward function**: `reward_functions/llm_v1_r1.py`
Design: stateless. dx capped ±5 (+0.5×dx forward, +0.2×dx backward), idle penalty -0.05, score delta /100 capped 5.0, coin +1.0, status change ±5.0, death -25, flag +50 + 0.05×time. Clamp [-100, 100].

**Training curve (shaped reward, eval every 25k):**
Best checkpoint: step 750,000 — mean_rew=957.3, min=622, max=1041, mean_len=181
Notable: oscillations early (125k-225k), recovered and stabilized from 575k onwards.

**Task metrics — 20 stochastic episodes, best_model.zip, unshaped reward:**
- x_pos: mean=1167, std=242, min=706, max=1511
- score: mean=95, std=22
- steps: mean=179, std=49
- all x_pos: [1511, 1127, 1127, 1511, 1127, 722, 1127, 1127, 1419, 1127, 1415, 1127, 706, 1127, 1127, 1127, 1127, 1511, 722, 1419]

---

## Round 2 — complete

**Reward function**: `reward_functions/llm_v1_r2.py`
Design: stateless. dx capped ±5 (+1.0×dx forward, +0.5×dx backward, -0.1 idle), milestone bonuses +5.0 at x=500/800/1200/1600/2000/2400/2800/3100, score /100 cap 5.0, coin +1.0, status ±10/5, death -30, flag +200+0.1×time. Clamp [-100, 250].

**Training curve (shaped reward, eval every 25k):**
Best checkpoint: step 475,000 — mean_rew=625.2, min=617, max=630, mean_len=133
Notable: high oscillation throughout, never broke into higher reward territory.

**Task metrics — 20 stochastic episodes, best_model.zip, unshaped reward:**
- x_pos: mean=689, std=89, min=303, max=722
- score: mean=45, std=50
- steps: mean=160, std=98
- all x_pos: [722, 705, 702, 722, 722, 706, 706, 702, 722, 703, 722, 702, 706, 303, 722, 701, 700, 703, 702, 703]

**Deterministic eval (20 episodes):**
- x_pos: mean=434, std=0, min=434, max=434 — fully collapsed, identical trajectory every episode

**Note:** Regression from R1 (1167→689 stochastic, 434 deterministic). Deterministic collapse at x=434 confirms entropy collapse in best checkpoint. Stronger velocity shaping + milestone bonuses didn't help; stochastic sampling actually outperforms deterministic here.

---

## Round 3 — complete

**Reward function**: `reward_functions/llm_v1_r3.py`
Design: stateless. reward=0.0 base (drops br, adds 0.2×br at end). dx capped ±5 (+1.0×dx forward, +0.1×dx backtrack, no idle penalty). Score min(dscore/50, 10.0), coin +2.0, status +15/-5, death -25, flag +300+0.2×time. Clamp [-100, 350].

**Training curve (shaped reward, eval every 25k):**
Best checkpoint: step 975,000 — mean_rew=1007.4, min=609, max=1325, mean_len=198
Notable: consistent strong performance from 575k onwards, maxes regularly hitting 1300-1562.

**Task metrics — 20 stochastic episodes, best_model.zip, unshaped reward:**
- x_pos: mean=1110, std=365, min=312, max=2011
- score: mean=120, std=144
- steps: mean=182, std=68
- all x_pos: [1128, 1129, 1130, 712, 1413, 1232, 690, 898, 1437, 825, 312, 1228, 1409, 831, 1127, 1130, 1411, 1436, 711, 2011]

**Deterministic eval (20 episodes):**
- x_pos: mean=434, std=0, min=434, max=434 — fully collapsed, identical trajectory every episode

**Note:** Max of 2011 is furthest any LLM model has reached in this attempt. Stochastic mean (1110) close to R1 (1167). Same deterministic collapse at x=434 as R2.

---

## Round 4 — complete

**Reward function**: `reward_functions/llm_v1_r4.py`
Design: stateless. reward=0.0 base. dx capped ±5 (+1.5×dx forward, +0.2×dx backtrack, no idle penalty). Block milestone: +2.0 per 32px block crossed (cur_block > prev_block). Score min(dscore/50, 10.0), coin +2.0, status +15/-5, death -30, flag +500+0.3×time, 0.1×br added. Clamp [-100, 600].

**Training curve (shaped reward, eval every 25k):**
Best checkpoint: step 350,000 — mean_rew=735.6, min=231, max=1306, mean_len=138
Notable: collapsed hard at ~450k, mean_len=40 with near-zero variance from 450k-1M. Aggressive dx + block bonus overfit to short-episode local optimum.

**Task metrics — 20 stochastic episodes, best_model.zip, unshaped reward:**
- x_pos: mean=921, std=321, min=296, max=1410
- score: mean=135, std=131
- steps: mean=195, std=97
- all x_pos: [675, 829, 696, 1409, 1409, 1410, 680, 814, 689, 296, 898, 684, 796, 1409, 677, 898, 696, 1140, 898, 1409]

**Deterministic eval (20 episodes):**
- x_pos: mean=696, std=0, min=696, max=696 — fully collapsed, identical trajectory every episode

**Note:** Despite training collapse, 350k checkpoint retained capable policy. Stochastic 921 better than R2 (689). Multiple episodes reaching 1409-1410.

---

## Round 5 — complete

**Reward function**: `reward_functions/llm_v1_r5.py`
Design: stateless. reward=br (capped ±15) as base. dx capped ±5 (+0.3×dx forward, +0.1×dx backtrack, no idle penalty). Score min(dscore/100, 5.0), coin +1.0, status +8/-3, death -15 (on top of br), flag +100+0.1×time. Clamp [-50, 150]. Most conservative design of all rounds — deliberate retreat after R4 collapse.

**Training curve (shaped reward, eval every 25k):**
Best checkpoint: step 850,000 — mean_rew=1049.2, min=1038, max=1053, mean_len=162
Notable: no full collapse, sustained strong performance from 600k onwards. Tight min/max at best checkpoint indicates very consistent policy.

**Task metrics — 20 stochastic episodes, best_model.zip, unshaped reward:**
- x_pos: mean=1140, std=2, min=1134, max=1142
- score: mean=100, std=0
- steps: mean=162, std=0
- all x_pos: [1141, 1141, 1141, 1137, 1137, 1137, 1141, 1142, 1141, 1141, 1141, 1142, 1141, 1141, 1141, 1141, 1140, 1134, 1142, 1141]

**Deterministic eval (20 episodes):**
- x_pos: mean=1141, std=0 — deterministic and stochastic nearly identical; policy fully converged to one fixed strategy

**Note:** Most consistent policy across all rounds (std=2) but zero upside variance. Reliably stomps one enemy and reaches x=1141 every episode. Conservative shaping produced a locked-in but limited strategy.

---

## Notes

### Feedback loop design
- After each round, script runs 20-episode stochastic eval on `best_model.zip` (deterministic=False) for task metrics
- Full learning curve (every eval checkpoint) included in feedback, labeled as unshaped/comparable
- Training diagnostics (last 8 rollout updates): ep_rew_mean, ep_len_mean, entropy_loss, explained_variance
- Task metrics shown first, learning curve second, training diagnostics third
- No hints, design vocabulary, or pre-classified summaries — raw numbers only
- Claude sees its own previous reward functions and their results across rounds
- Each round's config records the actual archived reward file used (`llm_v1_rN.py`)

### Key observations
- **Stateless guardrail worked**: All 5 rounds produced fully stateless functions — no module-level variables at all
- **R1 best mean** (stochastic x_pos=1167), **R3 best max** (2011) — both used br as base or close to it with moderate shaping
- **R2 and R4 collapsed**: Aggressive velocity shaping (1.0-1.5×dx) + dense bonuses led to local optima; policy learned short-episode reward-hacking strategies
- **R5 over-corrected**: Ultra-conservative shaping produced a locked-in policy (std=2) with no exploration ceiling
- **Deterministic collapse at x=434**: R2 and R3 both show deterministic policy stuck at 434 — likely a local structure in the level (small ledge/enemy) that the greedy policy loops on. Stochastic sampling breaks past it
- **R1 selected for final 5M run**: Best stochastic mean (1167), no collapse, moderate design that balances native br signal with modest shaping
