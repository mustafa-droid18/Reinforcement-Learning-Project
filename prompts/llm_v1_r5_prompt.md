# LLM v1 Round 5 — Prompt

## USER

Generate a reward function for training a PPO agent to play Super Mario Bros World 1-1.

---

## ASSISTANT

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = float(br)

    # --- Forward progress shaping ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    # Guard against teleport/wrap on level reset
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Small bonus for moving right, small penalty for moving left or standing still
    if dx > 0:
        reward += 0.5 * dx
    elif dx < 0:
        reward += 0.2 * dx  # mild penalty for backtracking
    else:
        reward -= 0.05  # tiny idle penalty to discourage standing still

    # --- Score-based shaping (stomps, coins, powerups, etc.) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        # Score deltas come in chunks of 100/200/etc; scale modestly
        reward += min(dscore / 100.0, 5.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 1.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 5.0
    elif s_now < s_prev:
        # Took damage but didn't die
        reward -= 5.0

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 50.0
        # Time bonus for finishing quickly
        t = info.get("time", 0)
        reward += 0.05 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 25.0

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp to keep magnitudes reasonable
    if reward > 100.0:
        reward = 100.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

---

## USER

Round 1 results.

Task-level metrics — best model evaluated over 10 episodes using unshaped/default Mario reward with the same evaluation wrappers (comparable across rounds):

  Mean across all episodes:
  - x_pos:          1167 / 3166
  - score:          95
  - coins:          0.0
  - completion:     0%
  - time left:      365
  - episode length: 179 steps

  Best single episode:
  - x_pos:          1511 / 3166
  - score:          100
  - coins:          0
  - flag reached:   False
  - time left:      358
  - episode length: 211 steps

Eval reward learning curve (eval every 25,000 steps, evaluated on the default unshaped Mario reward — comparable across rounds):
  step    25,000: 163.6
  step    50,000: 449.3
  step    75,000: 332.0
  step   100,000: 345.8
  step   125,000: 252.0
  step   150,000: 252.0
  step   175,000: 442.4
  step   200,000: 318.3
  step   225,000: 242.9
  step   250,000: 409.1
  step   275,000: 241.0
  step   300,000: 472.4
  step   325,000: 347.0
  step   350,000: 516.4
  step   375,000: 624.0
  step   400,000: 579.5
  step   425,000: 645.3
  step   450,000: 266.6
  step   475,000: 931.2
  step   500,000: 291.9
  step   525,000: 440.7
  step   550,000: 363.3
  step   575,000: 831.9
  step   600,000: 656.5
  step   625,000: 546.1
  step   650,000: 360.9
  step   675,000: 498.5
  step   700,000: 591.4
  step   725,000: 663.4
  step   750,000: 957.3
  step   775,000: 629.6
  step   800,000: 632.4
  step   825,000: 612.6
  step   850,000: 616.3
  step   875,000: 643.5
  step   900,000: 521.4
  step   925,000: 628.6
  step   950,000: 561.6
  step   975,000: 622.3
  step 1,000,000: 738.6

Training diagnostics (last 8 rollout updates, chronological order):
  ep_rew_mean (shaped):  987.107  →  991.399  →  993.351  →  997.122  →  997.198  →  996.554  →  994.553  →  998.222
  ep_len_mean:           224.010  →  219.460  →  220.190  →  219.630  →  217.620  →  217.860  →  208.920  →  200.980
  entropy_loss:          -0.044  →  -0.167  →  -0.114  →  -0.521  →  -0.464  →  -0.281  →  -0.131  →  -0.050
  explained_variance:    0.988  →  0.979  →  0.996  →  0.976  →  0.992  →  0.990  →  0.983  →  0.989

The reward function used in round 1:

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = float(br)

    # --- Forward progress shaping ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    # Guard against teleport/wrap on level reset
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Small bonus for moving right, small penalty for moving left or standing still
    if dx > 0:
        reward += 0.5 * dx
    elif dx < 0:
        reward += 0.2 * dx  # mild penalty for backtracking
    else:
        reward -= 0.05  # tiny idle penalty to discourage standing still

    # --- Score-based shaping (stomps, coins, powerups, etc.) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        # Score deltas come in chunks of 100/200/etc; scale modestly
        reward += min(dscore / 100.0, 5.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 1.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 5.0
    elif s_now < s_prev:
        # Took damage but didn't die
        reward -= 5.0

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 50.0
        # Time bonus for finishing quickly
        t = info.get("time", 0)
        reward += 0.05 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 25.0

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp to keep magnitudes reasonable
    if reward > 100.0:
        reward = 100.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

Revise the reward function. Return ONLY the revised Python code block.

---

## ASSISTANT

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = float(br)

    # --- Forward progress shaping ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    # Guard against teleport/wrap on level reset
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Stronger forward bias: reward right movement more, penalize backtracking more.
    # Asymmetric to push the agent past obstacles like pipes & enemies.
    if dx > 0:
        reward += 1.0 * dx
    elif dx < 0:
        reward += 0.5 * dx  # stronger penalty for backtracking
    else:
        reward -= 0.1  # idle penalty

    # Milestone bonuses for reaching new x positions (sparse extra signal).
    # Use thresholds based on prev/current crossing - encourages exploring further right.
    # Mario 1-1 has obstacles around 600 (pipes), 1300 (gap), 1600 (gap), 2400 (stairs), 3166 (flag).
    milestones = (500, 800, 1200, 1600, 2000, 2400, 2800, 3100)
    for m in milestones:
        if px < m <= x:
            reward += 5.0

    # --- Score-based shaping (stomps, coins, powerups, etc.) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        reward += min(dscore / 100.0, 5.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 1.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 10.0
    elif s_now < s_prev:
        reward -= 5.0  # took damage but didn't die

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 200.0
        t = info.get("time", 0)
        reward += 0.1 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 30.0

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp to keep magnitudes reasonable
    if reward > 250.0:
        reward = 250.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

---

## USER

Round 2 results.

Task-level metrics — best model evaluated over 10 episodes using unshaped/default Mario reward with the same evaluation wrappers (comparable across rounds):

  Mean across all episodes:
  - x_pos:          689 / 3166
  - score:          45
  - coins:          0.0
  - completion:     0%
  - time left:      369
  - episode length: 160 steps

  Best single episode:
  - x_pos:          722 / 3166
  - score:          100
  - coins:          0
  - flag reached:   False
  - time left:      335
  - episode length: 329 steps

Eval reward learning curve (eval every 25,000 steps, evaluated on the default unshaped Mario reward — comparable across rounds):
  step    25,000: 312.6
  step    50,000: 327.0
  step    75,000: 225.0
  step   100,000: 402.8
  step   125,000: 231.0
  step   150,000: 274.3
  step   175,000: 380.4
  step   200,000: 521.2
  step   225,000: 224.2
  step   250,000: 422.0
  step   275,000: 252.0
  step   300,000: 458.2
  step   325,000: 252.1
  step   350,000: 511.8
  step   375,000: 560.4
  step   400,000: 351.6
  step   425,000: 418.1
  step   450,000: 264.3
  step   475,000: 625.2
  step   500,000: 335.7
  step   525,000: 297.8
  step   550,000: 470.8
  step   575,000: 616.0
  step   600,000: 240.3
  step   625,000: 566.0
  step   650,000: 570.5
  step   675,000: 36.2
  step   700,000: 231.0
  step   725,000: 408.8
  step   750,000: 522.9
  step   775,000: 589.8
  step   800,000: 447.0
  step   825,000: 614.7
  step   850,000: 595.1
  step   875,000: 215.4
  step   900,000: 209.7
  step   925,000: 561.2
  step   950,000: 623.0
  step   975,000: 615.3
  step 1,000,000: 604.5

Training diagnostics (last 8 rollout updates, chronological order):
  ep_rew_mean (shaped):  1053.948  →  1054.908  →  1054.333  →  1054.378  →  1054.873  →  1054.100  →  1054.757  →  1055.267
  ep_len_mean:           188.870  →  188.680  →  190.810  →  190.860  →  187.740  →  189.220  →  185.570  →  183.380
  entropy_loss:          -0.318  →  -0.156  →  -0.353  →  -0.280  →  -0.302  →  -0.450  →  -0.304  →  -0.271
  explained_variance:    0.992  →  0.992  →  0.992  →  0.994  →  0.993  →  0.993  →  0.992  →  0.998

The reward function used in round 2:

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = float(br)

    # --- Forward progress shaping ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    # Guard against teleport/wrap on level reset
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Stronger forward bias: reward right movement more, penalize backtracking more.
    # Asymmetric to push the agent past obstacles like pipes & enemies.
    if dx > 0:
        reward += 1.0 * dx
    elif dx < 0:
        reward += 0.5 * dx  # stronger penalty for backtracking
    else:
        reward -= 0.1  # idle penalty

    # Milestone bonuses for reaching new x positions (sparse extra signal).
    # Use thresholds based on prev/current crossing - encourages exploring further right.
    # Mario 1-1 has obstacles around 600 (pipes), 1300 (gap), 1600 (gap), 2400 (stairs), 3166 (flag).
    milestones = (500, 800, 1200, 1600, 2000, 2400, 2800, 3100)
    for m in milestones:
        if px < m <= x:
            reward += 5.0

    # --- Score-based shaping (stomps, coins, powerups, etc.) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        reward += min(dscore / 100.0, 5.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 1.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 10.0
    elif s_now < s_prev:
        reward -= 5.0  # took damage but didn't die

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 200.0
        t = info.get("time", 0)
        reward += 0.1 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 30.0

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp to keep magnitudes reasonable
    if reward > 250.0:
        reward = 250.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

Revise the reward function. Return ONLY the revised Python code block.

---

## ASSISTANT

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping/level reset)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    # Use base_reward (which already encodes forward progress + time penalty + death)
    # but scale it down so our shaping doesn't get drowned out, and rebuild the pieces
    # we want explicitly.
    reward = 0.0

    # --- Forward progress shaping (the dominant signal) ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Symmetric-ish dx reward. Avoid over-penalizing backtracking, since
    # Mario sometimes needs to back up to time a jump over a pipe/gap.
    if dx > 0:
        reward += 1.0 * dx
    elif dx < 0:
        reward += 0.1 * dx  # mild penalty
    # No per-step idle penalty: stagnation wrapper handles long stalls,
    # and an idle penalty can punish necessary pauses before timing jumps.

    # --- Score-based shaping (stomps, fireballs hitting enemies, flag points) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        # Bigger bonus for stomp/kill events to encourage dealing with enemies.
        reward += min(dscore / 50.0, 10.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 2.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 15.0
    elif s_now < s_prev:
        reward -= 5.0  # took damage but didn't die (lost powerup)

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 300.0
        t = info.get("time", 0)
        reward += 0.2 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    died = life_now < life_prev
    if died:
        reward -= 25.0

    # Add a small fraction of base_reward to keep alignment with native shaping
    # (which already includes a forward-progress + time term).
    reward += 0.2 * br

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp
    if reward > 350.0:
        reward = 350.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

---

## USER

Round 3 results.

Task-level metrics — best model evaluated over 10 episodes using unshaped/default Mario reward with the same evaluation wrappers (comparable across rounds):

  Mean across all episodes:
  - x_pos:          1110 / 3166
  - score:          120
  - coins:          0.1
  - completion:     0%
  - time left:      364
  - episode length: 182 steps

  Best single episode:
  - x_pos:          2011 / 3166
  - score:          600
  - coins:          0
  - flag reached:   False
  - time left:      328
  - episode length: 365 steps

Eval reward learning curve (eval every 25,000 steps, evaluated on the default unshaped Mario reward — comparable across rounds):
  step    25,000: 423.8
  step    50,000: 373.5
  step    75,000: 722.9
  step   100,000: 801.7
  step   125,000: 250.0
  step   150,000: 567.2
  step   175,000: 785.5
  step   200,000: 715.2
  step   225,000: 421.7
  step   250,000: 773.7
  step   275,000: 612.5
  step   300,000: 599.8
  step   325,000: 825.2
  step   350,000: 758.2
  step   375,000: 346.0
  step   400,000: 697.1
  step   425,000: 535.6
  step   450,000: 764.5
  step   475,000: 734.8
  step   500,000: 632.2
  step   525,000: 666.9
  step   550,000: 600.7
  step   575,000: 950.9
  step   600,000: 796.3
  step   625,000: 806.9
  step   650,000: 875.8
  step   675,000: 751.5
  step   700,000: 821.2
  step   725,000: 743.5
  step   750,000: 896.5
  step   775,000: 963.3
  step   800,000: 914.6
  step   825,000: 810.3
  step   850,000: 747.7
  step   875,000: 890.9
  step   900,000: 787.2
  step   925,000: 592.6
  step   950,000: 821.7
  step   975,000: 1007.4
  step 1,000,000: 568.7

Training diagnostics (last 8 rollout updates, chronological order):
  ep_rew_mean (shaped):  610.266  →  607.155  →  593.288  →  589.522  →  595.359  →  591.223  →  596.222  →  596.222
  ep_len_mean:           153.140  →  148.770  →  139.790  →  138.700  →  140.920  →  138.320  →  134.590  →  134.590
  entropy_loss:          -0.495  →  -0.504  →  -0.439  →  -0.417  →  -0.419  →  -0.495  →  -0.455  →  -0.405
  explained_variance:    0.686  →  0.748  →  0.776  →  0.542  →  0.563  →  0.839  →  0.714  →  0.797

The reward function used in round 3:

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
    # Clamp the base reward to avoid extreme spikes (e.g. from x_pos wrapping/level reset)
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    # Use base_reward (which already encodes forward progress + time penalty + death)
    # but scale it down so our shaping doesn't get drowned out, and rebuild the pieces
    # we want explicitly.
    reward = 0.0

    # --- Forward progress shaping (the dominant signal) ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # Symmetric-ish dx reward. Avoid over-penalizing backtracking, since
    # Mario sometimes needs to back up to time a jump over a pipe/gap.
    if dx > 0:
        reward += 1.0 * dx
    elif dx < 0:
        reward += 0.1 * dx  # mild penalty
    # No per-step idle penalty: stagnation wrapper handles long stalls,
    # and an idle penalty can punish necessary pauses before timing jumps.

    # --- Score-based shaping (stomps, fireballs hitting enemies, flag points) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        # Bigger bonus for stomp/kill events to encourage dealing with enemies.
        reward += min(dscore / 50.0, 10.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 2.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 15.0
    elif s_now < s_prev:
        reward -= 5.0  # took damage but didn't die (lost powerup)

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 300.0
        t = info.get("time", 0)
        reward += 0.2 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    died = life_now < life_prev
    if died:
        reward -= 25.0

    # Add a small fraction of base_reward to keep alignment with native shaping
    # (which already includes a forward-progress + time term).
    reward += 0.2 * br

    # Sanity check
    if not math.isfinite(reward):
        return 0.0

    # Final clamp
    if reward > 350.0:
        reward = 350.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

Revise the reward function. Return ONLY the revised Python code block.

---

## ASSISTANT

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
    # Clamp base_reward to avoid level-reset spikes
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = 0.0

    # --- Position deltas ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # --- Forward progress (primary signal) ---
    # Asymmetric: strong reward for moving right, mild penalty for backtracking.
    if dx > 0:
        reward += 1.5 * dx
    elif dx < 0:
        reward += 0.2 * dx
    # No idle penalty (stagnation wrapper handles stalls).

    # --- "Furthest right" milestone bonus ---
    # Reward the agent every time it crosses a new ~32-pixel block to the right.
    # This gives a steady, dense bonus for genuine progress without hardcoded coords.
    block = 32
    prev_block = px // block
    cur_block = x // block
    if cur_block > prev_block:
        reward += 2.0 * (cur_block - prev_block)

    # --- Score-based shaping (stomps, fireballs hitting enemies, flag points) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        reward += min(dscore / 50.0, 10.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 2.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 15.0
    elif s_now < s_prev:
        reward -= 5.0

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 500.0
        t = info.get("time", 0)
        reward += 0.3 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 30.0

    # Small fraction of base reward to keep alignment with native shaping.
    reward += 0.1 * br

    if not math.isfinite(reward):
        return 0.0

    if reward > 600.0:
        reward = 600.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

---

## USER

Round 4 results.

Task-level metrics — best model evaluated over 10 episodes using unshaped/default Mario reward with the same evaluation wrappers (comparable across rounds):

  Mean across all episodes:
  - x_pos:          921 / 3166
  - score:          135
  - coins:          0.2
  - completion:     0%
  - time left:      361
  - episode length: 195 steps

  Best single episode:
  - x_pos:          1410 / 3166
  - score:          100
  - coins:          0
  - flag reached:   False
  - time left:      350
  - episode length: 250 steps

Eval reward learning curve (eval every 25,000 steps, evaluated on the default unshaped Mario reward — comparable across rounds):
  step    25,000: 44.0
  step    50,000: 318.6
  step    75,000: 549.7
  step   100,000: 358.1
  step   125,000: 553.5
  step   150,000: 518.8
  step   175,000: -30.3
  step   200,000: 520.8
  step   225,000: 197.9
  step   250,000: 248.1
  step   275,000: 354.6
  step   300,000: 377.7
  step   325,000: 433.3
  step   350,000: 735.6
  step   375,000: 275.0
  step   400,000: 330.0
  step   425,000: 330.9
  step   450,000: 230.5
  step   475,000: 231.0
  step   500,000: 330.9
  step   525,000: 342.0
  step   550,000: 230.5
  step   575,000: 342.0
  step   600,000: 272.4
  step   625,000: 230.5
  step   650,000: 233.4
  step   675,000: 232.2
  step   700,000: 232.6
  step   725,000: 232.0
  step   750,000: 230.7
  step   775,000: 227.3
  step   800,000: 228.8
  step   825,000: 228.4
  step   850,000: 229.0
  step   875,000: 229.0
  step   900,000: 229.0
  step   925,000: 228.8
  step   950,000: 229.0
  step   975,000: 229.0
  step 1,000,000: 229.0

Training diagnostics (last 8 rollout updates, chronological order):
  ep_rew_mean (shaped):  329.759  →  329.759  →  329.759  →  329.759  →  329.759  →  329.759  →  329.853  →  329.900
  ep_len_mean:           42.000  →  42.000  →  42.000  →  42.000  →  42.000  →  42.000  →  42.000  →  42.000
  entropy_loss:          -0.001  →  -0.001  →  -0.001  →  -0.001  →  -0.001  →  -0.001  →  -0.001  →  -0.001
  explained_variance:    1.000  →  1.000  →  1.000  →  1.000  →  1.000  →  1.000  →  1.000  →  1.000

The reward function used in round 4:

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
    # Clamp base_reward to avoid level-reset spikes
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    reward = 0.0

    # --- Position deltas ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    # --- Forward progress (primary signal) ---
    # Asymmetric: strong reward for moving right, mild penalty for backtracking.
    if dx > 0:
        reward += 1.5 * dx
    elif dx < 0:
        reward += 0.2 * dx
    # No idle penalty (stagnation wrapper handles stalls).

    # --- "Furthest right" milestone bonus ---
    # Reward the agent every time it crosses a new ~32-pixel block to the right.
    # This gives a steady, dense bonus for genuine progress without hardcoded coords.
    block = 32
    prev_block = px // block
    cur_block = x // block
    if cur_block > prev_block:
        reward += 2.0 * (cur_block - prev_block)

    # --- Score-based shaping (stomps, fireballs hitting enemies, flag points) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        reward += min(dscore / 50.0, 10.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 2.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 15.0
    elif s_now < s_prev:
        reward -= 5.0

    # --- Flag reached: large terminal bonus ---
    if info.get("flag_get", False):
        reward += 500.0
        t = info.get("time", 0)
        reward += 0.3 * t

    # --- Death penalty ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 30.0

    # Small fraction of base reward to keep alignment with native shaping.
    reward += 0.1 * br

    if not math.isfinite(reward):
        return 0.0

    if reward > 600.0:
        reward = 600.0
    elif reward < -100.0:
        reward = -100.0

    return float(reward)
```

Revise the reward function. Return ONLY the revised Python code block.

---

## ASSISTANT

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
    # The round-4 run collapsed: entropy_loss ~0, ep_len ~42, eval flatlined.
    # The policy converged to a near-deterministic short-episode strategy that
    # still scored ~330 shaped reward. That suggests reward shaping was
    # exploitable enough to outweigh actually playing the level.
    # Going back to a simpler, more conservative shaping that closely tracks
    # the native base reward (which is known to give a reasonable signal),
    # with modest event bonuses on top.

    # Clamp base reward to avoid level-reset spikes.
    br = base_reward
    if br > 15.0:
        br = 15.0
    elif br < -15.0:
        br = -15.0

    # Start from the native reward (forward progress + time penalty + death).
    reward = float(br)

    # --- Modest extra forward shaping (don't dominate base_reward) ---
    x = info.get("x_pos", 0)
    px = prev_info.get("x_pos", x)
    dx = x - px
    if dx > 5:
        dx = 5
    elif dx < -5:
        dx = -5

    if dx > 0:
        reward += 0.3 * dx
    elif dx < 0:
        reward += 0.1 * dx  # mild penalty

    # --- Score-based shaping (stomps, kills, coin score, flag points) ---
    score = info.get("score", 0)
    pscore = prev_info.get("score", score)
    dscore = score - pscore
    if dscore > 0:
        # Bound to avoid huge spikes (e.g. flag scoring).
        reward += min(dscore / 100.0, 5.0)

    # --- Coin pickup bonus ---
    coins = info.get("coins", 0)
    pcoins = prev_info.get("coins", coins)
    dcoins = coins - pcoins
    if dcoins > 0:
        reward += 1.0 * dcoins

    # --- Power-up / status change ---
    status_rank = {"small": 0, "tall": 1, "fireball": 2}
    s_now = status_rank.get(info.get("status", "small"), 0)
    s_prev = status_rank.get(prev_info.get("status", "small"), 0)
    if s_now > s_prev:
        reward += 8.0
    elif s_now < s_prev:
        reward -= 3.0

    # --- Flag reached: large terminal bonus, scaled to remaining time ---
    if info.get("flag_get", False):
        reward += 100.0
        t = info.get("time", 0)
        reward += 0.1 * t

    # --- Death penalty (in addition to base_reward's -15) ---
    life_now = info.get("life", 2)
    life_prev = prev_info.get("life", life_now)
    if life_now < life_prev:
        reward -= 15.0

    if not math.isfinite(reward):
        return 0.0

    # Final clamp.
    if reward > 150.0:
        reward = 150.0
    elif reward < -50.0:
        reward = -50.0

    return float(reward)
```

---

