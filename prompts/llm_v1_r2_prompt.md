# LLM v1 Round 2 — Prompt

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

