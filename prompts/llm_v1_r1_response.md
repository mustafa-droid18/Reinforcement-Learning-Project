# LLM v1 Round 1 — Response

## Raw response

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

## Extracted code

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
