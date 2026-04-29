# LLM v1 Round 3 — Response

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
