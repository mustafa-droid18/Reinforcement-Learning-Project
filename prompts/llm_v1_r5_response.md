# LLM v1 Round 5 — Response

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
