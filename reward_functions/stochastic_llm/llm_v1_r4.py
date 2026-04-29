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