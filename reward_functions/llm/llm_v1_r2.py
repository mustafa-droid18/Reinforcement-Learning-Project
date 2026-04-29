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