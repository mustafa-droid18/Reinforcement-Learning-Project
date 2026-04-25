from __future__ import annotations

import math

# Milestone x_pos checkpoints — each awards a one-time bonus to pull Mario
# through the level section by section. Without these, the flag bonus at x≈3000
# is never discovered, so the agent has no signal to go past its current peak.
_MILESTONES = [500, 1000, 1500, 2000, 2500]
_MILESTONE_BONUS = 100.0


def _sqrt_transform(r: float) -> float:
    # Compresses large reward magnitudes sublinearly while preserving sign.
    # Prevents rare large signals (flag, death) from swamping per-step gradients.
    return math.copysign(math.sqrt(abs(r) + 1) - 1, r) + 0.001 * r


def compute_reward(
    *,
    base_reward: float,
    prev_info: dict,
    info: dict,
    action: int,
    terminated: bool,
    truncated: bool,
) -> float:
    prev_x = float(prev_info.get("x_pos", 0))
    curr_x = float(info.get("x_pos", prev_x))
    delta_x = curr_x - prev_x

    prev_score = float(prev_info.get("score", 0))
    curr_score = float(info.get("score", prev_score))
    score_delta = max(curr_score - prev_score, 0.0)

    prev_time = float(prev_info.get("time", info.get("time", 0)))
    curr_time = float(info.get("time", prev_time))
    time_delta = max(prev_time - curr_time, 0.0)

    reward = float(base_reward)

    # Forward progress
    reward += 0.1 * max(delta_x, 0.0)

    # Mild backward penalty
    reward -= 0.02 * max(-delta_x, 0.0)

    # Score delta: enemy stomps, coins, block hits
    reward += score_delta / 100.0

    # Time-based stagnation: game clock ticks but Mario isn't moving
    if time_delta > 0.0 and abs(delta_x) < 1.0:
        reward -= 0.5 * time_delta

    # Milestone bonuses: one-time rewards for crossing key x_pos thresholds.
    # Fires the step Mario first crosses each checkpoint.
    for milestone in _MILESTONES:
        if prev_x < milestone <= curr_x:
            reward += _MILESTONE_BONUS

    # Flag completion
    if info.get("flag_get"):
        reward += 500.0

    # Death/timeout penalty
    if (terminated or truncated) and not info.get("flag_get", False):
        reward -= 100.0

    return _sqrt_transform(reward)
