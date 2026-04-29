```python
def compute_reward(*, base_reward: float, prev_info: dict, info: dict,
                   action: int, terminated: bool, truncated: bool) -> float:

    # --- Extract useful quantities ---
    x_now  = info.get("x_pos", 0)
    x_prev = prev_info.get("x_pos", 0)
    dx     = x_now - x_prev

    score_now  = info.get("score", 0)
    score_prev = prev_info.get("score", 0)
    dscore     = score_now - score_prev

    time_now  = info.get("time", 400)
    time_prev = prev_info.get("time", 400)
    dt        = time_prev - time_now

    flag_get = info.get("flag_get", False)

    reward = 0.0

    # -------------------------------------------------------------------------
    # DIAGNOSIS: We have been oscillating between x≈600-1137 across iterations.
    # The main culprits appear to be:
    #   (a) Reward hacking / instability from stacking too many shaping terms
    #       that interact poorly with each other.
    #   (b) The potential shaping term added in iter 4 created a constant
    #       background reward that the agent could exploit without actual progress.
    #   (c) Death penalty oscillation: too high = freezing, too low = reckless dying.
    #
    # STRATEGY: Return to a MINIMAL, CLEAN reward function based on what
    # actually worked best (iter 3 reached x≈1137). Strip out experimental
    # terms, keep only the three most reliable signals:
    #   1. Dense dx progress (strong)
    #   2. Score delta (enemy kills / coins)
    #   3. Death penalty at a moderate level
    # Then add ONE new idea: a clipped velocity reward to reward sustained
    # rightward momentum rather than individual steps.
    # -------------------------------------------------------------------------

    # 1. Forward progress — strong dense signal, same as iter 3 best performer.
    #    Asymmetric: reward right movement more than penalising left so the
    #    agent isn't paralysed by minor fluctuations near obstacles.
    if dx > 0:
        reward += dx * 0.5
    elif dx < 0:
        reward += dx * 0.3          # soft backtrack penalty only

    # 2. Sustained momentum bonus — NEW, replaces the unstable potential term.
    #    If Mario moved right by more than 1 pixel this step (i.e. actually
    #    running, not just drifting), give a small extra bonus. This rewards
    #    consistent forward movement without creating the always-on background
    #    gradient that the potential term produced.
    if dx >= 2:
        reward += 0.5               # flat bonus for clear rightward movement

    # 3. Score reward — keep, it's consistently useful across all iterations.
    #    Enemy kills and coins provide a natural signal for engaging with the
    #    level rather than passively running.
    reward += dscore * 0.03         # slightly reduced to keep magnitudes balanced

    # 4. Survival bonus — minimal, just enough to signal "alive = good".
    #    We do NOT want this to dominate; Mario should prefer moving over surviving.
    if not terminated:
        reward += 0.02

    # 5. Death penalty — set at intermediate level between iter 2 (-25) and iter 4 (-10).
    #    -15 was empirically the level we haven't tried that sits between
    #    "too conservative" (-25) and "too reckless" (-10).
    if terminated:
        reward -= 15.0

    # 6. Time penalty — REMOVED stall penalty, keep only a very small time cost.
    #    The stall penalty interacted badly with the death penalty near enemies,
    #    making the agent thrash rather than plan. A tiny time cost is enough
    #    to prevent indefinite idling.
    reward -= dt * 0.01

    # 7. Flag completion bonus — unchanged.
    if flag_get:
        reward += 200.0

    return reward
```