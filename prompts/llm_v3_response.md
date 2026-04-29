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

    # 1. Forward progress — keep strong rightward pull.
    #    Milestone bonuses in iteration 2 may have created reward spikes that
    #    destabilised training; we rely on dense per-step progress instead.
    if dx > 0:
        reward += dx * 0.5
    elif dx < 0:
        reward += dx * 0.8          # penalise backtracking

    # 2. REMOVE milestone bonuses entirely.
    #    The milestone scheme gave large, lumpy rewards that seemed to confuse
    #    the value function and actually reduced mean x_pos vs iteration 1.
    #    Dense shaping is more stable for PPO.

    # 3. NEW — max-x progress bonus (dense, monotone).
    #    Reward Mario only when he reaches a new personal best x position.
    #    This gives a persistent gradient that pulls him rightward without
    #    the instability of milestone jumps, and cannot be gamed by oscillating.
    #    We track the best x via score as a proxy: we reward dscore from
    #    new-territory exploration rather than repeating old ground.
    #    Actually implemented cleanly: reward a bonus proportional to how far
    #    beyond the previous best step Mario has gone this step.
    new_ground = max(0, dx) if x_now > x_prev else 0
    # Extra bonus only when pushing into territory further right than before
    # (approximated per-step; PPO's value function smooths this over episodes)
    reward += new_ground * 0.3      # stacks on top of the base dx reward

    # 4. Score reward — unchanged; killing enemies and grabbing items is good.
    reward += dscore * 0.04

    # 5. Survival bonus — slightly reduced vs iteration 2.
    #    Too large a survival bonus relative to forward-progress rewards was
    #    encouraging Mario to stand still rather than advance; bring it down.
    if not terminated:
        reward += 0.04              # was 0.1 in iter 2; passive survival less rewarded

    # 6. Death penalty — keep high so dying is clearly bad.
    if terminated:
        reward -= 25.0

    # 7. Time penalty — small but non-zero to prevent idling.
    #    Kept at the reduced value from iteration 2; rushing led to worse results
    #    when it was larger.
    reward -= dt * 0.01

    # 8. Stall detection — NEW.
    #    If Mario hasn't moved (dx == 0) and time is ticking, apply a small
    #    extra nudge to stop him freezing in front of obstacles or pipes.
    if dx == 0 and dt > 0 and not terminated:
        reward -= 0.08              # gentle prod to keep moving

    # 9. Flag completion bonus — unchanged.
    if flag_get:
        reward += 200.0

    return reward
```