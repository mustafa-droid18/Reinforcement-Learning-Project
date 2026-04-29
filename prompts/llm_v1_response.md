```python
def compute_reward(*, base_reward: float, prev_info: dict, info: dict,
                   action: int, terminated: bool, truncated: bool) -> float:

    # --- Extract useful quantities ---
    x_now  = info.get("x_pos", 0)
    x_prev = prev_info.get("x_pos", 0)
    dx     = x_now - x_prev                        # rightward displacement this step

    score_now  = info.get("score", 0)
    score_prev = prev_info.get("score", 0)
    dscore     = score_now - score_prev             # score delta (killing enemies, coins, etc.)

    time_now  = info.get("time", 400)
    time_prev = prev_info.get("time", 400)
    dt        = time_prev - time_now                # time ticked down (positive when time passes)

    flag_get = info.get("flag_get", False)

    reward = 0.0

    # 1. Forward progress reward
    #    Reward moving right, mildly penalise moving left to discourage stalling.
    if dx > 0:
        reward += dx * 0.3          # positive progress bonus
    else:
        reward += dx * 0.5          # stronger nudge against backtracking

    # 2. Score-based reward
    #    Score increases when Mario kills enemies, grabs coins, hits blocks.
    #    This teaches the agent that killing Goombas (instead of dying to them) is good.
    reward += dscore * 0.025        # scale raw score delta to moderate magnitude

    # 3. Survival reward
    #    A small per-step bonus just for staying alive encourages the agent
    #    to avoid obstacles rather than rushing into them.
    if not terminated:
        reward += 0.05

    # 4. Death penalty
    #    Large negative reward for dying so the agent learns to avoid Goombas.
    if terminated:
        reward -= 15.0

    # 5. Time penalty
    #    Small cost per tick of game-clock consumed; keeps Mario moving
    #    without being so large that it overrides caution.
    reward -= dt * 0.02

    # 6. Flag / level completion bonus
    #    Big reward for reaching the flag pole.
    if flag_get:
        reward += 200.0

    return reward
```