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

    # 1. Forward progress reward — INCREASED scaling
    #    Agent was stalling around x≈898; stronger signal for rightward movement
    #    helps push past obstacles. Backtrack penalty also increased to discourage
    #    retreating when scared of enemies.
    if dx > 0:
        reward += dx * 0.5          # was 0.3 — stronger pull toward the right
    else:
        reward += dx * 0.8          # was 0.5 — stronger push against backtracking

    # 2. Milestone / checkpoint bonuses
    #    The agent plateaus because there is no explicit gradient beyond x≈898.
    #    Sparse milestones every 200 units give the agent intermediate targets to
    #    aim for without hard-coding enemy positions.
    #    We award the bonus only the FIRST TIME each milestone is crossed by
    #    checking whether x_prev was below and x_now is at/above the boundary.
    milestone_spacing = 200
    prev_milestone = x_prev // milestone_spacing
    curr_milestone = x_now  // milestone_spacing
    if curr_milestone > prev_milestone:
        # reward scales with distance to incentivise pushing further right
        reward += curr_milestone * 3.0  # e.g. crossing into zone 5 gives +15

    # 3. Score-based reward — INCREASED slightly
    #    Killing Goombas/Koopas and grabbing power-ups gives score; rewarding
    #    this more strongly teaches Mario to fight through enemies rather than
    #    stall before them.
    reward += dscore * 0.04         # was 0.025

    # 4. Survival reward — INCREASED
    #    A bigger per-step survival bonus relative to the time penalty means the
    #    agent values staying alive more, but not so big that it becomes passive.
    if not terminated:
        reward += 0.1               # was 0.05

    # 5. Death penalty — INCREASED
    #    Dying should be significantly worse than any short-term gain from rushing;
    #    raising this discourages the risky dash-and-die behaviour.
    if terminated:
        reward -= 25.0              # was -15.0

    # 6. Time penalty — REDUCED slightly
    #    The previous time penalty may have been pressuring Mario to rush
    #    recklessly into enemies. A smaller penalty still discourages idling
    #    without punishing brief pauses to navigate obstacles.
    reward -= dt * 0.01             # was 0.02

    # 7. Flag / level completion bonus — unchanged, already generous
    if flag_get:
        reward += 200.0

    return reward
```