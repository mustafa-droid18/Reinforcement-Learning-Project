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
    # DIAGNOSIS: Agent plateaus at x≈1137 across iterations. The core problem
    # is likely that:
    #   (a) the per-step shaping signal is too weak to push through hard sections,
    #   (b) the death penalty dominates and makes the agent overly conservative,
    #   (c) there is no explicit reward gradient beyond the current plateau.
    #
    # CHANGES THIS ITERATION:
    #   - Significantly increase forward-progress reward magnitude
    #   - Reduce death penalty (was too conservative, agent freezes rather than risks)
    #   - Add a smooth x_pos-based potential shaping term so the gradient
    #     never flattens regardless of where Mario is in the level
    #   - Remove the redundant "new_ground" bonus (double-counted with dx reward)
    #   - Reduce stall penalty slightly (was counterproductive near gaps/enemies)
    #   - Keep score reward to encourage enemy kills
    # -------------------------------------------------------------------------

    # 1. Forward progress — SIGNIFICANTLY increased.
    #    This is the primary learning signal. The agent needs a strong, dense
    #    gradient pulling it rightward at all x positions, especially past 1137.
    if dx > 0:
        reward += dx * 1.0          # was 0.5; doubled to push through plateaus
    elif dx < 0:
        reward += dx * 0.5          # was 0.8; soften backtrack penalty slightly
                                    # so agent isn't terrified of minor retreats

    # 2. Potential-based shaping term — NEW.
    #    F = gamma * Phi(s') - Phi(s) where Phi = x_pos / scale.
    #    This gives a smooth, always-positive gradient toward the flag that
    #    complements the per-step dx reward. Using gamma ≈ 1 for simplicity.
    #    Scaled so the per-step contribution is small (fractions of a unit)
    #    but accumulates meaningfully over a full episode.
    gamma = 0.99
    phi_now  = x_now  / 3186.0     # normalise to [0, 1]
    phi_prev = x_prev / 3186.0
    reward += 3.0 * (gamma * phi_now - phi_prev)  # moderate weight

    # 3. Score reward — unchanged; enemy kills and coins are valuable signals.
    reward += dscore * 0.04

    # 4. Survival bonus — kept small so Mario doesn't learn to stand still.
    if not terminated:
        reward += 0.04

    # 5. Death penalty — REDUCED from -25 to -10.
    #    A very large death penalty was making the agent overly risk-averse:
    #    it would freeze before enemies rather than attempt to pass them,
    #    causing the plateau. Reducing it lets the agent explore riskier moves.
    if terminated:
        reward -= 10.0              # was -25.0

    # 6. Time penalty — slightly increased to prevent prolonged freezing.
    #    The agent freezes at x≈1137; a firmer time cost encourages action.
    reward -= dt * 0.02             # was 0.01

    # 7. Stall penalty — kept but reduced.
    #    Still useful near pipes/gaps but shouldn't overwhelm the agent.
    if dx == 0 and dt > 0 and not terminated:
        reward -= 0.05              # was 0.08

    # 8. Flag completion bonus — unchanged.
    if flag_get:
        reward += 200.0

    return reward
```