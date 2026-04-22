from __future__ import annotations


def compute_reward(*, base_reward: float, prev_info: dict, info: dict, action: int, terminated: bool, truncated: bool) -> float:
    prev_x = float(prev_info.get("x_pos", 0))
    curr_x = float(info.get("x_pos", prev_x))
    delta_x = curr_x - prev_x

    reward = float(base_reward)
    reward += 0.05 * max(delta_x, 0.0)

    if delta_x < 0:
        reward -= 0.02 * abs(delta_x)

    if info.get("flag_get"):
        reward += 100.0

    if terminated and not info.get("flag_get", False):
        reward -= 25.0

    return float(reward)

