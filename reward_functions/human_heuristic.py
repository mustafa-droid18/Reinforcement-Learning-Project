from __future__ import annotations


def compute_reward(*, base_reward: float, prev_info: dict, info: dict, action: int, terminated: bool, truncated: bool) -> float:
    prev_x = float(prev_info.get("x_pos", 0))
    curr_x = float(info.get("x_pos", prev_x))
    delta_x = curr_x - prev_x

    prev_time = float(prev_info.get("time", info.get("time", 0)))
    curr_time = float(info.get("time", prev_time))
    time_delta = prev_time - curr_time

    reward = float(base_reward)

    reward += 0.08 * max(delta_x, 0.0)
    reward -= 0.04 * max(-delta_x, 0.0)

    reward += 0.01
    reward -= 0.02 * max(time_delta, 0.0)

    if info.get("flag_get"):
        reward += 250.0

    if terminated and not info.get("flag_get", False):
        reward -= 75.0

    return float(reward)
