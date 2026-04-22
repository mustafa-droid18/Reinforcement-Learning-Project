from __future__ import annotations


def compute_reward(*, base_reward: float, prev_info: dict, info: dict, action: int, terminated: bool, truncated: bool) -> float:
    return float(base_reward)

