"""Reward function loader and validator.

Dynamically loads a reward function from a .py file at runtime so that
different reward designs can be swapped without modifying training code.

Each reward file must define a `compute_reward` function with signature:
    compute_reward(*, base_reward, prev_info, info, action, terminated, truncated) -> float

The function must be stateless — training uses 4 parallel environments that
share all module-level state, so any per-episode state stored at module level
will be contaminated across environments simultaneously.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Callable


RewardFn = Callable[..., float]


def default_reward(*, base_reward: float, prev_info: dict, info: dict, action: int, terminated: bool, truncated: bool) -> float:
    return float(base_reward)


def _load_module(path: Path) -> ModuleType:
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reward module from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reward_function(path: str | None) -> RewardFn:
    if path is None:
        return default_reward

    module_path = Path(path)
    if not module_path.exists():
        raise FileNotFoundError(f"Reward file does not exist: {module_path}")

    module = _load_module(module_path)
    if not hasattr(module, "compute_reward"):
        raise AttributeError(f"{module_path} must define compute_reward(...)")

    reward_fn = getattr(module, "compute_reward")
    return reward_fn


def validate_reward_output(value: float) -> float:
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("Reward function returned a non-finite value")
    return value

