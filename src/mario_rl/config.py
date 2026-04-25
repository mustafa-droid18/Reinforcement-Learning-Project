from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class PPOConfig:
    policy: str = "CnnPolicy"
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class ExperimentConfig:
    experiment_name: str
    env_id: str = "SuperMarioBros-1-1-v0"
    action_set: str = "SIMPLE_MOVEMENT"
    total_timesteps: int = 300_000
    eval_freq: int = 25_000
    n_eval_episodes: int = 5
    seed: int = 0
    frame_stack: int = 4
    grayscale: bool = True
    resize_shape: list[int] = field(default_factory=lambda: [84, 84])
    n_envs: int = 1
    frame_skip: int = 1
    max_stagnation_steps: int = 0
    train_reward_path: str | None = None
    log_dir: str = "artifacts"
    record_videos: bool = False
    video_freq: int | None = None
    video_length: int = 10000
    video_fps: int = 30
    video_format: str = "mp4"
    device: str = "auto"
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        ppo_data = data.pop("ppo", {})
        return cls(ppo=PPOConfig(**ppo_data), **data)

    def to_json_dict(self) -> dict:
        return asdict(self)
