# Project Log

This file tracks project decisions, implementation work, and remaining risks as
the RL project moves from proposal to final submission.

## 2026-04-24

### Repository setup

- Connected the local folder to the GitHub repository:
  `https://github.com/mustafa-droid18/Reinforcement-Learning-Project.git`
- Merged the existing remote `main` history into the local scaffold.
- Resolved the `README.md` conflict by keeping the expanded project README.
- Pushed the merged `main` branch to GitHub.

### Initial implementation scaffold

- Added a reproducible Mario PPO experiment structure.
- Added baseline and shaped-reward experiment configs under `configs/`.
- Added reward function files under `reward_functions/`.
- Added training and evaluation entrypoints under `src/mario_rl/`.
- Added `PLAN_NOTES.md` to track project plan fixes needed for grading.
- Added `requirements.txt` for the intended Python dependencies.

### Reward function note

- `reward_functions/default_reward.py` returns the environment reward unchanged.
- `reward_functions/forward_progress.py` is a first shaped-reward template.
- The shaped reward keeps the original environment reward, then adds small
  incentives for forward progress, a larger bonus for reaching the flag, and
  penalties for moving backward or dying before completion.

### Current risk

- The reward shaping constants are heuristic. They are useful for testing the
  pipeline, but they should not be presented as the final LLM-generated reward
  unless they are generated or revised through the documented LLM loop.
- The final comparison must evaluate all agents using task metrics like
  completion rate, `x_pos`, score, and sample efficiency, not only shaped reward.

### Source package explanation

- `src/mario_rl/config.py` defines the experiment and PPO config dataclasses.
- `src/mario_rl/reward_api.py` loads reward functions from Python files and
  validates that reward outputs are finite numbers.
- `src/mario_rl/env_factory.py` builds the Mario environment, applies the action
  set and observation preprocessing, and optionally wraps the environment with a
  custom reward function.
- `src/mario_rl/train.py` reads a config, creates vectorized training and
  evaluation environments, trains PPO, saves the best and final models, and logs
  the resolved config.
- `src/mario_rl/eval.py` loads a trained model and reports task-level metrics
  such as reward, `x_pos`, score, coins, flag completion, and time left.

### Config and action-set explanation

- The files in `configs/` define one experiment run each.
- `baseline.json` trains PPO with the original Mario environment reward.
- `llm_v1_template.json` and `llm_v2_template.json` are shaped-reward run
  templates. They currently point to `reward_functions/forward_progress.py`
  until actual LLM-generated reward files are added.
- `env_id` selects the exact Mario level and version. We currently use
  `SuperMarioBros-1-1-v0`.
- `action_set` controls which controller-button combinations the agent is
  allowed to choose from at each step.
- `RIGHT_ONLY` gives the agent a small set of actions focused on moving right.
- `SIMPLE_MOVEMENT` gives a moderate action set with useful Mario controls like
  moving, jumping, and running.
- `COMPLEX_MOVEMENT` gives the largest action set with more button
  combinations, which is more expressive but harder to learn.
- We currently use `SIMPLE_MOVEMENT` because it is a good first-project balance:
  the agent has enough actions to play the level but not so many that PPO has to
  explore an unnecessarily large action space.
- `total_timesteps` is how many environment steps PPO trains for.
- `eval_freq` controls how often training pauses to evaluate the current model.
- `n_eval_episodes` controls how many episodes are used during each evaluation.
- `seed` makes runs more reproducible.
- `frame_stack`, `grayscale`, and `resize_shape` define how raw game frames are
  converted into model input.
- The `ppo` section holds the learning algorithm hyperparameters.

### Starting point

- Start by getting one short baseline run working locally.
- Do not begin with LLM reward generation. The training and evaluation pipeline
  must work first, otherwise LLM-generated rewards cannot be tested fairly.
- Use Python 3.9 or 3.10 for setup. The current system Python observed locally
  is Python 3.13, which is likely too new for `gym-super-mario-bros`,
  `nes-py`, and the older Gym stack.
- First target:
  - create a virtual environment with Python 3.9 or 3.10
  - install `requirements.txt`
  - run a quick environment smoke test
  - run a very short baseline PPO training job
  - only then run the full baseline config
- Practical project order:
  - baseline training
  - baseline evaluation
  - heuristic shaped-reward training
  - LLM reward `v1`
  - LLM reward `v2`
  - plots and final presentation results

### 2026-04-24 training startup fix

- First baseline training attempt reached environment creation but failed before
  PPO started learning.
- Error:
  `AssertionError: The observation space must be an image or dictionary observation space`
- Cause: raw Gym `FrameStack` changed the observation space shape before
  Stable-Baselines3's image vector wrapper saw it.
- Fix: removed raw Gym frame stacking from `env_factory.py` and added
  `src/mario_rl/vec_env.py`, which applies `VecTransposeImage` and
  `VecFrameStack` in the SB3 vectorized environment layer.
- `train.py` and `eval.py` now share the same vectorized observation pipeline,
  which keeps model training and model evaluation compatible.
- Follow-up startup issue: `ResizeObservation` requires OpenCV. Added
  `opencv-python` to `requirements.txt`.
- Follow-up step issue: `nes-py` uses the old four-return Gym step API, while
  newer Gym wrappers expect five returns. Enabled `apply_api_compatibility=True`
  when creating the Mario environment.
- Added `configs/smoke_baseline.json` for quick training-loop validation before
  running the full 300,000-step baseline.
- Smoke training reached PPO setup and exposed a missing TensorBoard dependency.
  Added `tensorboard` to `requirements.txt`.
- Smoke training then exposed optional progress-bar dependencies (`tqdm` and
  `rich`). Disabled SB3's progress bar in `train.py`; TensorBoard and console
  logging are enough for this project.

### 2026-04-24 baseline run

- Started the full baseline command:
  `PYTHONPATH=src python -m mario_rl.train --config configs/baseline.json`
- The run is now training successfully on CPU.
- Early logs show PPO rollout and training updates every 512 timesteps.
- Current expected behavior:
  - Gym deprecation warnings are expected because `gym-super-mario-bros` uses an
    older Gym stack.
  - `total_timesteps` should continue increasing until 300,000.
  - Evaluation should run every 25,000 timesteps because `eval_freq` is set to
    25,000 in `configs/baseline.json`.

### 2026-04-24 baseline result debugging

- Completed baseline artifacts found under `artifacts/baseline_seed0/`.
- Stable-Baselines3 evaluation history exists at
  `artifacts/baseline_seed0/eval/evaluations.npz`.
- Saved models exist at:
  - `artifacts/baseline_seed0/models/best_model.zip`
  - `artifacts/baseline_seed0/models/final_model.zip`
- Initial task-level evaluation of both best and final model:
  - mean reward: `252.0`
  - mean `x_pos`: `315.0`
  - mean score: `0.0`
  - completion rate: `0.0`
  - episode length: `106` steps
- Interpretation: the baseline learned a short deterministic behavior that moves
  Mario to around `x_pos=315` but does not complete the level.
- Visual inspection of the baseline video confirms the behavior: Mario mostly
  runs right and fails at/near the first enemy instead of learning to jump or
  avoid it.
- Updated `eval.py` so evaluating `best_model` and `final_model` no longer
  overwrites the same JSON summary by default.
- Added `src/mario_rl/summarize_eval.py` to convert SB3 `.npz` evaluation
  history into a CSV summary.

### 2026-04-24 human heuristic run setup

- Added `reward_functions/human_heuristic.py`.
- Added `configs/human_heuristic.json`.
- The human heuristic reward is designed to address the observed baseline
  failure:
  - reward forward progress
  - penalize backward movement
  - add a large flag-completion bonus
  - penalize ending the episode without the flag
- Removed hardcoded `x_pos` milestone bonuses from the human heuristic reward.
  The reward should not depend on the specific `x_pos` where the baseline died.
  It now uses general progress, time, completion, and failure signals.
- Updated LLM template configs so they point to future LLM reward files
  (`reward_functions/llm_v1.py` and `reward_functions/llm_v2.py`) instead of
  incorrectly pointing at the human heuristic reward.

### 2026-04-24 video recording option

- Added config-controlled video recording fields:
  - `record_videos`
  - `video_freq`
  - `video_length`
  - `video_fps`
  - `video_format`
- Added `src/mario_rl/video.py`.
- Video workflow:
  - Set `record_videos` to `true` in a config to record checkpoint videos
    during training.
  - Or record a saved model after training with:
    `PYTHONPATH=src python -m mario_rl.video --config configs/baseline.json --model artifacts/baseline_seed0/models/final_model.zip --output artifacts/baseline_seed0/videos/final_model.gif`
- Default configs keep `record_videos` set to `false` so normal training is not
  slowed down unless video recording is explicitly enabled.
- Video environments are created with `render_mode="rgb_array"` so frames can be
  captured without changing the normal training environment.
- GIF was the initial default because it did not require FFmpeg.
- After upgrading `imageio-ffmpeg`, MP4 recording works with the bundled Mac ARM
  FFmpeg binary, so the default `video_format` is now `mp4`.
- Set main experiment `video_length` to `10000` steps so full episodes can be
  recorded and trimmed later if needed.

### 2026-04-24 device check

- The baseline log says `Using cpu device`.
- Checked PyTorch in `.venv`:
  - `torch==2.3.1`
  - `torch.backends.mps.is_built()` is `True`
  - `torch.backends.mps.is_available()` is `False`
  - `torch.cuda.is_available()` is `False`
- The machine is `arm64` on macOS `26.3.1`.
- Attempting to create a tensor on `device="mps"` fails with PyTorch reporting
  that MPS requires macOS 12.3+, even though the OS is newer. This points to a
  PyTorch/macOS compatibility detection issue with the pinned Torch version.
- Stable-Baselines3 also defaults to CPU when CUDA is unavailable unless a device
  is explicitly supplied, so CPU is expected with the current setup.

### 2026-04-24 MP4 and accelerator follow-up

- GIF was initially chosen as the default because MP4 writing needs FFmpeg, and
  the first local MP4 attempt failed with no FFmpeg executable available.
- Upgraded `imageio-ffmpeg` to `0.6.0`; it now provides a bundled Mac ARM FFmpeg
  binary, so MP4 recording is available.
- Upgraded PyTorch from `2.3.1` to `2.11.0` to test whether Apple MPS becomes
  available.
- MPS still reports unavailable:
  - `torch.backends.mps.is_built()` is `True`
  - `torch.backends.mps.is_available()` is `False`
  - creating a tensor on `device="mps"` raises a runtime error
- CUDA is not available on this Mac because CUDA requires an NVIDIA GPU.
- Added a `device` config field and pass it into PPO. It remains set to `auto`
  in the configs so training uses CPU unless PyTorch exposes a working
  accelerator.
- Tried PyTorch nightly CPU/Mac wheels:
  - `torch==2.13.0.dev20260424`
  - `torchvision==0.27.0.dev20260424`
  - `torchaudio==2.11.0.dev20260424`
- Stable-Baselines3 still imports with the nightly PyTorch build.
- MPS still reports unavailable with the same macOS-version runtime error.
- Current conclusion: this machine/environment should continue using CPU unless
  PyTorch starts reporting `torch.backends.mps.is_available() == True`.
- Created a clean `.venv-mps` and installed the stable macOS arm64 PyTorch
  packages. In that clean environment, MPS works:
  - `torch==2.11.0`
  - `torch.backends.mps.is_available()` is `True`
  - `torch.ones(1, device="mps")` succeeds
- This confirms the MPS failure was specific to the old `.venv`, not the Mac
  hardware.
- Updated `requirements.txt` back to stable Mac PyTorch package versions so a
  clean install can reproduce the working MPS setup.
- Installed the full RL dependency stack into `.venv-mps`.
- MPS availability can differ between sandboxed tool commands and the normal
  shell. Retesting `.venv-mps` outside the sandbox reports MPS available and can
  allocate a tensor on `mps:0`.
- Added MPS-specific configs:
  - `configs/smoke_baseline_mps.json`
  - `configs/baseline_mps.json`
- Ran `configs/smoke_baseline_mps.json` with `.venv-mps`.
- Stable-Baselines3 printed `Using mps device`, confirming PPO can train on
  Apple MPS.
- MPS smoke training completed, but it was much slower than CPU for this setup:
  around `11-12 fps` versus the prior CPU run around `200+ fps`.
- Current recommendation: use CPU for the Mario PPO experiments unless a later
  profiling run shows MPS is faster with different batch sizes or training
  settings.
