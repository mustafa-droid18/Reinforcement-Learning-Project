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
- Kept the non-completion penalty at `-100` rather than increasing it to a much
  larger value. A very large death penalty can make early exploration overly
  conservative.
- Updated the penalty to apply to both `terminated` and `truncated` non-flag
  endings so stalling until timeout is not a loophole.
- Updated LLM template configs so they point to future LLM reward files
  (`reward_functions/llm_v1.py` and `reward_functions/llm_v2.py`) instead of
  incorrectly pointing at the human heuristic reward.

### 2026-04-24 human heuristic reward redesign

- Rewrote `reward_functions/human_heuristic.py` based on analysis of the baseline
  failure mode.
- Baseline problem: Mario reliably dies at `x_pos=315` (first Goomba). The prior
  heuristic had no signal that teaches jumping — it only rewarded horizontal
  distance, which the baseline already maximised before dying.
- Key addition: **score delta bonus** (`score_delta / 100`). Mario's in-game score
  increases when the agent stomps enemies (+100 pts), collects coins, or hits
  blocks. Rewarding this delta creates a direct incentive to jump over and onto
  obstacles, which is exactly the behaviour the baseline lacked.
- Removed the per-step time penalty. It was redundant with the forward progress
  reward and created conflicting gradients — every step was both rewarded for
  rightward movement and penalised for time passing.
- Removed the per-step survival bonus. It added noise with no directional
  information.
- Increased the death penalty from −75 to −100 so it clearly outweighs any
  local progress gained near an enemy.
- Increased the flag completion bonus from +250 to +500 to make the level-clear
  signal unambiguous relative to accumulated shaped rewards.
- Retained `base_reward` as the foundation so the native environment reward is
  still present.
- Summary of new reward terms:
  - `base_reward` (unchanged)
  - `+0.1 × max(delta_x, 0)` — forward progress amplification
  - `−0.05 × max(−delta_x, 0)` — backward movement penalty
  - `+score_delta / 100` — enemy stomps, coins, block hits
  - `+500` on flag completion
  - `−100` on death without flag

### 2026-04-24 Goomba failure diagnosis

- Visual inspection after the human heuristic run showed Mario jumps early but
  still collides with the first Goomba.
- Verified Mario environment `info` fields include:
  `coins`, `flag_get`, `life`, `score`, `stage`, `status`, `time`, `world`,
  `x_pos`, and `y_pos`.
- `status` starts as `small`, so a first-Goomba collision usually does not
  create a useful `tall -> small` status-change signal. A status downgrade
  penalty is useful after power-ups, but it is not a direct first-Goomba contact
  signal.
- The main issue is likely sparse/timing-sensitive learning from pixels. Jumping
  over the Goomba gives no score reward; only stomping it produces score.

### 2026-04-25 human heuristic results

- Completed `human_heuristic_seed0` with `1,000,000` timesteps.
- Converted evaluation history to
  `artifacts/human_heuristic_seed0/eval/evaluation_history.csv`.
- Evaluation history is unstable but shows clear improvement over the default
  baseline at the best checkpoint.
- Default baseline best/final:
  - mean reward: `252.0`
  - mean `x_pos`: `315.0`
  - mean score: `0.0`
  - completion rate: `0.0`
- Human heuristic best model:
  - mean reward: `814.0`
  - mean `x_pos`: `898.0`
  - mean score: `100.0`
  - completion rate: `0.0`
  - episode length: `149`
- Human heuristic final model:
  - mean reward: `-21.0`
  - mean `x_pos`: `434.0`
  - mean score: `0.0`
  - completion rate: `0.0`
  - episode length: `2005`
- Interpretation: the human heuristic successfully found a better policy at the
  best checkpoint, likely handling the first enemy and reaching much farther
  than the default baseline. Later training regressed, so use
  `models/best_model.zip` for reporting this run unless a more stable retrain is

### 2026-04-25 human heuristic rerun and reward update

- The current `artifacts/human_heuristic_seed0/` directory no longer reflects
  the older full run above. It now contains a newer partial rerun that reached
  `100,000` timesteps before stopping.
- Current partial rerun evaluation history from
  `artifacts/human_heuristic_seed0/eval/evaluations.npz`:
  - `25,000`: reward `252`, episode length `27`
  - `50,000`: reward `231`, episode length `40`
  - `75,000`: reward `231`, episode length `40`
  - `100,000`: reward `342`, episode length `103`
- Interpretation: this rerun is still better than the default baseline by the
  end of `100k`, but it is not yet solving the obstacle timing problem and it
  has not reached the older run's best progress.
- Updated the human heuristic reward to target the new failure mode:
  Mario getting stuck behind a pipe or repeating a bad loop after early
  progress.
- Reward changes now in `reward_functions/human_heuristic.py`:
  - keep `base_reward`
  - `+0.1 * forward progress`
  - `-0.02 * backward movement`
  - `+score_delta / 100`
  - `-0.5 * time_delta` when time passes and `x_pos` does not change
  - `+500` on flag
  - `-100` on non-completion
  - final square-root reward compression
- Training-loop changes added to support the new heuristic:
  - `max_stagnation_steps` added to config
  - `StagnationTerminationWrapper` added to end episodes early when `x_pos`
    stops improving for too long
  - callback frequencies now scale correctly with `n_envs`, so `eval_freq` and
    `video_freq` are interpreted in real timesteps instead of vector-env calls
- The next retrain should use a new experiment name to avoid overwriting the
  partial rerun artifacts.

### 2026-04-25 human heuristic v2 start

- Added `configs/human_heuristic_v2.json` with experiment name
  `human_heuristic_v2_seed0` so the updated reward does not overwrite the
  partial rerun artifacts.
- Started the updated run with the new stagnation penalty and stagnation-based
  early truncation wrapper enabled.
- First measured checkpoint:
  - `25,000` timesteps
  - mean eval reward: `250`
  - mean eval episode length: `30`
- Comparison against the older partial rerun at `25,000`:
  - old partial rerun: reward `252`, episode length `27`
  - v2 run: reward `250`, episode length `30`
- Early interpretation:
  - the new reward is not yet clearly better at the first eval checkpoint
  - rollout reward during training climbed quickly into the `130-170` range
  - the next useful comparison point is `50,000` timesteps, where the prior
    partial rerun dropped to reward `231`
  performed.

### 2026-04-25 training loop fixes

- Added `max_stagnation_steps` to `ExperimentConfig`.
- Added `StagnationTerminationWrapper` in `src/mario_rl/env_factory.py`.
- When enabled, the wrapper truncates an episode if `x_pos` fails to improve for
  too many agent steps. This directly targets the "stuck behind the pipe"
  behavior and avoids wasting long rollouts on stalled trajectories.
- Set `max_stagnation_steps` to `64` in `configs/human_heuristic.json`.
- Fixed callback frequency semantics in `src/mario_rl/train.py`.
  `eval_freq` and `video_freq` are now interpreted in actual environment
  timesteps, not vectorized `env.step()` calls. With `n_envs=4`, a config value
  of `25000` now really means evaluate or record every `25000` timesteps.
- Cleaned `DummyVecEnv` construction in `src/mario_rl/vec_env.py` so each
  vectorized environment is created from its own factory closure.

### 2026-04-25 human heuristic revision

- Replaced the action-index-specific stagnation penalty in
  `reward_functions/human_heuristic.py`.
- New shaping is more general and less tied to the exact action numbering:
  - `+0.1 * forward progress`
  - `-0.02 * backward movement`
  - `+score_delta / 100`
  - `-0.5 * time_delta` when time advances but `x_pos` does not
  - `+500` on flag completion
  - `-100` on non-completion
- Motivation: the previous version only penalized "holding right while stuck."
  The new version penalizes wasting time without progress regardless of the
  exact action choice, which should better target pipe-stuck loops and repeated
  bad jump timing.

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

### 2026-04-24 human heuristic training improvements

Analysed an external Mario PPO reference implementation
(jiseongHAN/Super-Mario-RL) and applied three lessons to the human heuristic
experiment.

#### 1. Sqrt reward transformation

- Added `_sqrt_transform` to `reward_functions/human_heuristic.py`.
- Formula: `sign(r) * (sqrt(|r| + 1) - 1) + 0.001 * r`
- Motivation: our reward components span a 5000:1 ratio (+0.1 per-step progress
  up to +500 flag bonus). Without compression, the rare large signals dominate
  the value function gradient and slow down learning of the frequent small
  signals. The sqrt transform collapses this to roughly a 22:0.05 ratio while
  preserving sign and rank ordering.
- Applied as the final output step so all intermediate shaping logic stays
  readable and unchanged.

#### 2. Parallel environments

- Added `n_envs` field to `ExperimentConfig` (default `1` so existing configs
  are unaffected).
- Updated `build_vec_env` to spawn `config.n_envs` copies via `DummyVecEnv`.
- Set `n_envs: 4` in `configs/human_heuristic.json`.
- Motivation: more parallel envs produce decorrelated experience per PPO update,
  which is one of the highest-leverage PPO improvements. 4 envs keeps memory and
  CPU usage reasonable on this machine.

#### 3. Shorter rollouts with smaller minibatches

- Changed `ppo.n_steps` from `512` to `128` in `configs/human_heuristic.json`.
- Changed `ppo.batch_size` from `256` to `64`.
- With 4 envs × 128 steps the rollout buffer holds 512 transitions per update
  (same as before), but each chunk comes from a different point in 4 independent
  episodes, so the data is less temporally correlated.
- Smaller minibatches increase the number of gradient steps per epoch, which
  tends to improve sample efficiency for policy gradient methods.

#### 4. Increased timestep budget

- Changed `total_timesteps` from `300000` to `1000000` in
  `configs/human_heuristic.json`.
- Motivation: the baseline showed that 300k steps is not enough for the agent
  to learn precise jump timing from raw pixels. With 4 parallel envs this is
  roughly the same wall-clock time as the old 300k single-env run.
- `record_videos` set to `false` to avoid slowing down training.

### 2026-04-25 frame skipping and stagnation penalty

#### Observation: jump height too low to clear obstacles

- Observed during human heuristic run: Mario walks into pipes and Goombas
  without jumping, even after 1M steps of training.
- Root cause identified: without frame skipping each `env.step()` holds the
  action for exactly 1 game frame (1/60 s). A real Mario jump requires holding
  the A button for ~20 game frames. A single-frame A press produces only a tiny
  hop that cannot clear a pipe or Goomba. The agent is not "choosing not to
  jump" — it physically cannot jump high enough with the current setup.

#### Fix 1: FrameSkipWrapper

- Added `FrameSkipWrapper` to `src/mario_rl/env_factory.py`.
- Each selected action is repeated for `skip` game frames; rewards are summed
  and the final frame observation is returned.
- Placed after `GymApiCompatibilityWrapper` (needs 5-return API) and before
  `RecordEpisodeStatistics` (so episode stats reflect agent decisions, not
  raw game frames).
- Added `frame_skip: int = 1` field to `ExperimentConfig` (default 1 = no
  skip, so all existing configs are unaffected).
- Set `frame_skip: 4` in `configs/human_heuristic.json`.
- With skip=4 the agent acts at ~15fps instead of 60fps. A jump action held
  for 4 frames generates a proper Mario jump. This also reduces the temporal
  horizon the policy needs to reason over, which significantly helps PPO.

#### Fix 2: Stagnation penalty

- Added to `reward_functions/human_heuristic.py`.
- When Mario selects a rightward action (SIMPLE_MOVEMENT indices 1-4) but
  `|delta_x| < 1.0` (not making horizontal progress), apply a −0.3 penalty.
- Motivation: when Mario is pressed against a pipe wall, `delta_x = 0` every
  step. The stagnation penalty makes standing still against a wall clearly
  worse than backing up (mild −0.05×delta_x backward penalty) or jumping.
  The agent is pushed to discover the back-up-then-jump strategy.
- Backward penalty kept at −0.05 so that backing up to attempt a jump remains
  worthwhile when it leads to subsequent forward progress.
- Action index comment added to the reward file since the indices are
  SIMPLE_MOVEMENT-specific.

#### Config changes for human_heuristic.json

- Added `frame_skip: 4`.
- `record_videos` restored to `true` (user preference).

### 2026-04-25 video recording bug fix

- Fixed `src/mario_rl/video.py` to always record with `n_envs=1` regardless
  of the training config value.
- Previously, `record_policy_video` used `build_vec_env(config, ...)` which
  now spawns `config.n_envs` parallel environments (4 for human heuristic).
  This caused the recorded video to show a 2×2 tiled frame of all 4 envs.
- Fix: `dataclasses.replace(config, n_envs=1)` overrides only n_envs for the
  video env, keeping all other config fields intact.

### 2026-04-25 stagnation penalty increase and pipe-jump improvements

- Observed in human heuristic v2 videos: Mario gets stuck at pipes and never
  backs up to attempt a jump, despite the stagnation penalty.
- Root cause: the -0.3 stagnation penalty was compressed to ≈-0.14 by the
  sqrt transform — too mild to outweigh the large accumulated forward-progress
  reward the agent earned getting to the pipe in the first place.
- Additionally, the -0.05 backward penalty discouraged the exact back-up
  action the agent needed to take to create jump distance from the pipe.

#### Changes to reward_functions/human_heuristic.py

- Raised stagnation penalty from -0.3 to -2.0.
  After sqrt transform this is ≈-1.0 per stuck step — genuinely costly and
  enough to overpower the accumulated progress reward within a few steps.
- Removed the backward movement penalty entirely (was -0.05 × delta_x).
  Forward progress reward already makes moving right preferable when possible.
  Removing the backward penalty makes backing up to jump completely free,
  which is necessary for the back-up-then-jump pipe maneuver.

#### Changes to configs/human_heuristic.json

- Switched action_set from SIMPLE_MOVEMENT to COMPLEX_MOVEMENT.
  COMPLEX_MOVEMENT adds left+A and left+A+B actions, giving the agent explicit
  back-up-while-jumping capability that SIMPLE_MOVEMENT lacked.
  Rightward action indices 1-4 are identical in both sets so the stagnation
  penalty detection is unchanged.

### 2026-04-25 milestone rewards, more timesteps, more exploration

- Observation: human_heuristic_v2 run peaked at reward 816 (x_pos ≈ 856,
  ~28% through the level) but the agent has no signal to push further.
  The flag bonus (+500) is unreachable at the current policy's range, so it
  never fires and provides no learning gradient past x≈856.

#### reward_functions/human_heuristic.py — milestone bonuses

- Added five x_pos milestone checkpoints: 500, 1000, 1500, 2000, 2500.
- Each awards +100 the step Mario's x_pos first crosses the threshold.
- After sqrt transform: +100 → ≈+9.15, strong enough to pull the policy
  past each checkpoint without overwhelming the per-step signals.
- Milestones break the sparse reward problem: instead of needing to
  accidentally reach x=3000 to get the flag signal, the agent gets
  incremental dense feedback for pushing into new territory.

#### configs/human_heuristic_v2.json

- total_timesteps: 1000000 → 3000000. 3M steps with 4 envs gives the
  agent enough samples to learn the jump timing at each new obstacle
  it encounters beyond x=856.
- ppo.ent_coef: 0.01 → 0.03. More entropy encourages exploration of new
  actions at obstacles the agent hasn't seen before. Without this, the
  policy repeats the same moves at new pipes and Goombas.

### 2026-04-25 milestone spacing fix

- Observed: x=500 as the first milestone was too far — it only fired on good
  episodes, giving no learning signal during the critical early phase where the
  agent is dying around x=315.
- Updated milestones to [200, 400, 600, 800, 1000, 1500, 2000, 2500].
  x=200 fires in almost every episode (agent already reaches ~x=315 in the
  baseline), giving consistent early feedback from the very first episodes.
- Reduced bonus from +100 to +50 per milestone so total milestone reward
  (+400) stays below the flag bonus (+500).
- Stopped the 3M-step run at 1075k steps and restarting with updated milestones.

### 2026-04-25 switch to SIMPLE_MOVEMENT, increase stagnation cutoff

- COMPLEX_MOVEMENT run (1050k steps) peaked at only 620, trending downward
  to 230-238 by the end. Worse than the SIMPLE_MOVEMENT run that hit 816.
- Root cause analysis: COMPLEX_MOVEMENT (12 actions) is too large an
  exploration space. The agent kept finding ways to waste time with left, down,
  up combinations instead of learning the jump-right sequence.
- max_stagnation_steps 64 → 200: 64 agent steps × 4 frame_skip = 256 game
  frames (~4 seconds) was too tight. The agent needs more attempts per episode
  to discover the jump sequence at each new obstacle before the episode cuts off.
  200 steps = 800 game frames (~13 seconds) gives 3-4 full jump attempts.
- action_set: COMPLEX_MOVEMENT → SIMPLE_MOVEMENT. Our best result (816 reward)
  came from SIMPLE_MOVEMENT. It includes left and jump-in-place (A) for
  flexibility without the 12-action exploration penalty of COMPLEX_MOVEMENT.
- Restarting 3M step run with: SIMPLE_MOVEMENT, max_stagnation_steps=200,
  milestones [200,400,600,800,1000,1500,2000,2500] at +50 each, ent_coef=0.03.

### 2026-04-26 human heuristic iteration audit

The full history of manual tuning iterations that produced the final heuristic
is recorded here for the project comparison. Artifacts from early runs were
overwritten during experimentation; only v2 and v3 artifacts remain on disk.

**Total reward/config iterations: 8**

| # | What changed | Why | Outcome |
|---|---|---|---|
| 1 | Initial design: forward progress, backward penalty, score delta, flag +500, death -100, n_envs=4, n_steps=128, 1M steps | Baseline died at x=315, needed basic shaping | Agent still failed at first Goomba |
| 2 | Added sqrt reward transform | 5000:1 reward ratio swamped value function gradients | Stabilized gradient scale |
| 3 | Added FrameSkipWrapper (skip=4), action-specific stagnation penalty -0.3, StagnationTerminationWrapper (max=64) | Single-frame A press physically cannot produce a real Mario jump | Agent could now jump over obstacles |
| 4 | Raised stagnation penalty -0.3 → -2.0, removed backward penalty, SIMPLE_MOVEMENT → COMPLEX_MOVEMENT | -0.3 compressed to ≈-0.14 by sqrt transform, too mild; backward penalty blocked the back-up-then-jump maneuver | Agent more willing to back up at pipes |
| 5 | Added milestones [500,1000,1500,2000,2500] at +100 each, 1M → 3M timesteps, ent_coef 0.01 → 0.03 | Agent peaked at x≈856 with no signal to go further; flag bonus unreachable | Milestone structure added |
| 6 | Moved first milestone from 500 → 200, reduced bonus +100 → +50, revised spacing to [200,400,600,800,1000,1500,2000,2500] | x=500 milestone never fired during early training (agent dying at x=315) | Consistent early feedback signal |
| 7 | COMPLEX_MOVEMENT → SIMPLE_MOVEMENT, max_stagnation 64 → 200 | COMPLEX_MOVEMENT (12 actions) peaked at 620, worse than SIMPLE_MOVEMENT's 816; 64-step cutoff too tight for jump discovery | Became the v2 final config |
| 8 | Denser milestones every 100 units from x=1000–2000, ent_coef 0.03 → 0.02, 3M → 5M steps | v2 agent was inconsistent past x=1000 (500-unit gap with no milestone signal); lower entropy to consolidate learned behavior | v3 final config |

In addition to the 8 reward/config iterations, three infrastructure bugs required
fixes that affected training correctness (not logged as tuning iterations):
- Video recording produced a 2×2 tiled frame (n_envs=4 used for recording)
- Callback eval/video frequencies did not scale with n_envs
- DummyVecEnv lambda closure captured shared reference instead of per-env copy

**Key observation for the project write-up:** a human expert needed 8 deliberate
iterations, direct observation of failure modes through videos, and domain
knowledge about action sets, frame skip, and reward scale to reach the final
heuristic. The LLM comparison should be evaluated against a first or second
attempt — not against this fully iterated version.

### 2026-04-26 human heuristic v2 final results

- Completed `human_heuristic_v2_seed0` at 3,000,000 timesteps.
- Config: SIMPLE_MOVEMENT, frame_skip=4, max_stagnation_steps=200, n_envs=4,
  ent_coef=0.03, milestones [200,400,600,800,1000,1500,2000,2500] at +50 each.
- Evaluation summary (120 total checkpoints):
  - Best mean reward: **1435.0** at step 2,525,000
  - Evals above 800: 7
  - Evals above 500: 31
- Progressive best-reward history:
  - 231 → 250 → 347 → 504 → 653 → 812 → 816 → **1432 → 1435**
- Key observation: the agent plateaued near 816 from steps 1.05M–2.3M,
  then broke through to 1432–1435 in the final third of training.
  This matches the milestone structure working as intended — the agent
  eventually learned to push past the x≈800–850 obstacle cluster.
- Late-run variance is high (last 10 evals range from −28 to 757).
  The best_model.zip at step 2,525,000 is the recommended model for
  reporting; the final checkpoint regressed.
- Comparison against baseline (best model):
  - Baseline: reward 252, x_pos 315, completion rate 0%
  - Human heuristic v2: reward 1435, significantly further through level
- Next step: record best_model.zip as video to confirm actual x_pos reached.
- Next run candidate: add denser milestones in the 1000–2500 range
  (every 100–200 units) to provide stronger pull through the second half
  of the level where the agent is still inconsistent.

### 2026-04-26 human heuristic v3 final results

- Completed `human_heuristic_v3_seed0` at 5,000,000 timesteps.
- Config vs v2: denser milestones (17 checkpoints vs 8), ent_coef 0.03 → 0.02,
  total_timesteps 3M → 5M. All other settings identical to v2.
- Evaluation summary (200 total checkpoints):
  - Best mean reward: **2223.0** at step 4,925,000
  - Evals above 800: 62
  - Evals above 1000: 55
  - Evals above 1435 (v2 best): 15
- Late-run evals are consistently high: last 10 averaged ~1570, ranging 1088–2223.
  Much more stable than v2's final oscillations (−28 to 757).
- Note on v2 vs v3 comparison: two variables changed simultaneously (milestone
  density and timestep budget). The improvement cannot be cleanly attributed to
  either alone. For the project write-up, v3 is presented as the final tuned
  heuristic without decomposing v2→v3 causally.

### 2026-04-26 task-level evaluation — all heuristic runs

Ran `mario_rl.eval` (10 episodes, deterministic, best_model.zip) across all
complete runs to get environment-native metrics comparable across reward functions.

| Run | Steps | Mean x_pos | Mean Score | Completion | Mean Reward |
|---|---|---|---|---|---|
| Baseline | 300k | 315 | 0 | 0% | 252 |
| Human heuristic v2 | 3M | 1523 | 0 | 0% | 1435 |
| Human heuristic v3 | 5M | **2354** | **500** | 0% | 2223 |

- x_pos is the primary task metric for cross-run comparison (reward values
  are not comparable across different reward functions).
- v3 reaches x=2354, which is ~74% through World 1-1 (flag at x≈3166).
- v3 mean score of 500 confirms Mario is actively stomping enemies.
- No run achieved level completion. The remaining ~800 units to the flag
  represents the next learning challenge.

### 2026-04-26 LLM experiment framing and success criteria

- Human heuristic took 8 manual tuning iterations over multiple days.
- Fair LLM comparison: evaluate LLM v1 against human heuristic v1 (first
  attempt), not v3 (fully iterated). The honest question is whether an LLM
  can match or exceed a human's first-attempt reward function.
- LLM runs will use 5M timesteps to match v3's training budget.
- Success criteria for LLM runs (task metrics only, not shaped reward):
  - LLM v1 beats baseline (x_pos > 315) — LLM reward generation works at all
  - LLM v2 beats LLM v1 — LLM feedback loop improves the reward iteratively
  - LLM v1 is competitive with human heuristic v1 first attempt (x_pos ~898)
  - Stretch: LLM v2 approaches v3 x_pos of 2354 without 8 iterations of tuning

### 2026-04-27 LLM loop first attempt — aborted, prompts revised

First attempt at the automated LLM loop ran 3 rounds before being stopped and
the experiment redesigned. All artifacts from this attempt have been cleared.

Rounds completed before stopping:
- **Round 1** — peaked at shaped reward 634 at step 600k, ended at 252.
  Oscillated 250–347 throughout. No novelty/milestone signal in Claude's
  reward function.
- **Round 2** — best 502 at step 1M, also ending at 502. Claude added a
  module-level `_episode_state` dict to track max_x and stagnation. Late run
  consolidated at 347 then broke through to 502.
- **Round 3** — collapsed to shaped reward -40 at step 75k and never
  recovered. Claude added graduated stagnation penalties (-0.5/-1.0/-2.0
  cumulative). Telemetry confirmed dead policy: entropy_loss ≈ 0,
  approx_kl = 0, ep_rew_mean locked at -581. Stopped before completion.

Why aborted: identified two methodological flaws that would have biased the
v1 result.

1. **Best-round selection used shaped reward.** Shaped reward values are
   not comparable across rounds since each round uses a different reward
   function. Fixed to use mean x_pos (environment-native, comparable).

2. **System and feedback prompts contained subtle hints toward specific
   reward design.** Including "agent must learn to run right, jump over
   enemies", labeling the trend as "improving/declining", and bolding
   particular facts as "CRITICAL". A blind LLM experiment requires neutral
   data presentation.

Changes applied to `llm_reward_loop.py` for the next run:
- Initial message reduced to "Generate a reward function for training a
  PPO agent to play Super Mario Bros World 1-1." — no behavioral hints.
- System prompt now states only technical facts: n_envs=4 (shared module),
  StagnationTerminationWrapper exists, y_pos increases downward. No
  emphasis (no bold, no "CRITICAL").
- Feedback message removed pre-classified summaries (best/final/trend),
  shows task metrics first (cross-round comparable), shaped curve second
  with explicit caveat about non-comparability.
- Best-round selection now uses task metric `mean_x_pos`, not shaped
  reward.
- Removed "It never learns to jump over obstacles" from baseline section
  — that was telling Claude the failure mode.

The previous prompt had Claude implicitly using design vocabulary it was
fed (novelty bonus, stagnation penalty). The next attempt isolates Claude's
own reward design choices from the experimenter's framing.

Cleaned up: `artifacts/llm_v1_test_r{1,2,3}/`, `configs/llm_v1_test_r{1,2,3}.json`,
`reward_functions/llm_v1_{current,r1,r2,r3}.py`, `prompts/llm_v1_r{1,2}_*.md`,
`error.txt`. `LLM_LOOP_LOG.md` reset for the next run.

### 2026-04-27 LLM loop second attempt — aborted after r2

Second attempt ran 1 full round and partial r2 before being stopped.

**Issues found and fixed:**

1. **Reproducibility bug in `run_training()`.** Each round's per-round
   config (`configs/llm_v1_test_rN.json`) was being written with
   `train_reward_path` still pointing to the mutable
   `reward_functions/llm_v1_current.py` rather than the archived
   `llm_v1_rN.py`. Fixed by passing the archive path into
   `run_training(round_num, reward_path)` and writing it into the
   per-round config. Each round is now self-describing on disk.

2. **Mislabeled feedback curve.** The feedback message described the
   learning curve from `evaluations.npz` as "shaped reward, not
   comparable across rounds". This was wrong on both counts: eval is
   built with `reward_path=None` (verified at `train.py:47`), so the
   curve is the unshaped default Mario reward — and is therefore
   comparable across rounds. Identical methodology to the human
   heuristic eval. The mislabel told Claude to dismiss what is
   actually the most reliable cross-round signal. Fixed to:
   "Eval reward learning curve … evaluated on the default unshaped
   Mario reward — comparable across rounds".

3. **Reward hacking detected (informative finding, not a code bug).**
   In r2, training rollout `ep_rew_mean` reached 480 while eval
   `mean_reward` was locked at -40 and `ep_len_mean` at 201 (the
   stagnation termination limit). Claude's r2 reward keyed
   per-episode state by `id(prev_info)` — every step generated a new
   key with `max_x = 0`, so the "new ground" bonus fired every step
   regardless of actual progress. PPO converged to a deterministic
   policy that exploited this without moving forward. SB3's
   `best_model.zip` saved the step-25k checkpoint (eval reward 231
   before the exploit was discovered), so cross-round task selection
   still works correctly. Worth documenting as a real failure mode in
   the writeup: an LLM-designed reward function had an exploitable
   loophole, and PPO found it.

**R1 partial result preserved as a data point** (artifacts deleted, but
recorded here): with the corrected unbiased prompts, r1 achieved mean
x_pos 703 (vs baseline 315). Best shaped reward 632 at step 150k, then
oscillated. Claude correctly used y_pos direction (system prompt
clarification worked) but still used module-level state with a single
shared bucket.

Cleaned up: `artifacts/llm_v1_test_r{1,2}/`,
`configs/llm_v1_test_r{1,2}.json`,
`reward_functions/llm_v1_{current,r1,r2}.py`,
`prompts/llm_v1_r{1,2}_*.md`, `error.txt`. `LLM_LOOP_LOG.md` reset for
the third attempt.

### 2026-04-27 LLM v1 third attempt — completed (4 effective rounds)

Third attempt of the LLM reward loop completed end-to-end. Archived to
`archive/llm_v1_run3/` (artifacts, configs, reward functions, prompts,
loop log). Workspace cleared for a follow-up v1 run.

**Rounds completed:** 4 of 5 attempted. Round 4 was lost to the
`max_tokens=2048` API truncation — Claude's r4 response was cut
mid-statement at `reward +=` and failed validation. The loop's
retry-on-failure logic uses `continue`, which advances the for-loop
counter, so Claude's corrected response trained as r5 instead of r4.
Net effect: 4 effective iterations (r1, r2, r3, r5) instead of 5.

**Final task metrics** (best_model.zip, 10 deterministic episodes,
unshaped Mario reward):

| Round | Mean x_pos | Score | Best shaped (eval) | Final shaped (eval) |
|---|---|---|---|---|
| R1 | 594 | 100 | 502 @ 875k | -40 |
| R2 | 692 | 0 | 621 @ 825k | 251 |
| R3 | **722** | 100 | 630 @ 775k | 252 |
| R5 | 722 | 100 | 628 @ 600k | 231 |

**Best round: R3** (mean x_pos 722). `llm_v1_final.py` (now in
`archive/llm_v1_run3/reward_functions/`) is r3's reward function.

**Comparison to other agents (mean x_pos):**

| Agent | Steps | Mean x_pos |
|---|---|---|
| Baseline (no shaping) | 300k | 315 |
| LLM v1 best (r3) | 1M | 722 |
| Human heuristic v1 (first attempt) | ~1M | ~898 |
| Human heuristic v2 | 3M | 1523 |
| Human heuristic v3 | 5M | 2354 |

LLM v1 third attempt cleared baseline by 2.3× but plateaued below the
human first-attempt heuristic. R3 and R5 tied at exactly 722 x_pos —
two different reward functions converged to the same policy ceiling
(the agent reaches the first hard obstacle cluster at x≈720 and dies).

**Findings about Claude's reward design behavior:**

- **Claude reasoned explicitly in r3 and r5 docstrings.** R3 docstring
  diagnosed prior failures: "Heavy death penalty (-50) likely caused
  risk-aversion; per-step time penalty (-0.1) on a 4-frame skip stacks
  up and discouraged exploration; action-conditioned jump bonus
  probably biased the policy toward spamming jumps." R5 noted "agent
  learned a stable run-right-then-die policy … must break new ground
  to score well." This is real RL reasoning, not parameter twiddling.

- **Targeted corrections worked.** r2 had -50 death penalty and -0.1
  per-step time penalty; r3 softened death to -15 and removed time
  penalty. r2 had +0.5 jump-when-stuck action bonus; r3 removed it.
  r3's mean x_pos (722) > r2's (692), validating the diagnoses.

- **R5 attempted a strategic shift.** Pulled frontier bonus from
  1.0×gain to 5.0×gain while reducing symmetric dx to 0.1× — explicit
  attempt to break the "run right and stop" plateau by making *new
  ground* the dominant signal. Did not improve over r3 (also 722) but
  did not regress.

- **Persistent technical blind spot.** All four rounds used
  module-level state keyed by `id(prev_info)`, on the assumption that
  each env reuses its info dict across steps. The system prompt told
  Claude module state is shared across n_envs=4 but did not specify
  whether `prev_info` identity is stable. Claude filled the gap with
  an incorrect assumption. R2's first 175k steps locked at -40 with
  ep_len=201 (training rollout much higher) was likely caused by this
  — reward hacking from a broken state key. R2 recovered after the
  exploit became unstable; R1, R3, R5 didn't show the symptom as
  severely.

**Remaining open issues for next v1 run:**

1. `max_tokens=2048` causes occasional response truncation. Bump to
   4096 to avoid losing rounds.
2. Loop's retry-on-validation-failure increments `round_num`. Should
   retry within the same round so failed validation doesn't cost an
   iteration.
3. System prompt has a gap on info dict identity. Adding a neutral
   technical fact like "info and prev_info dict identity is not
   guaranteed stable across steps; using id() on them as a per-env
   state key is unreliable" would close the gap without dictating
   reward design.

**Status:** v1 third attempt is a complete, usable result on its own.
Workspace cleared and reset for a follow-up v1 run with the three
fixes above before running the 5M final.

### 2026-04-27 LLM v1 fourth attempt — completed (4 rounds)

Fourth attempt of the LLM reward loop completed all 5 round slots (r1–r4;
r5 was not reached). Three fixes applied vs third attempt:

1. `max_tokens` 2048 → 4096 — no truncation this run
2. Retry-on-validation-failure uses inner loop — failed validation no longer advances `round_num`
3. System prompt now includes: "info and prev_info dict identity is not guaranteed stable across steps; using id() on them as a per-env state key is unreliable"

**Rounds completed:** 4 of 5 (loop stopped after r4 as all 5 training slots were used).

**Final task metrics** (best_model.zip, 10 deterministic episodes + 20 stochastic episodes):

| Round | Det. x_pos | Stoch. mean x_pos | Stoch. std | Best eval reward | Notes |
|---|---|---|---|---|---|
| R1 | **1139** | **764** | 272 | 1053 @ 950k | Stateless, breakthrough |
| R2 | 296 | 315 | 0 | 252 @ 875k | Stagnation penalty, zero entropy |
| R3 | 705 | 595 | 201 | 629 @ 750k | Correct diagnosis, bimodal |
| R4 | 315 | 315 | 0 | 252 @ 150k | Slot-matching novelty, zero entropy |

**Best round: R1** (det. x_pos 1139, stoch. mean 764).

**Stochastic cross-run comparison** (20 episodes, deterministic=False):

| Agent | Steps | Stoch. mean x_pos | Std | Completion |
|---|---|---|---|---|
| Baseline | 1M | 579 | 96 | 0% |
| Human heuristic v3 | 5M | 2044 | 594 | 15% |
| LLM v1 R1 (best) | 1M | 764 | 272 | 0% |

Note on evaluation methodology: all prior task metrics used `deterministic=True`
(10 identical episodes in a deterministic env — effectively 1 episode replayed 10×).
This run also collected stochastic eval (`deterministic=False`, 20 episodes) which
better reflects the trained policy's actual distribution since PPO training itself
is stochastic. Human v3 deterministic eval = x_pos 2354; stochastic mean = 2044
with 15% completion rate (3/20 episodes reaching the flag). The stochastic numbers
are the more methodologically sound comparison.

**Findings about Claude's reward design behavior in this run:**

- **id(prev_info) instability fixed immediately.** After adding the neutral
  fact to the system prompt, Claude produced a fully stateless R1 with no
  module-level variables. The fix directly broke the x≈720 ceiling — R1
  reached x=1139 vs the third attempt's best of 722.

- **Claude still overcorrects on success.** R1 reached 1139; Claude's R2
  made aggressive changes (1.5×dx, -0.3 stagnation penalty per dx≤0 step)
  to push further. The stagnation penalty fires during mid-air jump frames,
  punishing the exact behavior needed to clear obstacles. Same failure mode
  as third attempt r2. Claude has no memory across runs.

- **R3 self-diagnosed R2 correctly again.** Docstring explicitly identified
  the stagnation-penalty-kills-jumping problem and reverted to lighter shaping.
  R3 is not as good as R1 but is a correct correction.

- **R4's novelty bonus creative but flawed.** Claude tried to reintroduce a
  frontier/novelty bonus without module state keyed by id(), using slot-matching
  by `prev_x` value. Two problems: multiple envs share starting x positions
  (slot confusion), and the implementation still uses module-level state (just
  with a different, also-unreliable keying scheme). Two-phase failure: -80
  reward first 125k (stagnation hacking), then fast-death trap for remaining
  875k. Zero entropy.

- **R2 and R4 both collapsed to zero entropy.** Both used module-level state
  with problematic keying — noisy shaped signals created gradients that drove
  the policy to deterministic local optima (fast death at x=315). Zero
  entropy means even stochastic sampling (`deterministic=False`) produces
  identical episodes.

- **Deterministic eval can misrepresent performance.** R1 deterministic=1139
  but stochastic mean=764 (std=272). The best_model.zip was saved at a
  checkpoint that happened to have one strong deterministic trajectory; the
  underlying policy is bimodal. This matters for honest reporting.

**R5 completed — selected as best round.** R5 (x_pos 1230) beat R1 (x_pos
1139) via stateless milestone bonuses every 250 pixels, scaling with depth.
`reward_functions/llm_v1_final.py` = R5's reward function.

**Final round results including R5:**

| Round | Det. x_pos | Stoch. mean | Stoch. std | Stoch. max | Notes |
|---|---|---|---|---|---|
| R1 | 1139 | 764 | 272 | 1139 | Stateless, late breakthrough @ 950k |
| R2 | 315 | 315 | 0 | 315 | Zero entropy collapse |
| R3 | 705 | 595 | 201 | 898 | Noisy bimodal |
| R4 | 315 | 315 | 0 | 315 | Zero entropy collapse |
| R5 | **1230** | **806** | 390 | **1395** | Milestone bonuses, selected |

**R5 key finding:** milestone bonuses (`+1.0 × bucket` when crossing x multiples
of 250) pushed the deterministic trajectory to 1230 and opened a stochastic
ceiling of 1395. Training diagnostics confirmed healthy training at end of run —
ep_rew_mean climbing 711→902, ep_len growing 87→113, entropy -0.07 to -0.34
(non-collapsed). Both R1 and R5 had identical late breakthrough timing (step
950k), suggesting training dynamics drives the breakthrough window regardless
of reward design.

**Feedback loop finding:** results did not monotonically improve across rounds.
R1 (no history): 1139. R2–R4: all regressed. R5 (full history): 1230. The
eval curve + task metrics alone were insufficient for consistent improvement.
Training diagnostics (entropy, ep_rew_mean, explained_variance) added to
`build_feedback_message()` for the next run to enable earlier collapse detection.

**Changes applied for next LLM v1 run:**
1. `build_feedback_message()` now includes last-8-update training diagnostics
   from TensorBoard (ep_rew_mean shaped, ep_len, entropy_loss, explained_variance)
2. System prompt updated with neutral technical descriptions of each diagnostic
3. EvalCallback: switch to `deterministic=False`, `n_eval_episodes=10`
4. Final comparison eval: 20 episodes, `deterministic=False`

**Next step:** archive this run, start new LLM v1 run with improved feedback.
Simultaneously retrain human heuristic v3 on separate system with `deterministic=False`
eval (n_eval_episodes=10). Both at 5M steps. Final comparison uses 20-episode
stochastic eval on all best models.

### 2026-04-27 LLM v1 fifth attempt — started

Fifth attempt of the LLM reward loop started. All fixes from fourth attempt
retained, plus new additions:

- `build_feedback_message()` now includes last-8-update training diagnostics
  from TensorBoard: ep_rew_mean (shaped), ep_len_mean, entropy_loss,
  explained_variance. Enables Claude to detect reward hacking (training
  ep_rew_mean >> eval reward), policy collapse (entropy → 0), and training
  stability issues without needing to infer them from the eval curve alone.
- System prompt updated with neutral technical descriptions of each diagnostic.
- EvalCallback: `deterministic=False`, `n_eval_episodes=10` — checkpoint
  selection now based on stochastic policy mean rather than single greedy
  trajectory. Methodologically consistent with stochastic PPO training.
- `eval_best_model()`: `deterministic=False`, `n_episodes=20` — best-round
  selection and task metrics use genuine stochastic distribution.

Human heuristic v3 retraining planned in parallel on separate system with
same eval changes for fair final comparison.

Additional fix before fifth attempt started:
- System prompt gap closed: now explicitly states there is no reliable
  per-env identifier in info dict (world/stage/life identical across all 4
  envs, id() unstable), and that stateless computation is the correct approach.
  All 5 rounds produced fully stateless reward functions — guardrail worked.

### 2026-04-28 LLM v1 fifth attempt — completed

All 5 rounds completed. Eval: 20-episode stochastic + deterministic, best_model.zip.

| Round | Stoch mean x_pos | Stoch max | Stoch std | Det mean | Notes |
|-------|-----------------|-----------|-----------|----------|-------|
| R1    | 1167            | 1511      | 242       | —        | Best mean; br as base, 0.5×dx |
| R2    | 689             | 722       | 89        | 434      | Regression; aggressive dx + hardcoded milestones |
| R3    | 1110            | 2011      | 365       | 434      | Best max ever; zero-base, 1.0×dx, no idle penalty |
| R4    | 921             | 1410      | 321       | 696      | Training collapsed at 450k; 350k checkpoint salvaged |
| R5    | 1140            | 1142      | 2         | 1141     | Most consistent; ultra-conservative, locked-in strategy |

Key observations:
- R1 has the best stochastic mean (1167); R3 has the best single-episode max (2011)
- R2 and R4 both collapsed due to overly aggressive velocity shaping and dense bonuses
- R5 over-corrected after R4 — ultra-conservative design produced a fixed, ceiling-less strategy
- Deterministic eval at x=434 for R2/R3 — greedy policy trapped by level structure; stochastic breaks past it
- All 5 rounds were fully stateless — the guardrail fix was effective

**Selected for final 5M run: R1** (`reward_functions/llm_v1_r1.py`)
Rationale: best stochastic mean, no collapse, moderate design that balances native br signal with modest shaping.

### 2026-04-28 LLM v1 final 5M run — completed and evaluated

Config: `configs/llm_v1_final.json` — R1 reward function (`llm_v1_final.py`), 5M steps, n_envs=4, ent_coef=0.02, deterministic=False eval.

**20-episode eval on best_model.zip:**

Stochastic (deterministic=False):
- x_pos: mean=1374, std=502, min=434, max=2130
- score: mean=305, std=213
- steps: mean=239, std=107
- all x_pos: [1508, 2130, 1417, 2022, 1130, 1128, 1945, 2130, 722, 434, 1495, 1517, 722, 677, 898, 1419, 1512, 1128, 1521, 2023]

Deterministic (deterministic=True):
- x_pos: mean=1129, std=0 — locked to fixed trajectory at 1129

**Final comparison (stochastic, 20 episodes):**

| Agent | Steps | x_pos mean | x_pos max | Notes |
|-------|-------|-----------|-----------|-------|
| Baseline | 1M | 579 | 722 | No reward shaping |
| LLM v1 R1 | 1M | 1167 | 1511 | Best 1M LLM round |
| LLM v1 Final | 5M | 1374 | 2130 | +18% over 1M R1 |
| Human v3 | 5M | 2044 | 3161 | 3 flag completions |

**Key findings:**
- LLM reward shaping improved 2.4× over baseline at 5M steps (579→1374)
- 5M training improved over 1M by +18% mean, +41% max
- Human heuristic outperformed LLM at same compute (2044 vs 1374 mean)
- Human v3 completed the level 3/20 times; LLM v1 did not complete the level
- LLM iterative loop showed R1 (first round) was the strongest — subsequent rounds either collapsed or overcorrected

### 2026-04-28 Human heuristic v3 — final eval (5M steps, existing run)

20-episode eval on `artifacts/human_heuristic_v3_seed0/models/best_model.zip` (best checkpoint: step 4,925,000).

**Stochastic (deterministic=False, 20 episodes):**
- x_pos: mean=2044, std=594, min=1431, max=3161
- score: mean=380, std=312
- steps: mean=239, std=124
- all x_pos: [3161, 2472, 2226, 1431, 1521, 3161, 1788, 1793, 1434, 2457, 1523, 1523, 1524, 2457, 3161, 1525, 1522, 1796, 2462, 1943]

**Deterministic (deterministic=True, 20 episodes):**
- x_pos: mean=2354, std=0, min=2354, max=2354
- score: mean=500, std=0
- steps: mean=455, std=0

**Note:** 3 episodes reached x=3161 (flag). Deterministic policy reliably reaches 2354. This is the strongest result across all agents evaluated so far. Sets the bar for the LLM v1 final 5M run.

---

### 2026-04-28 Human heuristic v3 stochastic retrain — final eval

Evaluated `artifacts/human_heuristic_v3_seed0_stochastic/models/best_model.zip` (5M steps, trained with det=False eval methodology).

**Stochastic (deterministic=False, 20 episodes):**
- x_pos: mean=1934, std=422, min=898, max=2475
- score: mean=410, std=259
- flags: 0/20
- all x_pos: [1787, 1794, 1671, 1665, 1654, 1664, 2469, 898, 2467, 2006, 1433, 1957, 1919, 1410, 2472, 2227, 2226, 2471, 2475, 2007]

**Deterministic (deterministic=True, 20 episodes):**
- x_pos: mean=1797, std=0

**Note:** Training with det=False eval produced a weaker best checkpoint than the original v3 (det=True) run. The original v3's best checkpoint at 4.925M was simply a stronger policy — methodology change alone does not guarantee improvement.

---

### 2026-04-28 Human heuristic v4 (10M) — final eval

Evaluated `artifacts/human_heuristic_v4_seed0/models/best_model.zip` (10M steps).

**Stochastic (deterministic=False, 20 episodes):**
- x_pos: mean=2378, std=935, min=594, max=3161
- score: mean=220, std=166
- flags: 9/20
- all x_pos: [2466, 2759, 3161, 3161, 3161, 2466, 594, 2466, 2469, 1228, 3161, 3161, 3161, 1959, 3161, 3161, 594, 594, 1523, 3161]

**Deterministic (deterministic=True, 20 episodes):**
- x_pos: mean=2402, std=0

**Note:** Dominant result. 9/20 flag completions. Deterministic policy consistently reaches x=2402. This is the strongest agent across all runs. Extra 5M steps over v3 yielded massive improvement in flag completion rate (3/20 → 9/20).

---

### 2026-04-28 LLM v1 R3 Final (5M) — final eval

Evaluated `artifacts/llm_v1_r3_final_seed0/models/best_model.zip`. Best checkpoint: step 4,775,000.

**Stochastic (deterministic=False, 20 episodes):**
- x_pos: mean=1441, std=355, min=696, max=1909
- score: mean=225, std=228
- steps: mean=192
- flags: 0/20
- all x_pos: [1759, 1429, 1672, 1670, 1435, 898, 1896, 1434, 1665, 1434, 1675, 696, 1127, 1674, 1909, 1433, 1528, 830, 898, 1767]

**Deterministic (deterministic=True, 20 episodes):**
- x_pos: mean=1904, std=0

**Note:** R3 at 5M slightly outperforms R1 at 5M in stochastic mean (1441 vs 1374) and substantially in deterministic (1904 vs 1129). Lower upside ceiling than R1 (max 1909 vs 2130). Neither clears the flag.

---

### 2026-04-28 All runs complete — final comparison

| Agent | Steps | Stoch mean x_pos | Stoch max | Flags | Det x_pos |
|-------|-------|-----------------|-----------|-------|-----------|
| Baseline | 1M | 579 | 722 | 0/20 | — |
| LLM R1 (loop round 1) | 1M | 1167 | 1511 | 0/20 | — |
| LLM v1 Final (R1) | 5M | 1374 | 2130 | 0/20 | 1129 |
| LLM v1 R3 Final | 5M | 1441 | 1909 | 0/20 | 1904 |
| Human v3 (original) | 5M | 2044 | 3161 | 3/20 | 2354 |
| Human v3 (stoch retrain) | 5M | 1934 | 2475 | 0/20 | 1797 |
| Human v4 | 10M | 2378 | 3161 | 9/20 | 2402 |

Improvement over baseline: LLM 1M +101%, LLM 5M +137%, Human 5M +253%, Human 10M +311%.
