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
