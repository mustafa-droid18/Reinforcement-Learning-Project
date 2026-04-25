# LLM Iteration Notes

This file explains the intended LLM reward-design loop for the Mario PPO
project and summarizes the current implementation and experiment status.

## Project goal

The project question is not "can PPO solve Mario from pixels?" by itself.
The actual question is:

`Can an LLM assist reward engineering through an iterative loop, and does that produce better PPO behavior than the default reward or a human heuristic reward?`

That means PPO should stay fixed as the learning algorithm while reward design
is the main experimental variable.

## Why the LLM process should be iterative

The reward-design discussion already showed that a single LLM response is not a
strong experiment. The better framing is:

1. Start from evidence.
2. Let the LLM propose a reward function.
3. Train for a short fixed budget.
4. Evaluate behavior using metrics and video.
5. Feed the failure mode back to the LLM.
6. Revise the reward and repeat for a bounded number of attempts.

This matches the original proposal's "LLM proposes -> PPO trains -> results fed
back -> LLM revises" loop. The bounded-search version is just a more concrete
implementation of that idea.

## Recommended LLM protocol

Use a fixed and documented loop:

1. Keep the environment, PPO hyperparameters, and action set fixed.
2. Allow the LLM to edit only `compute_reward(...)`.
3. Give each candidate a short training budget, ideally `25,000` timesteps.
4. Evaluate each candidate on task metrics, not shaped reward totals.
5. Keep the best candidate and revise only if the new one is worse.
6. Stop after a fixed maximum number of revisions, such as `5`.
7. Promote the best candidate to a longer run.

Suggested selection rule:

- Primary metric: mean evaluation reward under the same environment reward path.
- Secondary metrics: episode length, videos, score, and progress stability.
- For the final report, still use task-level metrics like `x_pos`,
  completion rate, and score where available.

## LLM prompt constraints

The LLM should be told:

- use only `base_reward`, `prev_info`, `info`, `action`, `terminated`,
  `truncated`
- do not hardcode level-specific coordinates such as a known enemy or pipe
  position
- keep reward magnitudes moderate
- avoid obvious reward hacking such as rewarding stalling
- explain each reward term in plain language

The reward file output should be saved exactly as generated, for example:

- `reward_functions/llm_v1.py`
- `reward_functions/llm_v2.py`

The corresponding prompts and responses should also be saved, for example:

- `prompts/llm_v1_prompt.md`
- `prompts/llm_v1_response.md`
- `prompts/llm_v2_prompt.md`
- `prompts/llm_v2_response.md`

## What has been implemented so far

### Core training pipeline

The repo now has a working Mario PPO pipeline under `src/mario_rl/`:

- `config.py`: experiment and PPO dataclasses
- `reward_api.py`: dynamic reward loading and validation
- `env_factory.py`: Mario environment construction and wrappers
- `vec_env.py`: SB3-compatible vectorized environment setup
- `train.py`: PPO training entrypoint
- `eval.py`: saved-model evaluation
- `video.py`: video export for trained policies
- `summarize_eval.py`: `.npz` evaluation history to CSV conversion

### Baseline experiment

Baseline config:

- `configs/baseline.json`

Baseline result:

- mean reward: `252`
- mean `x_pos`: `315`
- mean score: `0`
- completion rate: `0.0`

Observed behavior:

- Mario mostly runs right into the first enemy.

Conclusion:

- the default reward is a valid baseline, but it is weak.

### Human heuristic reward

Human heuristic config:

- `configs/human_heuristic.json`

Human heuristic reward evolved in two stages:

1. forward progress, backward penalty, score bonus, flag bonus, failure penalty
2. added a stagnation-oriented penalty for time passing without horizontal
   progress

Current human heuristic reward file:

- `reward_functions/human_heuristic.py`

Current key terms:

- keep `base_reward`
- `+0.1 * forward progress`
- `-0.02 * backward movement`
- `+score_delta / 100`
- `-0.5 * time_delta` when time passes and `x_pos` does not improve
- `+500` on flag
- `-100` on non-completion
- final square-root reward compression

### Training infrastructure improvements

To support the heuristic and later LLM loop, the codebase now includes:

- config-controlled video recording
- MP4 export support
- vector-env-safe frame stacking and image transposition
- old-Gym/Mario compatibility wrappers
- `max_stagnation_steps` in config
- `StagnationTerminationWrapper` to cut off stuck trajectories early
- callback frequency scaling so `eval_freq` and `video_freq` are interpreted in
  actual timesteps even when `n_envs > 1`

## Current experiment status

### `human_heuristic_seed0`

The current `artifacts/human_heuristic_seed0/` folder reflects a newer partial
rerun rather than the older full run that was discussed earlier.

Latest visible evaluation history from
`artifacts/human_heuristic_seed0/eval/evaluation_history.csv`:

- `25k`: reward `252`, episode length `27`
- `50k`: reward `231`, episode length `40`
- `75k`: reward `231`, episode length `40`
- `100k`: reward `342`, episode length `103`

Interpretation:

- this run improves over the default baseline by `100k`
- it still does not clearly solve the obstacle-timing problem

### `human_heuristic_v2_seed0`

Separate config:

- `configs/human_heuristic_v2.json`

This was created so the revised heuristic would not overwrite the older
artifacts.

Visible evaluation checkpoints in
`artifacts/human_heuristic_v2_seed0/eval/evaluations.npz` currently run from
`25k` to `700k`.

Representative pattern:

- `25k`: `250`
- `50k`: `7`
- `75k`: `372`
- `300k`: `528`
- `500k`: `-13`
- `575k`: `373`
- `700k`: `374`

Interpretation:

- the revised heuristic can produce much better checkpoints than the baseline
- training remains unstable
- this is exactly the kind of setting where a bounded LLM revision loop makes
  sense

## LLM stage status

There is now an untracked file:

- `reward_functions/llm_v1.py`

That suggests the repo is ready to start the actual LLM reward stage, but the
LLM workflow is not yet fully documented in files and has not been run in a
clean, bounded, reproducible way.

## Recommended next steps

1. Create prompt files for the first real LLM candidate.
2. Save the exact LLM response into `reward_functions/llm_v1.py`.
3. Add a dedicated config for the first LLM candidate.
4. Run the candidate for a short fixed budget such as `25k`.
5. Record:
   - evaluation metrics
   - video
   - short written diagnosis
6. Revise once based on that evidence and save the next version as
   `reward_functions/llm_v2.py`.
7. Compare:
   - default baseline
   - human heuristic
   - best LLM candidate
   - optional refined LLM candidate

## Bottom line

The project is already past the "does the code work?" phase.

The repo now has:

- a working baseline
- a documented human heuristic baseline
- video logging
- evaluation logging
- stagnation handling
- a clear path to an iterative LLM reward-search experiment

The remaining work is not basic scaffolding anymore. The remaining work is to
run the LLM loop in a disciplined and reproducible way.
