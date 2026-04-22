# Plan fixes before you spend time training

The proposal is good at the idea level. These are the implementation details
that need to be explicit before you trust any result.

## 1. Separate the training reward from the evaluation metrics

Right now the plan says "default reward or LLM-generated reward" and then lists
metrics. That is not enough. If you train on a shaped reward and also judge the
run using that same reward, the comparison is biased.

Fix:

- Train with the selected reward function.
- Evaluate every run using task-level metrics from `info`, not the shaped reward.
- Primary metrics should be:
  - completion rate
  - final `x_pos`
  - average max `x_pos`
  - score
  - sample efficiency under a fixed timestep budget

## 2. Lock the exact environment setup

The plan says "gym-super-mario-bros" and "Mario frames", but that still leaves
room for accidental experiment drift.

Fix:

- Pick one exact level such as `SuperMarioBros-1-1-v0`.
- Pick one action set such as `SIMPLE_MOVEMENT`.
- Fix the observation preprocessing:
  - grayscale or RGB
  - resize shape
  - frame stack count
- Fix the episode termination policy and maximum timestep budget.

## 3. Define the compute budget numerically

`Train PPO agent for fixed timestep budget` is too vague.

Fix:

- State the exact timesteps per run.
- State the number of seeds.
- State the evaluation frequency.
- State whether all runs use the same PPO hyperparameters.

## 4. Make the comparison fair across runs

The proposal says baseline vs LLM v1 vs LLM v2, which is correct, but fairness
depends on tighter controls.

Fix:

- Same seed set across all runs.
- Same PPO hyperparameters across all runs.
- Same environment and preprocessing across all runs.
- Same evaluation protocol across all runs.

## 5. Define what "iterative improvement" means

The current loop is conceptually clear, but the revision rule is underspecified.

Fix:

- Save the exact prompt sent to the LLM.
- Save the reward function returned by the LLM.
- Feed back only a standard summary:
  - learning curve
  - evaluation summary
  - failure modes observed
- Limit the iteration count in advance. For this project, `v1` and `v2` are enough.

## 6. Add a failure policy for invalid reward code

An LLM-generated reward file can crash, produce NaNs, or reward degenerate
behavior.

Fix:

- Validate every reward function before training.
- Clip or sanity-check reward magnitude.
- Abort the run if the reward returns non-finite values.

## 7. Tighten the expected outcome

`We expect the LLM-designed reward to improve learning efficiency` is fine as a
hypothesis, but the plan should also define what counts as a successful result.

Fix:

- Success criterion example:
  - at equal timesteps, LLM reward reaches higher mean `x_pos` than baseline
  - or reaches the same performance in fewer timesteps
- Negative result is still valid if reported clearly.

## Recommended coding order

1. Baseline PPO pipeline.
2. Evaluation script that reports task metrics.
3. Reward-loader interface.
4. Heuristic reward file to prove the pipeline works.
5. LLM generation loop after the pipeline is stable.

