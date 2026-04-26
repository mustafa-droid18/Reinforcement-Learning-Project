# LLM Reward Loop

## How it works

`llm_reward_loop.py` runs a fully automated iterative loop:

1. Calls the Claude API with the environment description and function signature
2. Extracts and validates the generated reward function
3. Trains PPO for 1M steps
4. Parses eval results into a feedback summary
5. Sends results back to Claude for revision
6. Repeats up to 5 rounds
7. Promotes the best reward function to `reward_functions/llm_v1_final.py`

No human input between rounds. Every prompt and response is saved to `prompts/`.

## File layout

| File | Purpose |
|---|---|
| `llm_reward_loop.py` | The loop script |
| `configs/llm_v1_test.json` | 1M step config used during the loop |
| `configs/llm_v1_final.json` | 5M step config for final v1 run |
| `configs/llm_v2_final.json` | 5M step config for final v2 run |
| `reward_functions/llm_v1_rN.py` | Reward from loop round N |
| `reward_functions/llm_v1_current.py` | Current round's reward (overwritten each round) |
| `reward_functions/llm_v1_final.py` | Best reward from the loop (used for final run) |
| `reward_functions/llm_v2_final.py` | LLM v2 revision based on final v1 results |
| `prompts/llm_v1_rN_prompt.md` | Full conversation up to round N |
| `prompts/llm_v1_rN_response.md` | Claude's response and extracted code for round N |
| `prompts/llm_v1_loop_summary.json` | Best round, best reward, all round results |

## Running the loop

```bash
export ANTHROPIC_API_KEY=your_key_here
PYTHONPATH=src python llm_reward_loop.py
```

## After the loop finishes

1. Check `prompts/llm_v1_loop_summary.json` to see which round was best
2. Run the final 5M training:
   ```bash
   PYTHONPATH=src python -m mario_rl.train --config configs/llm_v1_final.json
   ```
3. Evaluate:
   ```bash
   PYTHONPATH=src python -m mario_rl.eval --config configs/llm_v1_final.json \
     --model artifacts/llm_v1_final_seed0/models/best_model.zip --episodes 10
   ```

## LLM v2

After the v1 final run, do one more LLM revision using the 5M results as feedback,
save the output as `reward_functions/llm_v2_final.py`, then train with `configs/llm_v2_final.json`.

## Success criteria

| Metric | Threshold |
|---|---|
| LLM v1 beats baseline | mean x_pos > 315 |
| LLM v2 beats LLM v1 | mean x_pos improves |
| LLM v1 competitive with human v1 | mean x_pos approaching 898 |
| Stretch | LLM v2 approaches human v3 (x_pos 2354) without 8 iterations |
