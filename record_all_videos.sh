#!/usr/bin/env bash
# Records one video per LLM loop iteration from the saved best models.
# Run from the project root with the mario-rl conda env active:
#   bash record_all_videos.sh

set -e
export PYTHONPATH=src

PYTHON=/opt/anaconda3/envs/mario-rl/bin/python

for v in 1 2 3 4 5; do
  MODEL="artifacts/llm_loop_v${v}_seed0/models/best_model.zip"
  CONFIG="configs/llm_loop_v${v}.json"
  OUTPUT="artifacts/llm_loop_v${v}_seed0/videos/policy_video.mp4"

  if [ ! -f "$MODEL" ]; then
    echo "[skip] v${v} — model not found: $MODEL"
    continue
  fi

  mkdir -p "artifacts/llm_loop_v${v}_seed0/videos"
  echo "[record] v${v} → $OUTPUT"
  $PYTHON -m mario_rl.video \
    --config "$CONFIG" \
    --model  "$MODEL" \
    --output "$OUTPUT" \
    --max-steps 1200 \
    --fps 30
done

echo ""
echo "Done. Videos saved to artifacts/llm_loop_v*/videos/"
