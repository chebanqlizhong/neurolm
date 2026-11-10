#!/usr/bin/env bash
# Train NeuroLM on a corpus and sample from the result.
#
# Usage:
#   scripts/train_and_sample.sh [config] [prompt]
#
# Example:
#   scripts/train_and_sample.sh configs/small.yaml "The transformer "
set -euo pipefail

CONFIG="${1:-configs/small.yaml}"
PROMPT="${2:-The transformer }"
RUN_NAME="$(grep -E '^run_name:' "$CONFIG" | awk '{print $2}')"

export PYTHONPATH="src:${PYTHONPATH:-}"

echo "==> Training with $CONFIG"
python -m neurolm train --config "$CONFIG"

echo "==> Evaluating runs/${RUN_NAME}/final.npz"
python -m neurolm evaluate --config "$CONFIG" --checkpoint "runs/${RUN_NAME}/final.npz"

echo "==> Sampling"
python -m neurolm generate \
  --config "$CONFIG" \
  --checkpoint "runs/${RUN_NAME}/final.npz" \
  --prompt "$PROMPT" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40

echo "==> Loss curve"
python scripts/plot_history.py "runs/${RUN_NAME}/history.json"
