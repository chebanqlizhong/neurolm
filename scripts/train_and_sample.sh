#!/usr/bin/env bash
# 在语料上训练 NeuroLM，并从结果采样。
#
# 用法:
#   scripts/train_and_sample.sh [config] [prompt]
#
# 示例:
#   scripts/train_and_sample.sh configs/small.yaml "The transformer "
set -euo pipefail

CONFIG="${1:-configs/small.yaml}"
PROMPT="${2:-The transformer }"
RUN_NAME="$(grep -E '^run_name:' "$CONFIG" | awk '{print $2}')"

export PYTHONPATH="src:${PYTHONPATH:-}"

echo "==> 开始训练，配置：$CONFIG"
python -m neurolm train --config "$CONFIG"

echo "==> 评估 runs/${RUN_NAME}/final.npz"
python -m neurolm evaluate --config "$CONFIG" --checkpoint "runs/${RUN_NAME}/final.npz"

echo "==> 采样"
python -m neurolm generate \
  --config "$CONFIG" \
  --checkpoint "runs/${RUN_NAME}/final.npz" \
  --prompt "$PROMPT" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40

echo "==> 损失曲线"
python scripts/plot_history.py "runs/${RUN_NAME}/history.json"
