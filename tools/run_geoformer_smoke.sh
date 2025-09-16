#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/tools/train_geoformer.py" \
  --max-steps 1 \
  --limit-train-pairs 4 \
  --limit-eval-samples 2 \
  --eval-steps 1 \
  --disable-wandb \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --num-train-epochs 1 \
  --output-dir checkpoints/geoformer_pretrain_smoke
