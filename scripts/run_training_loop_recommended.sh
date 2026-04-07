#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/run_selfplay_train_loop.py \
  --run-name loop_v1 \
  --iterations 30 \
  --games-per-run 40 \
  --workers 8 \
  --simulations 100 \
  --time-budget-ms 60 \
  --max-plies 160 \
  --epochs 2 \
  --train-batch-size 256 \
  --replay-shards 5 \
  --checkpoint-every 5 \
  --size 10 \
  --seed 1 \
  --initial-model-path data/models/latest.torchscript.pt
