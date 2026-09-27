#!/usr/bin/env bash
# Low-cost end-to-end smoke test for one 24 GB GPU.
# Usage: bash reproduction/run_alfworld_smoke.sh [SEED] [ENGINE]

set -euo pipefail

SEED=${1:-1}
ENGINE=${2:-vllm}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export RUN_TAG=${RUN_TAG:-smoke}

exec bash reproduction/run_alfworld_single_gpu.sh gmsv "$SEED" "$ENGINE" \
    env.max_steps=5 \
    trainer.total_training_steps=8 \
    trainer.test_freq=8 \
    trainer.save_freq=8 \
    trainer.val_before_train=false \
    "data.val_files=$HOME/data/verl-agent/text/train.parquet" \
    data.val_batch_size=16
