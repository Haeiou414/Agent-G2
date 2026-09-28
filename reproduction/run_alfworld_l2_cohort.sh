#!/usr/bin/env bash
# Run one named, same-budget L2 cohort sequentially on a single GPU.
# Usage: bash reproduction/run_alfworld_l2_cohort.sh {primary|ablations} SEED [SEED ...]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 {primary|ablations} SEED [SEED ...]" >&2
    exit 2
fi

COHORT=$1
shift
SEEDS=("$@")
case "$COHORT" in
    primary)
        METHODS=(grpo target_acc gmsv)
        ;;
    ablations)
        METHODS=(fixed_sigma no_aux_sft deterministic_mean)
        ;;
    *)
        echo "unknown cohort '$COHORT'; expected primary or ablations" >&2
        exit 2
        ;;
esac

for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "SEED must be a non-negative integer: $seed" >&2
        exit 2
    fi
done

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"
export RUN_TAG=${RUN_TAG:-l2-v1}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "refusing to start a formal cohort from a dirty Git worktree" >&2
        exit 2
    fi
    python3 -m reproduction.verify_gpu_environment
fi

# Validate every destination before spending GPU time on the first run.
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_dir="outputs/limited_${method}_qwen2.5_1.5b_seed${seed}_${RUN_TAG}"
        if [[ -e "$run_dir" ]]; then
            echo "run directory already exists: $run_dir" >&2
            exit 2
        fi
    done
done

for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "=== L2 cohort=$COHORT method=$method seed=$seed tag=$RUN_TAG ==="
        bash reproduction/run_alfworld_single_gpu.sh "$method" "$seed" vllm
    done
done
