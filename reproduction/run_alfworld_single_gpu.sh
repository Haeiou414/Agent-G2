#!/usr/bin/env bash
# Same-budget ALFWorld comparison for one 24 GB NVIDIA GPU.
# Usage: bash reproduction/run_alfworld_single_gpu.sh METHOD SEED [ENGINE] [extra Hydra overrides]
# METHOD: gmsv | grpo | target_acc | fixed_sigma | no_aux_sft | deterministic_mean

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 {gmsv|grpo|target_acc|fixed_sigma|no_aux_sft|deterministic_mean} SEED [ENGINE] [extra Hydra overrides]" >&2
    exit 2
fi

METHOD=$1
SEED=$2
ENGINE=${3:-vllm}
if [[ $# -ge 3 ]]; then
    shift 3
else
    shift 2
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

case "$METHOD" in
    gmsv)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh"
        METHOD_OVERRIDES=(
            gmsv.num_difficulty_groups=3
            gmsv.lambda_value=1.0
            gmsv.mu_global_max=1.0
            gmsv.max_length=1.0
            "gmsv.seed=$SEED"
        )
        ;;
    grpo)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/grpo_trainer/run_alfworld.sh"
        # Keep a non-empty array for compatibility with the Bash 3.2 shipped
        # on macOS when `set -u` is enabled.
        METHOD_OVERRIDES=(gmsv.enable=false)
        ;;
    target_acc)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/fix_acc_hint_trainer/run_alfworld.sh"
        METHOD_OVERRIDES=(
            "fix_acc_hint.seed=$SEED"
        )
        ;;
    fixed_sigma)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh"
        METHOD_OVERRIDES=(
            gmsv.num_difficulty_groups=3
            gmsv.lambda_value=1.0
            gmsv.mu_global_max=1.0
            gmsv.max_length=1.0
            gmsv.sigma_mode=fixed_sigma
            gmsv.fixed_sigma=0.1
            "gmsv.seed=$SEED"
        )
        ;;
    no_aux_sft)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh"
        METHOD_OVERRIDES=(
            gmsv.num_difficulty_groups=3
            gmsv.lambda_value=1.0
            gmsv.mu_global_max=1.0
            gmsv.max_length=1.0
            gmsv.prefix_sft.enable=false
            "gmsv.seed=$SEED"
        )
        ;;
    deterministic_mean)
        UPSTREAM_SCRIPT="$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh"
        METHOD_OVERRIDES=(
            gmsv.num_difficulty_groups=3
            gmsv.lambda_value=1.0
            gmsv.mu_global_max=1.0
            gmsv.max_length=1.0
            gmsv.prefix_sample_mode=deterministic_mean
            "gmsv.seed=$SEED"
        )
        ;;
    *)
        echo "unknown METHOD '$METHOD'; expected gmsv, grpo, target_acc, fixed_sigma, no_aux_sft, or deterministic_mean" >&2
        exit 2
        ;;
esac

RUN_TAG=${RUN_TAG:-}
if [[ -n "$RUN_TAG" ]] && ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
BASE_MODEL_PATH=${BASE_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
RUN_NAME="limited_${METHOD}_qwen2.5_1.5b_seed${SEED}${RUN_SUFFIX}"
COMMON_OVERRIDES=(
    +data.seed="$SEED"
    data.train_batch_size=4
    data.val_batch_size=64
    data.max_prompt_length=7000
    actor_rollout_ref.model.path="$BASE_MODEL_PATH"
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.free_cache_engine=true
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    env.rollout.n=4
    "env.seed=$SEED"
    trainer.n_gpus_per_node=1
    trainer.total_training_steps=80
    trainer.test_freq=10
    trainer.save_freq=40
    trainer.val_before_train=true
    trainer.resume_mode=disable
    "trainer.default_local_dir=outputs/$RUN_NAME/checkpoints"
    "trainer.experiment_name=$RUN_NAME"
    "trainer.logger=[console,jsonl]"
    "trainer.rollout_data_dir=outputs/$RUN_NAME/rollouts"
    "trainer.validation_data_dir=outputs/$RUN_NAME/validation"
    "trainer.train_rollout_detail_data_dir=outputs/$RUN_NAME/train_details"
    trainer.rollout_detail_dump_freq=10
)

COMMAND=(
    bash "$UPSTREAM_SCRIPT" "$ENGINE"
    "${COMMON_OVERRIDES[@]}"
    "${METHOD_OVERRIDES[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

RUN_DIR="outputs/$RUN_NAME"
if [[ -e "$RUN_DIR" ]]; then
    echo "run directory already exists: $RUN_DIR; set RUN_TAG to create a distinct retry" >&2
    exit 2
fi

export PYTHONHASHSEED=$SEED
export VERL_METRICS_JSONL_PATH="$RUN_DIR/metrics.jsonl"
export VERL_RESOLVED_CONFIG_PATH="$ROOT_DIR/$RUN_DIR/resolved_config.yaml"
python3 -m reproduction.capture_run_manifest \
    --output "$RUN_DIR/run_manifest.json" \
    --method "$METHOD" \
    --seed "$SEED" \
    --command "${COMMAND[@]}"

set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/console.log"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e
python3 -m reproduction.finalize_run_manifest \
    --manifest "$RUN_DIR/run_manifest.json" \
    --exit-code "$RUN_EXIT_CODE"
exit "$RUN_EXIT_CODE"
