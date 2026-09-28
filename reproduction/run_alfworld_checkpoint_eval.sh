#!/usr/bin/env bash
# Evaluate the released Agent-G2 checkpoint without training or validation guidance.
# Usage: bash reproduction/run_alfworld_checkpoint_eval.sh CHECKPOINT_DIR SEED [ENGINE] [extra Hydra overrides]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 CHECKPOINT_DIR SEED [ENGINE] [extra Hydra overrides]" >&2
    exit 2
fi

CHECKPOINT_DIR=$1
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
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 2
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CHECKPOINT_DIR=$(cd "$CHECKPOINT_DIR" && pwd)
cd "$ROOT_DIR"
# Instance restarts clear exported variables.  Resolve the persistent
# ALFWorld location on every invocation instead of relying on login-shell
# state; an unresolved path makes TextWorld spin forever over zero games.
# shellcheck source=autodl_env.sh
source "$ROOT_DIR/reproduction/autodl_env.sh"
RUN_TAG=${RUN_TAG:-}
if [[ -n "$RUN_TAG" ]] && ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
RUN_NAME="checkpoint_eval_agent_g2_alfworld_1.5b_seed${SEED}${RUN_SUFFIX}"
UPSTREAM_SCRIPT="$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh"

COMMAND=(
    bash "$UPSTREAM_SCRIPT" "$ENGINE"
    +data.seed="$SEED"
    data.val_batch_size=128
    data.max_prompt_length=4096
    actor_rollout_ref.model.path="$CHECKPOINT_DIR"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=true
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    env.rollout.n=1
    "env.seed=$SEED"
    # The 128 validation environments plus 16 train placeholders otherwise
    # reserve 14.4 of 16 CPUs at the upstream 0.1 default. Ray then cannot
    # place the 1-CPU/1-GPU model worker and waits forever.
    env.resources_per_worker.num_cpus=0.05
    gmsv.apply_on_validation=false
    trainer.n_gpus_per_node=1
    trainer.val_before_train=true
    trainer.val_only=true
    trainer.resume_mode=disable
    "trainer.default_local_dir=outputs/$RUN_NAME/checkpoints"
    "trainer.experiment_name=$RUN_NAME"
    "trainer.logger=[console,jsonl]"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
    echo "ALFWorld training data is missing: $ALFWORLD_DATA/json_2.1.1/train" >&2
    echo "run: bash reproduction/install_alfworld_autodl.sh" >&2
    exit 2
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
    --method public_checkpoint_eval \
    --seed "$SEED" \
    --command "${COMMAND[@]}"
python3 -m reproduction.capture_checkpoint_identity \
    --checkpoint "$CHECKPOINT_DIR" \
    --output "$RUN_DIR/checkpoint_identity.json"

set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/console.log"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e
python3 -m reproduction.finalize_run_manifest \
    --manifest "$RUN_DIR/run_manifest.json" \
    --exit-code "$RUN_EXIT_CODE"
exit "$RUN_EXIT_CODE"
