#!/usr/bin/env bash
# Paper-faithful ALFWorld launcher for Appendix B, Table 5.
#
# The upstream recipe remains untouched. Hydra applies these trailing overrides
# after the official script's values so that both configurations stay runnable.

set -euo pipefail

ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then
    shift
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

exec bash "$ROOT_DIR/examples/gmsv_trainer/run_alfworld.sh" "$ENGINE" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    data.max_prompt_length=7000 \
    gmsv.num_difficulty_groups=3 \
    gmsv.lambda_value=1.0 \
    gmsv.mu_global_max=1.0 \
    gmsv.max_length=1.0 \
    trainer.total_epochs=200 \
    "$@"
