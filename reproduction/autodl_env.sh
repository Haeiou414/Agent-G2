#!/usr/bin/env bash
# Shared data/cache locations for an AutoDL instance. Source this file before
# downloading models or running experiments.

if [[ -z "${AGENT_G2_DATA_ROOT:-}" ]]; then
    if [[ -d /root/autodl-tmp ]]; then
        export AGENT_G2_DATA_ROOT=/root/autodl-tmp/agent-g2-data
    else
        export AGENT_G2_DATA_ROOT="$PWD/.agent-g2-data"
    fi
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-$AGENT_G2_DATA_ROOT/alfworld}"
export HF_HOME="${HF_HOME:-$AGENT_G2_DATA_ROOT/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$AGENT_G2_DATA_ROOT/pip-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$ALFWORLD_DATA" "$HF_HOME" "$PIP_CACHE_DIR"
