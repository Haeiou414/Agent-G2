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
export TMPDIR="${TMPDIR:-$AGENT_G2_DATA_ROOT/tmp}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$AGENT_G2_DATA_ROOT/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$AGENT_G2_DATA_ROOT/conda/pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

mkdir -p "$ALFWORLD_DATA" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"
