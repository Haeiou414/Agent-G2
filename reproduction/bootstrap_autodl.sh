#!/usr/bin/env bash
# Idempotent Agent-G2 ALFWorld environment setup for an AutoDL NVIDIA instance.
# Run from anywhere inside the checked-out repository:
#   bash reproduction/bootstrap_autodl.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${AGENT_G2_CONDA_ENV:-agent-g2}
SETUP_DIR="$ROOT_DIR/outputs/setup"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "bootstrap_autodl.sh requires Linux" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; start a GPU instance first" >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is unavailable; choose an AutoDL image with Miniconda/Anaconda" >&2
    exit 2
fi

# shellcheck source=autodl_env.sh
source "$ROOT_DIR/reproduction/autodl_env.sh"
mkdir -p "$SETUP_DIR"

if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    conda create -n "$ENV_NAME" python=3.12 -y
fi

PYTHON=(conda run -n "$ENV_NAME" python)
PIP=("${PYTHON[@]}" -m pip)

"${PIP[@]}" install --upgrade pip wheel packaging ninja
"${PIP[@]}" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"${PIP[@]}" install flash-attn==2.7.4.post1 --no-build-isolation
"${PIP[@]}" install -e "$ROOT_DIR"
"${PIP[@]}" install \
    vllm==0.8.5 \
    gymnasium==0.29.1 \
    stable-baselines3==2.6.0 \
    alfworld \
    huggingface_hub

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
    conda run -n "$ENV_NAME" alfworld-download -f
fi

"${PIP[@]}" check
conda run -n "$ENV_NAME" python -m unittest discover \
    -s "$ROOT_DIR/tests/reproduction" -v

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits > "$SETUP_DIR/nvidia-smi.csv"
"${PIP[@]}" freeze > "$SETUP_DIR/pip-freeze.txt"
conda run -n "$ENV_NAME" python -m reproduction.verify_gpu_environment --json \
    > "$SETUP_DIR/gpu-preflight.json"

echo "AutoDL environment is ready."
echo "Before each new shell or tmux session, run:"
echo "  source $ROOT_DIR/reproduction/autodl_env.sh"
echo "  conda activate $ENV_NAME"
echo "Evidence written to $SETUP_DIR"
