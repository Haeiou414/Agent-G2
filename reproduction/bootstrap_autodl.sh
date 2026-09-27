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

# The pinned PyTorch 2.6.0+cu124 / vLLM 0.8.5 stack predates Blackwell.
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]')
if ! [[ "$GPU_COMPUTE_CAP" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "cannot verify GPU compute capability with nvidia-smi; refusing to install an unverified CUDA stack" >&2
    exit 2
fi
if (( ${GPU_COMPUTE_CAP%%.*} >= 10 )); then
    echo "GPU compute capability $GPU_COMPUTE_CAP is Blackwell or newer; this CUDA 12.4 / PyTorch 2.6 recipe is unsupported. Use an Ada/Ampere GPU or a separately validated CUDA 12.8+ stack." >&2
    exit 2
fi

# shellcheck source=autodl_env.sh
source "$ROOT_DIR/reproduction/autodl_env.sh"
mkdir -p "$SETUP_DIR"
cd "$ROOT_DIR"
python3 -m reproduction.check_network > "$SETUP_DIR/network-preflight.txt"

if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    conda create -n "$ENV_NAME" python=3.12 -y
fi

PYTHON=(conda run -n "$ENV_NAME" python)
PIP=("${PYTHON[@]}" -m pip)
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$GPU_COMPUTE_CAP}

"${PIP[@]}" install --upgrade pip wheel packaging ninja
"${PIP[@]}" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"${PIP[@]}" install flash-attn==2.7.4.post1 --no-build-isolation
"${PIP[@]}" install -e "$ROOT_DIR"
"${PIP[@]}" install \
    vllm==0.8.5 \
    gymnasium==0.29.1 \
    stable-baselines3==2.6.0 \
    huggingface_hub
bash "$ROOT_DIR/reproduction/install_alfworld_autodl.sh"
"${PIP[@]}" install \
    wandb==0.19.11 \
    google-api-core==2.24.2 \
    proto-plus==1.26.1 \
    opentelemetry-exporter-prometheus==0.47b0

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
    conda run -n "$ENV_NAME" alfworld-download || {
        # ALFWorld's downloader fetches an optional visual detector after all
        # text-game archives. The text-only reproduction does not need it.
        if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ||
              ! -d "$ALFWORLD_DATA/json_2.1.1/valid_seen" ||
              ! -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ||
              ! -f "$ALFWORLD_DATA/logic/alfred.pddl" ||
              ! -f "$ALFWORLD_DATA/logic/alfred.twl2" ]]; then
            echo "ALFWorld text data is incomplete" >&2
            exit 1
        fi
        echo "ALFWorld visual detector download failed; text data is present"
    }
fi
for required in \
    "$ALFWORLD_DATA/json_2.1.1/train" \
    "$ALFWORLD_DATA/json_2.1.1/valid_seen" \
    "$ALFWORLD_DATA/json_2.1.1/valid_unseen" \
    "$ALFWORLD_DATA/logic/alfred.pddl" \
    "$ALFWORLD_DATA/logic/alfred.twl2"; do
    if [[ ! -e "$required" ]]; then
        echo "ALFWorld text data is incomplete: $required" >&2
        exit 1
    fi
done

"${PIP[@]}" check > "$SETUP_DIR/pip-check.txt" 2>&1 || {
    # These two upstream wheels have incorrect platform tags. TextWorld is
    # separately exercised by the ALFWorld reset/step smoke check; decord is
    # not used by the text-only ALFWorld reproduction.
    if grep -Ev '^(decord 0\.6\.0|textworld 1\.7\.0) is not supported on this platform$' \
        "$SETUP_DIR/pip-check.txt" | grep -q .; then
        cat "$SETUP_DIR/pip-check.txt" >&2
        exit 1
    fi
    cat "$SETUP_DIR/pip-check.txt"
}
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
