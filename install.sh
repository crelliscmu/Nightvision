#!/usr/bin/env bash
# Environment setup.
set -euo pipefail

# --- Require singularity ---
if ! command -v singularity &>/dev/null; then
    echo "ERROR: singularity not found. Load the singularity module first." >&2
    exit 1
fi

# --- Ensure uv is available ---
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Pull vLLM container if not present ---
pull_container() {
    local container_dir="${PROJECT:-.}/containers"
    local container_path="$container_dir/vllm.sif"
    mkdir -p "$container_dir"

    if [ ! -f "$container_path" ]; then
        echo "Pulling vLLM container to $container_path ..."
        singularity pull "$container_path" docker://vllm/vllm-openai:v0.17.0
    else
        echo "Container already exists at $container_path, skipping pull."
    fi

    echo "$container_path"
}

CONTAINER_PATH=$(pull_container)
uv venv --python 3.12 --seed
if [[ "$HOSTNAME" == "login-gpu" ]]; then
    echo "module load gcc cuda/12.6.1" >> .venv/bin/activate
    source .venv/bin/activate
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu126 --no-build-isolation
elif [[ "$HOSTNAME" == "delta.ncsa.illinois.edu" ]]; then
    source .venv/bin/activate
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132 --no-build-isolation
fi

uv pip install -e .

echo ""
echo "Installation complete."
echo "  Container: $CONTAINER_PATH"
echo "  Activate:  source .venv/bin/activate"
