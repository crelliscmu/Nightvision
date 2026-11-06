#!/usr/bin/env bash
# Download every model listed in model_configs.json into the HuggingFace hub
# cache (honours HF_HUB_CACHE / HF_HOME). Run from the repository root with the
# virtual environment active:
#   source .venv/bin/activate && bash download_models.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -c 'import json; print("\n".join(json.load(open("'"$REPO_DIR"'/model_configs.json"))))' |
while read -r model; do
    echo "=== $model"
    hf download "$model"
done
