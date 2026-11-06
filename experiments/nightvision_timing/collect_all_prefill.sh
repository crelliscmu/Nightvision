#!/usr/bin/env bash
#SBATCH --job-name=collect-all
#SBATCH --gres=gpu:h100-80:1 
#SBATCH --partition=GPU-shared
#SBATCH --time=12:00:00
#SBATCH --output=logs/collect_all_prefill_%A_%a.out
#SBATCH --error=logs/collect_all_prefill_%A_%a.out
set -euo pipefail
#
# Run with bash on a node with one GPU:
#   bash experiments/nightvision_timing/collect_all_prefill.sh [extra collect_prefill.py args...]

# The H200 measurements used the same arguments.
# Locate this directory. Under sbatch the script runs from a spool copy, so fall
# back to the submit directory (repo root or this directory).
HERE=""
for dir in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
           "${SLURM_SUBMIT_DIR:-.}" \
           "${SLURM_SUBMIT_DIR:-.}/experiments/nightvision_timing"; do
    if [[ -f "$dir/collect_prefill.py" ]]; then
        HERE="$(cd "$dir" && pwd)"
        break
    fi
done
[[ -n "$HERE" ]] || { echo "ERROR: collect_prefill.py not found; submit from the repo root or experiments/nightvision_timing" >&2; exit 1; }
PROJECT_DIR="$(cd "$HERE/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/timing_results_prefill}"
VENV="${VENV:-$PROJECT_DIR/.venv}"

cd "$PROJECT_DIR"

[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv not found at $VENV" >&2; exit 1; }

source "$VENV/bin/activate"
PYTHON="$(command -v python)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== $(date) host=$(hostname) out=$OUTPUT_DIR === project_dir=$PROJECT_DIR === python=$PYTHON === hf_hub_cache=${HF_HUB_CACHE:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

models_list="$("$PYTHON" -c '
import json, sys
for name, cfg in json.load(open(sys.argv[1])).items():
    print(name)
' "$HERE/model_configs.json")"
mapfile -t models <<< "$models_list"
[[ ${#models[@]} -gt 0 && -n "${models[0]}" ]] || { echo "ERROR: no Dense models found" >&2; exit 1; }

failed=()
for model in "${models[@]}"; do
    out="$OUTPUT_DIR/${model//\//_}_timing.json"
    if [[ -f "$out" ]]; then
        echo "=== $model: $out exists, skipping"
        continue
    fi
    echo "=== $(date) $model"
    # OLMo-2 barely fits let needs larger gpu-mem
    extra_args=()
    if [[ "$model" == *"allenai/OLMo-2-1124-7B-Instruct"* ]]; then
        extra_args+=(--gpu-mem 0.95)
    fi
    # Not fatal: one failing model should not stop the ones queued behind it.
    if ! "$PYTHON" "$HERE/collect_prefill.py" "$model" --output-dir "$OUTPUT_DIR" "${extra_args[@]}" "$@"; then
        echo "ERROR: collect_prefill.py failed for $model" >&2
        failed+=("$model")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "=== $(date) done with ${#failed[@]} failure(s): ${failed[*]}" >&2
    exit 1
fi
echo "=== $(date) all done ==="
