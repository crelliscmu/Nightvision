#!/usr/bin/env bash
#
# Run with bash on a node with one GPU:
#   bash experiments/nightvision_common_set/collect_logits/collect_all.sh [extra collect.py args...]
#
# Loops over every prompt file under $PROMPTS_DIR and runs collect.py on it,
# one model at a time on a single GPU, using the matching HuggingFace model
# for each file. The default is the unified prompt directory shared by every
# nightvision_common_set experiment (../prompts/qwen3_8b.json -> $OUTPUT_DIR/qwen3_8b_logits.pt, with
# $OUTPUT_DIR defaulting to collect_logits/results, which is where
# sampling/run_sampling.py reads logits from).
# Prompt sets may also live in subdirectories of $PROMPTS_DIR (e.g.
# variant/qwen3_8b.json); each set gets its own output subdirectory and its
# name is appended to the output filename, so that example lands in
# $OUTPUT_DIR/variant/qwen3_8b_variant_logits.pt.
#
# Files whose logits already exist are skipped, so if the loop is interrupted
# just rerun it -- it picks up at the first model that has not finished.
#
# The stem -> model mapping mirrors _STEM_TO_HF in sampling/run_sampling.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"
EXPERIMENT_DIR="$PROJECT_DIR/experiments/nightvision_common_set"
SCRIPT_DIR="${SCRIPT_DIR:-$EXPERIMENT_DIR/collect_logits}"
PROMPTS_DIR="${PROMPTS_DIR:-$EXPERIMENT_DIR/prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== $(date) host=$(hostname) ==="
echo "prompts=$PROMPTS_DIR  out=$OUTPUT_DIR"
# Informational only; must not trip `set -e` when run somewhere GPU-less.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# prompt filename stem -> HuggingFace model
declare -A MODELS=(
    # granite
    [granite_3.0_1b_a400m]="ibm-granite/granite-3.0-1b-a400m-instruct"
    [granite_3.0_2b]="ibm-granite/granite-3.0-2b-instruct"
    [granite_3.0_3b_a800m]="ibm-granite/granite-3.0-3b-a800m-instruct"
    [granite_3.0_8b]="ibm-granite/granite-3.0-8b-instruct"
    [granite_3.1_1b_a400m]="ibm-granite/granite-3.1-1b-a400m-instruct"
    [granite_3.1_2b]="ibm-granite/granite-3.1-2b-instruct"
    [granite_3.1_3b_a800m]="ibm-granite/granite-3.1-3b-a800m-instruct"
    [granite_3.1_8b]="ibm-granite/granite-3.1-8b-instruct"
    [granite_3.2_2b]="ibm-granite/granite-3.2-2b-instruct"
    [granite_3.2_8b]="ibm-granite/granite-3.2-8b-instruct"
    [granite_3.3_2b]="ibm-granite/granite-3.3-2b-instruct"
    [granite_3.3_8b]="ibm-granite/granite-3.3-8b-instruct"
    [granite_4.0_350m]="ibm-granite/granite-4.0-350m"
    [granite_4.0_micro]="ibm-granite/granite-4.0-micro"

    # llama
    [llama_3.1_8b]="meta-llama/Llama-3.1-8B-Instruct"
    [llama_3.2_1b]="meta-llama/Llama-3.2-1B-Instruct"
    [llama_3.2_3b]="meta-llama/Llama-3.2-3B-Instruct"

    # olmo
    [olmo_2_1b]="allenai/OLMo-2-0425-1B-Instruct"
    [olmo_2_7b]="allenai/OLMo-2-1124-7B-Instruct"
    [olmo_3_7b]="allenai/Olmo-3-7B-Instruct"
    [olmoe_7b]="allenai/OLMoE-1B-7B-0924-Instruct"

    # qwen2.5
    [qwen2.5_0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
    [qwen2.5_1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
    [qwen2.5_3b]="Qwen/Qwen2.5-3B-Instruct"
    [qwen2.5_7b]="Qwen/Qwen2.5-7B-Instruct"

    # qwen3
    [qwen3_0.6b]="Qwen/Qwen3-0.6B"
    [qwen3_1.7b]="Qwen/Qwen3-1.7B"
    [qwen3_4b]="Qwen/Qwen3-4B"
    [qwen3_8b]="Qwen/Qwen3-8B"
    [qwen3_30b]="Qwen/Qwen3-30B-A3B"
    [qwen3_30b_instruct]="Qwen/Qwen3-30B-A3B-Instruct-2507"

    # qwen3.5
    [qwen3.5_0.8b]="Qwen/Qwen3.5-0.8B"
    [qwen3.5_2b]="Qwen/Qwen3.5-2B"
    [qwen3.5_4b]="Qwen/Qwen3.5-4B"
    [qwen3.5_9b]="Qwen/Qwen3.5-9B"

    # smol
    [smol2_135m]="HuggingFaceTB/SmolLM2-135M-Instruct"
    [smol2_360m]="HuggingFaceTB/SmolLM2-360M-Instruct"
    [smol2_1.7b]="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    [smol3_3b]="HuggingFaceTB/SmolLM3-3B"

)

mapfile -d '' -t prompt_files < <(find "$PROMPTS_DIR" -type f -name '*.json' -print0 | sort -z)

failed=()

if [[ ${#prompt_files[@]} -eq 0 ]]; then
    echo "No prompt .json files found under $PROMPTS_DIR" >&2
    exit 1
fi

for prompt_file in "${prompt_files[@]}"; do
    stem="$(basename "$prompt_file" .json)"
    model="${MODELS[$stem]:-}"
    if [[ -z "$model" ]]; then
        echo "WARNING: no model mapping for $stem, skipping" >&2
        continue
    fi

    # Prompt-set name = path of the file's directory relative to PROMPTS_DIR
    # (e.g. "variant"); empty when the file sits directly in PROMPTS_DIR.
    rel_dir="$(dirname "${prompt_file#"$PROMPTS_DIR"/}")"
    if [[ "$rel_dir" == "." ]]; then
        set_name=""
        out_dir="$OUTPUT_DIR"
        out_name="$stem"
    else
        set_name="${rel_dir//\//_}"
        out_dir="$OUTPUT_DIR/$rel_dir"
        out_name="${stem}_${set_name}"
    fi

    if [[ -f "$out_dir/${out_name}_logits.pt" ]]; then
        echo "=== $stem: $out_dir/${out_name}_logits.pt exists, skipping"
        continue
    fi

    echo "=== $(date) ${set_name:+$set_name/}$stem -> $model"
    # Deliberately not fatal: an OOM on a single model should not stop the
    # models still queued behind it.
    if ! "$PYTHON" "$SCRIPT_DIR/collect.py" \
        "$prompt_file" \
        "$out_name" \
        "$model" \
        --output-dir "$out_dir" \
        "$@"; then
        echo "ERROR: collect.py failed for $stem ($model)" >&2
        failed+=("${set_name:+$set_name/}$stem")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "=== $(date) done with ${#failed[@]} failure(s): ${failed[*]}" >&2
    exit 1
fi

echo "=== $(date) all done ==="
