#!/usr/bin/env bash
#
# Run with bash on a machine with one GPU:
#   bash experiments/nightvision_common_set/prompt_generation/run_prompt_search.sh <5|8> [stem ...] [-- extra main.py args...]
#
# Runs the low-probability prompt search (main.py) with the options used to
# build the published prompt sets, for the given prompt length in characters
# (5 or 8) and the given model stems, one model at a time. With no stems every
# model in the table below is searched. Raw results are written to
# results/<N>_char/<stem>_<N>tokens.json; models whose result file already
# exists are skipped, so an interrupted run can simply be restarted.
#
# Every search uses one GPU (--tensor-parallel-size 1), --target 20000 and
# --iterations 40. The per-model vLLM limits below (max model length, max
# concurrent sequences, batch size, GPU memory fraction) were chosen for an
# 80 GB H100; lower --max-num-seqs and --batch-size on smaller GPUs.
# Arguments after `--` are appended to every main.py call and override the
# table (e.g. `-- --max-num-seqs 256`).
#
# Environment overrides:
#   PROJECT_DIR   repository root           (default: derived from the script location)
#   RESULTS_DIR   parent of <N>_char/       (default: prompt_generation/results)
#   PYTHON        interpreter               (default: $PROJECT_DIR/.venv/bin/python)
set -euo pipefail

usage() { echo "usage: $0 <5|8> [stem ...] [-- extra main.py args...]" >&2; exit 2; }
[[ $# -ge 1 && ( "$1" == 5 || "$1" == 8 ) ]] || usage
CHARS="$1"; shift

STEMS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do STEMS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

# Options used per model. lenN / seqsN / batchN are --max-model-len,
# --max-num-seqs and --batch-size for N-character prompts; gpu_mem is
# --gpu-memory-utilization.
TABLE=$(cat <<'TABLE_END'
# stem                model                                      len5  len8  seqs5  seqs8  batch5  batch8  gpu_mem
granite_3.0_1b_a400m  ibm-granite/granite-3.0-1b-a400m-instruct  128   128   2048   2048   20000   20000   0.9
granite_3.0_2b        ibm-granite/granite-3.0-2b-instruct        128   128   2048   2048   20000   20000   0.9
granite_3.0_3b_a800m  ibm-granite/granite-3.0-3b-a800m-instruct  128   128   2048   2048   20000   20000   0.9
granite_3.0_8b        ibm-granite/granite-3.0-8b-instruct        128   128   1024   1024   20000   20000   0.9
granite_3.1_1b_a400m  ibm-granite/granite-3.1-1b-a400m-instruct  128   128   2048   2048   20000   20000   0.9
granite_3.1_2b        ibm-granite/granite-3.1-2b-instruct        128   128   2048   2048   20000   20000   0.9
granite_3.1_3b_a800m  ibm-granite/granite-3.1-3b-a800m-instruct  128   128   2048   2048   20000   20000   0.9
granite_3.1_8b        ibm-granite/granite-3.1-8b-instruct        128   128   1024   1024   20000   20000   0.9
granite_3.2_2b        ibm-granite/granite-3.2-2b-instruct        128   128   2048   2048   20000   20000   0.9
granite_3.2_8b        ibm-granite/granite-3.2-8b-instruct        128   128   1024   1024   20000   10000   0.9
granite_3.3_2b        ibm-granite/granite-3.3-2b-instruct        128   128   2048   2048   20000   20000   0.9
granite_3.3_8b        ibm-granite/granite-3.3-8b-instruct        128   128   1024   1024   20000   20000   0.9
granite_4.0_350m      ibm-granite/granite-4.0-350m               128   128   2048   2048   20000   10000   0.9
granite_4.0_micro     ibm-granite/granite-4.0-micro              128   128   2048   2048   20000   20000   0.9
llama_3.1_8b          meta-llama/Llama-3.1-8B-Instruct           64    80    1024   1024   20000   20000   0.9
llama_3.2_1b          meta-llama/Llama-3.2-1B-Instruct           64    80    2048   2048   20000   20000   0.9
llama_3.2_3b          meta-llama/Llama-3.2-3B-Instruct           64    80    1024   1024   20000   20000   0.9
olmo_2_1b             allenai/OLMo-2-0425-1B-Instruct            64    64    2048   2048   20000   20000   0.9
olmo_2_7b             allenai/OLMo-2-1124-7B-Instruct            64    64    1024   1024   20000   20000   0.9
olmo_3_7b             allenai/Olmo-3-7B-Instruct                 64    80    1024   1024   20000   20000   0.75
olmoe_7b              allenai/OLMoE-1B-7B-0924-Instruct          64    64    1024   1024   20000   20000   0.9
qwen2.5_0.5b          Qwen/Qwen2.5-0.5B-Instruct                 64    64    2048   2048   20000   20000   0.9
qwen2.5_1.5b          Qwen/Qwen2.5-1.5B-Instruct                 64    64    2048   2048   20000   20000   0.9
qwen2.5_3b            Qwen/Qwen2.5-3B-Instruct                   64    64    2048   2048   20000   20000   0.9
qwen2.5_7b            Qwen/Qwen2.5-7B-Instruct                   64    64    1024   1024   20000   20000   0.9
qwen3.5_0.8b          Qwen/Qwen3.5-0.8B                          64    64    2048   2048   20000   20000   0.9
qwen3.5_2b            Qwen/Qwen3.5-2B                            64    64    2048   2048   20000   20000   0.9
qwen3.5_4b            Qwen/Qwen3.5-4B                            64    64    512    512    20000   20000   0.9
qwen3.5_9b            Qwen/Qwen3.5-9B                            64    64    1024   1024   20000   20000   0.9
qwen3_0.6b            Qwen/Qwen3-0.6B                            64    64    2048   2048   20000   20000   0.9
qwen3_1.7b            Qwen/Qwen3-1.7B                            64    64    2048   2048   20000   20000   0.9
qwen3_30b             Qwen/Qwen3-30B-A3B                         64    64    1024   1024   20000   20000   0.9
qwen3_30b_instruct    Qwen/Qwen3-30B-A3B-Instruct-2507           64    64    1024   1024   20000   20000   0.9
qwen3_4b              Qwen/Qwen3-4B                              64    64    512    512    20000   20000   0.9
qwen3_8b              Qwen/Qwen3-8B                              64    64    1024   1024   20000   20000   0.9
smol2_1.7b            HuggingFaceTB/SmolLM2-1.7B-Instruct        64    64    2048   2048   20000   20000   0.9
smol2_135m            HuggingFaceTB/SmolLM2-135M-Instruct        64    64    2048   2048   20000   20000   0.9
smol2_360m            HuggingFaceTB/SmolLM2-360M-Instruct        64    64    2048   2048   20000   20000   0.9
smol3_3b              HuggingFaceTB/SmolLM3-3B                   512   512   1024   512    20000   20000   0.9
TABLE_END
)

declare -A MODEL LEN SEQS BATCH GPU_MEM
ALL_STEMS=()
while read -r stem model len5 len8 seqs5 seqs8 batch5 batch8 gpu_mem; do
    [[ -z "$stem" || "$stem" == \#* ]] && continue
    ALL_STEMS+=("$stem")
    MODEL[$stem]="$model"
    GPU_MEM[$stem]="$gpu_mem"
    if [[ "$CHARS" == 5 ]]; then
        LEN[$stem]="$len5"; SEQS[$stem]="$seqs5"; BATCH[$stem]="$batch5"
    else
        LEN[$stem]="$len8"; SEQS[$stem]="$seqs8"; BATCH[$stem]="$batch8"
    fi
done <<< "$TABLE"

[[ ${#STEMS[@]} -gt 0 ]] || STEMS=("${ALL_STEMS[@]}")
for stem in "${STEMS[@]}"; do
    [[ -n "${MODEL[$stem]:-}" ]] || { echo "unknown stem: $stem (known: ${ALL_STEMS[*]})" >&2; exit 2; }
done

cd "$PROJECT_DIR"
source .venv/bin/activate

OUT_DIR="$RESULTS_DIR/${CHARS}_char"
mkdir -p "$OUT_DIR"

failed=()
for stem in "${STEMS[@]}"; do
    out="$OUT_DIR/${stem}_${CHARS}tokens.json"
    if [[ -f "$out" ]]; then
        echo "--- skip $stem: $out exists"
        continue
    fi

    echo "=== $(date) $stem (${MODEL[$stem]}), ${CHARS} chars ==="
    if ! "$PYTHON" "$HERE/main.py" \
        --model "${MODEL[$stem]}" \
        --tensor-parallel-size 1 \
        --max-prompt-chars "$CHARS" \
        --max-model-len "${LEN[$stem]}" \
        --max-num-seqs "${SEQS[$stem]}" \
        --gpu-memory-utilization "${GPU_MEM[$stem]}" \
        --target 20000 \
        --batch-size "${BATCH[$stem]}" \
        --iterations 40 \
        --output "$out" \
        "${EXTRA[@]}"; then
        failed+=("$stem")
    fi
done

echo "=== $(date) done ==="
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
