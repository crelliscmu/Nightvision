#!/usr/bin/env bash
#
# Run with bash on a machine with one GPU:
#   bash experiments/nightvision_common_set/sampling/run_sampling.sh [extra run_sampling.py args...]
#
# Runs run_sampling.py over every cached-logits .pt under $LOGITS (default:
# collect_logits/results), once per entry of $LOGPROB_DTYPES. Every step gives
# each of the $MAX_PROMPTS prompts the same number of draws. Each precision
# writes its own output directory, $OUT_ROOT/<run name>_<dtype>.
#
# Environment overrides:
#   LOGITS          logits file or directory          (default: ../collect_logits/results)
#   PATTERN         glob inside LOGITS                (default: *.pt)
#   OUT_ROOT        parent of the per-dtype output dirs (default: ./results)
#   RUN_NAME        output dir prefix                 (default: full_run)
#   LOGPROB_DTYPES  space-separated list             (default: "bfloat16")
#   SWEEP_PASSES    biclique sweep passes             (default: 2)
#   MAX_PROMPTS     n_max, prompts in the set         (default: 20000)
#   BUDGET          total samples over all prompts    (default: 20000000000)
#   N_STEPS         equal budget slices               (default: 10)
#   EVAL_EVERY      evaluate every k-th step          (default: 1)
#   PROJECT_DIR     repository root                   (default: derived from the script location)
#   PYTHON          interpreter                       (default: $PROJECT_DIR/.venv/bin/python)
#   PRED_SMOOTH     true/false; false passes --no-pred-smooth (default: false)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"
EXPERIMENT_DIR="$PROJECT_DIR/experiments/nightvision_common_set"
SAMPLING_DIR="${SAMPLING_DIR:-$EXPERIMENT_DIR/sampling}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

LOGITS="${LOGITS:-$EXPERIMENT_DIR/collect_logits/results}"
PATTERN="${PATTERN:-*.pt}"
OUT_ROOT="${OUT_ROOT:-$SAMPLING_DIR/results}"
RUN_NAME="${RUN_NAME:-full_run}"
LOGPROB_DTYPES="${LOGPROB_DTYPES:-bfloat16}"
SWEEP_PASSES="${SWEEP_PASSES:-2}"
MAX_PROMPTS="${MAX_PROMPTS:-20000}"
BUDGET="${BUDGET:-20000000000}"
N_STEPS="${N_STEPS:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PRED_SMOOTH="${PRED_SMOOTH:-false}"

smooth_flag=()
[[ "$PRED_SMOOTH" == "true" ]] || smooth_flag=(--no-pred-smooth)

cd "$PROJECT_DIR"
source .venv/bin/activate

for logprob_dtype in $LOGPROB_DTYPES; do
    OUT_DIR="$OUT_ROOT/${RUN_NAME}_${logprob_dtype}"
    mkdir -p "$OUT_DIR"

    echo "=== $(date) host=$(hostname) ==="
    echo "pred_smooth=$PRED_SMOOTH  logprob_dtype=$logprob_dtype  sweep_passes=$SWEEP_PASSES  max_prompts=$MAX_PROMPTS  budget=$BUDGET  n_steps=$N_STEPS  eval_every=$EVAL_EVERY"
    echo "logits=$LOGITS  pattern=$PATTERN  out=$OUT_DIR"

    "$PYTHON" "$SAMPLING_DIR/run_sampling.py" \
        "$LOGITS" "$OUT_DIR" \
        --pattern "$PATTERN" \
        --sweep-passes "$SWEEP_PASSES" \
        --budget "$BUDGET" \
        --max-prompts "$MAX_PROMPTS" \
        --n-steps "$N_STEPS" \
        --eval-every "$EVAL_EVERY" \
        --temperature 2.0 \
        --seed 42 \
        --logprob-dtype "$logprob_dtype" \
        "${smooth_flag[@]}" \
        "$@"

    echo "=== $(date) done with $logprob_dtype ==="
done
