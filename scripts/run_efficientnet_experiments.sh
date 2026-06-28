#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-3}"
PYTHON_RUNNER=(bash scripts/wsl_tensorflow.sh)

run_experiment() {
    local run_name="$1"
    local dropout="$2"
    local learning_rate="$3"

    "${PYTHON_RUNNER[@]}" train.py \
        --model efficientnet_b0 \
        --run-name "$run_name" \
        --run-type experiment \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --dropout "$dropout" \
        --learning-rate "$learning_rate"
}

# One-factor-at-a-time comparison around the repository baseline.
run_experiment efficientnet_b0_baseline 0.3 0.001
run_experiment efficientnet_b0_lr3e4 0.3 0.0003
run_experiment efficientnet_b0_drop04 0.4 0.001
