#!/usr/bin/env bash
set -euo pipefail

# Configuration selected by internal tuning macro F1 in Phase 4.
bash scripts/wsl_tensorflow.sh train.py \
    --model efficientnet_b0 \
    --run-name efficientnet_b0_final \
    --run-type final \
    --epochs 15 \
    --patience 3 \
    --dropout 0.3 \
    --learning-rate 0.001 \
    --evaluate-held-out
