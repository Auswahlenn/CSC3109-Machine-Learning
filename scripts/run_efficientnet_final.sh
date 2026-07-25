#!/usr/bin/env bash
set -euo pipefail

# Final configuration used by the integrated five-model comparison.
bash scripts/wsl_tensorflow.sh train.py \
    --model efficientnet_b0 \
    --run-name efficientnet_b0 \
    --run-type final \
    --epochs 32 \
    --patience 20 \
    --batch-size 16 \
    --dropout 0.3 \
    --learning-rate 0.001 \
    --evaluate-held-out
