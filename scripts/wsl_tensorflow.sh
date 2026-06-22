#!/usr/bin/env bash
set -euo pipefail

VENV="${CSC3109_VENV:-$HOME/.venvs/csc3109}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "TensorFlow environment not found at $VENV" >&2
    exit 1
fi

SITE_PACKAGES="$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_ROOT="$SITE_PACKAGES/nvidia"

if [[ ! -d "$NVIDIA_ROOT" ]]; then
    echo "NVIDIA pip libraries not found under $NVIDIA_ROOT" >&2
    exit 1
fi

NVIDIA_LIB_DIRS="$(find "$NVIDIA_ROOT" -type d -name lib -print | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$NVIDIA_ROOT/cuda_nvcc/bin:$PATH"

exec "$VENV/bin/python" "$@"
