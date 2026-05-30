#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/score_ar_convnext_grandstaff.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export DIST_BACKEND="${DIST_BACKEND:-gloo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

cd "$ROOT_DIR"

PIP_BREAK_SYSTEM_PACKAGES=1 "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_score_ar.py --config "$CONFIG_PATH" "$@"
