#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/score_unfolding.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

PIP_BREAK_SYSTEM_PACKAGES=1 "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ -f "$ROOT_DIR/data/grandstaff.tgz" ]]; then
  GRANDSTAFF_EXTRACT_ROOT="$ROOT_DIR/data/grandstaff_dataset/grandstaff"
  if [[ -z "$(find "$GRANDSTAFF_EXTRACT_ROOT" -name '*.bekrn' -print -quit 2>/dev/null)" ]]; then
    mkdir -p "$GRANDSTAFF_EXTRACT_ROOT"
    tar --warning=no-unknown-keyword -xf "$ROOT_DIR/data/grandstaff.tgz" -C "$GRANDSTAFF_EXTRACT_ROOT"
  fi
  touch "$ROOT_DIR/data/.grandstaff_extracted"
fi

"$PYTHON_BIN" scripts/prepare_image_cache.py --config "$CONFIG_PATH"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_score_unfolding.py --config "$CONFIG_PATH" "$@"
