#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/score_unfolding.yaml"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

python scripts/train_score_unfolding.py --config "$CONFIG_PATH" "$@"
