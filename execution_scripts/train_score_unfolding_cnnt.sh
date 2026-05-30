#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/score_unfolding_cnnt.yaml"

exec "$ROOT_DIR/execution_scripts/train_score_unfolding.sh" "$CONFIG_PATH" "$@"
