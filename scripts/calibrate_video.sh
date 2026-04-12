#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
CACHE_DIR="$ROOT_DIR/.cache"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing local virtualenv at $PYTHON_BIN"
  echo "Run: uv sync --extra detection"
  exit 1
fi

mkdir -p "$MPL_CACHE_DIR" "$CACHE_DIR/fontconfig"
export XDG_CACHE_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/calibrate_video.py" "$@"
