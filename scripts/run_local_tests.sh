#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venc}"
PY_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  echo "[run_local_tests] Kein venv gefunden unter $VENV_DIR. Starte Setup..."
  bash scripts/setup_venv.sh --venv "$VENV_DIR" --extras local-dev
fi

if [[ ! -x "$PY_BIN" ]]; then
  echo "ERROR: Test-Python fehlt: $PY_BIN" >&2
  exit 1
fi

"$PY_BIN" -m pytest -q tests "$@"
