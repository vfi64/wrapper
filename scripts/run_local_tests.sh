#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN=""
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
elif [[ -x ".venc/bin/python" ]]; then
  PY_BIN=".venc/bin/python"
else
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
  PY_BIN=".venv/bin/python"
fi

"$PY_BIN" -m pip install -U pip >/dev/null
"$PY_BIN" -m pip install -r requirements-dev.txt >/dev/null
"$PY_BIN" -m pytest -q tests
