#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venc}"
PY_REQ_DEFAULT="3.14"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.14 >/dev/null 2>&1; then
    PYTHON_BIN="python3.14"
  elif [[ -x "/opt/homebrew/opt/python@3.14/bin/python3.14" ]]; then
    PYTHON_BIN="/opt/homebrew/opt/python@3.14/bin/python3.14"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Kein passender Python-Interpreter gefunden (python3.14/python3)." >&2
    exit 1
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup_venv] Erzeuge venv: $VENV_DIR (via $PYTHON_BIN)"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup_venv] Verwende bestehendes venv: $VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv-Python nicht gefunden: $VENV_PY" >&2
  exit 1
fi

echo "[setup_venv] Aktualisiere pip/setuptools/wheel"
"$VENV_PY" -m pip install -U pip setuptools wheel

echo "[setup_venv] Installiere Projekt + lokale Dev-Extras aus pyproject.toml"
"$VENV_PY" -m pip install -e ".[local-dev]"

echo
echo "[setup_venv] Fertig."
echo "Naechste Schritte:"
echo "  source $VENV_DIR/bin/activate"
echo "  python Comm-SCI-Control-App.py"
echo
echo "Schnelltests:"
echo "  python -m pytest -q tests/test_app_bootstrap.py"
echo "  python -m pytest -q tests/test_app.py -k 'panel_asset_static_selftest or panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections or panel_action_accepts_panel_bootstrap_selftest_callback or on_panel_closed_ignores_stale_close_after_fallback_recreate'"
echo
echo "Hinweis: Bevorzugte Python-Version fuer lokale RC-Checks: ${PY_REQ_DEFAULT}+ (aktuell empfohlen: 3.14)."
