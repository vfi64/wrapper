#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venc}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-local-dev}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_venv.sh [options]

Options:
  --python <bin>      Interpreter explizit setzen (z. B. python3.12)
  --venv <dir>        Ziel-venv (default: .venc)
  --extras <name>     Optional-Extras aus pyproject (default: local-dev, use 'none' fuer keine Extras)
  --dry-run           Nur Interpreter-/Versionspruefung, keine Installation
  -h, --help          Hilfe anzeigen
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "ERROR: --python erwartet ein Argument." >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || { echo "ERROR: --venv erwartet ein Argument." >&2; exit 2; }
      VENV_DIR="$2"
      shift 2
      ;;
    --extras)
      [[ $# -ge 2 ]] || { echo "ERROR: --extras erwartet ein Argument." >&2; exit 2; }
      INSTALL_EXTRAS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unbekannte Option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

MIN_PY="$(sed -n 's/^[[:space:]]*requires-python[[:space:]]*=[[:space:]]*">=\([0-9]\+\.[0-9]\+\).*"/\1/p' pyproject.toml | head -n1)"
if [[ -z "$MIN_PY" ]]; then
  MIN_PY="3.10"
fi

version_ge() {
  local a="$1"
  local b="$2"
  [[ "$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -n1)" == "$b" ]]
}

python_mm() {
  local bin="$1"
  "$bin" - <<'PY' 2>/dev/null
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

pick_python() {
  local selected=""
  local selected_mm=""

  if [[ -n "$PYTHON_BIN" ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
      echo "ERROR: Interpreter nicht gefunden: $PYTHON_BIN" >&2
      exit 1
    fi
    selected="$PYTHON_BIN"
    selected_mm="$(python_mm "$selected" || true)"
    if [[ -z "$selected_mm" ]]; then
      echo "ERROR: Python-Version konnte nicht gelesen werden: $selected" >&2
      exit 1
    fi
    if ! version_ge "$selected_mm" "$MIN_PY"; then
      echo "ERROR: Interpreter $selected ist $selected_mm, benoetigt wird >= $MIN_PY." >&2
      exit 1
    fi
    echo "$selected"
    return
  fi

  local candidates=(
    python3.14
    /opt/homebrew/opt/python@3.14/bin/python3.14
    python3.13
    /opt/homebrew/opt/python@3.13/bin/python3.13
    python3.12
    /opt/homebrew/opt/python@3.12/bin/python3.12
    python3.11
    /opt/homebrew/opt/python@3.11/bin/python3.11
    python3.10
    /opt/homebrew/opt/python@3.10/bin/python3.10
    python3
  )

  local c mm
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then
      mm="$(python_mm "$c" || true)"
      if [[ -n "$mm" ]] && version_ge "$mm" "$MIN_PY"; then
        selected="$c"
        selected_mm="$mm"
        break
      fi
    fi
  done

  if [[ -z "$selected" ]]; then
    echo "ERROR: Kein passender Python-Interpreter gefunden (benoetigt >= $MIN_PY)." >&2
    echo "Hinweis: Installiere zuerst eine kompatible Python-Version und starte erneut." >&2
    exit 1
  fi

  echo "$selected"
}

PYTHON_BIN="$(pick_python)"
PY_MM="$(python_mm "$PYTHON_BIN")"

echo "[setup_venv] Root: $ROOT_DIR"
echo "[setup_venv] requires-python (min): >= $MIN_PY"
echo "[setup_venv] Interpreter: $PYTHON_BIN ($PY_MM)"
echo "[setup_venv] venv: $VENV_DIR"
echo "[setup_venv] Extras: $INSTALL_EXTRAS"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[setup_venv] Dry-run erfolgreich. Keine Installation ausgefuehrt."
  exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup_venv] Erzeuge venv: $VENV_DIR"
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

if [[ "$INSTALL_EXTRAS" == "none" ]]; then
  INSTALL_TARGET="-e ."
else
  INSTALL_TARGET="-e .[$INSTALL_EXTRAS]"
fi

echo "[setup_venv] Installiere: $INSTALL_TARGET"
if ! "$VENV_PY" -m pip install $INSTALL_TARGET; then
  echo "ERROR: Paketinstallation fehlgeschlagen." >&2
  echo "Hinweis: Entweder fehlen Build-Abhaengigkeiten oder die Python-Version ist zu alt." >&2
  echo "Benoetigt laut pyproject: >= $MIN_PY." >&2
  exit 1
fi

echo "[setup_venv] Fuehre 'pip check' aus"
"$VENV_PY" -m pip check

echo
echo "[setup_venv] Fertig."
echo "Naechste Schritte:"
echo "  source $VENV_DIR/bin/activate"
echo "  python Comm-SCI-Control-App.py"
echo
echo "Schnelltests:"
echo "  python -m pytest -q tests/test_app_bootstrap.py"
echo "  python -m pytest -q tests/test_app.py -k 'panel_asset_static_selftest or panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections or panel_action_accepts_panel_bootstrap_selftest_callback or on_panel_closed_ignores_stale_close_after_fallback_recreate'"
