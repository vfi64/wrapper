#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_DIR=""
APPLY=0
ALLOW_DIRTY_TARGET=0
RUN_VALIDATE=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/sync_to_wrapper_local.sh --target /path/to/wrapper [options]

Options:
  --target <dir>          Lokaler Pfad zum Ziel-Repo (vfi64/wrapper)
  --apply                 Tatsächlich kopieren (default ist Dry-Run)
  --allow-dirty-target    Dirty target erlauben (default: abbrechen)
  --validate              Nach Sync im Ziel: bash scripts/setup_venv.sh --dry-run
  -h, --help              Hilfe anzeigen

Verhalten:
  - Kopiert nur getrackte Dateien aus diesem privaten Repo.
  - Schreibt NICHT in entfernte Repositories, nur lokal.
  - Verifiziert, dass das Ziel-Remote auf vfi64/wrapper zeigt.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "ERROR: --target erwartet ein Argument." >&2; exit 2; }
      TARGET_DIR="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --allow-dirty-target)
      ALLOW_DIRTY_TARGET=1
      shift
      ;;
    --validate)
      RUN_VALIDATE=1
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

if [[ -z "$TARGET_DIR" ]]; then
  echo "ERROR: --target ist erforderlich." >&2
  usage
  exit 2
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: Zielverzeichnis existiert nicht: $TARGET_DIR" >&2
  exit 1
fi

if ! git -C "$TARGET_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Ziel ist kein Git-Repository: $TARGET_DIR" >&2
  exit 1
fi

TARGET_REMOTE="$(git -C "$TARGET_DIR" remote get-url origin 2>/dev/null || true)"
if [[ "$TARGET_REMOTE" != *"vfi64/wrapper"* ]]; then
  echo "ERROR: Ziel-Remote ist nicht vfi64/wrapper: $TARGET_REMOTE" >&2
  echo "Abbruch aus Sicherheitsgruenden." >&2
  exit 1
fi

if [[ "$ALLOW_DIRTY_TARGET" -eq 0 ]]; then
  if [[ -n "$(git -C "$TARGET_DIR" status --porcelain)" ]]; then
    echo "ERROR: Ziel-Repo hat lokale Aenderungen. Bitte commit/stash oder --allow-dirty-target verwenden." >&2
    exit 1
  fi
fi

should_skip() {
  local p="$1"
  case "$p" in
    .git/*|.git) return 0 ;;
    .venv/*|.venv|.venc/*|.venc) return 0 ;;
    .pytest_cache/*|.pytest_cache) return 0 ;;
    __pycache__/*|__pycache__) return 0 ;;
    *.pyc) return 0 ;;
    .DS_Store|*/.DS_Store) return 0 ;;
    Config/Comm-SCI-API-Keys.json) return 0 ;;
    Config/*models_cache.json) return 0 ;;
    Logs/*) return 0 ;;
    docs/proposals/*) return 0 ;;
  esac
  return 1
}

FILE_LIST=()
while IFS= read -r path; do
  if should_skip "$path"; then
    continue
  fi
  FILE_LIST+=("$path")
done < <(git ls-files)

if [[ "${#FILE_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: Keine zu synchronisierenden Dateien gefunden." >&2
  exit 1
fi

echo "[sync] Quelle: $ROOT_DIR"
echo "[sync] Ziel:   $TARGET_DIR"
echo "[sync] Dateien: ${#FILE_LIST[@]}"
echo "[sync] Modus:   $([[ "$APPLY" -eq 1 ]] && echo APPLY || echo DRY-RUN)"

if [[ "$APPLY" -eq 0 ]]; then
  for p in "${FILE_LIST[@]}"; do
    echo "[dry-run] $p"
  done
  echo
  echo "Kein Kopiervorgang ausgefuehrt. Fuer echten Sync: --apply"
  exit 0
fi

for p in "${FILE_LIST[@]}"; do
  src="$ROOT_DIR/$p"
  dst="$TARGET_DIR/$p"
  mkdir -p "$(dirname "$dst")"
  cp -p "$src" "$dst"
done

echo "[sync] Kopieren abgeschlossen."

if [[ "$RUN_VALIDATE" -eq 1 ]]; then
  echo "[sync] Validierung im Ziel: bash scripts/setup_venv.sh --dry-run"
  (cd "$TARGET_DIR" && bash scripts/setup_venv.sh --dry-run)
fi

echo "[sync] Fertig."
