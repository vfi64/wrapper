# Installation und Sync (Privat -> lokaler Wrapper-Klon)

Dieses Dokument definiert die lokale Uebergaberoutine von diesem Quell-Repository in den lokalen Klon von `vfi64/wrapper`.

## Ziel

- Eine kompatible Python-Version automatisch verwenden.
- Alle benoetigten Pakete deterministisch installieren.
- Getrackte Wrapper-Dateien sicher in den lokalen `vfi64/wrapper`-Klon uebernehmen.
- Das Remote von `vfi64/wrapper` bis zur finalen Freigabe unberuehrt lassen.

## 1) Umgebung aufsetzen

```bash
bash scripts/setup_venv.sh
```

Das Skript:
- liest `requires-python` aus `pyproject.toml`,
- waehlt einen kompatiblen Interpreter,
- erstellt/verwendet `.venc`,
- installiert das Projekt mit Extras (default: `local-dev`),
- fuehrt `pip check` aus.

Nutzliche Flags:

```bash
bash scripts/setup_venv.sh --dry-run
bash scripts/setup_venv.sh --python python3.12
bash scripts/setup_venv.sh --extras none
```

## 2) Lokaler Sync in den Wrapper-Klon

Zuerst Dry-Run:

```bash
bash scripts/sync_to_wrapper_local.sh --target /pfad/zu/wrapper
```

Dann echter Sync:

```bash
bash scripts/sync_to_wrapper_local.sh --target /pfad/zu/wrapper --apply --validate
```

Sicherheitsmechanismen:
- prueft, ob Ziel ein Git-Repo mit Origin `vfi64/wrapper` ist,
- blockiert dirty target repos (ausser mit `--allow-dirty-target`),
- kopiert nur getrackte Dateien aus dem Quell-Repo,
- ueberspringt lokale Secrets/Caches/Log-Artefakte.

## 3) Release-Gating (spaeterer Schritt)

In dieser Phase kein Push auf `vfi64/wrapper`.
Erst private Verifikation abschliessen, dann lokal kopieren, dann Bezeichner/Links auf public-Kontext anpassen, dann veroeffentlichen.
