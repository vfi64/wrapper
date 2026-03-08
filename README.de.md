# Comm-SCI-Control-App

Deterministische Python-Runtime fuer Comm-SCI-Governance-Workflows.

Aktuelle App-Version: **1.0.0**  
Standardmaessig geladenes Regelwerk beim Start: **`JSON/Comm-SCI-v20.2.1.json`**

## Positionierung

**Nicht mehr Autonomie um jeden Preis, sondern mehr Kontrolle pro Antwort.**

**Nicht bloss sprachliche Plausibilitaet, sondern sichtbare Einordnung, Fehlbarkeit und Pruefbarkeit.**

Der Wrapper existiert, weil ein Regelwerk ohne technische Ausfuehrung leicht nur Wunsch bleibt.

## Warum dieser Wrapper existiert

Comm-SCI-Control definiert normative Governance-Regeln. Die Wrapper-Linie fokussiert den operativen Vollzug und Wiederholbarkeit: Command-Contracts, SCI-State-Handling, QC/Verification-Vertraege und panel-auditierbares Runtime-Verhalten.

## Praktische Orientierung

- JSON-only reicht oft fuer konzeptionelle Tests und schnelle Exploration.
- Wrapper-Runtime ist klar im Vorteil bei reproduzierbaren Laeufen, expliziten Kontrollpfaden, Diagnostik und auditierbaren Vergleichen.

Vertiefungsseiten:

- [`docs/why-wrapper.de.md`](docs/why-wrapper.de.md)
- [`docs/why-wrapper.en.md`](docs/why-wrapper.en.md)
- [`docs/runtime-use-cases.de.md`](docs/runtime-use-cases.de.md)
- [`docs/runtime-use-cases.en.md`](docs/runtime-use-cases.en.md)
- Webseite: [`docs/why-wrapper.de.html`](docs/why-wrapper.de.html)
- Webseite: [`docs/runtime-scenarios.de.html`](docs/runtime-scenarios.de.html)
- Webseite: [`docs/limits-wrapper.de.html`](docs/limits-wrapper.de.html)

## Zweck dieses Repos

Dieses Repository enthaelt die Wrapper-Entwicklungslinie der Python-Runtime, die Comm-SCI-Governance deterministisch ausfuehrt.
Es ist die Implementierungsseite zum oeffentlichen Regelwerk.

- Oeffentliches Regelwerk: [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)
- Wrapper/Runtime-Repository: [vfi64/wrapper](https://github.com/vfi64/wrapper)
- Oeffentliche Wrapper-Webseite: [vfi64.github.io/wrapper](https://vfi64.github.io/wrapper/)
- Oeffentliche Regelwerk-Webseite: [vfi64.github.io/Comm-SCI-Control](https://vfi64.github.io/Comm-SCI-Control/)

## Dokumentations-Hub

- Webseite (EN): [`docs/index.html`](docs/index.html)
- Webseite (DE): [`docs/index.de.html`](docs/index.de.html)
- Why Wrapper (EN): [`docs/why-wrapper.html`](docs/why-wrapper.html)
- Why Wrapper (DE): [`docs/why-wrapper.de.html`](docs/why-wrapper.de.html)
- Runtime Scenarios (EN): [`docs/runtime-scenarios.html`](docs/runtime-scenarios.html)
- Runtime Scenarios (DE): [`docs/runtime-scenarios.de.html`](docs/runtime-scenarios.de.html)
- Wrapper-Grenzen (EN): [`docs/limits-wrapper.html`](docs/limits-wrapper.html)
- Wrapper-Grenzen (DE): [`docs/limits-wrapper.de.html`](docs/limits-wrapper.de.html)
- Glossar (EN): [`docs/glossary.html`](docs/glossary.html)
- Glossar (DE): [`docs/glossar.de.html`](docs/glossar.de.html)
- Installation Einsteiger (EN): [`docs/install-beginner.html`](docs/install-beginner.html)
- Installation Einsteiger (DE): [`docs/install-beginner.de.html`](docs/install-beginner.de.html)
- Installation Profis (EN): [`docs/install-pro.html`](docs/install-pro.html)
- Installation Profis (DE): [`docs/install-pro.de.html`](docs/install-pro.de.html)
- Handbook (EN): [`docs/HANDBOOK.md`](docs/HANDBOOK.md)
- Handbuch (DE): [`docs/HANDBOOK.de.md`](docs/HANDBOOK.de.md)
- Installations-/Sync-Runbook (EN): [`docs/INSTALL_SYNC.md`](docs/INSTALL_SYNC.md)
- Installations-/Sync-Runbook (DE): [`docs/INSTALL_SYNC.de.md`](docs/INSTALL_SYNC.de.md)
- Architektur-Rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Modularisierungs-Roadmap: [`MODULARIZATION.md`](MODULARIZATION.md)
- Proposal-Backlog: [`docs/proposals/README.md`](docs/proposals/README.md)

## DOI (aktuelle Zenodo-Records)

- Wrapper (Concept DOI / alle Versionen): [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Wrapper (Versions-DOI / getaggter Wrapper-Release): [10.5281/zenodo.18759479](https://doi.org/10.5281/zenodo.18759479)
- Regelwerk (Concept DOI / alle Versionen): [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Regelwerk (Versions-DOI / aktuell gepflegter Zenodo-Release): [10.5281/zenodo.18154098](https://doi.org/10.5281/zenodo.18154098)

## Kommando-Kompatibilitaet (v20.2.x)

- `Comm Anchor on` / `Comm Anchor off` sind die kanonischen Toggle-Kommandos.
- `Anchor auto on` / `Anchor auto off` sind deprecated und sollten nicht mehr verwendet werden.
- `Control on` / `Control off` sind **keine** gueltigen Command-Tokens im v20.2.x-Command-Model.
- `Color on` / `Color off` sind gueltige Command-Tokens (`commands.color_control`).
- `Comm Help` wird aus dem geladenen JSON gerendert und zeigt den aktuell gueltigen Kommandosatz.

## DOI-Linkstrategie

- Fuer einen stabilen Langzeit-Link nutze den **Concept DOI** (Wrapper + Regelwerk).
- Einen **Versions-DOI** nutze nur dann, wenn exakt ein archiviertes Release zitiert werden soll.
- Repository-Startseite (lesbar): [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)

## Schnellstart (empfohlen, reproduzierbar)

```bash
cd /pfad/zum/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

Das Setup-Skript erstellt (oder aktualisiert) `.venc` und installiert die lokale
Entwicklungsumgebung aus `pyproject.toml` via `pip install -e ".[local-dev]"`.

## Professionelle Installations- und lokale Sync-Routine

Fuer robusten lokalen Betrieb und kontrollierte Uebergabe an den lokalen Klon von `vfi64/wrapper`:

1. Umgebung aufsetzen (automatische Python-Kompatibilitaetspruefung aus `pyproject.toml`):

```bash
bash scripts/setup_venv.sh
```

Optionale Checks:

```bash
bash scripts/setup_venv.sh --dry-run
bash scripts/setup_venv.sh --python python3.12 --venv .venc --extras local-dev
```

2. Dry-Run-Sync vom Quell-Repo in den lokalen `vfi64/wrapper`-Klon:

```bash
bash scripts/sync_to_wrapper_local.sh --target /pfad/zu/wrapper
```

3. Echter Sync (weiterhin nur lokal, kein Remote-Push):

```bash
bash scripts/sync_to_wrapper_local.sh --target /pfad/zu/wrapper --apply --validate
```

Sicherheitsdefaults:
- prueft, ob das Ziel-Remote auf `vfi64/wrapper` zeigt
- bricht bei dirty target repo ab (ausser mit `--allow-dirty-target`)
- kopiert nur getrackte Dateien und ueberspringt lokale Secrets/Logs

## Start (manuell)

```bash
cd /pfad/zum/repo
python3.14 -m venv .venc   # alternativ: python3 -m venv .venc
source .venc/bin/activate
python -m pip install -U pip
python -m pip install -e ".[local-dev]"
python Comm-SCI-Control-App.py
```

## Tests

```bash
cd /pfad/zum/repo
source .venc/bin/activate
python -m pytest -q
```

Gezielter schneller Testlauf (S8-/Panel-Bootstrap):

```bash
python -m pytest -q tests/test_app_bootstrap.py
python -m pytest -q tests/test_app.py -k "panel_asset_static_selftest or panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections or panel_action_accepts_panel_bootstrap_selftest_callback or on_panel_closed_ignores_stale_close_after_fallback_recreate"
```

## Repository-Struktur (vollstaendige Top-Level-Liste, versionierte Eintraege)

```text
.
├── .github/
├── .gitignore
├── ARCHITECTURE.md
├── CHANGELOG.md
├── CITATION.cff
├── Comm-SCI-Control-App.py
├── Config/
├── JSON/
├── LICENSE
├── Logs/
├── MODULARIZATION.md
├── README.de.md
├── README.md
├── docs/
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── scripts/
├── src/
└── tests/
```

Hinweis: Lokale Artefakte wie `.git/`, `.venc/`, `.pytest_cache/` oder `.DS_Store` sind absichtlich nicht in dieser Repo-Liste enthalten.

## Wichtige Dateien und Ordner (Kurzueberblick)

| Pfad | Zweck |
|---|---|
| `Comm-SCI-Control-App.py` | Root-Launcher (startet die Runtime aus `src/` mit korrektem `sys.path`). |
| `src/Comm-SCI-Control-App.py` | Haupt-Runtime / Monolith (UI, Zustand, Providerfluss, Audit, Fallbacks). |
| `src/app_bootstrap.py` | S8-Composition-Root-/Bootstrap-Helfer (Dependency-Guards, Fensterreihenfolge, `webview.start(...)`). |
| `src/ui_assets/` | Externe UI-Assets (Panel/Chat/Manual-Test-Monitor) mit S7-Fallback-Unterstuetzung. |
| `tests/test_app.py` | Groesste Regressions-/Vertragstests fuer Runtime- und Panel-Verhalten. |
| `tests/test_app_bootstrap.py` | Gezielte S8-Tests fuer Bootstrap-/Window-Lifecycle-Reihenfolge und Guards. |
| `scripts/setup_venv.sh` | Reproduzierbares lokales Setup (`.venc`, `pip install -e ".[local-dev]"`). |
| `scripts/run_local_tests.sh` | Lokaler Test-Runner mit venv-Erkennung/-Erzeugung. |
| `pyproject.toml` | Packaging-/Installationskonfiguration und pytest-Basisoptionen. |
| `requirements.txt` | Kompatibilitaets-/Referenzliste fuer Runtime-Abhaengigkeiten. |
| `MODULARIZATION.md` | Stufenplan S0-S8 und Akzeptanzkriterien. |
| `ARCHITECTURE.md` | Architekturprinzipien / Design-Rationale (Governance vs. Wrapper-Verantwortung). |
| `JSON/` | Regelwerke / Governance-JSONs. |
| `Config/` | Lokale Konfiguration und Provider-Keys (teils gitignored). |
| `Logs/` | Chat-/Audit-/ManualTest-Exports, Model-Caches (`Logs/Cache/`) und Laufzeitspuren (Inhalte gitignored). |
| `docs/` | Zusatzdokumentation (z. B. Checklisten, Release-Hinweise). |
| `docs/proposals/` | Proposal-Notizen fuer geplante Erweiterungen (draft/proposed/accepted/deferred/rejected). |

## Detailstruktur (Auszug fuer `src/` / `tests/`)

```text
src/
├── Comm-SCI-Control-App.py    # Haupt-Runtime
├── app_bootstrap.py           # S8 Bootstrap/Composition Root helper
├── controller.py
├── governance_service.py
├── intents.py
├── provider_service.py
├── render_service.py
├── state.py
├── storage_service.py
├── transitions.py
├── ui_controller.py
├── ui_panel_model.py
├── ui_assets/
│   ├── chat_template.html
│   ├── panel.html
│   └── ...
└── Module/
    ├── auditstream.py
    ├── compliance_scan.py
    ├── rendering_pipeline_v192.py
    └── rendering_utils.py

tests/
├── test_app.py
├── test_app_bootstrap.py
├── test_contracts_ui_state.py
├── test_controller_dispatch.py
├── test_governance_service.py
├── test_storage_service.py
└── ...
```

## Sicherheit und lokale Daten

- `Config/Comm-SCI-API-Keys.json` ist nur lokal und wird von Git ignoriert.
- Als Vorlage liegt `Config/Comm-SCI-API-Keys.example.json` ohne echte API-Keys im Repo.
- Inhalte unter `Logs/**` werden von Git ignoriert (Ordnerstruktur bleibt erhalten).
- API-Keys koennen im Panel (`Provider & LLM` -> `API-Key`) pro Provider im Klartext (`api_key_plain`) oder verschluesselt (`api_key_enc`) gespeichert werden.
- Fuer verschluesselte Keys erscheint beim Programmstart und beim Provider-Wechsel ein Passphrase-Dialog, sobald fuer den Ziel-Provider ein `api_key_enc` hinterlegt ist.
- Fuer unverschluesselte Keys erscheint kein Passphrase-Dialog; der Key wird direkt aus der lokalen Datei gelesen.
- Eingabe-History der Hauptzeile wird automatisch in `Logs/History/InputLineHistory.json` gespeichert und beim naechsten Start geladen.

## Abhaengigkeiten

- Primaerer lokaler Setup-Pfad: `pyproject.toml` (`pip install -e ".[local-dev]"`)
- Kompatibilitaets-/Referenzliste: `requirements.txt`
