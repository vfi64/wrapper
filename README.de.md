# Comm-SCI-Control-App

Deterministische Python-Runtime fuer Comm-SCI-Governance-Workflows.

Aktuelle App-Version: **1.0.0**  
Empfohlene Regelwerk-Linie: **`JSON/Comm-SCI-v20.2.x`** (aktuelle Datei: **`JSON/Comm-SCI-v20.2.0.json`**)

## DOI (stabile Referenzen)

- Runtime-App (Concept DOI / alle Versionen, stabiler Link): [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Regelwerk (Concept DOI / alle Versionen, stabiler Link): [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Empfehlung fuer stabile README-/Handbuch-/Webseiten-Links: Concept DOIs verwenden.
- Versions-DOIs nur in Release-Notizen oder fuer exakt gepinnte Reproduzierbarkeit eines einzelnen Releases nutzen.
- DOI-Links in dieser README zeigen auf Zenodo-Records (Archiv + Metadaten), nicht direkt auf GitHub-Repository-Seiten.
- GitHub-Repository-URL zur Code-Navigation: `https://github.com/vfi64/Comm-SCI-Control`

## Kommando-Status (Regelwerk v20.2.0)

- Kanonische Anchor-Kommandos: `Comm Anchor`, `Comm Anchor on`, `Comm Anchor off`.
- `Anchor auto on/off` ist in der v20.2.0-Runtime-Logik entfernt. Bitte nur `Comm Anchor on/off` verwenden.
- `anchor_auto` kann weiterhin als internes Status-Flag erscheinen, ist aber kein eigenes Nutzer-Kommando.
- `Control on/off` sind in v20.2.0 **keine** kanonischen Kommando-Tokens.
- `Color on` und `Color off` sind kanonische Kommando-Tokens in v20.2.0.

## Installationspfade (nach Zielgruppe)

### A) Einsteigerpfad (Laien, Schueler, Studierende, Fachanwender)

```bash
cd /pfad/zum/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

### B) Profi-Pfad (Entwickler, fortgeschrittene Nutzer, CI-nah)

```bash
cd /pfad/zum/repo
PYTHON_BIN=python3.14 VENV_DIR=.venc bash scripts/setup_venv.sh
source .venc/bin/activate
python -m pytest -q tests
python Comm-SCI-Control-App.py
```

Das Setup-Skript erstellt (oder aktualisiert) `.venc` und installiert Abhaengigkeiten
aus `pyproject.toml` via `pip install -e ".[local-dev]"`.
Detaillierte Installationsanleitungen:

- Deutsches Handbuch: [`docs/HANDBOOK.de.md`](docs/HANDBOOK.de.md)
- Englisches Handbuch: [`docs/HANDBOOK.md`](docs/HANDBOOK.md)
- Web-Doku (wenn GitHub Pages via `docs/` aktiv ist): [`docs/index.de.html`](docs/index.de.html) / [`docs/index.html`](docs/index.html)

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
python -m pytest -q tests
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
| `ARCHITECTURE.md` | Architekturprinzipien / Design-Rationale (Governance vs. Runtime-Verantwortung). |
| `JSON/` | Regelwerke / Governance-JSONs. |
| `Config/` | Lokale Konfiguration, Provider-Keys, Caches (teils gitignored). |
| `Logs/` | Chat-/Audit-/ManualTest-Exports und Laufzeitspuren (Inhalte gitignored). |
| `docs/` | Zusatzdokumentation (z. B. Checklisten, Release-Hinweise). |

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
- Provider-Key-Onboarding, Kostenhinweise und sichere Handhabung sind im [`docs/HANDBOOK.de.md`](docs/HANDBOOK.de.md) dokumentiert.

## Abhaengigkeiten

- Primaerer lokaler Setup-Pfad: `pyproject.toml` (`pip install -e ".[local-dev]"`)
- Kompatibilitaets-/Referenzliste: `requirements.txt`
