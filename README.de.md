# Comm-SCI-Control-App

Deterministische Python-Runtime fuer Comm-SCI-Governance-Workflows.

Aktuelle App-Version: **20.0.3**  
Standardmaessig geladenes Regelwerk beim Start: **`JSON/Comm-SCI-v20.0.3.json`**

## Start

```bash
python3 Comm-SCI-Control-App.py
```

## Tests

```bash
pytest -q
```

## Repository-Struktur

```text
.
├── Comm-SCI-Control-App.py        # Root-Launcher
├── src/
│   ├── Comm-SCI-Control-App.py    # Haupt-Runtime
│   ├── controller.py
│   ├── intents.py
│   ├── state.py
│   ├── transitions.py
│   ├── ui_panel_model.py
│   └── Module/
│       ├── __init__.py
│       ├── auditstream.py
│       ├── compliance_scan.py
│       ├── rendering_pipeline_v192.py
│       └── rendering_utils.py
├── tests/
│   ├── test_app.py
│   ├── test_contracts_ui_state.py
│   ├── test_controller_dispatch.py
│   ├── test_transitions_intents.py
│   └── test_ui_panel_model.py
├── JSON/
├── Config/
└── Logs/
```

## Sicherheit und lokale Daten

- `Config/Comm-SCI-API-Keys.json` ist nur lokal und wird von Git ignoriert.
- Inhalte unter `Logs/**` werden von Git ignoriert (Ordnerstruktur bleibt erhalten).

## Abhaengigkeiten

Siehe `requirements.txt` (Runtime) und `pyproject.toml` (Packaging/Test-Konfiguration).
