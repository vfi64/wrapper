# Comm-SCI-Control-App

Deterministic Python runtime for Comm-SCI governance workflows.

Current app version: **20.0.3**  
Default ruleset loaded on startup: **`JSON/Comm-SCI-v20.0.3.json`**

## Start

```bash
python3 Comm-SCI-Control-App.py
```

## Tests

```bash
pytest -q
```

## Repository layout

```text
.
├── Comm-SCI-Control-App.py        # root launcher
├── src/
│   ├── Comm-SCI-Control-App.py    # main runtime
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

## Security and local data

- `Config/Comm-SCI-API-Keys.json` is local-only and ignored by Git.
- `Logs/**` content is ignored by Git (folder structure kept).

## Enforcement settings

- `enforcement_enabled`: feature flag for strict enforcement (`true` by default).
- `enforcement_policy`: `audit_only` | `strict_warn` | `strict_block` (default: `audit_only`).
- `enforcement_blocked_severities`: severities that can trigger blocking in strict mode (default: `["critical", "major"]`).
- Runtime status is available via local command `Comm Enforcement` (deterministic, no provider call).

## Dependencies

See `requirements.txt` (runtime) and `pyproject.toml` (packaging/test config).
