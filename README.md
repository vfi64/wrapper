# Comm-SCI-Control-App

Deterministic Python runtime for Comm-SCI governance workflows.

Current app version: **1.0.2**  
Default ruleset loaded on startup: **`JSON/Comm-SCI-v20.0.3.json`**

## DOI (current Zenodo records)

- Wrapper (concept DOI / all versions): [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Wrapper (version DOI / current public release `v1.0.0 (JSON-v20.0.3)`): [10.5281/zenodo.18759479](https://doi.org/10.5281/zenodo.18759479)
- Ruleset (concept DOI / all versions): [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Ruleset (version DOI / currently maintained Zenodo release): [10.5281/zenodo.18154098](https://doi.org/10.5281/zenodo.18154098)

## Quick Start (recommended, reproducible)

```bash
cd /path/to/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

The setup script creates (or refreshes) `.venc` and installs the local development
environment from `pyproject.toml` via `pip install -e ".[local-dev]"`.

## Start (manual)

```bash
cd /path/to/repo
python3.14 -m venv .venc   # alternatively: python3 -m venv .venc
source .venc/bin/activate
python -m pip install -U pip
python -m pip install -e ".[local-dev]"
python Comm-SCI-Control-App.py
```

## Tests

```bash
cd /path/to/repo
source .venc/bin/activate
python -m pytest -q
```

Targeted fast regression checks (S8 / panel bootstrap):

```bash
python -m pytest -q tests/test_app_bootstrap.py
python -m pytest -q tests/test_app.py -k "panel_asset_static_selftest or panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections or panel_action_accepts_panel_bootstrap_selftest_callback or on_panel_closed_ignores_stale_close_after_fallback_recreate"
```

## Repository layout (complete top-level list, versioned entries)

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

Note: local artifacts such as `.git/`, `.venc/`, `.pytest_cache/`, or `.DS_Store` are intentionally not included in this repository list.

## Key files and folders (short purpose map)

| Path | Purpose |
|---|---|
| `Comm-SCI-Control-App.py` | Root launcher (runs the runtime from `src/` with the expected `sys.path`). |
| `src/Comm-SCI-Control-App.py` | Main runtime / monolith (UI, state, provider flow, audit, fallbacks). |
| `src/app_bootstrap.py` | S8 composition-root/bootstrap helpers (dependency guards, window order, `webview.start(...)`). |
| `src/ui_assets/` | External UI assets (panel/chat/manual-test monitor) with S7 fallback support. |
| `tests/test_app.py` | Main regression/contract tests for runtime and panel behavior. |
| `tests/test_app_bootstrap.py` | Targeted S8 tests for bootstrap/window lifecycle sequencing and guards. |
| `scripts/setup_venv.sh` | Reproducible local setup (`.venc`, `pip install -e ".[local-dev]"`). |
| `scripts/run_local_tests.sh` | Local test runner with venv detection/creation. |
| `pyproject.toml` | Packaging/install config and base pytest settings. |
| `requirements.txt` | Compatibility/reference dependency list for runtime. |
| `MODULARIZATION.md` | S0-S8 roadmap and acceptance criteria. |
| `ARCHITECTURE.md` | Architecture rationale (governance vs wrapper responsibilities). |
| `JSON/` | Rulesets / governance JSON files. |
| `Config/` | Local configuration, provider keys, caches (partly gitignored). |
| `Logs/` | Chat/audit/manual-test exports and runtime traces (contents gitignored). |
| `docs/` | Additional documentation (checklists, release notes, etc.). |

## Detailed layout (excerpt for `src/` / `tests/`)

```text
src/
├── Comm-SCI-Control-App.py    # main runtime
├── app_bootstrap.py           # S8 bootstrap / composition root helper
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

## Security and local data

- `Config/Comm-SCI-API-Keys.json` is local-only and ignored by Git.
- A template file `Config/Comm-SCI-API-Keys.example.json` (without real API keys) is included in the repository.
- `Logs/**` content is ignored by Git (folder structure kept).

## Dependencies

- Primary local setup path: `pyproject.toml` (`pip install -e ".[local-dev]"`)
- Compatibility/reference list: `requirements.txt`
