# Comm-SCI-Control-App

Deterministic Python runtime for Comm-SCI governance workflows.

Current app version line: **1.0.x (current: 1.0.11)**  
Latest wrapper release: **https://github.com/vfi64/wrapper/releases/latest**  
Default ruleset loaded on startup: **`JSON/Comm-SCI-v20.2.2.json`**

## Positioning

**Not more autonomy at any price, but more control per answer.**

**Not mere linguistic plausibility, but visible classification, fallibility, and inspectability.**

The wrapper exists because a ruleset without technical execution can remain an aspiration instead of stable runtime behavior.

## Why this wrapper exists

Comm-SCI-Control defines normative governance behavior. The wrapper line focuses on operational enforcement and repeatability: command contracts, SCI state handling, QC/verification contracts, and panel-auditable runtime behavior.

## Practical orientation

- When JSON-only is often enough: conceptual tests, quick exploratory usage.
- When wrapper runtime is clearly preferable: reproducible runs, explicit control paths, diagnostics, and audit-oriented comparison.

See detailed pages:

- [`docs/why-wrapper.en.md`](docs/why-wrapper.en.md)
- [`docs/why-wrapper.de.md`](docs/why-wrapper.de.md)
- [`docs/runtime-use-cases.en.md`](docs/runtime-use-cases.en.md)
- [`docs/runtime-use-cases.de.md`](docs/runtime-use-cases.de.md)
- Website: [`docs/why-wrapper.html`](docs/why-wrapper.html)
- Website: [`docs/runtime-scenarios.html`](docs/runtime-scenarios.html)
- Website: [`docs/limits-wrapper.html`](docs/limits-wrapper.html)

## What This Repository Is

This repository contains the development line of the Python wrapper/runtime that executes Comm-SCI governance deterministically.
It is the implementation counterpart to the public ruleset repository.

- Public ruleset reference: [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)
- Wrapper/runtime repository: [vfi64/wrapper](https://github.com/vfi64/wrapper)
- Wrapper project website (public): [vfi64.github.io/wrapper](https://vfi64.github.io/wrapper/)
- Ruleset website (public): [vfi64.github.io/Comm-SCI-Control](https://vfi64.github.io/Comm-SCI-Control/)

## Documentation Hub

- Website (EN): [`docs/index.html`](docs/index.html)
- Website (DE): [`docs/index.de.html`](docs/index.de.html)
- Why Wrapper (EN): [`docs/why-wrapper.html`](docs/why-wrapper.html)
- Why Wrapper (DE): [`docs/why-wrapper.de.html`](docs/why-wrapper.de.html)
- Runtime scenarios (EN): [`docs/runtime-scenarios.html`](docs/runtime-scenarios.html)
- Runtime scenarios (DE): [`docs/runtime-scenarios.de.html`](docs/runtime-scenarios.de.html)
- Wrapper limits (EN): [`docs/limits-wrapper.html`](docs/limits-wrapper.html)
- Wrapper limits (DE): [`docs/limits-wrapper.de.html`](docs/limits-wrapper.de.html)
- Glossary (EN): [`docs/glossary.html`](docs/glossary.html)
- Glossary (DE): [`docs/glossar.de.html`](docs/glossar.de.html)
- Beginner install (EN): [`docs/install-beginner.html`](docs/install-beginner.html)
- Beginner install (DE): [`docs/install-beginner.de.html`](docs/install-beginner.de.html)
- Professional install (EN): [`docs/install-pro.html`](docs/install-pro.html)
- Professional install (DE): [`docs/install-pro.de.html`](docs/install-pro.de.html)
- Handbook (EN): [`docs/HANDBOOK.md`](docs/HANDBOOK.md)
- Handbook (DE): [`docs/HANDBOOK.de.md`](docs/HANDBOOK.de.md)
- Install + sync runbook (EN): [`docs/INSTALL_SYNC.md`](docs/INSTALL_SYNC.md)
- Install + sync runbook (DE): [`docs/INSTALL_SYNC.de.md`](docs/INSTALL_SYNC.de.md)
- Architecture rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Modularization roadmap: [`MODULARIZATION.md`](MODULARIZATION.md)
- Proposal backlog: [`docs/proposals/README.md`](docs/proposals/README.md)

## DOI (current Zenodo records)

- Wrapper (concept DOI / all versions): [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Wrapper (version DOI / tagged wrapper release): [10.5281/zenodo.18759479](https://doi.org/10.5281/zenodo.18759479)
- Ruleset (concept DOI / all versions): [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Ruleset (version DOI / currently maintained Zenodo release): [10.5281/zenodo.18154098](https://doi.org/10.5281/zenodo.18154098)

## Command Compatibility (v20.2.x)

- `Comm Anchor on` / `Comm Anchor off` are the canonical toggles.
- `Anchor auto on` / `Anchor auto off` are deprecated and should not be used.
- `Control on` / `Control off` are **not** valid command tokens in the v20.2.x command model.
- `Color on` / `Color off` are valid command tokens (`commands.color_control`).
- `Comm Help` is rendered from the loaded JSON and shows the currently valid command set.

## DOI Linking Strategy

- For a stable long-term link, use the **concept DOI** (wrapper + ruleset).
- Use a **version DOI** only when you need to cite one exact archived release.
- Repository landing page (human-readable): [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)

## Quick Start (recommended, reproducible)

```bash
cd /path/to/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

The setup script creates (or refreshes) `.venc` and installs the local development
environment from `pyproject.toml` via `pip install -e ".[local-dev]"`.

## Professional Install + Local Sync Routine

For robust local operation and controlled handover to the local clone of `vfi64/wrapper`:

1. Environment bootstrap (auto-detects compatible Python from `pyproject.toml`):

```bash
bash scripts/setup_venv.sh
```

Optional checks:

```bash
bash scripts/setup_venv.sh --dry-run
bash scripts/setup_venv.sh --python python3.12 --venv .venc --extras local-dev
```

2. Dry-run sync from source repo to local `vfi64/wrapper` clone:

```bash
bash scripts/sync_to_wrapper_local.sh --target /path/to/wrapper
```

3. Real sync (still local only, no remote push):

```bash
bash scripts/sync_to_wrapper_local.sh --target /path/to/wrapper --apply --validate
```

Safety defaults:
- verifies target remote points to `vfi64/wrapper`
- refuses dirty target repos unless `--allow-dirty-target` is set
- copies only tracked files and skips local secrets/log artifacts

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
| `Config/` | Local configuration and provider keys (partly gitignored). |
| `Logs/` | Chat/audit/manual-test exports, model caches (`Logs/Cache/`), and runtime traces (contents gitignored). |
| `docs/` | Additional documentation (checklists, release notes, etc.). |
| `docs/proposals/` | Proposal notes for planned extensions (draft/proposed/accepted/deferred/rejected). |

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
- API keys can be managed per provider in the panel (`Provider & LLM` -> `API-Key`) and stored either as plaintext (`api_key_plain`) or encrypted (`api_key_enc`).
- For encrypted keys, the runtime asks for a passphrase on startup and on provider switch when the target provider uses `api_key_enc`.
- For plaintext keys, no passphrase dialog is shown and the key is read directly from the local config file.
- Main input-line history is auto-saved to `Logs/History/InputLineHistory.json` on shutdown and loaded again on startup.

## Dependencies

- Primary local setup path: `pyproject.toml` (`pip install -e ".[local-dev]"`)
- Compatibility/reference list: `requirements.txt`
