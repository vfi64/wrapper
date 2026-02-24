# Changelog - Comm-SCI-Control-App

All notable changes are documented in this file.

## 1.0.1 - 2026-02-24

- Completed S9 panel/fallback helper modularization (asset loading, runtime bootstrap state, fallback race-guard decisions, panel HTML source selection) without changing `panel_action(...)` contracts.
- `src/Comm-SCI-Control-App.py` now delegates more panel/fallback decision logic to helper modules while keeping pywebview window operations in the monolith.
- Added dedicated unit tests for extracted S9 helper modules and kept targeted panel regression tests green.

## 1.0.0 - 2026-02-24

Public release milestone for the stabilized `20.0.3` baseline line (runtime/ruleset
strings remain on the `20.0.3` line in this release).

- Completed S7 UI asset decoupling hardening:
  - local `src/ui_assets/` loading with deterministic fallback
  - panel runtime self-test callback + automatic fallback to embedded panel
  - duplicate-panel race-condition fix after fallback replacement
- Completed S8 (Variant A) composition-root / desktop bootstrap isolation:
  - extracted desktop startup bootstrap helpers into `src/app_bootstrap.py`
  - preserved window creation order, close-event wiring, and `webview.start(...)` sequencing
  - added targeted bootstrap lifecycle regression tests (`tests/test_app_bootstrap.py`)
- Hardened manual test scenarios for professional release gating:
  - `qc_override_footer` uses a Gemini reference provider for QC/SCI contract checks
  - provider credit/limit failures (OpenRouter/HF) remain provider-path checks, not false contract failures
- Finalized reproducible local setup flow for release checks:
  - `scripts/setup_venv.sh` creates/updates `.venc`
  - local editable install via `pip install -e ".[local-dev]"` from `pyproject.toml`
  - `.gitignore` updated for `.venc/` and editable-install `*.egg-info/`
- Improved release-facing documentation:
  - generic (non-user-specific) paths in README / modularization docs
  - full top-level repository layout in `README.md` and `README.de.md`
  - short purpose map for key files/folders (including `src/app_bootstrap.py`, `src/ui_assets/`, `tests/test_app_bootstrap.py`)
  - explicit note about `Config/Comm-SCI-API-Keys.example.json` template file

## 20.0.3 - 2026-02-20

- Renamed runtime entrypoint to `Comm-SCI-Control-App.py`.
- Introduced `src/` layout:
  - runtime at `src/Comm-SCI-Control-App.py`
  - supporting modules in `src/` and `src/Module/`
- Moved tests into `tests/` and updated path handling.
- Switched default ruleset to `JSON/Comm-SCI-v20.0.3.json`.
- Added root launcher `Comm-SCI-Control-App.py` that executes the app from `src/`.
- Removed legacy versioned wrapper/test files from repository root.
- Updated packaging/test configuration in `pyproject.toml` for `src` + `tests`.
- Hardened ignore rules for secrets/log artifacts (`Config/Comm-SCI-API-Keys.json`, `Logs/**` content).
