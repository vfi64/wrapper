# Changelog - Comm-SCI-Control-App

All notable changes are documented in this file.

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
- Added Gemini dynamic model refresh/cache integration for panel model list updates.
- Added enforcement runtime settings:
  - `enforcement_enabled` feature flag
  - `enforcement_policy` (`audit_only`/`strict_warn`/`strict_block`)
  - `enforcement_blocked_severities` default list
- Added deterministic local command `Comm Enforcement` for runtime status (no provider call).
