# Changelog - Comm-SCI-Control-App

All notable changes are documented in this file.

## 1.0.4-s12 - 2026-02-26

- Synced wrapper runtime/tests/docs to the private-repo S12 completion state (panel/QC seam expansion + post-S12 hardening fixes).
- Included S12 seam modules in `src/` and their dedicated tests:
  - `panel_lifecycle_seam`, `qc_override_window_seam`, `qc_bridge`, `panel_bridge`
- Included post-S12 runtime fixes from private:
  - QC override clear one-shot prompt reset hint (prevents stale override carry-over)
  - Self-Debunking HTML fragment repair for malformed ordered-list output
  - mixed `QC:` + `QC-Matrix:` footer dedupe to one canonical footer
  - hidden `CONTROL LAYER NOTE` for format-only Self-Debunking repairs (repair/audit remain active)
- Updated `MODULARIZATION.md` with S12 completion status (functional seam expansion complete; net monolith reduction deferred to S13).
- Verified after sync via GitHub `tests` CI on wrapper `main`.

## 1.0.4 - 2026-02-25

- Synced wrapper runtime/tests to the current private-repo `main` state after S11 seam extraction closure (monolith reduction continues; no planned S12 changes included yet).
- Included the post-S11 `Comm Stop` hardening fix: with `Comm off`, governance formatting scaffolding (e.g. SCI Trace, Self-Debunking, QC footer) is suppressed while the safety core remains active.
- Normalized Self-Debunking ordered-list HTML rendering so split secondary labels (`Warum relevant`, `Pruefen/Widerlegen ...`) are merged back into the same `<li>` (consistent spacing in point 1 vs. point 2+).
- Updated/added Self-Debunking regression tests to match the canonical `<li> + <br>` rendering behavior.

## 1.0.3 - 2026-02-25

- Completed S10 monolith-thinning pass (no behavior change): further compacted panel fallback/report/wait/close cleanup paths in `src/Comm-SCI-Control-App.py` while preserving S7/S8/S9 fallback behavior and panel contracts.
- Fixed a manual-test false positive in `full_regression_light` by relaxing the Self-Debunking box detector for localized/inline-styled box variants (OpenRouter path).
- Hardened SCI Trace step recognition against common model label drift for the B-variant dialectic step (`Dialectic_6_Synthesis2` aliases such as `Dialectic_6_Syntheses_2` / `Dialectic_6_Synthesis_2`) and added regression tests.

## 1.0.2 - 2026-02-24

- Reduced noisy false-positive `CONTROL LAYER ALERTS (Python)` messages for CSC marker visibility by rendering the CSC transparency marker visibly in the header (policy-conform) instead of checking raw model text only.
- Preserved existing Verification Route Gate repair notes and other Control Layer signals.

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
