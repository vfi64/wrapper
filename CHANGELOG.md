# Changelog - Comm-SCI-Control-App

All notable changes are documented in this file.

## 20.0.17-s15.2 - 2026-03-01

- Fixed recurring duplicate header leak in answer rendering:
  - removes isolated `Profile <KnownProfile>` scaffold lines without `:` in plain-text cleanup
  - removes `<p>Profile <KnownProfile></p>` scaffold blocks in HTML fallback cleanup.
- Hardened manual-test panel bridge reliability (pywebview fallback paths):
  - robust `ask` fallback detection via literal bridge error matching
  - deterministic `export` fallback via `panel_action('export')` route when direct API export is unavailable.
- Stabilized Self-Debunking formatting under malformed model markdown in HTML fallback:
  - normalizes leaked emphasis patterns and nested `<strong>` artifacts
  - preserves deterministic numbering of weakness entries.
- Added regression tests for:
  - plain/HTML profile-without-colon scaffold removal
  - manual-test ask/export fallbacks and regex escaping invariants
  - Self-Debunking markdown-leak cleanup in fallback numbering path.

## 20.0.16-s15 - 2026-03-01

- Added S15.1 deterministic scenario-harness foundation:
  - `src/manual_scenario_harness.py`
  - `scripts/run_scenario_harness.py`
  - `tests/test_manual_scenario_harness.py`
  - `docs/manual_scenario_protocol.md`
- Added S15.1.1 manual-panel `komplexttest` hardening:
  - export checkpoints before destructive `clear_chat` phases and at finalization
  - partial export snapshots on `STOPPED` and `ERROR` test termination
  - monitor-driven stop route (`manual_test_stop`) with safe report persistence
  - localized monitor labels and stop-feedback in DE/EN
- Added/extended test coverage for:
  - `komplexttest` export checkpoints and stop wiring
  - monitor stop action + controller/API route integration
  - manual-test panel option/route invariants.

## 20.0.15-s14 - 2026-02-28

- Added explicit language-policy modes for output validation:
  - `production` (default): enforce language violations as hard contract failures
  - `benchmark`: log language violations as soft audit signals without repair rewrite
- Hardened language validation scope to content-focused checks (reduces false positives from control/meta sections).
- Added `language_policy_mode` propagation across config, runtime state, panel snapshot, and audit/JSONL metadata.
- Exposed runtime setter route for language-policy mode via panel action bridge.
- Extended test coverage for:
  - benchmark vs production enforcement behavior,
  - content-scope language checks with scientific symbols/control-line exclusions,
  - panel/state normalization for language-policy mode.

## 20.0.12-s12 - 2026-02-26

- Completed S12 panel/QC UI-orchestration seam expansion (private-first, behavior-preserving):
  - extracted panel visibility/closing/rebuild decision plans into `src/panel_lifecycle_seam.py`
  - extracted QC override dialog window/apply/clear UI planning into `src/qc_override_window_seam.py`
  - extracted `QCBridge` and `PanelBridge` into standalone modules with monolith fail-open fallbacks
- Preserved pywebview call locality in the monolith and kept `panel_action(...)` contracts unchanged.
- Closed post-S12 user-reported regressions before wrapper sync:
  - QC override clear now injects a one-shot prompt reset hint to prevent stale override carry-over in the next model turn
  - Self-Debunking HTML normalization hardened for fragmented ordered-list output (`<li>/<p>` split + color/markdown debris)
  - mixed model/footer variants (`QC:` + `QC-Matrix:`) are deduplicated to one canonical QC footer
  - `CONTROL LAYER NOTE` repair banner is hidden for format-only Self-Debunking repairs (audit + repair pass remain active)
- S12 closure verification:
  - focused seam/regression tests green locally
  - private repo CI green before wrapper sync
  - wrapper sync completed with green CI afterwards

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
