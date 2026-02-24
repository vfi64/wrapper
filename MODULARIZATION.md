# Wrapper Modularization Guide (Comm‑SCI‑Wrapper)

This document is the **source of truth** for stepwise, regression‑safe modularization of the local GUI wrapper (pywebview) that operationalizes the **Comm‑SCI‑Control** JSON ruleset.

## Objectives
- Split responsibilities into modules **without** breaking determinism, auditability, UI behavior, or test coverage.
- Keep every step **releasable** (tests green; no “big‑bang refactor”).

## Scope and terminology
- **Ruleset**: the external JSON governance spec (the core logic is *data*, not Python).
- **Wrapper**: the Python execution/control layer (UI + providers + logging + state machine).
- **Build artifacts**: files named like `Wrapper-<NNN>.py` and `Test-<NNN>.py`.  
  These are **iteration counters**, not architecture milestones.
- **Stages** (S0, S1, …): the modularization roadmap below.  
  Stages are **independent** of the `<NNN>` file numbers.

## Non‑negotiable constraints (must not regress)
- Deterministic behavior: no hidden state changes, no “smart” auto‑adjustments.
- Full auditability: every significant state transition is observable and logged.
- QC calculation and QC override semantics remain identical.
- SCI menu/format and command parsing remain identical.
- UI output must not change unless explicitly intended and tested.
- Provider/model switching is explicit and visible.
- Tests must not perform real network calls (use stubs/mocks).

## Repository conventions
- Keep governance JSON files under `JSON/`.
- Keep working from project root; run via `python3` from the current directory.
- Prefer small, test‑backed PRs.

---

## Roadmap (stages)

### S0 — Baseline hardening (no functional change)
**Goal:** make the current behavior reproducible and easy to verify.
- Document a “Golden Run” manual checklist (startup, panel lifecycle, provider/model switch, log export/load/fork, QC override apply/clear).
- Add/strengthen smoke tests (no network).
- Add a minimal observability hook: `log_event(kind, payload)` (or equivalent).
- Add invariants/guards (visible provider switch; stable defaults; UI payload shape).

#### S0 acceptance checklist
- [ ] `GOLDEN_RUN_STUFE0` exists and contains startup/panel/provider/export/QC-relevant steps.
- [ ] `Api.log_event(...)` is callable fail-safe and appends JSON-safe session events.
- [ ] Provider switch emits a visible event trail (`provider_switch`) in observability data.
- [ ] Smoke tests run locally without network calls (`pytest -q` from project root).

#### S0 test focus (current repo)
- `tests/test_stage0_baseline.py`
- Existing baseline guards in `tests/test_app.py` (startup defaults, UI header/title invariants, no-network smoke).

### S1 — Clear internal boundaries (still single file)
**Goal:** make later extraction safe by defining contracts first.
- Introduce explicit “sections” and minimal contracts (inputs/outputs).
- Reduce implicit globals where safe; pass a single `AppState`/`GovState` object.
- Add tests that lock the contracts (golden outputs and edge cases).

### S2 — Provider encapsulation
**Goal:** isolate LLM calls behind an interface.
- Introduce `ProviderClient` interface + registry + `NullClient`.
- Ensure explicit provider/model change events (no silent switching).
- Provider/model cache per provider, deterministic selection.

### S3 — Governance engine isolation
**Goal:** move policy/rules application into a dedicated service.
- Introduce `GovernanceService` (route/apply/wrap/postprocess/validate‑and‑repair/policy gate).
- Centralize reset rules (profile switch, clear chat, comm stop).
- Ensure **no UI logic** inside governance state.

#### S3 acceptance checklist
- [x] Raw output contract normalization is centralized via `GovernanceService`.
- [x] Reset rules for profile switch / clear chat / comm stop are centralized via `GovernanceService`.
- [x] Legacy governance command-state transitions (overlay/color/sci/comm/anchor/dynamic) are routed through `GovernanceService` (SCI recursion remains local by design).

### S4 — UI decoupling
**Goal:** make UI interactions a thin, testable layer.
- Introduce `UIController` with standardized actions/responses.
- Preserve UI behavior; change only through tests.

### S5 — Storage & logs as services
**Goal:** deterministic persistence and schema enforcement.
- Introduce `StorageService` for load/fork/export.
- Enforce schema‑versioned audit logs (e.g., “audit v2+”) in exports.

### S6 — Panel action routing split (UI action dispatcher extraction)
**Goal:** reduce `panel_action(...)` monolith size/risk by moving stable panel routes into `UIController`.
- Add a delegated `UIController` panel-action handler for stable panel routes.
- Keep routing behavior identical (same response schema, same fail-soft semantics).
- Remove duplicated wrapper branches only after tests/manual checks are green.

#### S6 acceptance checklist
- [x] `panel_action(...)` delegates a stable subset of routes to `UIController`.
- [x] Manual-test monitor and report routes are handled via `UIController`.
- [x] QC override panel routes are handled via `UIController`.
- [x] Provider/model/language/refresh panel routes are handled via `UIController`.
- [x] Chat log list/load/clear panel routes are handled via `UIController`.
- [x] Redundant delegated wrapper branches are removed without panel regression.

### S7 — UI / Panel assets decoupling (HTML/CSS/JS extraction)
**Goal:** reduce wrapper file size and make UI assets versionable/testable without changing runtime behavior.
- Extract large inline UI assets (chat/panel/manual-test monitor HTML/CSS/JS) into dedicated files under `src/ui_assets/` (or equivalent).
- Keep a deterministic loader/fallback in the wrapper (no dynamic remote loading).
- Preserve exact UI behavior and command bindings (`panel_action`, `ask`, monitor/report hooks).
- Add tests for asset loading/fallback and key invariants (required JS functions, panel controls present).

#### S7 acceptance checklist (planned)
- [x] Panel HTML/JS is loaded from local asset file(s) with deterministic fallback.
- [x] Chat HTML/JS/CSS is loaded from local asset file(s) with deterministic fallback.
- [ ] No change in `panel_action` command names or payload schemas.
- [ ] Existing panel/manual-test smoke checks remain green.
- [ ] Wrapper file size is materially reduced (documented delta).

### Optional later
- External templates (feature‑flagged)
- Multi‑file split into packages (only after stable interfaces)
- Additional stages only if required by new features

### S8 — Composition root / window bootstrap isolation (release-oriented)
**Goal:** make the launcher path smaller and safer without changing runtime behavior.
- Extract the `__main__` bootstrap sequence (dependency checks, window creation order, event binding, `webview.start(...)`) into a small composition/bootstrap module or function.
- Keep `Api` behavior, `panel_action(...)` names, payload schemas, and pywebview window timing semantics unchanged.
- Centralize window lifecycle orchestration for main/panel/QC windows (including pre-create order and close-event wiring) behind a narrow interface.
- Preserve deterministic local assets + fallback behavior introduced in S7 (including panel runtime self-test fallback).

**Out of scope (S8 must not do this)**
- No feature additions.
- No provider/rendering behavior changes.
- No broad package re-layout or multi-file split beyond bootstrap/composition extraction.
- No UI redesign or panel action contract changes.

#### S8 acceptance checklist (planned)
- [x] `if __name__ == '__main__'` block in `Comm-SCI-Control-App.py` is reduced to a thin bootstrap call (plus minimal guard logic).
- [x] Window creation order and behavior remain identical in manual smoke checks (main window, panel pre-create, QC window pre-create, close handling).
- [x] Panel runtime self-test + fallback path still works (including no duplicate panel after fallback replacement).
- [x] No change in `panel_action(...)` route names or response payload schemas.
- [x] Targeted regression tests cover bootstrap/window lifecycle seams introduced by S8.
- [x] Manual startup instructions use a reproducible `.venc` flow (documented for release).

Current S8 verification snapshot (2026-02-24):
- Automated: `tests/test_app_bootstrap.py` (`7 PASS`) for dependency guards, window creation order, close-event binding, and `webview.start(...)` composition-root sequencing.
- Automated: targeted S7 panel regression tests (`5 PASS`) for panel asset static self-test, runtime payload validation, bootstrap callback acceptance, and duplicate-panel race fix.
- Manual: panel/manual-test smoke runs passed (`qc_override_footer`, `full_regression_light`) after S8 bootstrap extraction; provider-credit warnings are treated as provider-path checks, not feature-contract failures.
- Manual: post-S8 pywebview runtime-fallback re-test (intentionally broken `panel_bootstrap_selftest` callback) passed: delayed fallback to embedded panel works, and no duplicate-panel regression on Panel toggle.

#### S8 execution notes (Variant A)
- Prefer extraction by *composition root* (new module/function) over moving core logic between existing services.
- Keep commits small and behavior-preserving; run manual pywebview smoke checks after each window-lifecycle change.
- Stop once acceptance criteria are met; defer deeper refactors to post-release hardening (Variant B).
- Reproducible local startup flow (current release candidate baseline):
  ```bash
  cd /path/to/repo
  source .venc/bin/activate
  python Comm-SCI-Control-App.py
  ```

### S9 — Panel fallback/helper modularization (post-release, behavior-preserving)
**Goal:** reduce panel/fallback-specific monolith logic by extracting pure helper modules while preserving pywebview runtime behavior.
- Extract panel asset/static-selftest helpers into a dedicated module.
- Extract panel bootstrap runtime state/fallback-decision logic into a dedicated module.
- Extract panel fallback recreate-plan / duplicate-panel race-guard decisions into a dedicated module.
- Keep pywebview window creation/destruction calls and `panel_action(...)` contracts unchanged in the monolith during S9.

**Out of scope (S9 must not do this)**
- No feature additions.
- No change to `panel_action(...)` names, payload schemas, or JS callback semantics.
- No provider/rendering behavior changes.
- No broad UI controller/orchestrator rewrite.

#### S9 acceptance checklist (completed)
- [x] Panel asset/static-selftest helper logic extracted into a separate module with dedicated tests.
- [x] Panel bootstrap runtime state/fallback-reason logic extracted into a separate module with dedicated tests.
- [x] Panel fallback recreate-plan / duplicate-panel race-guard decisions extracted into a separate module with dedicated tests.
- [x] Monolith delegates to extracted S9 helper modules using soft-import fail-open behavior.
- [x] Targeted panel regression tests remain green after S9a-S9d.
- [x] Manual pywebview runtime-fallback re-test passed after S9 changes (broken `panel_bootstrap_selftest` callback -> embedded fallback, no duplicate panel).
- [x] Manual panel/manual-test smoke checks re-run after S9 changes (`qc_override_footer`, `full_regression_light`).

Current S9 verification snapshot (2026-02-24):
- Implemented/committed in private repo:
  - `e0bf605` (`S9a`: `panel_asset_loader`)
  - `c7a1df1` (`S9b`: `panel_bootstrap_state`)
  - `49e8ba0` (`S9c`: `panel_window_fallback`)
  - `e2e4910` (`S9d`: `panel_html_source`)
- Automated: module tests for S9a-S9d (`tests/test_panel_asset_loader.py`, `tests/test_panel_bootstrap_state.py`, `tests/test_panel_window_fallback.py`, `tests/test_panel_html_source.py`) -> `22 PASS`.
- Automated: targeted panel regression subset in `tests/test_app.py` (static selftest, runtime payload validation, bootstrap callback acceptance, duplicate-panel race guard) -> `6 PASS`.
- Manual: pywebview broken-callback fallback re-test passed (delayed fallback to embedded panel, no duplicate-panel regression on Panel toggle); fallback evidence recorded in `Logs/Audit/AuditStream_20260224.jsonl` (`panel_bootstrap -> fallback_to_embedded`, `runtime_selftest_timeout`).
- Manual: `qc_override_footer` -> `PASS` (`Logs/ManualTests/ManualTest_20260224_192212_736057_qc_override_footer.json`), `full_regression_light` -> `PASS` (`Logs/ManualTests/ManualTest_20260224_192619_681365_full_regression_light.json`).

#### S9 execution notes
- Keep pywebview operations (`create_window`, `destroy`, event binding, show/hide/focus) in the monolith until S9 acceptance is complete.
- Extract only pure decisions/state transitions first; verify with focused tests before touching window-lifecycle calls.
- Use the existing S7/S8 manual fallback test procedure as the S9 manual gate.

---

## Contribution workflow (recommended)
1. Pick one stage item.
2. Write a failing test that captures the regression risk.
3. Implement the smallest change to make the test pass.
4. Keep PRs small and describable in one paragraph.
