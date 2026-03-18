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

### Output/Rendering-Track A-H (active, 2026-03-11)
This A-H track is the active migration plan for deterministic output/rendering modularization and runs in parallel to the historical S-stage roadmap.

Hard constraints for this track:
- UI visual structure stays unchanged.
- No merge without green tests.
- Mandatory gate after every relevant slice: `python3 scripts/quality_gate.py --mode all`.
- Self-Debunking, uncertainty markers, header/footer, and command outputs remain deterministic.
- No distributed regex end-state; target is a centralized output pipeline.

Current completion estimate (2026-03-11):
- Overall A-H progress: **~50-55%**.
- Output-pipeline-only progress (without monolith shrink target): **~65-70%**.

Phase status snapshot:
- [x] **Phase A** (`src/output/state_snapshot.py`) mostly done and integrated.
- [x] **Phase B** (`src/output/rules_registry.py`) mostly done and integrated.
- [~] **Phase C** (`src/output/resolver.py`, `src/output/dispatcher.py`, `src/output/routing_runtime.py`) partially done; module routing is active, legacy fallback still present.
- [~] **Phase D** (`src/output/response_model.py`, `src/output/pipeline.py`) partially done; post-render normalization is modularized, full structural parser/validator chain is not complete yet.
- [x] **Phase E** (renderer modules) extracted behind fail-soft wrapper interfaces:
  - done: `header.py`, `footer_qc.py`, `sci_trace.py`, `uncertainty.py`
  - done: `csc_warning.py`, `control_layer_note.py`, `cgi_line.py`, `color_markers.py`
- [~] **Phase F** (`src/output/commands/*.py`) started:
  - `output.commands.response_catalog` introduced for deterministic post-state command outputs
    (Profile switch audit line + post-command profile/comm-state response selection, fail-soft delegated).
  - deterministic command responses now delegated for:
    `QC Override`, `SCI on/menu`, and static renderer-map commands (`Comm Help/State/Config/Anchor`).
  - basic legacy command transitions delegated (`Strict/Explore/Color/SCI/Comm/Anchor/Dynamic one-shot`) via
    `apply_basic_command_state` (fail-soft fallback remains in monolith).
- [ ] **Phase G** (legacy path removal / monolith thinning) open; `src/Comm-SCI-Control-App.py` still contains substantial output logic and fail-open fallbacks.
- [~] **Phase H** (hardening + docs) partially done; replay contract exists, but matrix expansion is still open.

Acceptance criteria (DoD) for this track:
- Self-Debunking labels are bold, with `:`, one label per line, stable in DE/EN.
- Uncertainty markers appear only in content context, with tooltip; never on header/status/control-layer notes.
- Color off removes visual color markers/signal dots from final output.
- Profile Briefing output keeps one deterministic format including audit line.
- Exactly one final QC-Matrix footer line remains.
- No visible UI regression.

Execution order for next slices:
1. **E-complete**: extract missing renderer modules behind stable interfaces.
2. **F-start**: introduce `src/output/commands/` and move deterministic command responses there.
3. **G-start**: remove now-redundant output logic from monolith (keep wrapper as orchestration/fallback only).
4. **H-expand**: replay/test hardening for DE/EN, SCI off/A/B, profile transitions, color on/off, tooltip presence, and exactly-one QC footer.

Per-slice quality gate:
- targeted tests for touched surface
- `python3 scripts/quality_gate.py --mode all`

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

### S10 — Monolith thinning (visible size reduction, no behavior change)
**Goal:** make `src/Comm-SCI-Control-App.py` visibly smaller after S9 by removing now-redundant helper/fallback duplication while preserving runtime behavior.
- Reduce S9 delegation duplicates in the monolith (keep only compact emergency shims where needed).
- Keep pywebview window operations and panel lifecycle behavior unchanged.
- Preserve all `panel_action(...)` names/payloads and S7/S8/S9 panel fallback behavior.
- Keep the private-first workflow: refactor + tests/manual gates in private repo, then sync to public wrapper.

**Out of scope (S10 must not do this)**
- No feature additions or UI changes.
- No provider/governance/rendering behavior changes.
- No pywebview timing optimization (timeouts/fallback frequency analysis is separate work).
- No broad orchestrator/service rewrite outside the panel/fallback seam.

#### S10 acceptance checklist (completed)
- [x] S10a baseline measured and documented (current monolith size + hotspot references).
- [x] `Comm-SCI-Control-App.py` is materially smaller after S10 (documented delta vs S10a baseline: **-50 lines** after S10d).
- [x] S9 helper-module delegations in the monolith are reduced to compact shims / direct module calls where safe (S10b first pass: `_panel_accept_bootstrap_report`, `_panel_swap_to_embedded_fallback`).
- [x] No change in `panel_action(...)` route names or response payload schemas (S10 touched panel fallback/wait/close cleanup paths only; panel route subset regressions remained green).
- [x] S9 module tests remain green (`tests/test_panel_asset_loader.py`, `tests/test_panel_bootstrap_state.py`, `tests/test_panel_window_fallback.py`, `tests/test_panel_html_source.py`) after S10b (`22 PASS`).
- [x] Targeted panel regression subset in `tests/test_app.py` remains green after S10b (`6 PASS`).
- [x] Manual pywebview runtime-fallback re-test passes (broken `panel_bootstrap_selftest` callback -> embedded fallback, no duplicate panel).
- [x] Manual panel/manual-test smoke checks re-run (`qc_override_footer`, `full_regression_light`).

Current S10 baseline snapshot (2026-02-24, S10a):
- Monolith size: `src/Comm-SCI-Control-App.py` = **16,188 lines** (`wc -l`).
- Remaining panel/fallback hotspot methods still in the monolith:
  - `_panel_get_embedded_html(...)` (`src/Comm-SCI-Control-App.py:12070`)
  - `_panel_select_html_for_window(...)` (`src/Comm-SCI-Control-App.py:12093`)
  - `_panel_begin_bootstrap_probe(...)` (`src/Comm-SCI-Control-App.py:12123`)
  - `_panel_accept_bootstrap_report(...)` (`src/Comm-SCI-Control-App.py:12162`)
  - `_panel_swap_to_embedded_fallback(...)` (`src/Comm-SCI-Control-App.py:12255`)
  - `_panel_wait_bootstrap_or_fallback(...)` (`src/Comm-SCI-Control-App.py:12358`)
  - `on_panel_closed(...)` (`src/Comm-SCI-Control-App.py:12898`)
- S10 focus: shrink these monolith methods by removing duplicated fallback logic that now exists in `panel_asset_loader`, `panel_bootstrap_state`, `panel_window_fallback`, and `panel_html_source`.

#### S10 execution notes
- Prefer small, behavior-preserving deletions/compactions over moving additional pywebview calls into modules.
- After each S10 sub-step, run the focused panel test gates before touching another hotspot.
- Keep manual fallback smoke (`panel_bootstrap_selftest` broken callback) as the final gate before marking S10 complete.

S10b progress snapshot (2026-02-25):
- Monolith size after first S10b compaction: `src/Comm-SCI-Control-App.py` = **16,162 lines** (`wc -l`) => **-26 lines** vs S10a baseline (16,188).
- First visible compaction completed in `_panel_accept_bootstrap_report(...)` and `_panel_swap_to_embedded_fallback(...)` (duplicate side-effect and fallback-state paths reduced; no contract changes).
- Focused gates run after S10b: `py_compile` OK, S9 panel helper tests `22 PASS`, targeted panel regression subset `6 PASS`.

S10c progress snapshot (2026-02-25):
- Monolith size after second S10c compaction: `src/Comm-SCI-Control-App.py` = **16,154 lines** (`wc -l`) => **-34 lines** vs S10a baseline (16,188).
- `on_panel_closed(...)` compacted (retired-panel close-event race-guard fallback path + closed-state reset path deduplicated; behavior preserved).
- Focused gates re-run after S10c: `py_compile` OK, S9 panel helper tests `22 PASS`, targeted panel regression subset `6 PASS`.

S10d progress snapshot (2026-02-25):
- Monolith size after third S10d compaction: `src/Comm-SCI-Control-App.py` = **16,138 lines** (`wc -l`) => **-50 lines** vs S10a baseline (16,188).
- `_panel_wait_bootstrap_or_fallback(...)` compacted via shared local ready/reason helper (module path + fallback shim logic deduplicated; wait/fallback behavior preserved).
- Focused gates re-run after S10d: `py_compile` OK, S9 panel helper tests `22 PASS`, targeted panel regression subset `6 PASS`.

S10e completion snapshot (2026-02-25):
- Manual pywebview fallback re-test (broken `panel_bootstrap_selftest` callback) passed after S10d: delayed panel open OK, embedded fallback OK, no duplicate panel on `Panel` button (user-verified; `panel.html` restored afterwards).
- Manual smoke re-runs passed:
  - `Logs/ManualTests/ManualTest_20260225_073852_862939_full_regression_light.json` -> `PASS` (including `PASS: OpenRouter: Self-Debunking-Box erkannt`)
  - `Logs/ManualTests/ManualTest_20260225_073943_015319_qc_override_footer.json` -> `PASS`
- S10 status: **complete** (S10a-S10e).


### S11 — Panel lifecycle seam extraction (next visible monolith reduction)
**Goal:** continue thinning `src/Comm-SCI-Control-App.py` after S10 by extracting the remaining panel lifecycle orchestration seam (HTML source selection, bootstrap probe wiring, fallback swap coordination) while preserving pywebview runtime behavior.
- Reduce the remaining panel hotspot methods in the monolith to thin pywebview adapters and explicit fail-open shims.
- Consolidate panel bootstrap source/probe/fallback sequencing into a dedicated helper/controller module (exact module split may be one or two files, depending on testability).
- Preserve all S7/S8/S9/S10 runtime guarantees (external `panel.html` + runtime self-test + embedded fallback + duplicate-panel race guard).
- Keep the private-first workflow: plan/implement/test in private repo, then sync to public wrapper.

**Out of scope (S11 must not do this)**
- No provider/governance/SCI behavior changes (including QC/CSC/repair logic).
- No `panel_action(...)` contract changes (names, payloads, callback semantics).
- No pywebview timing/timeout tuning (frequent fallback diagnostics remain separate work).
- No broad UI controller/orchestrator rewrite outside the panel lifecycle seam.

#### S11 acceptance checklist (completed - functional seam extraction; size-reduction goal deferred)
- [x] S11a baseline measured and documented (post-S10 + post-SCI-fix monolith size and current hotspot line refs).
- [ ] `Comm-SCI-Control-App.py` is materially smaller after S11 (documented delta vs S11a baseline). Deferred to S12/S11-follow-up compaction after seam consolidation; S11 completed on functional/architectural acceptance.
- [x] Remaining panel lifecycle hotspot methods are reduced to thin pywebview adapters / compact fail-open shims.
- [x] No change in `panel_action(...)` route names or response payload schemas.
- [x] Focused panel test gates remain green (`tests/test_panel_asset_loader.py`, `tests/test_panel_bootstrap_state.py`, `tests/test_panel_window_fallback.py`, `tests/test_panel_html_source.py`, targeted `tests/test_app.py` subset).
- [x] Manual pywebview runtime-fallback re-test passes after S11 (broken `panel_bootstrap_selftest` callback -> embedded fallback, no duplicate panel).
- [x] Manual smoke checks re-run after S11 (`qc_override_footer`, `full_regression_light`).
- [x] Wrapper sync possible without follow-up hotfixes (pending wrapper sync execution, no private-side hotfix required after S11e gates).

Current S11 baseline snapshot (2026-02-25, S11a):
- Monolith size: `src/Comm-SCI-Control-App.py` = **16,182 lines** (`wc -l`).
- Remaining panel lifecycle hotspot methods still in the monolith:
  - `_panel_get_embedded_html(...)` (`src/Comm-SCI-Control-App.py:12114`)
  - `_panel_select_html_for_window(...)` (`src/Comm-SCI-Control-App.py:12137`)
  - `_panel_begin_bootstrap_probe(...)` (`src/Comm-SCI-Control-App.py:12167`)
  - `_panel_accept_bootstrap_report(...)` (`src/Comm-SCI-Control-App.py:12206`)
  - `_panel_swap_to_embedded_fallback(...)` (`src/Comm-SCI-Control-App.py:12282`)
  - `_panel_wait_bootstrap_or_fallback(...)` (`src/Comm-SCI-Control-App.py:12376`)
  - `on_panel_closed(...)` (`src/Comm-SCI-Control-App.py:12900`)
- S11 focus: reduce these methods further by moving orchestration/sequencing into a dedicated panel runtime seam while keeping direct pywebview calls behavior-preserving and test-gated.

S11b progress snapshot (2026-02-25):
- Added new seam helper module `src/panel_lifecycle_seam.py` (HTML source plan + bootstrap probe plan extraction, fail-open via monolith soft-import).
- Monolith methods `_panel_select_html_for_window(...)` and `_panel_begin_bootstrap_probe(...)` now delegate first to `panel_lifecycle_seam` and retain compact local fallbacks.
- Added focused seam tests in `tests/test_panel_lifecycle_seam.py` (selection/probe plan behavior).
- Focused gates after S11b:
  - `py_compile` OK (`src/Comm-SCI-Control-App.py`, `src/panel_lifecycle_seam.py`, `tests/test_panel_lifecycle_seam.py`)
  - Panel helper/seam tests: `26 PASS` (`tests/test_panel_lifecycle_seam.py` + S9 helper modules)
- Targeted panel regression subset in `tests/test_app.py`: `6 PASS`
- Monolith size after S11b extraction-first pass: `src/Comm-SCI-Control-App.py` = **16,198 lines** (`wc -l`) => **+16 lines** vs S11a baseline (expected for first seam extraction; visible size reduction is deferred to later S11 compaction steps).

S11c progress snapshot (2026-02-25):
- Extended `src/panel_lifecycle_seam.py` with panel bootstrap/fallback orchestration helpers:
  - `panel_bootstrap_ready_and_reason(...)`
  - `panel_bootstrap_wait_plan(...)`
  - `panel_embedded_fallback_swap_plan(...)`
- Monolith methods `_panel_wait_bootstrap_or_fallback(...)` and `_panel_swap_to_embedded_fallback(...)` now delegate decision/plan logic first to `panel_lifecycle_seam`, while pywebview operations (`Event.wait`, `_create_panel()`, `destroy()`) remain local.
- Expanded seam tests in `tests/test_panel_lifecycle_seam.py` (wait-plan + fallback-swap plan coverage).
- Focused gates after S11c:
  - `py_compile` OK (`src/Comm-SCI-Control-App.py`, `src/panel_lifecycle_seam.py`, `tests/test_panel_lifecycle_seam.py`)
  - Panel helper/seam tests: `30 PASS` (`tests/test_panel_lifecycle_seam.py` + S9 helper modules)
- Targeted panel regression subset in `tests/test_app.py`: `6 PASS`
- Monolith size after S11c extraction-first pass: `src/Comm-SCI-Control-App.py` = **16,249 lines** (`wc -l`) => **+67 lines** vs S11a baseline (still expected during seam consolidation; visible line reduction is deferred to later S11 compaction steps).

S11d progress snapshot (2026-02-25):
- Extended `src/panel_lifecycle_seam.py` with `panel_closed_event_plan(...)` to centralize retired-panel closed-event race-guard decision + closed bootstrap-state reset planning.
- Monolith method `on_panel_closed(...)` now delegates lifecycle decision/reset planning first to `panel_lifecycle_seam`, while geometry capture and direct window-handle cleanup remain local.
- Expanded seam tests in `tests/test_panel_lifecycle_seam.py` (retired close-event ignore path + normal closed-state path).
- Focused gates after S11d:
  - `py_compile` OK (`src/Comm-SCI-Control-App.py`, `src/panel_lifecycle_seam.py`, `tests/test_panel_lifecycle_seam.py`)
  - Panel helper/seam tests: `32 PASS` (`tests/test_panel_lifecycle_seam.py` + S9 helper modules)
- Targeted panel regression subset in `tests/test_app.py`: `6 PASS`
- Monolith size after S11d extraction-first pass: `src/Comm-SCI-Control-App.py` = **16,277 lines** (`wc -l`) => **+95 lines** vs S11a baseline (S11 is still in seam-consolidation mode; compaction/noise reduction remains for S11e or follow-up S11 cleanup before final acceptance).

S11e completion snapshot (2026-02-25):
- Manual pywebview runtime-fallback re-test (broken `panel_bootstrap_selftest` callback) passed after S11d:
  - delayed panel opening observed (user-verified)
  - embedded fallback panel opened successfully (user-verified)
  - no duplicate panel on `Panel` button (user-verified)
  - `panel.html` restored after the test
- Manual smoke re-runs passed:
  - `Logs/ManualTests/ManualTest_20260225_184043_289386_qc_override_footer.json` -> `summary.status=PASS`, `summary.fails=0`
  - `Logs/ManualTests/ManualTest_20260225_184314_847161_full_regression_light.json` -> `summary.status=PASS`, `summary.fails=0`
- Supporting audit trace evidence (panel fallback info events present in session stream):
  - `Logs/Audit/AuditStream_20260225.jsonl`
  - `Logs/Audit/Audit_20260225_184446_767756.json`
- S11 status: **functionally complete (seam extraction accepted)**; visible monolith size reduction deferred to a dedicated follow-up compaction stage.

#### S11 execution notes
- Prefer extraction of sequencing/orchestration helpers over moving raw pywebview calls on the first pass.
- Keep fail-open behavior explicit: if the new helper import fails, panel fallback path must still remain usable.
- Reuse the established S7/S8/S9/S10 manual fallback test procedure as the final gate.
- Treat SCI/repair fixes (like the `Dialectic_6_Synthesis2` alias hardening) as separate patches, not S11 scope.

### S12 — UI/Bridge orchestration seam expansion + post-S12 hardening (completed)
**Goal:** continue the post-S11 monolith-thinning path by extracting additional panel/QC orchestration decisions into seam modules while keeping direct pywebview calls in the monolith, then close user-found regressions before sync.
- Expand panel lifecycle seam coverage beyond bootstrap/fallback into visibility/toggle, closing/binding, and rebuild sequencing.
- Extract QC override dialog window/apply/clear UI planning into dedicated seam helpers.
- Extract thin `QCBridge` / `PanelBridge` classes into standalone modules (monolith keeps fail-open import fallback).
- Preserve `panel_action(...)` contracts and pywebview runtime behavior; treat user-reported rendering/QC issues as separate hardening patches.

#### S12 acceptance checklist (completed - functional seam expansion; net size reduction deferred to S13)
- [x] S12a post-S11 baseline measured and documented (`src/Comm-SCI-Control-App.py` size + next UI/panel hotspots).
- [x] Panel visibility/toggle, closing/binding, and rebuild decision/orchestration logic extracted into seam plan helpers with focused tests.
- [x] QC override dialog window + apply/clear UI planning extracted into seam helpers with focused tests.
- [x] `QCBridge` and `PanelBridge` extracted to standalone modules with monolith fail-open fallback imports.
- [x] User-reported regressions after S12 slices were fixed before sync (QC clear prompt-context reset, Self-Debunking HTML fragment repair, mixed QC footer dedupe, repair-banner UX filtering).
- [x] Manual smoke checks re-run on the private repo (panel toggle/rebuild, QC override apply/clear, content answers) with user verification after fixes.
- [x] Private repo push + GitHub `tests` CI green, then wrapper sync + wrapper GitHub `tests` CI green.
- [ ] `Comm-SCI-Control-App.py` is materially smaller after S12. Deferred to S13 composition/wiring compaction; S12 focused on seam coverage + hardening closure.

S12 completion snapshot (2026-02-26):
- Private-first S12 seam commits (functional extraction):
  - `26cec03` visibility decision plans
  - `28ccef8` closing/event-binding plans
  - `6fc53e4` rebuild orchestration plan
  - `2aa9900` QC override dialog window plans
  - `720e92b` QC override apply/clear UI plans
  - `b62f038` `QCBridge` module extraction
  - `15139ec` `PanelBridge` module extraction
- Post-S12 hardening/fixes (user-driven regressions closed before sync):
  - `5f2b1b9` QC override clear -> one-shot prompt reset directive (prevents stale override carry-over)
  - `cdaac9c` Self-Debunking HTML normalization hardening for fragmented `<ol>/<li>/<p>` output
  - `d1a3a5b` mixed `QC:` + `QC-Matrix:` HTML footer dedupe to one canonical footer
  - `06844f5` suppress `CONTROL LAYER NOTE` for format-only Self-Debunking repair passes (audit/repair preserved)
- S12 seam modules now present:
  - `src/panel_lifecycle_seam.py`
  - `src/qc_override_window_seam.py`
  - `src/qc_bridge.py`
  - `src/panel_bridge.py`
- Monolith size note:
  - S12a baseline (post-S11): `src/Comm-SCI-Control-App.py` = **16,349 lines** (`wc -l`)
  - Post-S12 completion + hardening fixes: `src/Comm-SCI-Control-App.py` = **16,801 lines** (`wc -l`)
  - Net growth is due to seam expansion plus regression hardening; explicit line-count reduction is deferred to S13 compaction/composition work.
- Verification summary (private + wrapper):
  - Focused seam and regression tests remained green locally during S12 slices/fixes (panel seam, QC override seam, bridge modules, targeted `tests/test_app.py`)
  - Private `main` CI green after S12 closure push (`tests` on `06844f5`)
  - Wrapper synced from private and wrapper `tests` CI green after sync commit

#### S12 execution notes
- Keep pywebview operations in the monolith until seam behavior is fully stabilized; extract plans/decisions first.
- Treat user-found rendering/QC regressions as release-blocking follow-up fixes before wrapper sync.
- S13 should prioritize net monolith reduction (composition/wiring compaction), not only additional seam coverage.

### S13 — Composition/Wiring compaction (next net monolith reduction stage)
**Goal:** convert the S11/S12 seam groundwork into visible monolith reduction by moving remaining UI/bridge composition and low-risk UI orchestration glue out of `src/Comm-SCI-Control-App.py`, while preserving behavior and contracts.
- Prioritize **net line reduction** in the monolith (not just new seam modules).
- Keep pywebview calls behavior-preserving; continue extracting plans/builders/factories first.
- Preserve `panel_action(...)` contracts, QC/QC-Override semantics, audit behavior, and provider switching behavior.
- Continue private-first workflow (implement + verify in private repo, then sync to wrapper).

**Out of scope (S13 must not do this)**
- No governance/repair/SCI feature refactor (e.g. `_apply_csc_strict(...)`, repair validator behavior) unless fixing a user-reported regression.
- No provider behavior changes or model-selection policy changes.
- No `panel_action(...)` route renames / payload schema changes.
- No UI redesign.

#### S13 acceptance checklist (planned)
- [x] S13a baseline measured and documented (post-S12 monolith size + hotspot map + first slice order).
- [ ] `Comm-SCI-Control-App.py` is materially smaller after S13 (documented delta vs S13a baseline).
- [ ] Remaining UI/bridge composition/wiring glue is reduced to thin adapters/fail-open shims where practical.
- [ ] `panel_action(...)` route names/payload schemas unchanged.
- [ ] Focused UI/bridge/panel tests remain green after each S13 slice.
- [ ] Manual smoke checks re-run before wrapper sync (panel, QC override, manual-test monitor, provider/model switch).
- [ ] Private CI green, then wrapper sync, then wrapper CI green.

S13a baseline snapshot (2026-02-26):
- Monolith size: `src/Comm-SCI-Control-App.py` = **16,801 lines** (`wc -l`).
- Highest-risk large methods remain:
  - `_apply_csc_strict(...)` (`src/Comm-SCI-Control-App.py:8129`) ~ 815 lines (governance/rendering-heavy; **not** the first S13 extraction target)
  - `ask(...)` (`src/Comm-SCI-Control-App.py:10244`) ~ 826 lines (request pipeline; high regression risk)
- Primary S13 UI/bridge compaction hotspots (lower-risk, wiring-heavy):
  - `get_ui(...)` (`src/Comm-SCI-Control-App.py:11599`) panel UI snapshot assembly + local/offline merges + UI normalization/gating
  - Manual test monitor window lifecycle/UI methods:
    - `_bind_manual_test_monitor_window_events(...)` (`src/Comm-SCI-Control-App.py:11410`)
    - `_create_manual_test_monitor(...)` (`src/Comm-SCI-Control-App.py:11421`)
    - `manual_test_monitor_show(...)` (`src/Comm-SCI-Control-App.py:11468`)
    - `manual_test_monitor_hide(...)` (`src/Comm-SCI-Control-App.py:11523`)
    - `manual_test_monitor_reset(...)` (`src/Comm-SCI-Control-App.py:11539`)
    - `manual_test_monitor_append(...)` (`src/Comm-SCI-Control-App.py:11559`)
    - `manual_test_monitor_set_header(...)` (`src/Comm-SCI-Control-App.py:11580`)
  - Bridge/fallback wiring region:
    - `_QCBridge` fallback binding (`src/Comm-SCI-Control-App.py:16490`)
    - `_PanelBridge` fallback binding (`src/Comm-SCI-Control-App.py:16525`)
    - `MainBridge` class (`src/Comm-SCI-Control-App.py:16544`)
- `panel_action(...)` (`src/Comm-SCI-Control-App.py:11225`) is already relatively compact after S6, but remains a contract-sensitive integration point to keep regression-tested during S13.

Planned S13 slice order (pragmatic):
- S13b (low risk): manual-test-monitor window/orchestration seam extraction (window create/show/hide + state/UI push planning), pywebview calls remain local.
- S13c: `get_ui(...)` split into pure snapshot-builder helpers (provider/model/language snapshot, governance merge, comm-off gating, log list merge).
- S13d: bridge/fallback wiring compaction (`MainBridge` / fallback bridge shims -> module/factory extraction with fail-open imports).
- S13e: cleanup/compaction pass + focused tests + manual smoke + wrapper sync.

#### S13 execution notes
- Prefer extracting pure builders/plans/factories before touching `ask(...)` or `_apply_csc_strict(...)`.
- Treat line-count reduction as an explicit acceptance metric in S13, not a side effect.
- Keep user-facing regression fixes separate from S13 scope accounting unless they are caused by S13 changes.

### S15 — Scenario harness + Manual-Test Komplextest hardening
**Goal:** provide reproducible, GUI-near scenario checks with machine-readable output and safe interruption paths.

#### S15.1 snapshot (2026-03-01)
- Deterministic harness groundwork added:
  - `src/manual_scenario_harness.py`
  - `scripts/run_scenario_harness.py`
  - `tests/test_manual_scenario_harness.py`
  - `docs/manual_scenario_protocol.md`
- Scope:
  - mandatory prompts (including long governance/fairness prompt),
  - deterministic matrix generation over `profile x sci_variant x qc_override x color`,
  - structured checks for QC footer, U markers, color markers, CGI/dynamic influence.

#### S15.1.1 snapshot (2026-03-01)
- Panel `komplexttest` now writes export checkpoints before destructive `clear_chat` steps and at finalization.
- Manual-test abort path hardened:
  - monitor window gets active `Stop` action,
  - new backend route `manual_test_stop` sets the running test to stop safely,
  - `STOPPED`/`ERROR` finalization now writes partial export snapshots and report JSON.
- Localization parity in monitor improved:
  - header/stop labels and status feedback respond to selected answer language (`de`/`en`).
- Focused verification gates:
  - `tests/test_panel_manual_test_scenarios.py`
  - `tests/test_app.py` (manual-test stop route and monitor wiring subset)
  - app selftest (`Comm-SCI-Control-App.py --selftest`)


---

## Contribution workflow (recommended)
1. Pick one stage item.
2. Write a failing test that captures the regression risk.
3. Implement the smallest change to make the test pass.
4. Keep PRs small and describable in one paragraph.
