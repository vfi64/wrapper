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
- [ ] Panel HTML/JS is loaded from local asset file(s) with deterministic fallback.
- [ ] Chat HTML/JS/CSS is loaded from local asset file(s) with deterministic fallback.
- [ ] No change in `panel_action` command names or payload schemas.
- [ ] Existing panel/manual-test smoke checks remain green.
- [ ] Wrapper file size is materially reduced (documented delta).

### Optional later
- External templates (feature‑flagged)
- Multi‑file split into packages (only after stable interfaces)
- Additional stages only if required by new features

---

## Contribution workflow (recommended)
1. Pick one stage item.
2. Write a failing test that captures the regression risk.
3. Implement the smallest change to make the test pass.
4. Keep PRs small and describable in one paragraph.
