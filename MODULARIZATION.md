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

### S4 — UI decoupling
**Goal:** make UI interactions a thin, testable layer.
- Introduce `UIController` with standardized actions/responses.
- Preserve UI behavior; change only through tests.

### S5 — Storage & logs as services
**Goal:** deterministic persistence and schema enforcement.
- Introduce `StorageService` for load/fork/export.
- Enforce schema‑versioned audit logs (e.g., “audit v2+”) in exports.

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

