---
name: "Roadmap: Wrapper modularization (tracking)"
about: Tracking issue for stepwise modularization (regression-safe)
title: "Roadmap: Wrapper modularization (tracking)"
labels: ["tracking", "refactor", "enhancement"]
assignees: []
---

# Roadmap: Wrapper modularization (tracking)

> Purpose: make the modularization effort visible, reviewable, and easy to contribute to.  
> Non‑goals: architecture purity; breaking changes without tests.

## Current status
- [ ] Baseline tests green on the **current** wrapper/test pair (e.g., `Wrapper-<NNN>.py` / `Test-<NNN>.py`)
- [ ] “Golden Run” manual checklist documented

## Constraints (must not regress)
- Determinism / no hidden state changes
- Audit logs remain correct and schema‑versioned
- QC calculation + QC override behavior preserved
- SCI menu/format preserved
- UI/HTML output preserved unless explicitly intended + tested
- No silent provider/model switching
- Tests must never trigger real network calls (mock/stub providers)

## Stages
### S0 — Baseline hardening
- [ ] Document Golden Run (startup, panel lifecycle, provider/model switch, log export/load/fork, QC override apply/clear)
- [ ] Add smoke tests (no network)
- [ ] Add minimal observability hook: `log_event(kind, payload)`
- [ ] Add invariants/guards (visible provider switch; stable defaults; UI payload shape)

### S1 — Clear internal boundaries (still single file)
- [ ] Add clear section boundaries + minimal contracts
- [ ] Reduce implicit globals where safe; pass explicit state object(s)
- [ ] Add contract/golden tests for boundaries

### S2 — Provider encapsulation
- [ ] Introduce `ProviderClient` interface + registry + `NullClient`
- [ ] Explicit provider/model change events (no silent switching)
- [ ] Deterministic provider/model selection

### S3 — Governance engine isolation
- [ ] Introduce `GovernanceService` (policy gate + rules application pipeline)
- [ ] Centralize reset rules (profile switch, clear chat, comm stop)
- [ ] Ensure no UI logic inside governance state

### S4 — UI decoupling
- [ ] Introduce `UIController` with standardized actions/responses
- [ ] Preserve UI behavior through tests

### S5 — Storage & logs as services
- [ ] Introduce `StorageService`
- [ ] Deterministic load/fork/export
- [ ] Enforce schema‑versioned audit logs in export (e.g., audit v2+)

## How to help
- Pick one checkbox, propose a minimal PR, and keep it test‑backed.
- If unsure: start by adding a regression test for a known risk.

