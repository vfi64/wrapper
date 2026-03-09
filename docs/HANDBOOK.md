# Comm-SCI-Control-Private Handbook (EN)

Version baseline: wrapper `1.0.x` (current: `1.0.11`)  
Runtime default ruleset: `JSON/Comm-SCI-v20.2.2.json`

## 1. Purpose

This repository contains the runtime implementation of Comm-SCI governance.
The public ruleset defines normative behavior; this wrapper enforces it operationally:

- deterministic command routing
- runtime state transitions
- output contract checks and repair
- audit/export functions
- UI/panel integration for real operation

The wrapper is intentionally strict on command tokens and fail-soft on optional tooling.

## 2. Relationship Between Repositories

- Public ruleset and concepts: [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)
- Runtime implementation: [vfi64/wrapper](https://github.com/vfi64/wrapper)
- Public wrapper website: [vfi64.github.io/wrapper](https://vfi64.github.io/wrapper/)
- This repository: staging and integration track for wrapper changes

Practical rule:
JSON is the governance source, wrapper is the execution source.

## 3. Architecture Snapshot

Core implementation files:

- `src/Comm-SCI-Control-App.py`: monolithic runtime orchestration
- `src/app_bootstrap.py`: startup/bootstrap and window lifecycle guards
- `src/provider_service.py`: provider and model handling
- `src/governance_service.py`: state/governance helpers
- `src/ui_assets/`: panel/chat HTML assets
- `src/Module/`: optional modules (`auditstream`, `rendering_utils`, `compliance_scan`)

Test focus:

- `tests/test_app.py`: behavioral contracts and regressions
- `tests/test_app_bootstrap.py`: bootstrap lifecycle and panel readiness

## 4. Runtime Execution Model

The operational flow follows deterministic stages:

1. Parse input and classify command vs content.
2. Route command intent and update governance state.
3. Build governed prompt (if content route).
4. Execute provider call.
5. Enforce contracts (SCI trace, Self-Debunking, QC footer, verification gates).
6. Render deterministic HTML output and session metadata.

Important:
UI commands such as `Comm Help`, `Comm State`, `Comm Config`, `Comm Anchor`, and `Comm Audit` should be local deterministic renderers and must not require a model call.

## 5. Command Discipline

Command parsing is standalone-only by design.
Expected behavior:

- exact command token line
- no mixed command + content in one line
- SCI variant letter `A-H` is interpreted as variant selection only while SCI is pending

This prevents accidental mode changes and supports reproducible automation.

## 6. SCI Governance

SCI mode activates trace-enforced reasoning format.

- `SCI on` arms variant selection.
- Variant letter `A-H` selects the trace template.
- `SCI recurse` is one-turn deepening while SCI is active.
- `SCI off` returns to non-trace output mode.

Variant metadata (name/focus/mapping/steps) is taken from loaded JSON and shown in help/UI.

## 7. QC, CGI, and Control Layer

### QC

QC footer values are normalized and checked against profile targets.
Overrides are allowed but must be visible (`Comm State` / footer metadata).

### CGI

CGI captures user feedback signals and can affect adjustment logic.
CGI input parsing should be intercepted locally where applicable.

### Control Layer

Control Layer enforces:

- output structure and ordering
- required/forbidden blocks by route
- deterministic fallback behavior

## 8. Language Policy: `production` vs `benchmark`

The runtime exposes `language_policy_mode` in state/config.

- `production`: user-facing operational mode, conservative and stable defaults
- `benchmark`: evaluation/comparison mode for controlled experiments and regression runs

Why this matters:

- separates day-to-day usage from measurement workflows
- enables explicit documentation of experiment context
- reduces ambiguity when comparing model behavior over time

## 9. API-key Modes and Passphrase Flow

The wrapper supports two local storage modes per provider in `Config/Comm-SCI-API-Keys.json`:

- `api_key_plain`: stored unencrypted, no passphrase dialog required
- `api_key_enc`: stored encrypted, unlocked via passphrase dialog

Runtime behavior:

- On startup, the wrapper prompts for a passphrase if the initial provider uses `api_key_enc`.
- On provider switch, the passphrase dialog appears only when the target provider is encrypted.
- Providers can be mixed (for example, Gemini encrypted and OpenRouter plaintext).
- API keys can be entered, updated, or deleted via the panel path `Provider & LLM` -> `API-Key`.

## 10. Operational Playbook

Recommended baseline session:

1. Load ruleset and verify with `Comm State`.
2. `Comm Start`
3. Select profile (`Profile Expert`, `Profile Standard`, etc.).
4. Optional: `SCI on` + variant letter.
5. Ask content question.
6. In longer sessions: run `Comm Anchor` and `Comm Audit` periodically.

When instability appears:

- check command formatting and state first
- verify provider/model/API key in panel
- re-anchor and audit before continuing
- Input-line history is auto-saved on shutdown to `Logs/History/InputLineHistory.json` and auto-loaded at startup.

## 11. Documentation and Website Structure

This repository keeps project-facing docs under `docs/`:

- `docs/index.html` / `docs/index.de.html`: bilingual landing pages
- `docs/glossary.html` / `docs/glossar.de.html`: didactic terminology pages including SCI A-H examples
- `docs/install-beginner.html` / `docs/install-beginner.de.html`: guided install for non-technical users
- `docs/install-pro.html` / `docs/install-pro.de.html`: advanced install for reproducible professional operation
- `docs/HANDBOOK.md` / `docs/HANDBOOK.de.md`: operational handbook
- `docs/manual-*.md`: release/manual test checklists
- `docs/proposals/`: design backlog and future ideas

## 12. Proposal and Planning Practice

Professional GitHub practice for non-committed ideas:

1. Open a lightweight proposal document in `docs/proposals/`.
2. Link the related issue.
3. Track status clearly (`draft`, `proposed`, `accepted`, `deferred`, `rejected`).
4. If accepted, migrate durable architecture decisions to `docs/ADR-light.md`.

This keeps implementation and strategy cleanly separated.

## 13. DOI and Citation

Current DOI references:

- Wrapper concept DOI: [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Wrapper release DOI: [10.5281/zenodo.18759479](https://doi.org/10.5281/zenodo.18759479)
- Ruleset concept DOI: [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Ruleset release DOI: [10.5281/zenodo.18154098](https://doi.org/10.5281/zenodo.18154098)

Primary citation metadata is stored in `CITATION.cff`.
