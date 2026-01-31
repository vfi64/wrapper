# Comm-SCI-Wrapper v136

**Deterministic single-file Python wrapper for auditable usage of LLMs under the normative governance specification of Comm-SCI-Control (v19.6.8).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Comm-SCI-Control](https://img.shields.io/badge/Comm--SCI--Control-v19.6.8-orange.svg)](https://github.com/vfi64/Comm-SCI-Control)
[![DOI](https://zenodo.org/badge/1137466025.svg)](https://doi.org/10.5281/zenodo.18445673)
[![Zenodo (ruleset)](https://zenodo.org/badge/DOI/10.5281/zenodo.18108395.svg)](https://doi.org/10.5281/zenodo.18108395)
[![tests](https://github.com/vfi64/wrapper/actions/workflows/tests.yml/badge.svg)](https://github.com/vfi64/wrapper/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> Core idea:  
> **Comm-SCI-Control** = governance ruleset (JSON, *no runtime, no code execution*).  
> **Wrapper-136.py** = technical enforcement and session layer (state machine, UI, export, audit, provider switch).

Reference ruleset: https://github.com/vfi64/Comm-SCI-Control
Wrapper DOI: https://doi.org/10.5281/zenodo.18445673

Recommended citation (ruleset): https://doi.org/10.5281/zenodo.18108395

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install pywebview requests markdown pytest bleach cryptography google-genai
   ```
2. Run the wrapper:
   ```bash
   python3 Wrapper-136.py
   ```
3. Run the regression suite:
   ```bash
   python3 -m pytest -vv -s --tb=long Test-136.py
   ```
4. In the chat window, try:
   - `Comm Help`
   - `Comm State`
   - `QC Override` (opens the slider UI for temporary QC adjustments, if enabled)

---

## Concept: Governance vs. Runtime

**Comm-SCI-Control** is a *pure specification* (JSON). It defines *what should happen*: profiles, commands, structured workflows (SCI), QC metrics with deviation reporting, an uncertainty taxonomy, and post-answer self-audits.

This wrapper is the *executor*: it maintains an external state machine and produces structured logs so that governance becomes **visible, testable, and auditable** (rather than “prompt vibes”).

---

## Key Features (v136)

Regression suite: **67 tests** (offline, deterministic; no GUI, no real provider calls).

- **Audit v2 export** (includes `trace_id`, provider/model context, ruleset hash, wrapper file hash, and an event stream)
- **QC override UI** (6 sliders) and deterministic QC-delta handling
- **Chat log replay**, and optional **load & fork** for new sessions
- **Multi-provider support** (Gemini / OpenRouter / Hugging Face catalog), config-driven
- **Guardrails** (e.g., “no network calls for UI-only actions”) + rate limiting
- **HTML sanitization** via `bleach` (prevents unsafe HTML in rendered output)
- **Optional encrypted key storage** (Fernet / `cryptography`)
- **Offline regression suite** in `Test-136.py` (no GUI, no real provider calls)

---

## What this repository contains

- [`Wrapper-136.py`](https://github.com/vfi64/wrapper/blob/main/Wrapper-136.py) — the runtime (UI, session state, guardrails, exports, audit)
- [`Test-136.py`](https://github.com/vfi64/wrapper/blob/main/Test-136.py) — the offline regression gate (**no GUI, no real provider calls**)
- `JSON/Comm-SCI-v19.6.8.json` — the ruleset (**source of truth**)

---

## Directory Structure

Expected layout:

```text
.
├── Wrapper-136.py
├── Test-136.py
├── JSON/
│   ├── Comm-SCI-v19.6.8.json
│   └── Comm-SCI-API-Keys.json        # local only, do NOT commit
├── Config/
│   └── Comm-SCI-Config.json
└── Logs/
    ├── Audit/
    └── Chats/
```

---

## Dependencies (short list)

- `pywebview` — UI/WebView
- `requests` — HTTP (provider-dependent)
- `markdown` — rendering (fallback possible)
- `bleach` — HTML sanitization
- `pytest` — tests
- `google-genai` — Gemini client (if used)
- `cryptography` — optional encrypted key storage

> Note: `pywebview` requires OS-specific WebView backends (macOS: WebKit/Cocoa; Windows: WebView2; Linux: GTK/QT — depending on your installation).

---

## Usage (short)

Common commands (from Comm-SCI-Control):

- `Comm Start` / `Comm Stop`
- `Comm Help`
- `Comm State`
- `Profile Expert`
- `SCI on` / `SCI menu`
- `Strict on` / `Strict off`
- `Color on` (if enabled in the ruleset)

Wrapper/UI actions:

- `QC Override` — opens a slider UI for temporary QC adjustments (if enabled in config/UI)

State changes are surfaced via **status + audit events**, and chat/audit logs can be exported.

---

## Known Limitations (honest)

- **Provider APIs evolve:** the wrapper may need small updates when providers change endpoints/SDKs (it follows OpenAI-compatible conventions where possible).
- **No cloud log sharing yet:** exports are local by design.
- **Governance improves discipline, not truth:** it increases transparency and reproducibility, but cannot guarantee factual correctness.

---

## Contribution

Issues and PRs are welcome, especially for:
- provider adapter improvements (while keeping “no network on UI-only actions”)
- tests and fixtures (pytest must stay green)
- documentation and minimal examples

---

## Licensing / Citation

- **Wrapper code:** Apache License 2.0 (see `LICENSE`).
- **Comm-SCI-Control (ruleset):** cite an archived Zenodo release (see DOI above). Licensing and attribution for the ruleset are governed by the upstream Comm-SCI-Control project.