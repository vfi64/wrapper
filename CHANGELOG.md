# Changelog – Comm-SCI-Wrapper

All notable changes to **Comm-SCI-Wrapper** are documented in this file.  
Format: newest versions first. Patch releases are additive and should be safe to upgrade.

> This repository provides the *runtime / executor* for the governance ruleset **Comm-SCI-Control**.  
> The ruleset itself is maintained separately: https://github.com/vfi64/Comm-SCI-Control

---

## [136] – 2026-01-31
### Audit v2 Export as Default (UI + Command Path)

#### Fixed
- **Audit export always uses Audit v2**:
  - Panel export and `Comm Audit` now write **Audit v2** (includes `trace_id`, provider/model fields, and other v2 metadata).
  - Removes the previous “v2 only on exception” behavior, which could silently keep producing Audit v1 in normal UI flows.

#### Changed
- **Export routing clarified**:
  - Chat log export remains unchanged.
  - Audit export path is explicitly routed to the v2 writer.

#### Tests
- **Regression suite remains green** (`Test-136.py`), with minimal adjustments where a specific “exported audit path” string was asserted.

---

## [135] – 2026-01-30
### Mixed Audit Export Paths (Known Issue)

#### Known issue (fixed in 136)
- `Comm Audit` and panel export still called the generic `export()` path, which wrote **Audit v1** by default.
- A v2 exporter existed but was effectively unused in the normal UI/command flow (fallback triggered only on exceptions).

---

## Compatibility Notes

- **Governance ruleset**: designed for `Comm-SCI-Control` **v19.6.8** (JSON Source of Truth).
- **Logging**:
  - Audit logs: v2 JSON (recommended / default since 136).
  - Chat logs: existing format (unchanged in 136).
- **CI / tests**: GitHub Actions workflow `.github/workflows/tests.yml` runs pytest on pushes/PRs.

