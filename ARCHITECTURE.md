# Design Rationale: Wrapper-Agnostic Governance

## 1. System model (roles)
Comm-SCI treats an LLM as a **probabilistic instrument** whose behavior is governed by an explicit, canonical rule system.

There are exactly two authoritative layers:

1) **Normative layer (Comm-SCI JSON ruleset)**
   - Defines *what must happen* (command tokens, state transitions, output contracts, QC policy, SCI workflows, uncertainty taxonomy U1–U6, verification routes).
   - Is the **single Source of Truth**.

2) **Execution/Observation layer (Python wrapper)**
   - Implements only:
     - standalone command parsing (exact token match),
     - deterministic state management,
     - UI rendering (help/state/menus),
     - audit and logging of observable artifacts.
   - Must remain **semantically neutral**.

## 2. Non-goals (hard constraints)
The wrapper MUST NOT:
- produce epistemic judgments (e.g., deciding that a query is ambiguous → U6),
- enforce uncertainty labeling via heuristics,
- post-process or “repair” QC scores or evidence values,
- rewrite model content to meet governance requirements.

Reason: these behaviors would create a second, unverified decision layer that competes with the model’s governed output.

## 3. Integrity properties
### 3.1 Epistemic integrity
Uncertainty labels (U1–U6) are **semantic content**, not UI metadata.
They must be authored by the model under governance control.
If the wrapper generates them, uncertainty becomes an implementation artifact rather than a governed disclosure.

### 3.2 Audit transparency (no silent adaptation)
All governance-relevant signals must be attributable:
- either to the canonical JSON ruleset,
- or to explicit, user-triggered commands.

Any wrapper-driven modification to QC values or uncertainty markers violates “No Silent Adaptation” because it changes the visible outcome without a traceable governance cause.

### 3.3 Cross-model comparability
The same ruleset must be usable to compare different providers/models.
Wrapper heuristics bias outcomes and destroy comparability.

## 4. What enforcement means in this project
“Enforcement” is **structural**, not semantic:
- The wrapper enforces token exactness for commands.
- It enforces deterministic state transitions.
- It ensures required UI/menu outputs are rendered when governance state demands it.
- It audits *observable conformance* signals (presence/format), not the truth of content.

## 5. Reviewer checklist
When reviewing changes, validate:
- No new heuristic classification of user input into U-labels.
- No wrapper-side QC score rewriting/capping.
- No output rewriting that changes semantic content.
- Command tokens remain validated against the loaded canonical JSON.
- State changes are traceable to explicit standalone commands only.