# Runtime Use Cases

## When JSON-only is often enough

JSON-only chat usage is often sufficient for:

- first conceptual tests,
- light exploratory workflows,
- didactic demonstration of command logic.

## When wrapper runtime has a clear advantage

The wrapper is strongly preferable for:

- reproducible runs,
- stricter command/state contracts,
- diagnostics and audit trails,
- model comparison under stable execution logic.

## Typical scenarios

1. Repeated benchmark prompts across models.
2. Long sessions where drift risk must stay visible.
3. QA checks before public release.
4. Teaching/demo sessions where runtime control must be transparent.

## Core distinction

- Ruleset: how answers should behave.
- Wrapper runtime: how that behavior is enforced and observed.
