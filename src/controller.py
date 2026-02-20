from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from intents import Intent, intent_from_command
from state import apply_state_to_runtime, state_from_runtime
from transitions import apply_intent


@dataclass(frozen=True)
class DispatchResult:
    applied: bool
    effects: List[str] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)


def dispatch_intent(
    *,
    intent: Intent,
    cmd: str,
    runtime_state: Any,
    ruleset_data: Optional[Dict[str, Any]] = None,
    mirror_callback: Optional[Callable[[], None]] = None,
) -> DispatchResult:
    token = (cmd or "").strip()
    if intent is None:
        return DispatchResult(applied=False, effects=[], audit_events=[])

    state_in = state_from_runtime(runtime_state)
    result = apply_intent(state_in, intent, ruleset_data if isinstance(ruleset_data, dict) else {})
    apply_state_to_runtime(result.state, runtime_state)

    try:
        if mirror_callback is not None:
            mirror_callback()
    except Exception:
        pass

    effects: List[str] = []
    if token == "Comm Stop":
        effects.append("recreate_session_without_governance")
    elif token == "Comm Start":
        effects.append("recreate_session_with_governance")
    else:
        effects.append("push_state_update")

    if token.startswith("Profile "):
        effects.append("profile_switch")

    return DispatchResult(applied=True, effects=effects, audit_events=list(result.audit_events or []))


def dispatch(
    *,
    cmd: str,
    runtime_state: Any,
    ruleset_data: Optional[Dict[str, Any]] = None,
    mirror_callback: Optional[Callable[[], None]] = None,
) -> DispatchResult:
    token = (cmd or "").strip()
    intent = intent_from_command(token)
    if intent is None:
        return DispatchResult(applied=False, effects=[], audit_events=[])
    return dispatch_intent(
        intent=intent,
        cmd=token,
        runtime_state=runtime_state,
        ruleset_data=ruleset_data,
        mirror_callback=mirror_callback,
    )
