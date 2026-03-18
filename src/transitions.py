from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from intents import (
    ActivateDynamicOneShot,
    ComplianceViolation,
    EnterSciRecursion,
    Intent,
    ProcessModelResponse,
    SelectProfile,
    SetAnchorAuto,
    SetOverlay,
    ToggleColor,
    ToggleComm,
    ToggleSCI,
)
from state import WrapperState, resolve_profile_color_default


@dataclass(frozen=True)
class TransitionResult:
    state: WrapperState
    command_strings: List[str] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)


def _default_profile(data: Dict[str, Any]) -> str:
    default = str(data.get("default_profile", "Standard") or "Standard")
    profiles = data.get("profiles") or {}
    if isinstance(profiles, dict) and default in profiles:
        return default
    return "Standard"


def _profile_exists(data: Dict[str, Any], profile: str) -> bool:
    profiles = data.get("profiles") or {}
    return isinstance(profiles, dict) and profile in profiles


def _profile_overlay(data: Dict[str, Any], profile: str) -> str:
    try:
        profiles = data.get("profiles") or {}
        p = (profiles.get(profile) or {}) if isinstance(profiles, dict) else {}
        ov = str(p.get("mode_overlay", "") or "").strip()
        if ov.lower() in {"strict", "explore"}:
            return ov.capitalize()
        return ""
    except Exception:
        return ""


def _max_sci_depth(data: Dict[str, Any]) -> int:
    try:
        return int((((data.get("sci") or {}).get("recursive_sci") or {}).get("max_depth", 2)) or 2)
    except Exception:
        return 2


def _apply_comm_start_defaults(state: WrapperState, data: Dict[str, Any]) -> WrapperState:
    default_profile = _default_profile(data)
    new_state = WrapperState(**state.__dict__)
    new_state.comm_active = True
    if _profile_exists(data, default_profile):
        new_state.active_profile = default_profile
        new_state.overlay = _profile_overlay(data, default_profile)
        new_state.color = resolve_profile_color_default(data, default_profile, fallback=new_state.color)
        new_state.sci_pending_turns = 0
        if default_profile not in {"Expert", "Sparring"}:
            new_state.sci_active = False
            new_state.sci_pending = False
            new_state.sci_variant = ""
    return new_state


def _norm_policy(value: str) -> str:
    p = (value or "").strip().lower()
    if p in {"audit_only", "strict_warn", "strict_block"}:
        return p
    return "audit_only"


def _norm_severity(value: str) -> str:
    s = (value or "").strip().lower()
    if s in {"critical", "major", "minor"}:
        return s
    return "major"


def _warning_prefix(violations: list[ComplianceViolation]) -> str:
    lines = ["⚠️ Compliance Warnung:", ""]
    for v in violations:
        sev = _norm_severity(v.severity).upper()
        msg = (v.message or "").strip()
        if msg:
            lines.append(f"- [{sev}] {v.rule}: {msg}")
        else:
            lines.append(f"- [{sev}] {v.rule}")
    lines.append("")
    return "\n".join(lines)


def _blocked_text(violations: list[ComplianceViolation]) -> str:
    lines = ["⛔ Antwort blockiert.", "", "Das Modell hat zwingende Regeln verletzt:"]
    for v in violations:
        sev = _norm_severity(v.severity).upper()
        msg = (v.message or "").strip()
        if msg:
            lines.append(f"- [{sev}] {v.rule}: {msg}")
        else:
            lines.append(f"- [{sev}] {v.rule}")
    return "\n".join(lines)


def apply_intent(state: WrapperState, intent: Intent, ruleset_data: Dict[str, Any]) -> TransitionResult:
    data = ruleset_data if isinstance(ruleset_data, dict) else {}
    new_state = WrapperState(**state.__dict__)
    events: List[Dict[str, Any]] = []

    if isinstance(intent, ToggleComm):
        if intent.turn_on:
            new_state = _apply_comm_start_defaults(new_state, data)
            events.append({"event": "comm_started", "profile": new_state.active_profile})
        else:
            new_state.comm_active = False
            # Strict Comm-off reset: clear SCI/one-shot/transient governance state to avoid leaks.
            new_state.sci_pending = False
            new_state.sci_active = False
            new_state.sci_variant = ""
            new_state.sci_pending_turns = 0
            new_state.sci_recursion_one_shot = False
            new_state.sci_recursion_parent_variant = ""
            new_state.dynamic_one_shot_active = False
            new_state.dynamic_nudge = ""
            new_state.qc_overrides = {}
            events.append({"event": "comm_stopped"})
        return TransitionResult(new_state, [("Comm Start" if intent.turn_on else "Comm Stop")], events)

    if isinstance(intent, ToggleSCI):
        if intent.turn_on:
            new_state.sci_pending = True
            new_state.sci_pending_turns = 0
            events.append({"event": "sci_pending_enabled"})
        else:
            new_state.sci_pending = False
            new_state.sci_active = False
            new_state.sci_variant = ""
            new_state.sci_pending_turns = 0
            events.append({"event": "sci_disabled"})
        return TransitionResult(new_state, [("SCI on" if intent.turn_on else "SCI off")], events)

    if isinstance(intent, SetOverlay):
        new_state.overlay = intent.value
        events.append({"event": "overlay_set", "overlay": intent.value or "off"})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, ToggleColor):
        new_state.color = "on" if intent.turn_on else "off"
        events.append({"event": "color_set", "color": new_state.color})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, SelectProfile):
        if _profile_exists(data, intent.profile):
            new_state.active_profile = intent.profile
            new_state.color = resolve_profile_color_default(data, intent.profile, fallback=new_state.color)
            new_state.qc_overrides = {}
            new_state.sci_pending_turns = 0
            if intent.profile not in {"Expert", "Sparring"}:
                new_state.sci_active = False
                new_state.sci_pending = False
                new_state.sci_variant = ""
            events.append({"event": "profile_selected", "profile": intent.profile})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, SetAnchorAuto):
        new_state.anchor_auto = bool(intent.enabled)
        new_state.anchor_auto_user_override = True
        if intent.enabled:
            new_state.anchor_force_next = False
        else:
            new_state.anchor_force_next = False
        events.append({"event": "anchor_auto_set", "enabled": bool(intent.enabled)})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, EnterSciRecursion):
        max_depth = _max_sci_depth(data)
        cur = int(new_state.sci_recursion_depth or 0)
        if cur < max_depth:
            new_state.sci_recursion_parent_variant = new_state.sci_variant or ""
            new_state.sci_recursion_depth = cur + 1
            new_state.sci_recursion_one_shot = True
            if not new_state.sci_active:
                new_state.sci_active = True
            if not (new_state.sci_variant or "").strip():
                new_state.sci_variant = "A"
            events.append({"event": "sci_recursion_entered", "depth": new_state.sci_recursion_depth})
        else:
            events.append({"event": "sci_recursion_blocked", "depth": cur, "max_depth": max_depth})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, ActivateDynamicOneShot):
        new_state.dynamic_one_shot_active = True
        new_state.dynamic_nudge = "one-shot"
        events.append({"event": "dynamic_one_shot_armed"})
        return TransitionResult(new_state, [], events)

    if isinstance(intent, ProcessModelResponse):
        violations = list(intent.violations or [])
        policy = _norm_policy(str(new_state.enforcement_policy or "audit_only"))
        enabled = bool(new_state.enforcement_enabled)
        blocked_set = {
            _norm_severity(s)
            for s in (new_state.blocked_severities or ["critical"])
        } or {"critical"}

        final_text = str(intent.raw_text or "")
        action = "pass"
        if enabled and violations:
            if policy == "strict_block":
                has_blocking = any(_norm_severity(v.severity) in blocked_set for v in violations)
                if has_blocking:
                    final_text = _blocked_text(violations)
                    action = "blocked"
            if action != "blocked" and policy == "strict_warn":
                final_text = _warning_prefix(violations) + final_text
                action = "warned"
            if action == "pass":
                action = "audited"

        events.append(
            {
                "event": "response_enforcement_evaluated",
                "policy": policy,
                "enabled": enabled,
                "violations_count": len(violations),
                "action": action,
            }
        )
        return TransitionResult(new_state, [final_text], events)

    return TransitionResult(new_state, [], events)
