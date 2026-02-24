from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _norm_overlay(value: str) -> str:
    v = (value or "").strip()
    if v.lower() in {"", "off", "none"}:
        return ""
    if v.lower() == "strict":
        return "Strict"
    if v.lower() == "explore":
        return "Explore"
    return v


def _default_overlay_from_ruleset(ruleset: Dict[str, Any], profile: str) -> str:
    try:
        profiles = (ruleset.get("profiles") or {}) if isinstance(ruleset, dict) else {}
        pdef = (profiles.get(profile) or {}) if isinstance(profiles, dict) else {}
        return _norm_overlay(str((pdef.get("mode_overlay") or "")))
    except Exception:
        return ""


def _norm_enforcement_policy(value: str) -> str:
    p = (value or "").strip().lower()
    if p in {"audit_only", "strict_warn", "strict_block"}:
        return p
    return "audit_only"


def _norm_blocked_severities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["critical"]
    out: list[str] = []
    for item in value:
        s = str(item or "").strip().lower()
        if s in {"critical", "major", "minor"} and s not in out:
            out.append(s)
    return out or ["critical"]


@dataclass
class WrapperState:
    comm_active: bool = True
    active_profile: str = "Standard"
    overlay: str = ""
    color: str = "on"
    conversation_language: str = "de"
    answer_language: str = "de"
    sci_pending: bool = False
    sci_variant: str = ""
    sci_active: bool = False
    sci_pending_turns: int = 0

    anchor_auto: bool = True
    anchor_force_next: bool = False
    anchor_auto_user_override: bool = False

    dynamic_one_shot_active: bool = False
    dynamic_nudge: str = ""

    sci_recursion_depth: int = 0
    sci_recursion_parent_variant: str = ""
    sci_recursion_one_shot: bool = False

    qc_overrides: Dict[str, int] = field(default_factory=dict)
    enforcement_policy: str = "audit_only"
    enforcement_enabled: bool = True
    blocked_severities: list[str] = field(default_factory=lambda: ["critical"])


def state_from_runtime(runtime_state: Any) -> WrapperState:
    if runtime_state is None:
        return WrapperState()
    return WrapperState(
        comm_active=bool(getattr(runtime_state, "comm_active", False)),
        active_profile=str(getattr(runtime_state, "active_profile", "Standard") or "Standard"),
        overlay=_norm_overlay(str(getattr(runtime_state, "overlay", "") or "")),
        color=str(getattr(runtime_state, "color", "on") or "on"),
        conversation_language=str(getattr(runtime_state, "conversation_language", "de") or "de"),
        answer_language=str(getattr(runtime_state, "answer_language", "de") or "de"),
        sci_pending=bool(getattr(runtime_state, "sci_pending", False)),
        sci_variant=str(getattr(runtime_state, "sci_variant", "") or ""),
        sci_active=bool(getattr(runtime_state, "sci_active", False)),
        sci_pending_turns=int(getattr(runtime_state, "sci_pending_turns", 0) or 0),
        anchor_auto=bool(getattr(runtime_state, "anchor_auto", True)),
        anchor_force_next=bool(getattr(runtime_state, "anchor_force_next", False)),
        anchor_auto_user_override=bool(getattr(runtime_state, "anchor_auto_user_override", False)),
        dynamic_one_shot_active=bool(getattr(runtime_state, "dynamic_one_shot_active", False)),
        dynamic_nudge=str(getattr(runtime_state, "dynamic_nudge", "") or ""),
        sci_recursion_depth=int(getattr(runtime_state, "sci_recursion_depth", 0) or 0),
        sci_recursion_parent_variant=str(getattr(runtime_state, "sci_recursion_parent_variant", "") or ""),
        sci_recursion_one_shot=bool(getattr(runtime_state, "sci_recursion_one_shot", False)),
        qc_overrides=dict(getattr(runtime_state, "qc_overrides", {}) or {}),
        enforcement_policy=_norm_enforcement_policy(str(getattr(runtime_state, "enforcement_policy", "audit_only") or "audit_only")),
        enforcement_enabled=bool(getattr(runtime_state, "enforcement_enabled", True)),
        blocked_severities=_norm_blocked_severities(getattr(runtime_state, "blocked_severities", ["critical"])),
    )


def apply_state_to_runtime(state: WrapperState, runtime_state: Any) -> None:
    if runtime_state is None:
        return
    for key, value in state.__dict__.items():
        try:
            setattr(runtime_state, key, value)
        except Exception:
            continue


def init_state_from_ruleset(
    ruleset: Optional[Dict[str, Any]],
    *,
    answer_language: str = "de",
    conversation_language: str = "de",
) -> WrapperState:
    data = ruleset if isinstance(ruleset, dict) else {}
    profiles = (data.get("profiles") or {}) if isinstance(data, dict) else {}
    default_profile = str(data.get("default_profile", "Standard") or "Standard")
    if not isinstance(profiles, dict) or default_profile not in profiles:
        default_profile = "Standard"

    overlay = _default_overlay_from_ruleset(data, default_profile)
    enforcement_cfg = data.get("enforcement") if isinstance(data.get("enforcement"), dict) else {}
    policy = _norm_enforcement_policy(
        str(
            (enforcement_cfg.get("policy") if isinstance(enforcement_cfg, dict) else "")
            or data.get("enforcement_policy")
            or "audit_only"
        )
    )
    enabled = bool(
        (enforcement_cfg.get("enabled") if isinstance(enforcement_cfg, dict) else True)
        if isinstance(enforcement_cfg, dict) else True
    )
    blocked = _norm_blocked_severities(
        (enforcement_cfg.get("blocked_severities") if isinstance(enforcement_cfg, dict) else ["critical"])
    )
    return WrapperState(
        comm_active=True,
        active_profile=default_profile,
        overlay=overlay,
        color="on",
        conversation_language=(conversation_language or "de"),
        answer_language=(answer_language or "de"),
        sci_pending=False,
        sci_variant="",
        sci_active=False,
        sci_pending_turns=0,
        enforcement_policy=policy,
        enforcement_enabled=enabled,
        blocked_severities=blocked,
    )
