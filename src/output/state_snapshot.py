from __future__ import annotations

from dataclasses import dataclass


def _norm_onoff(value, *, fallback: str = "off") -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    raw = str(value or "").strip().lower()
    if raw in {"on", "off"}:
        return raw
    fb = str(fallback or "off").strip().lower()
    return fb if fb in {"on", "off"} else "off"


def _norm_overlay(value) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"off", "none"}:
        return "off"
    return raw


def _norm_sci_variant(value) -> str:
    raw = str(value or "").strip()
    return raw or "off"


@dataclass(frozen=True)
class OutputStateSnapshot:
    """Minimal deterministic runtime snapshot for output rendering."""

    comm_active: bool = True
    active_profile: str = "Standard"
    sci_variant: str = "off"
    sci_pending: bool = False
    sci_active: bool = False
    overlay: str = "off"
    color: str = "off"
    control_layer: str = "on"
    qc: str = "on"
    cgi: str = "on"
    language_policy_mode: str = "production"
    anchor_auto: bool = True
    user_turns: int = 0
    dynamic_nudge: str = ""

    @staticmethod
    def from_runtime_state(gov_state) -> "OutputStateSnapshot":
        try:
            comm_active = bool(getattr(gov_state, "comm_active", True))
            profile = str(getattr(gov_state, "active_profile", "Standard") or "Standard")
            sci_variant = _norm_sci_variant(getattr(gov_state, "sci_variant", ""))
            sci_pending = bool(getattr(gov_state, "sci_pending", False))
            sci_active = bool(getattr(gov_state, "sci_active", False))
            overlay = _norm_overlay(getattr(gov_state, "overlay", ""))
            color = _norm_onoff(getattr(gov_state, "color", "off"), fallback="off")
            control_layer = _norm_onoff(getattr(gov_state, "control_layer", "on"), fallback="on")
            qc = _norm_onoff(getattr(gov_state, "qc", "on"), fallback="on")
            cgi = _norm_onoff(getattr(gov_state, "cgi", "on"), fallback="on")
            language_policy_mode = str(
                getattr(gov_state, "language_policy_mode", "production") or "production"
            ).strip() or "production"
            anchor_auto = bool(getattr(gov_state, "anchor_auto", True))
            user_turns = int(getattr(gov_state, "user_turns", 0) or 0)
            dynamic_nudge = str(getattr(gov_state, "dynamic_nudge", "") or "").strip()
        except Exception:
            return OutputStateSnapshot()
        return OutputStateSnapshot(
            comm_active=comm_active,
            active_profile=profile,
            sci_variant=sci_variant,
            sci_pending=sci_pending,
            sci_active=sci_active,
            overlay=overlay,
            color=color,
            control_layer=control_layer,
            qc=qc,
            cgi=cgi,
            language_policy_mode=language_policy_mode,
            anchor_auto=anchor_auto,
            user_turns=user_turns,
            dynamic_nudge=dynamic_nudge,
        )
