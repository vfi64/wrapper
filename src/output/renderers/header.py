from __future__ import annotations

import html
from typing import Iterable


def normalize_sci_display(
    sci_variant: str,
    *,
    sci_pending: bool = False,
    off_label: str = "off",
    pending_label: str = "PENDING",
    pending_mode: str = "when_pending_and_unset",
    uppercase_non_off: bool = False,
) -> str:
    """Return deterministic SCI display token for status/header lines.

    pending_mode:
    - "never": never emit pending_label
    - "when_pending_and_unset": pending_label only when pending and variant is unset/off
    - "always_when_pending": pending_label whenever pending is true
    """
    raw = str(sci_variant or "").strip()
    raw_low = raw.lower()
    sci_is_unset = (not raw) or (raw_low in {"off", "none"})
    mode = str(pending_mode or "when_pending_and_unset").strip().lower()

    if mode == "always_when_pending" and bool(sci_pending):
        return str(pending_label or "PENDING")
    if mode == "when_pending_and_unset" and bool(sci_pending) and sci_is_unset:
        return str(pending_label or "PENDING")

    if sci_is_unset:
        return str(off_label or "off")

    if bool(uppercase_non_off):
        return raw.upper()
    return raw


def build_active_profile_status_line(
    *,
    profile: str,
    sci_variant: str,
    overlay: str,
    control_layer: str = "on",
    qc: str = "on",
    cgi: str = "on",
    color: str = "off",
    sci_pending: bool = False,
    off_label: str = "off",
    pending_label: str = "PENDING",
    pending_mode: str = "when_pending_and_unset",
    uppercase_sci_non_off: bool = False,
    color_force_off_profiles: Iterable[str] | None = (),
) -> str:
    """Build canonical Active-profile status header line."""
    prof = str(profile or "Standard")
    ov = str(overlay or "off")
    ctl = str(control_layer or "on")
    qc_v = str(qc or "on")
    cgi_v = str(cgi or "on")
    color_v = str(color or "off")

    try:
        force_profiles = {str(x) for x in (color_force_off_profiles or ())}
    except Exception:
        force_profiles = set()
    if force_profiles and prof in force_profiles:
        color_v = "off"

    sci_out = normalize_sci_display(
        sci_variant,
        sci_pending=bool(sci_pending),
        off_label=off_label,
        pending_label=pending_label,
        pending_mode=pending_mode,
        uppercase_non_off=bool(uppercase_sci_non_off),
    )
    return (
        f"Active profile: {prof} · SCI: {sci_out} · Overlay: {ov} · "
        f"Control Layer: {ctl} · QC: {qc_v} · CGI: {cgi_v} · Color: {color_v}"
    )


def build_comm_status_line(
    *,
    comm: str,
    profile: str,
    sci_variant: str,
    overlay: str,
    control_layer: str = "on",
    qc: str = "on",
    cgi: str = "on",
    color: str = "off",
    language_policy: str = "",
    sci_pending: bool = False,
    off_label: str = "OFF",
    pending_label: str = "PENDING",
    pending_mode: str = "always_when_pending",
    uppercase_sci_non_off: bool = True,
) -> str:
    """Build canonical Comm State status line."""
    sci_out = normalize_sci_display(
        sci_variant,
        sci_pending=bool(sci_pending),
        off_label=off_label,
        pending_label=pending_label,
        pending_mode=pending_mode,
        uppercase_non_off=bool(uppercase_sci_non_off),
    )

    line = (
        f"Comm: {str(comm or 'off')} · Active profile: {str(profile or 'Standard')} · SCI: {sci_out} · "
        f"Overlay: {str(overlay or 'off')} · Control Layer: {str(control_layer or 'on')} · "
        f"QC: {str(qc or 'on')} · CGI: {str(cgi or 'on')} · Color: {str(color or 'off')}"
    )
    if str(language_policy or "").strip():
        line += f" · Language policy: {str(language_policy)}"
    return line


def render_profile_switch_control_html(
    *,
    timestamp: str,
    header_line: str,
    qc_line: str,
    audit_line: str = "",
) -> str:
    """Render deterministic minimal profile-switch control output."""
    out = []
    out.append('<div class="comm-help comm-state">')
    out.append(f'<div class="help-status">{html.escape(str(header_line or ""))}</div>')
    if str(audit_line or "").strip():
        out.append(f"<div style='margin-top:8px'>{html.escape(str(audit_line))}</div>")
    out.append(f"<div style='margin-top:10px'>{html.escape(str(qc_line or ''))}</div>")
    out.append("</div>")
    out.append(f'<div class="ts-footer">Response at {html.escape(str(timestamp or ""))}</div>')
    return "\n".join(out)
