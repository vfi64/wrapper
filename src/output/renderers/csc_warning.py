from __future__ import annotations

import html


def render_control_layer_block_html(message: str, *, suffix_html: str = "") -> str:
    """Render deterministic CONTROL LAYER BLOCK warning box."""
    safe_msg = html.escape(str(message or ""))
    return (
        '<div class="csc-warning" style="background:#fee; border-color:#c00; color:#a00;">'
        "<b>CONTROL LAYER BLOCK:</b><br>"
        + safe_msg
        + str(suffix_html or "")
        + "</div>"
    )


def render_strict_block_html(violation_items_html: str) -> str:
    """Render strict-block warning details box."""
    return (
        "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
        "<summary>⛔ STRICT BLOCK (hard violations)</summary>"
        "<div class='csc-details'>"
        "<p>The model response was blocked by the wrapper because hard rule violations remained after repair/enforcement.</p>"
        "<ul>"
        + str(violation_items_html or "")
        + "</ul>"
        "<p><i>(Content withheld by wrapper)</i></p>"
        "</div></details>"
    )


def render_strict_warn_banner_html(violation_items_html: str) -> str:
    """Render strict-warn banner above normal content."""
    return (
        "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
        "<summary>⚠️ RULE VIOLATION DETECTED (strict_warn)</summary>"
        "<div class='csc-details'>"
        "<p>The following response still contains hard rule violations after repair/enforcement:</p>"
        "<ul>"
        + str(violation_items_html or "")
        + "</ul>"
        "</div></details><hr>"
    )


def render_cross_version_guard_html(active_version: str) -> str:
    """Render deterministic cross-version guard alert."""
    safe_active_version = html.escape(str(active_version or ""))
    return (
        "<div class='csc-warning' style='background:#fff7ed; border:1px solid #fb923c; padding:10px; "
        "border-radius:10px; margin:8px 0; color:#9a3412;'>"
        "<b>CONTROL LAYER ALERT</b><br><b>Cross-Version Guard</b>: "
        f"Ignored foreign version token(s) in user input. Active: {safe_active_version}."
        "</div>"
    )
