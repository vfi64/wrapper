from __future__ import annotations

import html
from typing import Sequence


def render_control_layer_alert_html(
    *,
    message: str,
    title: str = "CONTROL LAYER ALERT",
    severity: str = "error",
    tooltip_text: str = "",
    action_switch_free: bool = False,
) -> str:
    """Render deterministic Control-Layer alert details box."""
    safe = html.escape(str(message or ""))
    safe = safe.replace("\n", "<br>")

    sev = str(severity or "").lower()
    if sev == "error":
        style = "border: 1px solid #c00; background: #fee; color: #600;"
    elif sev == "warn":
        style = "border: 1px solid #f9ab00; background: #fff7e0; color: #3c2b00;"
    else:
        style = "border: 1px solid #999; background: #f5f5f5; color: #222;"

    action_html = ""
    if bool(action_switch_free):
        action_html = (
            "<br><br>"
            "<a href=\"#\" class=\"ctl-action action-next-free\" "
            "style=\"display:inline-block;padding:2px 8px;border:1px solid #888;"
            "border-radius:10px;background:#eee;font-family:monospace;text-decoration:none;color:inherit;\">"
            "Tip: choose another :free model</a>"
        )

    safe_title = html.escape(str(title or "CONTROL LAYER ALERT"))
    tip = str(tooltip_text or "").strip()
    tip_attr = f" data-u-title='{tip}' style='cursor:help;'" if tip else ""
    return (
        f"<details class='csc-warning' open style='{style}'{tip_attr}>"
        f"<summary{tip_attr}>⚠️ {safe_title}</summary>"
        f"<div class='csc-details'>{safe}{action_html}</div>"
        f"</details>"
    )


def render_repair_pass_banner_html(
    *,
    violations: Sequence[str] | None = None,
    tooltip_text: str = "",
    include_violation_list: bool = True,
) -> str:
    """Render deterministic repair-pass note banner."""
    tip = html.escape(str(tooltip_text or ""), quote=True)
    tip_attr = f" data-u-title='{tip}' style='cursor:help;'" if tip else ""
    base = (
        "<div class='control-layer-note csc-warning' style='border:1px solid #f59e0b; background:#fffbeb; padding:10px; "
        "border-radius:10px; margin:8px 0; color:#92400e;'>"
        f"<b{tip_attr}>CONTROL LAYER NOTE</b><br>One repair pass was applied for hard contract violations."
    )

    if bool(include_violation_list):
        items = "".join(
            f"<li class='control-layer-violation'>{html.escape(str(v or ''))}</li>"
            for v in list(violations or [])
        )
        if items:
            return (
                base
                + "<ul class='control-layer-violations' style='margin:6px 0 0 18px; padding:0;'>"
                + items
                + "</ul></div>"
            )
    return base + "</div>"
