from __future__ import annotations


def qc_override_window_create_kwargs_plan(
    *,
    html_qc_override: str,
    js_api_obj,
    window_title: str = "Temporary QC override – Profile: ?",
) -> dict:
    """Build deterministic kwargs for the QC Override dialog window."""
    return {
        "kwargs": {
            "title": str(window_title),
            "html": html_qc_override,
            "width": 450,
            "height": 550,
            "resizable": False,
            "hidden": True,
            "on_top": True,
            "js_api": js_api_obj,
        },
        "clear_window_on_failure": True,
    }


def qc_override_show_plan(*, window_exists: bool) -> dict:
    if not bool(window_exists):
        return {
            "create_if_missing": True,
            "error_if_unavailable": "qc_win unavailable",
            "window_methods": (),
            "success_result": None,
        }
    return {
        "create_if_missing": False,
        "error_if_unavailable": "qc_win unavailable",
        "window_methods": ("show", "bring_to_front"),
        "success_result": {"ok": True},
    }


def qc_override_cancel_plan(*, window_exists: bool) -> dict:
    return {
        "window_methods": (("hide",) if bool(window_exists) else ()),
        "success_result": {"ok": True},
    }


def qc_override_apply_ui_plan(*, clean_overrides: dict, qc_window_exists: bool) -> dict:
    """Build deterministic UI/messaging plan after QC override values were applied."""
    clean = clean_overrides if isinstance(clean_overrides, dict) else {}
    disp = {
        "clarity": "Clarity",
        "brevity": "Brevity",
        "evidence": "Evidence",
        "empathy": "Empathy",
        "consistency": "Consistency",
        "neutrality": "Neutrality",
    }
    msg_parts = []
    for key in ("clarity", "brevity", "evidence", "empathy", "consistency", "neutrality"):
        if key in clean:
            msg_parts.append(f"{disp.get(key, key)}={clean[key]}")
    msg = "QC-Overrides gesetzt: " + (", ".join(msg_parts) if msg_parts else "(leer)")
    return {
        "history_message": msg,
        "main_ui_message": msg,
        "qc_window_methods": (("hide",) if bool(qc_window_exists) else ()),
        "success_result": {"ok": True, "overrides": dict(clean)},
        "warn_prefix": "[WARN] QC Override Apply failed: ",
    }


def qc_override_clear_ui_plan(*, qc_window_exists: bool) -> dict:
    msg = "QC-Overrides zurückgesetzt"
    return {
        "history_message": msg,
        "main_ui_message": msg,
        "qc_window_methods": (("hide",) if bool(qc_window_exists) else ()),
        "success_result": {"ok": True},
    }
