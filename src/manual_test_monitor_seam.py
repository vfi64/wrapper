from __future__ import annotations

import json


def manual_test_monitor_initial_state() -> dict:
    return {
        "scenario": "",
        "status": "idle",
        "summary": "-",
        "events": [],
    }


def manual_test_monitor_create_window_kwargs_plan(*, html_manual_test_monitor: str, js_api_obj) -> dict:
    return {
        "kwargs": {
            "title": "Comm-SCI Manual Test Monitor",
            "html": html_manual_test_monitor,
            "width": 760,
            "height": 700,
            "resizable": True,
            "hidden": True,
            "on_top": False,
            "js_api": js_api_obj,
        },
        "reset_state_before_create": True,
    }


def manual_test_monitor_bind_window_events_plan(
    *,
    window_exists: bool,
    has_events: bool,
    has_closed_event: bool,
) -> dict:
    return {
        "bind_closed": bool(window_exists and has_events and has_closed_event),
        "closed_handler_name": "on_manual_test_monitor_closed",
    }


def manual_test_monitor_show_plan(*, window_exists: bool) -> dict:
    return {
        "create_if_missing": not bool(window_exists),
        "error_if_unavailable": "manual_test_monitor_win unavailable",
        "show_methods": ("show",),
        "retry_after_show_failure": True,
        "clear_window_on_show_failure": True,
        "post_show_methods": ("bring_to_front",),
        "push_state_to_ui": True,
        "success_result": {"ok": True},
    }


def manual_test_monitor_hide_plan(*, window_exists: bool, has_hide: bool, has_minimize: bool) -> dict:
    if not bool(window_exists):
        return {
            "window_methods": (),
            "success_result": {"ok": True, "hidden": True, "skipped": True},
        }
    if bool(has_hide):
        methods = ("hide",)
    elif bool(has_minimize):
        methods = ("minimize",)
    else:
        methods = ()
    return {
        "window_methods": methods,
        "success_result": {"ok": True, "hidden": True},
    }


def manual_test_monitor_replace_js(state) -> str:
    st = state if isinstance(state, dict) else {}
    return f"mtmReplace({json.dumps(st, ensure_ascii=False)});"


def manual_test_monitor_reset_state_plan(*, state, payload) -> dict:
    p = payload if isinstance(payload, dict) else {}
    st = state if isinstance(state, dict) else {}
    st["scenario"] = str(p.get("scenario", "") or "")
    st["status"] = str(p.get("status", "running") or "running")
    st["summary"] = p.get("summary", "-")
    st["events"] = []
    return {
        "state": st,
        "js_code": manual_test_monitor_replace_js(st),
    }


def manual_test_monitor_append_state_plan(*, state, entry, max_events: int = 1000) -> dict:
    try:
        limit = int(max_events)
    except Exception:
        limit = 1000
    if limit < 1:
        limit = 1

    st = state if isinstance(state, dict) else {"events": []}
    events = st.get("events")
    if not isinstance(events, list):
        events = []
        st["events"] = events

    e = entry if isinstance(entry, dict) else {"message": str(entry)}
    events.append(dict(e))
    if len(events) > limit:
        del events[:-limit]

    return {
        "state": st,
        "entry": e,
        "js_code": f"mtmAppend({json.dumps(e, ensure_ascii=False)});",
    }


def manual_test_monitor_set_header_state_plan(*, state, payload) -> dict:
    p = payload if isinstance(payload, dict) else {}
    st = state if isinstance(state, dict) else {}
    for key in ("scenario", "status", "summary"):
        if key in p:
            st[key] = p.get(key)
    header = {
        "scenario": st.get("scenario", ""),
        "status": st.get("status", ""),
        "summary": st.get("summary", "-"),
    }
    return {
        "state": st,
        "header": header,
        "js_code": f"mtmSetHeader({json.dumps(header, ensure_ascii=False)});",
    }
