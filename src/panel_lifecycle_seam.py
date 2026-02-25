from __future__ import annotations

try:
    import panel_html_source as _panel_html_source_mod  # type: ignore
except Exception:
    _panel_html_source_mod = None  # type: ignore

try:
    import panel_bootstrap_state as _panel_bootstrap_state_mod  # type: ignore
except Exception:
    _panel_bootstrap_state_mod = None  # type: ignore

try:
    import panel_window_fallback as _panel_window_fallback_mod  # type: ignore
except Exception:
    _panel_window_fallback_mod = None  # type: ignore


def _fallback_embedded_html(html_panel_embedded, html_panel) -> str:
    if isinstance(html_panel_embedded, str) and html_panel_embedded:
        return html_panel_embedded
    if isinstance(html_panel, str):
        return html_panel
    return ""


def _coerce_int(value, default) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def panel_window_geometry_plan(panel_geom) -> dict:
    """Normalize persisted panel geometry into safe on-screen window bounds."""
    geom = panel_geom if isinstance(panel_geom, dict) else {}

    panel_x = _coerce_int(geom.get("x", 1100), 1100)
    panel_y = _coerce_int(geom.get("y", 0), 0)
    panel_w = _coerce_int(geom.get("width", 340), 340)
    panel_h = _coerce_int(geom.get("height", 1000), 1000)

    if panel_w < 250:
        panel_w = 250
    if panel_h < 300:
        panel_h = 300
    if panel_x < 0 or panel_x > 5000:
        panel_x = 50
    if panel_y < 0 or panel_y > 3000:
        panel_y = 50

    return {
        "x": panel_x,
        "y": panel_y,
        "width": panel_w,
        "height": panel_h,
    }


def panel_create_window_kwargs_plan(
    *,
    panel_geom,
    panel_window_title: str,
    panel_html: str,
    js_api_obj,
) -> dict:
    """Build deterministic kwargs for the panel `webview.create_window(...)` call.

    The actual pywebview call stays in the monolith; this helper only prepares inputs.
    """
    g = panel_window_geometry_plan(panel_geom)
    kwargs = {
        "title": panel_window_title,
        "html": panel_html,
        "js_api": js_api_obj,
        "width": int(g["width"]),
        "height": int(g["height"]),
        "on_top": False,
        "x": int(g["x"]),
        "y": int(g["y"]),
    }
    return {
        "geometry": g,
        "kwargs": kwargs,
        "precreate_hidden": True,
    }


def panel_show_plan(*, panel_window_exists: bool) -> dict:
    if not bool(panel_window_exists):
        return {
            "action": "create_panel",
            "wait_bootstrap_before_show": False,
            "window_methods": (),
            "panel_hidden": None,
        }
    return {
        "action": "show_existing",
        "wait_bootstrap_before_show": True,
        "window_methods": ("show", "restore", "focus"),
        "panel_hidden": False,
    }


def panel_hide_plan(*, panel_window_exists: bool, has_hide: bool, has_minimize: bool) -> dict:
    if not bool(panel_window_exists):
        return {
            "action": "noop",
            "remember_geometry": False,
            "panel_hidden": None,
            "clear_panel_window": False,
        }
    if bool(has_hide):
        action = "hide"
        panel_hidden = True
        clear_panel_window = False
    elif bool(has_minimize):
        action = "minimize"
        panel_hidden = True
        clear_panel_window = False
    else:
        action = "destroy"
        panel_hidden = False
        clear_panel_window = True
    return {
        "action": action,
        "remember_geometry": True,
        "panel_hidden": panel_hidden,
        "clear_panel_window": clear_panel_window,
    }


def panel_ensure_visible_plan(*, panel_window_exists: bool, panel_hidden: bool) -> dict:
    if not bool(panel_window_exists):
        return {
            "action": "create_panel",
            "window_methods": (),
        }
    if bool(panel_hidden):
        return {
            "action": "show_panel",
            "window_methods": (),
        }
    return {
        "action": "focus_existing",
        "window_methods": ("restore", "focus"),
    }


def panel_settings_toggle_plan(*, panel_window_exists: bool, panel_hidden: bool) -> dict:
    if not bool(panel_window_exists):
        return {"action": "create_panel"}
    if bool(panel_hidden):
        return {"action": "show_panel"}
    return {"action": "hide_panel"}


def panel_on_closing_plan(*, panel_window_exists: bool, has_hide: bool) -> dict:
    """Plan the close-intercept behavior (hide instead of destroy when possible)."""
    return {
        "call_hide_panel": True,
        "fallback_action": ("direct_hide" if bool(panel_window_exists and has_hide) else "noop"),
        "fallback_sets_panel_hidden": bool(panel_window_exists and has_hide),
        "return_value": False,  # cancel close on backends that support it
    }


def panel_bind_window_events_plan(
    *,
    window_exists: bool,
    has_events: bool,
    has_closing_event: bool,
    has_closed_event: bool,
) -> dict:
    """Plan which panel lifecycle events should be bound on the given backend/window."""
    if not bool(window_exists):
        return {
            "bind_closing": False,
            "bind_closed": False,
            "closing_handler_name": "on_panel_closing",
            "closed_handler_name": "on_panel_closed",
        }
    if not bool(has_events):
        return {
            "bind_closing": False,
            "bind_closed": False,
            "closing_handler_name": "on_panel_closing",
            "closed_handler_name": "on_panel_closed",
        }
    return {
        "bind_closing": bool(has_closing_event),
        "bind_closed": bool(has_closed_event),
        "closing_handler_name": "on_panel_closing",
        "closed_handler_name": "on_panel_closed",
    }


def panel_rebuild_plan(*, reason: str, panel_window_exists: bool) -> dict:
    """Plan the deterministic control flow of `_rebuild_panel(...)`.

    pywebview calls remain in the monolith; this returns only decisions/ordering/messages.
    """
    why = str(reason or "reload")
    return {
        "remember_geometry_before_destroy": bool(panel_window_exists),
        "destroy_old_window": bool(panel_window_exists),
        "reset_panel_window": True,
        "reset_panel_hidden": False,
        "recreate_panel": True,
        # Keep historical order (focus -> restore) to avoid behavior drift.
        "post_create_window_methods": ("focus", "restore"),
        "success_main_message": f"Panel rebuilt ({why}).",
        "failure_main_message_prefix": "Panel rebuild failed: ",
    }


def panel_window_html_plan(
    *,
    force_embedded_html: bool,
    html_panel,
    html_panel_embedded,
    panel_html_asset_meta,
) -> tuple[str, str]:
    if _panel_html_source_mod is not None:
        try:
            return _panel_html_source_mod.select_panel_html_for_window(
                force_embedded_html=bool(force_embedded_html),
                html_panel=html_panel,
                html_panel_embedded=html_panel_embedded,
                panel_html_asset_meta=panel_html_asset_meta,
            )
        except Exception:
            pass

    if bool(force_embedded_html):
        return _fallback_embedded_html(html_panel_embedded, html_panel), "embedded"

    txt = html_panel if isinstance(html_panel, str) and html_panel else _fallback_embedded_html(html_panel_embedded, html_panel)
    meta = panel_html_asset_meta if isinstance(panel_html_asset_meta, dict) else {}
    src = str(meta.get("source") or "embedded")
    if not txt:
        src = "embedded"
    return txt, ("external" if src == "external" else "embedded")


def panel_bootstrap_probe_plan(source: str, *, now_iso: str | None) -> dict:
    src = str(source or "embedded")
    if _panel_bootstrap_state_mod is not None:
        try:
            st = _panel_bootstrap_state_mod.panel_bootstrap_probe_state(src, now_iso=now_iso)
        except Exception:
            st = None
    else:
        st = None

    if not isinstance(st, dict):
        st = {
            "status": ("pending" if src == "external" else "skipped"),
            "source": src,
            "reason": "",
            "created_at": now_iso,
            "reported_at": None,
        }

    return {
        "source": src,
        "event_action": ("clear" if src == "external" else "set"),
        "bootstrap_state": st,
    }


def panel_window_html_and_probe_plan(
    *,
    force_embedded_html: bool,
    html_panel,
    html_panel_embedded,
    panel_html_asset_meta,
    now_iso: str | None,
) -> dict:
    txt, src = panel_window_html_plan(
        force_embedded_html=force_embedded_html,
        html_panel=html_panel,
        html_panel_embedded=html_panel_embedded,
        panel_html_asset_meta=panel_html_asset_meta,
    )
    probe = panel_bootstrap_probe_plan(src, now_iso=now_iso)
    probe["html"] = txt
    return probe


def panel_bootstrap_ready_and_reason(state) -> tuple[bool, str | None]:
    if _panel_bootstrap_state_mod is not None:
        try:
            if _panel_bootstrap_state_mod.panel_bootstrap_is_runtime_ready(state):
                return True, None
            return False, _panel_bootstrap_state_mod.panel_bootstrap_fallback_reason(state)
        except Exception:
            pass

    if not isinstance(state, dict):
        return True, None
    status = str(state.get("status") or "")
    if str(state.get("source") or "embedded") != "external" or status == "passed":
        return True, None
    reason = str(state.get("reason") or "").strip()
    if not reason:
        reason = ("runtime_selftest_timeout" if status == "pending" else "runtime_selftest_failed")
    return False, reason


def panel_bootstrap_wait_plan(state, *, timeout_s, default_timeout_s: float = 2.5) -> dict:
    ready, reason = panel_bootstrap_ready_and_reason(state)
    if ready:
        return {
            "ready": True,
            "reason": None,
            "wait_seconds": 0.0,
            "should_wait": False,
        }

    if _panel_bootstrap_state_mod is not None:
        try:
            wait_s = _panel_bootstrap_state_mod.panel_bootstrap_timeout_seconds(
                timeout_s, default=default_timeout_s
            )
        except Exception:
            wait_s = float(default_timeout_s)
    else:
        try:
            wait_s = max(0.0, float(timeout_s))
        except Exception:
            wait_s = float(default_timeout_s)

    return {
        "ready": False,
        "reason": reason,
        "wait_seconds": float(wait_s),
        "should_wait": bool(isinstance(state, dict) and str(state.get("status") or "") == "pending"),
    }


def panel_embedded_fallback_swap_plan(
    *,
    state,
    reason: str,
    now_iso: str | None,
    old_window_exists: bool,
    ignore_count,
) -> dict:
    why = str(reason or "runtime_selftest_failed")

    marked = False
    if _panel_bootstrap_state_mod is not None:
        try:
            st = _panel_bootstrap_state_mod.panel_bootstrap_mark_failed_for_fallback(
                state,
                reason=why,
                now_iso=now_iso,
            )
            marked = True
        except Exception:
            st = None
    else:
        st = None

    if not marked:
        if not isinstance(state, dict):
            st = {}
        else:
            st = state
        st["status"] = "failed"
        st["reason"] = why
        st["source"] = "external"
        st["reported_at"] = now_iso

    if _panel_window_fallback_mod is not None:
        try:
            recreate = _panel_window_fallback_mod.panel_embedded_fallback_recreate_plan(
                old_window_exists=bool(old_window_exists),
                ignore_count=ignore_count,
            )
        except Exception:
            recreate = None
    else:
        recreate = None

    if not isinstance(recreate, dict):
        recreate = {
            "clear_panel_window": True,
            "panel_hidden": False,
            "force_embedded_html": True,
            "next_ignore_count": 1,
        }

    return {
        "bootstrap_state": st if isinstance(st, dict) else {},
        "ready_event_action": "set",
        "log_event": {
            "event": "fallback_to_embedded",
            "reason": why,
        },
        "recreate_plan": recreate,
    }


def panel_closed_event_plan(
    *,
    panel_window_exists: bool,
    ignore_count,
    panel_html_source,
) -> dict:
    panel_exists = bool(panel_window_exists)
    n = None
    ignore = False
    if _panel_window_fallback_mod is not None:
        try:
            ignore, n = _panel_window_fallback_mod.panel_closed_retired_event_decision(
                panel_window_exists=panel_exists,
                ignore_count=ignore_count,
            )
        except Exception:
            n = None

    if n is None:
        try:
            n0 = int(ignore_count or 0)
        except Exception:
            n0 = 0
        if panel_exists and n0 > 0:
            ignore = True
            n = n0 - 1
        else:
            ignore = False
            n = max(0, n0)

    src = str(panel_html_source or "embedded")
    if _panel_bootstrap_state_mod is not None:
        try:
            closed_state = _panel_bootstrap_state_mod.panel_bootstrap_closed_state(src)
        except Exception:
            closed_state = None
    else:
        closed_state = None

    if not isinstance(closed_state, dict):
        closed_state = {
            "status": "idle",
            "source": src,
            "reason": "window_closed",
            "created_at": None,
            "reported_at": None,
        }

    return {
        "ignore_event": bool(ignore),
        "next_ignore_count": int(n if n is not None else 0),
        "bootstrap_state": closed_state,
        "ready_event_action": "set",
        "clear_panel_window": True,
        "panel_hidden": False,
    }
