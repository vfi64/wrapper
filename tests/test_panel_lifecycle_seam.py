from __future__ import annotations

import panel_lifecycle_seam as sut


def test_panel_window_html_plan_uses_external_html_when_available():
    txt, src = sut.panel_window_html_plan(
        force_embedded_html=False,
        html_panel="<external>",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
    )
    assert txt == "<external>"
    assert src == "external"


def test_panel_window_html_plan_respects_force_embedded():
    txt, src = sut.panel_window_html_plan(
        force_embedded_html=True,
        html_panel="<external>",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
    )
    assert txt == "<embedded>"
    assert src == "embedded"


def test_panel_bootstrap_probe_plan_marks_external_pending_and_clears_event():
    plan = sut.panel_bootstrap_probe_plan("external", now_iso="2026-02-25T08:00:00")
    assert plan["source"] == "external"
    assert plan["event_action"] == "clear"
    st = plan["bootstrap_state"]
    assert st["status"] == "pending"
    assert st["created_at"] == "2026-02-25T08:00:00"


def test_panel_window_html_and_probe_plan_combines_selection_and_probe():
    plan = sut.panel_window_html_and_probe_plan(
        force_embedded_html=False,
        html_panel="<external>",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
        now_iso="2026-02-25T08:00:01",
    )
    assert plan["html"] == "<external>"
    assert plan["source"] == "external"
    assert plan["event_action"] == "clear"
    assert (plan["bootstrap_state"] or {}).get("status") == "pending"


def test_panel_bootstrap_ready_and_reason_pending_external_returns_timeout_reason():
    ready, reason = sut.panel_bootstrap_ready_and_reason(
        {"source": "external", "status": "pending", "reason": ""}
    )
    assert ready is False
    assert reason == "runtime_selftest_timeout"


def test_panel_bootstrap_wait_plan_normalizes_timeout_and_marks_waitable_pending():
    plan = sut.panel_bootstrap_wait_plan(
        {"source": "external", "status": "pending", "reason": ""},
        timeout_s="2.75",
        default_timeout_s=2.5,
    )
    assert plan["ready"] is False
    assert plan["reason"] == "runtime_selftest_timeout"
    assert plan["should_wait"] is True
    assert plan["wait_seconds"] == 2.75


def test_panel_bootstrap_wait_plan_ready_when_non_external():
    plan = sut.panel_bootstrap_wait_plan(
        {"source": "embedded", "status": "skipped", "reason": ""},
        timeout_s=None,
        default_timeout_s=2.5,
    )
    assert plan["ready"] is True
    assert plan["reason"] is None
    assert plan["should_wait"] is False


def test_panel_embedded_fallback_swap_plan_marks_failed_and_builds_recreate_plan():
    plan = sut.panel_embedded_fallback_swap_plan(
        state={"source": "external", "status": "pending"},
        reason="runtime_selftest_timeout",
        now_iso="2026-02-25T14:00:00",
        old_window_exists=True,
        ignore_count=0,
    )
    st = plan["bootstrap_state"]
    assert st["status"] == "failed"
    assert st["source"] == "external"
    assert st["reason"] == "runtime_selftest_timeout"
    assert st["reported_at"] == "2026-02-25T14:00:00"
    assert plan["ready_event_action"] == "set"
    assert plan["log_event"]["event"] == "fallback_to_embedded"
    recreate = plan["recreate_plan"]
    assert recreate["clear_panel_window"] is True
    assert recreate["force_embedded_html"] is True
    assert recreate["next_ignore_count"] >= 1


def test_panel_closed_event_plan_ignores_retired_panel_close_and_decrements_counter():
    plan = sut.panel_closed_event_plan(
        panel_window_exists=True,
        ignore_count=2,
        panel_html_source="external",
    )
    assert plan["ignore_event"] is True
    assert plan["next_ignore_count"] == 1
    assert plan["ready_event_action"] == "set"
    assert plan["clear_panel_window"] is True
    assert plan["panel_hidden"] is False
    st = plan["bootstrap_state"]
    assert st["status"] == "idle"
    assert st["reason"] == "window_closed"
    assert st["source"] == "external"


def test_panel_closed_event_plan_non_retired_keeps_ignore_zero():
    plan = sut.panel_closed_event_plan(
        panel_window_exists=False,
        ignore_count=0,
        panel_html_source="embedded",
    )
    assert plan["ignore_event"] is False
    assert plan["next_ignore_count"] == 0
    st = plan["bootstrap_state"]
    assert st["status"] == "idle"
    assert st["source"] == "embedded"


def test_panel_window_geometry_plan_sanitizes_offscreen_and_min_sizes():
    plan = sut.panel_window_geometry_plan(
        {"x": -900, "y": 99999, "width": 120, "height": 200}
    )
    assert plan["x"] == 50
    assert plan["y"] == 50
    assert plan["width"] == 250
    assert plan["height"] == 300


def test_panel_create_window_kwargs_plan_builds_pywebview_kwargs_without_calls():
    api_obj = object()
    plan = sut.panel_create_window_kwargs_plan(
        panel_geom={"x": 12, "y": 34, "width": 345, "height": 678},
        panel_window_title="Comm-SCI-Control-App Panel",
        panel_html="<html>ok</html>",
        js_api_obj=api_obj,
    )
    kwargs = plan["kwargs"]
    assert plan["precreate_hidden"] is True
    assert kwargs["title"] == "Comm-SCI-Control-App Panel"
    assert kwargs["html"] == "<html>ok</html>"
    assert kwargs["js_api"] is api_obj
    assert kwargs["x"] == 12 and kwargs["y"] == 34
    assert kwargs["width"] == 345 and kwargs["height"] == 678
    assert kwargs["on_top"] is False


def test_panel_show_plan_requests_create_when_window_missing():
    plan = sut.panel_show_plan(panel_window_exists=False)
    assert plan["action"] == "create_panel"
    assert plan["wait_bootstrap_before_show"] is False
    assert tuple(plan["window_methods"]) == ()


def test_panel_show_plan_requests_bootstrap_wait_and_focus_sequence_when_window_exists():
    plan = sut.panel_show_plan(panel_window_exists=True)
    assert plan["action"] == "show_existing"
    assert plan["wait_bootstrap_before_show"] is True
    assert tuple(plan["window_methods"]) == ("show", "restore", "focus")
    assert plan["panel_hidden"] is False


def test_panel_show_plan_can_skip_panel_focus_when_called_from_main_toggle():
    plan = sut.panel_show_plan(panel_window_exists=True, activate_panel=False)
    assert plan["action"] == "show_existing"
    assert plan["wait_bootstrap_before_show"] is True
    assert tuple(plan["window_methods"]) == ("show", "restore")
    assert plan["panel_hidden"] is False


def test_panel_hide_plan_prefers_hide_then_minimize_then_destroy():
    hide_plan = sut.panel_hide_plan(panel_window_exists=True, has_hide=True, has_minimize=True)
    assert hide_plan["action"] == "hide"
    assert hide_plan["remember_geometry"] is True
    assert hide_plan["panel_hidden"] is True
    assert hide_plan["clear_panel_window"] is False

    min_plan = sut.panel_hide_plan(panel_window_exists=True, has_hide=False, has_minimize=True)
    assert min_plan["action"] == "minimize"
    assert min_plan["panel_hidden"] is True
    assert min_plan["clear_panel_window"] is False

    destroy_plan = sut.panel_hide_plan(panel_window_exists=True, has_hide=False, has_minimize=False)
    assert destroy_plan["action"] == "destroy"
    assert destroy_plan["panel_hidden"] is False
    assert destroy_plan["clear_panel_window"] is True

    noop_plan = sut.panel_hide_plan(panel_window_exists=False, has_hide=True, has_minimize=True)
    assert noop_plan["action"] == "noop"
    assert noop_plan["remember_geometry"] is False


def test_panel_ensure_visible_plan_distinguishes_create_show_and_focus():
    p_create = sut.panel_ensure_visible_plan(panel_window_exists=False, panel_hidden=False)
    assert p_create["action"] == "create_panel"

    p_show = sut.panel_ensure_visible_plan(panel_window_exists=True, panel_hidden=True)
    assert p_show["action"] == "show_panel"

    p_focus = sut.panel_ensure_visible_plan(panel_window_exists=True, panel_hidden=False)
    assert p_focus["action"] == "focus_existing"
    assert tuple(p_focus["window_methods"]) == ("restore", "focus")


def test_panel_settings_toggle_plan_distinguishes_create_show_and_hide():
    assert sut.panel_settings_toggle_plan(panel_window_exists=False, panel_hidden=False)["action"] == "create_panel"
    assert sut.panel_settings_toggle_plan(panel_window_exists=True, panel_hidden=True)["action"] == "show_panel"
    assert sut.panel_settings_toggle_plan(panel_window_exists=True, panel_hidden=False)["action"] == "hide_panel"


def test_panel_on_closing_plan_prefers_hide_and_returns_false():
    p1 = sut.panel_on_closing_plan(panel_window_exists=True, has_hide=True)
    assert p1["call_hide_panel"] is True
    assert p1["fallback_action"] == "direct_hide"
    assert p1["fallback_sets_panel_hidden"] is True
    assert p1["return_value"] is False

    p2 = sut.panel_on_closing_plan(panel_window_exists=False, has_hide=False)
    assert p2["fallback_action"] == "noop"
    assert p2["fallback_sets_panel_hidden"] is False
    assert p2["return_value"] is False


def test_panel_bind_window_events_plan_handles_missing_window_events_and_partial_backend_support():
    p_none = sut.panel_bind_window_events_plan(
        window_exists=False,
        has_events=False,
        has_closing_event=False,
        has_closed_event=False,
    )
    assert p_none["bind_closing"] is False
    assert p_none["bind_closed"] is False

    p_no_evs = sut.panel_bind_window_events_plan(
        window_exists=True,
        has_events=False,
        has_closing_event=True,
        has_closed_event=True,
    )
    assert p_no_evs["bind_closing"] is False
    assert p_no_evs["bind_closed"] is False

    p_partial = sut.panel_bind_window_events_plan(
        window_exists=True,
        has_events=True,
        has_closing_event=False,
        has_closed_event=True,
    )
    assert p_partial["bind_closing"] is False
    assert p_partial["bind_closed"] is True
    assert p_partial["closing_handler_name"] == "on_panel_closing"
    assert p_partial["closed_handler_name"] == "on_panel_closed"


def test_panel_rebuild_plan_preserves_rebuild_order_and_messages():
    plan = sut.panel_rebuild_plan(reason="reload", panel_window_exists=True)
    assert plan["remember_geometry_before_destroy"] is True
    assert plan["destroy_old_window"] is True
    assert plan["reset_panel_window"] is True
    assert plan["reset_panel_hidden"] is False
    assert plan["recreate_panel"] is True
    assert tuple(plan["post_create_window_methods"]) == ("focus", "restore")
    assert plan["success_main_message"] == "Panel rebuilt (reload)."
    assert plan["failure_main_message_prefix"] == "Panel rebuild failed: "


def test_panel_rebuild_plan_handles_missing_existing_window():
    plan = sut.panel_rebuild_plan(reason="", panel_window_exists=False)
    assert plan["remember_geometry_before_destroy"] is False
    assert plan["destroy_old_window"] is False
    assert plan["success_main_message"] == "Panel rebuilt (reload)."
