from __future__ import annotations

import manual_test_monitor_seam as sut


def test_manual_test_monitor_create_window_kwargs_plan_builds_hidden_dialog():
    api_obj = object()
    plan = sut.manual_test_monitor_create_window_kwargs_plan(
        html_manual_test_monitor="<html>monitor</html>",
        js_api_obj=api_obj,
    )
    kwargs = plan["kwargs"]
    assert kwargs["title"] == "Comm-SCI Manual Test Monitor"
    assert kwargs["html"] == "<html>monitor</html>"
    assert kwargs["width"] == 760 and kwargs["height"] == 700
    assert kwargs["hidden"] is True
    assert kwargs["resizable"] is True
    assert kwargs["js_api"] is api_obj


def test_manual_test_monitor_hide_plan_prefers_hide_then_minimize_and_skips_when_missing():
    p_hide = sut.manual_test_monitor_hide_plan(window_exists=True, has_hide=True, has_minimize=True)
    assert tuple(p_hide["window_methods"]) == ("hide",)
    assert p_hide["success_result"] == {"ok": True, "hidden": True}

    p_min = sut.manual_test_monitor_hide_plan(window_exists=True, has_hide=False, has_minimize=True)
    assert tuple(p_min["window_methods"]) == ("minimize",)
    assert p_min["success_result"] == {"ok": True, "hidden": True}

    p_missing = sut.manual_test_monitor_hide_plan(window_exists=False, has_hide=False, has_minimize=False)
    assert tuple(p_missing["window_methods"]) == ()
    assert p_missing["success_result"] == {"ok": True, "hidden": True, "skipped": True}


def test_manual_test_monitor_reset_state_plan_normalizes_defaults_and_builds_replace_js():
    st = {"events": [{"message": "old"}]}
    plan = sut.manual_test_monitor_reset_state_plan(
        state=st,
        payload={"scenario": "qc_override_footer", "summary": "ready"},
    )
    assert plan["state"] is st
    assert st["scenario"] == "qc_override_footer"
    assert st["status"] == "running"
    assert st["summary"] == "ready"
    assert st["events"] == []
    assert str(plan["js_code"]).startswith("mtmReplace(")


def test_manual_test_monitor_append_state_plan_wraps_non_dict_and_trims_in_place():
    st = {"events": [{"i": 0}]}
    for i in range(1, 5):
        plan = sut.manual_test_monitor_append_state_plan(state=st, entry={"i": i}, max_events=3)
    assert plan["state"] is st
    assert [e["i"] for e in st["events"]] == [2, 3, 4]

    p2 = sut.manual_test_monitor_append_state_plan(state=None, entry="hello")
    assert isinstance(p2["state"], dict)
    assert p2["entry"] == {"message": "hello"}
    assert str(p2["js_code"]).startswith("mtmAppend(")


def test_manual_test_monitor_set_header_state_plan_updates_only_supported_keys():
    st = {"scenario": "old", "status": "idle", "summary": "-", "events": [1]}
    plan = sut.manual_test_monitor_set_header_state_plan(
        state=st,
        payload={"status": "running", "summary": "halfway", "ignored": 1},
    )
    assert plan["state"] is st
    assert st["scenario"] == "old"
    assert st["status"] == "running"
    assert st["summary"] == "halfway"
    assert st["events"] == [1]
    assert plan["header"] == {"scenario": "old", "status": "running", "summary": "halfway"}
    assert str(plan["js_code"]).startswith("mtmSetHeader(")
