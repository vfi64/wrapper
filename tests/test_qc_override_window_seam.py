from __future__ import annotations

import qc_override_window_seam as sut


def test_qc_override_window_create_kwargs_plan_builds_hidden_dialog_kwargs():
    api_obj = object()
    plan = sut.qc_override_window_create_kwargs_plan(
        html_qc_override="<html>QC</html>",
        js_api_obj=api_obj,
    )
    kwargs = plan["kwargs"]
    assert kwargs["title"] == "Temporary QC override – Profile: ?"
    assert kwargs["html"] == "<html>QC</html>"
    assert kwargs["width"] == 450
    assert kwargs["height"] == 550
    assert kwargs["resizable"] is False
    assert kwargs["hidden"] is True
    assert kwargs["on_top"] is True
    assert kwargs["js_api"] is api_obj
    assert plan["clear_window_on_failure"] is True


def test_qc_override_show_plan_creates_if_missing_and_otherwise_shows_dialog():
    missing = sut.qc_override_show_plan(window_exists=False)
    assert missing["create_if_missing"] is True
    assert tuple(missing["window_methods"]) == ()
    assert missing["error_if_unavailable"] == "qc_win unavailable"
    assert missing["success_result"] is None

    existing = sut.qc_override_show_plan(window_exists=True)
    assert existing["create_if_missing"] is False
    assert tuple(existing["window_methods"]) == ("show", "bring_to_front")
    assert existing["success_result"] == {"ok": True}


def test_qc_override_cancel_plan_hides_when_window_exists_and_always_returns_ok():
    no_win = sut.qc_override_cancel_plan(window_exists=False)
    assert tuple(no_win["window_methods"]) == ()
    assert no_win["success_result"] == {"ok": True}

    has_win = sut.qc_override_cancel_plan(window_exists=True)
    assert tuple(has_win["window_methods"]) == ("hide",)
    assert has_win["success_result"] == {"ok": True}


def test_qc_override_apply_ui_plan_builds_ordered_message_and_close_action():
    plan = sut.qc_override_apply_ui_plan(
        clean_overrides={"evidence": 3, "brevity": 1},
        qc_window_exists=True,
    )
    # canonical order, not input dict order
    assert plan["history_message"] == "QC-Overrides gesetzt: Brevity=1, Evidence=3"
    assert plan["main_ui_message"] == "QC-Overrides gesetzt: Brevity=1, Evidence=3"
    assert tuple(plan["qc_window_methods"]) == ("hide",)
    assert plan["success_result"] == {"ok": True, "overrides": {"evidence": 3, "brevity": 1}}
    assert plan["warn_prefix"] == "[WARN] QC Override Apply failed: "


def test_qc_override_apply_ui_plan_handles_empty_overrides():
    plan = sut.qc_override_apply_ui_plan(clean_overrides={}, qc_window_exists=False)
    assert plan["history_message"] == "QC-Overrides gesetzt: (leer)"
    assert tuple(plan["qc_window_methods"]) == ()


def test_qc_override_clear_ui_plan_returns_reset_message_and_optional_hide():
    p1 = sut.qc_override_clear_ui_plan(qc_window_exists=False)
    assert p1["history_message"] == "QC-Overrides zurückgesetzt"
    assert p1["main_ui_message"] == "QC-Overrides zurückgesetzt"
    assert tuple(p1["qc_window_methods"]) == ()
    assert p1["success_result"] == {"ok": True}

    p2 = sut.qc_override_clear_ui_plan(qc_window_exists=True)
    assert tuple(p2["qc_window_methods"]) == ("hide",)
