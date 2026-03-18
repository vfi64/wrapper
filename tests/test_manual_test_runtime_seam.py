from __future__ import annotations

import manual_test_runtime_seam as sut


def test_manual_test_report_write_plan_uses_stable_filename_for_actual_test():
    plan = sut.manual_test_report_write_plan(
        {"scenario": "actual_test", "summary": {"status": "PASS"}},
        logs_dir="/tmp/Logs",
        ts="20260313_100000_000000",
    )
    assert plan["scenario"] == "actual_test"
    assert str(plan["target_path"]).endswith("ManualTest_ACTUAL-TEST.json")
    assert plan["overwritten"] is True


def test_manual_test_report_write_plan_uses_timestamped_name_for_other_scenarios():
    plan = sut.manual_test_report_write_plan(
        {"scenario": "profile self debunking"},
        logs_dir="/tmp/Logs",
        ts="20260313_100000_000000",
    )
    assert plan["scenario"] == "profile_self_debunking"
    assert str(plan["target_path"]).endswith("ManualTest_20260313_100000_000000_profile_self_debunking.json")
    assert plan["overwritten"] is False


def test_manual_test_main_chat_append_plan_builds_role_specific_js():
    p_user = sut.manual_test_main_chat_append_plan({"role": "user", "text": "Hallo"})
    assert p_user["role"] == "user"
    assert "addMsg('user'" in str(p_user["js"])

    p_bot = sut.manual_test_main_chat_append_plan(
        {
            "role": "bot",
            "html": "<p>Antwort</p>",
            "cgi_bar": True,
            "csc": {"score": 2},
            "answer_lang": "en",
        }
    )
    js = str(p_bot["js"])
    assert p_bot["role"] == "bot"
    assert "addMsg('bot'" in js
    assert "true" in js
    assert "answerLang" in js


def test_manual_test_request_stop_plan_contains_runner_stop_js():
    plan_de = sut.manual_test_request_stop_plan({"lang": "de"})
    js_de = str(plan_de["js"])
    assert "window.__manualTestRunner.stop = true;" in js_de
    assert "Stop angefordert (Monitor)." in js_de

    plan_en = sut.manual_test_request_stop_plan({"lang": "en"})
    js_en = str(plan_en["js"])
    assert "window.__manualTestRunner.stop = true;" in js_en
    assert "Stop requested (monitor)." in js_en
