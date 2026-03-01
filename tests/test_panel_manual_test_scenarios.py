from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_HTML = ROOT / "src" / "ui_assets" / "panel.html"
MANUAL_TEST_MONITOR_HTML = ROOT / "src" / "ui_assets" / "manual_test_monitor.html"
UI_CONTROLLER = ROOT / "src" / "ui_controller.py"
MONOLITH = ROOT / "src" / "Comm-SCI-Control-App.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_panel_manual_test_dropdown_contains_komplexttest_option():
    html = _read(PANEL_HTML)
    assert '<option value="komplexttest">' in html
    assert "Komplexttest (Matrix + Pflichtprompts + Influence-Checks)" in html


def test_panel_manual_test_contains_mirror_toggle_and_i18n_keys():
    html = _read(PANEL_HTML)
    assert 'id="manualTestMirrorMain"' in html
    assert "function setManualTestMirrorMode()" in html
    assert "const _MT_I18N =" in html
    assert "scenario_komplexttest" in html
    assert "Complex test (matrix + mandatory prompts + influence checks)" in html


def test_panel_komplexttest_uses_localized_prompts_and_u_marker_detector():
    html = _read(PANEL_HTML)
    assert "function _mtPromptLong()" in html
    assert "_mtPromptShort()" in html
    assert "_mtPromptLong()" in html
    assert "function _mtContainsUCode(html, plainText)" in html
    assert "data-u-code" in html
    assert "_mtContainsUCode(res.html || '', txt)" in html


def test_panel_komplexttest_adds_export_checkpoints_before_clear_and_at_end():
    html = _read(PANEL_HTML)
    assert "async function _mtExportChatAudit(label)" in html
    assert "case_checkpoint_before_clear_chat_" in html
    assert "before_influence_checks" in html
    assert "komplexttest_final" in html
    assert "manual_test_stopped_partial" in html


def test_manual_test_main_chat_mirror_is_wired_in_controller_and_api():
    controller_txt = _read(UI_CONTROLLER)
    monolith_txt = _read(MONOLITH)
    assert 'if action_s == "manual_test_main_chat_append":' in controller_txt
    assert "def manual_test_main_chat_append(self, payload=None):" in monolith_txt
    assert "'manual_test_main_chat_append'," in monolith_txt


def test_embedded_panel_manual_test_contains_komplexttest_option_and_route():
    txt = _read(MONOLITH)
    assert '<option value="komplexttest">' in txt
    assert "async function _mtScenarioKomplexttest()" in txt
    assert "else if(scenario === 'komplexttest') result = await _mtScenarioKomplexttest();" in txt


def test_manual_test_monitor_contains_stop_button_and_stop_action():
    html = _read(MANUAL_TEST_MONITOR_HTML)
    assert 'id="stopBtn"' in html
    assert "async function mtmRequestStop()" in html
    assert "manual_test_stop" in html


def test_manual_test_stop_is_wired_in_controller_and_api():
    controller_txt = _read(UI_CONTROLLER)
    monolith_txt = _read(MONOLITH)
    assert 'if action_s == "manual_test_stop":' in controller_txt
    assert "def manual_test_request_stop(self, payload=None):" in monolith_txt
    assert "if action_s == 'manual_test_stop':" in monolith_txt
