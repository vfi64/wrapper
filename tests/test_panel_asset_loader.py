from __future__ import annotations

from pathlib import Path

import panel_asset_loader as sut


_PANEL_HTML_MIN_OK = """
<html><body>
<select id="provider"></select>
<select id="model"></select>
<select id="answer-language"></select>
<select id="manual-test-scenario"></select>
<select id="monitor-visibility"></select>
<div class="comm-core-grid"></div>
<script>
function panelAction() {}
function buildUI() {}
const x = window.pywebview;
</script>
</body></html>
"""


def test_panel_asset_static_selftest_ok_accepts_required_markers():
    assert sut.panel_asset_static_selftest_ok(_PANEL_HTML_MIN_OK) is True


def test_panel_asset_static_selftest_report_lists_missing_markers():
    report = sut.panel_asset_static_selftest_report("<html><body></body></html>")
    assert report["ok"] is False
    assert "provider_select" in report["missing"]
    assert "action_fn" in report["missing"]


def test_panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections():
    ok, why = sut.panel_runtime_selftest_payload_ok(
        {
            "ok": True,
            "bridge_ping": True,
            "build_ui": True,
            "dom_ok": True,
            "data_loaded": True,
            "dynamic_section_count": 0,
        }
    )
    assert ok is False
    assert why == "loaded_ruleset_but_no_dynamic_sections"


def test_load_ui_asset_text_uses_fallback_for_missing_file(tmp_path: Path):
    out = sut.load_ui_asset_text(str(tmp_path), "missing.html", "<fallback/>")
    assert out == "<fallback/>"


def test_load_panel_asset_text_s7_prefers_external_asset_when_static_selftest_passes(tmp_path: Path):
    ui_dir = tmp_path / "ui_assets"
    ui_dir.mkdir()
    (ui_dir / "panel.html").write_text(_PANEL_HTML_MIN_OK, encoding="utf-8")

    messages: list[str] = []
    out_html, meta = sut.load_panel_asset_text_s7(str(ui_dir), "<embedded/>", print_fn=messages.append)

    assert out_html == _PANEL_HTML_MIN_OK
    assert meta["source"] == "external"
    assert meta["static_ok"] is True
    assert any("external asset enabled" in msg for msg in messages)
