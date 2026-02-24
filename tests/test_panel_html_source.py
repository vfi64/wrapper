from __future__ import annotations

import panel_html_source as sut


def test_panel_embedded_html_prefers_embedded_snapshot():
    assert sut.panel_embedded_html("<embedded>", "<current>") == "<embedded>"


def test_panel_embedded_html_falls_back_to_current_panel_html():
    assert sut.panel_embedded_html("", "<current>") == "<current>"
    assert sut.panel_embedded_html(None, "<current>") == "<current>"


def test_panel_html_source_from_asset_meta_normalizes_nonexternal_and_missing_html():
    assert sut.panel_html_source_from_asset_meta({"source": "external"}, has_html=True) == "external"
    assert sut.panel_html_source_from_asset_meta({"source": "embedded"}, has_html=True) == "embedded"
    assert sut.panel_html_source_from_asset_meta({"source": "external"}, has_html=False) == "embedded"
    assert sut.panel_html_source_from_asset_meta("bad", has_html=True) == "embedded"


def test_select_panel_html_for_window_respects_force_embedded_flag():
    txt, src = sut.select_panel_html_for_window(
        force_embedded_html=True,
        html_panel="<external>",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
    )
    assert txt == "<embedded>"
    assert src == "embedded"


def test_select_panel_html_for_window_uses_external_html_when_available_and_marked_external():
    txt, src = sut.select_panel_html_for_window(
        force_embedded_html=False,
        html_panel="<external>",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
    )
    assert txt == "<external>"
    assert src == "external"


def test_select_panel_html_for_window_falls_back_to_embedded_when_html_missing():
    txt, src = sut.select_panel_html_for_window(
        force_embedded_html=False,
        html_panel="",
        html_panel_embedded="<embedded>",
        panel_html_asset_meta={"source": "external"},
    )
    assert txt == "<embedded>"
    # Historical behavior keeps the asset-meta source marker even if the runtime
    # text falls back to the embedded snapshot.
    assert src == "external"
