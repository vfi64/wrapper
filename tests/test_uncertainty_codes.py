from __future__ import annotations

import re

import uncertainty_codes as sut


def test_find_uncertainty_codes_keeps_order_and_uniqueness():
    txt = "Uncertainty: U4 ... and later U1 ... then U4 again."
    out = sut.find_uncertainty_codes(txt)
    assert out == ["U4", "U1"]


def test_append_uncertainty_legend_html_is_disabled_and_strips_existing_legend():
    src = "<p>Uncertainty: U2 - assumptions unclear.</p>"
    with_legend = src + "<details class='uncertainty-legend'><summary>X</summary></details>"
    out1 = sut.append_uncertainty_legend_html(src, lang="de")
    out2 = sut.append_uncertainty_legend_html(with_legend, lang="de")
    assert "uncertainty-legend" not in out1
    assert "uncertainty-legend" not in out2


def test_build_uncertainty_legend_html_english_labels():
    out = sut.build_uncertainty_legend_html(["U5"], lang="en")
    assert "Uncertainty code legend" in out
    assert "Structural limitation" in out
    assert "<code>U5</code>" in out


def test_infer_uncertainty_codes_detects_structural_limitations():
    txt = "<p>Das ist eine komplexe Herausforderung. Es gibt keine einfache oder perfekte Loesung.</p>"
    out = sut.infer_uncertainty_codes(txt)
    assert "U5" in out


def test_ensure_uncertainty_annotations_adds_inline_marker_without_legend():
    src = "<p>Das ist eine komplexe Herausforderung ohne einfache Loesung.</p>"
    out1 = sut.ensure_uncertainty_annotations_html(
        src,
        lang="de",
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    out2 = sut.ensure_uncertainty_annotations_html(
        out1,
        lang="de",
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    assert "uncertainty-inline-marker" in out1
    assert "data-u-code='U5'" in out1
    assert "data-u-title='U5 - Strukturelle Grenze" in out1
    assert ">U5</span>)</span>" in out1
    assert "uncertainty-legend" not in out1
    assert out2 == out1


def test_inline_markers_skip_profile_overlay_sci_status_block():
    src = (
        "<p>Profile: Standard · Overlay: Strict · SCI: off · Color: on</p>"
        "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(
        src,
        lang="de",
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    blocks = re.findall(r"(?is)<p[^>]*>.*?</p>", out)
    assert len(blocks) >= 2
    assert "uncertainty-inline-marker" not in blocks[0]
    assert "uncertainty-inline-marker" in blocks[1]


def test_inline_markers_do_not_break_self_debunking_div_structure():
    src = (
        "<div class='self-debunking'>"
        "<div>Selbst-Debunking:</div>"
        "<div>1. Schwäche: Datenbasis ist begrenzt.</div>"
        "</div>"
        "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(
        src,
        lang="de",
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    sd_block = (
        "<div class='self-debunking'>"
        "<div>Selbst-Debunking:</div>"
        "<div>1. Schwäche: Datenbasis ist begrenzt.</div>"
        "</div>"
    )
    assert sd_block in out
    assert out.find("uncertainty-inline-marker") > out.find(sd_block)
    assert "uncertainty-inline-marker" in out


def test_append_uncertainty_legend_is_not_added_before_or_after_footer():
    src = (
        "<p>Uncertainty: U2 - assumptions unclear.</p>"
        "<div class='ts-footer'>Response at 2026-02-27 14:00:00</div>"
    )
    out = sut.append_uncertainty_legend_html(src, lang="en")
    assert "uncertainty-legend" not in out


def test_inline_marker_is_placed_after_matching_sentence_not_only_block_end():
    src = (
        "<p>Erster Satz bleibt ohne Unsicherheitscode. "
        "Zweiter Satz ist zeitlich instabil und sollte aktuell verifiziert werden.</p>"
    )
    out = sut.inject_inline_uncertainty_markers_html(src, codes=["U4"], lang="de")
    assert re.search(
        r"zeitlich instabil und sollte aktuell verifiziert werden\.\s*<span class='uncertainty-inline-wrap'",
        out,
    )
    assert "Erster Satz bleibt ohne Unsicherheitscode. <span class='uncertainty-inline-wrap'" not in out


def test_inline_marker_in_li_stays_sentence_near_and_not_after_closing_div():
    src = (
        "<li><div>Plan:</div><div>Die Faktlage kann sich kurzfristig aendern und sollte aktuell verifiziert werden.</div></li>"
    )
    out = sut.inject_inline_uncertainty_markers_html(src, codes=["U4"], lang="de")
    assert re.search(
        r"aktuell verifiziert werden\.\s*<span class='uncertainty-inline-wrap'",
        out,
    )
    assert "</div> <span class='uncertainty-inline-wrap'" not in out


def test_append_uncertainty_legend_removes_existing_block():
    src = (
        "<p>Uncertainty: U2 - assumptions unclear.</p>"
        "<div class='ts-footer'>Response at 2026-02-27 14:00:00</div>"
        "<details class='uncertainty-legend'><summary>U-Code-Legende</summary><ul><li><code>U2</code></li></ul></details>"
    )
    out = sut.append_uncertainty_legend_html(src, lang="de")
    assert "uncertainty-legend" not in out
