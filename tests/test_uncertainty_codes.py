from __future__ import annotations

import re

import uncertainty_codes as sut


def test_find_uncertainty_codes_keeps_order_and_uniqueness():
    txt = "Uncertainty: U4 ... and later U1 ... then U4 again."
    out = sut.find_uncertainty_codes(txt)
    assert out == ["U4", "U1"]


def test_find_uncertainty_codes_supports_u8():
    txt = "Retrieval issue: U8 and later U1."
    out = sut.find_uncertainty_codes(txt)
    assert out == ["U8", "U1"]


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


def test_infer_uncertainty_codes_detects_retrieval_metadata_gap():
    txt = "<p>WEB claim without QualityClass and retrieval tool unavailable in this run.</p>"
    out = sut.infer_uncertainty_codes(txt)
    assert "U8" in out


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


def test_inline_markers_skip_german_profil_overlay_sci_status_block():
    src = (
        "<p>Profil: Standard · Overlay: Strict · SCI: off · Control Layer: on · Color: on</p>"
        "<p>Der zweite Absatz bleibt in Teilen mehrdeutig und interpretationsoffen.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(
        src,
        lang="de",
        user_text="Bitte erklaere das ohne feste Annahmen; mehrere Deutungen sind moeglich.",
    )
    blocks = re.findall(r"(?is)<p[^>]*>.*?</p>", out)
    assert len(blocks) >= 2
    assert "uncertainty-inline-marker" not in blocks[0]
    assert "uncertainty-inline-marker" in blocks[1]


def test_inline_markers_skip_control_layer_note_and_keep_content_markers():
    src = (
        "<div class='control-layer-note csc-warning'>"
        "<b>CONTROL LAYER NOTE</b>"
        "<ul class='control-layer-violations'>"
        "<li class='control-layer-violation'>Verification Route Gate: RED claim requires uncertainty label (U1-U8).</li>"
        "</ul>"
        "</div>"
        "<p>Unsicherheit: U1 - Datenluecke im Inhaltsteil.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    note = re.search(r"(?is)<div[^>]*control-layer-note[^>]*>.*?</div>", out)
    body = re.search(r"(?is)<p[^>]*>.*?</p>", out)
    assert note is not None
    assert body is not None
    assert "uncertainty-inline-marker" not in note.group(0)
    assert "data-u-code='U1'" in body.group(0)
    assert "data-u-code='U6'" not in out


def test_inline_markers_assign_explicit_code_to_matching_block_not_first_candidate():
    src = (
        "<p>Der erste Absatz ist lang genug, enthaelt aber keinen Unsicherheitscode und dient nur als Kontext.</p>"
        "<p>Der zweite Absatz enthaelt explizit U1 und beschreibt eine Datenluecke fuer den Inhaltsteil.</p>"
    )
    out = sut.inject_inline_uncertainty_markers_html(src, codes=["U1"], lang="de")
    blocks = re.findall(r"(?is)<p[^>]*>.*?</p>", out)
    assert len(blocks) == 2
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


def test_inline_markers_skip_self_debunking_p_li_and_mark_main_content():
    src = (
        "<div class='self-debunking'>"
        "<div>Selbst-Debunking:</div>"
        "<ol>"
        "<p>Unsicherheit: U1 - Datenluecke im Debunking-Teil.</p>"
        "<li><p><strong>Unsicherheit</strong>: U1 - Zweiter Debunking-Punkt.</p></li>"
        "</ol>"
        "</div>"
        "<p>Der Hauptinhalt bleibt unsicher und braucht externe Verifikation.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    sd_start = out.find("<div class='self-debunking'>")
    main_p = out.rfind("<p>Der Hauptinhalt")
    marker_pos = out.find("uncertainty-inline-marker")
    assert sd_start >= 0 and main_p > sd_start and marker_pos > main_p
    assert "uncertainty-inline-marker" not in out[sd_start:main_p]


def test_existing_marker_only_inside_self_debunking_still_marks_main_content():
    src = (
        "<div class='self-debunking'>"
        "<div>Selbst-Debunking:</div>"
        "<ol>"
        "<li>Unsicherheit <span class='uncertainty-inline-wrap' style='color:#111827;'>(<span class='uncertainty-inline-marker' "
        "data-u-code='U1' data-u-title='U1 - Datenluecke' title='U1 - Datenluecke'>U1</span>)</span> - Debunking.</li>"
        "</ol>"
        "</div>"
        "<p>Der Hauptinhalt benoetigt externe Verifikation wegen Datenluecke.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    sd_start = out.find("<div class='self-debunking'>")
    main_p = out.rfind("<p>Der Hauptinhalt")
    assert sd_start >= 0 and main_p > sd_start
    assert "uncertainty-inline-marker" in out[main_p:]


def test_orphan_uncertainty_marker_paragraph_before_self_debunking_is_collapsed_into_previous_paragraph():
    src = (
        "<p>Beide Perspektiven ergaenzen sich zu einem Gesamtbild.</p>"
        "<p>(U5)</p>"
        "<div class='self-debunking'><div>Selbst-Debunking:</div><ol><li>A</li><li>B</li></ol></div>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    orphan_re = re.compile(
        r"(?is)<p[^>]*>\s*<span\b[^>]*\buncertainty-inline-wrap\b[^>]*>[\s\S]*?</span>\s*</p>\s*"
        r"(?=<div\b[^>]*class=(?:\"|')[^\"']*\bself-debunking\b)",
    )
    assert orphan_re.search(out) is None
    assert "Beide Perspektiven ergaenzen sich zu einem Gesamtbild." in out
    assert "uncertainty-inline-marker" in out
    assert out.find("uncertainty-inline-marker") < out.find("self-debunking")


def test_orphan_uncertainty_marker_paragraph_before_self_debunking_is_unwrapped_when_no_prev_paragraph():
    src = (
        "<ul><li>Kontext</li></ul>"
        "<p>(U5)</p>"
        "<div class='self-debunking'><div>Selbst-Debunking:</div><ol><li>A</li><li>B</li></ol></div>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    orphan_re = re.compile(
        r"(?is)<p[^>]*>\s*<span\b[^>]*\buncertainty-inline-wrap\b[^>]*>[\s\S]*?</span>\s*</p>\s*"
        r"(?=<div\b[^>]*class=(?:\"|')[^\"']*\bself-debunking\b)",
    )
    assert orphan_re.search(out) is None
    assert "uncertainty-inline-marker" in out
    assert out.find("uncertainty-inline-marker") < out.find("self-debunking")


def test_orphan_uncertainty_marker_paragraph_with_signal_dot_before_self_debunking_is_collapsed():
    marker = sut.build_uncertainty_inline_marker_html("U5", lang="de")
    src = (
        "<p>Kontextsatz.</p>"
        "<p><span class='signal-dot-marker'>🔴</span> "
        f"{marker}</p>"
        "<div class='self-debunking'><div>Selbst-Debunking:</div><ol><li>A</li><li>B</li></ol></div>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert re.search(r"(?is)<p[^>]*>\s*(?:<span\b[^>]*\bsignal-dot-marker\b[^>]*>[\s\S]*?</span>\s*)+"
                     r"(?:<span\b[^>]*\buncertainty-inline-wrap\b[^>]*>[\s\S]*?</span>\s*)+</p>\s*"
                     r"(?=<div\b[^>]*class=(?:\"|')[^\"']*\bself-debunking\b)", out) is None
    assert out.find("signal-dot-marker") < out.find("self-debunking")


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


def test_ensure_annotations_marks_plain_explicit_u_code_with_tooltip():
    src = (
        "<p>U1: Datenlücke. Benötigt: Kontinuierliche Beobachtung und Anpassung der Strategien.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U1'" in out
    assert "data-u-title='U1 - Datenluecke" in out
    assert "U1: Datenlücke" not in out
    assert out.count("data-u-code='U1'") == 1


def test_ensure_annotations_strips_uncertainty_template_phrase_to_marker_only():
    src = (
        "<p>Die Aussage bleibt vorlaeufig. "
        "Uncertainty: (U5) - Model limitation. Needed: Explain the structural limitation and, if possible, suggest an alternative approach or external method."
        "</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U5'" in out
    assert "Uncertainty:" not in out
    assert "Model limitation. Needed:" not in out


def test_ensure_annotations_strips_code_tail_phrase_without_uncertainty_prefix_for_all_u_codes():
    for code in ("U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8"):
        name = str((sut._U_CODES.get(code) or {}).get("en_name") or "Data gap")
        src = (
            f"<p>Critical claim remains open. ({code}) - {name}. "
            "Needed: Source/current context from the user or external verification.</p>"
        )
        out = sut.ensure_uncertainty_annotations_html(src, lang="en")
        assert "uncertainty-inline-marker" in out
        assert f"data-u-code='{code}'" in out
        assert "Needed:" not in out
        assert re.search(
            rf"(?is)\(\s*{re.escape(code)}\s*\)\s*(?:-|–|—)\s*{re.escape(name)}\s*\.\s*Needed\s*:",
            re.sub(r"(?is)<[^>]+>", " ", out),
        ) is None


def test_ensure_annotations_strips_spaced_parenthesized_u_code_tail_phrase_in_running_text():
    src = (
        "<p>"
        "Physik bleibt offen. 🔴 ( U1 ) – Datenlücke. Benötigt: Weitere Forschung zur Physik des Urknalls. "
        "Philosophie betrachtet Zeit weiterhin als Bewusstseinsphänomen."
        "</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    plain = re.sub(r"(?is)<[^>]+>", " ", out)
    plain = re.sub(r"\s+", " ", plain).strip()
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U1'" in out
    assert "Benötigt:" not in plain
    assert "Benoetigt:" not in plain
    assert "Philosophie betrachtet Zeit weiterhin als Bewusstseinsphänomen." in plain


def test_ensure_annotations_repairs_dangling_german_auxiliary_clause_before_uncertainty_marker():
    src = (
        "<p>"
        "In der klassischen Physik galt Zeit als absolut. "
        "Dieser Irrtum wurde, "
        "<span class='signal-dot-marker' data-u-title='Gelb' title='Gelb' style='cursor:help;'><span style='color:#f9a825; font-weight:600;'>🟡</span></span>"
        "(U1)"
        "</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    plain = re.sub(r"(?is)<[^>]+>", " ", out)
    plain = re.sub(r"\s+", " ", plain).strip()
    assert "Dieser Irrtum wurde," not in plain
    assert "Dieser Irrtum ist als unsicher einzuordnen." in plain
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U1'" in out


def test_ensure_annotations_repairs_dangling_english_auxiliary_clause_before_uncertainty_marker():
    src = "<p>This claim was, (U1)</p>"
    out = sut.ensure_uncertainty_annotations_html(src, lang="en")
    plain = re.sub(r"(?is)<[^>]+>", " ", out)
    plain = re.sub(r"\s+", " ", plain).strip()
    assert "This claim was," not in plain
    assert "This claim is uncertain." in plain
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U1'" in out


def test_ensure_annotations_avoids_plain_and_tooltip_duplicate_for_parenthesized_u_code():
    src = (
        "<p>Die Aussage bleibt vorlaeufig und ist von der Datenlage abhaengig (U1). "
        "Weitere Verifikation ist erforderlich.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "uncertainty-inline-marker" in out
    assert out.count("data-u-code='U1'") == 1
    assert "(U1)." not in out
    assert re.search(r"\(\s*<span class='uncertainty-inline-wrap'[^>]*>\s*\(", out) is None


def test_append_uncertainty_legend_removes_existing_block():
    src = (
        "<p>Uncertainty: U2 - assumptions unclear.</p>"
        "<div class='ts-footer'>Response at 2026-02-27 14:00:00</div>"
        "<details class='uncertainty-legend'><summary>U-Code-Legende</summary><ul><li><code>U2</code></li></ul></details>"
    )
    out = sut.append_uncertainty_legend_html(src, lang="de")
    assert "uncertainty-legend" not in out


def test_ensure_annotations_converts_bracketed_u_code_inside_leaf_div():
    src = (
        "<li>"
        "<div style='font-weight:700;'>Logician:</div>"
        "<div>Die Aussage bleibt vorlaeufig und braucht Verifikation. [U1]</div>"
        "</li>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "data-u-code='U1'" in out
    assert "[U1]" not in out
    assert "uncertainty-inline-marker" in out


def test_ensure_annotations_converts_plain_code_even_if_same_code_already_marked_elsewhere():
    src = (
        "<p>Erster Absatz mit Marker "
        "<span class='uncertainty-inline-wrap' style='color:#111827;'>(<span class='uncertainty-inline-marker' "
        "data-u-code='U1' data-u-title='U1 - Datenluecke: Es fehlen belastbare Quellen oder aktueller Kontext.' "
        "title='U1 - Datenluecke: Es fehlen belastbare Quellen oder aktueller Kontext.'>U1</span>)</span>.</p>"
        "<li><div>Zweiter Absatz mit rohem Marker [U1] im Inhalt.</div></li>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert out.count("data-u-code='U1'") >= 2
    assert "[U1]" not in out


def test_ensure_annotations_upgrades_existing_marker_without_tooltip_attrs():
    src = (
        "<p>Unsicher: "
        "<span class='uncertainty-inline-marker' data-u-code='U1' style='font-weight:600;'>U1</span>"
        "</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "data-u-code='U1'" in out
    assert "data-u-title='U1 - Datenluecke:" in out
    assert "title='U1 - Datenluecke:" in out


def test_ensure_annotations_removes_square_brackets_around_existing_uncertainty_wrap():
    src = (
        "<p>Aussage "
        "[<span class='uncertainty-inline-wrap' style='color:#111827;'>(<span class='uncertainty-inline-marker' "
        "data-u-code='U1' data-u-title='U1 - Datenluecke: Es fehlen belastbare Quellen oder aktueller Kontext.' "
        "title='U1 - Datenluecke: Es fehlen belastbare Quellen oder aktueller Kontext.'>U1</span>)</span>]"
        " bleibt vorsichtig.</p>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de")
    assert "[<span class='uncertainty-inline-wrap'" not in out
    assert "data-u-code='U1'" in out


def test_ensure_annotations_does_not_infer_codes_from_self_debunking_text_only():
    src = (
        "<p>Kurze Antwort ohne explizite Unsicherheitscodes.</p>"
        "<div class='self-debunking'>"
        "<div>Selbst-Debunking:</div>"
        "<ol>"
        "<li><strong>Schwäche</strong>: Die Antwort kann stillschweigende Annahmen machen.</li>"
        "<li><strong>Schwäche</strong>: Alternative Deutungen wurden nicht voll ausgeführt.</li>"
        "</ol>"
        "</div>"
    )
    out = sut.ensure_uncertainty_annotations_html(src, lang="de", user_text="Was ist Zeit?")
    assert "uncertainty-auto-marker" not in out


def test_ensure_annotations_auto_marker_keeps_code_tags_intact():
    src = "<p>Kurz.</p>"
    out = sut.ensure_uncertainty_annotations_html(
        src,
        lang="de",
        user_text="Was ist die objektiv beste weltweit faire Strategie in jeder Kultur?",
    )
    assert "uncertainty-auto-marker" in out
    assert "<code>U3</code>" in out
    assert "<code>U5</code>" in out
    assert "<code>U6</code>" in out
    assert "<code><span class='uncertainty-inline-wrap'" not in out
    assert ")</span></code>" not in out
