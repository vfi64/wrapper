from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_HTML = ROOT / "src" / "ui_assets" / "panel.html"
MANUAL_TEST_MONITOR_HTML = ROOT / "src" / "ui_assets" / "manual_test_monitor.html"
MANUAL_TEST_SHARED_JS = ROOT / "src" / "ui_assets" / "panel_manual_test_runner.js"
UI_CONTROLLER = ROOT / "src" / "ui_controller.py"
MONOLITH = ROOT / "src" / "Comm-SCI-Control-App.py"
MANUAL_TEST_SHARED_MARKER = "/* __PANEL_MANUAL_TEST_RUNNER_SHARED__ */"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _runtime_text(path: Path) -> str:
    txt = _read(path)
    if MANUAL_TEST_SHARED_MARKER in txt:
        txt = txt + "\n" + _read(MANUAL_TEST_SHARED_JS)
    return txt


def _panel_runtime_text() -> str:
    return _runtime_text(PANEL_HTML)


def _monolith_runtime_text() -> str:
    return _runtime_text(MONOLITH)


def test_manual_test_runner_single_source_marker_and_injection_are_wired():
    panel_html = _read(PANEL_HTML)
    monolith_txt = _read(MONOLITH)
    shared_txt = _read(MANUAL_TEST_SHARED_JS)
    assert MANUAL_TEST_SHARED_MARKER in panel_html
    assert MANUAL_TEST_SHARED_MARKER in monolith_txt
    assert "HTML_PANEL_MANUAL_TEST_SHARED = _load_panel_manual_test_shared_js(\"\")" in monolith_txt
    assert "HTML_PANEL = _inject_panel_manual_test_shared_js(HTML_PANEL, HTML_PANEL_MANUAL_TEST_SHARED)" in monolith_txt
    assert "HTML_PANEL_EMBEDDED = _inject_panel_manual_test_shared_js(HTML_PANEL_EMBEDDED, HTML_PANEL_MANUAL_TEST_SHARED)" in monolith_txt
    assert "async function _mtScenarioActualTest()" not in panel_html
    assert "async function _mtScenarioActualTest()" not in monolith_txt
    assert "async function _mtScenarioActualTest()" in shared_txt
    assert "function _mtCheckQcFooterWithSciAlertGuard(" in shared_txt
    assert "QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in shared_txt


def test_manual_test_qc_override_prefers_direct_api_and_fallback_warns_once():
    txt = _read(MANUAL_TEST_SHARED_JS)
    assert "if(api && typeof api.qc_override_apply === 'function')" in txt
    assert "const res = await _apiCall('qc_override_apply', [values || {}], 10000);" in txt
    assert "if(!mt.qcApplyFallbackNoted){" in txt
    assert "const res = await _mtPanelAction('qc_override_apply', {values: values || {}}, 10000);" in txt
    assert "if(api && typeof api.qc_override_clear === 'function')" in txt
    assert "const res = await _apiCall('qc_override_clear', [], 10000);" in txt
    assert "if(!mt.qcClearFallbackNoted){" in txt
    assert "const res = await _mtPanelAction('qc_override_clear', {}, 10000);" in txt


def test_panel_manual_test_dropdown_contains_komplexttest_option():
    html = _read(PANEL_HTML)
    assert '<option value="actual_test">' in html
    assert "ACTUAL-TEST" in html
    assert '<option value="komplexttest">' in html
    assert "Komplexttest (Matrix + Pflichtprompts + Influence-Checks)" in html
    assert '<option value="profile_self_debunking">' in html
    assert "Profile + Self-Debunking-Contract (alle Profile)" in html


def test_panel_manual_test_contains_mirror_toggle_and_i18n_keys():
    html = _read(PANEL_HTML)
    runtime_txt = _panel_runtime_text()
    assert 'id="manualTestMirrorMain"' in html
    assert "function setManualTestMirrorMode()" in runtime_txt
    assert "const _MT_I18N =" in runtime_txt
    assert "scenario_actual_test" in runtime_txt
    assert "scenario_komplexttest" in runtime_txt
    assert "scenario_profile_self_debunking" in runtime_txt
    assert "ACTUAL-TEST (fast, report file is overwritten)" in runtime_txt
    assert "Profiles + self-debunking contract (all profiles)" in runtime_txt
    assert "Complex test (matrix + mandatory prompts + influence checks)" in runtime_txt


def test_panel_komplexttest_uses_localized_prompts_and_u_marker_detector():
    runtime_txt = _panel_runtime_text()
    assert "function _mtPromptLong()" in runtime_txt
    assert "_mtPromptShort()" in runtime_txt
    assert "_mtPromptLong()" in runtime_txt
    assert "nicht die aktuelle Uhrzeit" in runtime_txt
    assert "not the current clock time" in runtime_txt
    assert "function _mtContainsUCode(html, plainText)" in runtime_txt
    assert "data-u-code" in runtime_txt
    assert "_mtContainsUCode(res.html || '', txt)" in runtime_txt
    assert "txt.replace(/\\s+:/g, ':')" in runtime_txt


def test_actual_test_checks_for_broken_red_span_around_uncertainty_marker_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtHasBrokenRedSpanAroundUncertaintyMarker(html)" in panel_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(r1.html)" in panel_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(rA.html)" in panel_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(rB.html)" in panel_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(r.html)" in panel_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(res.html)" in panel_txt
    assert "function _mtHasBrokenRedSpanAroundUncertaintyMarker(html)" in monolith_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(r1.html)" in monolith_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(rA.html)" in monolith_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(rB.html)" in monolith_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(r.html)" in monolith_txt
    assert "!_mtHasBrokenRedSpanAroundUncertaintyMarker(res.html)" in monolith_txt


def test_actual_test_checks_qc_dimension_tooltips_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtHasQcDimTooltips(html)" in panel_txt
    panel_fn = panel_txt.split("function _mtHasQcDimTooltips(html){", 1)[1].split("function _mtHasSelfDebunkingBox", 1)[0]
    assert "raw.match(/class=(?:\\\"|')[^\\\"']*qc-dim-tip[^\\\"']*(?:\\\"|')/g)" in panel_fn
    assert "\\\\bqc-dim-tip\\\\b" not in panel_fn
    assert "_mtHasQcDimTooltips(r1.html)" in panel_txt
    assert "_mtHasQcDimTooltips(rA.html)" in panel_txt
    assert "_mtHasQcDimTooltips(rB.html)" in panel_txt
    assert "_mtHasQcDimTooltips(r.html)" in panel_txt
    assert "_mtHasQcDimTooltips(res.html)" in panel_txt
    assert "function _mtHasQcDimTooltips(html)" in monolith_txt
    monolith_fn = monolith_txt.split("function _mtHasQcDimTooltips(html){", 1)[1].split("function _mtHasSelfDebunkingBox", 1)[0]
    assert "raw.match(/class=(?:\\\"|')[^\\\"']*qc-dim-tip[^\\\"']*(?:\\\"|')/g)" in monolith_fn
    assert "\\\\bqc-dim-tip\\\\b" not in monolith_fn
    assert "_mtHasQcDimTooltips(r1.html)" in monolith_txt
    assert "_mtHasQcDimTooltips(rA.html)" in monolith_txt
    assert "_mtHasQcDimTooltips(rB.html)" in monolith_txt
    assert "_mtHasQcDimTooltips(r.html)" in monolith_txt
    assert "_mtHasQcDimTooltips(res.html)" in monolith_txt


def test_actual_test_contains_image_embed_check_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtHasEmbeddedImageTagForUrl(html, url)" in panel_txt
    assert "https://example.com/comm-sci-manual-test.png" in panel_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImg.html, testImgUrl)" in panel_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgDot.html, testImgUrl)" in panel_txt
    assert "Image-URL mit Satzzeichen als <img> eingebettet" in panel_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgInlineCode.html, testImgUrl)" in panel_txt
    assert "Image-URL im Inline-Code als <img> eingebettet" in panel_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgFence.html, testImgUrl)" in panel_txt
    assert "Image-URL im Codeblock als <img> eingebettet" in panel_txt
    assert "Image-Embed-Check uebersprungen: Test-URL nicht in Antwort enthalten." in panel_txt
    assert "Image-Embed-Check (Satzzeichen) uebersprungen: Test-URL nicht in Antwort enthalten." in panel_txt
    assert "Image-Embed-Check (Inline-Code) uebersprungen: URL/Codeform nicht in Antwort enthalten." in panel_txt
    assert "Image-Embed-Check (Codeblock) uebersprungen: URL/Codeform nicht in Antwort enthalten." in panel_txt
    assert "function _mtHasEmbeddedImageTagForUrl(html, url)" in monolith_txt
    assert "https://example.com/comm-sci-manual-test.png" in monolith_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImg.html, testImgUrl)" in monolith_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgDot.html, testImgUrl)" in monolith_txt
    assert "Image-URL mit Satzzeichen als <img> eingebettet" in monolith_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgInlineCode.html, testImgUrl)" in monolith_txt
    assert "Image-URL im Inline-Code als <img> eingebettet" in monolith_txt
    assert "_mtHasEmbeddedImageTagForUrl(rImgFence.html, testImgUrl)" in monolith_txt
    assert "Image-URL im Codeblock als <img> eingebettet" in monolith_txt
    assert "Image-Embed-Check uebersprungen: Test-URL nicht in Antwort enthalten." in monolith_txt
    assert "Image-Embed-Check (Satzzeichen) uebersprungen: Test-URL nicht in Antwort enthalten." in monolith_txt
    assert "Image-Embed-Check (Inline-Code) uebersprungen: URL/Codeform nicht in Antwort enthalten." in monolith_txt
    assert "Image-Embed-Check (Codeblock) uebersprungen: URL/Codeform nicht in Antwort enthalten." in monolith_txt


def test_actual_test_contains_strict_banner_absence_check_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtHasStrictEnforcementBanner(html)" in panel_txt
    assert "function _mtHasRenderFallbackNote(html)" in panel_txt
    assert "!_mtHasStrictEnforcementBanner(r1.html)" in panel_txt
    assert "Kein Strict-Enforcement-Banner bei Basisantwort" in panel_txt
    assert "!_mtHasRenderFallbackNote(r1.html)" in panel_txt
    assert "Kein Render-Fallback-Hinweis bei Basisantwort" in panel_txt
    assert "function _mtHasStrictEnforcementBanner(html)" in monolith_txt
    assert "function _mtHasRenderFallbackNote(html)" in monolith_txt
    assert "!_mtHasStrictEnforcementBanner(r1.html)" in monolith_txt
    assert "Kein Strict-Enforcement-Banner bei Basisantwort" in monolith_txt
    assert "!_mtHasRenderFallbackNote(r1.html)" in monolith_txt
    assert "Kein Render-Fallback-Hinweis bei Basisantwort" in monolith_txt


def test_smoke_short_uses_gemini_and_transient_retry_guard():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "const r1 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'smoke-base'});" in panel_txt
    assert "const imgPrompt = _mtL(" in panel_txt
    assert "const rImg = await _mtAsk(imgPrompt, 120000);" in panel_txt
    assert "const imgPromptDot = _mtL(" in panel_txt
    assert "const rImgDot = await _mtAsk(imgPromptDot, 120000);" in panel_txt
    assert "Smoke-Check uebersprungen: transiente Provider-Fehlerantwort bei Basisprompt." in panel_txt
    assert "await _mtSetProvider('gemini');" in panel_txt
    assert "const r1 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'smoke-base'});" in monolith_txt
    assert "const imgPrompt = _mtL(" in monolith_txt
    assert "const rImg = await _mtAsk(imgPrompt, 120000);" in monolith_txt
    assert "const imgPromptDot = _mtL(" in monolith_txt
    assert "const rImgDot = await _mtAsk(imgPromptDot, 120000);" in monolith_txt
    assert "Smoke-Check uebersprungen: transiente Provider-Fehlerantwort bei Basisprompt." in monolith_txt
    assert "await _mtSetProvider('gemini');" in monolith_txt


def test_profile_self_debunking_uses_transient_retry_guard():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "const res = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'profile-self-debunking-' + profile});" in panel_txt
    assert "transient provider error response (checks skipped)." in panel_txt
    assert "function _mtLooksLikeClockTimeListAnswer(html)" in panel_txt
    assert "!_mtLooksLikeClockTimeListAnswer(res.html)" in panel_txt
    assert "Semantischer Drift: Uhrzeitliste statt Zeit-Begriff" in panel_txt
    assert "semantic drift: clock-time list instead of concept of time" in panel_txt
    assert "function _mtHasSelfDebunkingItalicSecondaryLabelLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingVerificationRouteLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingVerificationRouteHeaderLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingSourceTrainLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingRawDoubleStarLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingColorMarkerLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingPlaceholderFallbackLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(html)" in panel_txt
    assert "function _mtHasSelfDebunkingBrokenOlArtifacts(html)" in panel_txt
    assert "function _mtHasDanglingAuxiliaryBeforeUncertaintyMarker(html)" in panel_txt
    assert "function _mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(html)" in panel_txt
    assert "function _mtStripRawModelOutputDetails(html)" in panel_txt
    assert "function _mtHasUncertaintyTemplateLeak(html)" in panel_txt
    assert "function _mtHasUncertaintyTailPhraseLeakU1ToU8(html)" in panel_txt
    assert "function _mtFindUncertaintyTailPhraseLeakU1ToU8(html)" in panel_txt
    assert "function _mtFormatLeakSnippet(snippet, maxLen)" in panel_txt
    assert "function _mtHasFinalAnswerLabelInsideSciTrace(html)" in panel_txt
    assert "_mtStripHtml(_mtStripRawModelOutputDetails(html || ''))" in panel_txt
    assert "_mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingItalicSecondaryLabelLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingVerificationRouteLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingVerificationRouteHeaderLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingSourceTrainLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingRawDoubleStarLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingColorMarkerLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingPlaceholderFallbackLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(res.html)" in panel_txt
    assert "!_mtHasSelfDebunkingBrokenOlArtifacts(res.html)" in panel_txt
    assert "!_mtHasDanglingAuxiliaryBeforeUncertaintyMarker(res.html)" in panel_txt
    assert "!_mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(res.html)" in panel_txt
    assert "!_mtHasUncertaintyTailPhraseLeakU1ToU8(res.html)" in panel_txt
    assert "Self-Debunking-Secondary-Labels fett + neue Zeile (alle Punkte)" in panel_txt
    assert "self-debunking secondary labels bold + new line (all points)" in panel_txt
    assert "Self-Debunking ohne italic Secondary-Label-Leak" in panel_txt
    assert "self-debunking without italic secondary-label leak" in panel_txt
    assert "Self-Debunking ohne Verification-Route-Leak" in panel_txt
    assert "self-debunking without verification-route leak" in panel_txt
    assert "Self-Debunking ohne Verification-Route-Header im Block" in panel_txt
    assert "self-debunking without verification-route header inside block" in panel_txt
    assert "Self-Debunking ohne Source: TRAIN-Leak" in panel_txt
    assert "self-debunking without Source: TRAIN leak" in panel_txt
    assert "Self-Debunking ohne Secondary-Label-**-Leak" in panel_txt
    assert "self-debunking without secondary-label ** leak" in panel_txt
    assert "Self-Debunking ohne rohe **-Artefakte" in panel_txt
    assert "self-debunking without raw ** artifacts" in panel_txt
    assert "Self-Debunking ohne Farbmarker-Leak" in panel_txt
    assert "self-debunking without color-marker leak" in panel_txt
    assert "Self-Debunking ohne Placeholder-Fallback-Texte" in panel_txt
    assert "self-debunking without placeholder fallback phrases" in panel_txt
    assert "Self-Debunking ohne Leading-Secondary-Duplikat im ersten Feld" in panel_txt
    assert "self-debunking without leading-secondary duplicate in first field" in panel_txt
    assert "Self-Debunking ohne kaputte <ol>-Artefakte" in panel_txt
    assert "self-debunking without broken <ol> artifacts" in panel_txt
    assert "Keine abgebrochene Hilfsverb-Klausel vor U-Marker" in panel_txt
    assert "no dangling auxiliary clause before U marker" in panel_txt
    assert "Kein isolierter U-Only-Absatz vor Self-Debunking" in panel_txt
    assert "no isolated U-only paragraph before self-debunking" in panel_txt
    assert "Isolierter U-Only-Absatz vor Self-Debunking erkannt" in panel_txt
    assert "isolated U-only paragraph before self-debunking detected" in panel_txt
    assert "U1-U8-Guard ok (keine Tail-Phrasen) im Basisoutput" in panel_txt
    assert "U1-U8 guard ok (no tail phrases) in base output" in panel_txt
    assert "SCI A: U1-U8-Guard ok (keine Tail-Phrasen)" in panel_txt
    assert "SCI B: U1-U8-Guard ok (keine Tail-Phrasen)" in panel_txt
    assert "const sciATailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rA.html);" in panel_txt
    assert "const sciBTailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rB.html);" in panel_txt
    assert "SCI A: U1-U8-Tail-Leak erkannt [Match: " in panel_txt
    assert "SCI B: U1-U8-Tail-Leak erkannt [Match: " in panel_txt
    assert "SCI B: kein Final-Answer-Label im SCI-Trace" in panel_txt
    assert "SCI B + QC-Override: U1-U8-Guard ok (keine Tail-Phrasen)" in panel_txt
    assert "SCI B + QC-Override: kein Final-Answer-Label im SCI-Trace" in panel_txt
    assert "U1-U8-Guard ok (keine Tail-Phrasen)" in panel_txt
    assert "const res = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'profile-self-debunking-' + profile});" in monolith_txt
    assert "transient provider error response (checks skipped)." in monolith_txt
    assert "function _mtLooksLikeClockTimeListAnswer(html)" in monolith_txt
    assert "!_mtLooksLikeClockTimeListAnswer(res.html)" in monolith_txt
    assert "Semantischer Drift: Uhrzeitliste statt Zeit-Begriff" in monolith_txt
    assert "semantic drift: clock-time list instead of concept of time" in monolith_txt
    assert "function _mtHasSelfDebunkingItalicSecondaryLabelLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingVerificationRouteLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingVerificationRouteHeaderLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingSourceTrainLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingRawDoubleStarLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingColorMarkerLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingPlaceholderFallbackLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(html)" in monolith_txt
    assert "function _mtHasSelfDebunkingBrokenOlArtifacts(html)" in monolith_txt
    assert "function _mtHasDanglingAuxiliaryBeforeUncertaintyMarker(html)" in monolith_txt
    assert "function _mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(html)" in monolith_txt
    assert "function _mtStripRawModelOutputDetails(html)" in monolith_txt
    assert "function _mtHasUncertaintyTemplateLeak(html)" in monolith_txt
    assert "function _mtHasUncertaintyTailPhraseLeakU1ToU8(html)" in monolith_txt
    assert "function _mtFindUncertaintyTailPhraseLeakU1ToU8(html)" in monolith_txt
    assert "function _mtFormatLeakSnippet(snippet, maxLen)" in monolith_txt
    assert "function _mtHasFinalAnswerLabelInsideSciTrace(html)" in monolith_txt
    assert "_mtStripHtml(_mtStripRawModelOutputDetails(html || ''))" in monolith_txt
    assert "_mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingItalicSecondaryLabelLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingVerificationRouteLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingVerificationRouteHeaderLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingSourceTrainLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingRawDoubleStarLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingColorMarkerLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingPlaceholderFallbackLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(res.html)" in monolith_txt
    assert "!_mtHasSelfDebunkingBrokenOlArtifacts(res.html)" in monolith_txt
    assert "!_mtHasDanglingAuxiliaryBeforeUncertaintyMarker(res.html)" in monolith_txt
    assert "!_mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(res.html)" in monolith_txt
    assert "!_mtHasUncertaintyTailPhraseLeakU1ToU8(res.html)" in monolith_txt
    assert "Self-Debunking-Secondary-Labels fett + neue Zeile (alle Punkte)" in monolith_txt
    assert "self-debunking secondary labels bold + new line (all points)" in monolith_txt
    assert "Self-Debunking ohne italic Secondary-Label-Leak" in monolith_txt
    assert "self-debunking without italic secondary-label leak" in monolith_txt
    assert "Self-Debunking ohne Verification-Route-Leak" in monolith_txt
    assert "self-debunking without verification-route leak" in monolith_txt
    assert "Self-Debunking ohne Verification-Route-Header im Block" in monolith_txt
    assert "self-debunking without verification-route header inside block" in monolith_txt
    assert "Self-Debunking ohne Source: TRAIN-Leak" in monolith_txt
    assert "self-debunking without Source: TRAIN leak" in monolith_txt
    assert "Self-Debunking ohne Secondary-Label-**-Leak" in monolith_txt
    assert "self-debunking without secondary-label ** leak" in monolith_txt
    assert "Self-Debunking ohne rohe **-Artefakte" in monolith_txt
    assert "self-debunking without raw ** artifacts" in monolith_txt
    assert "Self-Debunking ohne Farbmarker-Leak" in monolith_txt
    assert "self-debunking without color-marker leak" in monolith_txt
    assert "Self-Debunking ohne Placeholder-Fallback-Texte" in monolith_txt
    assert "self-debunking without placeholder fallback phrases" in monolith_txt
    assert "Self-Debunking ohne Leading-Secondary-Duplikat im ersten Feld" in monolith_txt
    assert "self-debunking without leading-secondary duplicate in first field" in monolith_txt
    assert "Self-Debunking ohne kaputte <ol>-Artefakte" in monolith_txt
    assert "self-debunking without broken <ol> artifacts" in monolith_txt
    assert "Keine abgebrochene Hilfsverb-Klausel vor U-Marker" in monolith_txt
    assert "no dangling auxiliary clause before U marker" in monolith_txt
    assert "Kein isolierter U-Only-Absatz vor Self-Debunking" in monolith_txt
    assert "no isolated U-only paragraph before self-debunking" in monolith_txt
    assert "Isolierter U-Only-Absatz vor Self-Debunking erkannt" in monolith_txt
    assert "isolated U-only paragraph before self-debunking detected" in monolith_txt
    assert "U1-U8-Guard ok (keine Tail-Phrasen) im Basisoutput" in monolith_txt
    assert "U1-U8 guard ok (no tail phrases) in base output" in monolith_txt
    assert "SCI A: U1-U8-Guard ok (keine Tail-Phrasen)" in monolith_txt
    assert "SCI B: U1-U8-Guard ok (keine Tail-Phrasen)" in monolith_txt
    assert "const sciATailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rA.html);" in monolith_txt
    assert "const sciBTailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rB.html);" in monolith_txt
    assert "SCI A: U1-U8-Tail-Leak erkannt [Match: " in monolith_txt
    assert "SCI B: U1-U8-Tail-Leak erkannt [Match: " in monolith_txt
    assert "SCI B: kein Final-Answer-Label im SCI-Trace" in monolith_txt
    assert "SCI B + QC-Override: U1-U8-Guard ok (keine Tail-Phrasen)" in monolith_txt
    assert "SCI B + QC-Override: kein Final-Answer-Label im SCI-Trace" in monolith_txt
    assert "U1-U8-Guard ok (keine Tail-Phrasen)" in monolith_txt


def test_komplexttest_uses_transient_retry_helpers_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtLooksLikeTransientProviderError(" in panel_txt
    assert "async function _mtAskWithRetry(" in panel_txt
    assert "await _mtAskWithRetry(prompt, 180000" in panel_txt
    assert "function _mtLooksLikeTransientProviderError(" in monolith_txt
    assert "async function _mtAskWithRetry(" in monolith_txt
    assert "await _mtAskWithRetry(prompt, 180000" in monolith_txt
    assert "function _mtContainsUCode(html, plainText)" in monolith_txt
    assert "_mtContainsUCode(res.html || '', txt)" in monolith_txt


def test_komplexttest_skips_qc_and_u_hard_fails_for_unstable_sci_alert_responses():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtHasSciAlertMissingTraceStepContent(html)" in panel_txt
    assert "function _mtHasSciAlertMissingTraceStepContent(html)" in monolith_txt
    assert "QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "U-Marker-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt
    assert "U-Marker-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt


def test_manual_test_legacy_scenarios_use_qc_footer_sci_alert_guard():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "function _mtCheckQcFooterWithSciAlertGuard(" in panel_txt
    assert "function _mtCheckQcFooterWithSciAlertGuard(" in monolith_txt
    assert "QC-Check bei Basisantwort uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "Gemini: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "SCI A: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "SCI B + QC-Override: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in panel_txt
    assert "QC-Check bei Basisantwort uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt
    assert "Gemini: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt
    assert "SCI A: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt
    assert "SCI B + QC-Override: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content)." in monolith_txt


def test_panel_komplexttest_adds_export_checkpoints_before_clear_and_at_end():
    runtime_txt = _panel_runtime_text()
    assert "async function _mtExportChatAudit(label)" in runtime_txt
    assert "case_checkpoint_before_clear_chat_" in runtime_txt
    assert "before_influence_checks" in runtime_txt
    assert "komplexttest_final" in runtime_txt
    assert "manual_test_stopped_partial" in runtime_txt


def test_manual_test_main_chat_mirror_is_wired_in_controller_and_api():
    controller_txt = _read(UI_CONTROLLER)
    monolith_txt = _monolith_runtime_text()
    assert 'if action_s == "manual_test_main_chat_append":' in controller_txt
    assert "def manual_test_main_chat_append(self, payload=None):" in monolith_txt
    assert "'manual_test_main_chat_append'," in monolith_txt


def test_embedded_panel_manual_test_contains_komplexttest_option_and_route():
    txt = _monolith_runtime_text()
    assert '<option value="actual_test">' in txt
    assert "async function _mtScenarioActualTest()" in txt
    assert "if(scenario === 'actual_test') result = await _mtScenarioActualTest();" in txt
    assert '<option value="komplexttest">' in txt
    assert "async function _mtScenarioKomplexttest()" in txt
    assert "else if(scenario === 'komplexttest') result = await _mtScenarioKomplexttest();" in txt
    assert '<option value="profile_self_debunking">' in txt
    assert "async function _mtScenarioProfileSelfDebunking()" in txt
    assert "else if(scenario === 'profile_self_debunking') result = await _mtScenarioProfileSelfDebunking();" in txt
    assert ("txtNorm = txt.replace(/\\\\s+:/g, ':');" in txt) or ("txtNorm = txt.replace(/\\s+:/g, ':');" in txt)


def test_actual_test_includes_state_routing_core_slice_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "async function _mtScenarioStateRoutingCore()" in panel_txt
    assert "async function _mtScenarioCscMiddleBlock()" in panel_txt
    assert "async function _mtTryPanelAction(action, payload, timeoutMs){" in panel_txt
    assert "for(const fn of [_mtScenarioStateRoutingCore, _mtScenarioSmokeShort" in panel_txt
    assert "_mtScenarioCscMiddleBlock" in panel_txt
    assert "_mtHasControlLayerAlertsBox(res.html)" in panel_txt
    assert "kein Control-Layer-Alert-Container im Healthy-Path" in panel_txt
    assert "Runtime-Stage-Orchestrierung stabil (Strict-Gate-Dispatch + Hook-Resolver im Stage-Modul aktiv)" in panel_txt
    assert "kein Runtime-Renderer-Error" in panel_txt
    assert "!/Runtime Error in Renderer/i.test(_mtStripHtml(res.html || ''))" in panel_txt
    assert "_mtScenarioLogReplaySeam" in panel_txt
    assert "_mtScenarioProfileSelfDebunking" in panel_txt
    assert "await _mtAsk('Dynamic one-shot on', 30000);" in panel_txt
    assert "Verification route lines" in panel_txt
    assert "visible" in panel_txt
    assert "Verification-Route-Display-Policy Default" in panel_txt
    assert "set_hide_verification_route_lines" in panel_txt
    assert "provider=gemini -> hidden" in panel_txt
    assert "vr-policy-hidden-content" in panel_txt
    assert "Verification-Route-Hidden-Content-Check: kein Runtime-Renderer-Error" in panel_txt
    assert "Verification-Route-Hidden-Content-Check: Verification-Route-Header ausgeblendet" in panel_txt
    assert "Verification-Route-Hidden-Content-Check uebersprungen: transiente Provider-Fehlerantwort." in panel_txt
    assert "provider=gemini reset -> visible" in panel_txt
    assert "const stopRes = await _mtAsk('Comm Stop', 30000);" in panel_txt
    assert "const blockedAsk = await _mtTryPanelAction('ask', {text: _mtPromptShort()}, 30000);" in panel_txt
    assert "panel_action ask wird bei Comm-off geblockt" in panel_txt
    assert "const startViaPanel = await _mtTryPanelAction('ask', {text: 'Comm Start'}, 30000);" in panel_txt
    assert "const unknownAction = await _mtTryPanelAction('__unknown_action__', {}, 10000);" in panel_txt
    assert "panel_action unknown action liefert stabiles Fehlerschema" in panel_txt
    assert "_mtHasCommStateField(stopRes.html, 'Comm active', 'off')" in panel_txt
    assert "await checkCommState('Active profile', 'Standard', 'Comm Start');" in panel_txt
    assert "async function _mtScenarioStateRoutingCore()" in monolith_txt
    assert "async function _mtScenarioCscMiddleBlock()" in monolith_txt
    assert "async function _mtTryPanelAction(action, payload, timeoutMs){" in monolith_txt
    assert "for(const fn of [_mtScenarioStateRoutingCore, _mtScenarioSmokeShort" in monolith_txt
    assert "_mtScenarioCscMiddleBlock" in monolith_txt
    assert "_mtHasControlLayerAlertsBox(res.html)" in monolith_txt
    assert "kein Control-Layer-Alert-Container im Healthy-Path" in monolith_txt
    assert "Runtime-Stage-Orchestrierung stabil (Strict-Gate-Dispatch + Hook-Resolver im Stage-Modul aktiv)" in monolith_txt
    assert "kein Runtime-Renderer-Error" in monolith_txt
    assert "!/Runtime Error in Renderer/i.test(_mtStripHtml(res.html || ''))" in monolith_txt
    assert "_mtScenarioLogReplaySeam" in monolith_txt
    assert "_mtScenarioProfileSelfDebunking" in monolith_txt
    assert "await _mtAsk('Dynamic one-shot on', 30000);" in monolith_txt
    assert "Verification route lines" in monolith_txt
    assert "visible" in monolith_txt
    assert "Verification-Route-Display-Policy Default" in monolith_txt
    assert "set_hide_verification_route_lines" in monolith_txt
    assert "provider=gemini -> hidden" in monolith_txt
    assert "vr-policy-hidden-content" in monolith_txt
    assert "Verification-Route-Hidden-Content-Check: kein Runtime-Renderer-Error" in monolith_txt
    assert "Verification-Route-Hidden-Content-Check: Verification-Route-Header ausgeblendet" in monolith_txt
    assert "Verification-Route-Hidden-Content-Check uebersprungen: transiente Provider-Fehlerantwort." in monolith_txt
    assert "provider=gemini reset -> visible" in monolith_txt
    assert "const stopRes = await _mtAsk('Comm Stop', 30000);" in monolith_txt
    assert "const blockedAsk = await _mtTryPanelAction('ask', {text: _mtPromptShort()}, 30000);" in monolith_txt
    assert "panel_action ask wird bei Comm-off geblockt" in monolith_txt
    assert "const startViaPanel = await _mtTryPanelAction('ask', {text: 'Comm Start'}, 30000);" in monolith_txt
    assert "const unknownAction = await _mtTryPanelAction('__unknown_action__', {}, 10000);" in monolith_txt
    assert "panel_action unknown action liefert stabiles Fehlerschema" in monolith_txt
    assert "_mtHasCommStateField(stopRes.html, 'Comm active', 'off')" in monolith_txt
    assert "await checkCommState('Active profile', 'Standard', 'Comm Start');" in monolith_txt


def test_actual_test_includes_log_replay_seam_check_in_panel_and_embedded_code():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "async function _mtScenarioLogReplaySeam()" in panel_txt
    assert "Replay-Seam: preview_export_file ueber panel_action ok" in panel_txt
    assert "await _mtPanelAction('preview_export_file', {path: exportedChatPath, max_chars: 1200}, 20000);" in panel_txt
    assert "await _mtPanelAction('list_chat_logs', {limit: Number(limit || 200)}, 12000);" in panel_txt
    assert "function _mtLoadChatLogByName(name, fork){" in panel_txt
    assert "async function _mtTryLoadChatLogByName(name, fork){" in panel_txt
    assert "'load_chat_log'," in panel_txt
    assert "Replay-Seam: Chat-Log geladen (history_len > 0)" in panel_txt
    assert "Replay-Seam: ungueltiger Chat-Log wird sauber abgelehnt" in panel_txt
    assert "Replay-Seam: Traversal-Name wird geblockt" in panel_txt
    assert "path_traversal_blocked" in panel_txt
    assert "async function _mtScenarioLogReplaySeam()" in monolith_txt
    assert "Replay-Seam: preview_export_file ueber panel_action ok" in monolith_txt
    assert "await _mtPanelAction('preview_export_file', {path: exportedChatPath, max_chars: 1200}, 20000);" in monolith_txt
    assert "await _mtPanelAction('list_chat_logs', {limit: Number(limit || 200)}, 12000);" in monolith_txt
    assert "function _mtLoadChatLogByName(name, fork){" in monolith_txt
    assert "async function _mtTryLoadChatLogByName(name, fork){" in monolith_txt
    assert "'load_chat_log'," in monolith_txt
    assert "Replay-Seam: Chat-Log geladen (history_len > 0)" in monolith_txt
    assert "Replay-Seam: ungueltiger Chat-Log wird sauber abgelehnt" in monolith_txt
    assert "Replay-Seam: Traversal-Name wird geblockt" in monolith_txt
    assert "path_traversal_blocked" in monolith_txt


def test_profile_scenarios_use_explicit_profile_command_token():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "await _mtAsk('Profile ' + profile, 30000);" in panel_txt
    assert "await _mtAsk('Profile ' + profile, 30000);" in monolith_txt


def test_sci_and_qc_manual_scenarios_use_explicit_profile_commands():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "await _mtAsk('Profile Expert', 30000);" in panel_txt
    assert "await _mtAsk('Profile Expert', 30000);" in monolith_txt
    assert "await _mtAsk('Profile Standard', 30000);" in panel_txt
    assert "await _mtAsk('Profile Standard', 30000);" in monolith_txt
    assert "await _mtAsk('Expert', 30000);" not in panel_txt
    assert "await _mtAsk('Expert', 30000);" not in monolith_txt
    assert "await _mtAsk('Standard', 30000);" not in panel_txt
    assert "await _mtAsk('Standard', 30000);" not in monolith_txt


def test_sci_format_uses_retry_and_transient_skip_guards():
    panel_txt = _panel_runtime_text()
    monolith_txt = _monolith_runtime_text()
    assert "const rA = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A'});" in panel_txt
    assert "const rB = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B'});" in panel_txt
    assert "const rA2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A-tooltips-retry'});" in panel_txt
    assert "const rB2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B-tooltips-retry'});" in panel_txt
    assert "SCI A uebersprungen: transiente Provider-Fehlerantwort." in panel_txt
    assert "SCI B uebersprungen: transiente Provider-Fehlerantwort." in panel_txt
    assert "const rA = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A'});" in monolith_txt
    assert "const rB = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B'});" in monolith_txt
    assert "const rA2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A-tooltips-retry'});" in monolith_txt
    assert "const rB2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B-tooltips-retry'});" in monolith_txt
    assert "SCI A uebersprungen: transiente Provider-Fehlerantwort." in monolith_txt
    assert "SCI B uebersprungen: transiente Provider-Fehlerantwort." in monolith_txt


def test_manual_test_monitor_contains_stop_button_and_stop_action():
    html = _read(MANUAL_TEST_MONITOR_HTML)
    assert 'id="stopBtn"' in html
    assert "async function mtmRequestStop()" in html
    assert "manual_test_stop" in html


def test_manual_test_stop_is_wired_in_controller_and_api():
    controller_txt = _read(UI_CONTROLLER)
    monolith_txt = _monolith_runtime_text()
    assert 'if action_s == "manual_test_stop":' in controller_txt
    assert "def manual_test_request_stop(self, payload=None):" in monolith_txt
    assert "_panel_action_runtime_routes_mod = _load_optional_module_with_file_fallback(" in monolith_txt
    assert "_panel_action_gate_runtime_mod = _load_optional_module_with_file_fallback(" in monolith_txt
    assert "_panel_action_orchestrator_runtime_mod = _load_optional_module_with_file_fallback(" in monolith_txt
    assert "dispatch_panel_action" in monolith_txt
