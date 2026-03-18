from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
FIX_PATH = ROOT / "src" / "Comm-SCI-Control-App.py"


def _load_fix_module():
    spec = importlib.util.spec_from_file_location(FIX_PATH.stem, FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _normalize_render_contract_html(mod, html_in: str, *, lang: str, user_text: str) -> str:
    out = str(html_in or "")
    out = mod.ensure_self_debunking_box_html(out, lang=lang)
    out = mod.sanitize_self_debunking_markdown_in_html(out)
    out = mod.normalize_hash_subheadings_in_html(out)
    out = mod.strip_internal_scaffolding_status_html(out)
    out = mod.html_number_self_debunking(out, lang=lang)
    out = mod.sanitize_self_debunking_markdown_in_html(out)
    out = mod.ensure_self_debunking_box_html(out, lang=lang)
    uc = getattr(mod, "_uncertainty_codes_mod", None)
    if uc is not None and hasattr(uc, "ensure_uncertainty_annotations_html"):
        out = uc.ensure_uncertainty_annotations_html(out, lang=lang, user_text=user_text)
    out = mod.annotate_signal_dot_tooltips_html(out, lang=lang)
    return out


def _is_status_scaffold_text(text: str) -> bool:
    low = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not low:
        return False
    has_profile = any(k in low for k in ("active profile:", "profile:", "aktives profil:", "profil:"))
    has_matrix = any(
        k in low
        for k in ("overlay", "sci", "color", "farbe", "control layer", "steuerungsebene", "qc", "cgi")
    )
    return has_profile and has_matrix


def _status_marker_violations(final_html: str) -> list[str]:
    soup = BeautifulSoup(str(final_html or ""), "html.parser")
    out: list[str] = []
    for node in soup.find_all(["p", "div", "li"]):
        node_html = str(node)
        if "uncertainty-inline-marker" not in node_html:
            continue
        txt = node.get_text(" ", strip=True)
        if _is_status_scaffold_text(txt):
            out.append(re.sub(r"\s+", " ", txt)[:240])
    return out


def _count_markers_outside_status(final_html: str) -> int:
    soup = BeautifulSoup(str(final_html or ""), "html.parser")
    n = 0
    for node in soup.find_all(["p", "div", "li"]):
        txt = node.get_text(" ", strip=True)
        if _is_status_scaffold_text(txt):
            continue
        n += len(node.select(".uncertainty-inline-marker"))
    return n


def _self_debunking_ol_p_violations(final_html: str) -> list[str]:
    soup = BeautifulSoup(str(final_html or ""), "html.parser")
    out: list[str] = []
    for sd in soup.select("div.self-debunking"):
        for ol in sd.find_all("ol"):
            if ol.find("p") is not None:
                out.append("unexpected <p> inside <ol> in self-debunking block")
    return out


def _self_debunking_secondary_label_layout_violations(final_html: str, *, lang: str) -> list[str]:
    soup = BeautifulSoup(str(final_html or ""), "html.parser")
    out: list[str] = []
    if str(lang or "de").lower().startswith("de"):
        label_patterns = [
            r"Warum\s+das\s+wichtig\s+ist",
            r"Was\s+(?:würde|wuerde)\s+verifizieren(?:/|\s+oder\s+)falsifizieren\s+\((?:nächster|naechster)\s+Check\)",
        ]
    else:
        label_patterns = [
            r"Why\s+it\s+matters",
            r"What\s+would\s+verify(?:/|\s+or\s+)falsify\s+\(next\s+check\)",
        ]

    for sd_idx, sd in enumerate(soup.select("div.self-debunking"), start=1):
        for li_idx, li in enumerate(sd.select("ol > li"), start=1):
            li_html = str(li)
            for label_rx in label_patterns:
                if re.search(
                    rf"(?is)<br\s*/?>\s*<strong>\s*(?:{label_rx})\s*</strong>\s*:",
                    li_html,
                ) is None:
                    out.append(
                        f"sd#{sd_idx} li#{li_idx}: missing '<br><strong>...</strong>:' for secondary label pattern '{label_rx}'"
                    )
                if re.search(
                    rf"(?is)<em>\s*(?:<strong>\s*)?(?:{label_rx})(?:\s*</strong>)?\s*:?\s*</em>",
                    li_html,
                ) is not None:
                    out.append(f"sd#{sd_idx} li#{li_idx}: secondary label pattern '{label_rx}' is still italic")
    return out


def test_render_contract_replay_cases():
    mod = _load_fix_module()

    cases = [
        {
            "name": "de_status_line_must_not_receive_uncertainty_markers",
            "lang": "de",
            "user_text": "Was ist Zeit?",
            "require_sd": True,
            "require_content_marker": True,
            "html": (
                "<p>Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
                "<p>Profil: Standard · Overlay: Strict · SCI: off · Control Layer: on · Color: on (U2) (U6)</p>"
                "<p>Die Aussage bleibt mehrdeutig [U2] und muss gegen Primaerquellen geprueft werden.</p>"
                "<div class='self-debunking'>"
                "<div>Selbst-Debunking:</div>"
                "<ol>"
                "<li><strong>Weakness</strong>: Vereinfachung.<br><strong>Warum das wichtig ist</strong>: Randfaelle fehlen."
                "<br><strong>Was wuerde verifizieren/falsifizieren (naechster Check)</strong>: Gegenbeispiel testen.</li>"
                "<li><strong>Weakness</strong>: Grenzen fehlen.<br><strong>Warum das wichtig ist</strong>: Ueberdehnung moeglich."
                "<br><strong>Was wuerde verifizieren/falsifizieren (naechster Check)</strong>: Primaerquelle pruefen.</li>"
                "</ol>"
                "</div>"
                "<p>QC-Matrix: Clarity 3 (D0) · Brevity 2 (D0) · Evidence 2 (D0) · Empathy 2 (D0) · Consistency 3 (D0) · Neutrality 3 (D0)</p>"
            ),
        },
        {
            "name": "en_self_debunking_ol_must_not_keep_split_paragraph_rows",
            "lang": "en",
            "user_text": "What is time?",
            "require_sd": True,
            "require_content_marker": False,
            "html": (
                "<p>Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
                "<p>Time is a fundamental concept and can be interpreted in multiple ways [U6].</p>"
                "<div class='self-debunking'>"
                "<div>Self-Debunking:</div>"
                "<ol>"
                "<li><strong>Weakness</strong>: The answer may contain simplifications.</li>"
                "<p>Why this is important Simplifications can obscure edge cases.</p>"
                "<p><strong>What would verify/falsify (next check)</strong>: Check assumptions against primary sources.</p>"
                "<li><strong>Weakness</strong>: The answer may omit uncertainty limits. Why this is important : "
                "Missing restrictions can overstate validity."
                "<br><strong>What would verify/falsify (next check)</strong>: Add a strong counterexample.</li>"
                "</ol>"
                "</div>"
                "<p>QC-Matrix: Clarity 3 (D0) · Brevity 2 (D0) · Evidence 2 (D0) · Empathy 2 (D0) · Consistency 3 (D0) · Neutrality 3 (D0)</p>"
            ),
        },
        {
            "name": "de_self_debunking_log_artifact_secondary_labels_must_be_bold_and_linebroken",
            "lang": "de",
            "user_text": "Was ist Zeit?",
            "require_sd": True,
            "require_secondary_label_contract": True,
            "html": (
                "<p>Aus philosophischer Sicht beschreibt die Zeit das Fortschreiten der Gegenwart von der Vergangenheit in die Zukunft.</p>"
                "<div class='self-debunking'>"
                "<div>Selbst-Debunking:</div>"
                "<ol>"
                "<li><strong>Schwäche</strong>: Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen."
                "<br><strong>Warum das wichtig ist</strong>: Vereinfachungen können Randfälle oder alternative Deutungen verdecken."
                "<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Die zentralen Annahmen explizit machen.</li>"
                "<li><strong>Schwäche</strong>: Die Antwort kann wichtige Gegenpositionen auslassen. "
                "<em><strong>Warum das wichtig ist</strong>:</em>: Fehlende Einschränkungen können die Gültigkeit überdehnen. "
                "<em><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:</em>: Mindestens ein starkes Gegenbeispiel ergänzen.</li>"
                "</ol>"
                "</div>"
                "<p>QC-Matrix: Clarity 3 (D0) · Brevity 2 (D2) · Evidence 3 (D0) · Empathy 3 (D0) · Consistency 3 (D0) · Neutrality 3 (D0)</p>"
            ),
        },
        {
            "name": "de_self_debunking_log_artifact_ascii_labels_with_verification_route",
            "lang": "de",
            "user_text": "Was ist Zeit?",
            "require_sd": True,
            "require_secondary_label_contract": True,
            "html": (
                "<p>Zeit ist ein grundlegender Begriff.</p>"
                "<div class='self-debunking'>"
                "<div>Selbst-Debunking:</div>"
                "<ol>"
                "<li><strong>Schwaeche</strong>: Die Antwort kann Vereinfachungen enthalten."
                "<br><strong>Warum das wichtig ist</strong>: Randfaelle werden verdeckt."
                "<br><strong>Was wuerde verifizieren/falsifizieren (naechster Check)</strong>: Primaerquelle pruefen.</li>"
                "<li><strong>Schwaeche</strong>: Die Antwort kann Gegenpositionen auslassen.\n"
                " <em><strong>Warum das wichtig ist</strong>:</em>: Fehlende Einschraenkungen koennen Sicherheit vortaeuschen.\n"
                " <em><strong>Was wuerde verifizieren/falsifizieren (naechster Check)</strong>:</em>: Mindestens ein Gegenbeispiel ergaenzen.\n"
                " Verification Route: Source: TRAIN (allgemeines Hintergrundwissen)</li>"
                "</ol>"
                "</div>"
                "<p>QC-Matrix: Clarity 3 (D0) · Brevity 2 (D2) · Evidence 3 (D0) · Empathy 3 (D0) · Consistency 3 (D0) · Neutrality 3 (D0)</p>"
            ),
        },
        {
            "name": "en_self_debunking_italic_legacy_secondary_labels_are_canonicalized_and_linebroken",
            "lang": "en",
            "user_text": "What is time?",
            "require_sd": True,
            "require_secondary_label_contract": True,
            "html": (
                "<p>Time is a concept describing sequence and change.</p>"
                "<div class='self-debunking'>"
                "<div>Self-Debunking:</div>"
                "<ol>"
                "<li><strong>Weakness</strong>: The answer can overgeneralize."
                "<br><strong>Why it matters</strong>: Edge cases may be obscured."
                "<br><strong>What would verify/falsify (next check)</strong>: Compare assumptions with primary sources.</li>"
                "<li><strong>Weakness</strong>: The answer can omit uncertainty bounds. "
                "<em><strong>Why this is important</strong>:</em>: Missing caveats can overstate validity. "
                "<em><strong>What would verify/falsify (next check)</strong>:</em>: Add a strong counterexample and retest.</li>"
                "</ol>"
                "</div>"
                "<p>QC-Matrix: Clarity 3 (D0) · Brevity 2 (D0) · Evidence 2 (D0) · Empathy 2 (D0) · Consistency 3 (D0) · Neutrality 3 (D0)</p>"
            ),
        },
    ]

    failures: list[str] = []
    for case in cases:
        final_html = _normalize_render_contract_html(
            mod,
            case["html"],
            lang=str(case["lang"] or "de"),
            user_text=str(case["user_text"] or ""),
        )
        name = str(case["name"] or "case")
        lang = str(case["lang"] or "de")

        for v in _status_marker_violations(final_html):
            failures.append(f"{name}: uncertainty marker on status scaffold block: {v}")

        if bool(case.get("require_content_marker")) and _count_markers_outside_status(final_html) <= 0:
            failures.append(f"{name}: expected at least one uncertainty marker outside status scaffold blocks")

        require_sd = bool(case.get("require_sd"))
        if require_sd:
            if re.search(r'(?is)class=(?:"|\')[^"\']*\bself-debunking\b[^"\']*(?:"|\')', final_html) is None:
                failures.append(f"{name}: missing self-debunking box")
            if not bool(mod.detect_self_debunking_numbered_html(final_html)):
                failures.append(f"{name}: self-debunking is not numbered")

        if lang == "en":
            for v in _self_debunking_ol_p_violations(final_html):
                failures.append(f"{name}: {v}")
            if re.search(r"(?i)\bWhy\s+this\s+is\s+important\b", final_html):
                failures.append(f"{name}: non-canonical EN label 'Why this is important' leaked in final HTML")

        if bool(case.get("require_secondary_label_contract")):
            for v in _self_debunking_secondary_label_layout_violations(final_html, lang=lang):
                failures.append(f"{name}: {v}")

    assert failures == [], "\n".join(failures)
