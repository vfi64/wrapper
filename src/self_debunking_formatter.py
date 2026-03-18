from __future__ import annotations

import html
import importlib.util
import re
from pathlib import Path

_PRIMARY_LABELS = (
    "Weakness",
    "Schwäche",
    "Schwaeche",
    "Schäche",  # tolerate common typo in user prompts/outputs
    "Uncertainty",
    "Unsicherheit",
)

_SECONDARY_LABELS = (
    "Why it matters",
    "Why this is important",
    "Warum relevant",
    "Warum es wichtig ist",
    "Warum das wichtig ist",
    "What would verify/falsify (next check)",
    "What would verify or falsify (next check)",
    "Was würde verifizieren/falsifizieren (nächster Check)",
    "Was würde verifizieren oder falsifizieren (nächster Check)",
    "Was wuerde verifizieren/falsifizieren (naechster Check)",
    "Was wuerde verifizieren oder falsifizieren (naechster Check)",
    "Next check",
    "Next step",
    "Nächster Check",
    "Nächster Schritt",
    "Nächste Prüfung",
    "Prüfen/Widerlegen (nächster Schritt)",
    "Naechster Check",
    "Naechster Schritt",
    "Naechste Pruefung",
    "Pruefen/Widerlegen (naechster Schritt)",
    "Vereinfachung",
    "Simplification",
    "Subjektivität",
    "Subjectivity",
)


def _labels_rx(labels: tuple[str, ...]) -> str:
    return "|".join(re.escape(x) for x in labels)


def _label_token_rx(labels: tuple[str, ...]) -> str:
    """Match full label tokens only (no prefix match inside words)."""
    return rf"(?<!\w)(?:{_labels_rx(labels)})(?!\w)"


_PRIMARY_LABELS_RX = _labels_rx(_PRIMARY_LABELS)
_SECONDARY_LABELS_RX = _labels_rx(_SECONDARY_LABELS)
_ALL_KNOWN_LABELS_RX = _labels_rx(_PRIMARY_LABELS + _SECONDARY_LABELS)
_PRIMARY_LABELS_TOKEN_RX = _label_token_rx(_PRIMARY_LABELS)
_SECONDARY_LABELS_TOKEN_RX = _label_token_rx(_SECONDARY_LABELS)


try:
    from output.renderers import verification_route_policy as _vr_policy  # type: ignore
except Exception:
    _vr_policy = None  # type: ignore
if _vr_policy is None:
    try:
        _vr_path = Path(__file__).resolve().parent / "output" / "renderers" / "verification_route_policy.py"
        if _vr_path.exists():
            _vr_spec = importlib.util.spec_from_file_location("output_verification_route_policy", _vr_path)  # type: ignore[attr-defined]
            if _vr_spec is not None and _vr_spec.loader is not None:
                _vr_mod = importlib.util.module_from_spec(_vr_spec)  # type: ignore[attr-defined]
                _vr_spec.loader.exec_module(_vr_mod)  # type: ignore[attr-defined]
                _vr_policy = _vr_mod  # type: ignore
    except Exception:
        _vr_policy = None  # type: ignore

_VR_MARKER_LABEL_RX = str(getattr(_vr_policy, "VR_MARKER_LABEL_RX", "") or "") or (
    r"Verification\s+Route(?:\s+Gate)?"
    r"|Source"
    r"|Measurement"
    r"|Contrast"
    r"|Web[\s\-]*Check"
    r"|Quelle"
    r"|Messung"
    r"|Kontrast"
    r"|Web[\s\-]*Pr(?:ü|ue)fung"
)


def _is_verification_route_marker_line(raw_line: str) -> bool:
    """Return True for display marker rows that should not appear inside Self-Debunking."""
    if _vr_policy is not None and callable(getattr(_vr_policy, "is_verification_route_marker_line", None)):
        try:
            return bool(_vr_policy.is_verification_route_marker_line(raw_line))  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(raw_line or "")))
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain:
            return False
        return bool(
            re.match(
                rf"(?i)^(?:[-*•]\s*)?(?:{_VR_MARKER_LABEL_RX})\s*:?\s*.*$",
                plain,
            )
        )
    except Exception:
        return False


def _strip_broken_md_wrappers_around_known_sd_labels(raw_line: str) -> str:
    """Normalize one-sided markdown emphasis leaks around known SD labels.

    Example: "Warum das wichtig ist**:" -> "Warum das wichtig ist:"
    """
    try:
        out = str(raw_line or "")
        for lab in _PRIMARY_LABELS + _SECONDARY_LABELS:
            out = re.sub(
                rf"(?im)^(\s*)(?!\*\*|__){re.escape(lab)}\s*(?:\*\*|__)+\s*:\s*",
                rf"\1{lab}: ",
                out,
            )
            out = re.sub(
                rf"(?im)^(\s*)(?:\*\*|__)+\s*{re.escape(lab)}\s*:\s*",
                rf"\1{lab}: ",
                out,
            )
        return out
    except Exception:
        return raw_line


def _looks_like_uncertainty_tail_point(raw: str) -> bool:
    """Detect redundant SD tail points that are just global U-code explanations.

    Typical model artifact:
    - "3. Schwäche: U1 – Data gap. Needed: ..."
    - "<li><strong>Schwäche</strong>: U1 - Datenluecke. Benoetigt: ...</li>"
    """
    try:
        if not raw:
            return False
        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(raw or "")))
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain:
            return False
        plain = re.sub(r"(?i)^\s*\d+\s*[\.)]\s*", "", plain, count=1)
        plain = re.sub(
            rf"(?i)^\s*(?:\*\*|__)?(?:{_PRIMARY_LABELS_TOKEN_RX})(?:\*\*|__)?\s*:\s*",
            "",
            plain,
            count=1,
        ).strip()
        if not plain or re.search(rf"(?i)\b(?:{_SECONDARY_LABELS_TOKEN_RX})\b", plain):
            return False
        if re.search(r"(?i)\bverification\s+route\b|\bsource\s*:", plain):
            return False
        has_code = re.search(r"\bU[1-8]\b", plain) is not None
        has_needed = re.search(r"(?i)\b(?:needed|ben[oö]tigt|require(?:s|d)?)\b\s*:", plain) is not None
        has_u_name = re.search(
            r"(?i)\b(?:data\s+gap|datenluecke|datenlücke|assumption\s+gap|annahmen?\s+unklar|"
            r"perspective\s+conflict|temporal\s+instability|structural\s+limitation|"
            r"interpretation\s+ambiguity|retrieval)\b",
            plain,
        ) is not None
        return bool(has_code and (has_needed or has_u_name))
    except Exception:
        return False


def _strip_uncertainty_tail_fragments(raw: str) -> str:
    """Remove embedded U-code template tails from Self-Debunking content.

    Typical artifact inside an otherwise valid point:
    "... Schwäche: U1 – Data gap. Needed: Source/current context ..."
    """
    try:
        if not raw:
            return raw
        out = str(raw)
        tail_re = re.compile(
            rf"""
            (?is)
            (?:
                \s*(?:<strong>\s*)?(?:{_PRIMARY_LABELS_TOKEN_RX})(?:\s*</strong>)?\s*:\s*
            )?
            \(?\s*\bU[1-8]\b\s*\)?
            \s*(?:-|–|—|:)\s*
            [^<\n]*?
            \b(?:needed|ben[oö]tigt|require(?:s|d)?)\b\s*:\s*
            [^<\n]*
            (?=\s*(?:<br\s*/?>|</li>|</p>|</div>|\n|$))
            """,
            re.VERBOSE,
        )
        prev = None
        while out != prev:
            prev = out
            out = tail_re.sub("", out)
        out = re.sub(r"(?is)[ \t]+(?=<br\s*/?>)", "", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"(?is)[ \t]+([,.;:!?])", r"\1", out)
        out = re.sub(r"(?is)(?:<br\s*/?>\s*){2,}", "<br>", out)
        return out
    except Exception:
        return raw


def _is_empty_sd_point(raw: str) -> bool:
    """Detect SD points that are effectively empty after cleanup."""
    try:
        if not raw:
            return True
        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(raw or "")))
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain:
            return True
        plain = re.sub(r"(?i)^\s*\d+\s*[\.)]\s*", "", plain, count=1)
        plain = re.sub(
            rf"(?i)^\s*(?:{_PRIMARY_LABELS_TOKEN_RX})\s*:?\s*",
            "",
            plain,
            count=1,
        )
        plain = re.sub(rf"(?i)\b(?:{_SECONDARY_LABELS_TOKEN_RX})\b\s*:?", "", plain)
        plain = re.sub(r"[\s\-\u2013\u2014:;,.()*]+", "", plain)
        return not plain
    except Exception:
        return False


def inject_minimal_self_debunking(text: str, *, title: str = "Self-Debunking", lang: str = "en") -> str:
    """Deterministically inject a minimal compliant Self-Debunking block (2 points).

    This is a last-resort guard used only when the ruleset requires Self-Debunking but
    the model output omitted it (and a single repair pass didn't fix it).
    The injected content avoids new factual claims; it only states generic limitations
    and next checks.
    """
    if not text:
        return text
    if title in text:
        return text

    block = ""
    lang_norm = (lang or "en").lower().strip()
    if lang_norm.startswith("de"):
        block = (
            f"\n\n{title}:\n\n"
            "1. **Schwäche**: Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen.\n"
            "   **Warum das wichtig ist**: Vereinfachungen können Randfälle oder alternative Deutungen verdecken.\n"
            "   **Was würde verifizieren/falsifizieren (nächster Check)**: Die zentralen Annahmen explizit machen und gegen Primärquellen/Definitionen prüfen.\n\n"
            "2. **Schwäche**: Die Antwort kann wichtige Gegenpositionen oder Unsicherheitsgrenzen auslassen.\n"
            "   **Warum das wichtig ist**: Fehlende Einschränkungen können die Gültigkeit überdehnen oder Sicherheit vortäuschen.\n"
            "   **Was würde verifizieren/falsifizieren (nächster Check)**: Mindestens ein starkes Gegenbeispiel ergänzen und prüfen, ob die Kernaussagen bestehen bleiben.\n"
        )
    else:
        block = (
            f"\n\n{title}:\n\n"
            "1. **Weakness**: The answer may rely on simplified framing or implicit assumptions.\n"
            "   **Why it matters**: Simplifications can hide edge-cases or alternative interpretations.\n"
            "   **What would verify/falsify (next check)**: Identify key assumptions and test them against primary sources or formal definitions.\n\n"
            "2. **Weakness**: The answer may omit important counter-perspectives or uncertainty boundaries.\n"
            "   **Why it matters**: Missing caveats can overstate confidence or applicability.\n"
            "   **What would verify/falsify (next check)**: Add at least one strong counter-example and check whether conclusions still hold.\n"
        )
    # Place block BEFORE QC-Matrix if present, else append.
    m = re.search(r"(?im)^\s*QC-Matrix:\s*.*$", text)
    if not m:
        return text.rstrip() + block
    insert_at = m.start()
    return text[:insert_at].rstrip() + block + "\n\n" + text[insert_at:].lstrip()

def sanitize_self_debunking_markdown_in_html(html_text: str) -> str:
    """Normalize leaked markdown emphasis inside already-rendered Self-Debunking HTML blocks.

    Deterministic scope:
    - only runs if a self-debunking block exists
    - converts `**label**` / `__label__` to `<strong>label</strong>`
    - formatting only (no semantic rewrites)
    """
    try:
        if not html_text:
            return html_text
        if re.search(r"(?is)class=(?:\"|')[^\"']*self-debunking[^\"']*(?:\"|')", html_text) is None:
            return html_text
        out = re.sub(r"\*\*([^*\n][^*\n]*?)\*\*", r"<strong>\1</strong>", html_text)
        out = re.sub(r"__([^_\n][^_\n]*?)__", r"<strong>\1</strong>", out)
        # Remove orphan markdown bullet artifacts like "*<br>" inside full Self-Debunking blocks.
        sd_block_re = re.compile(
            r'(?is)<div[^>]*class=(?:"|\')[^"\']*self-debunking[^"\']*(?:"|\')[^>]*>.*?</div>(?=\s*<(?:p|div)\b|\s*\Z)'
        )

        def _clean_sd_block(m: re.Match) -> str:
            block = str(m.group(0) or "")
            block = re.sub(r"(?im)\s*\*\s*(?=<br\s*/?>)", "", block)
            block = re.sub(r"(?im)(^|>\s*)\*\s*(?=(?:<strong>|[A-Za-zÄÖÜäöü]))", r"\1", block)
            block = re.sub(r"(?im)\n[ \t]*\*[ \t]*(?=\n)", "\n", block)
            return block

        out = sd_block_re.sub(_clean_sd_block, out)
        return out
    except Exception:
        return html_text


def ensure_self_debunking_box_html(html_text: str, *, lang: str = "en") -> str:
    """Force a standalone Self-Debunking box when weak markdown rendering leaks it into normal lists."""
    try:
        if not html_text:
            return html_text
        src = str(html_text)
        if re.search(r"(?is)class=(?:\"|')[^\"']*self-debunking[^\"']*(?:\"|')", src):
            return src
        if re.search(r"(?i)Self[- ]?Debunking|Selbst[- ]?Debunking", src) is None:
            return src

        hdr_re = re.compile(
            r"(?is)(?:<strong>\s*(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:\s*</strong>|(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:)"
        )
        hm = hdr_re.search(src)
        if not hm:
            return src

        start = int(hm.start())
        rest = src[hm.end():]
        qc_m = re.search(r"(?is)<p[^>]*>\s*QC(?:-Matrix)?\s*:", rest)
        end = hm.end() + (qc_m.start() if qc_m else len(rest))

        segment = src[start:end]
        plain = html.unescape(re.sub(r"(?is)<[^>]+>", "\n", segment))
        plain = re.sub(r"[ \t]+", " ", plain)
        plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
        if not plain:
            return src

        m_plain = re.search(r"(?is)(Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:", plain)
        if m_plain:
            plain = plain[m_plain.start():]

        title = "Selbst-Debunking" if str(lang or "").lower().startswith("de") else "Self-Debunking"
        if re.match(r"(?is)^\s*(Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:", plain) is None:
            plain = f"{title}:\n{plain}"

        norm = normalize_inline_self_debunking_header(plain)
        norm = normalize_self_debunking_language(norm, lang)
        norm = normalize_self_debunking_field_linebreaks(norm, lang=lang)
        norm = bold_self_debunking_labels(norm, lang)
        norm = normalize_self_debunking_numbering_text(norm, lang=lang)

        block_m = re.search(
            r"(?is)(\b(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\b.*?)(?:\n\s*QC(?:-Matrix)?\s*:|\Z)",
            norm,
        )
        block = block_m.group(1) if block_m else norm
        lines = block.splitlines()
        if not lines:
            return src
        body = "\n".join(lines[1:])

        point_iter = list(re.finditer(r"(?m)^\s*\d+\.\s+", body))
        chunks = []
        for i, pm in enumerate(point_iter):
            p_start = pm.start()
            p_end = point_iter[i + 1].start() if i + 1 < len(point_iter) else len(body)
            ch = body[p_start:p_end].strip()
            if ch:
                chunks.append(ch)
        if not chunks:
            return src

        def _format_chunk(chunk: str) -> str:
            rows = []
            for raw_ln in str(chunk or "").splitlines():
                ln = (raw_ln or "").strip()
                if not ln:
                    continue
                ln = re.sub(r"^\*+\s*", "", ln)
                rows.append(ln)
            if not rows:
                return ""

            rows[0] = re.sub(r"^\d+\.\s*", "", rows[0], count=1)

            rendered = []
            for ln in rows:
                esc = html.escape(ln)
                esc = re.sub(r"\*\*([^*\n]{1,140})\*\*\s*:", r"<strong>\1</strong>:", esc)
                esc = re.sub(r"__([^_\n]{1,140})__\s*:", r"<strong>\1</strong>:", esc)
                esc = re.sub(
                    rf"(?i)^(?:{_PRIMARY_LABELS_TOKEN_RX})\s*:",
                    lambda mm: f"<strong>{html.escape((mm.group(0) or '').split(':', 1)[0].strip())}</strong>:",
                    esc,
                    count=1,
                )
                rendered.append(esc)
            return "<br>".join(rendered)

        items = [x for x in (_format_chunk(c) for c in chunks) if x]
        if not items:
            return src

        prefix = src[:start]
        suffix = src[end:]

        # If we cut from inside a legacy <ul>, close the still-open list tags to avoid list spillover.
        for tag in ("ol", "ul"):
            open_n = len(re.findall(rf"(?is)<{tag}\b", prefix))
            close_n = len(re.findall(rf"(?is)</{tag}>", prefix))
            if open_n > close_n:
                prefix = prefix.rstrip() + ("\n" + f"</{tag}>" * (open_n - close_n))

        title_render = "Selbst-Debunking:" if str(lang or "").lower().startswith("de") else "Self-Debunking:"
        box = (
            '<div class="self-debunking" style="border-left: 4px solid #a5b4fc; background-color: #eef2ff; '
            'padding: 10px; border-radius: 8px; margin: 10px 0;">'
            '<div style="font-weight:700; color:#4338ca; margin-bottom:6px;">'
            + html.escape(title_render)
            + "</div><ol>"
            + "".join(f"<li>{it}</li>" for it in items)
            + "</ol></div>"
        )

        return prefix.rstrip() + "\n" + box + "\n" + suffix.lstrip()
    except Exception:
        return html_text

def normalize_self_debunking_language(text: str, lang: str) -> str:
    """Translate Self-Debunking label tokens into the target language (currently DE),
    without changing the required header 'Self-Debunking' or adding new factual claims.

    This is a deterministic post-processing step for models that keep English label words
    (e.g., 'Weakness', 'Why it matters', 'What would verify/falsify (next check)') even
    when answer_language=de.
    """
    try:
        if not text or not lang:
            return text
        if not str(lang).lower().startswith("de"):
            return text

        # Isolate the Self-Debunking block up to the QC footer (or end of text).
        # We keep the section header unchanged but translate common label tokens inside.
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text

        block = m.group(1)
        tail_marker = m.group(2)  # either QC footer marker or end

        # Translate label phrases (keep punctuation/colon style flexible).
        repl = [
            (r"(?i)\bWeakness\b\s*:", "Schwäche:"),
            (r"(?i)\bUncertainty\b\s*:", "Schwäche:"),
            (r"(?i)\bUnsicherheit\b\s*:", "Schwäche:"),
            (r"(?i)\bSimplification\b\s*:", "Vereinfachung:"),
            (r"(?i)\bSubjectivity\b\s*:", "Subjektivität:"),
            (r"(?i)\bWhy\s+it\s+matters\b\s*:", "Warum das wichtig ist:"),
            (r"(?i)\bWhy\s+this\s+is\s+important\b\s*:", "Warum das wichtig ist:"),
            (r"(?i)\bWhat\s+would\s+verify\s*/\s*falsify\s*\(next\s+check\)\s*:",
             "Was würde verifizieren/falsifizieren (nächster Check):"),
            (r"(?i)\bWhat\s+would\s+verify\s+or\s+falsify\s*\(next\s+check\)\s*:",
             "Was würde verifizieren oder falsifizieren (nächster Check):"),
            (r"(?i)\bNext\s+check\b\s*:", "Nächster Check:"),
            (r"(?i)\bNext\s+step\b\s*:", "Nächster Schritt:"),
        ]
        for pat, rep in repl:
            block = re.sub(pat, rep, block)

        # Reassemble
        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def bold_self_debunking_labels(text: str, lang: str) -> str:
    """Bold the label token before the first colon inside Self-Debunking points.

    Deterministic post-processing. Formatting only.
    """
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text
        block = m.group(1)

        # Repair one-sided markdown leaks before bolding logic runs.
        block = "\n".join(_strip_broken_md_wrappers_around_known_sd_labels(ln) for ln in block.splitlines())

        # 1) Numbered point heads: "1. Weakness:" -> "1. **Weakness**:"
        def _bold_head(m2):
            lead = m2.group(1)
            label = (m2.group(2) or "").strip()
            if not label:
                return m2.group(0)
            if label.startswith("**") and label.endswith("**"):
                return m2.group(0)
            return f"{lead}**{label}**:"
        block = re.sub(r"(?m)^(\s*\d+\.\s*)([^\n:<]{1,80}?)(\s*):", lambda m2: _bold_head(m2), block)

        # 2) Field labels (possibly indented): "Why it matters:" -> "**Why it matters**:"
        for lab in _PRIMARY_LABELS + _SECONDARY_LABELS:
            block = re.sub(
                rf"(?m)^(\s*)(?!\*\*){re.escape(lab)}\s*:(?!\*)",
                rf"\1**{lab}**:",
                block
            )
        # Collapse accidental duplicate colons on known labels ("Why it matters:: ...")
        # before markdown conversion, which can otherwise leak as ":<strong>:" HTML.
        block = re.sub(
            rf"(?im)^(\s*(?:\d+\.\s*)?(?:\*\*|__)?(?:{_ALL_KNOWN_LABELS_RX})(?:\*\*|__)?\s*):\s*:\s*",
            r"\1: ",
            block,
        )

        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def normalize_self_debunking_field_linebreaks(text: str, *, lang: str = "en") -> str:
    """Ensure secondary Self-Debunking field labels start on a new line (no extra numbering).

    This keeps formatting stable when weaker models place
    "Warum das wichtig ist:" / "What would verify..." inline behind the Weakness sentence.
    """
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text
        block = m.group(1)

        labels_rx = _SECONDARY_LABELS_RX
        # Insert a line break before secondary field labels if they leak inline.
        block = re.sub(
            rf"(?i)([^\n])\s+(?=(?:\*\*|__)?(?:{labels_rx})(?:\*\*|__)?\s*:)",
            r"\1\n   ",
            block,
        )

        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def normalize_self_debunking_numbering_text(text: str, *, lang: str = "en") -> str:
    """Ensure stable numbered Self-Debunking points in plain text before HTML rendering."""
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text

        block = m.group(1)
        lines = block.splitlines()
        if not lines:
            return text

        out = []
        n = 0
        # Keep title line unchanged
        out.append(lines[0])
        for ln in lines[1:]:
            s = (ln or "").strip()
            # Remove orphan marker lines that cause visible double numbering.
            if re.fullmatch(r"\d+\.", s or ""):
                continue

            # Self-Debunking must not carry verification-route marker rows.
            if _is_verification_route_marker_line(ln):
                continue

            ln = _strip_broken_md_wrappers_around_known_sd_labels(ln)

            # Strip leaked list prefixes in front of known Self-Debunking labels.
            # Weakness/Schwäche lines get renumbered deterministically below.
            ln = re.sub(
                rf"(?im)^\s*\d+\.\s*(?=(?:\*\*|__)?(?:{_ALL_KNOWN_LABELS_RX})(?:\*\*|__)?\s*:)",
                "",
                ln,
                count=1,
            )

            if lang.lower().startswith("de"):
                ln = re.sub(r"(?i)\bWeakness\b\s*:", "Schwäche:", ln)
                ln = re.sub(r"(?i)\bUncertainty\b\s*:", "Schwäche:", ln)
                ln = re.sub(r"(?i)\bUnsicherheit\b\s*:", "Schwäche:", ln)
                ln = re.sub(r"(?i)\bSimplification\b\s*:", "Vereinfachung:", ln)
                ln = re.sub(r"(?i)\bSubjectivity\b\s*:", "Subjektivität:", ln)
                ln = re.sub(r"(?i)\bWhy\s+it\s+matters\b\s*:", "Warum das wichtig ist:", ln)
                ln = re.sub(r"(?i)\bWhy\s+this\s+is\s+important\b\s*:", "Warum das wichtig ist:", ln)
                ln = re.sub(
                    r"(?i)\bWhat\s+would\s+verify\s*/\s*falsify\s*\(next\s+check\)\s*:",
                    "Was würde verifizieren/falsifizieren (nächster Check):",
                    ln,
                )
                ln = re.sub(r"(?i)\bNext\s+step\b\s*:", "Nächster Schritt:", ln)

            # Number only the Weakness/Schwäche lead lines.
            if re.match(
                rf"(?im)^\s*(?:\d+\.\s*)?(?:\*\*|__)?(?:{_PRIMARY_LABELS_RX})(?:\*\*|__)?\s*:",
                ln or "",
            ):
                n += 1
                ln = re.sub(r"(?im)^\s*\d+\.\s*", "", ln, count=1)
                ln = f"{n}. {ln.lstrip()}"
            out.append(ln)

        start, end = m.span(1)
        return text[:start] + "\n".join(out) + text[end:]
    except Exception:
        return text


def normalize_inline_self_debunking_header(text: str) -> str:
    """Ensure Self-Debunking header starts on its own line for deterministic boxing."""
    try:
        if not text:
            return text
        out = str(text)
        # If title leaks inline after a sentence, split into a new block.
        out = re.sub(
            r"([^\n])\s+((?:SCI\s*Trace\s*:\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:)",
            r"\1\n\n\2",
            out,
            flags=re.IGNORECASE,
        )
        # If title line has immediate inline body, push the body to the next line.
        out = re.sub(
            r"(?im)^(\s*(?:SCI\s*Trace\s*:\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:)\s+(?=\S)",
            r"\1\n",
            out,
        )
        return out
    except Exception:
        return text


def dedupe_self_debunking_sections(text: str) -> str:
    """Keep exactly one Self-Debunking section when duplicates leak from weaker models.

    Rules (deterministic):
    - Detect section headers:
      - "Self-Debunking:" / "Selbst-Debunking:"
      - "SCI Trace: Self-Debunking:" / "SCI Trace: Selbst-Debunking:"
    - If multiple are present, keep the last *pure* Self-Debunking section.
      If none is pure, keep the last detected section.
    - Section ends at next SD header, QC-Matrix header, or end of text.
    """
    try:
        if not text:
            return text

        lines = str(text).splitlines()
        if not lines:
            return text

        re_sd_header = re.compile(
            r"(?i)^\s*(?:(?P<sci>SCI\s*Trace)\s*:\s*)?(?P<sd>Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:\s*$"
        )
        re_qc = re.compile(r"(?i)^\s*QC(?:-Matrix)?\s*:")

        starts = []
        for i, ln in enumerate(lines):
            m = re_sd_header.match((ln or "").strip())
            if m:
                starts.append((i, bool(m.group("sci"))))

        if len(starts) <= 1:
            return text

        # Build section ranges [start, end)
        ranges = []
        for idx, (start_i, is_sci_prefixed) in enumerate(starts):
            end_i = len(lines)
            for j in range(start_i + 1, len(lines)):
                s = (lines[j] or "").strip()
                if re_sd_header.match(s) or re_qc.match(s):
                    end_i = j
                    break
            ranges.append((start_i, end_i, is_sci_prefixed))

        # Keep last pure SD section if available; else last section.
        keep = None
        for r in ranges:
            if not r[2]:
                keep = r
        if keep is None:
            keep = ranges[-1]

        # Remove all non-kept SD ranges.
        remove_mask = [False] * len(lines)
        for r in ranges:
            if r is keep:
                continue
            for k in range(r[0], r[1]):
                remove_mask[k] = True

        out_lines = [ln for i, ln in enumerate(lines) if not remove_mask[i]]
        return "\n".join(out_lines)
    except Exception:
        return text


def html_number_self_debunking(html_text: str, *, lang: str = "en") -> str:
    """Best-effort: add stable 1./2./3. numbering to Self-Debunking in rendered HTML.

    Safety goals:
    - Never inject numbering inside existing <ol> lists.
    - Remove orphan list-marker lines like "1." that can appear after Markdown conversion.
    - Keep exactly one numeric prefix on each Weakness/Schwäche line.
    """
    try:
        if not html_text:
            return html_text
        # Normalize compact one-line HTML so line-wise processing can see block rows.
        html_text = str(html_text).replace("><", ">\n<")
        if re.search(r"(?i)Self-Debunking|Selbst[- ]?Debunking", html_text) is None:
            return html_text

        def _normalize_block(body: str) -> str:
            def _repair_broken_label_colon_markup(src: str) -> str:
                """Repair label-colon artifacts like '<strong>Why it matters</strong>:<strong>: ...'."""
                try:
                    if not src:
                        return src
                    out = str(src)
                    out = re.sub(
                        rf"(?is)(<strong>\s*(?:{_ALL_KNOWN_LABELS_RX})\s*</strong>)\s*:\s*<strong>\s*:\s*",
                        r"\1: ",
                        out,
                    )
                    out = re.sub(
                        rf"(?is)(<strong>\s*(?:{_ALL_KNOWN_LABELS_RX})\s*</strong>)\s*:\s*:\s*",
                        r"\1: ",
                        out,
                    )
                    return out
                except Exception:
                    return src

            def _drop_verification_route_marker_segments_html(src: str) -> str:
                """Drop leaked verification-route marker rows inside Self-Debunking HTML blocks."""
                try:
                    if not src:
                        return src
                    out = str(src)
                    # Drop standalone wrapped rows (p/div/li) that only carry route markers.
                    out = re.sub(
                        rf"(?is)<(p|div|li)[^>]*>\s*(?:<strong>\s*)?(?:{_VR_MARKER_LABEL_RX})(?:\s*</strong>)?\s*:?\s*[^<]*</\1>",
                        "",
                        out,
                    )
                    # Drop inline <br>-delimited marker fragments inside list items.
                    out = re.sub(
                        rf"(?is)<br\s*/?>\s*(?:<strong>\s*)?(?:{_VR_MARKER_LABEL_RX})(?:\s*</strong>)?\s*:?\s*[^<]*(?=(?:<br\s*/?>|</li>|</p>|</div>|$))",
                        "",
                        out,
                    )
                    out = re.sub(r"(?is)(?:<br\s*/?>\s*){2,}", "<br>", out)
                    out = re.sub(r"(?is)<li([^>]*)>\s*<br\s*/?>", r"<li\1>", out)
                    out = re.sub(r"(?is)<p([^>]*)>\s*<br\s*/?>", r"<p\1>", out)
                    return out
                except Exception:
                    return src

            if '<ol' in body.lower():
                # Keep ordered lists intact, but clean known leak artifacts:
                # - orphan marker rows like "<p>1.</p>"
                # - accidental numbering on non-Weakness field labels
                cleaned = body
                cleaned = _drop_verification_route_marker_segments_html(cleaned)
                prim_canon = "Schwäche" if lang.lower().startswith("de") else "Weakness"
                # Repair a common malformed sequence:
                # "<li><strong>Schwäche</strong>:<strong>1. Schwäche</strong></li><p>*: ...</p>"
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*(?:<strong>\s*)?(?:Schwäche|Weakness)(?:\s*</strong>)?\s*:\s*)"
                    rf"(?:<strong>\s*)?\d+\.\s*(?:Schwäche|Weakness)(?:\s*</strong>)?\s*",
                    r"\1",
                    cleaned,
                )
                cleaned = re.sub(
                    r"(?is)</li>\s*<p[^>]*>\s*\*+\s*:\s*([^<\s][^<]*)\s*</p>",
                    r"<br>\1</li>",
                    cleaned,
                )
                cleaned = re.sub(r"(?is)<p>\s*\d+\.\s*</p>", "", cleaned)
                cleaned = re.sub(r"(?is)<li>\s*\d+\.\s*</li>", "", cleaned)
                labels_rx = _SECONDARY_LABELS_TOKEN_RX
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*)(\d+\.\s*)(?=(?:<strong>\s*)?(?:{labels_rx})\s*:)",
                    r"\1",
                    cleaned,
                )
                # Normalize block wrappers inside list items.
                # If weak markdown conversion yields "<li><p>...</p><p>...</p></li>" or
                # "<li><div>...</div><div>...</div></li>", browser default margins can create
                # uneven line spacing between SD points. Flatten to one <li> with <br> rows.
                def _flatten_li_block_wrappers(mm: re.Match) -> str:
                    attrs = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "")
                    if re.search(r"(?is)<(?:p|div)\b", inner) is None:
                        return mm.group(0)
                    flat = re.sub(r"(?is)</(?:p|div)>\s*<(?:p|div)[^>]*>\s*", "<br>", inner)
                    flat = re.sub(r"(?is)<(?:p|div)[^>]*>\s*", "", flat)
                    flat = re.sub(r"(?is)\s*</(?:p|div)>", "", flat)
                    flat = re.sub(r"(?is)(?:<br>\s*){2,}", "<br>", flat)
                    flat = re.sub(r"(?is)^\s*<br>\s*", "", flat)
                    flat = flat.strip()
                    if not flat:
                        return mm.group(0)
                    return f"<li{attrs}>{flat}</li>"

                cleaned = re.sub(
                    r"(?is)<li([^>]*)>\s*(.*?)\s*</li>",
                    _flatten_li_block_wrappers,
                    cleaned,
                )
                # Drop leaked color-marker rows inside SD points.
                # Regression: weak SCI outputs can append rows like
                # "🔴 Eine Hyperantithesis ..." to SD list items, which must never render
                # inside the Self-Debunking box.
                def _is_sd_color_marker_row(seg_html: str) -> bool:
                    try:
                        if not seg_html:
                            return False
                        raw_seg = str(seg_html or "")
                        if re.search(r"(?is)\bsignal-dot-marker\b", raw_seg):
                            return True
                        plain_seg = html.unescape(re.sub(r"(?is)<[^>]+>", " ", raw_seg))
                        plain_seg = re.sub(r"\s+", " ", plain_seg).strip()
                        if not plain_seg:
                            return False
                        plain_seg = re.sub(
                            r"(?i)^\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]\s*",
                            "",
                            plain_seg,
                            count=1,
                        )
                        return bool(re.match(r"^[🟢🟡🔴⚪]\s*(?:\S|$)", plain_seg))
                    except Exception:
                        return False

                def _strip_sd_color_marker_rows_from_li(mm: re.Match) -> str:
                    attrs = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "")
                    if not inner:
                        return mm.group(0)
                    parts = re.split(r"(?is)<br\s*/?>", inner)
                    kept = []
                    for part in parts:
                        if _is_sd_color_marker_row(part):
                            continue
                        part = str(part or "").strip()
                        if part:
                            kept.append(part)
                    if not kept:
                        return ""
                    return f"<li{attrs}>{'<br>'.join(kept)}</li>"

                def _drop_sd_color_marker_paragraph_rows(mm: re.Match) -> str:
                    inner = str(mm.group(2) or "")
                    if _is_sd_color_marker_row(inner):
                        return ""
                    return mm.group(0)

                cleaned = re.sub(
                    r"(?is)<li([^>]*)>\s*(.*?)\s*</li>",
                    _strip_sd_color_marker_rows_from_li,
                    cleaned,
                )
                cleaned = re.sub(
                    r"(?is)<p([^>]*)>\s*(.*?)\s*</p>",
                    _drop_sd_color_marker_paragraph_rows,
                    cleaned,
                )
                # Keep secondary field labels on a new visual line inside list items.
                sec_rx = _SECONDARY_LABELS_TOKEN_RX
                # Self-Debunking boxes should not show CGI color bullets or isolated markdown debris.
                cleaned = re.sub(r"(?is)<span[^>]*>\s*[🟢🟡🔴]\s*</span>", "", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*[🟢🟡🔴]\s*</\1>", "", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</\1>", "", cleaned)
                cleaned = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", cleaned)
                cleaned = re.sub(
                    r"(?is)<strong>\s*(?:Unsicherheit|Uncertainty)\s*</strong>\s*:",
                    f"<strong>{prim_canon}</strong>:",
                    cleaned,
                )
                # Some weak repairs emit orphan paragraphs directly inside <ol>
                # before the first <li>. Normalize that leading paragraph into a list item.
                def _leading_orphan_p_to_li(mm: re.Match) -> str:
                    ol_open = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "").strip()
                    if not inner:
                        return mm.group(0)
                    plain_inner = re.sub(r"(?is)<[^>]+>", "", inner)
                    plain_inner = re.sub(r"\s+", " ", plain_inner).strip()
                    if not plain_inner:
                        return mm.group(0)
                    inner_norm = re.sub(
                        r"(?is)^\s*(?:<strong>\s*)?(?:Unsicherheit|Uncertainty)(?:\s*</strong>)?\s*:?\s*",
                        "",
                        inner,
                        count=1,
                    ).strip()
                    if inner_norm != inner:
                        if inner_norm:
                            return f"{ol_open}<li><strong>{prim_canon}</strong>: {inner_norm}</li>"
                        return mm.group(0)
                    return f"{ol_open}<li>{inner}</li>"

                cleaned = re.sub(
                    r"(?is)(<ol[^>]*>)\s*<p[^>]*>\s*(.*?)\s*</p>",
                    _leading_orphan_p_to_li,
                    cleaned,
                    count=1,
                )
                # Some weak repairs emit "<ol><p>Unsicherheit: ...</p><li>...</li></ol>".
                # Normalize these orphan primary rows into regular list items.
                cleaned = re.sub(
                    r"(?is)<p([^>]*)>\s*(?:<strong>\s*)?(?:Unsicherheit|Uncertainty)(?:\s*</strong>)?\s*:?\s*(.*?)\s*</p>",
                    lambda mm: (
                        f"<li><strong>{prim_canon}</strong>: {(mm.group(2) or '').strip()}</li>"
                        if (mm.group(2) or "").strip()
                        else mm.group(0)
                    ),
                    cleaned,
                )
                # If list items are present but primary labels are missing, add a deterministic
                # primary label so SD points stay machine-detectable and stable.
                def _prefix_missing_primary_li(mm: re.Match) -> str:
                    attrs = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "").strip()
                    if not inner:
                        return mm.group(0)
                    if re.match(
                        rf"(?is)^\s*(?:<strong>\s*)?(?:{_PRIMARY_LABELS_TOKEN_RX})(?:\s*</strong>)?\s*:",
                        inner,
                    ):
                        return mm.group(0)
                    if re.match(
                        rf"(?is)^\s*(?:<strong>\s*)?(?:{_SECONDARY_LABELS_TOKEN_RX})(?:\s*</strong>)?\s*:",
                        inner,
                    ):
                        return mm.group(0)
                    plain = re.sub(r"(?is)<[^>]+>", " ", inner)
                    plain = re.sub(r"\s+", " ", plain).strip().lower()
                    if re.fullmatch(r"\*+\s*(schwäche|weakness)\s*\*?", plain):
                        return mm.group(0)
                    if plain.startswith(("verification route", "source:", "quelle:")):
                        return mm.group(0)
                    return f"<li{attrs}><strong>{prim_canon}</strong>: {inner}</li>"

                cleaned = re.sub(
                    r"(?is)<li([^>]*)>\s*(.*?)\s*</li>",
                    _prefix_missing_primary_li,
                    cleaned,
                )
                cleaned = _strip_uncertainty_tail_fragments(cleaned)
                cleaned = re.sub(r"(?is)<li([^>]*)>\s*</li>", "", cleaned)
                cleaned = re.sub(
                    rf"(?is)<li([^>]*)>\s*(?:<strong>\s*)?(?:{_PRIMARY_LABELS_TOKEN_RX})(?:\s*</strong>)?\s*:?\s*</li>",
                    "",
                    cleaned,
                )
                # Drop redundant SD tail points like "Schwäche: U1 - Data gap. Needed: ..."
                # when at least two real SD points are already present.
                li_matches = list(re.finditer(r"(?is)<li[^>]*>.*?</li>", cleaned))
                if len(li_matches) > 2:
                    drop_idx = [i for i, mm in enumerate(li_matches) if _looks_like_uncertainty_tail_point(mm.group(0))]
                    max_drop = max(0, len(li_matches) - 2)
                    if drop_idx and max_drop > 0:
                        to_drop = set(drop_idx[:max_drop])
                        rebuilt = []
                        cursor = 0
                        for i, mm in enumerate(li_matches):
                            rebuilt.append(cleaned[cursor:mm.start()])
                            if i not in to_drop:
                                rebuilt.append(mm.group(0))
                            cursor = mm.end()
                        rebuilt.append(cleaned[cursor:])
                        cleaned = "".join(rebuilt)
                cleaned = re.sub(r"(?is)>\s*\*\s*(?=<br)", ">", cleaned)
                cleaned = re.sub(
                    rf"(?is)([^>\n])\s+(?=(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:)",
                    r"\1<br>",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                # Canonicalize + bold secondary labels even when models emit lowercase variants.
                canonical_sec = [
                    ("Why it matters", "Why it matters"),
                    ("Why this is important", "Why it matters"),
                    ("Warum relevant", "Warum relevant"),
                    ("Warum es wichtig ist", "Warum das wichtig ist"),
                    ("Warum das wichtig ist", "Warum das wichtig ist"),
                    ("What would verify/falsify (next check)", "What would verify/falsify (next check)"),
                    ("What would verify or falsify (next check)", "What would verify or falsify (next check)"),
                    ("Was würde verifizieren/falsifizieren (nächster Check)", "Was würde verifizieren/falsifizieren (nächster Check)"),
                    ("Was würde verifizieren oder falsifizieren (nächster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)"),
                    ("Was wuerde verifizieren/falsifizieren (naechster Check)", "Was würde verifizieren/falsifizieren (nächster Check)"),
                    ("Was wuerde verifizieren oder falsifizieren (naechster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)"),
                    ("Next check", "Next check"),
                    ("Next step", "Next step"),
                    ("Nächster Check", "Nächster Check"),
                    ("Nächster Schritt", "Nächster Schritt"),
                    ("Nächste Prüfung", "Nächste Prüfung"),
                    ("Prüfen/Widerlegen (nächster Schritt)", "Prüfen/Widerlegen (nächster Schritt)"),
                    ("Naechster Check", "Nächster Check"),
                    ("Naechster Schritt", "Nächster Schritt"),
                    ("Naechste Pruefung", "Nächste Prüfung"),
                    ("Pruefen/Widerlegen (naechster Schritt)", "Prüfen/Widerlegen (nächster Schritt)"),
                    ("Vereinfachung", "Vereinfachung"),
                    ("Simplification", "Simplification"),
                    ("Subjektivität", "Subjektivität"),
                    ("Subjectivity", "Subjectivity"),
                ]
                sec_canon_map = {str(_pat).lower(): _canon for _pat, _canon in canonical_sec}
                # Normalize italic/markdown-emphasized secondary labels to a bold canonical form.
                # Example from regression logs:
                #   <em><strong>Warum das wichtig ist</strong>:</em>:
                # -> <strong>Warum das wichtig ist</strong>:
                def _canon_secondary_label(raw: str) -> str:
                    key = str(raw or "").strip().lower()
                    return sec_canon_map.get(key, str(raw or "").strip())

                def _norm_em_secondary(mm: re.Match) -> str:
                    _raw = re.sub(r"(?is)<[^>]+>", "", mm.group("label") or "").strip()
                    if not _raw:
                        return mm.group(0)
                    return f"<strong>{_canon_secondary_label(_raw)}</strong>: "

                cleaned = re.sub(
                    rf"(?is)<em>\s*(?:<strong>\s*)?(?P<label>{sec_rx})(?:\s*</strong>)?\s*:?\s*</em>\s*:?\s*",
                    _norm_em_secondary,
                    cleaned,
                )
                cleaned = re.sub(
                    rf"(?is)\*+\s*(?P<label>{sec_rx})\s*:?\s*\*+\s*:?\s*",
                    _norm_em_secondary,
                    cleaned,
                )
                cleaned = re.sub(
                    rf"(?is)(^|<br\s*/?>|>\s*)(?:<strong>\s*)?(?P<label>{sec_rx})(?:\s*</strong>)?\s*(?:\*{{1,2}}|__)+\s*:?\s*",
                    lambda mm: f"{mm.group(1)}<strong>{_canon_secondary_label(mm.group('label') or '')}</strong>: ",
                    cleaned,
                )
                for _pat, _canon in canonical_sec:
                    cleaned = re.sub(
                        rf"(?is)(^|<br\s*/?>|>\s*)(?:(?:<strong>\s*)?{re.escape(_pat)}(?:\s*</strong>)?)(?!\w)\s*:?\s*",
                        rf"\1<strong>{_canon}</strong>: ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                # Remove accidental nested <strong> tags introduced by mixed markdown/html repairs.
                cleaned = re.sub(r"(?is)<strong>\s*<strong>\s*", "<strong>", cleaned)
                cleaned = re.sub(r"(?is)</strong>\s*</strong>", "</strong>", cleaned)
                # Re-run line-break normalization after label canonicalization so converted
                # "<strong>Warum ...</strong>:" / "<strong>What would ...</strong>:" rows
                # always start on a new visual line in every list item.
                cleaned = re.sub(
                    rf"(?is)([^>\n])\s+(?=(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:)",
                    r"\1<br>",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(r"(?is)</strong>:\s+<", "</strong>:<", cleaned)
                # Some weak models/HTML conversions split a single logical <li> item into
                # "<li>...Weakness...</li><p>Why...</p><p>What would...</p>" inside the same <ol>.
                # Normalize those sibling <p> rows as well (bold + canonical colon).
                for _pat, _canon in canonical_sec:
                    cleaned = re.sub(
                        rf"(?is)(<p[^>]*>\s*)(?:<strong>\s*)?{re.escape(_pat)}(?:\s*</strong>)?(?!\w)\s*:?\s*",
                        rf"\1<strong>{_canon}</strong>: ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                # Handle fragmented primary items from weak markdown conversion:
                # "<li>*Schwäche</li><p>*:</p><p>🟡</p><p>Text...</p>" -> one canonical <li>.
                split_primary_li = re.compile(
                    rf"(?is)<li[^>]*>\s*(?:<strong>\s*)?\*?\s*(Schwäche|Weakness)\s*\*?(?:\s*</strong>)?\s*</li>"
                    rf"(?:\s*<p[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</p>)*"
                    rf"(?:\s*<p[^>]*>\s*[🟢🟡🔴]\s*</p>)*"
                    rf"\s*<p[^>]*>\s*(?!\s*(?:<strong>\s*)?(?:{sec_rx}))(.*?)\s*</p>"
                )

                def _merge_split_primary_li(mm: re.Match) -> str:
                    _label = (mm.group(1) or "").strip()
                    _txt = (mm.group(2) or "").strip()
                    if not _txt:
                        return mm.group(0)
                    return f"<li><strong>{_label}</strong>: {_txt}</li>"

                cleaned = split_primary_li.sub(_merge_split_primary_li, cleaned)
                # Merge sibling secondary rows back into the preceding item:
                #   </li><li>Warum ...</li><p>Text</p>  ->  <br><strong>Warum ...</strong>: Text</li>
                #   </li><p><strong>Nächster Check</strong>:</p><p>Text</p> -> same
                split_secondary_rows = re.compile(
                    rf"(?is)</li>\s*<(?:li|p)[^>]*>\s*(?:<strong>\s*)?(?P<label>{sec_rx})(?:\s*</strong>)?\s*:?\s*</(?:li|p)>"
                    rf"(?:\s*<p[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</p>)*"
                    rf"(?:\s*<p[^>]*>\s*[🟢🟡🔴]\s*</p>)*"
                    rf"\s*<p[^>]*>\s*(?P<txt>.*?)\s*</p>"
                )

                def _merge_split_secondary_rows(mm: re.Match) -> str:
                    _lab_raw = re.sub(r"(?is)<[^>]+>", "", mm.group("label") or "").strip()
                    _lab = sec_canon_map.get(_lab_raw.lower(), _lab_raw)
                    _txt = (mm.group("txt") or "").strip()
                    if not _lab or not _txt:
                        return mm.group(0)
                    return f"<br><strong>{_lab}</strong>: {_txt}</li>"

                for _ in range(8):
                    _new = split_secondary_rows.sub(_merge_split_secondary_rows, cleaned)
                    if _new == cleaned:
                        break
                    cleaned = _new
                # If a secondary label became a separate <li>, merge it into the previous weakness item.
                split_secondary_li = re.compile(
                    rf"(?is)</li>\s*<li([^>]*)>\s*((?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?)(?=</li>)</li>"
                )

                def _merge_split_secondary_li(mm: re.Match) -> str:
                    _inner = (mm.group(2) or "").strip()
                    if not _inner:
                        return mm.group(0)
                    return f"<br>{_inner}</li>"

                for _ in range(8):
                    _new = split_secondary_li.sub(_merge_split_secondary_li, cleaned)
                    if _new == cleaned:
                        break
                    cleaned = _new
                # Some renderers split one logical SD ordered list into multiple tiny
                # "<ol><li>Weakness...</li></ol>" chunks and place secondary fields
                # as paragraphs between these chunks. Merge those fragments back so each
                # Weakness item contains its secondary rows and the block has one stable <ol>.
                split_ol_chunks_with_secondary_paras = re.compile(
                    rf"(?is)</li>\s*</ol>\s*(?P<paras>(?:<p[^>]*>\s*(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?</p>\s*)+)\s*<ol[^>]*>\s*(?P<li_open><li[^>]*>)"
                )

                def _merge_split_ol_chunks_with_secondary_paras(mm: re.Match) -> str:
                    paras_blob = str(mm.group("paras") or "")
                    li_open = str(mm.group("li_open") or "<li>")
                    rows = []
                    for pm in re.finditer(r"(?is)<p[^>]*>\s*(.*?)\s*</p>", paras_blob):
                        inner = str(pm.group(1) or "").strip()
                        if inner:
                            rows.append(inner)
                    if not rows:
                        return mm.group(0)
                    return "".join(f"<br>{row}" for row in rows) + "</li>" + li_open

                for _ in range(8):
                    _new = split_ol_chunks_with_secondary_paras.sub(
                        _merge_split_ol_chunks_with_secondary_paras,
                        cleaned,
                    )
                    if _new == cleaned:
                        break
                    cleaned = _new
                # Also handle the terminal case where the last tiny <ol> chunk is followed by
                # secondary paragraphs before the SD box closes.
                trailing_ol_secondary_paras = re.compile(
                    rf"(?is)</li>\s*</ol>\s*(?P<paras>(?:<p[^>]*>\s*(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?</p>\s*)+)(?=(?:</div>|<p[^>]*>\s*QC(?:-Matrix)?\s*:|$))"
                )

                def _merge_trailing_ol_secondary_paras(mm: re.Match) -> str:
                    paras_blob = str(mm.group("paras") or "")
                    rows = []
                    for pm in re.finditer(r"(?is)<p[^>]*>\s*(.*?)\s*</p>", paras_blob):
                        inner = str(pm.group(1) or "").strip()
                        if inner:
                            rows.append(inner)
                    if not rows:
                        return mm.group(0)
                    return "".join(f"<br>{row}" for row in rows) + "</li></ol>"

                for _ in range(8):
                    _new = trailing_ol_secondary_paras.sub(
                        _merge_trailing_ol_secondary_paras,
                        cleaned,
                    )
                    if _new == cleaned:
                        break
                    cleaned = _new
                # Handle a variant where the trailing secondary <p>-rows are followed by
                # a dangling opening "<ol>" (its closing tag sits in the outer block tail).
                trailing_ol_secondary_paras_with_open_ol = re.compile(
                    rf"(?is)</li>\s*</ol>\s*(?P<paras>(?:<p[^>]*>\s*(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?</p>\s*)+)\s*<ol[^>]*>\s*$"
                )

                def _merge_trailing_ol_secondary_paras_with_open_ol(mm: re.Match) -> str:
                    paras_blob = str(mm.group("paras") or "")
                    rows = []
                    for pm in re.finditer(r"(?is)<p[^>]*>\s*(.*?)\s*</p>", paras_blob):
                        inner = str(pm.group(1) or "").strip()
                        if inner:
                            rows.append(inner)
                    if not rows:
                        return mm.group(0)
                    # Keep only one closing </ol>: the outer block tail already carries it.
                    return "".join(f"<br>{row}" for row in rows) + "</li>"

                for _ in range(8):
                    _new = trailing_ol_secondary_paras_with_open_ol.sub(
                        _merge_trailing_ol_secondary_paras_with_open_ol,
                        cleaned,
                    )
                    if _new == cleaned:
                        break
                    cleaned = _new
                # If list fragments now directly touch, keep a single continuous <ol>.
                cleaned = re.sub(r"(?is)</ol>\s*<ol[^>]*>\s*", "", cleaned)
                cleaned = re.sub(r"(?is)(</ol>\s*){2,}", "</ol>", cleaned)
                cleaned = _repair_broken_label_colon_markup(cleaned)
                cleaned = re.sub(r"(?is)(<strong>[^<]+</strong>:)\s*</strong>", r"\1", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*\*\s*</\1>", "", cleaned)
                # If a logical list item was split into "</li><p>Why...</p><p>Next check...</p>",
                # merge those secondary paragraphs back into the preceding <li> using <br>.
                split_li_paras = re.compile(
                    rf"(?is)</li>((?:\s*<p[^>]*>\s*(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?</p>)+)"
                )

                def _merge_split_li_paras(mm: re.Match) -> str:
                    paras_blob = mm.group(1) or ""
                    parts = []
                    for pm in re.finditer(r"(?is)<p[^>]*>\s*(.*?)\s*</p>", paras_blob):
                        inner = (pm.group(1) or "").strip()
                        if inner:
                            parts.append(inner)
                    if not parts:
                        return mm.group(0)
                    return "".join(f"<br>{p}" for p in parts) + "</li>"

                cleaned = split_li_paras.sub(_merge_split_li_paras, cleaned)
                # Repair broken inline tag nests in malformed list items like
                # "<li><strong>Schwäche</strong>:<p><strong></strong>Schwäche<strong><em>*: ...".
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*<strong>\s*(?:Schwäche|Weakness)\s*</strong>\s*:)\s*<p[^>]*>\s*",
                    r"\1<br>",
                    cleaned,
                )
                cleaned = re.sub(r"(?is)</p>\s*</li>", "</li>", cleaned)
                cleaned = re.sub(r"(?is)<strong>\s*</strong>", "", cleaned)
                cleaned = re.sub(r"(?is)</em>\s*<strong>", "", cleaned)
                cleaned = re.sub(r"(?is)<em>\s*\*:\s*", "", cleaned)
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*<strong>\s*(?:Schwäche|Weakness)\s*</strong>\s*:\s*)<strong>\s*",
                    r"\1",
                    cleaned,
                )
                cleaned = re.sub(
                    r"(?is)</strong>\s*(<br>\s*<strong>\s*(?:Warum das wichtig ist|Why it matters|Was würde verifizieren/falsifizieren \(nächster Check\)|What would verify/falsify \(next check\)|Nächster Check|Next check))",
                    r"\1",
                    cleaned,
                )
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*<strong>\s*(?:Schwäche|Weakness)\s*</strong>\s*:)\s*"
                    rf"(?:<br>\s*)?(?:<strong>\s*)?(?:Schwäche|Weakness)(?:\s*</strong>)?\s*",
                    r"\1 ",
                    cleaned,
                )
                # Keep primary label + content on the same visual line:
                # "<li><strong>Schwäche</strong>:<br>Text..." -> "<li><strong>Schwäche</strong>: Text..."
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*(?:<strong>\s*)?(?:{_PRIMARY_LABELS_RX})(?:\s*</strong>)?\s*:)\s*<br\s*/?>\s*",
                    r"\1 ",
                    cleaned,
                )
                cleaned = re.sub(
                    r"(?is)(<strong>\s*(?:Schwäche|Weakness)\s*</strong>:\s*)(?=[^<\s])",
                    r"\1 ",
                    cleaned,
                )
                # Ensure each SD item includes mandatory secondary fields
                # even when the source only contains bare Weakness rows.
                reason_lbl = "Warum das wichtig ist" if lang.lower().startswith("de") else "Why it matters"
                check_lbl = (
                    "Was würde verifizieren/falsifizieren (nächster Check)"
                    if lang.lower().startswith("de")
                    else "What would verify/falsify (next check)"
                )
                reason_txt = (
                    "Die benannte Schwäche kann Reichweite, Präzision oder Belastbarkeit der Aussage einschränken."
                    if lang.lower().startswith("de")
                    else "This weakness can reduce scope, precision, or robustness of the claim."
                )
                check_txt = (
                    "Den betroffenen Punkt mit Primärquelle, Gegenbeispiel oder Zusatzkontext gezielt nachprüfen."
                    if lang.lower().startswith("de")
                    else "Verify the affected point using a primary source, a concrete counterexample, or added context."
                )

                def _ensure_secondary_fields_li(mm: re.Match) -> str:
                    attrs = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "").strip()
                    if not inner:
                        return mm.group(0)
                    if re.match(
                        rf"(?is)^\s*(?:<strong>\s*)?(?:{_SECONDARY_LABELS_TOKEN_RX})(?:\s*</strong>)?\s*:",
                        inner,
                    ):
                        return mm.group(0)
                    plain = re.sub(r"(?is)<[^>]+>", " ", inner)
                    plain = re.sub(r"\s+", " ", plain).strip()
                    has_reason = bool(
                        re.search(
                            r"(?i)\b(?:why it matters|why this is important|warum relevant|warum es wichtig ist|warum das wichtig ist)\b",
                            plain,
                        )
                    )
                    has_check = bool(
                        re.search(
                            r"(?i)\b(?:what would verify/falsify \(next check\)|what would verify or falsify \(next check\)|"
                            r"was würde verifizieren/falsifizieren \(nächster check\)|was würde verifizieren oder falsifizieren \(nächster check\)|"
                            r"prüfen/widerlegen \(nächster schritt\)|next check|next step|nächster check|nächster schritt|nächste prüfung)\b",
                            plain,
                        )
                    )
                    if has_reason and has_check:
                        return mm.group(0)
                    inner_norm = re.sub(r"(?is)(?:<br\s*/?>\s*)+$", "", inner).strip()
                    tail = ""
                    if not has_reason:
                        tail += f"<br><strong>{reason_lbl}</strong>: {reason_txt}"
                    if not has_check:
                        tail += f"<br><strong>{check_lbl}</strong>: {check_txt}"
                    return f"<li{attrs}>{inner_norm}{tail}</li>"

                cleaned = re.sub(
                    r"(?is)<li([^>]*)>\s*(.*?)\s*</li>",
                    _ensure_secondary_fields_li,
                    cleaned,
                )
                # Repair a rare but visible corruption where fallback secondary labels
                # are prepended right after the primary label and the canonical labels
                # appear again later in the same item.
                reason_token_rx = (
                    r"(?:Warum\s+(?:das|es)\s+wichtig\s+ist|Warum\s+relevant|"
                    r"Why\s+it\s+matters|Why\s+this\s+is\s+important)"
                )
                check_token_rx = (
                    r"(?:Was\s+(?:würde|wuerde)\s+verifizieren(?:/|\s+oder\s+)falsifizieren\s*\((?:nächster|naechster)\s+Check\)|"
                    r"What\s+would\s+verify(?:/|\s+or\s+)falsify\s*\(next\s+check\)|"
                    r"N(?:ä|ae)chster\s+(?:Check|Schritt)|Next\s+(?:check|step)|"
                    r"Pr(?:ü|ue)fen/Widerlegen\s*\((?:nächster|naechster)\s+Schritt\))"
                )
                lead_secondary_rx = re.compile(
                    rf"(?is)^\s*(?:<strong>\s*)?(?:{_PRIMARY_LABELS_RX})(?:\s*</strong>)?\s*:\s*"
                    rf"(?:<br\s*/?>\s*)*(?:<strong>\s*)?(?:{reason_token_rx}|{check_token_rx})(?:\s*</strong>)?\s*:"
                )
                reason_label_rx = re.compile(
                    rf"(?is)(?:<strong>\s*)?(?:{reason_token_rx})(?:\s*</strong>)?\s*:"
                )
                check_label_rx = re.compile(
                    rf"(?is)(?:<strong>\s*)?(?:{check_token_rx})(?:\s*</strong>)?\s*:"
                )
                de_fallback_pair_rx = re.compile(
                    r"(?is)^\s*"
                    r"(?:<strong>\s*)?Warum\s+das\s+wichtig\s+ist(?:\s*</strong>)?\s*:\s*"
                    r"Die\s+benannte\s+Schwäche\s+kann\s+Reichweite,\s+Präzision\s+oder\s+Belastbarkeit\s+der\s+Aussage\s+einschränken\.\s*"
                    r"<br\s*/?>\s*"
                    r"(?:<strong>\s*)?Was\s+w(?:ü|ue)rde\s+verifizieren/falsifizieren\s*\((?:nächster|naechster)\s+Check\)(?:\s*</strong>)?\s*:\s*"
                    r"Den\s+betroffenen\s+Punkt\s+mit\s+Primärquelle,\s+Gegenbeispiel\s+oder\s+Zusatzkontext\s+gezielt\s+nachprüfen\.\s*"
                    r"(?:<br\s*/?>\s*)?"
                )
                en_fallback_pair_rx = re.compile(
                    r"(?is)^\s*"
                    r"(?:<strong>\s*)?Why\s+it\s+matters(?:\s*</strong>)?\s*:\s*"
                    r"This\s+weakness\s+can\s+reduce\s+scope,\s+precision,\s+or\s+robustness\s+of\s+the\s+claim\.\s*"
                    r"<br\s*/?>\s*"
                    r"(?:<strong>\s*)?What\s+would\s+verify/falsify\s*\(next\s+check\)(?:\s*</strong>)?\s*:\s*"
                    r"Verify\s+the\s+affected\s+point\s+using\s+a\s+primary\s+source,\s+a\s+concrete\s+counterexample,\s+or\s+added\s+context\.\s*"
                    r"(?:<br\s*/?>\s*)?"
                )

                def _repair_leading_secondary_duplicate_pair_li(mm: re.Match) -> str:
                    attrs = str(mm.group(1) or "")
                    inner = str(mm.group(2) or "").strip()
                    if not inner:
                        return mm.group(0)
                    if lead_secondary_rx.search(inner) is None:
                        return mm.group(0)
                    reason_count = len(reason_label_rx.findall(inner))
                    check_count = len(check_label_rx.findall(inner))
                    if reason_count < 2 or check_count < 2:
                        return mm.group(0)
                    pm = re.match(
                        rf"(?is)^\s*(?P<pfx>(?:<strong>\s*)?(?:{_PRIMARY_LABELS_RX})(?:\s*</strong>)?\s*:)\s*",
                        inner,
                    )
                    if pm is None:
                        return mm.group(0)
                    pfx = str(pm.group("pfx") or "").strip()
                    rest = inner[pm.end():]
                    rest_new = de_fallback_pair_rx.sub("", rest, count=1)
                    if rest_new == rest:
                        rest_new = en_fallback_pair_rx.sub("", rest, count=1)
                    if rest_new == rest:
                        return mm.group(0)
                    rest_new = re.sub(r"(?is)^\s*(?:<br\s*/?>\s*)+", "", rest_new).strip()
                    if not rest_new:
                        return mm.group(0)
                    return f"<li{attrs}>{pfx} {rest_new}</li>"

                cleaned = re.sub(
                    r"(?is)<li([^>]*)>\s*(.*?)\s*</li>",
                    _repair_leading_secondary_duplicate_pair_li,
                    cleaned,
                )
                cleaned = _drop_verification_route_marker_segments_html(cleaned)
                cleaned = _repair_broken_label_colon_markup(cleaned)
                return cleaned

            # Drop standalone marker lines ("1.", "2.") that cause visible double numbering.
            # Also normalize common markdown-leak patterns from weak model output where
            # labels appear as "*Schwäche*" and nested <strong> tags.
            body = re.sub(
                r"(?is)<strong>\s*<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?\s*:\s*</strong>",
                r"<strong>\1</strong>:",
                body,
            )
            body = re.sub(
                r"(?is)<strong>\s*<strong>\s*([^<]+?)\s*</strong>\s*:\s*</strong>",
                r"<strong>\1</strong>:",
                body,
            )
            body = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", body)
            body = re.sub(r"(?im)^\s*(?:<[^>]+>\s*)*\*\s*(?:</[^>]+>\s*)*$", "", body)
            body = re.sub(r"(?is)>\s*\*\s*(?=<br\s*/?>|</(?:p|div|li)>)", ">", body)
            body = re.sub(
                r'(?im)^\s*<div[^>]*>\s*\d+\.\s*</div>\s*$',
                '',
                body,
            )
            body = re.sub(
                r'(?im)^\s*\d+\.\s*(?:<br\s*/?>\s*)?$',
                '',
                body,
            )

            n = 0
            out_lines = []
            for ln in body.splitlines():
                # Localize the three canonical labels inside the block.
                if lang.lower().startswith('de'):
                    ln = ln.replace('Weakness:', 'Schwäche:')
                    ln = ln.replace('Uncertainty:', 'Schwäche:')
                    ln = ln.replace('Unsicherheit:', 'Schwäche:')
                    ln = ln.replace('Simplification:', 'Vereinfachung:')
                    ln = ln.replace('Subjectivity:', 'Subjektivität:')
                    ln = ln.replace('Why it matters:', 'Warum das wichtig ist:')
                    ln = ln.replace('What would verify/falsify (next check):', 'Was würde verifizieren/falsifizieren (nächster Check):')
                    ln = ln.replace('Next step:', 'Nächster Schritt:')
                    ln = ln.replace('<strong>Weakness</strong>:', '<strong>Schwäche</strong>:')
                    ln = ln.replace('<strong>Uncertainty</strong>:', '<strong>Schwäche</strong>:')
                    ln = ln.replace('<strong>Unsicherheit</strong>:', '<strong>Schwäche</strong>:')
                    ln = ln.replace('<strong>Simplification</strong>:', '<strong>Vereinfachung</strong>:')
                    ln = ln.replace('<strong>Subjectivity</strong>:', '<strong>Subjektivität</strong>:')
                    ln = ln.replace('<strong>Why it matters</strong>:', '<strong>Warum das wichtig ist</strong>:')
                    ln = ln.replace(
                        '<strong>What would verify/falsify (next check)</strong>:',
                        '<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:'
                    )
                    ln = ln.replace('<strong>Next step</strong>:', '<strong>Nächster Schritt</strong>:')
                else:
                    ln = ln.replace('Schwäche:', 'Weakness:')
                    ln = ln.replace('Unsicherheit:', 'Weakness:')
                    ln = ln.replace('Vereinfachung:', 'Simplification:')
                    ln = ln.replace('Subjektivität:', 'Subjectivity:')
                    ln = ln.replace('Warum das wichtig ist:', 'Why it matters:')
                    ln = ln.replace('Was würde verifizieren/falsifizieren (nächster Check):', 'What would verify/falsify (next check):')
                    ln = ln.replace('Nächster Schritt:', 'Next step:')
                    ln = ln.replace('<strong>Unsicherheit</strong>:', '<strong>Weakness</strong>:')

                ln = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", ln)
                ln = re.sub(r"(?i)\*+\s*(Schwäche|Weakness)\s*\*+\s*:", r"\1:", ln)
                ln = re.sub(r"(?is)<strong>\s*<strong>\s*", "<strong>", ln)
                ln = re.sub(r"(?is)</strong>\s*</strong>", "</strong>", ln)
                ln = re.sub(r"(?is)(<strong>[^<]+</strong>:)\s*</strong>", r"\1", ln)

                # Bold known labels (formatting only).
                for lab in _PRIMARY_LABELS + _SECONDARY_LABELS:
                    ln = re.sub(
                        rf"(?i)\b{re.escape(lab)}\s*:",
                        rf"<strong>{lab}</strong>:",
                        ln,
                    )

                # Force secondary field labels onto a new visual line if they leaked inline after
                # the Weakness sentence (common with weaker models / compact HTML rendering).
                _sec_rx = _SECONDARY_LABELS_TOKEN_RX
                ln = re.sub(
                    rf"(?is)([^>\n])\s+(?=(?:<strong>\s*)?(?:{_sec_rx})(?:\s*</strong>)?\s*:)",
                    r"\1<br>",
                    ln,
                    flags=re.IGNORECASE,
                )

                # Count/number only Weakness lines.
                plain = re.sub(r'<[^>]+>', '', ln).strip()
                plain_norm = re.sub(r"\*+", "", plain).strip()
                # Non-Weakness field labels must not carry list numbering.
                if re.match(
                    rf"(?i)^(?:\d+\.\s*)(?:{_SECONDARY_LABELS_TOKEN_RX})\s*:",
                    plain_norm,
                ):
                    ln = re.sub(r'(?i)(<div[^>]*>\s*)\d+\.\s*', r'\1', ln, count=1)
                    ln = re.sub(r'(?i)^\s*\d+\.\s*', '', ln, count=1)
                    plain = re.sub(r'<[^>]+>', '', ln).strip()
                    plain_norm = re.sub(r"\*+", "", plain).strip()
                if re.match(rf'(?i)^(?:\d+\.\s*)?(?:{_PRIMARY_LABELS_RX})\b\s*:', plain_norm):
                    n += 1
                    # Remove any existing numeric prefix immediately after opening div/start.
                    ln = re.sub(r'(?i)(<div[^>]*>\s*)\d+\.\s*', r'\1', ln, count=1)
                    ln = re.sub(r'(?i)^\s*\d+\.\s*', '', ln, count=1)
                    if '<div' in ln:
                        ln = re.sub(r'(<div[^>]*>\s*)', lambda m: f"{m.group(1)}{n}. ", ln, count=1)
                    else:
                        ln = f'{n}. ' + ln

                out_lines.append(ln)

            return _repair_broken_label_colon_markup('\n'.join([x for x in out_lines if x is not None]))

        # Primary path: normalize explicit self-debunking boxes.
        # Prefer full-box capture that ends at "</ol></div>" so nested title <div> does not
        # truncate the match at the first closing </div>.
        block_re_ol = re.compile(
            r'(?is)(<div[^>]*class=(?:"|\')[^"\']*self-debunking[^"\']*(?:"|\')[^>]*>)(.*?)(</ol>\s*</div>)'
        )
        block_re_fallback = re.compile(
            r'(?is)(<div[^>]*class=(?:"|\')[^"\']*self-debunking[^"\']*(?:"|\')[^>]*>)(.*?)(</div>)'
        )

        def _block_sub(mb: re.Match) -> str:
            return mb.group(1) + _normalize_block(mb.group(2)) + mb.group(3)

        out_text = block_re_ol.sub(_block_sub, html_text)
        # Fallback for non-<ol> legacy SD containers.
        if out_text == html_text:
            out_text = block_re_fallback.sub(_block_sub, html_text)

        # Fallback path: line-wise region between Self-Debunking header and QC footer.
        lines = out_text.splitlines()
        out = []
        in_sd = False
        chunk = []

        def _flush_chunk() -> None:
            nonlocal chunk
            if chunk:
                out.extend(_normalize_block('\n'.join(chunk)).splitlines())
                chunk = []

        for ln in lines:
            plain = re.sub(r'<[^>]+>', '', ln)
            if (not in_sd) and re.search(r'(?i)Self-Debunking|Selbst[- ]?Debunking', plain):
                in_sd = True
                out.append(ln)
                continue
            if in_sd and re.search(r'(?im)^\s*QC(?:-Matrix)?\s*:', plain):
                _flush_chunk()
                in_sd = False
                out.append(ln)
                continue
            if in_sd:
                chunk.append(ln)
            else:
                out.append(ln)

        _flush_chunk()
        return '\n'.join(out)
    except Exception:
        return html_text

def detect_self_debunking_numbered_html(html_text: str) -> bool:
    """Audit helper: detect numbered Self-Debunking points in rendered HTML."""
    try:
        if not html_text:
            return False
        raw = str(html_text)
        if re.search(r"(?i)self[- ]?debunking|selbst[- ]?debunking", raw) is None:
            return False
        plain = re.sub(r"<[^>]+>", "\n", raw)
        m = re.search(
            r"(?is)(Self[- ]?Debunking|Selbst[- ]?Debunking).*?(?:\n\s*QC(?:-Matrix)?\s*:|\Z)",
            plain,
        )
        block = m.group(0) if m else plain
        if re.search(rf"(?im)^\s*\d+\.\s*(?:{_PRIMARY_LABELS_RX})\b", block):
            return True
        # HTML <ol>/<li> numbering is visual and may not appear as text prefixes.
        if re.search(
            rf"(?is)<ol[^>]*>.*?<li[^>]*>.*?(?:<strong>\s*)?(?:{_PRIMARY_LABELS_RX})(?:\s*</strong>)?\s*:.*?</li>",
            raw,
        ):
            return True
        return False
    except Exception:
        return False


def enforce_self_debunking_contract(text: str, gov_mgr, profile_name: str, *, is_command: bool = False, lang: str = "en") -> str:
    """Deterministically enforce the Self-Debunking contract (2–3 numbered points) when required.

    Note: is_command disables enforcement for command-only responses.

    - If missing: inject a minimal compliant block (no new factual claims).
    - If too many points: keep the first 3.
    - If too few points: add generic points to reach the minimum.
    - Ensure placement BEFORE the QC footer.
    """
    try:
        # Commands must NEVER trigger Self-Debunking enforcement/injection.
        if is_command:
            return text
        if not text or not gov_mgr or not getattr(gov_mgr, 'loaded', False):
            return text

        gd = (gov_mgr.data.get('global_defaults', {}) or {})
        oc = (gd.get('output_contract', {}) or {})
        contract = (oc.get('self_debunking_contract', {}) or {})
        if not contract.get('enabled', False):
            return text

        module = (gd.get('self_debunking', {}) or {})
        if not module.get('enabled', False):
            return text

        exceptions = set(module.get('exceptions') or [])
        if (profile_name or '') in exceptions:
            return text

        title = (contract.get('required_block_title') or (module.get('block', {}) or {}).get('title') or 'Self-Debunking').strip() or 'Self-Debunking'

        min_p = int(contract.get('required_min_points', 2) or 2)
        max_p = int(contract.get('required_max_points', 3) or 3)
        min_p = 2 if min_p < 2 else min_p
        max_p = 3 if max_p > 6 else max_p  # safety cap

        def _finalize_sd(t: str) -> str:
            # Apply language normalization first (for DE) then bold labels (formatting only).
            t2 = normalize_self_debunking_language(t, lang)
            t2 = normalize_self_debunking_field_linebreaks(t2, lang=lang)
            t2 = bold_self_debunking_labels(t2, lang)
            t2 = normalize_self_debunking_numbering_text(t2, lang=lang)
            return t2

        # Locate QC footer (insertion anchor)
        qc_m = re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$", text)
        qc_pos = qc_m.start() if qc_m else None

        # Locate Self-Debunking title line
        title_re = re.compile(rf"(?im)^\s*(?:#+\s*)?\*{{0,2}}{re.escape(title)}\*{{0,2}}\s*:?\s*$")
        m = title_re.search(text)
        if not m:
            # Missing entirely -> inject minimal
            out = inject_minimal_self_debunking(text, title=title, lang=lang)
            return _finalize_sd(out)
        # If title occurs after QC, remove that trailing block and inject at the correct place.
        if qc_pos is not None and m.start() > qc_pos:
            trimmed = text[:m.start()].rstrip()
            out = inject_minimal_self_debunking(trimmed, title=title, lang=lang)
            return _finalize_sd(out)

        # Extract the block region: from title line to QC footer (or end)
        end = qc_pos if qc_pos is not None else len(text)
        before = text[:m.end()]
        block = _strip_uncertainty_tail_fragments(text[m.end():end])
        after = text[end:] if end < len(text) else ""

        # Split numbered points (keep their multi-line bodies)
        # We detect lines starting with "<n>." or "<n>)".
        point_iter = list(re.finditer(r"(?m)^\s*(\d+)\s*[\.)]\s+", block))
        points = []
        for i, pm in enumerate(point_iter):
            p_start = pm.start()
            p_end = point_iter[i + 1].start() if i + 1 < len(point_iter) else len(block)
            points.append(block[p_start:p_end].rstrip())
        if points:
            points = [_strip_uncertainty_tail_fragments(p).rstrip() for p in points]
            points = [p for p in points if not _is_empty_sd_point(p)]
        # Filter redundant uncertainty tail points (e.g. "3. Weakness: U1 ... Needed: ...")
        # but only if at least the required minimum of substantive points remains.
        if len(points) > min_p:
            filtered = [p for p in points if not _looks_like_uncertainty_tail_point(p)]
            if len(filtered) >= min_p:
                points = filtered

        # If there are no numbered points at all, attempt to convert common unnumbered formats
        # (e.g., repeated 'Weakness:' blocks) into a numbered list without inventing new facts.
        if not points:
            # Try to extract unnumbered points from repeated label blocks inside this Self-Debunking section.
            extracted = []
            try:
                # Accept both plain labels and already-bolded markdown/HTML labels.
                # This avoids a failure mode where an earlier pass bolded labels (e.g. "**Schwäche**:")
                # and this extractor would no longer recognize unnumbered point starts.
                label_iter = list(
                    re.finditer(
                        rf"(?im)^\s*(?:\*\*|<strong>)?\s*(?:{_PRIMARY_LABELS_RX})\b\s*(?:\*\*|</strong>)?\s*:\s*",
                        block,
                    )
                )
                if label_iter:
                    for i, lm in enumerate(label_iter):
                        p_start = lm.start()
                        p_end = label_iter[i + 1].start() if i + 1 < len(label_iter) else len(block)
                        chunk = _strip_uncertainty_tail_fragments(block[p_start:p_end]).strip()
                        if chunk:
                            extracted.append(chunk)
            except Exception:
                extracted = []

            if extracted:
                # Trim to contract max and normalize to 1..k numbering.
                extracted = extracted[:max_p]
                normalized = []
                for i, chunk in enumerate(extracted, 1):
                    lines = chunk.splitlines()
                    if not lines:
                        continue
                    # Ensure the first line contains the label (Weakness/Schwäche) as in the model output.
                    first = lines[0].strip()
                    rest = [ln.rstrip() for ln in lines[1:]]
                    body = first
                    if rest:
                        body += "\n" + "\n".join("   " + ln.lstrip() for ln in rest if ln.strip() != "")
                    normalized.append(f"{i}. {body.strip()}")
                if normalized:
                    new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"
                    out = before.rstrip() + new_block + after.lstrip()
                    return _finalize_sd(out)

            # If the model used bullet points, convert the first 2–3 bullets into numbered points
            # without inventing content.
            try:
                bullet_lines = [ln.rstrip() for ln in block.splitlines()]
                bullets = []
                for ln in bullet_lines:
                    m_b = re.match(r"^\s*(?:[-*+]|•)\s+(.+?)\s*$", ln)
                    if m_b:
                        b = m_b.group(1).strip()
                        if b:
                            bullets.append(b)
                if bullets:
                    bullets = bullets[:max_p]
                    normalized = [f"{i}. {b}" for i, b in enumerate(bullets, 1)]
                    new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"
                    out = before.rstrip() + new_block + after.lstrip()
                    return _finalize_sd(out)
            except Exception:
                pass

            # Fallback: remove the broken/empty block and inject a minimal compliant numbered block.
            try:
                base = text[:m.start()].rstrip() + "\n\n" + text[end:].lstrip()
            except Exception:
                base = before.rstrip() + "\n\n" + after.lstrip()
            injected = inject_minimal_self_debunking(base, title=title, lang=lang)
            return _finalize_sd(injected)
        # Normalize number of points to the contract window.
        if len(points) > max_p:
            points = points[:max_p]
        while len(points) < min_p:
            n = len(points) + 1
            if str(lang).lower().startswith("de"):
                points.append(
                    f"{n}. Schwäche: Die Antwort könnte wichtige Einschränkungen oder Randbedingungen auslassen.\n"
                    f"   Warum das wichtig ist: Fehlende Vorbehalte können Sicherheit oder Gültigkeit überzeichnen.\n"
                    f"   Was würde verifizieren/falsifizieren (nächster Check): Formuliere ein konkretes Gegenbeispiel und prüfe, ob die Schlussfolgerung dann noch gilt."
                )
            else:
                points.append(
                    f"{n}. Weakness: The answer may omit important limitations or boundary conditions.\n"
                    f"   **Why it matters**: Missing caveats can overstate confidence or applicability.\n"
                    f"   **What would verify/falsify (next check)**: Identify a concrete counterexample and test whether the conclusion still holds."
                )
        # Some models emit only "Weakness/Schwäche" rows without mandatory secondary fields.
        # Add deterministic fallback lines so the SD structure remains stable across profiles/providers.
        def _ensure_secondary_fields(point: str) -> str:
            raw = str(point or "")
            if not raw.strip():
                return raw
            plain = re.sub(r"(?is)<[^>]+>", " ", raw)
            plain = plain.replace("**", "").replace("__", "")
            plain = re.sub(r"\s+", " ", plain).strip()

            has_reason = bool(
                re.search(
                    r"(?i)\b(?:why it matters|why this is important|warum relevant|warum es wichtig ist|warum das wichtig ist)\b",
                    plain,
                )
            )
            has_check = bool(
                re.search(
                    r"(?i)\b(?:what would verify/falsify \(next check\)|what would verify or falsify \(next check\)|"
                    r"was würde verifizieren/falsifizieren \(nächster check\)|was würde verifizieren oder falsifizieren \(nächster check\)|"
                    r"prüfen/widerlegen \(nächster schritt\)|next check|next step|nächster check|nächster schritt|nächste prüfung)\b",
                    plain,
                )
            )
            if has_reason and has_check:
                return raw

            lines = [ln.rstrip() for ln in raw.splitlines() if ln is not None]
            if not lines:
                return raw
            if str(lang).lower().startswith("de"):
                if not has_reason:
                    lines.append(
                        "   Warum das wichtig ist: Die benannte Schwäche kann Reichweite, Präzision "
                        "oder Belastbarkeit der Aussage einschränken."
                    )
                if not has_check:
                    lines.append(
                        "   Was würde verifizieren/falsifizieren (nächster Check): "
                        "Den betroffenen Punkt mit Primärquelle, Gegenbeispiel oder Zusatzkontext gezielt nachprüfen."
                    )
            else:
                if not has_reason:
                    lines.append(
                        "   Why it matters: This weakness can reduce scope, precision, or robustness of the claim."
                    )
                if not has_check:
                    lines.append(
                        "   What would verify/falsify (next check): "
                        "Verify the affected point using a primary source, a concrete counterexample, or added context."
                    )
            return "\n".join(lines).strip()

        points = [_ensure_secondary_fields(p) for p in points]
        # Re-number points sequentially (1..k)
        normalized = []
        for i, p in enumerate(points, 1):
            # Keep continuation lines inside the numbered item for stable Markdown rendering.
            lines = [ln.rstrip() for ln in str(p).splitlines()]
            if not lines:
                continue
            first = re.sub(r"^\s*\d+\s*[\.)]\s+", f"{i}. ", lines[0].strip(), count=1)
            rest = []
            for ln in lines[1:]:
                if ln.strip():
                    rest.append("   " + ln.lstrip())
            item = first if not rest else (first + "\n" + "\n".join(rest))
            normalized.append(item.strip())

        new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"

        # Reassemble
        out = before.rstrip() + new_block + after.lstrip()
        out = normalize_self_debunking_language(out, lang)
        out = normalize_self_debunking_field_linebreaks(out, lang=lang)
        out = bold_self_debunking_labels(out, lang)
        return out

    except Exception:
        return text
