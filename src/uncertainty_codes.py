from __future__ import annotations

import html
import re


_U_CODE_RE = re.compile(r"\b(U[1-8])\b")
_U_MARKED_RE = re.compile(r"(?i)data-u-code\s*=\s*(?:\"|')?(U[1-8])(?:\"|')?")
_BLOCK_TAG_RE = re.compile(r"(?is)<(p|li)([^>]*)>(.*?)</\1>")
_STATUS_KEY_RE = re.compile(r"(?i)\b(?:active profile|profile|overlay|sci|control layer|qc|cgi|color|comm)\s*:")
_TS_FOOTER_RE = re.compile(
    r"(?is)<div\b[^>]*class=(?:\"|')[^\"']*\bts-footer\b[^\"']*(?:\"|')[^>]*>.*?</div>"
)
_LEGEND_BLOCK_RE = re.compile(
    r"(?is)<details\b[^>]*class=(?:\"|')[^\"']*\buncertainty-legend\b[^\"']*(?:\"|')[^>]*>.*?</details>"
)
_CONTROL_LAYER_CLASS_BLOCK_RE = re.compile(
    r"(?is)<(div|details)\b[^>]*class=(?:\"|')[^\"']*\b(?:csc-warning|control-layer-note|control-layer-alert)\b[^\"']*(?:\"|')[^>]*>.*?</\1>"
)
_CONTROL_LAYER_LEGACY_BLOCK_RE = re.compile(
    r"(?is)<div\b[^>]*>\s*<b>\s*CONTROL\s+LAYER\s+(?:NOTE|ALERT|BLOCK)\s*</b>.*?</div>"
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_INLINE_CLOSING_TAG_RUN_RE = re.compile(
    r"(?is)(?:\s*</(?:span|strong|em|b|i|u|a|code|small|mark|sup|sub)\s*>)+"
)

_U_CODES = {
    "U1": {
        "de_name": "Datenluecke",
        "de_desc": "Es fehlen belastbare Quellen oder aktueller Kontext.",
        "en_name": "Data gap",
        "en_desc": "Reliable sources or current context are missing.",
    },
    "U2": {
        "de_name": "Annahmen unklar",
        "de_desc": "Begriffe oder Voraussetzungen sind mehrdeutig und muessen geklaert werden.",
        "en_name": "Assumption gap",
        "en_desc": "Terms or assumptions are ambiguous and must be clarified.",
    },
    "U3": {
        "de_name": "Perspektivenkonflikt",
        "de_desc": "Positionen oder Bewertungen widersprechen sich; Fakten und Werte trennen.",
        "en_name": "Perspective conflict",
        "en_desc": "Positions or value judgments conflict; separate facts from values.",
    },
    "U4": {
        "de_name": "Zeitliche Instabilitaet",
        "de_desc": "Faktlage kann sich kurzfristig aendern; aktuelle Verifikation noetig.",
        "en_name": "Temporal instability",
        "en_desc": "Facts may change quickly; current verification is needed.",
    },
    "U5": {
        "de_name": "Strukturelle Grenze",
        "de_desc": "Die Aufgabe hat methodische Grenzen; alternative Wege sollten benannt werden.",
        "en_name": "Structural limitation",
        "en_desc": "The task has methodological limits; alternatives should be stated.",
    },
    "U6": {
        "de_name": "Interpretationsspielraum",
        "de_desc": "Mehrere Deutungen sind plausibel; gezielte Rueckfrage empfohlen.",
        "en_name": "Interpretation ambiguity",
        "en_desc": "Multiple interpretations are plausible; targeted clarification is recommended.",
    },
    "U7": {
        "de_name": "Retrieval-Konflikt",
        "de_desc": "Abgerufene Quellen widersprechen sich; Konflikt muss explizit offengelegt werden.",
        "en_name": "Retrieval conflict",
        "en_desc": "Retrieved sources conflict; the conflict must be disclosed explicitly.",
    },
    "U8": {
        "de_name": "Retrieval-Metadatenluecke",
        "de_desc": "Provenienz-/Qualitaetsmetadaten fehlen oder Retrieval-Tools sind nicht verfuegbar.",
        "en_name": "Retrieval metadata gap",
        "en_desc": "Provenance/quality metadata is missing or retrieval tools are unavailable.",
    },
}


def _strip_html_tags(text: str) -> str:
    try:
        s = str(text or "")
    except Exception:
        return ""
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    return html.unescape(s)


def _normalize_text(text: str) -> str:
    s = _strip_html_tags(text)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_any(text_norm: str, needles: list[str]) -> bool:
    for n in needles:
        if n and n in text_norm:
            return True
    return False


_INFER_RULES = {
    "U1": [
        "missing data",
        "insufficient data",
        "missing source",
        "missing sources",
        "no source",
        "nicht belegt",
        "keine daten",
        "fehlende daten",
        "fehlende quellen",
        "ohne quelle",
    ],
    "U2": [
        "assumption",
        "assumptions",
        "depends on definition",
        "depends on assumptions",
        "annahme",
        "annahmen",
        "voraussetzung",
        "voraussetzungen",
        "mehrdeutig",
        "unklar",
    ],
    "U3": [
        "trade-off",
        "zielkonflikt",
        "widerspruch",
        "konflikt",
        "competing values",
        "interessenkonflikt",
    ],
    "U4": [
        "time-sensitive",
        "rapidly changing",
        "verification route gate",
        "aktuell verifizieren",
        "kurzfristig aendern",
        "kurzfristig ändern",
        "zeitlich instabil",
    ],
    "U5": [
        "no simple solution",
        "no perfect solution",
        "hard limitation",
        "structural limitation",
        "keine einfache loesung",
        "keine einfache lösung",
        "keine perfekte loesung",
        "keine perfekte lösung",
        "methodische grenze",
        "strukturelle grenze",
        "komplexe herausforderung",
    ],
    "U6": [
        "interpretation",
        "ambiguous",
        "multiple interpretations",
        "deutung",
        "interpretationsspielraum",
        "mehrere plausible",
        "nicht eindeutig",
    ],
    "U7": [
        "retrieval conflict",
        "source conflict",
        "conflicting sources",
        "widerspruechliche quellen",
        "quellenkonflikt",
        "konfligierende quelle",
    ],
    "U8": [
        "qualityclass",
        "quality class",
        "metadata gap",
        "provenance missing",
        "tool unavailable",
        "retrieval unavailable",
        "metadatenluecke",
        "metadatenluecke",
        "provenienz fehlt",
        "retrieval nicht verfuegbar",
    ],
}


_HIGH_COMPLEXITY_PROMPT_TERMS = [
    "objektiv beste",
    "dauerhaft fair",
    "weltweit",
    "einheitliches ki-regelwerk",
    "alle llms",
    "identische antworten",
    "ohne negative folgen",
    "in jeder sprache",
    "in jeder kultur",
    "rechtsordnung",
]

_INLINE_SKIP_TERMS = (
    "qc-matrix",
    "selbst-debunking",
    "self-debunking",
    "u-code-legende",
    "uncertainty code legend",
    "unsicherheitsmarker",
    "uncertainty markers",
    "control layer note",
    "active profile",
    "verification route",
)
_CONTROL_LAYER_ATTR_HINTS = (
    "csc-warning",
    "control-layer-note",
    "control-layer-alert",
    "control-layer-violation",
    "csc-details",
)


def _looks_like_status_scaffold_block(inner_html: str) -> bool:
    """Detect internal Profile/Overlay/SCI status scaffolding blocks conservatively."""
    raw = str(inner_html or "")
    if not raw:
        return False

    txt = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    txt = html.unescape(txt or "")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    low = txt.lower()
    if not low:
        return False

    # Fast path for dense one-line status composites.
    if low.startswith(("profile:", "active profile:", "comm:")) and len(_STATUS_KEY_RE.findall(low)) >= 3:
        return True

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    keys = []
    for ln in lines:
        m = re.match(
            r"(?i)^(active profile|profile|overlay|sci|control layer|qc|cgi|color|comm)\s*:\s*.+$",
            ln,
        )
        if m is None:
            return False
        keys.append(str(m.group(1) or "").strip().lower())

    uniq = set(keys)
    has_profile = ("profile" in uniq) or ("active profile" in uniq)
    has_meta = bool({"overlay", "sci", "control layer", "qc", "cgi", "color"} & uniq)
    return has_profile and has_meta


def _strip_control_layer_blocks_for_analysis(src: str) -> str:
    txt = str(src or "")
    if not txt:
        return txt
    out = _CONTROL_LAYER_CLASS_BLOCK_RE.sub(" ", txt)
    out = _CONTROL_LAYER_LEGACY_BLOCK_RE.sub(" ", out)
    return out


def _build_plain_text_index_map(html_text: str) -> tuple[str, list[int]]:
    plain_chars: list[str] = []
    html_after_positions: list[int] = []
    in_tag = False
    for i, ch in enumerate(str(html_text or "")):
        if ch == "<":
            in_tag = True
        if not in_tag:
            plain_chars.append(ch)
            html_after_positions.append(i + 1)
        if ch == ">":
            in_tag = False
    return "".join(plain_chars), html_after_positions


def _plain_to_html_span(pos_map: list[int], plain_start: int, plain_end: int) -> tuple[int, int]:
    if not pos_map:
        return -1, -1
    ps = int(plain_start)
    pe = int(plain_end)
    if ps < 0 or pe <= ps or ps >= len(pos_map):
        return -1, -1
    end_idx = pe - 1
    if end_idx >= len(pos_map):
        return -1, -1
    html_start = 0 if ps == 0 else int(pos_map[ps - 1])
    html_end = int(pos_map[end_idx])
    if html_end <= html_start:
        return -1, -1
    return html_start, html_end


def _replace_first_plain_code_with_marker(inner_html: str, code: str, *, lang: str = "de") -> tuple[str, bool]:
    src = str(inner_html or "")
    cc = str(code or "").strip().upper()
    if cc not in _U_CODES:
        return src, False

    plain, pos_map = _build_plain_text_index_map(src)
    if not plain or not pos_map:
        return src, False

    pat = re.compile(rf"\b{re.escape(cc)}\b")
    for m in pat.finditer(plain):
        ps, pe = int(m.start()), int(m.end())
        prev_ch = plain[ps - 1] if ps > 0 else ""
        next_ch = plain[pe] if pe < len(plain) else ""
        if prev_ch == "-" or next_ch == "-":
            # Keep range notations like U1-U8 untouched.
            continue

        # If the source already contains "(U1)", replace the whole parenthesized token
        # to avoid nested double parentheses after marker injection.
        rep_ps = ps
        rep_pe = pe
        if ps > 0 and pe < len(plain) and plain[ps - 1] == "(" and plain[pe] == ")":
            rep_ps = ps - 1
            rep_pe = pe + 1

        hs, he = _plain_to_html_span(pos_map, rep_ps, rep_pe)
        if hs < 0 or he <= hs:
            continue

        raw_token = src[hs:he]
        raw_plain = html.unescape(re.sub(r"(?is)<[^>]+>", "", str(raw_token or "")))
        raw_plain = re.sub(r"\s+", "", raw_plain).upper()
        expected_plain = re.sub(r"\s+", "", plain[rep_ps:rep_pe]).upper()
        if raw_plain != expected_plain:
            continue

        marker = build_uncertainty_inline_marker_html(cc, lang=lang)
        if not marker:
            return src, False
        return src[:hs] + marker + src[he:], True

    return src, False


def _advance_past_closing_tags(html_text: str, pos: int) -> int:
    src = str(html_text or "")
    cur = max(0, min(len(src), int(pos)))
    while cur < len(src):
        m = _INLINE_CLOSING_TAG_RUN_RE.match(src[cur:])
        if m is None:
            break
        cur += int(m.end())
    return cur


def _collect_sentence_slots(inner_html: str) -> list[dict]:
    plain, pos_map = _build_plain_text_index_map(inner_html)
    if not plain.strip() or not pos_map:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(plain):
        end = int(m.end())
        spans.append((start, end))
        start = end
    if start < len(plain):
        spans.append((start, len(plain)))

    out: list[dict] = []
    for s, e in spans:
        ss = int(s)
        ee = int(e)
        while ss < ee and plain[ss].isspace():
            ss += 1
        while ee > ss and plain[ee - 1].isspace():
            ee -= 1
        if ss >= ee:
            continue
        raw_pos = pos_map[ee - 1] if (ee - 1) < len(pos_map) else len(inner_html)
        raw_pos = _advance_past_closing_tags(inner_html, raw_pos)
        txt = plain[ss:ee]
        slot_codes: list[str] = []
        for c in (find_uncertainty_codes(txt) + infer_uncertainty_codes(txt, user_text="")):
            cc = str(c or "").strip().upper()
            if cc in _U_CODES and cc not in slot_codes:
                slot_codes.append(cc)
        out.append(
            {
                "text": txt,
                "codes": slot_codes,
                "insert_pos": raw_pos,
            }
        )
    return out


def _inject_markers_sentence_precise(inner_html: str, block_codes: list[str], *, lang: str = "de") -> str:
    if not block_codes:
        return str(inner_html or "")
    src = str(inner_html or "")

    uniq_codes: list[str] = []
    for c in block_codes:
        cc = str(c or "").strip().upper()
        if cc in _U_CODES and cc not in uniq_codes:
            uniq_codes.append(cc)

    if not uniq_codes:
        return src

    remaining_codes: list[str] = []
    for c in uniq_codes:
        src, replaced = _replace_first_plain_code_with_marker(src, c, lang=lang)
        if not replaced:
            remaining_codes.append(c)

    if not remaining_codes:
        return src

    slots = _collect_sentence_slots(src)
    if not slots:
        markers = " ".join(build_uncertainty_inline_marker_html(c, lang=lang) for c in remaining_codes if c in _U_CODES)
        return f"{src} {markers}".rstrip()

    assigned: dict[int, list[str]] = {}
    fallback_idx = len(slots) - 1
    for code in remaining_codes:
        tgt = fallback_idx
        for i, slot in enumerate(slots):
            if code in list(slot.get("codes") or []):
                tgt = i
                break
        assigned.setdefault(tgt, []).append(code)

    by_pos: dict[int, list[str]] = {}
    for idx, codes_here in assigned.items():
        pos = int(slots[idx].get("insert_pos") or len(src))
        uniq: list[str] = []
        for c in codes_here:
            if c in _U_CODES and c not in uniq:
                uniq.append(c)
        if not uniq:
            continue
        bucket = by_pos.setdefault(pos, [])
        for c in uniq:
            if c not in bucket:
                bucket.append(c)

    if not by_pos:
        return src

    out: list[str] = []
    cursor = 0
    for pos in sorted(by_pos.keys()):
        p = max(0, min(len(src), int(pos)))
        if p < cursor:
            continue
        out.append(src[cursor:p])
        prev = src[p - 1] if p > 0 else ""
        prefix = "" if (not prev or prev.isspace()) else " "
        markers = " ".join(build_uncertainty_inline_marker_html(c, lang=lang) for c in by_pos[pos] if c in _U_CODES)
        if markers:
            out.append(prefix + markers)
        cursor = p
    out.append(src[cursor:])
    return "".join(out)


def _ensure_legend_before_footer(src: str) -> str:
    txt = str(src or "")
    if not txt:
        return txt
    footer = _TS_FOOTER_RE.search(txt)
    if footer is None:
        return txt
    legends = list(_LEGEND_BLOCK_RE.finditer(txt))
    if not legends:
        return txt
    if legends[0].start() < footer.start():
        return txt

    legend_html = legends[0].group(0)
    cleaned = _LEGEND_BLOCK_RE.sub("", txt)
    footer2 = _TS_FOOTER_RE.search(cleaned)
    if footer2 is None:
        return txt
    before = cleaned[:footer2.start()]
    after = cleaned[footer2.start():]
    join_before = "" if before.endswith(("\n", "\r")) else "\n"
    join_after = "" if after.startswith(("\n", "\r")) else "\n"
    return before + join_before + legend_html + join_after + after


def _remove_legend_blocks(src: str) -> str:
    out = _LEGEND_BLOCK_RE.sub("", str(src or ""))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def find_uncertainty_codes(text: str) -> list[str]:
    seen = set()
    out = []
    for m in _U_CODE_RE.finditer(str(text or "")):
        code = str(m.group(1) or "").strip().upper()
        if code and code not in seen and code in _U_CODES:
            seen.add(code)
            out.append(code)
    return out


def find_marked_uncertainty_codes(text: str) -> list[str]:
    seen = set()
    out = []
    for m in _U_MARKED_RE.finditer(str(text or "")):
        code = str(m.group(1) or "").strip().upper()
        if code and code not in seen and code in _U_CODES:
            seen.add(code)
            out.append(code)
    return out


def infer_uncertainty_codes(text: str, *, user_text: str = "") -> list[str]:
    body = _normalize_text(text)
    user = _normalize_text(user_text)
    merged = (body + " " + user).strip()
    if not merged:
        return []

    out: list[str] = []
    seen = set()

    def _add(code: str):
        c = str(code or "").strip().upper()
        if c in _U_CODES and c not in seen:
            seen.add(c)
            out.append(c)

    for code in ("U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8"):
        needles = _INFER_RULES.get(code, [])
        if _contains_any(merged, needles):
            _add(code)

    # Deterministic high-complexity fallback for impossible "global optimum" prompt shapes.
    hits = sum(1 for t in _HIGH_COMPLEXITY_PROMPT_TERMS if t in user)
    if hits >= 2:
        _add("U3")
        _add("U5")
        _add("U6")

    # Last-resort fallback: uncertainty wording but no mapped code yet.
    if not out and _contains_any(merged, ["unsicher", "uncertain", "complex", "komplex", "offene frage"]):
        _add("U6")

    return out


def _code_meta(code: str, *, lang: str = "de") -> tuple[str, str]:
    meta = _U_CODES.get(str(code or "").upper(), {})
    use_en = str(lang or "").strip().lower() == "en"
    name = str(meta.get("en_name") if use_en else meta.get("de_name") or "").strip()
    desc = str(meta.get("en_desc") if use_en else meta.get("de_desc") or "").strip()
    return name, desc


def build_uncertainty_inline_marker_html(code: str, *, lang: str = "de") -> str:
    c = str(code or "").strip().upper()
    if c not in _U_CODES:
        return ""
    name, desc = _code_meta(c, lang=lang)
    title = f"{c} - {name}: {desc}".strip()
    return (
        "<span class='uncertainty-inline-wrap' style='color:#111827;'>("
        f"<span class='uncertainty-inline-marker' data-u-code='{html.escape(c)}' "
        f"data-u-title='{html.escape(title)}' title='{html.escape(title)}' "
        "style='color:#1d4ed8; text-decoration:underline; cursor:pointer; font-weight:600;'>"
        f"{html.escape(c)}</span>)</span>"
    )


def inject_inline_uncertainty_markers_html(
    text: str,
    *,
    codes: list[str] | None = None,
    lang: str = "de",
    user_text: str = "",
) -> str:
    src = str(text or "")
    if not src:
        return src
    if "data-u-code=" in src:
        return src

    inferred = [str(c or "").strip().upper() for c in (codes or infer_uncertainty_codes(src, user_text=user_text))]
    target_codes = []
    seen = set()
    existing_marked = set(find_marked_uncertainty_codes(src))
    for c in inferred:
        if c in _U_CODES and c not in seen and c not in existing_marked:
            seen.add(c)
            target_codes.append(c)
    if not target_codes:
        return src

    candidates = []
    for m in _BLOCK_TAG_RE.finditer(src):
        tag = str(m.group(1) or "")
        attrs = str(m.group(2) or "")
        inner = str(m.group(3) or "")
        attrs_low = attrs.lower()
        if any(h in attrs_low for h in _CONTROL_LAYER_ATTR_HINTS):
            continue
        if "uncertainty-legend" in attrs or "uncertainty-auto-marker" in attrs:
            continue
        if _looks_like_status_scaffold_block(inner):
            continue
        norm = _normalize_text(inner)
        if not norm or len(norm) < 40:
            continue
        if ("control layer note" in norm) or ("control layer alert" in norm) or ("control layer block" in norm):
            continue
        if any(t in norm for t in _INLINE_SKIP_TERMS):
            continue
        block_codes = []
        for c in (find_uncertainty_codes(inner) + infer_uncertainty_codes(inner, user_text="")):
            cc = str(c or "").strip().upper()
            if cc in _U_CODES and cc not in block_codes:
                block_codes.append(cc)
        candidates.append(
            {
                "start": m.start(),
                "end": m.end(),
                "tag": tag,
                "attrs": attrs,
                "inner": inner,
                "codes": block_codes,
            }
        )
    if not candidates:
        return src

    assigned: dict[int, list[str]] = {}
    for code in target_codes:
        placed = False
        for i, cand in enumerate(candidates):
            if code in list(cand.get("codes") or []):
                assigned.setdefault(i, []).append(code)
                placed = True
                break
        if not placed:
            assigned.setdefault(0, []).append(code)

    out = []
    cursor = 0
    for i, cand in enumerate(candidates):
        start = int(cand["start"])
        end = int(cand["end"])
        out.append(src[cursor:start])
        block_codes = []
        for c in assigned.get(i, []):
            if c not in block_codes:
                block_codes.append(c)
        if block_codes:
            inner_marked = _inject_markers_sentence_precise(str(cand["inner"] or ""), block_codes, lang=lang)
            repl = f"<{cand['tag']}{cand['attrs']}>{inner_marked}</{cand['tag']}>"
            out.append(repl)
        else:
            out.append(src[start:end])
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def build_uncertainty_legend_html(codes: list[str], *, lang: str = "de") -> str:
    use_en = str(lang or "").strip().lower() == "en"
    title = "Uncertainty code legend" if use_en else "U-Code-Legende"
    subtitle = (
        "Explanation of uncertainty markers used in this answer."
        if use_en
        else "Erklaerung der in dieser Antwort verwendeten Unsicherheitsmarker."
    )
    items = []
    for code in codes:
        if code not in _U_CODES:
            continue
        name, desc = _code_meta(code, lang=lang)
        items.append(
            "<li>"
            f"<code>{html.escape(code)}</code> - "
            f"<b>{html.escape(str(name or ''))}</b>: "
            f"{html.escape(str(desc or ''))}"
            "</li>"
        )
    if not items:
        return ""
    return (
        "<details class='uncertainty-legend' style='margin:10px 0; border:1px solid #d1d5db; "
        "background:#f8fafc; border-radius:10px; padding:8px;'>"
        f"<summary style='cursor:pointer; font-weight:600;'>{html.escape(title)}</summary>"
        f"<div style='margin-top:6px; color:#374151;'>{html.escape(subtitle)}</div>"
        "<ul style='margin:8px 0 0 18px; padding:0;'>"
        + "".join(items)
        + "</ul></details>"
    )


def build_uncertainty_auto_marker_html(codes: list[str], *, lang: str = "de") -> str:
    if not codes:
        return ""
    use_en = str(lang or "").strip().lower() == "en"
    title = "Uncertainty markers (auto)" if use_en else "Unsicherheitsmarker (auto)"
    subtitle = (
        "Inferred deterministically from prompt/answer uncertainty signals."
        if use_en
        else "Deterministisch aus Unsicherheitssignalen in Frage/Antwort abgeleitet."
    )
    code_html = " ".join(f"<code>{html.escape(c)}</code>" for c in codes if c in _U_CODES)
    return (
        "<div class='uncertainty-auto-marker' style='margin:8px 0; border:1px solid #fde68a; "
        "background:#fffbeb; color:#92400e; border-radius:10px; padding:8px;'>"
        f"<b>{html.escape(title)}:</b> {code_html}"
        f"<div style='margin-top:4px; font-size:12px;'>{html.escape(subtitle)}</div>"
        "</div>"
    )


def append_uncertainty_legend_html(text: str, *, lang: str = "de") -> str:
    # Product decision: legend output is disabled. Keep this function as
    # compatibility shim and strip any preexisting legend blocks.
    return _remove_legend_blocks(text)


def ensure_uncertainty_annotations_html(text: str, *, lang: str = "de", user_text: str = "") -> str:
    src = str(text or "")
    if not src:
        return src

    analysis_src = _strip_control_layer_blocks_for_analysis(src)
    codes = find_uncertainty_codes(analysis_src)
    if codes:
        src = inject_inline_uncertainty_markers_html(src, codes=codes, lang=lang, user_text=user_text)
        codes = find_uncertainty_codes(_strip_control_layer_blocks_for_analysis(src))
    else:
        inferred = infer_uncertainty_codes(analysis_src, user_text=user_text)
        src = inject_inline_uncertainty_markers_html(src, codes=inferred, lang=lang, user_text=user_text)
        codes = find_uncertainty_codes(_strip_control_layer_blocks_for_analysis(src))
        if (not codes) and inferred and ("uncertainty-auto-marker" not in src):
            src = src + "\n" + build_uncertainty_auto_marker_html(inferred, lang=lang)
            codes = list(inferred)

    if not codes:
        return _remove_legend_blocks(src)
    return _remove_legend_blocks(src)
