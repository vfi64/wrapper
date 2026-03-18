from __future__ import annotations

import html
import re


_U_CODE_RE = re.compile(r"\b(U[1-8])\b")
_U_MARKED_RE = re.compile(r"(?i)data-u-code\s*=\s*(?:\"|')?(U[1-8])(?:\"|')?")
_BLOCK_TAG_RE = re.compile(r"(?is)<(p|li)([^>]*)>(.*?)</\1>")
_LEAF_DIV_RE = re.compile(r"(?is)<div([^>]*)>(.*?)</div>")
_STATUS_KEY_RE = re.compile(
    r"(?i)\b(?:active profile|profile|aktives profil|profil|overlay|sci|control layer|steuerungsebene|qc|cgi|color|farbe|comm)\s*:"
)
_TS_FOOTER_RE = re.compile(
    r"(?is)<div\b[^>]*class=(?:\"|')[^\"']*\bts-footer\b[^\"']*(?:\"|')[^>]*>.*?</div>"
)
_UNCERTAINTY_MARKER_SPAN_RE = re.compile(
    r"(?is)<span\b[^>]*class=(?:\"|')[^\"']*\buncertainty-inline-marker\b[^\"']*(?:\"|')[^>]*>.*?</span>"
)
_BRACKETED_WRAP_RE = re.compile(
    r"(?is)\[\s*(<span\b[^>]*class=(?:\"|')[^\"']*\buncertainty-inline-wrap\b[^\"']*(?:\"|')[^>]*>\s*\(\s*"
    r"<span\b[^>]*class=(?:\"|')[^\"']*\buncertainty-inline-marker\b[^\"']*(?:\"|')[^>]*>\s*U[1-8]\s*</span>\s*\)\s*</span>)\s*\]"
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
_CODE_BLOCK_RE = re.compile(r"(?is)<code\b[^>]*>.*?</code>")
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

_SCI_TRACE_CLASS_RE = re.compile(
    r"(?is)\bclass\s*=\s*(?:\"|')[^\"']*\bsci-trace\b[^\"']*(?:\"|')"
)

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
    "profile:",
    "profil:",
    "aktives profil",
    "verification route",
)
_CONTROL_LAYER_ATTR_HINTS = (
    "csc-warning",
    "control-layer-note",
    "control-layer-alert",
    "control-layer-violation",
    "csc-details",
)


def _self_debunking_div_ranges(src: str) -> list[tuple[int, int]]:
    """Return [start,end) ranges for top-level self-debunking <div> blocks."""
    text = str(src or "")
    if not text:
        return []

    tag_re = re.compile(r"(?is)<div\b[^>]*>|</div\s*>")
    class_re = re.compile(
        r"(?is)\bclass\s*=\s*(?:\"|')[^\"']*\bself-debunking\b[^\"']*(?:\"|')"
    )

    out: list[tuple[int, int]] = []
    in_sd = False
    depth = 0
    start = -1

    for m in tag_re.finditer(text):
        tag = str(m.group(0) or "")
        low = tag.lower()
        is_open = low.startswith("<div")
        if is_open:
            if (not in_sd) and (class_re.search(tag) is not None):
                in_sd = True
                depth = 1
                start = int(m.start())
                continue
            if in_sd:
                depth += 1
            continue

        # closing </div>
        if in_sd:
            depth -= 1
            if depth <= 0:
                out.append((start, int(m.end())))
                in_sd = False
                depth = 0
                start = -1

    if in_sd and start >= 0:
        out.append((start, len(text)))
    return out


def _sci_trace_div_ranges(src: str) -> list[tuple[int, int]]:
    """Return [start,end) ranges for top-level sci-trace <div> blocks."""
    text = str(src or "")
    if not text:
        return []

    tag_re = re.compile(r"(?is)<div\b[^>]*>|</div\s*>")

    out: list[tuple[int, int]] = []
    in_sci = False
    depth = 0
    start = -1

    for m in tag_re.finditer(text):
        tag = str(m.group(0) or "")
        low = tag.lower()
        is_open = low.startswith("<div")
        if is_open:
            if (not in_sci) and (_SCI_TRACE_CLASS_RE.search(tag) is not None):
                in_sci = True
                depth = 1
                start = int(m.start())
                continue
            if in_sci:
                depth += 1
            continue

        if in_sci:
            depth -= 1
            if depth <= 0:
                out.append((start, int(m.end())))
                in_sci = False
                depth = 0
                start = -1

    if in_sci and start >= 0:
        out.append((start, len(text)))
    return out


def _uncertainty_auto_marker_div_ranges(src: str) -> list[tuple[int, int]]:
    """Return [start,end) ranges for top-level uncertainty-auto-marker <div> blocks."""
    text = str(src or "")
    if not text:
        return []

    tag_re = re.compile(r"(?is)<div\b[^>]*>|</div\s*>")
    class_re = re.compile(
        r"(?is)\bclass\s*=\s*(?:\"|')[^\"']*\buncertainty-auto-marker\b[^\"']*(?:\"|')"
    )

    out: list[tuple[int, int]] = []
    in_auto = False
    depth = 0
    start = -1

    for m in tag_re.finditer(text):
        tag = str(m.group(0) or "")
        low = tag.lower()
        is_open = low.startswith("<div")
        if is_open:
            if (not in_auto) and (class_re.search(tag) is not None):
                in_auto = True
                depth = 1
                start = int(m.start())
                continue
            if in_auto:
                depth += 1
            continue

        if in_auto:
            depth -= 1
            if depth <= 0:
                out.append((start, int(m.end())))
                in_auto = False
                depth = 0
                start = -1

    if in_auto and start >= 0:
        out.append((start, len(text)))
    return out


def _code_tag_ranges(src: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in _CODE_BLOCK_RE.finditer(str(src or "")):
        out.append((int(m.start()), int(m.end())))
    return out


def _is_inside_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    s = int(start)
    e = int(end)
    for a, b in ranges:
        if s >= int(a) and e <= int(b):
            return True
    return False


def _overlaps_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    s = int(start)
    e = int(end)
    for a, b in ranges:
        aa = int(a)
        bb = int(b)
        if s < bb and e > aa:
            return True
    return False


def _find_marked_uncertainty_codes_outside_ranges(
    src: str, ranges: list[tuple[int, int]]
) -> list[str]:
    seen = set()
    out = []
    for m in _U_MARKED_RE.finditer(str(src or "")):
        if _is_inside_ranges(m.start(), m.end(), ranges):
            continue
        code = str(m.group(1) or "").strip().upper()
        if code and code not in seen and code in _U_CODES:
            seen.add(code)
            out.append(code)
    return out


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
    if low.startswith(("profile:", "active profile:", "profil:", "aktives profil:", "comm:")) and len(
        _STATUS_KEY_RE.findall(low)
    ) >= 3:
        return True

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    keys = []
    for ln in lines:
        m = re.match(
            r"(?i)^(active profile|profile|aktives profil|profil|overlay|sci|control layer|steuerungsebene|qc|cgi|color|farbe|comm)\s*:\s*.+$",
            ln,
        )
        if m is None:
            return False
        keys.append(str(m.group(1) or "").strip().lower())

    uniq = set(keys)
    has_profile = ("profile" in uniq) or ("active profile" in uniq) or ("profil" in uniq) or ("aktives profil" in uniq)
    has_meta = bool({"overlay", "sci", "control layer", "steuerungsebene", "qc", "cgi", "color", "farbe"} & uniq)
    return has_profile and has_meta


def _strip_control_layer_blocks_for_analysis(src: str) -> str:
    txt = str(src or "")
    if not txt:
        return txt
    out = _CONTROL_LAYER_CLASS_BLOCK_RE.sub(" ", txt)
    out = _CONTROL_LAYER_LEGACY_BLOCK_RE.sub(" ", out)
    return out


def _strip_self_debunking_blocks_for_analysis(src: str) -> str:
    txt = str(src or "")
    if not txt:
        return txt
    ranges = _self_debunking_div_ranges(txt)
    if not ranges:
        return txt
    out = []
    cursor = 0
    for start, end in ranges:
        s = max(0, int(start))
        e = max(s, int(end))
        out.append(txt[cursor:s])
        out.append(" ")
        cursor = e
    out.append(txt[cursor:])
    return "".join(out)


def _strip_sci_trace_blocks_for_analysis(src: str) -> str:
    txt = str(src or "")
    if not txt:
        return txt
    ranges = _sci_trace_div_ranges(txt)
    if not ranges:
        return txt
    out = []
    cursor = 0
    for start, end in ranges:
        s = max(0, int(start))
        e = max(s, int(end))
        out.append(txt[cursor:s])
        out.append(" ")
        cursor = e
    out.append(txt[cursor:])
    return "".join(out)


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
        hs = _advance_past_closing_tags(src, hs)
        if hs >= he:
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


def _replace_all_plain_u_codes_with_markers(inner_html: str, *, lang: str = "de") -> str:
    src = str(inner_html or "")
    if not src:
        return src

    out = _BRACKETED_WRAP_RE.sub(r"\1", src)

    max_iters = 256
    for _ in range(max_iters):
        plain, pos_map = _build_plain_text_index_map(out)
        if not plain or not pos_map:
            break
        marker_ranges = [(m.start(), m.end()) for m in _UNCERTAINTY_MARKER_SPAN_RE.finditer(out)]
        replaced = False
        for m in _U_CODE_RE.finditer(plain):
            code = str(m.group(1) or "").strip().upper()
            if code not in _U_CODES:
                continue
            ps, pe = int(m.start()), int(m.end())
            hs_code, he_code = _plain_to_html_span(pos_map, ps, pe)
            if hs_code < 0 or he_code <= hs_code:
                continue
            if _is_inside_ranges(hs_code, he_code, marker_ranges):
                continue
            prev_ch = plain[ps - 1] if ps > 0 else ""
            next_ch = plain[pe] if pe < len(plain) else ""
            if prev_ch == "-" or next_ch == "-":
                # Keep range notations like U1-U8 untouched.
                continue

            rep_ps, rep_pe = ps, pe
            if ps > 0 and pe < len(plain):
                if plain[ps - 1] == "(" and plain[pe] == ")":
                    rep_ps, rep_pe = ps - 1, pe + 1
                elif plain[ps - 1] == "[" and plain[pe] == "]":
                    rep_ps, rep_pe = ps - 1, pe + 1

            hs, he = _plain_to_html_span(pos_map, rep_ps, rep_pe)
            if hs < 0 or he <= hs:
                continue
            hs = _advance_past_closing_tags(out, hs)
            if hs >= he:
                continue
            if _overlaps_ranges(hs, he, marker_ranges):
                continue

            raw_token = out[hs:he]
            raw_plain = html.unescape(re.sub(r"(?is)<[^>]+>", "", str(raw_token or "")))
            raw_norm = re.sub(r"\s+", "", raw_plain).upper()
            exp_norm = re.sub(r"\s+", "", plain[rep_ps:rep_pe]).upper()
            if raw_norm != exp_norm:
                continue

            marker = build_uncertainty_inline_marker_html(code, lang=lang)
            if not marker:
                continue
            out = out[:hs] + marker + out[he:]
            replaced = True
            break
        if not replaced:
            break
    return out


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


def _strip_uncertainty_template_phrases_html(src: str) -> str:
    """Collapse leaked template phrases like 'Uncertainty: (U5) - Model limitation. Needed: ...'
    to the canonical inline marker only.
    """
    out = str(src or "")
    if not out:
        return out
    def _name_variants(term: str) -> list[str]:
        t = str(term or "").strip()
        if not t:
            return []
        out_v = {t}
        # Accept both ASCII transliteration and umlaut spellings in weak-model leaks.
        out_v.add(t.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü").replace("ss", "ß"))
        out_v.add(t.replace("Ae", "Ä").replace("Oe", "Ö").replace("Ue", "Ü"))
        out_v.add(t.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
        return [v for v in out_v if v]
    u_name_terms = []
    for _meta in _U_CODES.values():
        for _k in ("en_name", "de_name"):
            _v = str((_meta or {}).get(_k) or "").strip()
            for _vv in _name_variants(_v):
                if _vv and _vv not in u_name_terms:
                    u_name_terms.append(_vv)
    # Legacy phrase variants observed in provider outputs.
    for _legacy in ("Model limitation",):
        if _legacy not in u_name_terms:
            u_name_terms.append(_legacy)
    u_name_rx = "|".join(re.escape(x) for x in u_name_terms) if u_name_terms else r"(?:Data\s+gap|Model\s+limitation)"
    marker_span_rx = (
        r"<span\b[^>]*class=(?:\"|')[^\"']*\buncertainty-inline-wrap\b[^\"']*(?:\"|')[^>]*>"
        r"[\s\S]*?</span>"
    )
    marker_rx = (
        rf"(?:{marker_span_rx}"
        r"|\(\s*U[1-8]\s*\)"
        rf"|\(\s*{marker_span_rx}\s*\))"
    )
    needed_rx = r"(?:Needed|Ben(?:ö|oe)tigt)\s*:"
    # Inline leak body seen in weak-model outputs:
    # "<marker> – Datenlücke. Benötigt: ... ." also inside running text (not only block tail).
    needed_tail_rx = rf"{needed_rx}\s*[^<\n]{{0,220}}?(?:[.!?])"
    out = re.sub(
        rf"(?is)(?:Uncertainty|Unsicherheit)\s*:\s*(?P<marker>{marker_rx})\s*"
        rf"(?:&ndash;|&mdash;|–|—|-)\s*(?:{u_name_rx})\s*\.\s*{needed_tail_rx}",
        r"\g<marker>",
        out,
    )
    out = re.sub(
        rf"(?is)(?:Uncertainty|Unsicherheit)\s*:\s*(?P<marker>{marker_rx})",
        r"\g<marker>",
        out,
    )
    # Also strip leaks without explicit 'Uncertainty:' prefix:
    # "(U1) - Data gap. Needed: ..." / "<marker> – Strukturelle Grenze. Benoetigt: ..."
    out = re.sub(
        rf"(?is)(?P<marker>{marker_rx})\s*(?:&ndash;|&mdash;|–|—|-)\s*(?:{u_name_rx})\s*\.\s*{needed_tail_rx}",
        r"\g<marker>",
        out,
    )
    # Repair dangling clause fragments immediately before uncertainty markers,
    # e.g. "... wurde, (U1)" after model-side truncation.
    signal_dot_marker_span_rx = (
        r"<span\b[^>]*class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')[^>]*>"
        r"[\s\S]*?</span>"
    )
    out = re.sub(
        rf"(?is)\b(?:wurde|wurden|wird|werden|war|waren|ist|sind)\b\s*,\s*"
        rf"(?P<dots>(?:{signal_dot_marker_span_rx}\s*)*)(?P<marker>{marker_rx})",
        r"ist als unsicher einzuordnen. \g<dots>\g<marker>",
        out,
    )
    out = re.sub(
        rf"(?is)\b(?:was|were|is|are|be|been)\b\s*,\s*"
        rf"(?P<dots>(?:{signal_dot_marker_span_rx}\s*)*)(?P<marker>{marker_rx})",
        r"is uncertain. \g<dots>\g<marker>",
        out,
    )
    out = re.sub(r"(?is)\s{2,}", " ", out)
    return out


def _collapse_orphan_uncertainty_marker_paragraph_before_self_debunking(src: str) -> str:
    """Avoid marker-only U paragraphs directly in front of Self-Debunking blocks.

    Weak-model outputs occasionally emit a standalone ``<p>(U5)</p>`` right before
    the self-debunking box. Keep the marker but merge/unwrap that paragraph so the
    transition into Self-Debunking remains structurally stable and readable.
    """
    out = str(src or "")
    if not out:
        return out

    wrapped_marker_rx = (
        r"<span\b[^>]*class=(?:\"|')[^\"']*\buncertainty-inline-wrap\b[^\"']*(?:\"|')[^>]*>"
        r"[\s\S]*?</span>"
    )
    signal_dot_marker_span_rx = (
        r"<span\b[^>]*class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')[^>]*>"
        r"[\s\S]*?</span>"
    )
    plain_marker_rx = r"\(\s*U[1-8]\s*\)"
    marker_seq_rx = (
        rf"(?:(?:{signal_dot_marker_span_rx}\s*)*(?:{wrapped_marker_rx}|{plain_marker_rx})\s*)+"
    )
    sd_open_rx = r"<div\b[^>]*class=(?:\"|')[^\"']*\bself-debunking\b[^\"']*(?:\"|')[^>]*>"

    # Preferred repair: attach marker sequence to the end of the preceding paragraph.
    out = re.sub(
        rf"(?is)</p>\s*<p\b[^>]*>\s*(?P<markers>{marker_seq_rx})\s*</p>\s*(?={sd_open_rx})",
        lambda m: " " + str(m.group("markers") or "").strip() + "</p>\n",
        out,
    )
    # Fallback: if no preceding paragraph exists, unwrap the marker paragraph.
    out = re.sub(
        rf"(?is)<p\b[^>]*>\s*(?P<markers>{marker_seq_rx})\s*</p>\s*(?={sd_open_rx})",
        lambda m: str(m.group("markers") or "").strip() + "\n",
        out,
    )
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _control_layer_ranges(src: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    txt = str(src or "")
    if not txt:
        return out
    for m in _CONTROL_LAYER_CLASS_BLOCK_RE.finditer(txt):
        out.append((int(m.start()), int(m.end())))
    for m in _CONTROL_LAYER_LEGACY_BLOCK_RE.finditer(txt):
        out.append((int(m.start()), int(m.end())))
    return out


def _replace_all_plain_u_codes_global(src: str, *, lang: str = "de") -> str:
    out = str(src or "")
    if not out:
        return out

    max_iters = 512
    for _ in range(max_iters):
        plain, pos_map = _build_plain_text_index_map(out)
        if not plain or not pos_map:
            break

        sd_ranges = _self_debunking_div_ranges(out)
        ctl_ranges = _control_layer_ranges(out)
        auto_ranges = _uncertainty_auto_marker_div_ranges(out)
        code_ranges = _code_tag_ranges(out)
        marker_ranges = [(m.start(), m.end()) for m in _UNCERTAINTY_MARKER_SPAN_RE.finditer(out)]
        skip_ranges = list(sd_ranges) + list(ctl_ranges) + list(auto_ranges) + list(code_ranges) + list(marker_ranges)

        replaced = False
        for m in _U_CODE_RE.finditer(plain):
            code = str(m.group(1) or "").strip().upper()
            if code not in _U_CODES:
                continue
            ps, pe = int(m.start()), int(m.end())
            hs_code, he_code = _plain_to_html_span(pos_map, ps, pe)
            if hs_code < 0 or he_code <= hs_code:
                continue
            if _is_inside_ranges(hs_code, he_code, skip_ranges):
                continue

            prev_ch = plain[ps - 1] if ps > 0 else ""
            next_ch = plain[pe] if pe < len(plain) else ""
            if prev_ch == "-" or next_ch == "-":
                continue

            rep_ps, rep_pe = ps, pe
            if ps > 0 and pe < len(plain):
                if plain[ps - 1] == "(" and plain[pe] == ")":
                    rep_ps, rep_pe = ps - 1, pe + 1
                elif plain[ps - 1] == "[" and plain[pe] == "]":
                    rep_ps, rep_pe = ps - 1, pe + 1

            hs, he = _plain_to_html_span(pos_map, rep_ps, rep_pe)
            if hs < 0 or he <= hs:
                continue
            hs = _advance_past_closing_tags(out, hs)
            if hs >= he:
                continue
            if _overlaps_ranges(hs, he, skip_ranges):
                continue

            raw_token = out[hs:he]
            raw_plain = html.unescape(re.sub(r"(?is)<[^>]+>", "", str(raw_token or "")))
            raw_norm = re.sub(r"\s+", "", raw_plain).upper()
            exp_norm = re.sub(r"\s+", "", plain[rep_ps:rep_pe]).upper()
            if raw_norm != exp_norm:
                continue

            marker = build_uncertainty_inline_marker_html(code, lang=lang)
            if not marker:
                continue
            out = out[:hs] + marker + out[he:]
            replaced = True
            break

        if not replaced:
            break
    return out


def canonicalize_explicit_uncertainty_codes_html(text: str, *, lang: str = "de") -> str:
    src = str(text or "")
    if not src:
        return src

    def _should_skip_block(attrs: str, inner: str) -> bool:
        attrs_low = str(attrs or "").lower()
        if any(h in attrs_low for h in _CONTROL_LAYER_ATTR_HINTS):
            return True
        if "uncertainty-legend" in attrs_low or "uncertainty-auto-marker" in attrs_low:
            return True
        if _looks_like_status_scaffold_block(inner):
            return True
        norm = _normalize_text(inner)
        if not norm:
            return True
        if ("control layer note" in norm) or ("control layer alert" in norm) or ("control layer block" in norm):
            return True
        return False

    out = src
    sd_ranges = _self_debunking_div_ranges(out)

    def _repl_p_li(m: re.Match) -> str:
        start, end = int(m.start()), int(m.end())
        if _is_inside_ranges(start, end, sd_ranges):
            return m.group(0)
        tag = str(m.group(1) or "")
        attrs = str(m.group(2) or "")
        inner = str(m.group(3) or "")
        if _should_skip_block(attrs, inner):
            return m.group(0)
        inner2 = _replace_all_plain_u_codes_with_markers(inner, lang=lang)
        if inner2 == inner:
            return m.group(0)
        return f"<{tag}{attrs}>{inner2}</{tag}>"

    out = _BLOCK_TAG_RE.sub(_repl_p_li, out)

    sd_ranges = _self_debunking_div_ranges(out)

    def _repl_leaf_div(m: re.Match) -> str:
        start, end = int(m.start()), int(m.end())
        if _is_inside_ranges(start, end, sd_ranges):
            return m.group(0)
        attrs = str(m.group(1) or "")
        inner = str(m.group(2) or "")
        if re.search(r"(?is)<\s*(?:div|p|li|ol|ul|table|blockquote)\b", inner):
            return m.group(0)
        if _should_skip_block(attrs, inner):
            return m.group(0)
        inner2 = _replace_all_plain_u_codes_with_markers(inner, lang=lang)
        if inner2 == inner:
            return m.group(0)
        return f"<div{attrs}>{inner2}</div>"

    out = _LEAF_DIV_RE.sub(_repl_leaf_div, out)
    # Global fallback pass for nested structures (e.g., SCI Trace with nested <div> blocks).
    out = _replace_all_plain_u_codes_global(out, lang=lang)
    return out


def ensure_uncertainty_marker_tooltips_html(text: str, *, lang: str = "de") -> str:
    src = str(text or "")
    if not src:
        return src

    marker_re = re.compile(
        r"(?is)<span(?P<attrs1>[^>]*)class=(?:\"|')(?P<class>[^\"']*\buncertainty-inline-marker\b[^\"']*)(?:\"|')"
        r"(?P<attrs2>[^>]*)>(?P<body>.*?)</span>"
    )

    def _repl(m: re.Match) -> str:
        attrs1 = str(m.group("attrs1") or "")
        klass = str(m.group("class") or "")
        attrs2 = str(m.group("attrs2") or "")
        body = str(m.group("body") or "")
        attrs = attrs1 + attrs2

        has_data_title = re.search(r"(?i)\bdata-u-title\s*=", attrs) is not None
        has_title = re.search(r"(?i)\btitle\s*=", attrs) is not None
        if has_data_title and has_title:
            return m.group(0)

        m_code = re.search(r"(?i)\bdata-u-code\s*=\s*(?:\"|')?(U[1-8])(?:\"|')?", attrs)
        code = str(m_code.group(1) or "").strip().upper() if m_code else ""
        if code not in _U_CODES:
            body_plain = html.unescape(re.sub(r"(?is)<[^>]+>", "", body)).strip().upper()
            if re.fullmatch(r"U[1-8]", body_plain or ""):
                code = body_plain
        if code not in _U_CODES:
            return m.group(0)

        name, desc = _code_meta(code, lang=lang)
        tip = html.escape(f"{code} - {name}: {desc}", quote=True)
        open_tag = f"<span{attrs1}class='{klass}'{attrs2}"
        if not has_data_title:
            open_tag += f" data-u-title='{tip}'"
        if not has_title:
            open_tag += f" title='{tip}'"
        return f"{open_tag}>{body}</span>"

    return marker_re.sub(_repl, src)


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
    # RED/critical signal fallback: if a red marker leaks into content without an explicit U-code,
    # infer U1 so the verification-route contract can be satisfied deterministically.
    if not out and _contains_any(merged, ["🔴", "[red]", " red claim", "kritische aussage", "critical claim"]):
        _add("U1")

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

    inferred = [str(c or "").strip().upper() for c in (codes or infer_uncertainty_codes(src, user_text=user_text))]
    target_codes = []
    seen = set()
    sd_ranges = _self_debunking_div_ranges(src)
    sci_ranges = _sci_trace_div_ranges(src)
    skip_ranges = sorted([*sd_ranges, *sci_ranges], key=lambda x: int(x[0]))
    existing_marked = set(_find_marked_uncertainty_codes_outside_ranges(src, skip_ranges))
    for c in inferred:
        if c in _U_CODES and c not in seen and c not in existing_marked:
            seen.add(c)
            target_codes.append(c)
    if not target_codes:
        return src

    candidates = []
    for m in _BLOCK_TAG_RE.finditer(src):
        if _is_inside_ranges(m.start(), m.end(), skip_ranges):
            # Do not inject uncertainty markers into Self-Debunking/SCI Trace internals.
            continue
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
    analysis_src = _strip_self_debunking_blocks_for_analysis(analysis_src)
    analysis_src = _strip_sci_trace_blocks_for_analysis(analysis_src)
    codes = find_uncertainty_codes(analysis_src)
    if codes:
        src = inject_inline_uncertainty_markers_html(src, codes=codes, lang=lang, user_text=user_text)
        recalc_src = _strip_self_debunking_blocks_for_analysis(_strip_control_layer_blocks_for_analysis(src))
        codes = find_uncertainty_codes(recalc_src)
    else:
        inferred = infer_uncertainty_codes(analysis_src, user_text=user_text)
        src = inject_inline_uncertainty_markers_html(src, codes=inferred, lang=lang, user_text=user_text)
        recalc_src = _strip_self_debunking_blocks_for_analysis(_strip_control_layer_blocks_for_analysis(src))
        codes = find_uncertainty_codes(recalc_src)
        if (not codes) and inferred and ("uncertainty-auto-marker" not in src):
            src = src + "\n" + build_uncertainty_auto_marker_html(inferred, lang=lang)
            codes = list(inferred)

    src = canonicalize_explicit_uncertainty_codes_html(src, lang=lang)
    src = ensure_uncertainty_marker_tooltips_html(src, lang=lang)
    src = _strip_uncertainty_template_phrases_html(src)
    src = _collapse_orphan_uncertainty_marker_paragraph_before_self_debunking(src)

    if not codes:
        return _remove_legend_blocks(src)
    return _remove_legend_blocks(src)
