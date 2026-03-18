from __future__ import annotations

import html
import re
from typing import Callable, Mapping, Sequence


_DEFAULT_EVIDENCE_COLOR = {
    "GREEN": "#137333",
    "YELLOW": "#f9ab00",
    "RED": "#d93025",
    "GRAY": "#5f6368",
}

_DEFAULT_EVIDENCE_ICON = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
    "GRAY": "⚪",
}

_SIGNAL_DOT_SPAN_RE = re.compile(r"(?is)<span(?P<attrs>[^>]*)>(?P<body>\s*[🟢🟡🔴]\s*)</span>")
_SIGNAL_DOT_BLOCK_RE = re.compile(r"(?is)<(?P<tag>p|li)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>")
_SIGNAL_DOT_MARKER_SPAN_RE = re.compile(
    r"(?is)<span(?P<attrs1>[^>]*)class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')(?P<attrs2>[^>]*)>\s*"
    r"(?P<body>(?:<span\b[^>]*>\s*[🟢🟡🔴]\s*</span>|[🟢🟡🔴]))\s*</span>"
)
_SIGNAL_DOT_MARKER_RE = re.compile(
    r"(?is)<span\b[^>]*class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')[^>]*>\s*"
    r"(?:<span\b[^>]*>\s*[🟢🟡🔴]\s*</span>|[🟢🟡🔴])\s*</span>\s*"
)
_SIGNAL_DOT_STATUS_PREFIX_RE = re.compile(r"(?i)^(?:active profile|profile|overlay|sci|control layer|qc|cgi|color|comm)\s*:")
_SIGNAL_DOT_COLOR = {
    "🟢": "#2e7d32",
    "🟡": "#f9a825",
    "🔴": "#c62828",
}


def _tooltip_lang(lang: str = "de") -> str:
    l = str(lang or "").strip().lower()
    return "en" if l.startswith("en") else "de"


def signal_dot_tooltip_text(icon: str, *, lang: str = "de") -> str:
    use_en = (_tooltip_lang(lang) == "en")
    ic = str(icon or "").strip()
    if ic == "🟢":
        return (
            "Green: high reliability and comparatively robust evidence."
            if use_en
            else "Gruen: hohe Verlaesslichkeit und vergleichsweise robuste Evidenz."
        )
    if ic == "🟡":
        return (
            "Yellow: medium reliability; relevant uncertainty remains."
            if use_en
            else "Gelb: mittlere Verlaesslichkeit; relevante Unsicherheit bleibt."
        )
    if ic == "🔴":
        return (
            "Red: low reliability; substantial uncertainty or weak support."
            if use_en
            else "Rot: niedrige Verlaesslichkeit; erhebliche Unsicherheit oder schwache Absicherung."
        )
    return ""


def _resolve_signal_dot_tip(
    icon: str,
    *,
    lang: str = "de",
    signal_dot_tooltip_text_fn: Callable | None = None,
) -> str:
    if callable(signal_dot_tooltip_text_fn):
        try:
            return str(signal_dot_tooltip_text_fn(icon=icon, lang=lang) or "")
        except TypeError:
            try:
                return str(signal_dot_tooltip_text_fn(icon, lang=lang) or "")
            except Exception:
                pass
        except Exception:
            pass
    return signal_dot_tooltip_text(icon, lang=lang)


def annotate_signal_dot_tooltips_html(
    html_text: str,
    *,
    lang: str = "de",
    signal_dot_tooltip_text_fn: Callable | None = None,
) -> str:
    """Wrap color signal dots with hold-tooltip metadata in answer language."""
    src = str(html_text or "")
    if not src:
        return src

    protected: list[str] = []

    def _protect_existing_marker(m: re.Match) -> str:
        attrs = f"{m.group('attrs1') or ''}{m.group('attrs2') or ''}"
        body = str(m.group("body") or "")
        block = m.group(0)
        if re.search(r"(?i)\b(?:data-u-title|title)\s*=", attrs) is None:
            im = re.search(r"[🟢🟡🔴]", body)
            icon = str(im.group(0) if im else "")
            tip = _resolve_signal_dot_tip(
                icon,
                lang=lang,
                signal_dot_tooltip_text_fn=signal_dot_tooltip_text_fn,
            )
            if tip:
                esc = html.escape(tip)
                if re.search(r"(?i)\bstyle\s*=", block):
                    block = re.sub(
                        r"(?is)<span\b",
                        f"<span data-u-title='{esc}' title='{esc}'",
                        block,
                        count=1,
                    )
                else:
                    block = re.sub(
                        r"(?is)<span\b",
                        f"<span data-u-title='{esc}' title='{esc}' style='cursor:help;'",
                        block,
                        count=1,
                    )
        token = f"__SIGNAL_DOT_MARKER_PROTECT_{len(protected)}__"
        protected.append(block)
        return token

    stage = _SIGNAL_DOT_MARKER_SPAN_RE.sub(_protect_existing_marker, src)

    def _repl(m: re.Match) -> str:
        body = str(m.group("body") or "")
        icon = re.sub(r"\s+", "", body)
        tip = _resolve_signal_dot_tip(
            icon,
            lang=lang,
            signal_dot_tooltip_text_fn=signal_dot_tooltip_text_fn,
        )
        if not tip:
            return m.group(0)
        esc = html.escape(tip)
        return (
            "<span class='signal-dot-marker' "
            f"data-u-title='{esc}' title='{esc}' style='cursor:help;'>"
            f"{m.group(0)}"
            "</span>"
        )

    stage = _SIGNAL_DOT_SPAN_RE.sub(_repl, stage)
    for idx, block in enumerate(protected):
        stage = stage.replace(f"__SIGNAL_DOT_MARKER_PROTECT_{idx}__", block)
    return stage


def fallback_signal_dot_icon_for_text(
    text: str,
    *,
    infer_uncertainty_codes_fn: Callable[[str], Sequence[str]] | None = None,
) -> str:
    """Map block uncertainty signal to a deterministic fallback dot icon."""
    norm = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(text or "")))
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return "🟢"

    codes: list[str] = []
    if callable(infer_uncertainty_codes_fn):
        try:
            raw = infer_uncertainty_codes_fn(norm) or []
            codes = [str(c or "") for c in list(raw)]
        except Exception:
            codes = []

    code_set = {str(c or "").strip().upper() for c in codes}
    if "U1" in code_set or "U4" in code_set:
        return "🔴"
    if code_set:
        return "🟡"
    return "🟢"


def inject_fallback_signal_dots_html(
    html_text: str,
    *,
    lang: str = "de",
    infer_uncertainty_codes_fn: Callable[[str], Sequence[str]] | None = None,
    signal_dot_tooltip_text_fn: Callable | None = None,
) -> str:
    """Insert deterministic signal dots when Color=on but model emitted no evidence dots."""
    src = str(html_text or "")
    if not src:
        return src
    if "signal-dot-marker" in src:
        return src

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        tag = str(m.group("tag") or "")
        attrs = str(m.group("attrs") or "")
        body = str(m.group("body") or "")
        out.append(src[cursor:start])

        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", body))
        plain = re.sub(r"\s+", " ", plain).strip()
        low = plain.lower()

        skip = False
        if not plain or len(plain) < 24:
            skip = True
        elif _SIGNAL_DOT_STATUS_PREFIX_RE.match(plain):
            skip = True
        elif low.startswith("qc-matrix:") or low.startswith("verification route"):
            skip = True
        elif "response at " in low or "selbst-debunking" in low or "self-debunking" in low:
            skip = True
        elif "uncertainty-auto-marker" in attrs:
            skip = True

        if skip:
            out.append(src[start:end])
        else:
            icon = fallback_signal_dot_icon_for_text(plain, infer_uncertainty_codes_fn=infer_uncertainty_codes_fn)
            tip = _resolve_signal_dot_tip(
                icon,
                lang=lang,
                signal_dot_tooltip_text_fn=signal_dot_tooltip_text_fn,
            )
            esc_tip = html.escape(tip) if tip else ""
            color = _SIGNAL_DOT_COLOR.get(icon, "#5f6368")
            dot_html = (
                "<span class='signal-dot-marker' "
                f"data-u-title='{esc_tip}' title='{esc_tip}' style='cursor:help;'>"
                f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"
                "</span> "
            )
            marked_body = re.sub(r"^(\s*)", r"\1" + dot_html, body, count=1)
            out.append(f"<{tag}{attrs}>{marked_body}</{tag}>")
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def limit_signal_dot_marker_density_html(html_text: str, *, max_per_block: int = 1) -> str:
    """Keep at most N signal-dot markers per content block to avoid visual marker overload."""
    src = str(html_text or "")
    cap = max(1, int(max_per_block or 1))
    if not src or "signal-dot-marker" not in src:
        return src

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        tag = str(m.group("tag") or "")
        attrs = str(m.group("attrs") or "")
        body = str(m.group("body") or "")
        out.append(src[cursor:start])

        if "signal-dot-marker" not in body:
            out.append(src[start:end])
            cursor = end
            continue

        kept = 0

        def _trim(mm: re.Match) -> str:
            nonlocal kept
            kept += 1
            if kept <= cap:
                return mm.group(0)
            return ""

        cleaned = _SIGNAL_DOT_MARKER_RE.sub(_trim, body)
        out.append(f"<{tag}{attrs}>{cleaned}</{tag}>")
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def strip_signal_dots_from_heading_only_blocks_html(html_text: str) -> str:
    """Remove signal-dot markers from pure heading blocks (<p>/<li> with only <strong>/<hN>)."""
    src = str(html_text or "")
    if not src or "signal-dot-marker" not in src:
        return src

    def _is_heading_only(inner_html: str) -> bool:
        content = re.sub(r"(?is)^\s*(?:<br\s*/?>|\s|&nbsp;)+", "", str(inner_html or ""))
        content = re.sub(r"(?is)(?:<br\s*/?>|\s|&nbsp;)+\s*$", "", content)
        if not content:
            return False
        if re.fullmatch(r"(?is)<strong\b[^>]*>.*?</strong>", content):
            return True
        if re.fullmatch(r"(?is)<h[1-6]\b[^>]*>.*?</h[1-6]>", content):
            return True
        return False

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        block = str(m.group(0) or "")
        out.append(src[cursor:start])
        if "signal-dot-marker" not in block:
            out.append(block)
        else:
            block_wo_markers = _SIGNAL_DOT_MARKER_RE.sub("", block)
            mm = re.match(r"(?is)<(?P<tag>p|li)\b[^>]*>(?P<body>.*?)</(?P=tag)>", block_wo_markers)
            body = str(mm.group("body") if mm else "")
            if _is_heading_only(body):
                out.append(block_wo_markers)
            else:
                out.append(block)
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def apply_color_spans(
    text: str,
    *,
    enabled: bool = True,
    evidence_color: Mapping[str, str] | None = None,
    evidence_icon: Mapping[str, str] | None = None,
) -> str:
    """Render Evidence-Linker tags with actual HTML colors (does not invent tags)."""
    if not enabled or not text:
        return text

    color_map = dict(_DEFAULT_EVIDENCE_COLOR)
    icon_map = dict(_DEFAULT_EVIDENCE_ICON)
    if isinstance(evidence_color, Mapping):
        color_map.update({str(k).upper(): str(v) for k, v in evidence_color.items()})
    if isinstance(evidence_icon, Mapping):
        icon_map.update({str(k).upper(): str(v) for k, v in evidence_icon.items()})

    def repl(m: re.Match) -> str:
        tag = (m.group("tag") or "").upper()
        emoji = m.group("emoji") or ""
        color = color_map.get(tag, "#616161")
        icon = emoji or icon_map.get(tag, "⚪")
        return f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"

    pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?")
    return pat.sub(repl, text)


def reapply_color_styles_if_stripped(
    html_text: str,
    *,
    evidence_color: Mapping[str, str] | None = None,
    evidence_icon: Mapping[str, str] | None = None,
) -> str:
    """Re-apply safe color styles on stripped evidence spans (`style=""`)."""
    if not html_text:
        return html_text

    color_map = dict(_DEFAULT_EVIDENCE_COLOR)
    icon_map = dict(_DEFAULT_EVIDENCE_ICON)
    if isinstance(evidence_color, Mapping):
        color_map.update({str(k).upper(): str(v) for k, v in evidence_color.items()})
    if isinstance(evidence_icon, Mapping):
        icon_map.update({str(k).upper(): str(v) for k, v in evidence_icon.items()})

    def repl(m: re.Match) -> str:
        tag = (m.group("tag") or "").upper()
        emoji = m.group("emoji") or ""
        color = color_map.get(tag, "#616161")
        icon = emoji or icon_map.get(tag, "⚪")
        return f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"

    pat = re.compile(
        r"<span\s+style=\"\"\s*>\s*\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?\s*</span>",
        flags=re.IGNORECASE,
    )
    return pat.sub(repl, html_text)


def strip_color_markers_for_color_off_text(text: str, *, strip_policy_fn: Callable[[str], str] | None = None) -> str:
    """Hide color marker artifacts when Color=off (tags + glyph markers)."""
    if callable(strip_policy_fn):
        try:
            out = strip_policy_fn(text)
            return text if out is None else out
        except Exception:
            pass
    src = str(text or "")
    if not src:
        return src
    out = src
    out = re.sub(r"(?i)\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]\s*[🟢🟡🔴⚪⚪️]*", "", out)
    out = re.sub(r"[🟢🟡🔴⚪⚪️]", "", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


def strip_color_markers_for_color_off_html(
    html_text: str,
    *,
    strip_policy_fn: Callable[[str], str] | None = None,
) -> str:
    """Hide color marker artifacts in rendered HTML when Color=off."""
    if callable(strip_policy_fn):
        try:
            out = strip_policy_fn(html_text)
            return html_text if out is None else out
        except Exception:
            pass
    out = str(html_text or "")
    if not out:
        return out
    out = re.sub(
        r"(?is)<span\b[^>]*class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')[^>]*>.*?</span>",
        "",
        out,
    )
    out = re.sub(r"(?i)\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]", "", out)
    out = re.sub(r"[🟢🟡🔴⚪⚪️]", "", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out
