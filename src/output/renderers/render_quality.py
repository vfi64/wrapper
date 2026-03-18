from __future__ import annotations

import re
from typing import Any, Callable


_QC_FOOTER_RE = re.compile(r"(?im)^\s*QC(?:-Matrix)?\s*:")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SD_BOX_RE = re.compile(r'(?is)class=(?:"|\')[^"\']*\bself-debunking\b[^"\']*(?:"|\')')


def looks_like_rendered_html(html_text: str) -> bool:
    src = str(html_text or "")
    if not src:
        return False
    hl = src.lstrip().lower()
    if hl.startswith("<pre") and "&lt;" in hl:
        return False
    if src.count("&lt;") > 10 and ("<p" not in hl and "<div" not in hl and "<ol" not in hl):
        return False
    return any(t in hl for t in ("<p", "<div", "<ol", "<ul", "<table", "<pre", "<blockquote"))


def build_normalization_summary(
    *,
    raw_text: str,
    html_text: str,
    detect_self_debunking_numbered_html_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    raw = str(raw_text or "")
    html = str(html_text or "")
    raw_qc_count = len(_QC_FOOTER_RE.findall(raw))
    html_plain = _HTML_TAG_RE.sub("", html)
    html_qc_count = len(_QC_FOOTER_RE.findall(html_plain))
    sd_boxed = bool(_SD_BOX_RE.search(html))
    sd_numbered = False
    if callable(detect_self_debunking_numbered_html_fn):
        try:
            sd_numbered = bool(detect_self_debunking_numbered_html_fn(html))
        except Exception:
            sd_numbered = False
    return {
        "qc_footer_raw_count": raw_qc_count,
        "qc_footer_html_count": html_qc_count,
        "qc_footer_deduped": (raw_qc_count > 1 and html_qc_count == 1),
        "self_debunking_boxed": bool(sd_boxed),
        "self_debunking_numbered": bool(sd_numbered),
    }
