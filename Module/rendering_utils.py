import re
from typing import Optional

try:
    import bleach  # type: ignore
except Exception:
    bleach = None  # type: ignore

_css_sanitizer_cached = None

_EVIDENCE_COLOR = {
    "GREEN": "#2e7d32",
    "YELLOW": "#f9a825",
    "RED": "#c62828",
    "GRAY": "#616161",
}

def _get_css_sanitizer():
    """Return a Bleach CSSSanitizer allowing only the minimal inline styles we inject."""
    global _css_sanitizer_cached
    if _css_sanitizer_cached is not None:
        return _css_sanitizer_cached
    try:
        from bleach.css_sanitizer import CSSSanitizer  # type: ignore
        _css_sanitizer_cached = CSSSanitizer(allowed_css_properties=["color", "font-weight"])
    except Exception:
        _css_sanitizer_cached = None
    return _css_sanitizer_cached

def apply_color_spans(text: str, enabled: bool = True) -> str:
    """Render Evidence-Linker tags with actual HTML colors (does not invent tags)."""
    if not enabled or not text:
        return text

    def repl(m: re.Match) -> str:
        tag = m.group("tag")
        suffix = m.group("suffix") or ""
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag, "#616161")
        token = f"[{tag}{suffix}]"
        if emoji:
            token = f"{token} {emoji}"
        return f"<span style=\"color:{color}; font-weight:600;\">{token}</span>"

    pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?")
    return pat.sub(repl, text)

def _reapply_color_styles_if_stripped(html_text: str) -> str:
    """If Bleach stripped inline CSS (style=""), re-apply our own safe color styles."""
    if not html_text:
        return html_text

    def repl(m: re.Match) -> str:
        tag = (m.group("tag") or "").upper()
        suffix = m.group("suffix") or ""
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag, "#616161")
        token = f"[{tag}{suffix}]"
        if emoji:
            token = f"{token} {emoji}"
        return f"<span style=\"color:{color}; font-weight:600;\">{token}</span>"

    pat = re.compile(
        r"<span\s+style=\"\"\s*>\s*\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?\s*</span>",
        flags=re.IGNORECASE,
    )
    return pat.sub(repl, html_text)

def sanitize_html(html_text: str) -> str:
    """Sanitize HTML (best-effort) while preserving our injected spans/images."""
    if not html_text:
        return ""
    if bleach is None:
        return html_text

    allowed_tags = [
        "p","br","b","strong","i","em","u","code","pre","blockquote","details","summary",
        "ul","ol","li","table","thead","tbody","tr","th","td","hr",
        "div","span","img","a",
        "h1","h2","h3","h4","h5","h6"
    ]
    allowed_attrs = {
        "*": ["class","style"],
        "a": ["href","title","target","rel","class","style"],
        "img": ["src","alt","title","style","loading","class"],
        "code": ["class"],
        "pre": ["class"],
        "details": ["open","class","style"],
        "summary": ["class","style"],
        "th": ["colspan","rowspan","class","style"],
        "td": ["colspan","rowspan","class","style"],
    }
    try:
        css_sanitizer = _get_css_sanitizer()
        cleaned = bleach.clean(
            html_text,
            tags=allowed_tags,
            attributes=allowed_attrs,
            protocols=["http","https","mailto"],
            strip=True,
            css_sanitizer=css_sanitizer,
        )
        if css_sanitizer is None:
            cleaned = _reapply_color_styles_if_stripped(cleaned)
        return cleaned
    except Exception:
        return html_text

def html_number_self_debunking(html_text: str, *, lang: str = "en") -> str:
    """Best-effort: add 1./2./3. numbering to Self-Debunking in already-rendered HTML."""
    try:
        if not html_text:
            return html_text
        if re.search(r"(?i)Self-Debunking|Selbst[- ]?Debunking", html_text) is None:
            return html_text
        lines = html_text.splitlines()
        out = []
        in_sd = False
        in_ol = False
        n = 0
        for ln in lines:
            if re.search(r"(?i)Self-Debunking|Selbst[- ]?Debunking", ln):
                in_sd = True
                in_ol = False
                n = 0
                out.append(ln)
                continue
            if in_sd:
                if re.search(r"(?i)>\s*QC(?:-Matrix)?\s*:", ln) or re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:", re.sub(r"<[^>]+>","",ln)):
                    in_sd = False
                    out.append(ln)
                    continue
                if "<ol" in ln.lower():
                    in_ol = True
                if "</ol" in ln.lower():
                    in_ol = False
                plain = re.sub(r"<[^>]+>", "", ln).strip()
                if (not in_ol) and re.match(r"(?i)^(Weakness|Schwäche)\b\s*:", plain):
                    n += 1
                    if n <= 6 and (f"{n}." not in plain):
                        if "<div" in ln:
                            ln = re.sub(r"(<div[^>]*>\s*)", r"\1" + str(n) + ". ", ln, count=1)
                        else:
                            ln = f"{n}. " + ln
                out.append(ln)
            else:
                out.append(ln)
        return "\n".join(out)
    except Exception:
        return html_text
