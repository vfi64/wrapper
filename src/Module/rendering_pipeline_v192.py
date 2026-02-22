
"""
rendering_pipeline_v192.py

Deterministic-ish rendering pipeline for Wrapper v192.

Design goals (practical):
- Single entry point for all "answer-like" render paths.
- Plaintext normalization BEFORE Markdown.
- DOM normalization AFTER Markdown, BEFORE final sanitization.
- Sanitization LAST (Bleach) with CSS sanitizer allowing only safe properties.
- Robust fallbacks when optional deps are missing (never crash).
- Avoid regex-on-HTML for structural edits when DOM parser is available.

NOTE: True determinism requires pinned dependency versions in CI.
"""

from __future__ import annotations

import html as _html
import re as _re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

# --- Optional deps ---
try:
    import markdown as _markdown_lib  # type: ignore
except Exception:  # pragma: no cover
    _markdown_lib = None

try:
    import bleach  # type: ignore
    from bleach.css_sanitizer import CSSSanitizer as _CSSSanitizer  # type: ignore
except Exception:  # pragma: no cover
    bleach = None
    _CSSSanitizer = None

try:
    from bs4 import BeautifulSoup, Tag, NavigableString  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None
    Tag = None
    NavigableString = None


# --- Evidence color palette (stable + matches your tests) ---
_EVIDENCE_COLOR: Dict[str, str] = {
    "GREEN": "#2e7d32",
    "YELLOW": "#f9a825",
    "RED": "#c62828",
    "GRAY": "#616161",
    "WHITE": "#616161",   # treat white circle as gray (legacy)
}
_EVIDENCE_ICON: Dict[str, str] = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
    "GRAY": "⚪",
    "WHITE": "⚪",
}

_ALLOWED_CSS_PROPS = [
    "color", "font-weight",
    "background", "background-color",
    "border", "border-left",
    "padding", "margin", "border-radius",
    "font-size",
]

# --- Self-Debunking styling (kept minimal) ---
_SD_BOX_STYLE = (
    "border-left: 4px solid #a5b4fc; "
    "background-color: #eef2ff; "
    "padding: 10px; "
    "border-radius: 8px; "
    "margin: 10px 0;"
)

_SD_TITLE_STYLE = "font-weight:700; color:#4338ca; margin-bottom:6px;"

# --- Core config ---
@dataclass
class RenderContext:
    ui_lang: str = "en"          # "en" or "de"
    color: str = "off"           # "on"/"off"
    is_command: bool = False     # command vs answer
    comm_active: bool = True     # affects wrapper paths; here just for parity
    strict: bool = True          # whether to enforce wrapper invariants


# ---------------------------
# Plaintext normalization
# ---------------------------

# SCI menu detection: require a "cluster" (header + instruction and/or options list),
# so we don't delete legitimate content quoting a single sentence.
_SCI_MENU_HEADER_RE = _re.compile(
    r"(?im)^\s*(SCI\s*[- ]?\s*Varianten(?:menü)?\s*(?:\(|:)?|SCI\s*variants\s*(?:\(|:)?).*?$"
)
_SCI_MENU_INSTR_RE = _re.compile(
    r"(?im)(antworte.*buchstaben|select.*variant|choose.*variant|within.*frist|timeout.*assumed)"
)
_SCI_MENU_OPTIONS_RE = _re.compile(
    r"(?im)(?:^|\n)\s*([A-Ia-i])\s*[\)\.\-:]\s+"
)

def strip_sci_menu_leaks_plaintext(text: str) -> str:
    """
    Remove leaked SCI menu blocks from answer text.

    Conservative, but robust:
    - Only removes when we see a header AND (instruction OR >=3 option lines).
    - Removes only the contiguous "menu cluster".
    - Stops when real content begins (e.g., "Antwort:", "Answer:", normal paragraphs),
      even if there is only a single blank line.
    """
    if not text or "SCI" not in text:
        return text

    lines = text.splitlines()
    out: List[str] = []
    i = 0

    # helper patterns
    header_re = _SCI_MENU_HEADER_RE
    instr_re = _SCI_MENU_INSTR_RE
    opt_line_re = _re.compile(r"(?im)^\s*[A-Ia-i]\s*[\)\.\-:]\s+")
    # "real content" starts: common answer starters OR anything that is not option/instruction once we've entered menu
    answer_start_re = _re.compile(r"(?im)^\s*(Antwort|Answer)\b\s*:")  # prefix, not whole line
    qc_start_re = _re.compile(r"(?im)^\s*QC(?:-Matrix)?\s*:")

    while i < len(lines):
        line = lines[i]
        if header_re.search(line):
            window = "\n".join(lines[i:i+60])
            has_instr = bool(instr_re.search(window))
            # count option lines in the next 60 lines
            opt_count = sum(1 for ln in lines[i:i+60] if opt_line_re.search(ln))
            has_opts = opt_count >= 3

            if has_instr or has_opts:
                j = i + 1
                # consume menu lines until we hit real content
                while j < len(lines):
                    ln = lines[j]

                    # stop if we hit an obvious answer/QC start (these are not part of the menu)
                    if answer_start_re.search(ln) or qc_start_re.search(ln):
                        break

                    # allow typical menu lines: blanks, instructions, option lines, short hints
                    if (not ln.strip()) or instr_re.search(ln) or opt_line_re.search(ln):
                        j += 1
                        continue

                    # if we've already seen options/instruction cluster, any other non-empty line ends the menu
                    break

                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


# ---------------------------
# Evidence color spans
# ---------------------------

# Supports:
# [GREEN] 🟢
# [GREEN: 90%] 🟢
# [GREEN] (no emoji)
# also "⚪" treated as WHITE
_EVIDENCE_TAG_RE = _re.compile(
    r"\[(?P<tag>GREEN|YELLOW|RED|GRAY|WHITE)(?P<suffix>(?:-[A-Z0-9]+)*|\s*:[^\]]+|\s*\d{1,3}%\s*)?\]\s*(?P<emoji>🟢|🟡|🔴|⚪)?",
    flags=_re.IGNORECASE
)

def apply_color_spans(text: str, enabled: bool) -> str:
    """
    Inject <span> wrappers and render only the evidence icon.
    """
    if not enabled or not text:
        return text

    def repl(m: _re.Match) -> str:
        tag_raw = (m.group("tag") or "").upper()
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag_raw, _EVIDENCE_COLOR["GRAY"])
        vis = emoji or _EVIDENCE_ICON.get(tag_raw, "⚪")
        return f'<span data-evidence="{tag_raw}" style="color: {color}; font-weight: 700;">{_html.escape(vis)}</span>'

    return _EVIDENCE_TAG_RE.sub(repl, text)


# ---------------------------
# Markdown -> HTML
# ---------------------------

def markdown_to_html(text: str) -> str:
    if not text:
        return ""
    if _markdown_lib is None:
        # Deterministic fallback (no partial HTML execution): escape and show preformatted
        return "<pre>" + _html.escape(str(text)) + "</pre>"
    try:
        return _markdown_lib.markdown(text, extensions=["fenced_code", "tables"])
    except Exception:
        try:
            return _markdown_lib.markdown(text, extensions=["fenced_code"])
        except Exception:
            return "<pre>" + _html.escape(str(text)) + "</pre>"


# ---------------------------
# DOM normalization
# ---------------------------

_QC_RE = _re.compile(r"(?i)^\s*QC(?:-Matrix)?\s*:")

def _dom_iter_block_candidates(root) -> List[Any]:
    if root is None:
        return []
    # prioritize blocks likely to contain "footer-like" lines
    return list(root.find_all(["p", "div", "li", "blockquote"]))

def _fix_double_sci_trace_headers_dom(soup):
    """
    Remove a standalone block that is only 'SCI Trace:' if the next significant sibling
    is an SCI trace card/block that already contains a title.
    """
    if soup is None:
        return
    patt = _re.compile(r"(?i)^\s*SCI\s*Trace\s*:?\s*$")
    for block in soup.find_all(["p", "div", "h3", "h4"]):
        txt = block.get_text(strip=True)
        if not patt.match(txt):
            continue

        # find next element sibling
        nxt = block.next_sibling
        while nxt and isinstance(nxt, NavigableString) and not str(nxt).strip():
            nxt = nxt.next_sibling
        if not nxt or not isinstance(nxt, Tag):
            continue

        nxt_text = nxt.get_text(" ", strip=True)
        nxt_class = " ".join(nxt.get("class", []) or [])
        # If next block looks like an SCI trace container
        if ("SCI Trace" in nxt_text) or ("sci-trace" in nxt_class.lower()):
            block.decompose()


def _canonicalize_qc_footer_dom(soup):
    """
    Ensure only one QC footer block remains and it is moved to the end of <body> (or soup).
    Conservative: operate on blocks whose text starts with QC:
    """
    if soup is None:
        return
    body = soup.body or soup
    candidates = []
    for block in _dom_iter_block_candidates(body):
        txt = block.get_text("\n", strip=True)
        if _QC_RE.match(txt):
            candidates.append(block)

    if not candidates:
        return

    last = candidates[-1]
    for c in candidates[:-1]:
        c.decompose()

    # Move 'last' to end of body (preserving structure)
    try:
        last.extract()
        body.append(last)
    except Exception:
        pass


def _reapply_color_styles_if_stripped_dom(soup):
    """
    CI-safe: if sanitization strips inline styles, reapply 'color' and 'font-weight'
    based on data-evidence attribute or visible marker.
    """
    if soup is None:
        return
    for sp in soup.find_all("span"):
        # Determine evidence tag
        tag = (sp.get("data-evidence") or "").upper().strip()
        if not tag:
            txt = sp.get_text(" ", strip=True)
            m = _re.search(r"\[(GREEN|YELLOW|RED|GRAY|WHITE)", txt, flags=_re.IGNORECASE)
            if m:
                tag = m.group(1).upper()
        if not tag:
            continue
        # If style missing or doesn't contain 'color'
        style = sp.get("style") or ""
        if ("color" not in style) or ("#" not in style):
            color = _EVIDENCE_COLOR.get(tag, _EVIDENCE_COLOR["GRAY"])
            sp["style"] = f"color: {color}; font-weight: 700;"


_SD_HEADER_RE = _re.compile(r"(?i)^\s*(Self|Selbst)\s*[- ]?\s*Debunking\s*:?\s*$")
_SD_LABELS_EN = ["Weakness", "Why it matters", "What would verify/falsify (next check)"]
_SD_LABELS_DE = ["Schwäche", "Warum das wichtig ist", "Was würde prüfen/widerlegen (nächster Check)"]

def _sd_localized_title(ui_lang: str) -> str:
    return "Selbst-Debunking:" if ui_lang.lower().startswith("de") else "Self-Debunking:"

def _sd_labels(ui_lang: str) -> List[str]:
    return _SD_LABELS_DE if ui_lang.lower().startswith("de") else _SD_LABELS_EN


def _process_self_debunking_dom(soup, ui_lang: str):
    """
    Wrap SD section in a box; enforce numbering and bold labels (strict, but localized).
    Strategy:
      - find a block that is exactly "Self-Debunking:" or "Selbst-Debunking:"
      - wrap subsequent sibling blocks until QC footer (or end)
      - inside box:
          - ensure each "Weakness/Schwäche" starts with "N. **Label**:"
          - bold the other two labels when present at line starts
    """
    if soup is None:
        return
    body = soup.body or soup

    header_block = None
    for block in body.find_all(["p", "div", "h3", "h4"]):
        if _SD_HEADER_RE.match(block.get_text(" ", strip=True)):
            header_block = block
            break
    if header_block is None:
        return

    # Collect siblings until QC
    nodes: List[Any] = []
    curr = header_block.next_sibling
    while curr:
        nxt = curr.next_sibling
        if isinstance(curr, NavigableString) and not str(curr).strip():
            curr = nxt
            continue

        # Stop at QC block
        if isinstance(curr, Tag):
            txt = curr.get_text("\n", strip=True)
            if _QC_RE.match(txt):
                break
        nodes.append(curr)
        curr = nxt

    # Create box
    box = soup.new_tag("div")
    box["class"] = (box.get("class", []) or []) + ["self-debunking"]
    box["style"] = _SD_BOX_STYLE

    title = soup.new_tag("div")
    title["style"] = _SD_TITLE_STYLE
    title.string = _sd_localized_title(ui_lang)
    box.append(title)

    # Move nodes into box
    for n in nodes:
        try:
            n.extract()
            box.append(n)
        except Exception:
            pass

    # Replace header with box
    try:
        header_block.replace_with(box)
    except Exception:
        return

    # Enforce numbering + bold labels
    # Step 0: If SD content is emitted as a single paragraph with embedded newlines,
    # split it into multiple <p> blocks so labeling/numbering works reliably.
    def _split_multiline_sd_blocks(box_tag):
        try:
            for blk in list(box_tag.find_all(["p", "div", "li"], recursive=True)):
                if blk is title:
                    continue
                txt = blk.get_text("\n", strip=False)
                if not txt or "\n" not in txt:
                    continue
                raw_lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
                # Merge continuation lines starting with ':' into previous line
                lines = []
                for ln in raw_lines:
                    if ln.startswith(":") and lines:
                        lines[-1] = (lines[-1].rstrip() + " " + ln.lstrip(":").strip()).strip()
                        continue
                    lines.append(ln)
                if len(lines) <= 1:
                    continue
                blk.clear()
                blk.append(NavigableString(lines[0]))
                insert_after = blk
                for ln in lines[1:]:
                    newp = soup.new_tag("p")
                    newp.append(NavigableString(ln))
                    insert_after.insert_after(newp)
                    insert_after = newp
        except Exception:
            return

    _split_multiline_sd_blocks(box)

    labels = _sd_labels(ui_lang)
    # Patterns accept plain, markdown-bold, or <strong>
    weakness_keys = [labels[0], "Weakness", "Schwäche"]
    weakness_re = _re.compile(
        r"(?i)^\s*(?:(\d+)\.\s*)?(?:\*\*)?(Weakness|Schwäche)(?:\*\*)?\s*:?\s*(.*)$"
    )
    other_label_res = [
        (_re.compile(r"(?i)^\s*(?:\*\*)?(" + _re.escape(labels[1]) + r")(?:\*\*)?\s*:?\s*(.*)$"), labels[1]),
        (_re.compile(r"(?i)^\s*(?:\*\*)?(" + _re.escape(labels[2]) + r")(?:\*\*)?\s*:?\s*(.*)$"), labels[2]),
    ]

    counter = 0
    for block in box.find_all(["p", "div", "li"]):
        # Skip the title itself
        if block is title:
            continue
        txt = block.get_text(" ", strip=True)
        m = weakness_re.match(txt)
        if m:
            counter += 1
            label = labels[0]  # localized label
            rest = m.group(3) or ""
            rest = rest.lstrip(":").strip()
            # rebuild: <strong>N. Label</strong>: rest
            block.clear()
            strong = soup.new_tag("strong")
            # Avoid double numbering if we're already inside an <ol><li>
            parent_li = block if getattr(block, 'name', None) == 'li' else block.find_parent('li')
            in_ordered_list = bool(parent_li and parent_li.find_parent('ol'))
            strong.string = f"{label}" if in_ordered_list else f"{counter}. {label}"
            block.append(strong)
            if rest:
                block.append(NavigableString(": "))
                # keep rest as text (escaped by serializer later / sanitized later)
                block.append(NavigableString(rest))
            continue

        # bold other labels if they appear as "Label: ..."
        for rx, canonical in other_label_res:
            mm = rx.match(txt)
            if not mm:
                continue
            rest = mm.group(2) or ""
            block.clear()
            strong = soup.new_tag("strong")
            strong.string = canonical
            block.append(strong)
            block.append(NavigableString(": "))
            block.append(NavigableString(rest))
            break


def dom_normalize(html_text: str, ctx: RenderContext) -> str:
    if not html_text or BeautifulSoup is None:
        return html_text
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        # stage order matters
        _fix_double_sci_trace_headers_dom(soup)
        _process_self_debunking_dom(soup, ctx.ui_lang)
        _canonicalize_qc_footer_dom(soup)
        # (after structural edits) ensure evidence spans remain colored
        _reapply_color_styles_if_stripped_dom(soup)
        return str(soup)
    except Exception:
        return html_text


# ---------------------------
# Sanitization (final)
# ---------------------------

_css_sanitizer_cached = None

def _get_css_sanitizer():
    global _css_sanitizer_cached
    if _css_sanitizer_cached is not None:
        return _css_sanitizer_cached
    if _CSSSanitizer is None:
        _css_sanitizer_cached = None
        return None
    try:
        _css_sanitizer_cached = _CSSSanitizer(allowed_css_properties=_ALLOWED_CSS_PROPS)
    except Exception:
        _css_sanitizer_cached = None
    return _css_sanitizer_cached


_ALLOWED_TAGS = [
    "p", "br", "strong", "b", "em", "i", "u",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "code", "pre", "span", "div", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td", "a",
]
_ALLOWED_ATTRS = {
    "*": ["class", "style", "data-evidence"],
    "a": ["href", "title", "target"],
}

def sanitize_html(html_text: str) -> str:
    if not html_text:
        return ""

    # If bleach is unavailable, do a best-effort scrub and still return renderable HTML.
    # This is safer than passing raw HTML, and avoids the "<pre>&lt;...&gt;</pre>" behavior
    # that makes the UI show source code instead of rendering.
    if bleach is None:
        if BeautifulSoup is None:
            # Last-resort: allow rendering, but remove the most dangerous constructs via regex.
            txt = html_text
            txt = _re.sub(r"(?is)<\s*(script|iframe|object|embed|link|meta)[^>]*>.*?<\s*/\s*\1\s*>", "", txt)
            txt = _re.sub(r"(?is)<\s*(script|iframe|object|embed|link|meta)[^>]*/\s*>", "", txt)
            txt = _re.sub(r"(?i)\son\w+\s*=\s*([\"']).*?\1", "", txt)  # strip onClick/onLoad...
            txt = _re.sub(r"(?i)(href|src)\s*=\s*([\"'])\s*javascript:.*?\2", r"\1=\2#\2", txt)
            return txt

        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # Drop dangerous tags entirely
            for bad in soup.find_all(["script", "iframe", "object", "embed", "link", "meta"]):
                bad.decompose()

            # Strip event handler attributes and javascript: URLs
            for tag in soup.find_all(True):
                attrs = dict(tag.attrs) if getattr(tag, "attrs", None) else {}
                for k in list(attrs.keys()):
                    if k.lower().startswith("on"):
                        del tag.attrs[k]
                for url_attr in ("href", "src"):
                    v = tag.get(url_attr)
                    if isinstance(v, str) and v.strip().lower().startswith("javascript:"):
                        tag[url_attr] = "#"

                # Keep only a conservative set of attributes
                allowed = {"class", "style", "data-evidence", "href", "title", "target"}
                for k in list(tag.attrs.keys()):
                    if k not in allowed:
                        del tag.attrs[k]

            return str(soup)
        except Exception:
            return html_text

    try:
        return bleach.clean(
            html_text,
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRS,
            css_sanitizer=_get_css_sanitizer(),
            strip=True,
        )
    except Exception:
        # If bleach itself errors, fall back to best-effort DOM scrub (if possible)
        if BeautifulSoup is None:
            return html_text
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for bad in soup.find_all(["script", "iframe", "object", "embed", "link", "meta"]):
                bad.decompose()
            return str(soup)
        except Exception:
            return html_text


# ---------------------------
# Single entry point
# ---------------------------

def render_llm_text_to_html(raw_text: str, ctx: Optional[RenderContext] = None) -> str:
    """
    The one function you call from Wrapper v192 for answer rendering.

    - For commands (ctx.is_command=True), you can still use it, but you may choose to bypass
      SCI/SD specific normalization in your wrapper logic.
    """
    ctx = ctx or RenderContext()
    text = raw_text or ""

    # 1) Plaintext normalization (safe + deterministic)
    if ctx.strict and not ctx.is_command:
        text = strip_sci_menu_leaks_plaintext(text)

    # 2) Evidence spans (still plaintext; inserted as HTML snippets; markdown will pass them through)
    if ctx.color.lower() == "on":
        text = apply_color_spans(text, enabled=True)

    # 3) Markdown -> HTML
    html_out = markdown_to_html(text)

    # 4) DOM normalize (structure edits)
    if ctx.strict and not ctx.is_command:
        html_out = dom_normalize(html_out, ctx)

    # 5) Sanitize last
    html_out = sanitize_html(html_out)

    # 6) CI safety: if bleach stripped styles, reapply deterministically (DOM if possible)
    # (Do this after sanitize; safe because we only apply whitelisted props and evidence palette.)
    if BeautifulSoup is not None and ctx.color.lower() == "on":
        try:
            soup = BeautifulSoup(html_out, "html.parser")
            _reapply_color_styles_if_stripped_dom(soup)
            html_out = str(soup)
        except Exception:
            pass

    return html_out


# ---------------------------
# Self-check fixtures (internal)
# ---------------------------

def _self_check() -> Tuple[bool, List[str]]:
    """
    Dependency-aware quick checks.

    If optional deps (markdown/bs4/bleach) are missing, the pipeline will fall back to
    safe <pre> rendering; in that mode, structural DOM invariants cannot be enforced.
    This check therefore verifies:
      - invariants that MUST hold even in fallback mode, and
      - richer invariants only when deps are available.
    """
    failures: List[str] = []

    def assert_true(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    ctx = RenderContext(ui_lang="de", color="on", is_command=False, strict=True)

    # 1) SCI menu leak removal is plaintext-stage, must work even without deps.
    raw = (
        "SCI-Variantenmenü (Auswahl a–i)\n"
        "Antworte mit genau einem Buchstaben.\n"
        "A) Foo\nB) Bar\nC) Baz\n\n"
        "Antwort: Zeit ist ...\n\n"
        "QC-Matrix: X"
    )
    stripped = strip_sci_menu_leaks_plaintext(raw)
    assert_true("SCI-Varianten" not in stripped, "SCI menu leak not removed (plaintext)")
    assert_true("Antwort: Zeit ist" in stripped, "Answer content removed by SCI strip")

    # 2) Evidence span injection is plaintext-stage. Should inject <span> when enabled.
    raw2 = "Hello\n\n[GREEN] 🟢 claim"
    colored = apply_color_spans(raw2, enabled=True)
    assert_true("<span" in colored and "data-evidence" in colored, "Evidence spans not injected")

    # If markdown or bs4 is missing, we cannot enforce DOM-based invariants in this environment.
    if _markdown_lib is None or BeautifulSoup is None:
        return (len(failures) == 0), failures

    # 3) Self-debunking DOM: localized title + numbering on Weakness/Schwäche
    raw3 = (
        "Self-Debunking:\n\n"
        "Schwäche: Test\n"
        "Warum das wichtig ist: X\n"
        "Was würde prüfen/widerlegen (nächster Check): Y\n\n"
        "QC-Matrix: end"
    )
    out3 = render_llm_text_to_html(raw3, ctx)
    assert_true("Selbst-Debunking" in out3, "SD title not localized to DE")
    assert_true("1. Schwäche" in _html.unescape(out3), "SD weakness not numbered")

    # 4) QC footer unique
    raw4 = "Text\n\nQC-Matrix: first\n\nMore\n\nQC-Matrix: last"
    out4 = render_llm_text_to_html(raw4, ctx)
    assert_true(_html.unescape(out4).count("QC-Matrix") == 1, "QC footer not deduplicated")

    return (len(failures) == 0), failures
