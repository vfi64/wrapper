from __future__ import annotations

import re

# Evidence marker emojis used by the wrapper/runtime.
_MARKER_EMOJI_RX = r"[🟢🟡🔴⚪⚪️]"
_TAG_RX = r"\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]"


def strip_color_markers_text(text: str) -> str:
    """Remove color marker artifacts for Color=off (both tags and marker glyphs)."""
    if not text:
        return text

    out = str(text)
    # Remove bracket tags and adjacent marker emoji in one pass.
    out = re.sub(rf"(?i){_TAG_RX}\s*{_MARKER_EMOJI_RX}*", "", out)
    # Remove any remaining standalone bracket tags.
    out = re.sub(rf"(?i){_TAG_RX}", "", out)
    # Remove standalone marker runs at line starts (common in weak-model outputs).
    out = re.sub(rf"(?m)^\s*(?:{_MARKER_EMOJI_RX}\s*)+", "", out)
    # Remove remaining marker emojis conservatively.
    out = re.sub(_MARKER_EMOJI_RX, "", out)
    # Avoid spaces before punctuation introduced by marker stripping.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    # Normalize spaces after removals.
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


def strip_color_markers_html(html_text: str) -> str:
    """Remove color marker artifacts from rendered HTML for Color=off."""
    if not html_text:
        return html_text

    out = str(html_text)
    # Unwrap helper marker wrapper spans introduced by tooltip/fallback logic.
    out = re.sub(
        r"(?is)<span\b[^>]*class=(?:\"|')[^\"']*signal-dot-marker[^\"']*(?:\"|')[^>]*>(.*?)</span>",
        r"\1",
        out,
    )
    # Remove textual evidence tags that may survive markdown rendering as plain text.
    out = re.sub(rf"(?i){_TAG_RX}", "", out)
    # Drop marker-only span containers.
    out = re.sub(rf"(?is)<span\b[^>]*>\s*{_MARKER_EMOJI_RX}\s*</span>", "", out)
    # Remove remaining marker emojis.
    out = re.sub(_MARKER_EMOJI_RX, "", out)
    # Avoid spaces before punctuation introduced by marker stripping.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    # Clean up empty marker paragraphs/list rows.
    out = re.sub(r"(?is)<(p|li|div)\b[^>]*>\s*</\1>", "", out)
    return out
