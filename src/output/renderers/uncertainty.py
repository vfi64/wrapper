from __future__ import annotations

import re
from typing import Callable


def append_uncertainty_explanation_if_needed(
    html_text: str,
    *,
    user_text: str = "",
    lang: str = "de",
    uncertainty_codes_mod=None,
    annotate_signal_dot_tooltips_fn: Callable | None = None,
) -> str:
    """Apply deterministic uncertainty markers/tooltips and strip legend blocks."""
    txt = str(html_text or "")
    if not txt:
        return txt

    out = txt
    try:
        if uncertainty_codes_mod is not None and hasattr(uncertainty_codes_mod, "ensure_uncertainty_annotations_html"):
            out = uncertainty_codes_mod.ensure_uncertainty_annotations_html(
                txt,
                lang=lang,
                user_text=str(user_text or ""),
            )
        elif uncertainty_codes_mod is not None and hasattr(uncertainty_codes_mod, "append_uncertainty_legend_html"):
            out = uncertainty_codes_mod.append_uncertainty_legend_html(txt, lang=lang)
    except Exception:
        out = txt

    try:
        if callable(annotate_signal_dot_tooltips_fn):
            out = annotate_signal_dot_tooltips_fn(out, lang=lang)
    except Exception:
        pass

    try:
        out = re.sub(
            r"(?is)<details\b[^>]*class=(?:\"|')[^\"']*\buncertainty-legend\b[^\"']*(?:\"|')[^>]*>.*?</details>\s*",
            "",
            str(out or ""),
        )
    except Exception:
        pass
    return out

