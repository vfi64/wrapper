from __future__ import annotations

import re
from typing import Callable


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def render_final_html_body_stage(
    *,
    raw_for_render: str,
    color_mode: str = "off",
    answer_lang: str = "de",
    ui_lang_fallback: str = "en",
    rendering_pipeline_v192_mod=None,
    auto_embed_image_urls_fn: Callable | None = None,
    apply_color_spans_fn: Callable | None = None,
    strip_color_markers_for_color_off_text_fn: Callable | None = None,
    normalize_markdown_list_spacing_fn: Callable | None = None,
    normalize_known_markdown_control_headings_fn: Callable | None = None,
    markdown_mod=None,
    sanitize_html_fn: Callable | None = None,
    html_number_self_debunking_fn: Callable | None = None,
    apply_post_render_normalization_seam_fn: Callable | None = None,
):
    """Render body runtime stage (pre-render text transforms + HTML render + post-normalization)."""
    out_raw = str(raw_for_render or "")
    color = str(color_mode or "off")

    lang_seed = str(answer_lang or ui_lang_fallback or "en")
    ans_lang = "de" if lang_seed.lower().startswith("de") else "en"

    tmp = _safe_call(auto_embed_image_urls_fn, out_raw)
    if tmp is not None:
        out_raw = str(tmp)

    if color == "on":
        tmp = _safe_call(apply_color_spans_fn, out_raw, enabled=True)
    else:
        tmp = _safe_call(strip_color_markers_for_color_off_text_fn, out_raw)
    if tmp is not None:
        out_raw = str(tmp)

    tmp = _safe_call(normalize_markdown_list_spacing_fn, out_raw)
    if tmp is not None:
        out_raw = str(tmp)
    out_raw = re.sub(r"(?<!\n)\nQC-Matrix:", r"\n\nQC-Matrix:", out_raw)

    tmp = _safe_call(normalize_known_markdown_control_headings_fn, out_raw)
    if tmp is not None:
        out_raw = str(tmp)

    final_html_body = ""
    if rendering_pipeline_v192_mod is not None:
        try:
            rctx_cls = getattr(rendering_pipeline_v192_mod, "RenderContext", None)
            render_fn = getattr(rendering_pipeline_v192_mod, "render_llm_text_to_html", None)
            if callable(rctx_cls) and callable(render_fn):
                rctx = rctx_cls(
                    ui_lang=ans_lang,
                    color=color,
                    is_command=False,
                    comm_active=True,
                    strict=True,
                )
                final_html_body = str(render_fn(out_raw or "", rctx) or "")
        except Exception:
            final_html_body = ""

    if not final_html_body:
        try:
            md_fn = getattr(markdown_mod, "markdown", None) if markdown_mod is not None else None
            if callable(md_fn):
                try:
                    final_html_body = str(md_fn(out_raw, extensions=["extra", "codehilite"]) or "")
                except Exception:
                    final_html_body = str(md_fn(out_raw, extensions=["fenced_code", "tables"]) or "")
            else:
                final_html_body = out_raw
        except Exception:
            final_html_body = out_raw

        tmp = _safe_call(sanitize_html_fn, final_html_body)
        if tmp is not None:
            final_html_body = str(tmp)
        tmp = _safe_call(html_number_self_debunking_fn, final_html_body, lang=ans_lang)
        if tmp is not None:
            final_html_body = str(tmp)

    tmp = _safe_call(
        apply_post_render_normalization_seam_fn,
        str(final_html_body or ""),
        answer_lang=ans_lang,
        color=color,
    )
    if tmp is not None:
        final_html_body = str(tmp)

    return str(out_raw or ""), str(final_html_body or "")

