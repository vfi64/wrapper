from __future__ import annotations

from typing import Callable


def _safe_apply_text(text: str, fn: Callable | None) -> str:
    if not callable(fn):
        return str(text or "")
    try:
        out = fn(str(text or ""))
    except Exception:
        return str(text or "")
    if out is None:
        return str(text or "")
    return str(out)


def _safe_apply_html(html_body: str, fn: Callable | None, *, kwargs: dict | None = None) -> str:
    if not callable(fn):
        return str(html_body or "")
    try:
        if kwargs:
            return str(fn(str(html_body or ""), **kwargs) or "")
        return str(fn(str(html_body or "")) or "")
    except Exception:
        return str(html_body or "")


def apply_self_debunking_text_postprocess(
    text: str,
    *,
    output_pipeline_mod=None,
    normalize_inline_header_fn: Callable | None = None,
    enforce_contract_fn: Callable | None = None,
    normalize_numbering_fn: Callable | None = None,
    dedupe_sections_fn: Callable | None = None,
) -> str:
    """Central seam for deterministic Self-Debunking text postprocessing (fail-soft)."""
    out = str(text or "")
    if output_pipeline_mod is not None and hasattr(output_pipeline_mod, "normalize_self_debunking_postprocess_text"):
        try:
            return str(
                output_pipeline_mod.normalize_self_debunking_postprocess_text(
                    out,
                    normalize_inline_self_debunking_header_fn=normalize_inline_header_fn,
                    enforce_self_debunking_contract_fn=enforce_contract_fn,
                    normalize_self_debunking_numbering_text_fn=normalize_numbering_fn,
                    dedupe_self_debunking_sections_fn=dedupe_sections_fn,
                )
                or out
            )
        except Exception:
            pass
    out = _safe_apply_text(out, normalize_inline_header_fn)
    out = _safe_apply_text(out, enforce_contract_fn)
    out = _safe_apply_text(out, normalize_numbering_fn)
    out = _safe_apply_text(out, dedupe_sections_fn)
    return out


def apply_post_render_normalization(
    html_body: str,
    *,
    answer_lang: str = "de",
    color: str = "off",
    output_pipeline_mod=None,
    ensure_self_debunking_box_html_fn: Callable | None = None,
    sanitize_self_debunking_markdown_in_html_fn: Callable | None = None,
    normalize_hash_subheadings_in_html_fn: Callable | None = None,
    strip_internal_scaffolding_status_html_fn: Callable | None = None,
    html_number_self_debunking_fn: Callable | None = None,
    strip_color_markers_for_color_off_html_fn: Callable | None = None,
) -> str:
    """Central seam for deterministic post-render normalization (fail-soft)."""
    out = str(html_body or "")
    if output_pipeline_mod is not None and hasattr(output_pipeline_mod, "post_render_normalization"):
        try:
            return str(
                output_pipeline_mod.post_render_normalization(
                    out,
                    answer_lang=answer_lang,
                    color=str(color or "off"),
                    ensure_self_debunking_box_html_fn=ensure_self_debunking_box_html_fn,
                    sanitize_self_debunking_markdown_in_html_fn=sanitize_self_debunking_markdown_in_html_fn,
                    normalize_hash_subheadings_in_html_fn=normalize_hash_subheadings_in_html_fn,
                    strip_internal_scaffolding_status_html_fn=strip_internal_scaffolding_status_html_fn,
                    html_number_self_debunking_fn=html_number_self_debunking_fn,
                    strip_color_markers_for_color_off_html_fn=strip_color_markers_for_color_off_html_fn,
                )
                or out
            )
        except Exception:
            pass

    out = _safe_apply_html(out, ensure_self_debunking_box_html_fn, kwargs={"lang": answer_lang})
    out = _safe_apply_html(out, sanitize_self_debunking_markdown_in_html_fn)
    out = _safe_apply_html(out, normalize_hash_subheadings_in_html_fn)
    out = _safe_apply_html(out, strip_internal_scaffolding_status_html_fn)
    out = _safe_apply_html(out, html_number_self_debunking_fn, kwargs={"lang": answer_lang})
    out = _safe_apply_html(out, sanitize_self_debunking_markdown_in_html_fn)
    out = _safe_apply_html(out, ensure_self_debunking_box_html_fn, kwargs={"lang": answer_lang})
    if str(color or "off").strip().lower() != "on":
        out = _safe_apply_html(out, strip_color_markers_for_color_off_html_fn)
    return out
