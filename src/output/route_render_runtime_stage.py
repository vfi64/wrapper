from __future__ import annotations

from typing import Callable


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def render_command_html_stage(
    *,
    raw_response: str,
    color_mode: str = "off",
    ui_lang: str = "en",
    comm_active: bool = False,
    rendering_pipeline_v192_mod=None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    normalize_known_markdown_control_headings_fn: Callable | None = None,
    apply_color_spans_fn: Callable | None = None,
    strip_color_markers_for_color_off_text_fn: Callable | None = None,
    strip_color_markers_for_color_off_html_fn: Callable | None = None,
    markdown_mod=None,
    sanitize_html_fn: Callable | None = None,
):
    """Render command responses (deterministic, fail-soft)."""
    raw = str(raw_response or "")
    color = str(color_mode or "off")
    ui = str(ui_lang or "en")

    tmp = _safe_call(unwrap_accidental_full_text_codefence_fn, raw)
    if tmp is not None:
        raw = str(tmp)
    tmp = _safe_call(normalize_known_markdown_control_headings_fn, raw)
    if tmp is not None:
        raw = str(tmp)

    if rendering_pipeline_v192_mod is not None:
        try:
            rctx_cls = getattr(rendering_pipeline_v192_mod, "RenderContext", None)
            render_fn = getattr(rendering_pipeline_v192_mod, "render_llm_text_to_html", None)
            if callable(rctx_cls) and callable(render_fn):
                rctx = rctx_cls(
                    ui_lang=ui,
                    color=color,
                    is_command=True,
                    comm_active=bool(comm_active),
                    strict=False,
                )
                out_html = str(render_fn(raw or "", rctx) or "")
                if color != "on":
                    tmp = _safe_call(strip_color_markers_for_color_off_html_fn, out_html)
                    if tmp is not None:
                        out_html = str(tmp)
                return out_html, None
        except Exception:
            pass

    raw_for_md = raw
    if color == "on":
        tmp = _safe_call(apply_color_spans_fn, raw_for_md, enabled=True)
    else:
        tmp = _safe_call(strip_color_markers_for_color_off_text_fn, raw_for_md)
    if tmp is not None:
        raw_for_md = str(tmp)

    try:
        md_fn = getattr(markdown_mod, "markdown", None) if markdown_mod is not None else None
        if callable(md_fn):
            out_html = str(md_fn(raw_for_md, extensions=["extra", "codehilite"]) or "")
        else:
            out_html = raw_for_md
    except Exception:
        out_html = raw_for_md
    if color != "on":
        tmp = _safe_call(strip_color_markers_for_color_off_html_fn, out_html)
        if tmp is not None:
            out_html = str(tmp)
    tmp = _safe_call(sanitize_html_fn, out_html)
    if tmp is not None:
        out_html = str(tmp)
    return out_html, None


def render_comm_inactive_html_stage(
    *,
    raw_response: str,
    color_mode: str = "off",
    ui_lang: str = "en",
    rendering_pipeline_v192_mod=None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    normalize_known_markdown_control_headings_fn: Callable | None = None,
    strip_governance_scaffolding_when_comm_inactive_fn: Callable | None = None,
    apply_color_spans_fn: Callable | None = None,
    strip_color_markers_for_color_off_text_fn: Callable | None = None,
    strip_color_markers_for_color_off_html_fn: Callable | None = None,
    markdown_mod=None,
    sanitize_html_fn: Callable | None = None,
    build_render_normalization_summary_fn: Callable | None = None,
    looks_like_rendered_html_runtime_fn: Callable | None = None,
    html_escape_fn: Callable | None = None,
):
    """Render comm-inactive responses (deterministic, fail-soft + normalization meta)."""
    raw = str(raw_response or "")
    color = str(color_mode or "off")
    ui = str(ui_lang or "en")
    html_out = ""

    try:
        tmp = _safe_call(unwrap_accidental_full_text_codefence_fn, raw)
        if tmp is not None:
            raw = str(tmp)
        tmp = _safe_call(normalize_known_markdown_control_headings_fn, raw)
        if tmp is not None:
            raw = str(tmp)
        tmp = _safe_call(strip_governance_scaffolding_when_comm_inactive_fn, raw)
        if tmp is not None:
            raw = str(tmp)

        if rendering_pipeline_v192_mod is not None:
            try:
                rctx_cls = getattr(rendering_pipeline_v192_mod, "RenderContext", None)
                render_fn = getattr(rendering_pipeline_v192_mod, "render_llm_text_to_html", None)
                if callable(rctx_cls) and callable(render_fn):
                    rctx = rctx_cls(
                        ui_lang=ui,
                        color=color,
                        is_command=False,
                        comm_active=False,
                        strict=False,
                    )
                    html_out = str(render_fn(raw or "", rctx) or "")
            except Exception:
                html_out = ""

        if not html_out:
            raw_for_md = raw
            if color == "on":
                tmp = _safe_call(apply_color_spans_fn, raw_for_md, enabled=True)
            else:
                tmp = _safe_call(strip_color_markers_for_color_off_text_fn, raw_for_md)
            if tmp is not None:
                raw_for_md = str(tmp)

            try:
                md_fn = getattr(markdown_mod, "markdown", None) if markdown_mod is not None else None
                if callable(md_fn):
                    html_out = str(md_fn(raw_for_md, extensions=["extra", "codehilite"]) or "")
                else:
                    html_out = raw_for_md
            except Exception:
                html_out = raw_for_md
            if color != "on":
                tmp = _safe_call(strip_color_markers_for_color_off_html_fn, html_out)
                if tmp is not None:
                    html_out = str(tmp)
            tmp = _safe_call(sanitize_html_fn, html_out)
            if tmp is not None:
                html_out = str(tmp)
    except Exception:
        esc = _safe_call(html_escape_fn, str(raw or ""))
        html_out = f"<pre>{str(esc or '')}</pre>" if esc is not None else "<pre></pre>"

    if color != "on":
        tmp = _safe_call(strip_color_markers_for_color_off_html_fn, html_out)
        if tmp is not None:
            html_out = str(tmp)

    try:
        norm = _safe_call(build_render_normalization_summary_fn, str(raw or ""), str(html_out or ""))
        if isinstance(norm, dict):
            probe = _safe_call(looks_like_rendered_html_runtime_fn, str(html_out or ""))
            render_ok = bool(probe) if probe is not None else bool(html_out and "<" in html_out and ">" in html_out)
            norm["render_ok"] = bool(render_ok)
            norm["render_fallback"] = (not bool(render_ok))
            return str(html_out or ""), {"normalization": norm}
    except Exception:
        pass

    return str(html_out or ""), {"normalization": {"render_ok": True, "render_fallback": False}}

