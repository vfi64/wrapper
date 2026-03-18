from __future__ import annotations

import html
from typing import Callable


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def finalize_render_end_html_stage(
    *,
    alert_html: str,
    final_html_body: str,
    raw_for_render: str,
    raw_original: str,
    raw_response: str,
    csc_meta,
    answer_lang: str = "de",
    build_render_normalization_summary_fn: Callable | None = None,
    looks_like_rendered_html_runtime_fn: Callable | None = None,
    detect_probable_truncation_fn: Callable | None = None,
    control_layer_alert_html_fn: Callable | None = None,
    format_response_timestamp_fn: Callable | None = None,
    render_ts_footer_html_fn: Callable | None = None,
):
    """Late render-end assembly stage for chat HTML (deterministic + fail-soft)."""
    out_alert = str(alert_html or "")
    out_body = str(final_html_body or "")
    meta = csc_meta

    try:
        norm_summary = _safe_call(
            build_render_normalization_summary_fn,
            str(raw_for_render or ""),
            str(out_body or ""),
        )
        if isinstance(norm_summary, dict):
            if meta is None:
                meta = {}
            if isinstance(meta, dict):
                meta["normalization"] = norm_summary
    except Exception:
        pass

    render_ok_probe = _safe_call(looks_like_rendered_html_runtime_fn, str(out_body or ""))
    if render_ok_probe is None:
        render_ok = bool(out_body and "<" in out_body and ">" in out_body and "&lt;" not in out_body)
    else:
        render_ok = bool(render_ok_probe)

    try:
        raw_probe = (raw_original or raw_response or "")
        trunc_probe = _safe_call(detect_probable_truncation_fn, raw_probe, out_body or "")
        trunc = False
        trunc_msg = ""
        if isinstance(trunc_probe, (tuple, list)) and len(trunc_probe) >= 2:
            trunc = bool(trunc_probe[0])
            trunc_msg = str(trunc_probe[1] or "")
        if trunc and trunc_msg:
            note_html = _safe_call(
                control_layer_alert_html_fn,
                trunc_msg + " Bitte gegenprüfen oder Antwort neu generieren.",
                title="CONTROL LAYER NOTE",
                severity="warn",
                lang=str(answer_lang or "de"),
            )
            out_alert = str(note_html or "") + (out_alert or "")
            if meta is None:
                meta = {}
            if isinstance(meta, dict):
                meta["probable_truncation"] = True
    except Exception:
        pass

    try:
        if meta is None:
            meta = {}
        if isinstance(meta, dict):
            ns = meta.get("normalization")
            if isinstance(ns, dict):
                ns["render_ok"] = bool(render_ok)
                ns["render_fallback"] = (not bool(render_ok))
    except Exception:
        pass

    timestamp = _safe_call(format_response_timestamp_fn)
    timestamp_s = str(timestamp or "")
    ts_footer_html = _safe_call(render_ts_footer_html_fn, timestamp_s)
    if ts_footer_html is None:
        ts_footer_html = f'<div class="ts-footer">Response at {html.escape(timestamp_s)}</div>'
    ts_footer_html = str(ts_footer_html or "")

    if render_ok:
        # Normal chat output should stay clean. Raw model output is reserved for fallback/error paths.
        return out_alert + out_body + ts_footer_html, meta

    try:
        raw_esc = html.escape(str(raw_original or raw_response or ""))
        fallback_note = '<div class="note-box"><b>Render fallback</b>: showing raw model output.</div>'
        return out_alert + fallback_note + '<pre class="raw-output-pre">' + raw_esc + "</pre>" + ts_footer_html, meta
    except Exception:
        return out_alert + ts_footer_html, meta
