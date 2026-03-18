from __future__ import annotations

from typing import Callable


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def ensure_qc_footer_html_consistency_html_stage(
    *,
    final_html_body: str,
    raw_for_render: str,
    profile_name: str,
    gov_mgr,
    overrides: dict | None,
    qc_footer_for_profile_fn,
    ensure_qc_footer_present_fn,
    enforce_qc_footer_deltas_fn,
    ensure_qc_footer_is_last_fn,
    qc_probe_is_complete_fn=None,
    footer_renderer_mod=None,
) -> str:
    """Run QC footer HTML guard through the modular footer renderer (fail-soft)."""
    out_html = str(final_html_body or "")
    fn = (
        getattr(footer_renderer_mod, "ensure_qc_footer_html_consistency", None)
        if footer_renderer_mod is not None
        else None
    )
    out = _safe_call(
        fn,
        final_html_body=out_html,
        raw_for_render=str(raw_for_render or ""),
        profile_name=str(profile_name or "Standard"),
        gov_mgr=gov_mgr,
        overrides=(overrides if isinstance(overrides, dict) else {}),
        qc_footer_for_profile_fn=qc_footer_for_profile_fn,
        ensure_qc_footer_present_fn=ensure_qc_footer_present_fn,
        enforce_qc_footer_deltas_fn=enforce_qc_footer_deltas_fn,
        ensure_qc_footer_is_last_fn=ensure_qc_footer_is_last_fn,
        qc_probe_is_complete_fn=qc_probe_is_complete_fn,
    )
    if out is None:
        return out_html
    return str(out)


def finalize_qc_footer_html_stage(
    *,
    final_html_body: str,
    raw_for_render: str,
    profile_name: str,
    gov_mgr,
    overrides: dict | None,
    qc_footer_for_profile_fn,
    ensure_qc_footer_present_fn,
    enforce_qc_footer_deltas_fn,
    ensure_qc_footer_is_last_fn,
    qc_probe_is_complete_fn=None,
    lang: str = "de",
    footer_renderer_mod=None,
    annotate_qc_matrix_tooltips_html_fn: Callable | None = None,
) -> str:
    """Run QC footer guard + tooltip annotation as one deterministic HTML stage."""
    out_html = str(final_html_body or "")
    guarded = ensure_qc_footer_html_consistency_html_stage(
        final_html_body=out_html,
        raw_for_render=raw_for_render,
        profile_name=profile_name,
        gov_mgr=gov_mgr,
        overrides=overrides,
        qc_footer_for_profile_fn=qc_footer_for_profile_fn,
        ensure_qc_footer_present_fn=ensure_qc_footer_present_fn,
        enforce_qc_footer_deltas_fn=enforce_qc_footer_deltas_fn,
        ensure_qc_footer_is_last_fn=ensure_qc_footer_is_last_fn,
        qc_probe_is_complete_fn=qc_probe_is_complete_fn,
        footer_renderer_mod=footer_renderer_mod,
    )
    out = _safe_call(
        annotate_qc_matrix_tooltips_html_fn,
        str(guarded or ""),
        lang=str(lang or "de"),
    )
    if out is None:
        return str(guarded or "")
    return str(out)
