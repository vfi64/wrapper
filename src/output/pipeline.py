from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Mapping

try:
    from output.response_model import OutputResponseModel  # type: ignore
except Exception:
    try:
        from response_model import OutputResponseModel  # type: ignore
    except Exception:
        _rm_path = Path(__file__).resolve().parent / "response_model.py"
        _rm_spec = importlib.util.spec_from_file_location("output_response_model", _rm_path)  # type: ignore[attr-defined]
        if _rm_spec is None or _rm_spec.loader is None:
            raise
        _rm_mod = importlib.util.module_from_spec(_rm_spec)  # type: ignore[attr-defined]
        _rm_spec.loader.exec_module(_rm_mod)  # type: ignore[attr-defined]
        OutputResponseModel = _rm_mod.OutputResponseModel  # type: ignore

try:
    from output.renderers import verification_route_policy as _verification_route_policy  # type: ignore
except Exception:
    try:
        from renderers import verification_route_policy as _verification_route_policy  # type: ignore
    except Exception:
        _vr_path = Path(__file__).resolve().parent / "renderers" / "verification_route_policy.py"
        _vr_spec = importlib.util.spec_from_file_location("output_verification_route_policy", _vr_path)  # type: ignore[attr-defined]
        if _vr_spec is not None and _vr_spec.loader is not None:
            _vr_mod = importlib.util.module_from_spec(_vr_spec)  # type: ignore[attr-defined]
            _vr_spec.loader.exec_module(_vr_mod)  # type: ignore[attr-defined]
            _verification_route_policy = _vr_mod  # type: ignore
        else:
            _verification_route_policy = None  # type: ignore


def _safe_apply_html(
    html_body: str,
    fn: Callable | None,
    *,
    kwargs: dict | None = None,
) -> str:
    if not callable(fn):
        return str(html_body or "")
    try:
        if kwargs:
            return str(fn(str(html_body or ""), **kwargs) or "")
        return str(fn(str(html_body or "")) or "")
    except Exception:
        return str(html_body or "")


def _safe_apply_text(
    text: str,
    fn: Callable | None,
) -> str:
    if not callable(fn):
        return str(text or "")
    try:
        out = fn(str(text or ""))
    except Exception:
        return str(text or "")
    if out is None:
        return str(text or "")
    return str(out)


def normalize_self_debunking_postprocess_text(
    text: str,
    *,
    normalize_inline_self_debunking_header_fn: Callable | None = None,
    enforce_self_debunking_contract_fn: Callable | None = None,
    normalize_self_debunking_numbering_text_fn: Callable | None = None,
    dedupe_self_debunking_sections_fn: Callable | None = None,
) -> str:
    """Central, deterministic Self-Debunking text postprocessing seam (fail-soft)."""
    out = str(text or "")
    out = _safe_apply_text(out, normalize_inline_self_debunking_header_fn)
    out = _safe_apply_text(out, enforce_self_debunking_contract_fn)
    out = _safe_apply_text(out, normalize_self_debunking_numbering_text_fn)
    out = _safe_apply_text(out, dedupe_self_debunking_sections_fn)
    return out


def _parse_bool_like(value, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "on", "yes", "y"}:
        return True
    if s in {"0", "false", "off", "no", "n"}:
        return False
    return bool(default)


def resolve_hide_verification_route_lines(
    *,
    config: Mapping | None = None,
    provider: str = "gemini",
) -> bool:
    """Central display-policy decision for Verification Route marker visibility."""
    policy_mod = _verification_route_policy
    if policy_mod is not None and callable(getattr(policy_mod, "resolve_hide_verification_route_lines", None)):
        try:
            return bool(
                policy_mod.resolve_hide_verification_route_lines(  # type: ignore[attr-defined]
                    config=config,
                    provider=provider,
                )
            )
        except Exception:
            pass
    conf = config if isinstance(config, Mapping) else {}
    pid = str(provider or "gemini").strip().lower() or "gemini"
    provs = conf.get("providers") if isinstance(conf, Mapping) else {}
    pconf = provs.get(pid) if isinstance(provs, Mapping) else {}
    raw = None
    if isinstance(pconf, Mapping):
        raw = pconf.get("hide_verification_route_lines")
    if raw is None and isinstance(conf, Mapping):
        raw = conf.get("hide_verification_route_lines")
    return _parse_bool_like(raw, default=False)


def apply_verification_route_display_policy(
    text: str,
    *,
    config: Mapping | None = None,
    provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
) -> str:
    """Apply central Verification Route display-policy to render text (fail-soft)."""
    policy_mod = _verification_route_policy
    if policy_mod is not None and callable(getattr(policy_mod, "apply_verification_route_display_policy", None)):
        try:
            return str(
                policy_mod.apply_verification_route_display_policy(  # type: ignore[attr-defined]
                    text,
                    config=config,
                    provider=provider,
                    strip_verification_route_display_lines_fn=strip_verification_route_display_lines_fn,
                )
                or ""
            )
        except Exception:
            pass
    out = str(text or "")
    if not out:
        return out
    if not resolve_hide_verification_route_lines(config=config, provider=provider):
        return out
    if not callable(strip_verification_route_display_lines_fn):
        return out
    try:
        stripped = strip_verification_route_display_lines_fn(out)
    except Exception:
        return out
    if stripped is None:
        return out
    return str(stripped)


def normalize_post_render_html(
    model: OutputResponseModel,
    *,
    ensure_self_debunking_box_html_fn: Callable | None = None,
    sanitize_self_debunking_markdown_in_html_fn: Callable | None = None,
    normalize_hash_subheadings_in_html_fn: Callable | None = None,
    strip_internal_scaffolding_status_html_fn: Callable | None = None,
    html_number_self_debunking_fn: Callable | None = None,
    strip_color_markers_for_color_off_html_fn: Callable | None = None,
) -> OutputResponseModel:
    """Deterministic post-render normalization pipeline (fail-soft)."""
    out = str(getattr(model, "html_body", "") or "")
    lang = str(getattr(model, "answer_lang", "de") or "de")

    # Preserve existing wrapper order for deterministic output compatibility.
    out = _safe_apply_html(out, ensure_self_debunking_box_html_fn, kwargs={"lang": lang})
    out = _safe_apply_html(out, sanitize_self_debunking_markdown_in_html_fn)
    out = _safe_apply_html(out, normalize_hash_subheadings_in_html_fn)
    out = _safe_apply_html(out, strip_internal_scaffolding_status_html_fn)
    out = _safe_apply_html(out, html_number_self_debunking_fn, kwargs={"lang": lang})
    out = _safe_apply_html(out, sanitize_self_debunking_markdown_in_html_fn)
    out = _safe_apply_html(out, ensure_self_debunking_box_html_fn, kwargs={"lang": lang})

    if str(getattr(model, "color", "off") or "off").strip().lower() != "on":
        out = _safe_apply_html(out, strip_color_markers_for_color_off_html_fn)

    return model.with_html_body(out)


def post_render_normalization(
    html_body: str,
    *,
    answer_lang: str = "de",
    color: str = "off",
    ensure_self_debunking_box_html_fn: Callable | None = None,
    sanitize_self_debunking_markdown_in_html_fn: Callable | None = None,
    normalize_hash_subheadings_in_html_fn: Callable | None = None,
    strip_internal_scaffolding_status_html_fn: Callable | None = None,
    html_number_self_debunking_fn: Callable | None = None,
    strip_color_markers_for_color_off_html_fn: Callable | None = None,
) -> str:
    model = OutputResponseModel.from_values(
        html_body=str(html_body or ""),
        answer_lang=answer_lang,
        color=color,
    )
    out = normalize_post_render_html(
        model,
        ensure_self_debunking_box_html_fn=ensure_self_debunking_box_html_fn,
        sanitize_self_debunking_markdown_in_html_fn=sanitize_self_debunking_markdown_in_html_fn,
        normalize_hash_subheadings_in_html_fn=normalize_hash_subheadings_in_html_fn,
        strip_internal_scaffolding_status_html_fn=strip_internal_scaffolding_status_html_fn,
        html_number_self_debunking_fn=html_number_self_debunking_fn,
        strip_color_markers_for_color_off_html_fn=strip_color_markers_for_color_off_html_fn,
    )
    return str(getattr(out, "html_body", "") or "")
