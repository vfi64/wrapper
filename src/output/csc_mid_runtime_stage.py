from __future__ import annotations

import html
import re
from typing import Callable


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


STRICT_GATE_RUNTIME_SCOPE_HOOK_MAP = {
    "evaluate_strict_enforcement_fn": "evaluate_strict_enforcement",
    "sanitize_html_fn": "sanitize_html",
    "strip_basic_html_for_enforcement_fn": "_strip_basic_html_for_enforcement",
    "ensure_qc_footer_present_fn": "ensure_qc_footer_present",
    "enforce_qc_footer_deltas_fn": "enforce_qc_footer_deltas",
    "ensure_qc_footer_is_last_fn": "ensure_qc_footer_is_last",
    "normalize_evidence_tags_fn": "normalize_evidence_tags",
    "strip_empty_citation_placeholders_fn": "strip_empty_citation_placeholders",
    "strip_sci_menu_from_answer_fn": "strip_sci_menu_from_answer",
    "strip_sci_trace_line_when_inactive_fn": "strip_sci_trace_line_when_inactive",
    "apply_self_debunking_text_postprocess_fn": "apply_self_debunking_text_postprocess_seam",
    "normalize_sci_trace_numbering_fn": "normalize_sci_trace_numbering",
    "qc_override_runtime_violations_fn": "qc_override_runtime_violations",
    "strip_internal_scaffolding_status_lines_fn": "strip_internal_scaffolding_status_lines",
    "strip_exact_status_header_line_fn": "strip_exact_status_header_line",
    "strip_verification_route_display_lines_fn": "strip_verification_route_display_lines",
    "unwrap_accidental_full_text_codefence_fn": "unwrap_accidental_full_text_codefence",
    "strip_pathological_repetition_display_noise_fn": "strip_pathological_repetition_display_noise",
}

STRICT_GATE_RUNTIME_APP_HOOK_ATTR_MAP = {
    "append_system_message_fn": "_append_system_message",
    "get_enforcement_settings_fn": "_get_enforcement_settings",
    "render_sci_trace_as_html_runtime_fn": "_render_sci_trace_as_html_runtime",
    "hide_verification_route_lines_in_chat_fn": "_hide_verification_route_lines_in_chat",
}

STRICT_GATE_RUNTIME_GOV_HOOK_ATTR_MAP = {
    "normalize_qc_overrides_fn": "normalize_qc_overrides",
    "get_effective_qc_corridor_fn": "get_effective_qc_corridor",
}

STRICT_GATE_RUNTIME_CHAIN_HOOK_KEYS = (
    "append_system_message_fn",
    "get_enforcement_settings_fn",
    "evaluate_strict_enforcement_fn",
    "sanitize_html_fn",
    "normalize_qc_overrides_fn",
    "get_effective_qc_corridor_fn",
    "strip_basic_html_for_enforcement_fn",
    "ensure_qc_footer_present_fn",
    "enforce_qc_footer_deltas_fn",
    "ensure_qc_footer_is_last_fn",
    "normalize_evidence_tags_fn",
    "strip_empty_citation_placeholders_fn",
    "strip_sci_menu_from_answer_fn",
    "strip_sci_trace_line_when_inactive_fn",
    "apply_self_debunking_text_postprocess_fn",
    "normalize_sci_trace_numbering_fn",
    "render_sci_trace_as_html_runtime_fn",
    "qc_override_runtime_violations_fn",
    "strip_internal_scaffolding_status_lines_fn",
    "strip_exact_status_header_line_fn",
    "apply_verification_route_display_policy_fn",
    "strip_verification_route_display_lines_fn",
    "hide_verification_route_lines_in_chat_fn",
    "unwrap_accidental_full_text_codefence_fn",
    "strip_pathological_repetition_display_noise_fn",
)

STRICT_GATE_RUNTIME_ULTIMATE_HOOK_KEYS = (
    "append_system_message_fn",
    "get_enforcement_settings_fn",
    "evaluate_strict_enforcement_fn",
    "sanitize_html_fn",
    "apply_verification_route_display_policy_fn",
    "strip_verification_route_display_lines_fn",
    "hide_verification_route_lines_in_chat_fn",
    "unwrap_accidental_full_text_codefence_fn",
    "strip_pathological_repetition_display_noise_fn",
)


def _resolve_hooks_from_scope(
    *,
    hook_scope: dict | None,
    hook_map: dict[str, str],
):
    hooks = {}
    if not isinstance(hook_scope, dict):
        return hooks
    for hook_key, symbol_name in hook_map.items():
        try:
            cand = hook_scope.get(symbol_name)
        except Exception:
            cand = None
        if callable(cand):
            hooks[hook_key] = cand
    return hooks


def _resolve_attr_hooks(obj, hook_attr_map: dict[str, str]):
    hooks = {}
    if obj is None:
        return hooks
    for hook_key, attr_name in hook_attr_map.items():
        cand = getattr(obj, attr_name, None)
        if callable(cand):
            hooks[hook_key] = cand
    return hooks


def resolve_pre_render_policy_strict_gate_hook_overrides(
    *,
    hook_scope: dict | None = None,
    hook_overrides: dict | None = None,
):
    """Resolve external strict-gate hooks from a symbol scope and explicit overrides."""
    hooks = _resolve_hooks_from_scope(
        hook_scope=(hook_scope if isinstance(hook_scope, dict) else None),
        hook_map=STRICT_GATE_RUNTIME_SCOPE_HOOK_MAP,
    )
    if isinstance(hook_overrides, dict):
        try:
            hooks.update(hook_overrides)
        except Exception:
            hooks = {**hooks, **dict(hook_overrides)}
    return hooks


def _apply_verification_route_display_policy_runtime(
    text_in: str,
    *,
    apply_verification_route_display_policy_fn: Callable | None = None,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
    hide_verification_route_lines_in_chat_fn: Callable | None = None,
) -> str:
    out = str(text_in or "")
    tmp = _safe_call(apply_verification_route_display_policy_fn, out)
    if tmp is not None:
        return str(tmp)

    try:
        if output_pipeline_mod is not None and hasattr(output_pipeline_mod, "apply_verification_route_display_policy"):
            cfg = verification_route_config if isinstance(verification_route_config, dict) else {}
            provider = str(verification_route_provider or "gemini").strip().lower()
            tmp = _safe_call(
                getattr(output_pipeline_mod, "apply_verification_route_display_policy", None),
                out or "",
                config=cfg,
                provider=provider,
                strip_verification_route_display_lines_fn=strip_verification_route_display_lines_fn,
            )
            if tmp is not None:
                return str(tmp)
    except Exception:
        pass

    try:
        hide = _safe_call(hide_verification_route_lines_in_chat_fn)
        if bool(hide):
            tmp = _safe_call(strip_verification_route_display_lines_fn, out or "")
            if tmp is not None:
                return str(tmp)
    except Exception:
        pass

    return out


def build_csc_refiner_meta_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    overlay_name: str,
    answer_lang: str = "de",
    csc_warning_text: str = "",
    refiner_obj=None,
    csc_score_tooltip_text_fn: Callable | None = None,
    csc_thresholds_tooltip_text_fn: Callable | None = None,
):
    """Build CSC refiner metadata deterministically (fail-soft)."""
    csc_meta = None
    prof = str(profile_name or "Standard")
    overlay = str(overlay_name or "")
    mult = 2 if overlay == "Explore" else 1

    txt = str(raw_response or "")
    txt_l = txt.lower()
    uncertainty_u4 = bool(re.search(r"\bU[4-6]\b", txt))
    web_check = bool(re.search(r"\bweb\s*[- ]\s*check\b", txt_l))
    strong_claim = any(x in txt_l for x in ["immer", "niemals", "definitiv", "guarantee", "prove"])

    decide_fn = getattr(refiner_obj, "decide", None) if refiner_obj is not None else None
    if callable(decide_fn):
        dec = _safe_call(
            decide_fn,
            comm_active=True,
            active_profile=prof,
            input_raw=str(user_raw or ""),
            uncertainty_U4_active=uncertainty_u4,
            web_check_hook_active=web_check,
            strong_claim_detected=strong_claim,
            neutrality_delta_negative=False,
            threshold_multiplier=mult,
        )
        if dec is not None and bool(getattr(dec, "apply", False)):
            msg = str(csc_warning_text or "")
            thr_fs = int(getattr(refiner_obj, "_refine_params", {}).get("threshold_f_score", 8) or 8)
            thr_tok = int(getattr(refiner_obj, "_refine_params", {}).get("min_token_count", 80) or 80)
            gov_min_tok = int(
                getattr(refiner_obj, "_gov_params", {}).get("min_token_count_governance", 40) or 40
            )
            if mult != 1:
                thr_fs *= mult
                thr_tok *= mult
                gov_min_tok *= mult
            score_tip = _safe_call(
                csc_score_tooltip_text_fn,
                lang=str(answer_lang or "de"),
                f_score=int(getattr(dec, "f_score", 0)),
                token_count=int(getattr(dec, "token_count", 0)),
            )
            thr_tip = _safe_call(
                csc_thresholds_tooltip_text_fn,
                lang=str(answer_lang or "de"),
                thr_fs=thr_fs,
                thr_tok=thr_tok,
                gov_min_tok=gov_min_tok,
                mult=mult,
            )
            csc_meta = {
                "applied": True,
                "message": msg,
                "trigger": str(getattr(dec, "trigger_source", "")),
                "mode": str(getattr(dec, "mode", "")),
                "governance_triggered": bool(getattr(dec, "governance_triggered", False)),
                "token_count": int(getattr(dec, "token_count", 0)),
                "f_score": int(getattr(dec, "f_score", 0)),
                "score_tooltip": str(score_tip or ""),
                "thresholds_tooltip": str(thr_tip or ""),
                "overlay": overlay,
                "profile": prof,
                "threshold_multiplier": int(mult or 1),
                "threshold_f_score": int(thr_fs),
                "threshold_token_count": int(thr_tok),
                "min_token_count": int(thr_tok),
                "min_token_count_governance": int(gov_min_tok),
                "schema_version": "1.0",
            }

    return {
        "csc_meta": csc_meta,
        "threshold_multiplier": int(mult or 1),
    }


def build_alerts_and_header_stage(
    *,
    raw_response: str,
    profile_name: str,
    overlay_name: str,
    csc_meta,
    refiner_obj,
    gov_mgr,
    runtime_state,
    check_verification_route_gate_fn: Callable | None = None,
    normalize_qc_overrides_fn: Callable | None = None,
    get_effective_qc_corridor_fn: Callable | None = None,
    strip_basic_html_for_enforcement_fn: Callable | None = None,
    ensure_qc_footer_present_fn: Callable | None = None,
    enforce_qc_footer_deltas_fn: Callable | None = None,
    ensure_qc_footer_is_last_fn: Callable | None = None,
    build_active_profile_header_line_fn: Callable | None = None,
):
    """Build control-layer alerts and status header (fail-soft)."""
    prof = str(profile_name or "Standard")
    overlay = str(overlay_name or "")
    raw = str(raw_response or "")
    alerts = []
    csc_visible_marker = ""

    try:
        if csc_meta and csc_meta.get("applied"):
            cl = (getattr(gov_mgr, "data", {}) or {}).get("control_layer", {}) or {}
            bridge = (cl.get("components", {}) or {}).get("csc_trigger_bridge", {}) or {}
            constraints = bridge.get("constraints", {}) or {}
            tm = constraints.get("transparency_marker", {}) or {}
            marker = str(tm.get("marker", getattr(refiner_obj, "marker", "") or "") or "").strip()
            marker_enabled = bool(tm.get("enabled", True))
            marker_visibility = str(tm.get("visibility", "") or "").strip().lower()
            if marker_enabled and marker and marker_visibility == "always_visible_if_applied":
                csc_visible_marker = marker
    except Exception:
        pass

    try:
        vr_msg = _safe_call(check_verification_route_gate_fn, raw)
        if vr_msg:
            alerts.append(("Verification Route Gate", vr_msg))
    except Exception:
        pass

    try:
        _prof = str(prof or getattr(runtime_state, "active_profile", "Standard") or "Standard")
        _ovr_raw = getattr(runtime_state, "qc_overrides", {}) or {}
        _ovr = _safe_call(normalize_qc_overrides_fn, _ovr_raw)
        if not isinstance(_ovr, dict):
            _ovr = _ovr_raw if isinstance(_ovr_raw, dict) else {}
        _corr = _safe_call(get_effective_qc_corridor_fn, _prof, _ovr)

        tmp = _safe_call(strip_basic_html_for_enforcement_fn, raw)
        if tmp is not None:
            raw = str(tmp)
        tmp = _safe_call(ensure_qc_footer_present_fn, raw, gov_mgr, _prof, _ovr)
        if tmp is not None:
            raw = str(tmp)
        tmp = _safe_call(enforce_qc_footer_deltas_fn, raw, _corr, _prof)
        if tmp is not None:
            raw = str(tmp)
        tmp = _safe_call(ensure_qc_footer_is_last_fn, raw)
        if tmp is not None:
            raw = str(tmp)

        cur_qc = None
        rep_delta = None
        if hasattr(gov_mgr, "parse_qc_footer"):
            parsed = _safe_call(getattr(gov_mgr, "parse_qc_footer", None), raw)
            if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                cur_qc, rep_delta = parsed[0], parsed[1]
        if cur_qc:
            exp_delta = {}
            if hasattr(gov_mgr, "expected_qc_deltas"):
                exp = _safe_call(getattr(gov_mgr, "expected_qc_deltas", None), _prof, cur_qc, overrides=_ovr)
                if isinstance(exp, dict):
                    exp_delta = exp
            if rep_delta:
                mism = [
                    f"{k}: expected Δ{v}, got Δ{rep_delta[k]}"
                    for k, v in exp_delta.items()
                    if rep_delta.get(k) != v
                ]
                if mism:
                    alerts.append(("QC-Matrix", "Delta mismatch: " + "; ".join(mism)))
            else:
                alerts.append(("QC-Matrix", "QC detected but no deltas found."))
    except Exception:
        pass

    try:
        hits = list(getattr(runtime_state, "cross_version_guard_hits", []) or [])
        active_v = str(
            getattr(runtime_state, "active_ruleset_version", "")
            or str((getattr(gov_mgr, "data", {}) or {}).get("version", "") or "")
        ).strip()
        if hits:
            alerts.append(("Cross-Version Guard", f"Ignored foreign version token(s) in user input (active {active_v})."))
    except Exception:
        pass

    alert_html = ""
    if alerts:
        items = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>" for k, v in alerts)
        alert_html = (
            "<div style='border:1px solid #b00; background:#fff5f5; padding:10px; "
            "border-radius:10px; margin:8px 0;'><b>CONTROL LAYER ALERTS (Python)</b>"
            f"<ul style='margin:6px 0 0 18px; padding:0;'>{items}</ul></div>"
        )

    header = ""
    try:
        sci = str(getattr(runtime_state, "sci_variant", "") or "").strip()
        sci_pending = bool(getattr(runtime_state, "sci_pending", False))
        color = str(getattr(runtime_state, "color", "off") or "off")
        built = _safe_call(
            build_active_profile_header_line_fn,
            profile=str(prof or "Standard"),
            sci_variant=str(sci or ""),
            overlay=str(overlay or "off"),
            control_layer="on",
            qc="on",
            cgi="on",
            color=str(color or "off"),
            sci_pending=bool(sci_pending),
            off_label="off",
            pending_label="PENDING",
            pending_mode="when_pending_and_unset",
            uppercase_sci_non_off=False,
            color_force_off_profiles=(),
        )
        header = str(built or "")
        try:
            if bool(getattr(runtime_state, "dynamic_one_shot_active", False)):
                header += " · Dynamic: one-shot (active)"
        except Exception:
            pass
        try:
            if csc_visible_marker and (csc_visible_marker not in header):
                header += f" · {csc_visible_marker}"
        except Exception:
            pass
    except Exception:
        header = ""

    return {
        "raw_response": str(raw or ""),
        "alert_html": str(alert_html or ""),
        "header": str(header or ""),
    }


def apply_pre_render_policy_strict_gate_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    is_command: bool,
    ctx: dict | None,
    header: str,
    alert_html: str,
    gov_mgr,
    runtime_state,
    validator_obj,
    append_system_message_fn: Callable | None = None,
    get_enforcement_settings_fn: Callable | None = None,
    evaluate_strict_enforcement_fn: Callable | None = None,
    sanitize_html_fn: Callable | None = None,
    normalize_qc_overrides_fn: Callable | None = None,
    get_effective_qc_corridor_fn: Callable | None = None,
    strip_basic_html_for_enforcement_fn: Callable | None = None,
    ensure_qc_footer_present_fn: Callable | None = None,
    enforce_qc_footer_deltas_fn: Callable | None = None,
    ensure_qc_footer_is_last_fn: Callable | None = None,
    normalize_evidence_tags_fn: Callable | None = None,
    strip_empty_citation_placeholders_fn: Callable | None = None,
    strip_sci_menu_from_answer_fn: Callable | None = None,
    strip_sci_trace_line_when_inactive_fn: Callable | None = None,
    apply_self_debunking_text_postprocess_fn: Callable | None = None,
    normalize_sci_trace_numbering_fn: Callable | None = None,
    render_sci_trace_as_html_runtime_fn: Callable | None = None,
    qc_override_runtime_violations_fn: Callable | None = None,
    strip_internal_scaffolding_status_lines_fn: Callable | None = None,
    strip_exact_status_header_line_fn: Callable | None = None,
    apply_verification_route_display_policy_fn: Callable | None = None,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
    hide_verification_route_lines_in_chat_fn: Callable | None = None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    strip_pathological_repetition_display_noise_fn: Callable | None = None,
):
    """Apply pre-render policy transforms and strict-gate evaluation (fail-soft)."""
    prof = str(profile_name or "Standard")
    raw_response_out = str(raw_response or "")
    raw_for_render = raw_response_out
    out_alert_html = str(alert_html or "")

    try:
        _ovr_raw = getattr(runtime_state, "qc_overrides", {}) or {}
        _ovr = _safe_call(normalize_qc_overrides_fn, _ovr_raw)
        if not isinstance(_ovr, dict):
            _ovr = _ovr_raw if isinstance(_ovr_raw, dict) else {}
        corr = _safe_call(get_effective_qc_corridor_fn, prof, _ovr)

        tmp = _safe_call(strip_basic_html_for_enforcement_fn, raw_response_out)
        if tmp is not None:
            raw_response_out = str(tmp)
        tmp = _safe_call(ensure_qc_footer_present_fn, raw_response_out, gov_mgr, prof, _ovr)
        if tmp is not None:
            raw_response_out = str(tmp)
        tmp = _safe_call(enforce_qc_footer_deltas_fn, raw_response_out, corr, prof)
        raw_for_render = str(tmp) if tmp is not None else str(raw_response_out or "")
        tmp = _safe_call(ensure_qc_footer_is_last_fn, raw_for_render)
        if tmp is not None:
            raw_for_render = str(tmp)
    except Exception:
        raw_for_render = raw_response_out

    try:
        cur_qc = None
        if hasattr(gov_mgr, "parse_qc_footer"):
            parsed = _safe_call(getattr(gov_mgr, "parse_qc_footer", None), raw_for_render)
            if isinstance(parsed, (tuple, list)) and len(parsed) >= 1:
                cur_qc = parsed[0]
        if cur_qc:
            exp_delta = {}
            if hasattr(gov_mgr, "expected_qc_deltas"):
                exp = _safe_call(
                    getattr(gov_mgr, "expected_qc_deltas", None),
                    prof,
                    cur_qc,
                    overrides=getattr(runtime_state, "qc_overrides", {}),
                )
                if isinstance(exp, dict):
                    exp_delta = exp
            try:
                setattr(runtime_state, "last_qc", dict(cur_qc))
                setattr(runtime_state, "last_qc_deltas", dict(exp_delta or {}))
            except Exception:
                pass
    except Exception:
        pass

    tmp = _safe_call(normalize_evidence_tags_fn, raw_for_render)
    if tmp is not None:
        raw_for_render = str(tmp)
    tmp = _safe_call(strip_empty_citation_placeholders_fn, raw_for_render)
    if tmp is not None:
        raw_for_render = str(tmp)

    try:
        sci_pending = bool((ctx or {}).get("sci_pending"))
        if (not bool(is_command)) and (not sci_pending):
            tmp = _safe_call(strip_sci_menu_from_answer_fn, raw_for_render)
            if tmp is not None:
                raw_for_render = str(tmp)
    except Exception:
        pass

    tmp = _safe_call(
        strip_sci_trace_line_when_inactive_fn,
        raw_for_render,
        sci_active=bool(getattr(runtime_state, "sci_active", False)),
        sci_variant=(getattr(runtime_state, "sci_variant", "") or ""),
        sci_pending=bool(getattr(runtime_state, "sci_pending", False)),
    )
    if tmp is not None:
        raw_for_render = str(tmp)

    _sd_lang = getattr(runtime_state, "answer_language", "de")
    tmp = _safe_call(
        apply_self_debunking_text_postprocess_fn,
        raw_for_render,
        gov_mgr=gov_mgr,
        profile_name=str(prof or "Standard"),
        is_command=bool(is_command),
        lang=str(_sd_lang or "de"),
    )
    if tmp is not None:
        raw_for_render = str(tmp)

    tmp = _safe_call(normalize_sci_trace_numbering_fn, raw_for_render, gov_mgr)
    if tmp is not None:
        raw_for_render = str(tmp)
    tmp = _safe_call(render_sci_trace_as_html_runtime_fn, raw_for_render)
    if tmp is not None:
        raw_for_render = str(tmp)

    override_vios = []
    try:
        _ovr_runtime = getattr(runtime_state, "qc_overrides", {}) or {}
        _vios = _safe_call(qc_override_runtime_violations_fn, raw_for_render, _ovr_runtime)
        if isinstance(_vios, (list, tuple)):
            override_vios = [str(x) for x in list(_vios or []) if str(x)]
    except Exception:
        override_vios = []

    tmp = _safe_call(strip_internal_scaffolding_status_lines_fn, raw_for_render or "")
    if tmp is not None:
        raw_for_render = str(tmp)
    tmp = _safe_call(strip_exact_status_header_line_fn, raw_for_render or "", str(header or ""))
    if tmp is not None:
        raw_for_render = str(tmp)

    if header:
        raw_for_render = str(header or "") + "\n\n" + str(raw_for_render or "")

    ens = _safe_call(get_enforcement_settings_fn)
    ens = ens if isinstance(ens, dict) else {}
    strict_eval = _safe_call(
        evaluate_strict_enforcement_fn,
        raw_for_render=raw_for_render,
        user_raw=str(user_raw or ""),
        profile_name=str(prof or "Standard"),
        override_violations=override_vios,
        settings=ens,
        validator_obj=validator_obj,
        runtime_state=runtime_state,
        append_system_message_fn=append_system_message_fn,
    )
    if isinstance(strict_eval, dict) and bool(strict_eval.get("blocked")):
        blocked_html = str(strict_eval.get("blocked_html") or "")
        strict_meta = strict_eval.get("meta")
        blocked_payload_html = blocked_html
        tmp = _safe_call(sanitize_html_fn, blocked_html)
        if tmp is not None:
            blocked_payload_html = str(tmp)
        return {
            "blocked": True,
            "blocked_response": {"html": blocked_payload_html, "text": "", "csc": None},
            "strict_meta": strict_meta,
            "alert_html": str(out_alert_html or ""),
            "raw_for_render": str(raw_for_render or ""),
            "raw_response": str(raw_response_out or ""),
        }

    strict_banner_html = ""
    if isinstance(strict_eval, dict):
        strict_banner_html = str(strict_eval.get("strict_banner_html") or "")
    if strict_banner_html:
        out_alert_html = strict_banner_html + out_alert_html

    raw_for_render = _apply_verification_route_display_policy_runtime(
        raw_for_render or "",
        apply_verification_route_display_policy_fn=apply_verification_route_display_policy_fn,
        output_pipeline_mod=output_pipeline_mod,
        verification_route_config=verification_route_config,
        verification_route_provider=str(verification_route_provider or "gemini"),
        strip_verification_route_display_lines_fn=strip_verification_route_display_lines_fn,
        hide_verification_route_lines_in_chat_fn=hide_verification_route_lines_in_chat_fn,
    )

    tmp = _safe_call(unwrap_accidental_full_text_codefence_fn, raw_for_render or "")
    if tmp is not None:
        raw_for_render = str(tmp)
    tmp = _safe_call(
        strip_pathological_repetition_display_noise_fn,
        raw_for_render or "",
        lang=getattr(runtime_state, "answer_language", "de"),
    )
    if tmp is not None:
        raw_for_render = str(tmp)

    return {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": str(out_alert_html or ""),
        "raw_for_render": str(raw_for_render or ""),
        "raw_response": str(raw_response_out or ""),
    }


def apply_pre_render_policy_strict_gate_fallback_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    header: str,
    alert_html: str,
    runtime_state,
    validator_obj,
    append_system_message_fn: Callable | None = None,
    get_enforcement_settings_fn: Callable | None = None,
    evaluate_strict_enforcement_fn: Callable | None = None,
    sanitize_html_fn: Callable | None = None,
    apply_verification_route_display_policy_fn: Callable | None = None,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
    hide_verification_route_lines_in_chat_fn: Callable | None = None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    strip_pathological_repetition_display_noise_fn: Callable | None = None,
):
    """Fallback stage for strict gate and pre-render policy (fail-soft)."""
    prof = str(profile_name or "Standard")
    raw_response_out = str(raw_response or "")
    out_alert_html = str(alert_html or "")

    raw_for_render = str(raw_response_out or "")
    if header:
        raw_for_render = str(header or "") + "\n\n" + raw_for_render

    ens = _safe_call(get_enforcement_settings_fn)
    ens = ens if isinstance(ens, dict) else {}
    strict_eval = _safe_call(
        evaluate_strict_enforcement_fn,
        raw_for_render=raw_for_render,
        user_raw=str(user_raw or ""),
        profile_name=str(prof or "Standard"),
        override_violations=[],
        settings=ens,
        validator_obj=validator_obj,
        runtime_state=runtime_state,
        append_system_message_fn=append_system_message_fn,
    )
    if isinstance(strict_eval, dict) and bool(strict_eval.get("blocked")):
        blocked_html = str(strict_eval.get("blocked_html") or "")
        strict_meta = strict_eval.get("meta")
        blocked_payload_html = blocked_html
        tmp = _safe_call(sanitize_html_fn, blocked_html)
        if tmp is not None:
            blocked_payload_html = str(tmp)
        return {
            "blocked": True,
            "blocked_response": {"html": blocked_payload_html, "text": "", "csc": None},
            "strict_meta": strict_meta,
            "alert_html": str(out_alert_html or ""),
            "raw_for_render": str(raw_for_render or ""),
            "raw_response": str(raw_response_out or ""),
        }

    strict_banner_html = ""
    if isinstance(strict_eval, dict):
        strict_banner_html = str(strict_eval.get("strict_banner_html") or "")
    if strict_banner_html:
        out_alert_html = strict_banner_html + out_alert_html

    raw_for_render = _apply_verification_route_display_policy_runtime(
        raw_for_render or "",
        apply_verification_route_display_policy_fn=apply_verification_route_display_policy_fn,
        output_pipeline_mod=output_pipeline_mod,
        verification_route_config=verification_route_config,
        verification_route_provider=str(verification_route_provider or "gemini"),
        strip_verification_route_display_lines_fn=strip_verification_route_display_lines_fn,
        hide_verification_route_lines_in_chat_fn=hide_verification_route_lines_in_chat_fn,
    )

    tmp = _safe_call(unwrap_accidental_full_text_codefence_fn, raw_for_render or "")
    if tmp is not None:
        raw_for_render = str(tmp)
    tmp = _safe_call(
        strip_pathological_repetition_display_noise_fn,
        raw_for_render or "",
        lang=getattr(runtime_state, "answer_language", "de"),
    )
    if tmp is not None:
        raw_for_render = str(tmp)

    return {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": str(out_alert_html or ""),
        "raw_for_render": str(raw_for_render or ""),
        "raw_response": str(raw_response_out or ""),
    }


def apply_pre_render_policy_strict_gate_ultimate_fallback_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    header: str,
    alert_html: str,
    runtime_state,
    validator_obj,
    append_system_message_fn: Callable | None = None,
    get_enforcement_settings_fn: Callable | None = None,
    evaluate_strict_enforcement_fn: Callable | None = None,
    sanitize_html_fn: Callable | None = None,
    apply_verification_route_display_policy_fn: Callable | None = None,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
    hide_verification_route_lines_in_chat_fn: Callable | None = None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    strip_pathological_repetition_display_noise_fn: Callable | None = None,
):
    """Ultimate runtime-stage fallback for strict gate / pre-render policy."""
    return apply_pre_render_policy_strict_gate_fallback_stage(
        raw_response=str(raw_response or ""),
        user_raw=str(user_raw or ""),
        profile_name=str(profile_name or "Standard"),
        header=str(header or ""),
        alert_html=str(alert_html or ""),
        runtime_state=runtime_state,
        validator_obj=validator_obj,
        append_system_message_fn=append_system_message_fn,
        get_enforcement_settings_fn=get_enforcement_settings_fn,
        evaluate_strict_enforcement_fn=evaluate_strict_enforcement_fn,
        sanitize_html_fn=sanitize_html_fn,
        apply_verification_route_display_policy_fn=apply_verification_route_display_policy_fn,
        output_pipeline_mod=output_pipeline_mod,
        verification_route_config=verification_route_config,
        verification_route_provider=str(verification_route_provider or "gemini"),
        strip_verification_route_display_lines_fn=strip_verification_route_display_lines_fn,
        hide_verification_route_lines_in_chat_fn=hide_verification_route_lines_in_chat_fn,
        unwrap_accidental_full_text_codefence_fn=unwrap_accidental_full_text_codefence_fn,
        strip_pathological_repetition_display_noise_fn=strip_pathological_repetition_display_noise_fn,
    )


def _normalize_pre_render_policy_strict_gate_stage_out(
    out,
    *,
    default_alert_html: str,
    default_raw_response: str,
):
    if not isinstance(out, dict):
        return None
    return {
        "blocked": bool(out.get("blocked", False)),
        "blocked_response": out.get("blocked_response"),
        "strict_meta": out.get("strict_meta"),
        "alert_html": str(out.get("alert_html") or default_alert_html),
        "raw_for_render": str(out.get("raw_for_render") or ""),
        "raw_response": str(out.get("raw_response") or default_raw_response),
    }


def normalize_pre_render_policy_strict_gate_stage_out(
    out,
    *,
    default_alert_html: str,
    default_raw_response: str,
):
    """Public normalizer for app/runtime bridge callers."""
    return _normalize_pre_render_policy_strict_gate_stage_out(
        out,
        default_alert_html=str(default_alert_html or ""),
        default_raw_response=str(default_raw_response or ""),
    )


def build_pre_render_policy_strict_gate_runtime_passthrough_out(
    *,
    raw_response: str,
    header: str,
    alert_html: str,
):
    """Deterministic emergency payload if runtime dispatch cannot produce stage output."""
    raw_response_out = str(raw_response or "")
    raw_for_render = str(raw_response_out or "")
    header_out = str(header or "")
    if header_out:
        raw_for_render = header_out + "\n\n" + raw_for_render
    return {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": str(alert_html or ""),
        "raw_for_render": str(raw_for_render or ""),
        "raw_response": str(raw_response_out or ""),
    }


def build_pre_render_policy_strict_gate_runtime_bundle(
    *,
    gov_mgr,
    runtime_state,
    validator_obj,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str = "gemini",
    app_obj=None,
    hook_overrides: dict | None = None,
    hook_scope: dict | None = None,
):
    """Build runtime bundle for strict-gate chain dispatch.

    The app bridge can pass only externally-owned hooks in ``hook_overrides`` while
    app-/gov-bound hooks are resolved here to keep monolith orchestration thin.
    """
    hooks = resolve_pre_render_policy_strict_gate_hook_overrides(
        hook_scope=(hook_scope if isinstance(hook_scope, dict) else None),
        hook_overrides=hook_overrides,
    )
    app_hooks = _resolve_attr_hooks(app_obj, STRICT_GATE_RUNTIME_APP_HOOK_ATTR_MAP)
    gov_hooks = _resolve_attr_hooks(gov_mgr, STRICT_GATE_RUNTIME_GOV_HOOK_ATTR_MAP)
    for hook_key, fn in {**app_hooks, **gov_hooks}.items():
        hooks.setdefault(hook_key, fn)

    return {
        "gov_mgr": gov_mgr,
        "runtime_state": runtime_state,
        "validator_obj": validator_obj,
        "output_pipeline_mod": output_pipeline_mod,
        "verification_route_config": (
            verification_route_config if isinstance(verification_route_config, dict) else {}
        ),
        "verification_route_provider": str(verification_route_provider or "gemini"),
        "hooks": hooks,
    }


def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    is_command: bool,
    ctx: dict | None,
    header: str,
    alert_html: str,
    runtime_bundle: dict | None = None,
):
    """Single runtime dispatch entry for strict-gate pre-render policy handling."""
    raw_response_out = str(raw_response or "")
    out_alert_html = str(alert_html or "")
    header_out = str(header or "")

    out = apply_pre_render_policy_strict_gate_runtime_chain_stage(
        raw_response=raw_response_out,
        user_raw=str(user_raw or ""),
        profile_name=str(profile_name or "Standard"),
        is_command=bool(is_command),
        ctx=(ctx if isinstance(ctx, dict) else {}),
        header=header_out,
        alert_html=out_alert_html,
        runtime_bundle=runtime_bundle,
    )
    out_norm = _normalize_pre_render_policy_strict_gate_stage_out(
        out,
        default_alert_html=out_alert_html,
        default_raw_response=raw_response_out,
    )
    if out_norm is not None:
        return out_norm

    out = apply_pre_render_policy_strict_gate_runtime_ultimate_fallback_from_bundle_stage(
        raw_response=raw_response_out,
        user_raw=str(user_raw or ""),
        profile_name=str(profile_name or "Standard"),
        header=header_out,
        alert_html=out_alert_html,
        runtime_bundle=runtime_bundle,
    )
    out_norm = _normalize_pre_render_policy_strict_gate_stage_out(
        out,
        default_alert_html=out_alert_html,
        default_raw_response=raw_response_out,
    )
    if out_norm is not None:
        return out_norm

    return build_pre_render_policy_strict_gate_runtime_passthrough_out(
        raw_response=raw_response_out,
        header=header_out,
        alert_html=out_alert_html,
    )


def _resolve_runtime_chain_callable(
    *,
    explicit,
    hook_bundle: dict | None,
    key: str,
):
    if callable(explicit):
        return explicit
    if isinstance(hook_bundle, dict):
        cand = hook_bundle.get(key)
        if callable(cand):
            return cand
    return explicit


def _resolve_runtime_chain_hook_set(
    *,
    hook_bundle: dict | None,
    explicit_hooks: dict | None,
    keys: tuple[str, ...],
):
    resolved = {}
    explicit_map = explicit_hooks if isinstance(explicit_hooks, dict) else {}
    for key in keys:
        resolved[key] = _resolve_runtime_chain_callable(
            explicit=explicit_map.get(key),
            hook_bundle=hook_bundle,
            key=key,
        )
    return resolved


def apply_pre_render_policy_strict_gate_runtime_ultimate_fallback_from_bundle_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    header: str,
    alert_html: str,
    runtime_bundle: dict | None = None,
):
    """Dispatch ultimate fallback stage by resolving hooks/objects from runtime bundle."""
    bundle = runtime_bundle if isinstance(runtime_bundle, dict) else {}
    hook_bundle = bundle.get("hooks", {}) if isinstance(bundle.get("hooks", {}), dict) else {}
    resolved_hooks = _resolve_runtime_chain_hook_set(
        hook_bundle=hook_bundle,
        explicit_hooks=None,
        keys=STRICT_GATE_RUNTIME_ULTIMATE_HOOK_KEYS,
    )

    return apply_pre_render_policy_strict_gate_ultimate_fallback_stage(
        raw_response=str(raw_response or ""),
        user_raw=str(user_raw or ""),
        profile_name=str(profile_name or "Standard"),
        header=str(header or ""),
        alert_html=str(alert_html or ""),
        runtime_state=bundle.get("runtime_state"),
        validator_obj=bundle.get("validator_obj"),
        append_system_message_fn=resolved_hooks.get("append_system_message_fn"),
        get_enforcement_settings_fn=resolved_hooks.get("get_enforcement_settings_fn"),
        evaluate_strict_enforcement_fn=resolved_hooks.get("evaluate_strict_enforcement_fn"),
        sanitize_html_fn=resolved_hooks.get("sanitize_html_fn"),
        apply_verification_route_display_policy_fn=resolved_hooks.get("apply_verification_route_display_policy_fn"),
        output_pipeline_mod=bundle.get("output_pipeline_mod"),
        verification_route_config=bundle.get("verification_route_config"),
        verification_route_provider=str(bundle.get("verification_route_provider") or "gemini"),
        strip_verification_route_display_lines_fn=resolved_hooks.get("strip_verification_route_display_lines_fn"),
        hide_verification_route_lines_in_chat_fn=resolved_hooks.get("hide_verification_route_lines_in_chat_fn"),
        unwrap_accidental_full_text_codefence_fn=resolved_hooks.get("unwrap_accidental_full_text_codefence_fn"),
        strip_pathological_repetition_display_noise_fn=resolved_hooks.get(
            "strip_pathological_repetition_display_noise_fn"
        ),
    )


def apply_pre_render_policy_strict_gate_runtime_chain_stage(
    *,
    raw_response: str,
    user_raw: str,
    profile_name: str,
    is_command: bool,
    ctx: dict | None,
    header: str,
    alert_html: str,
    runtime_bundle: dict | None = None,
    gov_mgr=None,
    runtime_state=None,
    validator_obj=None,
    append_system_message_fn: Callable | None = None,
    get_enforcement_settings_fn: Callable | None = None,
    evaluate_strict_enforcement_fn: Callable | None = None,
    sanitize_html_fn: Callable | None = None,
    normalize_qc_overrides_fn: Callable | None = None,
    get_effective_qc_corridor_fn: Callable | None = None,
    strip_basic_html_for_enforcement_fn: Callable | None = None,
    ensure_qc_footer_present_fn: Callable | None = None,
    enforce_qc_footer_deltas_fn: Callable | None = None,
    ensure_qc_footer_is_last_fn: Callable | None = None,
    normalize_evidence_tags_fn: Callable | None = None,
    strip_empty_citation_placeholders_fn: Callable | None = None,
    strip_sci_menu_from_answer_fn: Callable | None = None,
    strip_sci_trace_line_when_inactive_fn: Callable | None = None,
    apply_self_debunking_text_postprocess_fn: Callable | None = None,
    normalize_sci_trace_numbering_fn: Callable | None = None,
    render_sci_trace_as_html_runtime_fn: Callable | None = None,
    qc_override_runtime_violations_fn: Callable | None = None,
    strip_internal_scaffolding_status_lines_fn: Callable | None = None,
    strip_exact_status_header_line_fn: Callable | None = None,
    apply_verification_route_display_policy_fn: Callable | None = None,
    output_pipeline_mod=None,
    verification_route_config=None,
    verification_route_provider: str | None = None,
    strip_verification_route_display_lines_fn: Callable | None = None,
    hide_verification_route_lines_in_chat_fn: Callable | None = None,
    unwrap_accidental_full_text_codefence_fn: Callable | None = None,
    strip_pathological_repetition_display_noise_fn: Callable | None = None,
):
    """Run strict-gate runtime-stage chain and normalize all stage outputs."""
    bundle = runtime_bundle if isinstance(runtime_bundle, dict) else {}
    hook_bundle = bundle.get("hooks", {}) if isinstance(bundle.get("hooks", {}), dict) else {}
    module_bundle = bundle.get("modules", {}) if isinstance(bundle.get("modules", {}), dict) else {}

    if gov_mgr is None:
        gov_mgr = bundle.get("gov_mgr")
    if runtime_state is None:
        runtime_state = bundle.get("runtime_state")
    if validator_obj is None:
        validator_obj = bundle.get("validator_obj")

    resolved_hooks = _resolve_runtime_chain_hook_set(
        hook_bundle=hook_bundle,
        explicit_hooks={
            "append_system_message_fn": append_system_message_fn,
            "get_enforcement_settings_fn": get_enforcement_settings_fn,
            "evaluate_strict_enforcement_fn": evaluate_strict_enforcement_fn,
            "sanitize_html_fn": sanitize_html_fn,
            "normalize_qc_overrides_fn": normalize_qc_overrides_fn,
            "get_effective_qc_corridor_fn": get_effective_qc_corridor_fn,
            "strip_basic_html_for_enforcement_fn": strip_basic_html_for_enforcement_fn,
            "ensure_qc_footer_present_fn": ensure_qc_footer_present_fn,
            "enforce_qc_footer_deltas_fn": enforce_qc_footer_deltas_fn,
            "ensure_qc_footer_is_last_fn": ensure_qc_footer_is_last_fn,
            "normalize_evidence_tags_fn": normalize_evidence_tags_fn,
            "strip_empty_citation_placeholders_fn": strip_empty_citation_placeholders_fn,
            "strip_sci_menu_from_answer_fn": strip_sci_menu_from_answer_fn,
            "strip_sci_trace_line_when_inactive_fn": strip_sci_trace_line_when_inactive_fn,
            "apply_self_debunking_text_postprocess_fn": apply_self_debunking_text_postprocess_fn,
            "normalize_sci_trace_numbering_fn": normalize_sci_trace_numbering_fn,
            "render_sci_trace_as_html_runtime_fn": render_sci_trace_as_html_runtime_fn,
            "qc_override_runtime_violations_fn": qc_override_runtime_violations_fn,
            "strip_internal_scaffolding_status_lines_fn": strip_internal_scaffolding_status_lines_fn,
            "strip_exact_status_header_line_fn": strip_exact_status_header_line_fn,
            "apply_verification_route_display_policy_fn": apply_verification_route_display_policy_fn,
            "strip_verification_route_display_lines_fn": strip_verification_route_display_lines_fn,
            "hide_verification_route_lines_in_chat_fn": hide_verification_route_lines_in_chat_fn,
            "unwrap_accidental_full_text_codefence_fn": unwrap_accidental_full_text_codefence_fn,
            "strip_pathological_repetition_display_noise_fn": strip_pathological_repetition_display_noise_fn,
        },
        keys=STRICT_GATE_RUNTIME_CHAIN_HOOK_KEYS,
    )

    if output_pipeline_mod is None:
        output_pipeline_mod = module_bundle.get("output_pipeline_mod", bundle.get("output_pipeline_mod"))
    if not isinstance(verification_route_config, dict):
        cfg_bundle = bundle.get("verification_route_config")
        if isinstance(cfg_bundle, dict):
            verification_route_config = cfg_bundle
    if not verification_route_provider:
        verification_route_provider = str(bundle.get("verification_route_provider") or "gemini")

    prof = str(profile_name or "Standard")
    raw_response_out = str(raw_response or "")
    out_alert_html = str(alert_html or "")
    header_out = str(header or "")
    route_cfg = verification_route_config if isinstance(verification_route_config, dict) else {}
    provider = str(verification_route_provider or "gemini")

    try:
        out = apply_pre_render_policy_strict_gate_stage(
            raw_response=raw_response_out,
            user_raw=str(user_raw or ""),
            profile_name=prof,
            is_command=bool(is_command),
            ctx=(ctx if isinstance(ctx, dict) else {}),
            header=header_out,
            alert_html=out_alert_html,
            gov_mgr=gov_mgr,
            runtime_state=runtime_state,
            validator_obj=validator_obj,
            append_system_message_fn=resolved_hooks.get("append_system_message_fn"),
            get_enforcement_settings_fn=resolved_hooks.get("get_enforcement_settings_fn"),
            evaluate_strict_enforcement_fn=resolved_hooks.get("evaluate_strict_enforcement_fn"),
            sanitize_html_fn=resolved_hooks.get("sanitize_html_fn"),
            normalize_qc_overrides_fn=resolved_hooks.get("normalize_qc_overrides_fn"),
            get_effective_qc_corridor_fn=resolved_hooks.get("get_effective_qc_corridor_fn"),
            strip_basic_html_for_enforcement_fn=resolved_hooks.get("strip_basic_html_for_enforcement_fn"),
            ensure_qc_footer_present_fn=resolved_hooks.get("ensure_qc_footer_present_fn"),
            enforce_qc_footer_deltas_fn=resolved_hooks.get("enforce_qc_footer_deltas_fn"),
            ensure_qc_footer_is_last_fn=resolved_hooks.get("ensure_qc_footer_is_last_fn"),
            normalize_evidence_tags_fn=resolved_hooks.get("normalize_evidence_tags_fn"),
            strip_empty_citation_placeholders_fn=resolved_hooks.get("strip_empty_citation_placeholders_fn"),
            strip_sci_menu_from_answer_fn=resolved_hooks.get("strip_sci_menu_from_answer_fn"),
            strip_sci_trace_line_when_inactive_fn=resolved_hooks.get("strip_sci_trace_line_when_inactive_fn"),
            apply_self_debunking_text_postprocess_fn=resolved_hooks.get("apply_self_debunking_text_postprocess_fn"),
            normalize_sci_trace_numbering_fn=resolved_hooks.get("normalize_sci_trace_numbering_fn"),
            render_sci_trace_as_html_runtime_fn=resolved_hooks.get("render_sci_trace_as_html_runtime_fn"),
            qc_override_runtime_violations_fn=resolved_hooks.get("qc_override_runtime_violations_fn"),
            strip_internal_scaffolding_status_lines_fn=resolved_hooks.get(
                "strip_internal_scaffolding_status_lines_fn"
            ),
            strip_exact_status_header_line_fn=resolved_hooks.get("strip_exact_status_header_line_fn"),
            apply_verification_route_display_policy_fn=resolved_hooks.get(
                "apply_verification_route_display_policy_fn"
            ),
            output_pipeline_mod=output_pipeline_mod,
            verification_route_config=route_cfg,
            verification_route_provider=provider,
            strip_verification_route_display_lines_fn=resolved_hooks.get("strip_verification_route_display_lines_fn"),
            hide_verification_route_lines_in_chat_fn=resolved_hooks.get("hide_verification_route_lines_in_chat_fn"),
            unwrap_accidental_full_text_codefence_fn=resolved_hooks.get("unwrap_accidental_full_text_codefence_fn"),
            strip_pathological_repetition_display_noise_fn=resolved_hooks.get(
                "strip_pathological_repetition_display_noise_fn"
            ),
        )
        out_norm = _normalize_pre_render_policy_strict_gate_stage_out(
            out,
            default_alert_html=out_alert_html,
            default_raw_response=raw_response_out,
        )
        if out_norm is not None:
            return out_norm
    except Exception:
        pass

    try:
        out = apply_pre_render_policy_strict_gate_fallback_stage(
            raw_response=raw_response_out,
            user_raw=str(user_raw or ""),
            profile_name=prof,
            header=header_out,
            alert_html=out_alert_html,
            runtime_state=runtime_state,
            validator_obj=validator_obj,
            append_system_message_fn=resolved_hooks.get("append_system_message_fn"),
            get_enforcement_settings_fn=resolved_hooks.get("get_enforcement_settings_fn"),
            evaluate_strict_enforcement_fn=resolved_hooks.get("evaluate_strict_enforcement_fn"),
            sanitize_html_fn=resolved_hooks.get("sanitize_html_fn"),
            apply_verification_route_display_policy_fn=resolved_hooks.get(
                "apply_verification_route_display_policy_fn"
            ),
            output_pipeline_mod=output_pipeline_mod,
            verification_route_config=route_cfg,
            verification_route_provider=provider,
            strip_verification_route_display_lines_fn=resolved_hooks.get("strip_verification_route_display_lines_fn"),
            hide_verification_route_lines_in_chat_fn=resolved_hooks.get("hide_verification_route_lines_in_chat_fn"),
            unwrap_accidental_full_text_codefence_fn=resolved_hooks.get("unwrap_accidental_full_text_codefence_fn"),
            strip_pathological_repetition_display_noise_fn=resolved_hooks.get(
                "strip_pathological_repetition_display_noise_fn"
            ),
        )
        out_norm = _normalize_pre_render_policy_strict_gate_stage_out(
            out,
            default_alert_html=out_alert_html,
            default_raw_response=raw_response_out,
        )
        if out_norm is not None:
            return out_norm
    except Exception:
        pass

    try:
        out = apply_pre_render_policy_strict_gate_ultimate_fallback_stage(
            raw_response=raw_response_out,
            user_raw=str(user_raw or ""),
            profile_name=prof,
            header=header_out,
            alert_html=out_alert_html,
            runtime_state=runtime_state,
            validator_obj=validator_obj,
            append_system_message_fn=resolved_hooks.get("append_system_message_fn"),
            get_enforcement_settings_fn=resolved_hooks.get("get_enforcement_settings_fn"),
            evaluate_strict_enforcement_fn=resolved_hooks.get("evaluate_strict_enforcement_fn"),
            sanitize_html_fn=resolved_hooks.get("sanitize_html_fn"),
            apply_verification_route_display_policy_fn=resolved_hooks.get(
                "apply_verification_route_display_policy_fn"
            ),
            output_pipeline_mod=output_pipeline_mod,
            verification_route_config=route_cfg,
            verification_route_provider=provider,
            strip_verification_route_display_lines_fn=resolved_hooks.get("strip_verification_route_display_lines_fn"),
            hide_verification_route_lines_in_chat_fn=resolved_hooks.get("hide_verification_route_lines_in_chat_fn"),
            unwrap_accidental_full_text_codefence_fn=resolved_hooks.get("unwrap_accidental_full_text_codefence_fn"),
            strip_pathological_repetition_display_noise_fn=resolved_hooks.get(
                "strip_pathological_repetition_display_noise_fn"
            ),
        )
        out_norm = _normalize_pre_render_policy_strict_gate_stage_out(
            out,
            default_alert_html=out_alert_html,
            default_raw_response=raw_response_out,
        )
        if out_norm is not None:
            return out_norm
    except Exception:
        pass

    return None
