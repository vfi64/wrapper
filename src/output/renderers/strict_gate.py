from __future__ import annotations

import html
from typing import Any, Callable, Mapping, Sequence


def _render_violation_lines_html(
    *,
    hard_violations: Sequence[str],
    violations_struct: Sequence[Mapping[str, Any]] | None,
) -> str:
    if isinstance(violations_struct, Sequence) and violations_struct:
        items: list[str] = []
        for vv in violations_struct:
            if not isinstance(vv, Mapping):
                continue
            sev = str(vv.get("severity") or "").strip().upper()
            msg = str(vv.get("message") or "").strip()
            code = str(vv.get("code") or "").strip()
            if msg:
                items.append(f"<li>[{html.escape(sev)}] {html.escape(msg)}</li>")
            else:
                items.append(f"<li>[{html.escape(sev)}] {html.escape(code)}</li>")
        return "".join(items)
    return "".join(f"<li>{html.escape(str(x))}</li>" for x in (hard_violations or []))


def _to_violation_struct(
    *,
    hard_violations: Sequence[str],
    classify_violation_messages_best_effort_fn: Callable | None,
) -> list[dict[str, str]]:
    if not callable(classify_violation_messages_best_effort_fn):
        return []
    try:
        out = classify_violation_messages_best_effort_fn(
            [str(x) for x in (hard_violations or [])],
            default_severity="critical",
        )
    except Exception:
        return []
    if not isinstance(out, list):
        return []
    norm: list[dict[str, str]] = []
    for vv in out:
        if not isinstance(vv, Mapping):
            continue
        norm.append(
            {
                "code": str(vv.get("code") or "hard_violation"),
                "severity": str(vv.get("severity") or "critical"),
                "message": str(vv.get("message") or ""),
            }
        )
    return norm


def _resolve_strict_action(
    *,
    policy: str,
    enforcement_enabled: bool,
    blocked_severities: Sequence[str],
    hard_violations: Sequence[str],
    violations_struct: Sequence[Mapping[str, str]],
    runtime_state: Any,
    state_from_runtime_fn: Callable | None,
    apply_intent_fn: Callable | None,
    process_model_response_cls: Any,
    compliance_violation_cls: Any,
) -> str | None:
    if not (
        callable(state_from_runtime_fn)
        and callable(apply_intent_fn)
        and process_model_response_cls is not None
        and compliance_violation_cls is not None
    ):
        return None
    try:
        dom_state = state_from_runtime_fn(runtime_state)
    except Exception:
        return None
    try:
        dom_state.enforcement_policy = str(policy or "audit_only")
    except Exception:
        pass
    try:
        dom_state.enforcement_enabled = bool(enforcement_enabled)
    except Exception:
        pass
    try:
        bsl = [str(x).strip().lower() for x in (blocked_severities or []) if str(x).strip()]
        dom_state.blocked_severities = bsl or ["critical", "major"]
    except Exception:
        pass

    vio_objs = []
    if isinstance(violations_struct, Sequence) and violations_struct:
        for vv in violations_struct:
            if not isinstance(vv, Mapping):
                continue
            vio_objs.append(
                compliance_violation_cls(
                    rule=str(vv.get("code") or "hard_violation"),
                    severity=str(vv.get("severity") or "critical"),
                    message=str(vv.get("message") or ""),
                )
            )
    else:
        for msg in hard_violations or []:
            vio_objs.append(
                compliance_violation_cls(
                    rule="hard_violation",
                    severity="critical",
                    message=str(msg),
                )
            )

    try:
        tr = apply_intent_fn(
            dom_state,
            process_model_response_cls(raw_text="", violations=tuple(vio_objs)),
            {},
        )
        evs = list(getattr(tr, "audit_events", []) or [])
        if not evs:
            return None
        return str((evs[-1] or {}).get("action") or "").strip().lower() or None
    except Exception:
        return None


def evaluate_strict_enforcement(
    *,
    raw_for_render: str,
    user_raw: str,
    profile_name: str,
    override_violations: Sequence[str] | None,
    settings: Mapping[str, Any] | None,
    validate_fn: Callable[[str], tuple[Sequence[str], Sequence[str]]] | None,
    classify_violation_messages_best_effort_fn: Callable | None,
    runtime_state: Any,
    state_from_runtime_fn: Callable | None,
    apply_intent_fn: Callable | None,
    process_model_response_cls: Any,
    compliance_violation_cls: Any,
    render_strict_block_warning_html_fn: Callable[[str], str] | None,
    render_strict_warn_banner_html_fn: Callable[[str], str] | None,
    append_system_message_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate strict-enforcement stage and return banner/block decisions."""
    ens = dict(settings or {})
    policy = str(ens.get("policy") or "audit_only")
    enabled = bool(ens.get("enabled", True))
    blocked_severities = list(ens.get("blocked_severities") or ["critical", "major"])

    if (not enabled) or policy not in ("strict_warn", "strict_block"):
        return {"blocked": False, "strict_banner_html": "", "meta": None}

    hard_violations: list[str] = []
    if callable(validate_fn):
        try:
            hv2, _sv2 = validate_fn(str(raw_for_render or ""))
            hard_violations = [str(x) for x in (hv2 or [])]
        except Exception as exc:
            hard_violations = []
            if callable(append_system_message_fn):
                try:
                    append_system_message_fn(f"⚠️ QC/Validator error in strict enforcement: {exc}")
                except Exception:
                    pass
    if override_violations:
        hard_violations += [str(x) for x in list(override_violations)]
    if not hard_violations:
        return {"blocked": False, "strict_banner_html": "", "meta": None}

    violations_struct = _to_violation_struct(
        hard_violations=hard_violations,
        classify_violation_messages_best_effort_fn=classify_violation_messages_best_effort_fn,
    )

    strict_action = _resolve_strict_action(
        policy=policy,
        enforcement_enabled=enabled,
        blocked_severities=blocked_severities,
        hard_violations=hard_violations,
        violations_struct=violations_struct,
        runtime_state=runtime_state,
        state_from_runtime_fn=state_from_runtime_fn,
        apply_intent_fn=apply_intent_fn,
        process_model_response_cls=process_model_response_cls,
        compliance_violation_cls=compliance_violation_cls,
    )
    if strict_action is None:
        strict_action = "blocked" if policy == "strict_block" else "warned"

    vio_lines_html = _render_violation_lines_html(
        hard_violations=hard_violations,
        violations_struct=violations_struct,
    )

    if strict_action == "blocked":
        blocked_html = (
            render_strict_block_warning_html_fn(vio_lines_html)
            if callable(render_strict_block_warning_html_fn)
            else "<div><b>STRICT BLOCK</b></div>"
        )
        return {
            "blocked": True,
            "blocked_html": str(blocked_html or ""),
            "strict_banner_html": "",
            "meta": {
                "strict_enforcement": "blocked",
                "hard_violations": hard_violations,
                "violations_struct": violations_struct,
            },
        }

    warn_html = (
        render_strict_warn_banner_html_fn(vio_lines_html)
        if callable(render_strict_warn_banner_html_fn)
        else ""
    )
    return {
        "blocked": False,
        "blocked_html": "",
        "strict_banner_html": str(warn_html or ""),
        "meta": {
            "strict_enforcement": "warned",
            "hard_violations": hard_violations,
            "violations_struct": violations_struct,
        },
    }
