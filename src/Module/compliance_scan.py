"""
compliance_scan.py — minimal extraction from Wrapper monolith (Stage 3c).

Scope: best-effort message compliance scan used by "Comm Audit".
Design goals:
- Stateless functions, stdlib-only
- No dependencies on UI, providers, or pywebview
- Keep output strings identical to the monolith behavior (v170)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Tuple


def _norm_severity(value: str) -> str:
    s = (value or "").strip().lower()
    if s in {"critical", "major", "minor"}:
        return s
    return "major"


def _classify_violation_severity(code: str, message: str) -> str:
    """Best-effort severity mapping used for additive structured scanner output.

    This function is intentionally conservative:
    - critical: SCI trace contract failures during active SCI context
    - major: core contract failures (QC, Self-Debunking, Verification Route Gate)
    - minor: everything else / unknown
    """
    c = (code or "").strip().lower()
    m = (message or "").strip().lower()
    if c == "missing_sci_trace_block":
        return "critical"
    if c in {"missing_qc_footer", "missing_self_debunking", "verification_route_gate"}:
        return "major"
    if "missing sci trace" in m:
        return "critical"
    if ("missing qc" in m) or ("self-debunk" in m) or ("verification route" in m):
        return "major"
    return "minor"


def classify_violation_messages_best_effort(
    messages: List[str],
    *,
    default_severity: str = "major",
) -> List[Dict[str, str]]:
    """Map plain violation messages to structured objects with best-effort severity.

    Useful for compatibility paths where only textual violation lists exist.
    """
    out: List[Dict[str, str]] = []
    dsev = _norm_severity(default_severity)
    for msg in messages or []:
        m = str(msg or "").strip()
        if not m:
            continue
        # Best-effort code derivation from known phrases.
        ml = m.lower()
        if "missing sci trace" in ml or "sci trace step" in ml:
            code = "missing_sci_trace_block"
        elif "missing qc" in ml or "qc-matrix" in ml:
            code = "missing_qc_footer"
        elif "self-debunk" in ml:
            code = "missing_self_debunking"
        elif "verification route" in ml:
            code = "verification_route_gate"
        else:
            code = "hard_violation"
        sev = _classify_violation_severity(code, m)
        # Unknowns use caller-provided default.
        if code == "hard_violation":
            sev = dsev
        out.append({"code": code, "message": m, "severity": _norm_severity(sev)})
    return out


def build_prev_user_by_bot_id(history: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Build a mapping: id(assistant_message_dict) -> previous user message content (best-effort).
    This mirrors the monolith's logic and is intentionally forgiving.
    """
    prev_user_by_bot_id: Dict[int, str] = {}
    try:
        _hist = history or []
        for _ix, _m in enumerate(_hist):
            if (_m or {}).get('role') not in ('assistant', 'bot'):
                continue
            _prev_user = ''
            for _jx in range(_ix - 1, -1, -1):
                _pm = _hist[_jx] or {}
                if (_pm.get('role') or '') in ('user', 'human'):
                    _prev_user = str(_pm.get('content', '') or '')
                    break
            prev_user_by_bot_id[id(_m)] = _prev_user
    except Exception:
        return {}
    return prev_user_by_bot_id


def scan_message_compliance_best_effort(
    *,
    sample: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    build_route_ctx: Callable[[Any, str, bool], Dict[str, Any]],
    api: Any,
    gov: Any,
) -> List[Tuple[int, str]]:
    """
    Return rows: [(index, status_string), ...] where status_string is either
      '✓ Compliant' or '⚠ <violations...>'
    """
    detailed = scan_message_compliance_best_effort_detailed(
        sample=sample,
        history=history,
        build_route_ctx=build_route_ctx,
        api=api,
        gov=gov,
    )
    return [(idx, status) for idx, status, _v in detailed]


def scan_message_compliance_best_effort_detailed(
    *,
    sample: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    build_route_ctx: Callable[[Any, str, bool], Dict[str, Any]],
    api: Any,
    gov: Any,
) -> List[Tuple[int, str, List[Dict[str, str]]]]:
    """Return rows with structured violations (additive to legacy scan output).

    Row shape:
      (index, status_string, violations)
    where violations is a list of:
      {'code': str, 'message': str, 'severity': 'critical'|'major'|'minor'}
    """
    prev_user_by_bot_id = build_prev_user_by_bot_id(history)

    rows: List[Tuple[int, str, List[Dict[str, str]]]] = []
    for i, msg_obj in enumerate(sample, 1):
        txt = (msg_obj or {}).get('content', '') or ''
        vios: List[Dict[str, str]] = []

        # Command responses are deterministic UI/control outputs; they must not be
        # evaluated against Self-Debunking / QC footer / SCI Trace contracts.
        # Classification is based primarily on the preceding user message (Comm ...),
        # and only secondarily on the assistant's first line.
        try:
            prev_user = (prev_user_by_bot_id.get(id(msg_obj), '') or '').lstrip()
            prev_user_first = ''
            for _ln in str(prev_user).splitlines():
                if _ln.strip():
                    prev_user_first = _ln.strip()
                    break
            first_line = ''
            for _ln in str(txt).splitlines():
                if _ln.strip():
                    first_line = _ln.strip()
                    break

            ctx_turn = build_route_ctx(api, prev_user_first, prev_user_first.startswith('Comm '))
            is_cmd_msg = bool(ctx_turn.get('is_command'))
            if not is_cmd_msg:
                is_cmd_msg = bool(re.match(
                    r"^(?:Command executed:\s+)?(?:Comm\s+\w+|Profile\s+\w+|SCI\s+(?:on|off|menu)|QC\s+Override)\b",
                    first_line
                ))
        except Exception:
            is_cmd_msg = False

        def _add_vio(code: str, message: str) -> None:
            sev = _classify_violation_severity(code, message)
            vios.append({
                "code": str(code or "").strip() or "unknown_violation",
                "message": str(message or "").strip(),
                "severity": _norm_severity(sev),
            })

        # QC footer present? (skip for command responses)
        if (not is_cmd_msg) and ('QC-Matrix:' not in txt and 'QC:' not in txt):
            _add_vio("missing_qc_footer", "Missing QC footer")

        # Self-Debunking required? (skip for command responses)
        try:
            prof_now = getattr(getattr(api, 'gov_state', None), 'active_profile', '') or 'Standard'
            if not is_cmd_msg:
                sd_msg = gov.check_self_debunking(txt, prof_now)
                if sd_msg:
                    _add_vio("missing_self_debunking", str(sd_msg))
        except Exception:
            pass

        # Verification Route Gate
        try:
            vr_msg = gov.check_verification_route_gate(txt)
            if vr_msg:
                _add_vio("verification_route_gate", str(vr_msg))
        except Exception:
            pass

        # SCI Trace contract (if a variant is active)
        try:
            vk = getattr(getattr(api, 'gov_state', None), 'sci_variant', '') or ''
            if vk and not is_cmd_msg:
                if 'SCI Trace' not in txt:
                    _add_vio("missing_sci_trace_block", "Missing SCI Trace block")
        except Exception:
            pass

        status = '✓ Compliant' if not vios else '⚠ ' + '; '.join(v.get("message", "") for v in vios)
        rows.append((i, status, vios))

    return rows
