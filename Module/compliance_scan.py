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
    prev_user_by_bot_id = build_prev_user_by_bot_id(history)

    rows: List[Tuple[int, str]] = []
    for i, msg_obj in enumerate(sample, 1):
        txt = (msg_obj or {}).get('content', '') or ''
        vios: List[str] = []

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

        # QC footer present? (skip for command responses)
        if (not is_cmd_msg) and ('QC-Matrix:' not in txt and 'QC:' not in txt):
            vios.append('Missing QC footer')

        # Self-Debunking required? (skip for command responses)
        try:
            prof_now = getattr(getattr(api, 'gov_state', None), 'active_profile', '') or 'Standard'
            if not is_cmd_msg:
                sd_msg = gov.check_self_debunking(txt, prof_now)
                if sd_msg:
                    vios.append(sd_msg)
        except Exception:
            pass

        # Verification Route Gate
        try:
            vr_msg = gov.check_verification_route_gate(txt)
            if vr_msg:
                vios.append(vr_msg)
        except Exception:
            pass

        # SCI Trace contract (if a variant is active)
        try:
            vk = getattr(getattr(api, 'gov_state', None), 'sci_variant', '') or ''
            if vk and not is_cmd_msg:
                if 'SCI Trace' not in txt:
                    vios.append('Missing SCI Trace block')
        except Exception:
            pass

        status = '✓ Compliant' if not vios else '⚠ ' + '; '.join(vios)
        rows.append((i, status))

    return rows
