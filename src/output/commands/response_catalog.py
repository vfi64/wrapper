from __future__ import annotations

import html
import os
import re
from typing import Callable, Mapping, Sequence


def build_profile_switch_audit_line(command: str, from_profile: str, to_profile: str) -> str:
    """Build deterministic profile-switch audit line."""
    return (
        f"Profile-Switch-Audit: command={command} · from={from_profile} · "
        f"to={to_profile} · rule=explicit-standalone-only"
    )


def _safe_call(fn: Callable | None, *args, **kwargs):
    if not callable(fn):
        return ""
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn(*args)
        except Exception:
            return ""
    except Exception:
        return ""


def _build_clickable_file_detail(path: str, *, cwd: str = "") -> tuple[str, str]:
    """Return (relative_label, HTML detail line with in-app clickable file link)."""
    path_s = str(path or "").strip()
    if not path_s:
        return "", ""

    rel_path = ""
    try:
        wd = str(cwd or "").strip() or os.getcwd()
        rel_path = os.path.relpath(path_s, start=wd)
        if str(rel_path).startswith(".."):
            rel_path = os.path.join("Logs", "Audit", os.path.basename(path_s))
    except Exception:
        rel_path = os.path.basename(path_s)

    label_html = html.escape(str(rel_path))
    detail = (
        "<br><a href='#' class='export-file-link' data-export-path='"
        + html.escape(path_s, quote=True)
        + "' onclick='return openExportFileFromAnchor(this);'><code>"
        + label_html
        + "</code></a>"
    )
    return str(rel_path), detail


_COMM_OVERLAY_COMMANDS = {
    "Comm Start",
    "Comm Stop",
    "Strict on",
    "Strict off",
    "Explore on",
    "Explore off",
    "Color on",
    "Color off",
    "SCI off",
    "SCI recurse",
}

_BASIC_STATE_COMMANDS = {
    "Strict on",
    "Strict off",
    "Explore on",
    "Explore off",
    "Color on",
    "Color off",
    "SCI on",
    "SCI off",
    "SCI recurse",
    "Comm Stop",
    "Comm Start",
    "Comm Anchor off",
    "Comm Anchor on",
    "Dynamic one-shot on",
}


def is_basic_command_supported(cmd: str) -> bool:
    """Return True if command belongs to the basic state-transition catalog set."""
    return str(cmd or "").strip() in _BASIC_STATE_COMMANDS


def resolve_comm_overlay_command_html(
    *,
    cmd: str,
    current_profile: str,
    prev_profile_for_audit: str,
    render_comm_state_html_fn: Callable | None = None,
) -> dict | None:
    """Resolve deterministic post-state HTML for Comm/overlay/color/SCI toggles."""
    cmd_s = str(cmd or "").strip()
    if cmd_s not in _COMM_OVERLAY_COMMANDS:
        return None

    cur = str(current_profile or "")
    prev = str(prev_profile_for_audit or "")
    comm_start_audit_line = ""
    if cmd_s == "Comm Start" and cur != prev:
        comm_start_audit_line = build_profile_switch_audit_line("Comm Start", prev, cur)

    html_content = _safe_call(
        render_comm_state_html_fn,
        audit_line=comm_start_audit_line,
    )
    return {
        "html": str(html_content or ""),
        "comm_start_audit_line": comm_start_audit_line,
    }


def resolve_comm_validate_command_html(
    *,
    cmd: str,
    render_comm_state_html_fn: Callable | None = None,
) -> dict | None:
    """Resolve deterministic post-state HTML for Comm Validate."""
    cmd_s = str(cmd or "").strip()
    if cmd_s != "Comm Validate":
        return None
    html_content = _safe_call(render_comm_state_html_fn, audit_line="")
    return {"html": str(html_content or "")}


_COMM_ANCHOR_TOGGLE_COMMANDS = {
    "Comm Anchor on",
    "Comm Anchor off",
}


def resolve_comm_anchor_toggle_command_html(
    *,
    cmd: str,
    render_comm_state_html_fn: Callable | None = None,
) -> dict | None:
    """Resolve deterministic post-state HTML for Comm Anchor on/off."""
    cmd_s = str(cmd or "").strip()
    if cmd_s not in _COMM_ANCHOR_TOGGLE_COMMANDS:
        return None
    html_content = _safe_call(render_comm_state_html_fn, audit_line="")
    return {"html": str(html_content or "")}


def resolve_dynamic_one_shot_command_html(
    *,
    cmd: str,
    render_comm_state_html_fn: Callable | None = None,
) -> dict | None:
    """Resolve deterministic post-state HTML for Dynamic one-shot on."""
    cmd_s = str(cmd or "").strip()
    if cmd_s != "Dynamic one-shot on":
        return None
    html_content = _safe_call(render_comm_state_html_fn, audit_line="")
    return {"html": str(html_content or "")}


def resolve_sci_on_command_html(
    *,
    cmd: str,
    lang: str = "de",
    set_sci_pending_fn: Callable[[], None] | None = None,
    render_sci_menu_html_fn: Callable | None = None,
) -> dict | None:
    """Resolve deterministic SCI-on post-state HTML (SCI variant menu)."""
    cmd_s = str(cmd or "").strip()
    if cmd_s != "SCI on":
        return None

    if callable(set_sci_pending_fn):
        try:
            set_sci_pending_fn()
        except Exception:
            pass

    html_content = _safe_call(render_sci_menu_html_fn, lang=str(lang or "de"))
    return {
        "html": str(html_content or ""),
        "triggered_sci": True,
    }


def build_qc_override_opened_result() -> dict:
    """Deterministic response payload for local QC-Override dialog opening."""
    return {"html": "<div class='sys'>QC Override dialog opened.</div>", "csc": None}


def _first_nonempty_line(text: str) -> str:
    for ln in str(text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _normalize_usage_in_out(usage: dict | None) -> tuple[object, object]:
    u = usage if isinstance(usage, dict) else {}
    u_in = u.get("prompt_tokens", u.get("input_tokens", u.get("input", u.get("in", 0))))
    u_out = u.get("completion_tokens", u.get("output_tokens", u.get("output", u.get("out", 0))))
    return u_in, u_out


def _build_comm_audit_rows_fallback(
    *,
    sample: Sequence[dict] | None,
    history: Sequence[dict] | None,
    sci_variant: str,
    profile: str,
    build_route_ctx_fn: Callable[[str, bool], Mapping[str, object]] | None = None,
    check_self_debunking_fn: Callable[[str, str], str] | None = None,
    check_verification_route_gate_fn: Callable[[str], str] | None = None,
) -> list[tuple[int, str]]:
    sample_list = list(sample or [])
    hist_list = list(history or [])

    prev_user_by_bot_id: dict[int, str] = {}
    try:
        for ix, msg_obj in enumerate(hist_list):
            if (msg_obj or {}).get("role") != "bot":
                continue
            prev_user = ""
            for jx in range(ix - 1, -1, -1):
                prev = hist_list[jx] or {}
                if (prev.get("role") or "") in ("user", "human"):
                    prev_user = str(prev.get("content", "") or "")
                    break
            prev_user_by_bot_id[id(msg_obj)] = prev_user
    except Exception:
        prev_user_by_bot_id = {}

    rows: list[tuple[int, str]] = []
    for i, msg_obj in enumerate(sample_list, 1):
        txt = str((msg_obj or {}).get("content", "") or "")
        vios: list[str] = []

        try:
            prev_user = str(prev_user_by_bot_id.get(id(msg_obj), "") or "").lstrip()
            prev_user_first = _first_nonempty_line(prev_user)
            first_line = _first_nonempty_line(txt)
            is_cmd_msg = False
            if callable(build_route_ctx_fn):
                ctx_turn = build_route_ctx_fn(prev_user_first, prev_user_first.startswith("Comm "))
                is_cmd_msg = bool((ctx_turn or {}).get("is_command"))
            if not is_cmd_msg:
                is_cmd_msg = bool(
                    re.match(
                        r"^(?:Command executed:\s+)?(?:Comm\s+\w+|Profile\s+\w+|SCI\s+(?:on|off|menu)|QC\s+Override)\b",
                        first_line,
                    )
                )
        except Exception:
            is_cmd_msg = False

        if (not is_cmd_msg) and ("QC-Matrix:" not in txt and "QC:" not in txt):
            vios.append("Missing QC footer")

        if (not is_cmd_msg) and callable(check_self_debunking_fn):
            try:
                sd_msg = str(check_self_debunking_fn(txt, str(profile or "Standard")) or "")
                if sd_msg:
                    vios.append(sd_msg)
            except Exception:
                pass

        if callable(check_verification_route_gate_fn):
            try:
                vr_msg = str(check_verification_route_gate_fn(txt) or "")
                if vr_msg:
                    vios.append(vr_msg)
            except Exception:
                pass

        if str(sci_variant or "").strip() and (not is_cmd_msg):
            if "SCI Trace" not in txt:
                vios.append("Missing SCI Trace block")

        status = "✓ Compliant" if not vios else "⚠ " + "; ".join(vios)
        rows.append((i, status))

    return rows


def build_comm_audit_command_result(
    *,
    cmd: str,
    timestamp: str,
    now_iso: str,
    profile: str,
    overlay: str,
    sci_pending: bool,
    sci_variant: str,
    sci_active: bool,
    history: Sequence[dict] | None = None,
    last_call_info: dict | None = None,
    deps_status: dict | None = None,
    comm_audit_window: int = 5,
    export_audit_fn: Callable[[dict], tuple[object, object] | None] | None = None,
    scan_rows_fn: Callable[[Sequence[dict], Sequence[dict]], Sequence[tuple[int, str]]] | None = None,
    build_route_ctx_fn: Callable[[str, bool], Mapping[str, object]] | None = None,
    check_self_debunking_fn: Callable[[str, str], str] | None = None,
    check_verification_route_gate_fn: Callable[[str], str] | None = None,
    append_history_fn: Callable[[str], None] | None = None,
    cwd: str = "",
    session_tokens_in: int = 0,
    session_tokens_out: int = 0,
) -> dict | None:
    """Build deterministic Comm-Audit output payload (no provider call)."""
    cmd_s = str(cmd or "").strip()
    if cmd_s != "Comm Audit":
        return None

    audit_event = {
        "event": "comm_audit_called",
        "ts": str(now_iso or ""),
        "n_last": 25,
        "profile": str(profile or ""),
        "overlay": str(overlay or ""),
        "sci": {
            "pending": bool(sci_pending),
            "variant": str(sci_variant or ""),
            "active": bool(sci_active),
        },
        "last_call": dict(last_call_info or {}),
    }

    chat_path = None
    audit_path = None
    if callable(export_audit_fn):
        try:
            pair = export_audit_fn(audit_event)
            if isinstance(pair, tuple) and len(pair) >= 2:
                chat_path, audit_path = pair[0], pair[1]
        except Exception:
            chat_path, audit_path = (None, None)

    _ = chat_path  # reserved for compatibility with legacy tuple return

    # Keep history available for deterministic side-effects (e.g., append_history policy),
    # but do not render compliance scan tables in the visible Comm-Audit output.
    hist_list = list(history or [])
    _ = hist_list

    msg = "Audit exported."

    dep_warn_html = ""
    try:
        ds = deps_status if isinstance(deps_status, dict) else {}
        missing = [k for k in ("auditstream", "rendering_utils", "compliance_scan") if not (ds.get(k, (False, ""))[0])]
        if missing:
            dep_warn_html = (
                "<div style='margin-top:8px; padding:8px; border:1px solid #d77; background:#fff3f3; border-radius:8px;'>"
                "<b>Warning:</b> Optional modules missing; fallback paths active: "
                + html.escape(", ".join(missing))
                + "</div>"
            )
    except Exception:
        dep_warn_html = ""

    detail = ""
    rel_audit = ""
    if audit_path:
        rel_audit, detail = _build_clickable_file_detail(str(audit_path), cwd=str(cwd or ""))

    last_line = ""
    try:
        lc = dict(last_call_info or {})
        prov = str(lc.get("provider") or "").strip()
        modl = str(lc.get("model") or "").strip()
        ms = int(lc.get("ms") or 0)
        usage = lc.get("usage") if isinstance(lc.get("usage"), dict) else {}
        u_in, u_out = _normalize_usage_in_out(usage)
        if prov or modl or ms or usage:
            last_line = (
                "<div style='margin-top:6px; font-size:12px; color:#444;'>"
                + f"Last call: <code>{html.escape(prov or 'n/a')}</code> · <code>{html.escape(modl or 'n/a')}</code> · {ms} ms"
                + (f" · usage in/out: {html.escape(str(u_in))}/{html.escape(str(u_out))}" if (u_in or u_out) else "")
                + "</div>"
            )
    except Exception:
        last_line = ""

    html_content = (
        "<div style='border:1px solid #bbb; background:#f7f7f7; padding:10px; border-radius:10px; margin:8px 0;'>"
        f"<b>Comm Audit</b><br>{msg}{detail}{dep_warn_html}{last_line}</div>"
    )
    html_content += f'<div class="ts-footer">Response at {html.escape(str(timestamp or ""))}</div>'

    bot_txt = "Comm Audit"
    if rel_audit:
        bot_txt += f"\nExportiert (Audit): {rel_audit}"
    if callable(append_history_fn):
        try:
            append_history_fn(bot_txt)
        except Exception:
            pass

    return {
        "html": html_content,
        "t_in": 0,
        "t_out": 0,
        "total_in": int(session_tokens_in or 0),
        "total_out": int(session_tokens_out or 0),
        "csc": None,
    }


def build_sci_menu_command_result(
    *,
    cmd: str,
    timestamp: str,
    render_sci_menu_html_fn: Callable[[], str] | None = None,
    render_ts_footer_html_fn: Callable[[str], str] | None = None,
    set_sci_state_fn: Callable[[], None] | None = None,
    increment_user_turn_fn: Callable[[], None] | None = None,
    append_history_fn: Callable[[str], None] | None = None,
) -> dict:
    """Deterministic response payload for explicit SCI-menu commands."""
    cmd_s = str(cmd or "").strip()
    if callable(set_sci_state_fn):
        try:
            set_sci_state_fn()
        except Exception:
            pass
    if callable(increment_user_turn_fn):
        try:
            increment_user_turn_fn()
        except Exception:
            pass

    html_content = str(_safe_call(render_sci_menu_html_fn) or "")
    footer = str(_safe_call(render_ts_footer_html_fn, str(timestamp or "")) or "")
    if footer:
        html_content = html_content + footer

    if callable(append_history_fn):
        try:
            append_history_fn(cmd_s or "SCI menu")
        except Exception:
            pass

    return {"html": html_content, "csc": None}


def build_sci_selection_result(
    *,
    letter: str,
    data: dict | None = None,
    lang: str = "de",
    translate_fn: Callable[[str, str], str] | None = None,
) -> dict:
    """Build deterministic SCI-variant activation output payload."""
    char = str(letter or "").strip().upper()
    if not char:
        char = "A"

    def _tr(key: str, fallback: str = "") -> str:
        if callable(translate_fn):
            try:
                return str(translate_fn(key, fallback) or fallback or "")
            except TypeError:
                try:
                    return str(translate_fn(key) or fallback or "")
                except Exception:
                    return str(fallback or "")
            except Exception:
                return str(fallback or "")
        return str(fallback or "")

    rules = data if isinstance(data, dict) else {}
    variants = (((rules.get("sci") or {}).get("variant_menu") or {}).get("variants") or {})
    variant = (variants.get(char) or {}) if isinstance(variants, dict) else {}
    if not isinstance(variant, dict):
        variant = {}

    vname = str(variant.get("name") or "")
    vfocus = str(variant.get("focus") or "")
    title = _tr(f"sci_name_{char}", "") or _tr(f"sci_var_{char}", "") or (vname or f"Variant {char}")
    desc = _tr(f"sci_focus_{char}", "") or vfocus

    # Keep stable output strings for deterministic replay/contracts.
    footer = "SCI activated"
    proto = "Protocol"
    lang_s = str(lang or "").strip().lower()
    if lang_s.startswith("de"):
        footer = "SCI activated"
        proto = "Protocol"

    html_out = f"""
        <div style="border: 2px solid #1a73e8; background: #f0f7ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="font-weight: bold; color: #1a73e8; font-size: 14px; margin-bottom: 5px;">SCI ACTIVE: {html.escape(char)}</div>
            <div style="font-size: 18px; font-weight: bold; color: #333; margin-bottom: 8px;">{html.escape(title)}</div>
    """

    if desc:
        html_out += f"""
            <div style="font-size: 14px; color: #444; line-height: 1.4;">
                <i>"{html.escape(desc)}"</i>
            </div>
            """

    html_out += f"""
            <hr style="border: 0; border-top: 1px solid #ccd; margin: 10px 0;">
            <div style="font-size: 11px; color: #666;">
                <b>{proto}:</b> Plan &rarr; Solution &rarr; Check.<br>
                Control Layer strictly monitors compliance with this role.
            </div>
        </div>
        <div class="ts-footer">{html.escape(footer)}</div>
        """

    return {
        "char": char,
        "html": str(html_out or ""),
        "history_text": f"SCI Variant {char} activated.",
    }


def resolve_renderer_map_command(
    *,
    cmd: str,
    timestamp: str,
    renderer_map: Mapping[str, Sequence[Callable | str]] | None = None,
    safe_html_fn: Callable[[str, Callable], str] | None = None,
    render_ts_footer_html_fn: Callable[[str], str] | None = None,
    increment_user_turn_fn: Callable[[], None] | None = None,
    append_history_fn: Callable[[str], None] | None = None,
) -> dict | None:
    """Resolve deterministic command outputs backed by a static renderer map."""
    cmd_s = str(cmd or "").strip()
    mapping = renderer_map if isinstance(renderer_map, Mapping) else {}
    spec = mapping.get(cmd_s)
    if not spec or len(spec) < 3:
        return None

    label = str(spec[0] or cmd_s)
    raw_fn = spec[1] if callable(spec[1]) else None
    html_fn = spec[2] if callable(spec[2]) else None

    if callable(increment_user_turn_fn):
        try:
            increment_user_turn_fn()
        except Exception:
            pass

    raw_text = _safe_call(raw_fn) if callable(raw_fn) else ""
    if not str(raw_text or "").strip():
        raw_text = label

    html_content = ""
    if callable(safe_html_fn):
        try:
            html_content = str(safe_html_fn(label, html_fn) or "")
        except Exception:
            html_content = ""
    if not html_content:
        html_content = str(_safe_call(html_fn) or "")

    footer = str(_safe_call(render_ts_footer_html_fn, str(timestamp or "")) or "")
    if footer:
        html_content = str(html_content or "") + footer

    if callable(append_history_fn):
        try:
            append_history_fn(str(raw_text or label))
        except Exception:
            pass

    return {"html": str(html_content or ""), "csc": None}


def apply_profile_switch_state(
    *,
    cmd: str,
    state,
    data: dict | None = None,
    resolve_profile_color_default_fn: Callable[[dict, str, str], str] | None = None,
    on_profile_qc_reset_fn: Callable[[], None] | None = None,
) -> bool:
    """Apply deterministic profile-switch transitions.

    Returns True for any `Profile ...` command (handled no-op for unknown profile names).
    """
    cmd_s = str(cmd or "").strip()
    if not cmd_s.startswith("Profile "):
        return False

    pname = cmd_s.split(" ", 1)[1].strip()
    rules = data if isinstance(data, dict) else {}
    profiles = (rules.get("profiles") or {}) if isinstance(rules, dict) else {}
    if not (isinstance(profiles, dict) and pname in profiles):
        return True

    state.active_profile = pname
    if callable(resolve_profile_color_default_fn):
        try:
            state.color = resolve_profile_color_default_fn(
                rules,
                pname,
                str(getattr(state, "color", "on") or "on"),
            )
        except Exception:
            pass

    # QC overrides are session-local and reset on any explicit profile switch.
    try:
        state.qc_overrides = {}
    except Exception:
        pass
    if callable(on_profile_qc_reset_fn):
        try:
            on_profile_qc_reset_fn()
        except Exception:
            pass

    try:
        state.sci_pending_turns = 0
    except Exception:
        pass

    if pname not in ("Expert", "Sparring"):
        state.sci_active = False
        state.sci_pending = False
        try:
            state.sci_variant = ""
        except Exception:
            pass
    return True


def apply_basic_command_state(
    *,
    cmd: str,
    state,
    data: dict | None = None,
    resolve_profile_color_default_fn: Callable[[dict, str, str], str] | None = None,
    try_enter_sci_recursion_fn: Callable[..., bool] | None = None,
) -> bool:
    """Apply basic deterministic legacy command transitions to runtime state.

    Returns True when a command was handled; False otherwise.
    """
    cmd_s = str(cmd or "").strip()
    rules = data if isinstance(data, dict) else {}
    if cmd_s not in _BASIC_STATE_COMMANDS:
        return False

    if cmd_s == "Strict on":
        state.overlay = "Strict"
        return True
    if cmd_s == "Strict off":
        state.overlay = ""
        return True
    if cmd_s == "Explore on":
        state.overlay = "Explore"
        return True
    if cmd_s == "Explore off":
        state.overlay = ""
        return True

    if cmd_s == "Color on":
        state.color = "on"
        return True
    if cmd_s == "Color off":
        state.color = "off"
        return True

    if cmd_s == "SCI on":
        state.sci_pending = True
        try:
            state.sci_pending_turns = 0
        except Exception:
            pass
        return True

    if cmd_s == "SCI off":
        state.sci_pending = False
        state.sci_active = False
        state.sci_variant = ""
        try:
            state.sci_pending_turns = 0
        except Exception:
            pass
        return True

    if cmd_s == "SCI recurse":
        max_depth = 2
        try:
            max_depth = int(((rules.get("sci") or {}).get("recursive_sci") or {}).get("max_depth", 2))
        except Exception:
            max_depth = 2
        if callable(try_enter_sci_recursion_fn):
            try:
                try_enter_sci_recursion_fn(state, max_depth=max_depth)
            except Exception:
                pass
        return True

    if cmd_s == "Comm Stop":
        state.comm_active = False
        return True

    if cmd_s == "Comm Start":
        state.comm_active = True
        try:
            default_prof = (rules.get("default_profile") or "Standard")
            profiles = (rules.get("profiles") or {}) if isinstance(rules, dict) else {}
            if isinstance(profiles, dict) and default_prof in profiles:
                state.active_profile = default_prof
                if callable(resolve_profile_color_default_fn):
                    try:
                        state.color = resolve_profile_color_default_fn(
                            rules,
                            str(default_prof),
                            str(getattr(state, "color", "on") or "on"),
                        )
                    except Exception:
                        pass
                try:
                    state.sci_pending_turns = 0
                except Exception:
                    pass
                if default_prof not in ("Expert", "Sparring"):
                    state.sci_active = False
                    state.sci_pending = False
                    state.sci_variant = ""
        except Exception:
            pass
        return True

    if cmd_s == "Comm Anchor off":
        try:
            state.anchor_auto = False
            state.anchor_force_next = False
            state.anchor_auto_user_override = True
        except Exception:
            pass
        return True

    if cmd_s == "Comm Anchor on":
        try:
            state.anchor_auto = True
            state.anchor_auto_user_override = True
            state.anchor_force_next = False
        except Exception:
            pass
        return True

    if cmd_s == "Dynamic one-shot on":
        try:
            state.dynamic_one_shot_active = True
        except Exception:
            pass
        try:
            state.dynamic_nudge = "one-shot"
        except Exception:
            pass
        return True

    return False


def resolve_post_state_command_html(
    *,
    cmd: str,
    current_profile: str,
    prev_profile_for_audit: str,
    timestamp: str,
    lang: str = "de",
    set_sci_pending_fn: Callable[[], None] | None = None,
    render_profile_switch_control_html_fn: Callable | None = None,
    render_sci_menu_html_fn: Callable | None = None,
    render_comm_state_html_fn: Callable | None = None,
) -> dict:
    """Resolve deterministic command HTML after state transition.

    This encapsulates the legacy wrapper routing for:
    - Profile switch output (with audit line and optional SCI menu append)
    - Comm Start audit-to-comm-state path
    - Generic command -> comm-state path
    """
    cmd_s = str(cmd or "").strip()
    cur = str(current_profile or "")
    prev = str(prev_profile_for_audit or "")

    comm_start_audit_line = ""
    if cmd_s == "Comm Start" and cur != prev:
        comm_start_audit_line = build_profile_switch_audit_line("Comm Start", prev, cur)

    profile_switch_audit_line = ""
    if cmd_s.startswith("Profile "):
        profile_switch_audit_line = build_profile_switch_audit_line(cmd_s, prev, cur)

    if cmd_s.startswith("Profile "):
        if cur in ("Expert", "Sparring") and callable(set_sci_pending_fn):
            try:
                set_sci_pending_fn()
            except Exception:
                pass
        html_content = _safe_call(
            render_profile_switch_control_html_fn,
            str(timestamp or ""),
            audit_line=profile_switch_audit_line,
        )
        if cur in ("Expert", "Sparring"):
            menu_html = _safe_call(render_sci_menu_html_fn, lang=str(lang or "de"))
            html_content = str(html_content or "") + "\n<div style='margin-top:12px'></div>\n" + str(menu_html or "")
        return {
            "html": str(html_content or ""),
            "profile_switch_audit_line": profile_switch_audit_line,
            "comm_start_audit_line": comm_start_audit_line,
            "triggered_sci": False,
        }

    triggered_sci = cmd_s in ("SCI on", "SCI menu")
    if triggered_sci:
        if callable(set_sci_pending_fn):
            try:
                set_sci_pending_fn()
            except Exception:
                pass
        html_content = _safe_call(render_sci_menu_html_fn, lang=str(lang or "de"))
    else:
        html_content = _safe_call(render_comm_state_html_fn, audit_line=comm_start_audit_line)

    return {
        "html": str(html_content or ""),
        "profile_switch_audit_line": profile_switch_audit_line,
        "comm_start_audit_line": comm_start_audit_line,
        "triggered_sci": bool(triggered_sci),
    }
