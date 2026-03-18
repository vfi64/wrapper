from __future__ import annotations

from typing import Any, Callable


def try_apply_panel_action_gate(
    api: Any,
    *,
    action: str,
    payload: dict | None,
    err_fn: Callable[[str], dict],
) -> dict | None:
    """Apply modal and comm-state panel gates. Returns a panel_action error dict or None."""
    action_s = str(action or "").strip()
    payload_d = payload if isinstance(payload, dict) else {}

    # Exit-confirm modal gate: exclusive program-wide modal state.
    try:
        if api._is_exit_confirm_modal_active():
            try:
                api.log_event(
                    "exit_confirm_block",
                    {"source": "panel_action", "action": action_s},
                )
            except Exception:
                pass
            return err_fn("exit_confirm_open_blocked")
    except Exception:
        pass

    # QC-Override modal gate: block non-QC actions while dialog is open.
    try:
        if api._is_qc_override_modal_active():
            allowed_modal_actions = {
                "qc_override_apply",
                "qc_override_clear",
                "qc_override_cancel",
                "panel_bootstrap_selftest",
            }
            if action_s not in allowed_modal_actions:
                try:
                    api._bring_qc_override_to_front()
                except Exception:
                    pass
                try:
                    api.log_event(
                        "qc_override_modal_block",
                        {"source": "panel_action", "action": action_s},
                    )
                except Exception:
                    pass
                return err_fn("qc_override_modal_blocked")
    except Exception:
        pass

    # Strict Comm-off panel gate.
    try:
        comm_on = bool(getattr(getattr(api, "gov_state", None), "comm_active", False))
    except Exception:
        comm_on = False
    if not comm_on:
        blocked_actions = {
            "qc_override_apply",
            "qc_override_clear",
            "manual_test_monitor_show",
            "manual_test_monitor_hide",
            "manual_test_monitor_reset",
            "manual_test_monitor_append",
            "manual_test_monitor_header",
            "save_manual_test_report",
            "manual_test_main_chat_append",
        }
        if action_s in blocked_actions:
            return err_fn("comm_off_blocked")
        if action_s in {"cmd", "ask"}:
            try:
                txt = str((payload_d or {}).get("text", "") or "").strip()
            except Exception:
                txt = ""
            if txt != "Comm Start":
                return err_fn("comm_off_blocked")

    return None
