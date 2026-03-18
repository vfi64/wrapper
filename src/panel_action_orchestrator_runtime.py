from __future__ import annotations

from typing import Any


def dispatch_panel_action(
    api: Any,
    *,
    action: str,
    payload: dict | None,
    gate_mod: Any = None,
    runtime_routes_mod: Any = None,
) -> dict:
    """Orchestrate panel_action dispatch with stable schema and deterministic routing."""
    try:
        action_s = str(action or "").strip()
    except Exception:
        action_s = ""
    payload_d = payload or {}
    try:
        if not isinstance(payload_d, dict):
            payload_d = {"value": payload_d}
    except Exception:
        payload_d = {}

    def _ok(result=None, **extra):
        out = {"ok": True, "action": action_s, "result": result, "error": None}
        if extra:
            out.update(extra)
        return out

    def _err(message: str):
        return {"ok": False, "action": action_s, "result": None, "error": str(message or "error")}

    if gate_mod is not None:
        try:
            gate_fn = getattr(gate_mod, "try_apply_panel_action_gate", None)
            if callable(gate_fn):
                gate_out = gate_fn(
                    api,
                    action=action_s,
                    payload=payload_d,
                    err_fn=_err,
                )
                if isinstance(gate_out, dict):
                    return gate_out
        except Exception:
            pass

    try:
        ui = getattr(api, "ui_controller", None)
        if ui is not None and hasattr(ui, "try_handle_panel_aux_action"):
            delegated = ui.try_handle_panel_aux_action(api, action_s, payload_d)
            if isinstance(delegated, dict):
                return delegated
    except Exception:
        pass

    if action_s == "panel_bootstrap_selftest":
        try:
            info = api._panel_accept_bootstrap_report(payload_d)
        except Exception as e:
            info = {"accepted": False, "runtime_ok": False, "reason": f"{type(e).__name__}: {e}"}
        return _ok(info, runtime_ok=bool((info or {}).get("runtime_ok")))

    if runtime_routes_mod is not None:
        try:
            fn = getattr(runtime_routes_mod, "try_handle_panel_action_runtime", None)
            if callable(fn):
                out = fn(
                    api,
                    action=action_s,
                    payload=payload_d,
                    ok_fn=_ok,
                    err_fn=_err,
                )
                if isinstance(out, dict):
                    return out
        except Exception:
            pass

    return _err("unknown action")
