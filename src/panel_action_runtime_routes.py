from __future__ import annotations

from typing import Any, Callable


def try_handle_panel_action_runtime(
    api: Any,
    *,
    action: str,
    payload: dict | None,
    ok_fn: Callable[..., dict],
    err_fn: Callable[[str], dict],
) -> dict | None:
    """Handle panel_action routes that should stay deterministic and local."""
    action_s = str(action or "").strip()
    payload_d = payload if isinstance(payload, dict) else {}

    if action_s == "cmd":
        text = payload_d.get("text", "")
        try:
            if hasattr(api, "remote_cmd"):
                api.remote_cmd(str(text or ""))
                return ok_fn({"queued": True}, queued=True)
        except Exception as e:
            return err_fn(str(e))
        return err_fn("remote_cmd_unavailable")

    if action_s == "ask":
        text = payload_d.get("text", "")
        try:
            if hasattr(api, "ask"):
                res = api.ask(str(text or ""))
                if isinstance(res, dict):
                    return ok_fn(res, **res)
                return ok_fn({"html": str(res or "")})
        except Exception as e:
            return err_fn(str(e))
        return err_fn("ask_unavailable")

    if action_s == "export":
        try:
            chat_path, audit_path = api.export()
            return ok_fn(
                {"chat_path": chat_path, "audit_path": audit_path},
                chat_path=chat_path,
                audit_path=audit_path,
            )
        except Exception as e:
            return err_fn(str(e))

    if action_s == "open_export_preview":
        try:
            path = str((payload_d or {}).get("path", "") or "")
            max_chars = int((payload_d or {}).get("max_chars", 0) or 0)
            out = api.open_export_preview(path, max_chars=max_chars)
            if isinstance(out, dict):
                if bool(out.get("ok", False)):
                    return ok_fn(out, **out)
                return err_fn(str(out.get("error") or "preview_open_failed"))
            return err_fn("preview_open_failed")
        except Exception as e:
            return err_fn(str(e))

    if action_s == "preview_export_file":
        try:
            path = str((payload_d or {}).get("path", "") or "")
            max_chars = int((payload_d or {}).get("max_chars", 8000) or 8000)
            out = api.preview_export_file(path, max_chars=max_chars)
            if isinstance(out, dict):
                if bool(out.get("ok", False)):
                    return ok_fn(out, **out)
                return err_fn(str(out.get("error") or "preview_failed"))
            return err_fn("preview_failed")
        except Exception as e:
            return err_fn(str(e))

    if action_s == "manual_test_stop":
        fn = getattr(api, "manual_test_request_stop", None)
        res = fn(payload_d or {}) if callable(fn) else {"ok": False, "error": "manual_test_request_stop unavailable"}
        if isinstance(res, dict):
            if bool(res.get("ok", True)):
                return ok_fn(res, **res)
            return err_fn(str(res.get("error", "manual_test_stop_failed")))
        return ok_fn({"ok": True})

    return None
