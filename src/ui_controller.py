from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class UIController:
    """Thin pywebview UI facade (fail-soft)."""

    def eval_js(self, win: Any, script: str) -> bool:
        if win is None:
            return False
        try:
            win.evaluate_js(str(script))
            return True
        except Exception:
            return False

    def add_message(self, win: Any, role: str, message: str) -> bool:
        try:
            role_s = str(role or "sys")
            msg_s = str(message or "")
            script = f"addMsg({json.dumps(role_s)}, {json.dumps(msg_s, ensure_ascii=False)});"
            return self.eval_js(win, script)
        except Exception:
            return False

    def add_system_message(self, win: Any, message: str) -> bool:
        return self.add_message(win, "sys", message)

    def add_error_message(self, win: Any, message: str) -> bool:
        return self.add_message(win, "err", message)

    def update_stats(self, win: Any, text: str) -> bool:
        try:
            script = f"updateStats({json.dumps(str(text or ''), ensure_ascii=False)});"
            return self.eval_js(win, script)
        except Exception:
            return False

    def update_rule_file(self, win: Any, filename: str) -> bool:
        try:
            script = f"updateRuleFile({json.dumps(str(filename or ''), ensure_ascii=False)});"
            return self.eval_js(win, script)
        except Exception:
            return False

    def remote_input(self, win: Any, cmd: str) -> bool:
        try:
            script = f"remoteInput({json.dumps(str(cmd or ''), ensure_ascii=False)});"
            return self.eval_js(win, script)
        except Exception:
            return False

    def try_handle_panel_aux_action(self, app: Any, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Handle a small, stable subset of panel_action routes.

        Returns:
            dict result in the panel_action schema if handled, otherwise ``None``.
        """
        action_s = str(action or "").strip()
        payload = payload or {}

        def _ok(result=None, **extra):
            out = {"ok": True, "action": action_s, "result": result, "error": None}
            if extra:
                out.update(extra)
            return out

        def _err(message: str):
            return {"ok": False, "action": action_s, "result": None, "error": str(message or "error")}

        try:
            if action_s == "set_provider":
                fn = getattr(app, "set_provider", None)
                if not callable(fn):
                    return _err("set_provider unavailable")
                provider = payload.get("provider", "")
                res = fn(str(provider or ""))
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "set_provider_failed")))
                return _ok(res)

            if action_s == "set_model":
                fn = getattr(app, "set_model", None)
                if not callable(fn):
                    return _err("set_model unavailable")
                model = payload.get("model", "")
                return _ok(fn(str(model or "")))

            if action_s == "set_answer_language":
                fn = getattr(app, "set_answer_language", None)
                if not callable(fn):
                    return _err("set_answer_language unavailable")
                lang = payload.get("lang", "")
                return _ok(fn(str(lang or "")))

            if action_s == "set_api_key":
                fn = getattr(app, "set_api_key_for_provider", None)
                if not callable(fn):
                    return _err("set_api_key_for_provider unavailable")
                provider = payload.get("provider", "")
                api_key = payload.get("api_key", "")
                persist = bool(payload.get("persist", True))
                encrypt = bool(payload.get("encrypt", False))
                passphrase = payload.get("passphrase", "")
                write_path = payload.get("write_path", "")
                res = fn(
                    str(provider or ""),
                    str(api_key or ""),
                    persist=persist,
                    write_path=str(write_path or ""),
                    encrypt=encrypt,
                    passphrase=str(passphrase or ""),
                )
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "set_api_key_failed")))
                return _ok({"ok": True})

            if action_s == "set_key_passphrase":
                fn = getattr(app, "set_key_passphrase", None)
                if not callable(fn):
                    return _err("set_key_passphrase unavailable")
                passphrase = payload.get("passphrase", "")
                provider = payload.get("provider", "")
                reason = payload.get("reason", "")
                reconnect = bool(payload.get("reconnect", True))
                res = fn(
                    str(passphrase or ""),
                    str(provider or ""),
                    reason=str(reason or ""),
                    reconnect=reconnect,
                )
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "set_key_passphrase_failed")))
                return _ok({"ok": True})

            if action_s == "delete_api_key":
                fn = getattr(app, "delete_api_key_for_provider", None)
                provider = payload.get("provider", "")
                persist = bool(payload.get("persist", True))
                write_path = payload.get("write_path", "")
                if callable(fn):
                    res = fn(
                        str(provider or ""),
                        persist=persist,
                        write_path=str(write_path or ""),
                    )
                else:
                    fn_set = getattr(app, "set_api_key_for_provider", None)
                    if not callable(fn_set):
                        return _err("delete_api_key_for_provider unavailable")
                    res = fn_set(
                        str(provider or ""),
                        "",
                        persist=persist,
                        write_path=str(write_path or ""),
                        encrypt=False,
                        passphrase="",
                    )
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "delete_api_key_failed")))
                return _ok({"ok": True})

            if action_s == "set_language_policy_mode":
                fn = getattr(app, "set_language_policy_mode", None)
                if not callable(fn):
                    return _err("set_language_policy_mode unavailable")
                mode = payload.get("mode", "")
                return _ok(fn(str(mode or "")))

            if action_s == "set_hide_verification_route_lines":
                fn = getattr(app, "set_hide_verification_route_lines", None)
                if not callable(fn):
                    return _err("set_hide_verification_route_lines unavailable")
                enabled = payload.get("enabled", None)
                if bool(payload.get("clear", False)):
                    enabled = None
                scope = payload.get("scope", "provider")
                provider = payload.get("provider", "")
                res = fn(enabled, scope=str(scope or "provider"), provider=str(provider or ""))
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "set_hide_verification_route_lines_failed")))
                return _ok({"ok": True})

            if action_s == "refresh_models":
                fn_refresh = getattr(app, "refresh_models", None)
                if not callable(fn_refresh):
                    return _err("refresh_models unavailable")
                provider = payload.get("provider", "")
                try:
                    p = str(provider or "").strip().lower()
                except Exception:
                    p = ""
                if p:
                    try:
                        cfg = getattr(app, "cfg", None)
                        curp = (cfg.get_active_provider() or "gemini").strip().lower() if cfg is not None else "gemini"
                    except Exception:
                        curp = "gemini"
                    if p != curp:
                        try:
                            fn_set_provider = getattr(app, "set_provider", None)
                            if callable(fn_set_provider):
                                fn_set_provider(p)
                        except Exception:
                            pass
                return _ok(fn_refresh())

            if action_s == "hf_catalog":
                fn = getattr(app, "hf_catalog", None)
                if not callable(fn):
                    return _err("hf_catalog unavailable")
                top_n = payload.get("top_n", 200)
                provider_filter = payload.get("provider_filter", "all")
                force_refresh = bool(payload.get("force_refresh", False))
                return _ok(
                    fn(
                        top_n=int(top_n or 200),
                        provider_filter=str(provider_filter or "all"),
                        force_refresh=force_refresh,
                    )
                )

            if action_s == "list_chat_logs":
                fn = getattr(app, "list_chat_logs", None)
                if not callable(fn):
                    return _err("list_chat_logs unavailable")
                limit = payload.get("limit", 200)
                lr = fn(limit=int(limit or 200))
                logs = []
                try:
                    if isinstance(lr, dict):
                        logs = lr.get("logs") or []
                    elif isinstance(lr, list):
                        logs = lr
                except Exception:
                    logs = []
                if not isinstance(logs, list):
                    logs = []
                return _ok({"logs": logs}, logs=logs)

            if action_s == "load_chat_log":
                fn = getattr(app, "load_chat_log", None)
                if not callable(fn):
                    return _err("load_chat_log unavailable")
                name = payload.get("name", "")
                fork = bool(payload.get("fork", True))
                res = fn(str(name or ""), fork=fork)
                try:
                    if isinstance(res, dict) and res.get("ok") is True:
                        replay = getattr(app, "_ui_replay_loaded_history", None)
                        if callable(replay):
                            replay(status_msg=f"Loaded: {str(name or '').split('/')[-1]} ({'fork' if fork else 'no fork'})")
                except Exception:
                    pass
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "load_chat_log_failed")))
                return _ok({"ok": True})

            if action_s == "clear_chat":
                fn = getattr(app, "clear_chat", None)
                if not callable(fn):
                    return _err("clear_chat unavailable")
                res = fn()
                try:
                    ok = bool(res.get("ok", True)) if isinstance(res, dict) else True
                except Exception:
                    ok = True
                if ok:
                    if isinstance(res, dict):
                        return _ok(res, **res)
                    return _ok({"ok": True})
                if isinstance(res, dict):
                    return _err(str(res.get("error", "clear_chat_failed")))
                return _err("clear_chat_failed")

            if action_s == "qc_override_apply":
                fn = getattr(app, "qc_override_apply", None)
                res = fn(payload.get("values", {})) if callable(fn) else {"ok": False, "error": "qc_override_apply unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "qc_override_apply_failed")))
                return _ok({"ok": True})

            if action_s == "qc_override_clear":
                fn = getattr(app, "qc_override_clear", None)
                res = fn({}) if callable(fn) else {"ok": False, "error": "qc_override_clear unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "qc_override_clear_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_monitor_show":
                fn = getattr(app, "manual_test_monitor_show", None)
                res = fn(payload or {}) if callable(fn) else {"ok": False, "error": "manual_test_monitor_show unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_monitor_show_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_monitor_hide":
                fn = getattr(app, "manual_test_monitor_hide", None)
                res = fn() if callable(fn) else {"ok": False, "error": "manual_test_monitor_hide unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_monitor_hide_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_monitor_reset":
                fn = getattr(app, "manual_test_monitor_reset", None)
                res = fn(payload or {}) if callable(fn) else {"ok": False, "error": "manual_test_monitor_reset unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_monitor_reset_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_monitor_append":
                fn = getattr(app, "manual_test_monitor_append", None)
                entry = payload.get("entry", payload)
                res = fn(entry) if callable(fn) else {"ok": False, "error": "manual_test_monitor_append unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_monitor_append_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_monitor_header":
                fn = getattr(app, "manual_test_monitor_set_header", None)
                res = fn(payload or {}) if callable(fn) else {"ok": False, "error": "manual_test_monitor_set_header unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_monitor_set_header_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_stop":
                fn = getattr(app, "manual_test_request_stop", None)
                res = fn(payload or {}) if callable(fn) else {"ok": False, "error": "manual_test_request_stop unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_stop_failed")))
                return _ok({"ok": True})

            if action_s == "save_manual_test_report":
                fn = getattr(app, "save_manual_test_report", None)
                res = fn(payload.get("report", {})) if callable(fn) else {"ok": False, "error": "save_manual_test_report unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "save_manual_test_report_failed")))
                return _ok({"ok": True})

            if action_s == "manual_test_main_chat_append":
                fn = getattr(app, "manual_test_main_chat_append", None)
                res = fn(payload.get("payload", {})) if callable(fn) else {"ok": False, "error": "manual_test_main_chat_append unavailable"}
                if isinstance(res, dict):
                    if bool(res.get("ok", True)):
                        return _ok(res, **res)
                    return _err(str(res.get("error", "manual_test_main_chat_append_failed")))
                return _ok({"ok": True})
        except Exception as e:
            return _err(str(e))

        return None
