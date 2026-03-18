from __future__ import annotations

import json
from typing import Callable


def _as_dict(x, fallback: dict | None = None) -> dict:
    if isinstance(x, dict):
        return x
    return dict(fallback or {})


def build_create_window_plan(
    *,
    seam_mod,
    html_manual_test_monitor: str,
    js_api_obj,
) -> dict:
    default = {
        "kwargs": {
            "title": "Comm-SCI Manual Test Monitor",
            "html": html_manual_test_monitor,
            "width": 760,
            "height": 700,
            "resizable": True,
            "hidden": True,
            "on_top": False,
            "js_api": js_api_obj,
        },
        "reset_state_before_create": True,
    }
    try:
        if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_create_window_kwargs_plan"):
            out = seam_mod.manual_test_monitor_create_window_kwargs_plan(
                html_manual_test_monitor=html_manual_test_monitor,
                js_api_obj=js_api_obj,
            )
            return _as_dict(out, default)
    except Exception:
        pass
    return default


def build_initial_state(*, seam_mod) -> dict:
    default = {"scenario": "", "status": "idle", "summary": "-", "events": []}
    try:
        if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_initial_state"):
            out = seam_mod.manual_test_monitor_initial_state()
            return _as_dict(out, default)
    except Exception:
        pass
    return default


def bind_window_events(*, seam_mod, win, closed_handler) -> None:
    evs = getattr(win, "events", None) if win else None
    closed_ev = getattr(evs, "closed", None)
    default = {"bind_closed": bool(win and evs is not None and closed_ev is not None)}
    plan = default
    try:
        if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_bind_window_events_plan"):
            out = seam_mod.manual_test_monitor_bind_window_events_plan(
                window_exists=bool(win),
                has_events=bool(evs is not None),
                has_closed_event=bool(closed_ev is not None),
            )
            plan = _as_dict(out, default)
    except Exception:
        plan = default
    if not bool(plan.get("bind_closed")):
        return
    if closed_ev is None or closed_handler is None:
        return
    try:
        closed_ev += closed_handler
    except Exception:
        pass


def _invoke_first_supported_method(win, methods: tuple[str, ...]) -> bool:
    for m in tuple(methods or ()):
        if hasattr(win, str(m)):
            getattr(win, str(m))()
            return True
    return False


def show_monitor(
    *,
    seam_mod,
    win,
    ensure_window_fn: Callable[[], object | None],
    clear_window_fn: Callable[[], None],
    eval_fn: Callable[[str], bool] | None,
    state,
) -> dict:
    default = {
        "create_if_missing": True,
        "error_if_unavailable": "manual_test_monitor_win unavailable",
        "show_methods": ("show",),
        "retry_after_show_failure": True,
        "clear_window_on_show_failure": True,
        "post_show_methods": ("bring_to_front",),
        "push_state_to_ui": True,
        "success_result": {"ok": True},
    }
    plan = default
    try:
        if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_show_plan"):
            out = seam_mod.manual_test_monitor_show_plan(window_exists=bool(win))
            plan = _as_dict(out, default)
    except Exception:
        plan = default

    if win is None and bool(plan.get("create_if_missing", True)):
        win = ensure_window_fn()
    if win is None:
        return {"ok": False, "error": str(plan.get("error_if_unavailable") or "manual_test_monitor_win unavailable")}

    show_methods = tuple(plan.get("show_methods") or ("show",))
    retry_after_fail = bool(plan.get("retry_after_show_failure", True))
    clear_on_fail = bool(plan.get("clear_window_on_show_failure", True))
    post_show_methods = tuple(plan.get("post_show_methods") or ("bring_to_front",))
    push_state = bool(plan.get("push_state_to_ui", True))
    success_result = plan.get("success_result")

    try:
        show_ok = _invoke_first_supported_method(win, show_methods)
    except Exception:
        show_ok = False

    if (not show_ok) and retry_after_fail:
        if clear_on_fail:
            try:
                clear_window_fn()
            except Exception:
                pass
        win = ensure_window_fn()
        if win is None:
            return {"ok": False, "error": str(plan.get("error_if_unavailable") or "manual_test_monitor_win unavailable")}
        try:
            _invoke_first_supported_method(win, show_methods)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    for m in post_show_methods:
        try:
            if hasattr(win, str(m)):
                getattr(win, str(m))()
        except Exception:
            pass

    if push_state and callable(eval_fn):
        try:
            if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_replace_js"):
                js_code = seam_mod.manual_test_monitor_replace_js(state if isinstance(state, dict) else {})
            else:
                js_code = f"mtmReplace({json.dumps(state if isinstance(state, dict) else {}, ensure_ascii=False)});"
            eval_fn(str(js_code or ""))
        except Exception:
            pass

    return success_result if isinstance(success_result, dict) else {"ok": True}


def hide_monitor(*, seam_mod, win) -> dict:
    default = {
        "window_methods": (("hide",) if win is not None and hasattr(win, "hide") else (("minimize",) if win is not None and hasattr(win, "minimize") else ())),
        "success_result": ({"ok": True, "hidden": True} if win is not None else {"ok": True, "hidden": True, "skipped": True}),
    }
    plan = default
    try:
        if seam_mod is not None and hasattr(seam_mod, "manual_test_monitor_hide_plan"):
            out = seam_mod.manual_test_monitor_hide_plan(
                window_exists=bool(win is not None),
                has_hide=bool(win is not None and hasattr(win, "hide")),
                has_minimize=bool(win is not None and hasattr(win, "minimize")),
            )
            plan = _as_dict(out, default)
    except Exception:
        plan = default

    for m in tuple(plan.get("window_methods") or ()):
        try:
            if win is not None and hasattr(win, str(m)):
                getattr(win, str(m))()
                break
        except Exception:
            pass
    res = plan.get("success_result")
    return res if isinstance(res, dict) else {"ok": True, "hidden": True}
