from __future__ import annotations

import manual_test_monitor_runtime as sut


class _ClosedEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _Events:
    def __init__(self):
        self.closed = _ClosedEvent()


class _Win:
    def __init__(self, *, fail_show: bool = False):
        self.events = _Events()
        self.fail_show = bool(fail_show)
        self.calls = []

    def show(self):
        self.calls.append("show")
        if self.fail_show:
            raise RuntimeError("show boom")

    def bring_to_front(self):
        self.calls.append("bring_to_front")

    def hide(self):
        self.calls.append("hide")


def test_build_create_window_plan_prefers_seam_output():
    class _Seam:
        @staticmethod
        def manual_test_monitor_create_window_kwargs_plan(**_kwargs):
            return {"kwargs": {"title": "X"}, "reset_state_before_create": False}

    plan = sut.build_create_window_plan(
        seam_mod=_Seam(),
        html_manual_test_monitor="<html/>",
        js_api_obj=object(),
    )
    assert plan["kwargs"]["title"] == "X"
    assert plan["reset_state_before_create"] is False


def test_bind_window_events_uses_closed_handler_plan():
    class _Seam:
        @staticmethod
        def manual_test_monitor_bind_window_events_plan(**_kwargs):
            return {"bind_closed": True}

    win = _Win()

    def _handler():
        return None

    sut.bind_window_events(seam_mod=_Seam(), win=win, closed_handler=_handler)
    assert win.events.closed.handlers == [_handler]


def test_show_monitor_retries_after_first_show_failure():
    class _Seam:
        @staticmethod
        def manual_test_monitor_show_plan(**_kwargs):
            return {
                "create_if_missing": True,
                "error_if_unavailable": "missing",
                "show_methods": ("show",),
                "retry_after_show_failure": True,
                "clear_window_on_show_failure": True,
                "post_show_methods": ("bring_to_front",),
                "push_state_to_ui": True,
                "success_result": {"ok": True},
            }

        @staticmethod
        def manual_test_monitor_replace_js(state):
            return "mtmReplace({});"

    first = _Win(fail_show=True)
    second = _Win(fail_show=False)
    win_ref = {"win": first}
    cleared = {"count": 0}
    eval_calls = []

    def _ensure():
        return win_ref["win"]

    def _clear():
        cleared["count"] += 1
        win_ref["win"] = second

    out = sut.show_monitor(
        seam_mod=_Seam(),
        win=first,
        ensure_window_fn=_ensure,
        clear_window_fn=_clear,
        eval_fn=lambda js: eval_calls.append(str(js)) or True,
        state={"scenario": "s1"},
    )
    assert out.get("ok") is True
    assert cleared["count"] == 1
    assert "show" in second.calls
    assert "bring_to_front" in second.calls
    assert eval_calls and eval_calls[0].startswith("mtmReplace(")


def test_hide_monitor_returns_skipped_when_window_missing():
    out = sut.hide_monitor(seam_mod=None, win=None)
    assert out == {"ok": True, "hidden": True, "skipped": True}
