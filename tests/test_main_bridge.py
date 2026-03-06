from __future__ import annotations

import main_bridge as sut


class _ApiStub:
    def __init__(self):
        self.calls = []

    def ask(self, txt):
        self.calls.append(("ask", txt))
        return {"ok": True, "txt": txt}

    def remote_cmd(self, txt):
        self.calls.append(("remote_cmd", txt))
        return {"ok": True, "cmd": txt}

    def ui_qc_bar_enabled(self):
        self.calls.append(("ui_qc_bar_enabled",))
        return True

    def is_ready(self):
        self.calls.append(("is_ready",))
        return True

    def ping(self, payload=None):
        self.calls.append(("ping", payload))
        return {"ok": True}

    def update_stats_ui(self):
        self.calls.append(("update_stats_ui",))
        return {"ok": True}

    def ensure_panel_visible(self):
        self.calls.append(("ensure_panel_visible",))
        return {"ok": True}

    def load_rule_file(self):
        self.calls.append(("load_rule_file",))
        return {"ok": True}

    def export(self):
        self.calls.append(("export",))
        return {"ok": True}

    def settings(self):
        self.calls.append(("settings",))
        return {"ok": True}

    def close_app(self):
        self.calls.append(("close_app",))
        return {"ok": True}

    def set_exit_confirm_open(self, is_open):
        self.calls.append(("set_exit_confirm_open", bool(is_open)))
        return {"ok": True, "open": bool(is_open)}

    def get_help_content(self):
        self.calls.append(("get_help_content",))
        return {"ok": True, "lang": "de", "payload": {"title": "Hilfe"}}


def test_main_bridge_forwards_methods_to_api():
    api = _ApiStub()
    br = sut.MainBridge(api)

    assert br.ask("hi")["ok"] is True
    assert br.remote_cmd("Comm State")["ok"] is True
    assert br.ui_qc_bar_enabled() is True
    assert br.is_ready() is True
    assert br.ping({})["ok"] is True
    assert br.update_stats_ui()["ok"] is True
    assert br.ensure_panel_visible()["ok"] is True
    assert br.load_rule_file()["ok"] is True
    assert br.export()["ok"] is True
    assert br.settings()["ok"] is True
    assert br.close_app()["ok"] is True
    assert br.set_exit_confirm_open(True)["ok"] is True
    assert br.get_help_content()["ok"] is True

    assert api.calls == [
        ("ask", "hi"),
        ("remote_cmd", "Comm State"),
        ("ui_qc_bar_enabled",),
        ("is_ready",),
        ("ping", {}),
        ("update_stats_ui",),
        ("ensure_panel_visible",),
        ("load_rule_file",),
        ("export",),
        ("settings",),
        ("close_app",),
        ("set_exit_confirm_open", True),
        ("get_help_content",),
    ]
