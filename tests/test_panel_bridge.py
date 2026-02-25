from __future__ import annotations

import panel_bridge as sut


class _ApiStub:
    def __init__(self):
        self.calls = []

    def ping(self):
        self.calls.append(("ping",))
        return {"ok": True, "pong": 1}

    def get_ui(self):
        self.calls.append(("get_ui",))
        return {"ok": True, "comm_active": True}

    def panel_action(self, action, payload=None):
        self.calls.append(("panel_action", action, payload))
        return {"ok": True, "action": action, "payload": payload}


def test_panel_bridge_forwards_methods_to_api():
    api = _ApiStub()
    br = sut.PanelBridge(api)

    assert br.ping({})["ok"] is True
    assert br.get_ui()["ok"] is True
    out = br.panel_action("ask", {"text": "hi"})
    assert out["ok"] is True
    assert out["action"] == "ask"
    assert out["payload"] == {"text": "hi"}

    assert api.calls == [
        ("ping",),
        ("get_ui",),
        ("panel_action", "ask", {"text": "hi"}),
    ]
