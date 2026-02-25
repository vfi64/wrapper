from __future__ import annotations

import qc_bridge as sut


class _ApiStub:
    def __init__(self):
        self.calls = []

    def qc_get_state(self):
        self.calls.append(("qc_get_state",))
        return {"ok": True, "profile": "Standard"}

    def qc_override_apply(self, values):
        self.calls.append(("qc_override_apply", values))
        return {"ok": True, "overrides": dict(values or {})}

    def qc_override_clear(self):
        self.calls.append(("qc_override_clear",))
        return {"ok": True}

    def qc_override_cancel(self):
        self.calls.append(("qc_override_cancel",))
        return {"ok": True}


def test_qc_bridge_forwards_qc_override_calls():
    api = _ApiStub()
    br = sut.QCBridge(api)

    assert br.qc_get_state({})["ok"] is True
    assert br.qc_override_apply({"Brevity": 1})["ok"] is True
    assert br.qc_override_clear({})["ok"] is True
    assert br.qc_override_cancel({})["ok"] is True

    assert api.calls == [
        ("qc_get_state",),
        ("qc_override_apply", {"Brevity": 1}),
        ("qc_override_clear",),
        ("qc_override_cancel",),
    ]


def test_qc_bridge_wraps_api_exceptions_as_error_dict():
    class _BoomApi:
        def qc_get_state(self):
            raise RuntimeError("boom")

    br = sut.QCBridge(_BoomApi())
    res = br.qc_get_state({})
    assert res["ok"] is False
    assert "RuntimeError: boom" in str(res.get("error"))


def test_qc_bridge_ping_returns_ok():
    br = sut.QCBridge(_ApiStub())
    res = br.ping({})
    assert res.get("ok") is True
