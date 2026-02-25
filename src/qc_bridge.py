from __future__ import annotations


class QCBridge:
    """Minimal JS bridge for the QC Override dialog."""

    def __init__(self, api):
        self._api = api

    def ping(self, _payload=None):
        try:
            import time as _time

            return {"ok": True, "ts": _time.time()}
        except Exception:
            return {"ok": True}

    def qc_get_state(self, _payload=None):
        try:
            return self._api.qc_get_state()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_apply(self, values):
        try:
            return self._api.qc_override_apply(values)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_clear(self, _payload=None):
        try:
            return self._api.qc_override_clear()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_cancel(self, _payload=None):
        try:
            return self._api.qc_override_cancel()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
