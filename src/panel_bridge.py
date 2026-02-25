from __future__ import annotations


class PanelBridge:
    """Separate JS-API bridge for the Panel window."""

    def __init__(self, api):
        self._api = api

    def ping(self, _payload=None):
        return self._api.ping()

    def get_ui(self):
        return self._api.get_ui()

    def panel_action(self, action, payload=None):
        return self._api.panel_action(action, payload)
