from __future__ import annotations


class MainBridge:
    """Slim JS-API bridge for the main chat window."""

    def __init__(self, api):
        self._api = api

    def ask(self, txt):
        return self._api.ask(txt)

    def remote_cmd(self, txt):
        return self._api.remote_cmd(txt)

    def ui_qc_bar_enabled(self):
        return self._api.ui_qc_bar_enabled()

    def is_ready(self):
        return self._api.is_ready()

    def ping(self, _payload=None):
        return self._api.ping(_payload)

    def update_stats_ui(self):
        return self._api.update_stats_ui()

    def ensure_panel_visible(self):
        return self._api.ensure_panel_visible()

    def load_rule_file(self):
        return self._api.load_rule_file()

    def export(self):
        return self._api.export()

    def settings(self):
        return self._api.settings()

    def close_app(self):
        return self._api.close_app()

    def set_exit_confirm_open(self, is_open):
        return self._api.set_exit_confirm_open(is_open)

    def get_help_content(self):
        return self._api.get_help_content()

    def submit_cgi_feedback(self, clarity, insight, efficiency, mode="repeat"):
        return self._api.submit_cgi_feedback(clarity, insight, efficiency, mode)

    def get_input_history(self, max_entries=200):
        return self._api.get_input_history(max_entries)

    def append_input_history(self, raw):
        return self._api.append_input_history(raw)
