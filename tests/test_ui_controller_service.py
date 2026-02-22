from ui_controller import UIController


class _Win:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, script):
        self.calls.append(script)


class _BoomWin:
    def evaluate_js(self, script):
        raise RuntimeError("ui down")


def test_ui_controller_add_message_and_update_helpers():
    ui = UIController()
    win = _Win()

    assert ui.add_system_message(win, "Hallo 'Welt'") is True
    assert ui.add_error_message(win, "Fehler") is True
    assert ui.update_stats(win, "Reqs: 1 | In: 2 | Out: 3") is True
    assert ui.update_rule_file(win, "Comm-SCI-v20.0.3.json") is True
    assert ui.remote_input(win, "Comm Start") is True

    joined = "\n".join(win.calls)
    assert "addMsg(" in joined
    assert "updateStats(" in joined
    assert "updateRuleFile(" in joined
    assert "remoteInput(" in joined


def test_ui_controller_fail_soft_when_window_missing_or_broken():
    ui = UIController()
    assert ui.add_system_message(None, "x") is False
    assert ui.update_stats(None, "y") is False
    assert ui.remote_input(None, "z") is False

    bad = _BoomWin()
    assert ui.add_system_message(bad, "x") is False
    assert ui.update_stats(bad, "y") is False
    assert ui.remote_input(bad, "z") is False
