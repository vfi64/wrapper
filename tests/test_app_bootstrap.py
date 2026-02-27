import pytest

import app_bootstrap as sut


class _FakeClosedEvent:
    def __init__(self, log):
        self._log = log
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        self._log.append(("bind_closed", handler))
        return self


class _FakeEvents:
    def __init__(self, log):
        self.closed = _FakeClosedEvent(log)


class _FakeWindow:
    def __init__(self, log):
        self.events = _FakeEvents(log)


class _FakeScreen:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class _FakeWebview:
    def __init__(self, log, screens=None):
        self._log = log
        self.screens = list(screens or [])
        self.start_callbacks = []

    def create_window(self, title, **kwargs):
        self._log.append(("create_window", title, kwargs))
        return _FakeWindow(self._log)

    def start(self, callback):
        self.start_callbacks.append(callback)
        self._log.append(("webview_start", callback))


class _FakeApi:
    def __init__(self, log, main_bridge=None):
        self._log = log
        self.main_bridge = main_bridge
        self.main_win = None
        self.on_main_window_close = lambda: None
        self.start_background_thread = lambda: None

    def _create_panel(self):
        self._log.append(("create_panel",))

    def _create_qc_override(self):
        self._log.append(("create_qc_override",))


def test_bootstrap_desktop_windows_preserves_window_order_and_close_binding():
    log = []
    api = _FakeApi(log)
    webview = _FakeWebview(log)

    win = sut.bootstrap_desktop_windows(api, webview, title="Main", html_chat="<html/>")

    assert win is api.main_win
    assert [entry[0] for entry in log] == [
        "create_window",
        "create_panel",
        "create_qc_override",
        "bind_closed",
    ]
    create_entry = log[0]
    kwargs = create_entry[2]
    assert kwargs["js_api"] is api
    assert kwargs["html"] == "<html/>"


def test_bootstrap_desktop_windows_prefers_main_bridge_for_js_api():
    log = []
    bridge = object()
    api = _FakeApi(log, main_bridge=bridge)
    webview = _FakeWebview(log)

    sut.bootstrap_desktop_windows(api, webview, title="Main", html_chat="X")

    create_entry = log[0]
    kwargs = create_entry[2]
    assert kwargs["js_api"] is bridge


def test_compute_startup_window_layout_is_side_by_side_without_overlap(monkeypatch):
    log = []
    webview = _FakeWebview(log, screens=[_FakeScreen(20, 30, 1800, 1000)])
    monkeypatch.setattr(sut, "_tk_screen_rect", lambda: None)
    layout = sut.compute_startup_window_layout(webview)
    main = layout["main"]
    panel = layout["panel"]
    assert main["x"] == 20
    assert main["y"] == 30
    assert panel["y"] == 30
    assert main["width"] + panel["width"] == 1800
    assert panel["x"] == (main["x"] + main["width"])
    assert panel["x"] > main["x"]
    assert main["height"] == 1000
    assert panel["height"] == 1000


def test_bootstrap_desktop_windows_sets_panel_geom_from_split_layout(monkeypatch):
    log = []
    api = _FakeApi(log)
    webview = _FakeWebview(log, screens=[_FakeScreen(0, 0, 1720, 960)])
    monkeypatch.setattr(sut, "_tk_screen_rect", lambda: None)
    sut.bootstrap_desktop_windows(api, webview, title="Main", html_chat="<html/>")
    create_entry = log[0]
    kwargs = create_entry[2]
    panel_geom = getattr(api, "panel_geom", {})
    assert kwargs["x"] == 0
    assert kwargs["y"] == 0
    assert kwargs["height"] == 960
    assert kwargs["width"] + int(panel_geom["width"]) == 1720
    assert int(panel_geom["x"]) == kwargs["x"] + kwargs["width"]
    assert int(panel_geom["y"]) == 0
    assert int(panel_geom["height"]) == 960


@pytest.mark.parametrize(
    "webview_module,genai_module,genai_types,pattern",
    [
        (None, object(), object(), "pywebview is required"),
        (object(), None, object(), "google-genai is required"),
        (object(), object(), None, "google-genai is required"),
    ],
)
def test_validate_desktop_runtime_dependencies_raises_expected_errors(
    webview_module, genai_module, genai_types, pattern
):
    with pytest.raises(SystemExit, match=pattern):
        sut.validate_desktop_runtime_dependencies(webview_module, genai_module, genai_types)


def test_validate_desktop_runtime_dependencies_accepts_present_modules():
    sut.validate_desktop_runtime_dependencies(object(), object(), object())


def test_run_desktop_app_preserves_composition_root_order_and_starts_webview():
    log = []
    webview = _FakeWebview(log)
    created = []

    def _factory():
        api = _FakeApi(log)
        created.append(api)
        log.append(("api_factory",))
        return api

    api = sut.run_desktop_app(
        api_factory=_factory,
        webview_module=webview,
        genai_module=object(),
        genai_types=object(),
        title="Main",
        html_chat="<html/>",
    )

    assert api is created[0]
    assert [entry[0] for entry in log] == [
        "api_factory",
        "create_window",
        "create_panel",
        "create_qc_override",
        "bind_closed",
        "webview_start",
    ]
    assert webview.start_callbacks == [api.start_background_thread]
