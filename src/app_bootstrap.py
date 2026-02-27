"""S8 bootstrap helpers for the pywebview composition root.

Behavior-preserving extraction from the monolithic launcher path.
"""

from __future__ import annotations

import os


def validate_desktop_runtime_dependencies(webview_module, genai_module, genai_types):
    if webview_module is None:
        raise SystemExit('pywebview is required. Install with: pip install pywebview')
    if genai_module is None or genai_types is None:
        raise SystemExit('google-genai is required. Install with: pip install google-genai')


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _rect_from_screen_obj(screen_obj):
    if isinstance(screen_obj, dict):
        x = screen_obj.get("x", 0)
        y = screen_obj.get("y", 0)
        w = screen_obj.get("width", 0)
        h = screen_obj.get("height", 0)
        return _safe_int(x, 0), _safe_int(y, 0), _safe_int(w, 0), _safe_int(h, 0)
    x = getattr(screen_obj, "x", 0)
    y = getattr(screen_obj, "y", 0)
    w = getattr(screen_obj, "width", 0)
    h = getattr(screen_obj, "height", 0)
    return _safe_int(x, 0), _safe_int(y, 0), _safe_int(w, 0), _safe_int(h, 0)


def _mac_visible_frame_rect():
    """Return macOS visible frame (excludes menu bar + dock) when available."""
    # PyObjC/AppKit screen access can abort in headless pytest contexts.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        import AppKit  # type: ignore
    except Exception:
        return None
    try:
        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            return None
        frame = screen.visibleFrame()
        x = _safe_int(getattr(frame, "origin", None).x if getattr(frame, "origin", None) is not None else 0, 0)
        y = _safe_int(getattr(frame, "origin", None).y if getattr(frame, "origin", None) is not None else 0, 0)
        w = _safe_int(getattr(frame, "size", None).width if getattr(frame, "size", None) is not None else 0, 0)
        h = _safe_int(getattr(frame, "size", None).height if getattr(frame, "size", None) is not None else 0, 0)
        if w < 800 or h < 500:
            return None
        return (x, y, w, h)
    except Exception:
        return None


def _primary_screen_rect(webview_module):
    fallback = (0, 0, 1440, 1000)
    try:
        mac_visible = _mac_visible_frame_rect()
        if mac_visible:
            return mac_visible
        screens = getattr(webview_module, "screens", None)
        if not screens:
            return fallback
        first = screens[0]
        x, y, w, h = _rect_from_screen_obj(first)
        if w < 800 or h < 500:
            return fallback
        return x, y, w, h
    except Exception:
        return fallback


def compute_startup_window_layout(webview_module):
    """Compute a deterministic side-by-side startup layout for main + panel windows."""
    sx, sy, sw, sh = _primary_screen_rect(webview_module)

    panel_w = max(320, min(420, int(round(sw * 0.26))))
    if panel_w > sw - 480:
        panel_w = max(280, sw - 480)
    panel_w = max(220, min(panel_w, max(sw - 220, 220)))

    main_w = max(sw - panel_w, 220)
    panel_w = max(sw - main_w, 220)
    total = main_w + panel_w
    if total != sw:
        panel_w = max(sw - main_w, 220)
        if main_w + panel_w > sw:
            panel_w = max(sw - main_w, 1)

    main = {"x": sx, "y": sy, "width": main_w, "height": sh}
    panel = {"x": sx + main_w, "y": sy, "width": panel_w, "height": sh}
    return {"main": main, "panel": panel}


def bootstrap_desktop_windows(api, webview_module, *, title, html_chat):
    """Create and wire desktop windows in the required order.

    Order is intentionally preserved:
    1. main window
    2. panel pre-create
    3. QC override pre-create
    4. main close-event binding
    """
    layout = compute_startup_window_layout(webview_module)
    main = dict(layout.get("main") or {})
    panel = dict(layout.get("panel") or {})
    try:
        api.panel_geom = dict(panel)
    except Exception:
        pass

    api.main_win = webview_module.create_window(
        title,
        html=html_chat,
        js_api=(getattr(api, 'main_bridge', None) or api),
        width=_safe_int(main.get("width", 1100), 1100),
        height=_safe_int(main.get("height", 1000), 1000),
        x=_safe_int(main.get("x", 0), 0),
        y=_safe_int(main.get("y", 0), 0),
    )
    # Pre-create the Panel window *before* webview.start().
    # On macOS/Cocoa, creating secondary windows from a JS->Python callback can leave
    # the JS API bridge uninitialized, causing a stuck "Loading panel..." state.
    api._create_panel()
    api._create_qc_override()
    api.main_win.events.closed += api.on_main_window_close
    return api.main_win


def run_desktop_app(
    *,
    api_factory,
    webview_module,
    genai_module,
    genai_types,
    title,
    html_chat,
):
    """Run the pywebview desktop composition root with preserved ordering."""
    validate_desktop_runtime_dependencies(webview_module, genai_module, genai_types)
    api = api_factory()
    bootstrap_desktop_windows(api, webview_module, title=title, html_chat=html_chat)
    webview_module.start(api.start_background_thread)
    return api
