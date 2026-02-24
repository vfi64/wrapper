"""S8 bootstrap helpers for the pywebview composition root.

Behavior-preserving extraction from the monolithic launcher path.
"""

from __future__ import annotations


def validate_desktop_runtime_dependencies(webview_module, genai_module, genai_types):
    if webview_module is None:
        raise SystemExit('pywebview is required. Install with: pip install pywebview')
    if genai_module is None or genai_types is None:
        raise SystemExit('google-genai is required. Install with: pip install google-genai')


def bootstrap_desktop_windows(api, webview_module, *, title, html_chat):
    """Create and wire desktop windows in the required order.

    Order is intentionally preserved:
    1. main window
    2. panel pre-create
    3. QC override pre-create
    4. main close-event binding
    """
    api.main_win = webview_module.create_window(
        title,
        html=html_chat,
        js_api=(getattr(api, 'main_bridge', None) or api),
        width=1100,
        height=1000,
        x=0,
        y=0,
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
