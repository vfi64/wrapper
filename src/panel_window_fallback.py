from __future__ import annotations


def _coerce_ignore_count(value) -> int:
    try:
        n = int(value or 0)
    except Exception:
        n = 0
    return max(0, n)


def panel_closed_retired_event_decision(*, panel_window_exists: bool, ignore_count) -> tuple[bool, int]:
    """Return (ignore_event, next_ignore_count) for a delayed closed callback of a retired panel."""
    n = _coerce_ignore_count(ignore_count)
    if bool(panel_window_exists) and n > 0:
        return True, (n - 1)
    return False, n


def panel_embedded_fallback_recreate_plan(*, old_window_exists: bool, ignore_count) -> dict:
    """Prepare state changes before recreating the panel with embedded fallback HTML."""
    n = _coerce_ignore_count(ignore_count)
    if bool(old_window_exists):
        n += 1
    return {
        "clear_panel_window": True,
        "panel_hidden": False,
        "force_embedded_html": True,
        "next_ignore_count": n,
    }
