from __future__ import annotations

import panel_window_fallback as sut


def test_panel_closed_retired_event_decision_ignores_once_when_replacement_exists():
    ignore, next_n = sut.panel_closed_retired_event_decision(panel_window_exists=True, ignore_count=1)
    assert ignore is True
    assert next_n == 0


def test_panel_closed_retired_event_decision_does_not_ignore_without_replacement_window():
    ignore, next_n = sut.panel_closed_retired_event_decision(panel_window_exists=False, ignore_count=3)
    assert ignore is False
    assert next_n == 3


def test_panel_closed_retired_event_decision_coerces_invalid_counts_to_zero():
    ignore, next_n = sut.panel_closed_retired_event_decision(panel_window_exists=True, ignore_count="x")
    assert ignore is False
    assert next_n == 0


def test_panel_embedded_fallback_recreate_plan_increments_ignore_count_when_old_window_exists():
    plan = sut.panel_embedded_fallback_recreate_plan(old_window_exists=True, ignore_count=2)
    assert plan["clear_panel_window"] is True
    assert plan["panel_hidden"] is False
    assert plan["force_embedded_html"] is True
    assert plan["next_ignore_count"] == 3


def test_panel_embedded_fallback_recreate_plan_keeps_nonnegative_count_without_old_window():
    plan = sut.panel_embedded_fallback_recreate_plan(old_window_exists=False, ignore_count=-5)
    assert plan["next_ignore_count"] == 0
