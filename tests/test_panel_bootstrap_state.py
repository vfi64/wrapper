from __future__ import annotations

import panel_bootstrap_state as sut


def test_panel_bootstrap_probe_state_marks_external_pending_and_embedded_skipped():
    ext = sut.panel_bootstrap_probe_state("external", now_iso="2026-02-24T10:00:00")
    emb = sut.panel_bootstrap_probe_state("embedded", now_iso="2026-02-24T10:00:01")

    assert ext["status"] == "pending"
    assert ext["source"] == "external"
    assert ext["created_at"] == "2026-02-24T10:00:00"
    assert emb["status"] == "skipped"
    assert emb["source"] == "embedded"


def test_panel_bootstrap_accept_report_ignores_non_external_source():
    state = sut.panel_bootstrap_initial_state("embedded")

    state_out, result = sut.panel_bootstrap_accept_report(
        state,
        payload={"ok": True},
        validate_report=lambda payload: (True, ""),
        now_iso="2026-02-24T10:00:00",
    )

    assert state_out is state
    assert result == {"accepted": False, "ignored": True, "reason": "panel_source_not_external"}
    assert state["status"] == "idle"


def test_panel_bootstrap_accept_report_sets_failed_state_and_reason():
    state = sut.panel_bootstrap_probe_state("external", now_iso="2026-02-24T10:00:00")

    state_out, result = sut.panel_bootstrap_accept_report(
        state,
        payload={"x": 1},
        validate_report=lambda payload: (False, "dom_markers_missing"),
        now_iso="2026-02-24T10:00:02",
    )

    assert state_out is state
    assert state["status"] == "failed"
    assert state["reason"] == "dom_markers_missing"
    assert state["reported_at"] == "2026-02-24T10:00:02"
    assert result == {"accepted": True, "runtime_ok": False, "reason": "dom_markers_missing"}


def test_panel_bootstrap_fallback_reason_defaults_to_timeout_for_pending_external_state():
    state = sut.panel_bootstrap_probe_state("external", now_iso="2026-02-24T10:00:00")
    assert sut.panel_bootstrap_fallback_reason(state) == "runtime_selftest_timeout"
    assert sut.panel_bootstrap_is_runtime_ready(state) is False


def test_panel_bootstrap_mark_failed_for_fallback_and_closed_state():
    state = sut.panel_bootstrap_probe_state("external", now_iso="2026-02-24T10:00:00")
    out = sut.panel_bootstrap_mark_failed_for_fallback(
        state,
        reason="runtime_selftest_failed",
        now_iso="2026-02-24T10:00:03",
    )
    assert out is state
    assert state["status"] == "failed"
    assert state["reason"] == "runtime_selftest_failed"
    assert state["reported_at"] == "2026-02-24T10:00:03"

    closed = sut.panel_bootstrap_closed_state("external")
    assert closed["status"] == "idle"
    assert closed["reason"] == "window_closed"


def test_panel_bootstrap_timeout_seconds_coerces_invalid_and_negative_values():
    assert sut.panel_bootstrap_timeout_seconds(None, default=2.5) == 2.5
    assert sut.panel_bootstrap_timeout_seconds("abc", default=2.5) == 2.5
    assert sut.panel_bootstrap_timeout_seconds(-1) == 0.0
