from __future__ import annotations

from typing import Callable, Tuple


def panel_bootstrap_initial_state(source: str, *, reason: str = "") -> dict:
    src = str(source or "embedded")
    return {
        "status": "idle",
        "source": src,
        "reason": str(reason or ""),
        "created_at": None,
        "reported_at": None,
    }


def panel_bootstrap_probe_state(source: str, *, now_iso: str | None) -> dict:
    src = str(source or "embedded")
    return {
        "status": ("pending" if src == "external" else "skipped"),
        "source": src,
        "reason": "",
        "created_at": now_iso,
        "reported_at": None,
    }


def panel_bootstrap_closed_state(source: str) -> dict:
    return panel_bootstrap_initial_state(source, reason="window_closed")


def panel_bootstrap_ensure_state(state, *, default_source: str = "embedded") -> dict:
    if isinstance(state, dict):
        return state
    return panel_bootstrap_initial_state(default_source)


def panel_bootstrap_accept_report(
    state,
    payload,
    *,
    validate_report: Callable[[object], Tuple[bool, str]],
    now_iso: str | None,
) -> tuple[dict, dict]:
    st = panel_bootstrap_ensure_state(state)
    if str(st.get("source") or "embedded") != "external":
        return st, {"accepted": False, "ignored": True, "reason": "panel_source_not_external"}

    ok, why = validate_report(payload)
    st["status"] = "passed" if ok else "failed"
    st["reason"] = ("" if ok else str(why or "invalid_runtime_selftest"))
    st["reported_at"] = now_iso
    return st, {"accepted": True, "runtime_ok": bool(ok), "reason": ("" if ok else str(why or ""))}


def panel_bootstrap_mark_failed_for_fallback(
    state,
    *,
    reason: str = "runtime_selftest_failed",
    now_iso: str | None,
) -> dict:
    st = panel_bootstrap_ensure_state(state)
    st["status"] = "failed"
    st["reason"] = str(reason or "runtime_selftest_failed")
    st["source"] = "external"
    st["reported_at"] = now_iso
    return st


def panel_bootstrap_is_runtime_ready(state) -> bool:
    if not isinstance(state, dict):
        return True
    if str(state.get("source") or "embedded") != "external":
        return True
    return str(state.get("status") or "") == "passed"


def panel_bootstrap_fallback_reason(state) -> str | None:
    if not isinstance(state, dict):
        return None
    if str(state.get("source") or "embedded") != "external":
        return None
    if str(state.get("status") or "") == "passed":
        return None

    reason = str(state.get("reason") or "").strip()
    if reason:
        return reason
    if str(state.get("status") or "") == "pending":
        return "runtime_selftest_timeout"
    return "runtime_selftest_failed"


def panel_bootstrap_timeout_seconds(value, *, default: float = 2.5) -> float:
    try:
        out = float(value if value is not None else default)
    except Exception:
        out = float(default)
    try:
        out = max(0.0, float(out))
    except Exception:
        out = float(default)
    return out
