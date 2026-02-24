import os
from typing import Callable


def load_ui_asset_text(ui_assets_dir: str, filename: str, fallback_text: str) -> str:
    """Load optional external UI asset text with deterministic fail-open fallback."""
    try:
        path = os.path.join(ui_assets_dir, filename)
        if not os.path.isfile(path):
            return fallback_text
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return txt if txt else fallback_text
    except Exception:
        return fallback_text


def panel_asset_static_selftest_report(panel_html: str) -> dict:
    """Static sanity checks for external panel.html across legacy/current marker variants."""
    txt = panel_html or ''

    # Accept both historical and current marker names so the self-test checks
    # structure/behavioral anchors instead of one exact HTML snapshot.
    marker_groups = {
        "action_fn": ('function panelAction(', 'function run('),
        "build_ui_fn": ('function buildUI(',),
        "pywebview_bridge": ('window.pywebview',),
        "provider_select": ('id="provider"',),
        "model_select": ('id="model"',),
        "answer_language": ('id="answer-language"', 'id="anslang"'),
        "manual_test_scenario": ('id="manual-test-scenario"', 'id="manualTestScenario"'),
        "monitor_visibility": ('id="monitor-visibility"', 'id="manualTestMonitorMode"'),
        "dynamic_sections": (
            'comm-core-grid',
            "section('Comm Core'",
            'section("Comm Core"',
        ),
    }

    missing = []
    matched = {}
    for key, variants in marker_groups.items():
        hit = None
        for marker in variants:
            if marker in txt:
                hit = marker
                break
        matched[key] = hit
        if hit is None:
            missing.append(key)

    return {
        "ok": (len(missing) == 0),
        "missing": missing,
        "matched": matched,
    }


def panel_asset_static_selftest_ok(panel_html: str) -> bool:
    try:
        return bool((panel_asset_static_selftest_report(panel_html) or {}).get("ok"))
    except Exception:
        return False


def panel_runtime_selftest_payload_ok(payload) -> tuple[bool, str]:
    """Validate panel runtime-selftest callback payload from external panel.html."""
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    if payload.get("ok") is not True:
        return False, "js_report_not_ok"
    if payload.get("bridge_ping") is not True:
        return False, "bridge_ping_missing"
    if payload.get("build_ui") is not True:
        return False, "build_ui_missing"
    if payload.get("dom_ok") is not True:
        return False, "dom_markers_missing"

    try:
        loaded = (payload.get("data_loaded") is True)
    except Exception:
        loaded = False
    try:
        dyn_count = int(payload.get("dynamic_section_count", 0) or 0)
    except Exception:
        dyn_count = 0
    if loaded and dyn_count <= 0:
        return False, "loaded_ruleset_but_no_dynamic_sections"
    return True, ""


def load_panel_asset_text_s7(
    ui_assets_dir: str,
    fallback_text: str,
    *,
    print_fn: Callable[[str], None] = print,
):
    """Prefer external panel.html when static self-test passes; runtime self-test happens pre-show."""
    txt = load_ui_asset_text(ui_assets_dir, 'panel.html', '')
    if not txt:
        print_fn('[S7] panel.html asset missing/unreadable; using embedded panel.')
        return fallback_text, {"source": "embedded", "reason": "missing_asset", "static_ok": False}

    report = panel_asset_static_selftest_report(txt)
    if not report.get("ok"):
        miss = ",".join(report.get("missing") or []) or "unknown"
        print_fn(f"[S7] panel.html static self-test failed ({miss}); using embedded panel.")
        return fallback_text, {"source": "embedded", "reason": "static_selftest_failed", "static_ok": False, "report": report}

    print_fn('[S7] panel.html external asset enabled (static self-test passed; runtime self-test pending).')
    return txt, {"source": "external", "reason": "static_selftest_passed", "static_ok": True, "report": report}
