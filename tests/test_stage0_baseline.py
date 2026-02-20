import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIX_PATH = ROOT / "src" / "Comm-SCI-Control-App.py"


def load_fix_module():
    spec = importlib.util.spec_from_file_location(FIX_PATH.stem, FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_stage0_golden_run_checklist_exists_and_is_actionable():
    mod = load_fix_module()
    checklist = getattr(mod, "GOLDEN_RUN_STUFE0", None)
    assert isinstance(checklist, list) and checklist
    joined = " | ".join(str(x) for x in checklist)
    assert "Comm Start" in joined
    assert "Profile switch" in joined
    assert "Comm Audit exports" in joined
    assert "QC overrides" in joined


def test_stage0_log_event_is_callable_and_records_event():
    mod = load_fix_module()
    api = mod.Api()
    api.session_events = []
    api.log_event("stage0_smoke", {"ok": True})
    assert isinstance(api.session_events, list) and api.session_events
    last = api.session_events[-1]
    assert last.get("type") == "stage0_smoke"
    assert isinstance(last.get("data"), dict) and last["data"].get("ok") is True


def test_stage0_provider_switch_is_visible_in_event_stream():
    mod = load_fix_module()
    api = mod.Api()
    api.session_events = []
    api.main_win = None
    api.panel_win = None

    api.set_provider("openrouter")
    found = False
    for ev in api.session_events:
        if ev.get("type") != "provider":
            continue
        data = ev.get("data") or {}
        if isinstance(data, dict) and data.get("event") == "provider_switch":
            found = True
            assert "old_provider" in data
            assert "new_provider" in data
            break
    assert found, "Expected provider_switch event in session_events."
