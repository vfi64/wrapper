from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIX_PATH = SRC / "Comm-SCI-Control-App.py"
JSON_PATH = ROOT / "JSON" / "Comm-SCI-v20.0.3.json"


def _load_fix_module():
    spec = importlib.util.spec_from_file_location(FIX_PATH.stem, FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _prime_module_gov(mod):
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    mod.gov.data = data
    mod.gov.loaded = True
    try:
        mod.gov.filepath = str(JSON_PATH)
    except Exception:
        pass

    try:
        cfg = getattr(mod, "cfg", None)
        conf = getattr(cfg, "config", None)
        if isinstance(conf, dict):
            conf["active_provider"] = "gemini"
            conf.setdefault("model", "gemini-2.0-flash")
    except Exception:
        pass


def _set_test_log_dirs(mod, tmp_path):
    mod.PROJECT_DIR = str(tmp_path)
    mod.CONFIG_DIR = str(tmp_path / "Config")
    mod.LOGS_DIR = str(tmp_path / "Logs")
    mod.AUDIT_LOG_DIR = str(tmp_path / "Logs" / "Audit")
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    for d in [mod.CONFIG_DIR, mod.LOGS_DIR, mod.AUDIT_LOG_DIR, mod.CHAT_LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def test_stage0_golden_run_checklist_exists_and_is_actionable():
    mod = _load_fix_module()
    items = list(getattr(mod, "GOLDEN_RUN_STUFE0", []) or [])

    assert len(items) >= 5
    text = " | ".join(str(x).lower() for x in items)
    assert "comm start" in text
    assert "profile switch" in text
    assert "comm audit" in text
    assert "clear chat" in text


def test_stage0_log_event_is_callable_and_records_event(tmp_path):
    mod = _load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    api.log_event("stage0_test", {"k": "v"})

    ev = (getattr(api, "session_events", []) or [])[-1]
    assert ev.get("type") == "stage0_test"
    data = ev.get("data") or {}
    assert data.get("k") == "v"


def test_stage0_provider_switch_is_visible_in_event_stream(tmp_path):
    mod = _load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    api.set_provider("openrouter")
    events = list(getattr(api, "session_events", []) or [])

    assert any(
        (e.get("type") == "provider" and isinstance(e.get("data"), dict) and e["data"].get("event") == "provider_switch")
        for e in events
    )
