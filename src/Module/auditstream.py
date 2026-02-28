import os
import json
import datetime
from typing import Optional, Dict, Any

_jsonl_date_cache: Optional[str] = None
_jsonl_path_cache: Optional[str] = None

def get_audit_stream_jsonl_path(now: datetime.datetime, *, audit_log_dir: str) -> str:
    """Return JSONL audit stream path for current day (best-effort)."""
    global _jsonl_date_cache, _jsonl_path_cache
    try:
        day = now.strftime("%Y%m%d")
    except Exception:
        day = "unknown"
    if _jsonl_date_cache != day or not _jsonl_path_cache:
        _jsonl_date_cache = day
        try:
            _jsonl_path_cache = os.path.join(audit_log_dir, f"AuditStream_{day}.jsonl")
        except Exception:
            _jsonl_path_cache = None
    return _jsonl_path_cache or ""

def append_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    """Append one JSON object as a single JSONL line. Must never raise."""
    try:
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return

def build_jsonl_meta(*, wrapper_name: Optional[str], wrapper_version: Optional[str],
                     ruleset_version: Optional[str]=None, provider: Optional[str]=None,
                     model: Optional[str]=None, language_policy_mode: Optional[str]=None) -> Dict[str, Any]:
    """Best-effort metadata (no secrets)."""
    meta: Dict[str, Any] = {
        "wrapper_name": wrapper_name,
        "wrapper_version": wrapper_version,
    }
    if ruleset_version:
        meta["ruleset_version"] = ruleset_version
    if provider:
        meta["provider"] = provider
    if model:
        meta["model"] = model
    if language_policy_mode:
        meta["language_policy_mode"] = language_policy_mode
    return meta
