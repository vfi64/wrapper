from __future__ import annotations

import os
import pathlib
from typing import Any


def _normalize_limit(limit: Any) -> int:
    try:
        lim = int(limit) if limit is not None else 200
    except Exception:
        lim = 200
    if lim <= 0:
        lim = 200
    return lim


def list_chat_logs(api: Any, *, limit: int = 200, chat_log_dir: str = "") -> dict:
    """List available chat logs from the configured chat-log directory."""
    lim = _normalize_limit(limit)
    try:
        base = str(chat_log_dir or "")
        if not base:
            return {"ok": True, "logs": []}

        svc = getattr(api, "storage_service", None)
        if svc is not None and hasattr(svc, "list_json_filenames"):
            return {"ok": True, "logs": svc.list_json_filenames(base, limit=lim)}

        p = pathlib.Path(base)
        if not p.exists() or not p.is_dir():
            return {"ok": True, "logs": []}

        files = []
        for f in p.iterdir():
            if f.is_file() and f.name.lower().endswith(".json"):
                files.append(f.name)
        files.sort(reverse=True)
        if len(files) > lim:
            files = files[:lim]
        return {"ok": True, "logs": files}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "logs": []}


def _resolve_chat_log_candidate(api: Any, *, base_dir: str, filename: str) -> tuple[str | None, str | None]:
    name = str(filename or "").strip()
    if not name:
        return None, "missing_filename"
    if os.path.basename(name) != name:
        return None, "path_traversal_blocked"

    base_abs = os.path.abspath(str(base_dir or ""))
    if not base_abs:
        return None, "chat_log_dir_missing"
    base_prefix = base_abs + os.sep

    svc = getattr(api, "storage_service", None)
    candidate = None
    if svc is not None and hasattr(svc, "safe_resolve_in_dir"):
        try:
            candidate = svc.safe_resolve_in_dir(base_abs, name)
        except Exception:
            candidate = None

    if not candidate:
        candidate = os.path.join(base_abs, name)

    cand_abs = os.path.abspath(str(candidate))
    if not (cand_abs == base_abs or cand_abs.startswith(base_prefix)):
        return None, "path_traversal_blocked"
    return cand_abs, None


def load_chat_log(api: Any, *, filename: str, fork: bool = True, chat_log_dir: str = "") -> dict:
    """Load a chat log by filename from the configured chat-log directory."""
    try:
        base = str(chat_log_dir or "")
        if not base:
            return {"ok": False, "error": "chat_log_dir_missing"}

        candidate, err = _resolve_chat_log_candidate(api, base_dir=base, filename=filename)
        if err:
            return {"ok": False, "error": err}
        if not candidate:
            return {"ok": False, "error": "file_not_found"}

        svc = getattr(api, "storage_service", None)
        if svc is not None and hasattr(svc, "exists"):
            exists = bool(svc.exists(candidate))
        else:
            exists = os.path.exists(candidate)
        if not exists:
            return {"ok": False, "error": "file_not_found"}

        fn = getattr(api, "load_log_from_path", None)
        if callable(fn):
            return fn(candidate, fork=bool(fork))
        return {"ok": False, "error": "load_log_from_path_unavailable"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
