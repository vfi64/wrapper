from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime


def _build_fork_parent_trace_id(data: dict) -> str | None:
    try:
        sm = data.get("session_metadata") or {}
        parent_trace_id = (
            data.get("trace_id")
            or sm.get("trace_id")
            or data.get("session_id")
            or sm.get("session_id")
            or data.get("meta_trace_id")
            or None
        )
        if parent_trace_id:
            return str(parent_trace_id)
        try:
            blob = json.dumps(
                {
                    "session_metadata": sm,
                    "provider_model_history": data.get("provider_model_history", []),
                    "history": data.get("history", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            blob = repr(data)
        h = hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"legacy_{h}"
    except Exception:
        return None


def load_log_from_path(api, path: str, *, fork: bool = False) -> dict:
    """Load a legacy chat log JSON from disk into runtime history."""
    try:
        p = str(path or "")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        hist = data.get("history") or data.get("conversation") or []
        if not isinstance(hist, list):
            return {"ok": False, "error": "history_not_list"}

        norm = []
        for msg in hist:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role == "assistant":
                role = "bot"
            elif role == "system":
                role = "system"
            norm.append({**msg, "role": role})

        api.history = norm

        if fork:
            try:
                import uuid as _uuid

                api.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + _uuid.uuid4().hex[:6]
                api.trace_id = api.session_id
                api.session_start_dt = datetime.now()
                api.session_requests = 0
                api.session_rate_limit_hits = 0
                api.session_repair_passes = 0
                api.session_csc_applied_count = 0
                api.session_guard_hits = 0
                api.session_events = []
                try:
                    api.forked_from_log_path = p
                except Exception:
                    pass
                try:
                    api.fork_parent_trace_id = _build_fork_parent_trace_id(data)
                except Exception:
                    api.fork_parent_trace_id = None
                try:
                    msg = f"Forked from chat log: {os.path.basename(p)}"
                    api.history.append({"role": "sys", "content": msg, "ts": datetime.now().isoformat()})
                except Exception:
                    pass
            except Exception:
                pass

        return {"ok": True, "history_len": len(api.history), "forked": bool(fork)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
