from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class StorageService:
    """Thin storage facade (Stage 5 prep, fail-soft)."""

    def append_text(self, path: str, text: str, *, encoding: str = "utf-8") -> bool:
        try:
            with open(path, "a", encoding=encoding) as f:
                f.write(text)
            return True
        except Exception:
            return False

    def write_json(
        self,
        path: str,
        payload: Any,
        *,
        indent: int = 2,
        encoding: str = "utf-8",
        ensure_ascii: bool = True,
    ) -> bool:
        try:
            parent = os.path.dirname(str(path or ""))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding=encoding) as f:
                json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
            return True
        except Exception:
            return False

    def read_json(self, path: str, *, encoding: str = "utf-8") -> Any:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except Exception:
            return None

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> bool:
        try:
            parent = os.path.dirname(str(path or ""))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding=encoding) as f:
                f.write(str(text))
            return True
        except Exception:
            return False

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str | None:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception:
            return None

    def exists(self, path: str) -> bool:
        try:
            return os.path.exists(path)
        except Exception:
            return False

    def safe_resolve_in_dir(self, base_dir: str, filename: str) -> str | None:
        """Resolve filename under base_dir; block traversal by prefix check."""
        try:
            base_abs = os.path.abspath(str(base_dir or ""))
            name = os.path.basename(str(filename or ""))
            candidate = os.path.abspath(os.path.join(base_abs, name))
            if not candidate.startswith(base_abs):
                return None
            return candidate
        except Exception:
            return None

    def list_json_filenames(self, base_dir: str, *, limit: int = 200) -> list[str]:
        """Return JSON filenames in base_dir, sorted descending. Never raises."""
        try:
            lim = int(limit) if limit is not None else 200
        except Exception:
            lim = 200
        if lim <= 0:
            lim = 200
        try:
            if not base_dir:
                return []
            if not os.path.isdir(base_dir):
                return []
            out: list[str] = []
            for name in os.listdir(base_dir):
                full = os.path.join(base_dir, name)
                if os.path.isfile(full) and str(name).lower().endswith(".json"):
                    out.append(str(name))
            out.sort(reverse=True)
            return out[:lim]
        except Exception:
            return []
