from __future__ import annotations

import html
from typing import Callable


_ALLOWED_ROUTE_KINDS = {"noop", "command", "chat", "error"}


def _fallback_route(raw_txt: str) -> dict:
    txt = str(raw_txt or "").strip()
    if not txt:
        return {"kind": "noop"}
    return {"kind": "chat", "query_text": txt}


def normalize_route_shape(route: dict, *, raw_txt: str = "") -> dict:
    """Normalize a route object to the legacy contract shape."""
    if not isinstance(route, dict):
        return _fallback_route(raw_txt)

    kind = str(route.get("kind") or "").strip().lower()
    if kind not in _ALLOWED_ROUTE_KINDS:
        return {
            "kind": "error",
            "html": (
                '<div class="csc-warning" style="background:#fee; border-color:#c00; color:#a00;">'
                "<b>CONTROL LAYER BLOCK:</b><br>"
                + html.escape(f"Invalid route kind: {kind or 'n/a'}")
                + "</div>"
            ),
        }

    out = dict(route)
    out["kind"] = kind
    if kind == "command":
        cmd = str(out.get("canonical_cmd") or "").strip()
        if not cmd:
            return _fallback_route(raw_txt)
        out["canonical_cmd"] = cmd
        return out

    if kind == "chat":
        q = str(out.get("query_text") or out.get("txt") or "").strip()
        if not q:
            return _fallback_route(raw_txt)
        out["query_text"] = q
        return out

    if kind == "error":
        h = str(out.get("html") or "").strip()
        if not h:
            return _fallback_route(raw_txt)
        out["html"] = h
        return out

    return out


def resolve_input(
    raw_txt: str,
    state,
    api_instance,
    *,
    gov_manager=None,
    route_input_fn: Callable | None = None,
) -> dict:
    """Resolve input via injected legacy router and normalize shape deterministically."""
    if callable(route_input_fn):
        try:
            route = route_input_fn(raw_txt, state, api_instance, gov_manager=gov_manager)
            return normalize_route_shape(route, raw_txt=raw_txt)
        except Exception:
            return _fallback_route(raw_txt)
    return _fallback_route(raw_txt)


def contract_route_shape(route: dict) -> bool:
    """Best-effort contract check for resolver output."""
    if not isinstance(route, dict):
        return False
    kind = route.get("kind")
    if kind not in _ALLOWED_ROUTE_KINDS:
        return False
    if kind == "command":
        return isinstance(route.get("canonical_cmd"), str) and bool(route.get("canonical_cmd"))
    if kind == "chat":
        return isinstance(route.get("query_text"), str) and bool(route.get("query_text"))
    if kind == "error":
        return isinstance(route.get("html"), str) and bool(route.get("html"))
    return True

