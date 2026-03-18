from __future__ import annotations

from typing import Callable


def route_kind(route: dict) -> str:
    try:
        return str((route or {}).get("kind") or "").strip().lower()
    except Exception:
        return ""


def is_error_route(route: dict) -> bool:
    return route_kind(route) == "error"


def is_noop_route(route: dict) -> bool:
    return route_kind(route) == "noop"


def is_command_route(route: dict) -> bool:
    return route_kind(route) == "command"


def is_chat_route(route: dict) -> bool:
    return route_kind(route) == "chat"


def is_sci_selection_route(route: dict) -> bool:
    try:
        return bool((route or {}).get("is_sci_selection"))
    except Exception:
        return False


def route_meta(route: dict) -> dict:
    r = route if isinstance(route, dict) else {}
    return {
        "kind": route_kind(r),
        "is_command": is_command_route(r),
        "is_sci_selection": is_sci_selection_route(r),
        "is_error": is_error_route(r),
        "is_noop": is_noop_route(r),
        "standalone_only_violation": bool(r.get("standalone_only_violation")),
    }


def route_contract_ok(route: dict, *, contract_route_shape_fn: Callable | None = None) -> bool:
    """Evaluate route contract via injected checker, else conservative local checks."""
    if callable(contract_route_shape_fn):
        try:
            return bool(contract_route_shape_fn(route))
        except Exception:
            return False

    if not isinstance(route, dict):
        return False
    kind = route_kind(route)
    if kind not in {"noop", "command", "chat", "error"}:
        return False
    if kind == "command":
        return isinstance(route.get("canonical_cmd"), str) and bool(str(route.get("canonical_cmd") or "").strip())
    if kind == "chat":
        q = route.get("query_text")
        return isinstance(q, str) and bool(str(q or "").strip())
    if kind == "error":
        h = route.get("html")
        return isinstance(h, str) and bool(str(h or "").strip())
    return True


def route_audit_payload(route: dict, route_meta_obj: dict | None = None) -> dict:
    r = route if isinstance(route, dict) else {}
    m = route_meta_obj if isinstance(route_meta_obj, dict) else {}
    return {
        "kind": str(m.get("kind") or route_kind(r)),
        "is_command": bool(m.get("is_command", is_command_route(r))),
        "is_sci_selection": bool(m.get("is_sci_selection", is_sci_selection_route(r))),
        "standalone_only_violation": bool(r.get("standalone_only_violation")),
    }
