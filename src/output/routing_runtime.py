from __future__ import annotations

from typing import Callable


def resolve_route_context(
    raw_txt: str,
    state,
    api_instance,
    *,
    gov_manager=None,
    route_input_fn: Callable | None = None,
    output_resolver_mod=None,
    output_dispatcher_mod=None,
) -> tuple[dict, dict]:
    """Resolve route and derive deterministic route flags (fail-soft)."""
    route = None
    try:
        if output_resolver_mod is not None and hasattr(output_resolver_mod, "resolve_input"):
            route = output_resolver_mod.resolve_input(
                raw_txt,
                state,
                api_instance,
                gov_manager=gov_manager,
                route_input_fn=route_input_fn,
            )
    except Exception:
        route = None

    if not isinstance(route, dict):
        try:
            if callable(route_input_fn):
                route = route_input_fn(raw_txt, state, api_instance)
        except Exception:
            txt = str(raw_txt or "").strip()
            route = {"kind": "chat", "query_text": txt} if txt else {"kind": "noop"}

    if not isinstance(route, dict):
        txt = str(raw_txt or "").strip()
        route = {"kind": "chat", "query_text": txt} if txt else {"kind": "noop"}

    try:
        if output_resolver_mod is not None and hasattr(output_resolver_mod, "normalize_route_shape"):
            route = output_resolver_mod.normalize_route_shape(route, raw_txt=str(raw_txt or ""))
    except Exception:
        pass

    route_kind = str(route.get("kind") or "")
    is_command = bool(route_kind == "command")
    is_sci_selection = bool(route.get("is_sci_selection"))
    is_error = bool(route_kind == "error")
    is_noop = bool(route_kind == "noop")
    if output_dispatcher_mod is not None and hasattr(output_dispatcher_mod, "route_meta"):
        try:
            meta = output_dispatcher_mod.route_meta(route) or {}
            route_kind = str(meta.get("kind") or route_kind)
            is_command = bool(meta.get("is_command", is_command))
            is_sci_selection = bool(meta.get("is_sci_selection", is_sci_selection))
            is_error = bool(meta.get("is_error", is_error))
            is_noop = bool(meta.get("is_noop", is_noop))
        except Exception:
            pass
    elif output_dispatcher_mod is not None:
        try:
            route_kind = output_dispatcher_mod.route_kind(route)
            is_command = output_dispatcher_mod.is_command_route(route)
            is_sci_selection = output_dispatcher_mod.is_sci_selection_route(route)
            is_error = output_dispatcher_mod.is_error_route(route)
            is_noop = output_dispatcher_mod.is_noop_route(route)
        except Exception:
            pass

    route_meta = {
        "kind": str(route_kind or ""),
        "is_command": bool(is_command),
        "is_sci_selection": bool(is_sci_selection),
        "is_error": bool(is_error),
        "is_noop": bool(is_noop),
        "standalone_only_violation": bool(route.get("standalone_only_violation")),
    }
    return route, route_meta


def route_contract_ok(route: dict, *, output_resolver_mod=None, output_dispatcher_mod=None, local_contract_fn: Callable | None = None) -> bool:
    """Evaluate route contract via dispatcher/resolver contract helpers."""
    contract_fn = None
    try:
        if output_resolver_mod is not None and hasattr(output_resolver_mod, "contract_route_shape"):
            contract_fn = output_resolver_mod.contract_route_shape
    except Exception:
        contract_fn = None

    if contract_fn is None and callable(local_contract_fn):
        contract_fn = local_contract_fn

    if output_dispatcher_mod is not None and hasattr(output_dispatcher_mod, "route_contract_ok"):
        try:
            return bool(output_dispatcher_mod.route_contract_ok(route, contract_route_shape_fn=contract_fn))
        except Exception:
            pass

    if callable(contract_fn):
        try:
            return bool(contract_fn(route))
        except Exception:
            return False
    return False


def route_audit_payload(route: dict, route_meta: dict | None = None, *, output_dispatcher_mod=None) -> dict:
    """Build deterministic route audit payload."""
    meta = route_meta if isinstance(route_meta, dict) else {}
    payload = {
        "kind": str(meta.get("kind") or str((route or {}).get("kind") or "")),
        "is_command": bool(meta.get("is_command", False)),
        "is_sci_selection": bool(meta.get("is_sci_selection", False)),
        "standalone_only_violation": bool((route or {}).get("standalone_only_violation")),
    }
    if output_dispatcher_mod is not None and hasattr(output_dispatcher_mod, "route_audit_payload"):
        try:
            payload = dict(payload)
            payload.update(output_dispatcher_mod.route_audit_payload(route, meta) or {})
        except Exception:
            try:
                payload.update(output_dispatcher_mod.route_audit_payload(route) or {})
            except Exception:
                pass
    payload["standalone_only_violation"] = bool((route or {}).get("standalone_only_violation"))
    return payload
