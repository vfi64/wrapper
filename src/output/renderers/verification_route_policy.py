from __future__ import annotations

import html
import re
from typing import Callable, Mapping


VR_MARKER_LABEL_RX = (
    r"Verification\s+Route(?:\s+Gate)?"
    r"|Source"
    r"|Measurement"
    r"|Contrast"
    r"|Web[\s\-]*Check"
    r"|Quelle"
    r"|Messung"
    r"|Kontrast"
    r"|Web[\s\-]*Pr(?:ü|ue)fung"
)


def is_verification_route_marker_line(raw_line: str) -> bool:
    """Return True when a line is a pure verification-route display marker."""
    try:
        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(raw_line or "")))
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain:
            return False
        return bool(
            re.match(
                rf"(?i)^(?:[-*•]\s*)?(?:{VR_MARKER_LABEL_RX})\s*:?\s*.*$",
                plain,
            )
        )
    except Exception:
        return False


def strip_verification_route_display_lines(text: str) -> str:
    """Hide noisy verification-route marker lines from chat display."""
    try:
        if not text:
            return text
        out_lines = []
        for ln in str(text).splitlines():
            if is_verification_route_marker_line(ln):
                continue
            out_lines.append(str(ln or ""))
        return "\n".join(out_lines)
    except Exception:
        return text


def parse_bool_like(value, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "on", "yes", "y"}:
        return True
    if s in {"0", "false", "off", "no", "n"}:
        return False
    return bool(default)


def resolve_hide_verification_route_lines(
    *,
    config: Mapping | None = None,
    provider: str = "gemini",
) -> bool:
    """Central display-policy decision for Verification Route marker visibility."""
    conf = config if isinstance(config, Mapping) else {}
    pid = str(provider or "gemini").strip().lower() or "gemini"
    provs = conf.get("providers") if isinstance(conf, Mapping) else {}
    pconf = provs.get(pid) if isinstance(provs, Mapping) else {}
    raw = None
    if isinstance(pconf, Mapping):
        raw = pconf.get("hide_verification_route_lines")
    if raw is None and isinstance(conf, Mapping):
        raw = conf.get("hide_verification_route_lines")
    return parse_bool_like(raw, default=False)


def apply_verification_route_display_policy(
    text: str,
    *,
    config: Mapping | None = None,
    provider: str = "gemini",
    strip_verification_route_display_lines_fn: Callable | None = None,
) -> str:
    """Apply central Verification Route display-policy to render text (fail-soft)."""
    out = str(text or "")
    if not out:
        return out
    if not resolve_hide_verification_route_lines(config=config, provider=provider):
        return out
    strip_fn = (
        strip_verification_route_display_lines_fn
        if callable(strip_verification_route_display_lines_fn)
        else strip_verification_route_display_lines
    )
    try:
        stripped = strip_fn(out)
    except Exception:
        return out
    if stripped is None:
        return out
    return str(stripped)


def update_hide_verification_route_lines_config(
    config: dict,
    *,
    enabled=None,
    scope: str = "provider",
    provider: str = "gemini",
) -> dict:
    """Mutate config dict for verification-route display policy toggles.

    Returns schema:
    - ok: bool
    - error: str (only if ok=False)
    - scope/provider/value
    """
    if not isinstance(config, dict):
        return {"ok": False, "error": "config_unavailable"}

    s = str(scope or "provider").strip().lower()
    if s not in ("provider", "root"):
        s = "provider"

    p = str(provider or "gemini").strip().lower() or "gemini"

    val = enabled
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("", "none", "null", "unset", "clear"):
            val = None
        elif v in ("1", "true", "on", "yes", "y"):
            val = True
        elif v in ("0", "false", "off", "no", "n"):
            val = False
        else:
            return {"ok": False, "error": "invalid_enabled_value"}

    if val is not None and not isinstance(val, bool):
        return {"ok": False, "error": "invalid_enabled_value"}

    if s == "root":
        if val is None:
            config.pop("hide_verification_route_lines", None)
        else:
            config["hide_verification_route_lines"] = bool(val)
    else:
        provs = config.setdefault("providers", {})
        if not isinstance(provs, dict):
            return {"ok": False, "error": "providers_config_invalid"}
        pconf = provs.setdefault(p, {})
        if not isinstance(pconf, dict):
            return {"ok": False, "error": "provider_config_invalid"}
        if val is None:
            pconf.pop("hide_verification_route_lines", None)
        else:
            pconf["hide_verification_route_lines"] = bool(val)

    return {
        "ok": True,
        "scope": s,
        "provider": p,
        "value": (None if val is None else bool(val)),
    }
