"""
ui_panel_model.py — Deterministic panel UI normalization.

This module is intentionally small and pure:
- It does not perform IO.
- It does not call network.
- It does not mutate global state.
- It derives toggle buttons as a pure function of the provided state snapshot.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class StateSnapshot:
    comm_active: bool = False
    sci_on: bool = False          # True if SCI active or pending
    overlay: str = "off"          # "strict" | "explore" | "off"
    color_on: bool = False

def _norm_cmd(s: str) -> str:
    return (s or "").strip().lower()

def _cmd_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("cmd", "") or item.get("name", "") or "")
    return ""

def _toggle(label: str, is_on: bool, cmd_on: str, cmd_off: str, desc_on: str = "", desc_off: str = "") -> Dict[str, str]:
    """
    Contract: button shows the ACTION (inverse of state).
    - is_on True  -> label ': OFF' and command cmd_off
    - is_on False -> label ': ON'  and command cmd_on
    """
    if is_on:
        return {"name": f"{label} ⏻: OFF", "cmd": cmd_off, "desc": desc_off or f"Turn {label} off"}
    return {"name": f"{label} ⏻: ON", "cmd": cmd_on, "desc": desc_on or f"Turn {label} on"}

def _remove_cmds(items: List[Dict[str, Any]], cmds: Tuple[str, ...]) -> List[Dict[str, Any]]:
    wanted = {_norm_cmd(c) for c in cmds}
    out: List[Dict[str, Any]] = []
    for it in items:
        cmd = _norm_cmd(_cmd_of(it))
        if cmd in wanted:
            continue
        out.append(it)
    return out

def _find_first(items: List[Dict[str, Any]], cmd: str) -> Optional[Dict[str, Any]]:
    n = _norm_cmd(cmd)
    for it in items:
        if _norm_cmd(_cmd_of(it)) == n:
            return it
    return None

def normalize_panel_ui(data: Dict[str, Any], state: StateSnapshot) -> Dict[str, Any]:
    """
    Normalize the panel UI schema in `data` in a deterministic manner, without changing
    the underlying command language.
    Returns a NEW dict (does not mutate the input dict).
    """
    out = dict(data)

    # --- Comm toggle: collapse Start/Stop pair ---
    comm = list(out.get("comm") or [])
    if isinstance(comm, list) and comm:
        has_start = _find_first(comm, "Comm Start") is not None
        has_stop  = _find_first(comm, "Comm Stop") is not None
        if has_start and has_stop:
            comm = _remove_cmds(comm, ("Comm Start", "Comm Stop"))
            comm.insert(0, _toggle("Comm", state.comm_active, "Comm Start", "Comm Stop",
                                  desc_on="Start control layer", desc_off="Stop control layer"))
    out["comm"] = comm

    # --- SCI toggle + conditional tools ---
    sci = list(out.get("sci") or [])
    if isinstance(sci, list) and sci:
        has_on  = _find_first(sci, "SCI on") is not None
        has_off = _find_first(sci, "SCI off") is not None
        if has_on and has_off:
            sci = _remove_cmds(sci, ("SCI on", "SCI off"))
            sci.insert(0, _toggle("SCI", state.sci_on, "SCI on", "SCI off",
                                  desc_on="Enable SCI mode", desc_off="Disable SCI mode"))

        # show menu/recurse only if SCI is on
        if not state.sci_on:
            sci = _remove_cmds(sci, ("SCI menu", "SCI recurse"))
    out["sci"] = sci

    # --- Overlay toggles (Strict / Explore) ---
    overlays = list(out.get("overlays") or [])
    if isinstance(overlays, list) and overlays:
        strict_on = state.overlay == "strict"
        explore_on = state.overlay == "explore"

        has_s_on  = _find_first(overlays, "Strict on") is not None
        has_s_off = _find_first(overlays, "Strict off") is not None
        if has_s_on and has_s_off:
            overlays = _remove_cmds(overlays, ("Strict on", "Strict off"))
            overlays.insert(0, _toggle("Strict", strict_on, "Strict on", "Strict off",
                                       desc_on="Enable Strict overlay", desc_off="Disable Strict overlay"))

        has_e_on  = _find_first(overlays, "Explore on") is not None
        has_e_off = _find_first(overlays, "Explore off") is not None
        if has_e_on and has_e_off:
            overlays = _remove_cmds(overlays, ("Explore on", "Explore off"))
            # put Explore right after Strict toggle if present
            idx = 1 if overlays and _norm_cmd(_cmd_of(overlays[0])).startswith("strict") else 0
            overlays.insert(idx, _toggle("Explore", explore_on, "Explore on", "Explore off",
                                         desc_on="Enable Explore overlay", desc_off="Disable Explore overlay"))
    out["overlays"] = overlays

    # --- Color toggle (may live in tools in some versions, but usually overlays/tools) ---
    tools = list(out.get("tools") or [])
    if isinstance(tools, list) and tools:
        has_c_on  = _find_first(tools, "Color on") is not None
        has_c_off = _find_first(tools, "Color off") is not None
        if has_c_on and has_c_off:
            tools = _remove_cmds(tools, ("Color on", "Color off"))
            tools.insert(0, _toggle("Color", state.color_on, "Color on", "Color off",
                                    desc_on="Enable color", desc_off="Disable color"))
    out["tools"] = tools

    return out
