from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class ToggleComm:
    turn_on: bool


@dataclass(frozen=True)
class ToggleSCI:
    turn_on: bool


@dataclass(frozen=True)
class SetOverlay:
    value: str  # "Strict" | "Explore" | ""


@dataclass(frozen=True)
class ToggleColor:
    turn_on: bool


@dataclass(frozen=True)
class SelectProfile:
    profile: str


@dataclass(frozen=True)
class SetAnchorAuto:
    enabled: bool


@dataclass(frozen=True)
class EnterSciRecursion:
    pass


@dataclass(frozen=True)
class ActivateDynamicOneShot:
    pass


Intent = Union[
    ToggleComm,
    ToggleSCI,
    SetOverlay,
    ToggleColor,
    SelectProfile,
    SetAnchorAuto,
    EnterSciRecursion,
    ActivateDynamicOneShot,
]


def intent_from_command(cmd: str) -> Optional[Intent]:
    token = (cmd or "").strip()
    if not token:
        return None
    if token == "Comm Start":
        return ToggleComm(turn_on=True)
    if token == "Comm Stop":
        return ToggleComm(turn_on=False)
    if token == "SCI on":
        return ToggleSCI(turn_on=True)
    if token == "SCI off":
        return ToggleSCI(turn_on=False)
    if token == "Strict on":
        return SetOverlay(value="Strict")
    if token == "Strict off":
        return SetOverlay(value="")
    if token == "Explore on":
        return SetOverlay(value="Explore")
    if token == "Explore off":
        return SetOverlay(value="")
    if token == "Color on":
        return ToggleColor(turn_on=True)
    if token == "Color off":
        return ToggleColor(turn_on=False)
    if token in ("Comm Anchor on", "Anchor auto on"):
        return SetAnchorAuto(enabled=True)
    if token in ("Comm Anchor off", "Anchor auto off"):
        return SetAnchorAuto(enabled=False)
    if token == "SCI recurse":
        return EnterSciRecursion()
    if token == "Dynamic one-shot on":
        return ActivateDynamicOneShot()
    if token.startswith("Profile "):
        prof = token.split(" ", 1)[1].strip()
        if prof:
            return SelectProfile(profile=prof)
    return None
