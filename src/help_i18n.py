from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SUPPORTED_LANGS = ("de", "en")


def normalize_lang(lang: str | None) -> str:
    raw = str(lang or "").strip().lower()
    if raw.startswith("de"):
        return "de"
    if raw.startswith("en"):
        return "en"
    return "de"


def _default_payload(lang: str) -> dict[str, Any]:
    if lang == "en":
        return {
            "button_label": "? Help",
            "title": "Comm-SCI Help",
            "subtitle": "Quickstart, commands, SCI and troubleshooting",
            "close_label": "Close",
            "sections": [
                {
                    "id": "quickstart",
                    "title": "Quickstart",
                    "items": [
                        "Load a ruleset via the folder icon.",
                        "Open panel, choose provider/model, then connect.",
                        "Use 'Comm Start' to initialize deterministic control mode.",
                        "Ask your content question after profile/SCI setup.",
                    ],
                },
                {
                    "id": "commands",
                    "title": "Core Commands",
                    "items": [
                        "Comm Start / Comm Stop: enable or disable governance mode.",
                        "Comm State / Comm Config / Comm Anchor: inspect active runtime state.",
                        "Profile <name>: switch profile (for example Expert, Briefing, Sandbox).",
                        "SCI on + A-H: activate SCI variant workflow.",
                    ],
                },
                {
                    "id": "sci",
                    "title": "SCI Workflow",
                    "items": [
                        "SCI traces are required when SCI mode is active.",
                        "Choose the variant letter A-H after 'SCI on' or 'SCI menu'.",
                        "Use 'SCI off' to return to normal output mode.",
                    ],
                },
                {
                    "id": "troubleshooting",
                    "title": "Troubleshooting",
                    "items": [
                        "If output format drifts: run 'Comm Anchor'.",
                        "If commands seem ignored: check 'Comm State'.",
                        "If provider errors occur: verify API key and selected model in panel.",
                    ],
                },
            ],
            "footer": "Tip: Press F1 to open this help at any time.",
        }

    return {
        "button_label": "? Hilfe",
        "title": "Comm-SCI Hilfe",
        "subtitle": "Quickstart, Kommandos, SCI und Troubleshooting",
        "close_label": "Schliessen",
        "sections": [
            {
                "id": "quickstart",
                "title": "Quickstart",
                "items": [
                    "Ruleset ueber das Ordner-Symbol laden.",
                    "Panel oeffnen, Provider/Modell waehlen, dann verbinden.",
                    "Mit 'Comm Start' den deterministischen Control-Modus aktivieren.",
                    "Nach Profil/SCI-Setup die Inhaltsfrage stellen.",
                ],
            },
            {
                "id": "commands",
                "title": "Kern-Kommandos",
                "items": [
                    "Comm Start / Comm Stop: Governance-Modus aktivieren oder beenden.",
                    "Comm State / Comm Config / Comm Anchor: aktiven Runtime-Zustand pruefen.",
                    "Profile <name>: Profil wechseln (z. B. Expert, Briefing, Sandbox).",
                    "SCI on + A-H: SCI-Variantenworkflow aktivieren.",
                ],
            },
            {
                "id": "sci",
                "title": "SCI-Workflow",
                "items": [
                    "Bei aktivem SCI-Modus sind SCI-Traces verpflichtend.",
                    "Nach 'SCI on' oder 'SCI menu' den Variantenbuchstaben A-H waehlen.",
                    "Mit 'SCI off' zur normalen Ausgabe zurueckkehren.",
                ],
            },
            {
                "id": "troubleshooting",
                "title": "Troubleshooting",
                "items": [
                    "Bei Format-Drift: 'Comm Anchor' ausfuehren.",
                    "Wenn Kommandos ignoriert wirken: 'Comm State' pruefen.",
                    "Bei Providerfehlern: API-Key und Modell im Panel kontrollieren.",
                ],
            },
        ],
        "footer": "Tipp: Mit F1 kann diese Hilfe jederzeit geoeffnet werden.",
    }


def _sanitize_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    sections_raw = data.get("sections")
    if not isinstance(sections_raw, list) or not sections_raw:
        return None

    sections: list[dict[str, Any]] = []
    for item in sections_raw:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        items_raw = item.get("items")
        if not sid or not title or not isinstance(items_raw, list):
            continue
        points = [str(x).strip() for x in items_raw if str(x or "").strip()]
        if not points:
            continue
        sections.append({"id": sid, "title": title, "items": points})

    if not sections:
        return None

    button_label = str(data.get("button_label") or "").strip()
    title = str(data.get("title") or "").strip()
    close_label = str(data.get("close_label") or "").strip()
    subtitle = str(data.get("subtitle") or "").strip()
    footer = str(data.get("footer") or "").strip()

    if not button_label or not title or not close_label:
        return None

    out = {
        "button_label": button_label,
        "title": title,
        "close_label": close_label,
        "sections": sections,
    }
    if subtitle:
        out["subtitle"] = subtitle
    if footer:
        out["footer"] = footer
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    return _sanitize_payload(data)


def load_help_payload(lang: str | None = None, base_dir: str | Path | None = None) -> dict[str, Any]:
    use_lang = normalize_lang(lang)
    root = Path(base_dir).resolve() if base_dir else (Path(__file__).resolve().parent / "i18n")

    primary = _load_json(root / f"help.{use_lang}.json")
    if primary is not None:
        return primary

    fallback = _load_json(root / "help.en.json")
    if fallback is not None:
        return fallback

    return _default_payload(use_lang)
