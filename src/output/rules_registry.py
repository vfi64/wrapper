from __future__ import annotations

import re


QC_DIMENSIONS_EN = ("Clarity", "Brevity", "Evidence", "Empathy", "Consistency", "Neutrality")
QC_DIMENSIONS_DE_PRIMARY = ("Klarheit", "Kürze", "Evidenz", "Empathie", "Konsistenz", "Neutralität")
QC_DIMENSIONS_DE_ASCII = ("Klarheit", "Kuerze", "Evidenz", "Empathie", "Konsistenz", "Neutralitaet")


STATUS_SCAFFOLD_KEYS = (
    "active profile",
    "profile",
    "aktives profil",
    "profil",
    "overlay",
    "sci",
    "control layer",
    "steuerungsebene",
    "qc",
    "cgi",
    "color",
    "farbe",
    "comm",
)


def qc_probe_is_complete(text: str) -> bool:
    """True when all canonical QC dimensions are present in EN or DE form."""
    probe = re.sub(r"\s+", " ", str(text or "")).strip()
    if not probe:
        return False
    if all(f"{label} " in probe for label in QC_DIMENSIONS_EN):
        return True
    if all(f"{label} " in probe for label in QC_DIMENSIONS_DE_PRIMARY):
        return True
    if all(f"{label} " in probe for label in QC_DIMENSIONS_DE_ASCII):
        return True
    return False


def looks_like_status_scaffold_text(text: str) -> bool:
    """Conservative status-scaffold detector for output-level guards."""
    low = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not low:
        return False
    has_profile = any(k in low for k in ("active profile:", "profile:", "aktives profil:", "profil:"))
    has_matrix = any(k in low for k in ("overlay", "sci", "color", "farbe", "control layer", "steuerungsebene", "qc", "cgi"))
    return has_profile and has_matrix

