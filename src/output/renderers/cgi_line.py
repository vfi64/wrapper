from __future__ import annotations

from typing import Sequence


def get_cgi_ui_texts(*, lang: str = "de") -> dict[str, str]:
    use_en = str(lang or "").strip().lower().startswith("en")
    if use_en:
        return {
            "saved": "CGI saved: C={c} · I={i} · E={e}",
            "applied": "CGI applied (one-shot): C={c} · I={i} · E={e}",
            "no_prompt": "No previous content question available for repeat.",
            "invalid": "Please provide values from 0 to 3 for all three CGI criteria.",
            "repeat_failed": "CGI repeat could not be executed.",
        }
    return {
        "saved": "CGI gespeichert: K={c} · E={i} · F={e}",
        "applied": "CGI angewendet (one-shot): K={c} · E={i} · F={e}",
        "no_prompt": "Keine vorherige Inhaltsfrage fuer die Wiederholung vorhanden.",
        "invalid": "Bitte fuer alle drei CGI-Kriterien Werte von 0 bis 3 angeben.",
        "repeat_failed": "CGI-Wiederholung konnte nicht ausgefuehrt werden.",
    }


def render_cgi_feedback_block(*, user_feedback_triplet: str = "", process_feedback: str = "") -> str:
    rows = []
    user_val = str(user_feedback_triplet or "").strip()
    process_val = str(process_feedback or "").strip()
    if user_val:
        rows.append(f"User feedback triplet (CGI): {user_val}")
    if process_val:
        rows.append(f"Process CGI feedback: {process_val}")
    if not rows:
        return ""
    return "[CGI Feedback]\n" + "\n".join(rows)


def render_cgi_constraints_block(constraint_lines: Sequence[str] | None = None) -> str:
    rows = [str(line or "").strip() for line in list(constraint_lines or []) if str(line or "").strip()]
    if not rows:
        return ""
    return "[CGI One-Shot Rewrite Constraints]\n" + "\n".join(rows)
