from __future__ import annotations

import html
import json
from typing import Any, Callable


def _as_bot_html(
    content: str,
    *,
    sanitize_html_fn: Callable[[str], str],
    markdown_mod: Any,
) -> str:
    try:
        text = str(content or "")
        if text.lstrip().startswith("<"):
            return sanitize_html_fn(text)
        rendered = markdown_mod.markdown(text, extensions=["extra", "codehilite"])
        return sanitize_html_fn(rendered)
    except Exception:
        return sanitize_html_fn(html.escape(str(content or "")))


def _render_legacy_comm_config_dump(api: Any, content: str) -> str:
    try:
        raw = str(content or "")
        if "Loaded rules file:" not in raw:
            return content
        if ("QC-Matrix:" not in raw) and ("QC Matrix:" not in raw):
            return content
        lines = raw.splitlines()
        if not lines or ("Loaded rules file:" not in lines[0]):
            return content

        json_start = None
        for i in range(1, len(lines)):
            ls = lines[i].lstrip()
            if ls.startswith("{") or ls.startswith("["):
                json_start = i
                break
        if json_start is None:
            return content

        qc_idx = None
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].strip()
            if s.startswith("QC-Matrix:") or s.startswith("QC Matrix:"):
                qc_idx = i
                break

        status = lines[0].strip()
        qc = lines[qc_idx].strip() if qc_idx is not None else ""
        json_end = qc_idx if (qc_idx is not None and qc_idx > json_start) else len(lines)
        raw_json = "\n".join(lines[json_start:json_end]).strip()
        try:
            ui_lang = api._lang()  # noqa: SLF001
        except Exception:
            ui_lang = "de"
        summary = "Raw JSON anzeigen" if ui_lang == "de" else "Show raw JSON"
        minor = (
            "Read-only view of the full governance configuration (deterministic from JSON, no LLM)."
            if ui_lang != "de"
            else "Nur-Lese-Ansicht der vollstaendigen Governance-Konfiguration (deterministisch aus JSON, ohne LLM)."
        )
        return (
            '<div class="comm-help comm-config">'
            f'<div class="help-status">{html.escape(status)}</div>'
            f'<div class="minor">{html.escape(minor)}</div>'
            '<details class="config-details">'
            f'<summary>{html.escape(summary)}</summary>'
            f'<pre class="raw-json">{html.escape(raw_json)}</pre>'
            "</details>"
            + (f"<div style='margin-top:10px'>{html.escape(qc)}</div>" if qc else "")
            + "</div>"
        )
    except Exception:
        return content


def ui_replay_loaded_history(
    *,
    api: Any,
    status_msg: str,
    sanitize_html_fn: Callable[[str], str],
    markdown_mod: Any,
) -> None:
    """Rebuild main chat UI from history without triggering model calls."""
    try:
        win = getattr(api, "main_win", None)
        if not win:
            return
        hist = getattr(api, "history", None)
        if not isinstance(hist, list):
            hist = []

        ui_hist = []
        for msg in hist:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role", "") or "").strip().lower()
            content = msg.get("content", "") if "content" in msg else msg.get("text", "")
            if content is None:
                content = ""
            if role == "assistant":
                role = "bot"
            elif role == "system":
                role = "sys"
            elif role not in ("user", "bot", "sys"):
                role = "user"

            if role == "bot":
                rendered = _render_legacy_comm_config_dump(api, str(content))
                bot_html = _as_bot_html(
                    rendered,
                    sanitize_html_fn=sanitize_html_fn,
                    markdown_mod=markdown_mod,
                )
                ui_hist.append({"role": "bot", "html": bot_html})
            else:
                ui_hist.append({"role": role, "content": str(content)})

        payload = json.dumps(ui_hist, ensure_ascii=False)
        sm = json.dumps(str(status_msg or "Loaded chat log."), ensure_ascii=False)

        try:
            js = (
                "(function(){try{"
                "if(window.resetChatFromHistory){window.resetChatFromHistory(%s,%s); return 'OK';}"
                "return 'NOFUNC';"
                "}catch(e){return 'ERR:'+String(e);}})()"
            ) % (payload, sm)
            res = win.evaluate_js(js)
            if isinstance(res, str) and res == "OK":
                return
        except Exception:
            pass

        try:
            win.evaluate_js(f"resetChatToStatus({sm});")
        except Exception:
            return

        for m in ui_hist:
            try:
                r = m.get("role") or "user"
                if r == "bot":
                    h_js = json.dumps(str(m.get("html", "")), ensure_ascii=False)
                    win.evaluate_js(f"addMsg('bot', {h_js}, false, null);")
                else:
                    c_js = json.dumps(html.escape(str(m.get("content", ""))), ensure_ascii=False)
                    rr = "sys" if r == "sys" else "user"
                    win.evaluate_js(f"addMsg('{rr}', {c_js});")
            except Exception:
                continue
    except Exception:
        return
