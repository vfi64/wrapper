from __future__ import annotations

import html
import re
from typing import Callable, Iterable


def render_sci_trace_as_html_runtime(
    text_in: str,
    *,
    sci_variant: str,
    sci_active: bool,
    required_steps: Iterable[str] | None,
    match_required_sci_step_header_fn: Callable | None,
) -> str:
    """Repair + render SCI Trace deterministically for display.

    Goals:
    - If the model emits an empty 'SCI Trace' (only step names), rebuild it from
      step-labeled content elsewhere in the response.
    - Ensure all required steps for the active SCI variant are shown, and never as empty bullets.
    - Avoid duplicate content: extracted step sections are removed from the final-answer body.
    """
    try:
        if not text_in or "SCI Trace" not in text_in:
            return text_in

        variant = str(sci_variant or "").strip().upper()
        if (not bool(sci_active)) or (not variant):
            return text_in

        req_steps = [str(s) for s in (required_steps or []) if str(s).strip()]
        if not req_steps:
            return text_in

        if not callable(match_required_sci_step_header_fn):
            return text_in

        lines = text_in.splitlines()

        def _is_sci_trace_heading_line(plain_line: str) -> bool:
            p = re.sub(r"\s+", " ", str(plain_line or "")).strip()
            p = p.strip("*").strip()
            return bool(
                re.match(
                    r"^(?:#+\s*)?(?:\d+[\.\)]\s*)?SCI\s+Trace\b",
                    p,
                    flags=re.IGNORECASE,
                )
            )

        # Find the earliest SCI Trace marker line (also accepts forms like
        # "4. SCI Trace (Variante A: Standard)".
        sci_idx = None
        for i, ln in enumerate(lines):
            ln_plain = re.sub(r"<[^>]+>", "", ln).strip()
            ln_plain = ln_plain.strip().strip("*").strip()
            if _is_sci_trace_heading_line(ln_plain):
                sci_idx = i
                break
        if sci_idx is None:
            return text_in

        # Identify the end of the immediate list after 'SCI Trace' (bullets/numbering only)
        list_pat = re.compile(r"^\s*(?:[*+-]|•|\d+\.)\s+")
        k = sci_idx + 1
        while k < len(lines) and (not lines[k].strip() or list_pat.match(re.sub(r"<[^>]+>", "", lines[k]))):
            k += 1
        trace_list_end = k  # exclusive

        # If this immediate list already contains SCI step headers (e.g. "• Plan: ..."),
        # keep it as content; otherwise it is typically an empty placeholder list.
        try:
            _has_step_header = False
            for _ln in lines[sci_idx + 1:trace_list_end]:
                _plain = re.sub(r"<[^>]+>", "", _ln or "")
                _step, _rest = match_required_sci_step_header_fn(_plain, req_steps)
                if _step:
                    _has_step_header = True
                    break
            if _has_step_header:
                trace_list_end = sci_idx + 1
        except Exception:
            pass

        # Build a working copy without the original (often empty) trace list block
        pre = lines[:sci_idx]
        rest = lines[trace_list_end:]

        # Split off trailing governance blocks we must keep in place (Self-Debunking, QC-Matrix)
        boundary_pat = re.compile(r"^\s*(Self-?Debunking\s*:|QC-?Matrix\s*:)", re.IGNORECASE)
        tail_start = None
        for i, ln in enumerate(rest):
            plain_ln = re.sub(r"<[^>]+>", "", ln or "")
            if (
                boundary_pat.match(plain_ln)
                or re.search(r"(?i)\bself[- ]?debunking\b", plain_ln)
                or re.search(r"(?i)\bQC-?Matrix\s*:", plain_ln)
                or re.search(r"(?is)class=(?:\"|')[^\"']*self-debunking[^\"']*(?:\"|')", ln or "")
            ):
                tail_start = i
                break
        if tail_start is None:
            main = rest
            tail = []
        else:
            main = rest[:tail_start]
            tail = rest[tail_start:]

        # Normalize basic HTML bold headers that may leak into raw text
        def _strip_basic_tags(s: str) -> str:
            s = re.sub(r"</?(strong|b)>", "", s, flags=re.IGNORECASE)
            s = re.sub(r"</?p>", "", s, flags=re.IGNORECASE)
            return s

        def _strip_md_edge_emphasis_tokens(s: str) -> str:
            """Remove common leaked markdown emphasis tokens at line edges only.

            Keeps inline emphasis inside the sentence untouched.
            """
            t = str(s or "")
            # Leading token forms often leaked by weaker models:
            # "** text", "__ text", and "* * text".
            t = re.sub(r"^\s*\*\s*\*\s*", "", t)
            t = re.sub(r"^\s*(?:\*\*|__)\s*", "", t)
            # Trailing orphan emphasis markers.
            t = re.sub(r"\s*(?:\*\*|__)\s*$", "", t)
            return t

        blocks = {}
        out_main = []
        cur_step = None
        buf = []
        last_step = req_steps[-1] if req_steps else ""
        after_last_step_break = False

        def _looks_like_final_answer_start(plain_line: str, raw_line: str = "") -> bool:
            s = str(plain_line or "").strip()
            if not s:
                return False
            if re.match(r"^(?:[*+-]|•|\d+\.)\s+", s):
                return False
            if re.match(r"(?i)^(?:Final\s+Answer|Antwort|Answer)\b", s):
                return True
            if re.match(r"(?i)^(?:Self[- ]?Debunking|Selbst[- ]?Debunking|QC(?:-Matrix)?)\b", s):
                return False
            # Evidence/uncertainty markers in raw HTML are a strong indicator that
            # we have entered the narrative final-answer body.
            raw = str(raw_line or "")
            if ("signal-dot-marker" in raw) or ("uncertainty-inline-marker" in raw):
                return True
            words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", s)
            return (len(words) >= 8) or (len(s) >= 80)

        def _extract_answer_marker_payload(raw_line: str, plain_line: str = "") -> tuple[bool, str]:
            marker_re = re.compile(
                r"(?is)^\s*(?:(?:[*+\-]|•)\s+|\d+\s*[\.\)]\s+)*"
                r"(?:Final\s+Answer|Antwort|Answer)\s*(?::|-|–|—)?\s*(.*)$"
            )
            for src in (raw_line, plain_line):
                cand = str(src or "")
                cand = re.sub(r"<[^>]+>", " ", cand)
                cand = _strip_md_edge_emphasis_tokens(cand)
                cand = re.sub(r"\s+", " ", cand).strip()
                m = marker_re.match(cand)
                if m is not None:
                    return True, str(m.group(1) or "").strip()
            return False, ""

        def flush() -> None:
            nonlocal cur_step, buf
            if cur_step is None:
                return
            cleaned = []
            for x in buf:
                t = re.sub(r"^\s*\d+[\.\)]\s+", "", str(x or ""))
                t = _strip_md_edge_emphasis_tokens(t)
                cleaned.append(t)
            while cleaned and not cleaned[0].strip():
                cleaned.pop(0)
            while cleaned and not cleaned[-1].strip():
                cleaned.pop()
            # Keep the first non-empty capture for a step when duplicated SCI Trace
            # sections leak into the same answer.
            if (cur_step not in blocks) or (not blocks.get(cur_step)):
                blocks[cur_step] = cleaned
            cur_step = None
            buf = []

        recognized_steps = 0
        for ln in main:
            ln2 = _strip_basic_tags(re.sub(r"<[^>]+>", "", ln))
            if _is_sci_trace_heading_line(ln2):
                # Drop duplicate SCI Trace headings from the body once we rebuild.
                continue
            step_name, rest_line = match_required_sci_step_header_fn(ln2, req_steps)
            if step_name:
                flush()
                cur_step = step_name
                recognized_steps += 1
                after_last_step_break = False
                if rest_line:
                    buf.append(rest_line)
                continue
            if cur_step is not None:
                marker_hit, marker_payload = _extract_answer_marker_payload(ln, ln2)
                if marker_hit:
                    # Answer marker belongs between trace and narrative.
                    flush()
                    after_last_step_break = False
                    if marker_payload:
                        out_main.append(marker_payload)
                    continue
                if cur_step == last_step:
                    if not ln2.strip():
                        after_last_step_break = True
                        buf.append(ln2)
                        continue
                    if after_last_step_break and _looks_like_final_answer_start(ln2, ln):
                        # Split here: remaining narrative belongs to final answer,
                        # not to the last SCI step.
                        flush()
                        after_last_step_break = False
                        marker_hit2, marker_payload2 = _extract_answer_marker_payload(ln, ln2)
                        if marker_hit2:
                            if marker_payload2:
                                out_main.append(marker_payload2)
                        else:
                            out_main.append(ln)
                        continue
                buf.append(ln2)
            else:
                marker_hit, marker_payload = _extract_answer_marker_payload(ln, ln2)
                if marker_hit:
                    # Keep the narrative payload, but strip marker label.
                    if marker_payload:
                        out_main.append(marker_payload)
                    continue
                out_main.append(ln)

        flush()

        # If we didn't recognize any step content, do not fabricate trace items.
        if recognized_steps == 0:
            return text_in

        # Render only steps that have real content; never emit empty steps.
        missing = []
        for s in req_steps:
            if s in missing:
                continue
            if not blocks.get(s):
                missing.append(s)

        # Optional deterministic alert if step content is missing
        alert_html = ""
        if missing:
            safe = ", ".join([html.escape(x) for x in missing])
            alert_html = (
                "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                "<b>CONTROL LAYER ALERT (SCI)</b><br>"
                "Missing SCI Trace step content for: " + safe +
                "</div>"
            )

        # Render deterministic HTML trace block
        html_parts = [
            "<!-- SCI Trace: -->",
            "<div class='sci-trace' style='margin:10px 0; padding:10px; border:1px solid #ddd; border-radius:12px;'>",
            "<div style='font-weight:700; margin-bottom:6px;'>SCI Trace</div>",
            "<ol style='margin:0 0 0 22px; padding:0;'>",
        ]
        for s in req_steps:
            if s in missing:
                continue
            html_parts.append("<li style='margin:4px 0 10px 0;'>")
            html_parts.append(f"<div style='font-weight:700; margin:0 0 4px 0;'>{html.escape(s)}:</div>")
            for ln in (blocks.get(s) or []):
                t = (ln or "").rstrip("\n")
                if not t.strip():
                    html_parts.append("<div style='height:6px'></div>")
                    continue
                m2 = re.match(r"^\s*([*+-]|•)\s+(.*)$", t)
                if m2:
                    bullet_text = _strip_md_edge_emphasis_tokens(m2.group(2).strip())
                    html_parts.append(f"<div style='margin-left:14px;'>• {html.escape(bullet_text)}</div>")
                else:
                    plain_text = _strip_md_edge_emphasis_tokens(t.strip())
                    html_parts.append(f"<div>{html.escape(plain_text)}</div>")
            html_parts.append("</li>")
        html_parts.extend(["</ol>", "</div>"])

        # Deterministic ordering:
        # Header is injected later by the wrapper, so within this body we enforce:
        # SCI Trace -> narrative final answer -> governance tail (Self-Debunking/QC).
        narrative_lines = []
        narrative_lines.extend(pre)
        narrative_lines.extend(out_main)

        out_lines = []
        if alert_html:
            out_lines.append(alert_html)
        out_lines.append("\n".join(html_parts))
        out_lines.extend(narrative_lines)
        out_lines.extend(tail)
        return "\n".join(out_lines)
    except Exception:
        return text_in
