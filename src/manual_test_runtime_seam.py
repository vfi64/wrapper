from __future__ import annotations

import json
import os
import re


def manual_test_report_write_plan(report, *, logs_dir: str, ts: str) -> dict:
    payload = report if isinstance(report, dict) else {"raw": report}
    scenario = str((payload or {}).get("scenario") or "manual_test").strip().lower()
    scenario = re.sub(r"[^a-z0-9._-]+", "_", scenario).strip("_") or "manual_test"
    target_dir = os.path.join(str(logs_dir or ""), "ManualTests")
    stable_name_map = {
        "actual_test": "ManualTest_ACTUAL-TEST.json",
        "temp_test": "ManualTest_TEMP-TEST.json",
    }
    stable_name = str(stable_name_map.get(str(scenario or ""), "") or "").strip()
    if stable_name:
        target = os.path.join(target_dir, stable_name)
    else:
        target = os.path.join(target_dir, f"ManualTest_{str(ts or '').strip() or '0'}_{scenario}.json")
    return {
        "payload": payload,
        "scenario": scenario,
        "target_dir": target_dir,
        "target_path": target,
        "overwritten": bool(stable_name),
    }


def manual_test_main_chat_append_plan(payload=None) -> dict:
    p = payload if isinstance(payload, dict) else {}
    role = str((p or {}).get("role") or "sys").strip().lower()
    if role not in {"user", "bot", "sys"}:
        role = "sys"
    text = str((p or {}).get("text") or "")
    html_text = str((p or {}).get("html") or "")
    cgi_bar = bool((p or {}).get("cgi_bar", False))
    answer_lang = str((p or {}).get("answer_lang") or "").strip().lower()
    if answer_lang not in {"de", "en"}:
        answer_lang = ""

    csc = (p or {}).get("csc")
    if not isinstance(csc, (dict, list, str, int, float, bool)) and csc is not None:
        try:
            csc = str(csc)
        except Exception:
            csc = None

    if role == "bot":
        content = html_text if html_text else text
        opts = {"answerLang": answer_lang} if answer_lang else {}
        js = (
            f"addMsg('bot', {json.dumps(str(content or ''), ensure_ascii=False)}, "
            f"{'true' if cgi_bar else 'false'}, "
            f"{json.dumps(csc, ensure_ascii=False)}, "
            f"{json.dumps(opts, ensure_ascii=False)});"
        )
    elif role == "user":
        content = text if text else html_text
        js = f"addMsg('user', {json.dumps(str(content or ''), ensure_ascii=False)});"
    else:
        content = text if text else html_text
        js = f"addMsg('sys', {json.dumps(str(content or ''), ensure_ascii=False)});"
    return {
        "ok": True,
        "role": role,
        "js": js,
    }


def manual_test_request_stop_plan(payload=None) -> dict:
    p = payload if isinstance(payload, dict) else {}
    lang = str((p or {}).get("lang") or "").strip().lower()
    msg = "Stop requested (monitor)." if lang == "en" else "Stop angefordert (Monitor)."
    js = (
        "(function(){"
        "try{"
        "if(!window.__manualTestRunner){return {ok:false,error:'manual_test_runner unavailable'};}"
        "window.__manualTestRunner.stop = true;"
        f"if(typeof _mtLog==='function'){{_mtLog({json.dumps(msg, ensure_ascii=False)});}}"
        "return {ok:true,running:!!window.__manualTestRunner.running};"
        "}catch(e){return {ok:false,error:String(e&&e.message?e.message:e)};}"
        "})();"
    )
    return {"js": js}
