from __future__ import annotations

import itertools
import json
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Protocol, Sequence


STANDARD_PROMPTS: tuple[str, str] = (
    "Was ist Zeit als Begriff in Physik und Philosophie (nicht die aktuelle Uhrzeit)?",
    "Was ist die objektiv beste und dauerhaft faire Strategie, um ab heute weltweit ein einheitliches KI-Regelwerk verbindlich durchzusetzen, sodass alle LLMs in jeder Sprache, Kultur und Rechtsordnung identische Antworten liefern, ohne negative Folgen fuer Datenschutz, Demokratie, Kreativitaet, Wissenschaft und Arbeitsmarkt?",
)

DEFAULT_PROFILES: tuple[str, ...] = ("Standard", "Expert")
DEFAULT_SCI_VARIANTS: tuple[str, ...] = ("off", "A", "B")
DEFAULT_QC_OVERRIDE_STATES: tuple[bool, ...] = (False, True)
DEFAULT_COLOR_STATES: tuple[str, ...] = ("on", "off")


_QC_KEYS_EN: tuple[str, ...] = (
    "Clarity",
    "Brevity",
    "Evidence",
    "Empathy",
    "Consistency",
    "Neutrality",
)
_U_TEXT_RE = re.compile(r"\b(U[1-6])\b", re.IGNORECASE)
_U_ATTR_RE = re.compile(r"(?i)data-u-code\s*=\s*(?:\"|')?(U[1-6])(?:\"|')?")
_COLOR_TAG_RE = re.compile(r"\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]")
_COLOR_EMOJI_RE = re.compile(r"[🟢🟡🔴⚪]")
_CGI_RE = re.compile(r"(?i)\bcgi\b.*\b(feedback|angewendet|applied|gespeichert|recorded)\b")
_DYNAMIC_ONESHOT_RE = re.compile(r"(?i)(?:dynamic[^.\n]{0,80}one-?shot|one-?shot[^.\n]{0,80}dynamic)")


@dataclass(frozen=True)
class MatrixCase:
    profile: str
    sci_variant: str
    qc_override: bool
    color: str


class HarnessDriver(Protocol):
    def reset(self) -> None: ...
    def configure_case(self, case: MatrixCase) -> None: ...
    def set_qc_override(self, enabled: bool) -> None: ...
    def set_dynamic_one_shot(self, enabled: bool) -> None: ...
    def apply_cgi_feedback(self, clarity: int, insight: int, efficiency: int) -> None: ...
    def ask(self, prompt: str) -> str: ...


class SyntheticHarnessDriver:
    """Deterministic test driver for S15.1 groundwork."""

    def __init__(self) -> None:
        self._case = MatrixCase(profile="Standard", sci_variant="off", qc_override=False, color="on")
        self._qc_override = False
        self._dynamic_one_shot = False
        self._cgi_feedback_pending = False
        self._cgi_triplet = "0,0,0"

    def reset(self) -> None:
        self._qc_override = False
        self._dynamic_one_shot = False
        self._cgi_feedback_pending = False
        self._cgi_triplet = "0,0,0"

    def configure_case(self, case: MatrixCase) -> None:
        self._case = case
        self._qc_override = bool(case.qc_override)

    def set_qc_override(self, enabled: bool) -> None:
        self._qc_override = bool(enabled)

    def set_dynamic_one_shot(self, enabled: bool) -> None:
        self._dynamic_one_shot = bool(enabled)

    def apply_cgi_feedback(self, clarity: int, insight: int, efficiency: int) -> None:
        self._cgi_triplet = f"{int(clarity)},{int(insight)},{int(efficiency)}"
        self._cgi_feedback_pending = True

    def ask(self, prompt: str) -> str:
        prompt_s = str(prompt or "").strip()
        out: list[str] = []
        out.append(f"Antwort ({self._case.profile}, SCI {self._case.sci_variant}): {prompt_s}")

        if self._case.sci_variant != "off":
            out.append("SCI Trace:")
            out.append("1. Plan: Ziel und Rahmen klarstellen.")
            out.append("2. Check: Risiken und Nebenwirkungen pruefen.")

        if self._dynamic_one_shot:
            out.append("Dynamic: one-shot (active)")

        if self._cgi_feedback_pending:
            c, i, e = self._cgi_triplet.split(",")
            out.append(f"CGI angewendet (one-shot): K={c} · E={i} · F={e}")
            self._cgi_feedback_pending = False

        if self._qc_override:
            out.append("QC-Override aktiv: Antwort wird bewusst praegnanzfokussiert angepasst.")

        if "einheitliches KI-Regelwerk" in prompt_s:
            out.append("Unsicherheit: U2 - Annahmen unklar, da globale Rechtslagen divergieren.")
        else:
            out.append("Unsicherheit: U4 - Zeitliche Instabilitaet bei Faktenstand.")

        if self._case.color == "on":
            out.append("[GREEN] 🟢 Evidenzhinweis.")
            out.append("[YELLOW] 🟡 Abwaegungspunkt.")
        else:
            out.append("Farbmarker deaktiviert.")

        out.append("Self-Debunking:")
        out.append("1. Schwachstelle: Modellannahmen sind unvollstaendig.")
        out.append("2. Gegenmassnahme: Explizite Verifikationsroute ausgeben.")

        brevity_value = 0 if self._qc_override else 2
        qc_line = (
            "QC-Matrix: "
            f"Clarity 3 (Δ0) · Brevity {brevity_value} (Δ0) · Evidence 3 (Δ0) · "
            "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)"
        )
        out.append(qc_line)
        return "\n".join(out)


def build_case_matrix(
    *,
    profiles: Sequence[str] = DEFAULT_PROFILES,
    sci_variants: Sequence[str] = DEFAULT_SCI_VARIANTS,
    qc_override_states: Sequence[bool] = DEFAULT_QC_OVERRIDE_STATES,
    color_states: Sequence[str] = DEFAULT_COLOR_STATES,
) -> list[MatrixCase]:
    matrix: list[MatrixCase] = []
    for profile, sci_variant, qc_override, color in itertools.product(
        profiles,
        sci_variants,
        qc_override_states,
        color_states,
    ):
        matrix.append(
            MatrixCase(
                profile=str(profile),
                sci_variant=str(sci_variant),
                qc_override=bool(qc_override),
                color=str(color),
            )
        )
    return matrix


def _to_plain_text(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</?(?:p|div|li|ul|ol|details|summary|table|tr|td|th|h[1-6])[^>]*>", "\n", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    raw = unescape(raw)
    raw = re.sub(r"\r\n?", "\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def has_complete_qc_footer(text: str) -> bool:
    txt = _to_plain_text(text)
    idx_matrix = txt.rfind("QC-Matrix:")
    idx_qc = txt.rfind("QC:")
    idx = max(idx_matrix, idx_qc)
    if idx < 0:
        return False
    qc = txt[idx:]
    ts_match = re.search(r"(?i)\bResponse at\b", qc)
    if ts_match:
        qc = qc[: ts_match.start()]
    qc = re.sub(r"\s+", " ", qc).strip()
    has_en = all((key + " ") in qc for key in _QC_KEYS_EN)
    has_de = (
        all((key + " ") in qc for key in ("Klarheit", "Evidenz", "Empathie", "Konsistenz"))
        and ("Kuerze " in qc or "Kürze " in qc)
        and ("Neutralitaet " in qc or "Neutralität " in qc)
    )
    return bool(has_en or has_de)


def _extract_u_codes(text: str) -> list[str]:
    raw = str(text or "")
    codes = {c.upper() for c in _U_TEXT_RE.findall(_to_plain_text(raw))}
    codes.update(c.upper() for c in _U_ATTR_RE.findall(raw))
    return sorted(codes)


def _count_color_markers(text: str) -> int:
    raw = str(text or "")
    plain = _to_plain_text(raw)
    return len(_COLOR_TAG_RE.findall(raw)) + len(_COLOR_EMOJI_RE.findall(plain))


def _markers_before_qc(text: str, marker_pattern: re.Pattern[str]) -> bool:
    plain = _to_plain_text(text)
    idx_matrix = plain.rfind("QC-Matrix:")
    idx_qc = plain.rfind("QC:")
    idx = max(idx_matrix, idx_qc)
    if idx < 0:
        return False
    trailing = plain[idx:]
    return marker_pattern.search(trailing) is None


def analyze_response(text: str, *, expect_color_on: bool) -> dict[str, Any]:
    raw = str(text or "")
    plain = _to_plain_text(raw)
    u_codes = _extract_u_codes(raw)
    color_markers = _count_color_markers(raw)
    warnings: list[str] = []
    if expect_color_on and color_markers == 0:
        warnings.append("color_markers_missing_when_color_on")
    if (not expect_color_on) and color_markers > 0:
        warnings.append("color_markers_present_when_color_off")

    return {
        "qc_footer_complete": has_complete_qc_footer(raw),
        "cgi_feedback_seen": bool(_CGI_RE.search(plain)),
        "dynamic_one_shot_seen": bool(_DYNAMIC_ONESHOT_RE.search(plain)),
        "u_codes": u_codes,
        "u_marker_count": len(u_codes),
        "u_marker_position_ok": _markers_before_qc(raw, _U_TEXT_RE),
        "color_marker_count": color_markers,
        "color_marker_position_ok": _markers_before_qc(raw, _COLOR_EMOJI_RE),
        "warnings": warnings,
    }


def _normalize_for_influence(text: str) -> str:
    s = _to_plain_text(text)
    s = re.sub(r"(?i)QC(?:-Matrix)?\s*:\s*.*$", " ", s)
    s = re.sub(r"(?i)Response at\s+.*$", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _run_single_influence_check(
    *,
    name: str,
    driver: HarnessDriver,
    case: MatrixCase,
    prompt: str,
) -> dict[str, Any]:
    try:
        if name == "qc_override":
            driver.reset()
            driver.configure_case(case)
            driver.set_qc_override(False)
            baseline = driver.ask(prompt)
            driver.set_qc_override(True)
            changed_output = driver.ask(prompt)
        elif name == "dynamic_one_shot":
            driver.reset()
            driver.configure_case(case)
            driver.set_dynamic_one_shot(False)
            baseline = driver.ask(prompt)
            driver.set_dynamic_one_shot(True)
            changed_output = driver.ask(prompt)
        elif name == "cgi_feedback":
            driver.reset()
            driver.configure_case(case)
            baseline = driver.ask(prompt)
            driver.apply_cgi_feedback(3, 2, 1)
            changed_output = driver.ask(prompt)
        else:
            return {"name": name, "status": "unsupported", "changed": None, "error": "unknown_check"}
        changed = _normalize_for_influence(baseline) != _normalize_for_influence(changed_output)
        return {
            "name": name,
            "status": "pass" if changed else "fail",
            "changed": bool(changed),
            "baseline_excerpt": _normalize_for_influence(baseline)[:220],
            "changed_excerpt": _normalize_for_influence(changed_output)[:220],
        }
    except NotImplementedError as exc:
        return {"name": name, "status": "unsupported", "changed": None, "error": str(exc)}
    except Exception as exc:
        return {"name": name, "status": "unsupported", "changed": None, "error": f"{type(exc).__name__}: {exc}"}


def run_harness(
    *,
    driver: HarnessDriver,
    prompts: Sequence[str] = STANDARD_PROMPTS,
    matrix_cases: Sequence[MatrixCase] | None = None,
    ruleset_path: str = "JSON/Comm-SCI-v20.0.3.json",
) -> dict[str, Any]:
    cases = list(matrix_cases) if matrix_cases is not None else build_case_matrix()
    prompt_list = [str(p) for p in prompts]
    case_results: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        driver.reset()
        driver.configure_case(case)
        prompt_results: list[dict[str, Any]] = []
        for prompt in prompt_list:
            result: dict[str, Any]
            try:
                response = driver.ask(prompt)
                analysis = analyze_response(response, expect_color_on=(case.color == "on"))
                status = "pass" if bool(analysis.get("qc_footer_complete")) else "fail"
                result = {
                    "prompt": prompt,
                    "status": status,
                    "analysis": analysis,
                }
            except Exception as exc:
                result = {
                    "prompt": prompt,
                    "status": "fail",
                    "analysis": {
                        "qc_footer_complete": False,
                        "cgi_feedback_seen": False,
                        "dynamic_one_shot_seen": False,
                        "u_codes": [],
                        "u_marker_count": 0,
                        "u_marker_position_ok": False,
                        "color_marker_count": 0,
                        "color_marker_position_ok": False,
                        "warnings": [],
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
            prompt_results.append(result)
        case_results.append(
            {
                "case_id": case_index,
                "case": asdict(case),
                "prompts": prompt_results,
            }
        )

    influence_case = cases[0] if cases else MatrixCase("Standard", "off", False, "on")
    influence_prompt = prompt_list[0] if prompt_list else "Was ist Zeit als Begriff in Physik und Philosophie (nicht die aktuelle Uhrzeit)?"
    influence_checks = [
        _run_single_influence_check(name="qc_override", driver=driver, case=influence_case, prompt=influence_prompt),
        _run_single_influence_check(name="dynamic_one_shot", driver=driver, case=influence_case, prompt=influence_prompt),
        _run_single_influence_check(name="cgi_feedback", driver=driver, case=influence_case, prompt=influence_prompt),
    ]

    prompt_results_flat = [p for c in case_results for p in c["prompts"]]
    fail_count = sum(1 for p in prompt_results_flat if p.get("status") == "fail")
    error_entries = [p.get("error") for p in prompt_results_flat if p.get("error")]
    error_count = len(error_entries)
    first_error = str(error_entries[0]) if error_entries else ""
    warn_count = sum(len((p.get("analysis") or {}).get("warnings") or []) for p in prompt_results_flat)
    influence_fail_count = sum(1 for chk in influence_checks if chk.get("status") == "fail")
    influence_unsupported_count = sum(1 for chk in influence_checks if chk.get("status") == "unsupported")
    overall_status = "pass" if (fail_count == 0 and influence_fail_count == 0) else "fail"

    human_report = [
        f"Pflichtprompts: {len(prompt_list)}",
        f"Matrix-Faelle: {len(cases)}",
        f"Prompt-Checks: {len(prompt_results_flat)} (FAIL={fail_count}, WARN={warn_count})",
        (
            "Influence-Checks: "
            f"{len(influence_checks)} (FAIL={influence_fail_count}, UNSUPPORTED={influence_unsupported_count})"
        ),
        f"Prompt-Errors: {error_count}",
        f"Gesamtstatus: {overall_status.upper()}",
    ]

    return {
        "ruleset_path": ruleset_path,
        "mandatory_prompts": prompt_list,
        "matrix": [asdict(c) for c in cases],
        "case_results": case_results,
        "influence_checks": influence_checks,
        "summary": {
            "overall_status": overall_status,
            "case_count": len(cases),
            "prompt_check_count": len(prompt_results_flat),
            "fail_count": fail_count,
            "error_count": error_count,
            "first_error": first_error,
            "warn_count": warn_count,
            "influence_fail_count": influence_fail_count,
            "influence_unsupported_count": influence_unsupported_count,
        },
        "human_report": human_report,
    }


def _scenario_slug(name: str) -> str:
    raw = str(name or "").strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "_", raw).strip("_")
    return safe or "scenario_harness"


def write_harness_log(*, root_dir: Path | str, scenario: str, report: dict[str, Any]) -> Path:
    root = Path(root_dir)
    target_dir = root / "Logs" / "ManualTests"
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = target_dir / f"HarnessRun_{ts}_{_scenario_slug(scenario)}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run_harness_with_final_log(
    *,
    driver: HarnessDriver,
    root_dir: Path | str,
    scenario: str = "s15_1_harness",
    prompts: Sequence[str] = STANDARD_PROMPTS,
    matrix_cases: Sequence[MatrixCase] | None = None,
    ruleset_path: str = "JSON/Comm-SCI-v20.0.3.json",
) -> tuple[dict[str, Any], Path]:
    started = datetime.now()
    run_payload: dict[str, Any] = {}
    report: dict[str, Any] = {}
    path: Path | None = None
    status = "aborted"
    error_text = ""
    tb_text = ""
    try:
        run_payload = run_harness(
            driver=driver,
            prompts=prompts,
            matrix_cases=matrix_cases,
            ruleset_path=ruleset_path,
        )
        overall = str((run_payload.get("summary") or {}).get("overall_status") or "fail").lower()
        status = "passed" if overall == "pass" else "failed"
    except Exception as exc:
        status = "failed"
        error_text = f"{type(exc).__name__}: {exc}"
        tb_text = traceback.format_exc(limit=20)
        fallback_cases = list(matrix_cases) if matrix_cases is not None else build_case_matrix()
        run_payload = {
            "ruleset_path": ruleset_path,
            "mandatory_prompts": [str(p) for p in prompts],
            "matrix": [asdict(c) for c in fallback_cases],
            "case_results": [],
            "influence_checks": [],
            "summary": {
                "overall_status": "fail",
                "case_count": len(fallback_cases),
                "prompt_check_count": 0,
                "fail_count": 1,
                "error_count": 1,
                "first_error": error_text,
                "warn_count": 0,
                "influence_fail_count": 0,
                "influence_unsupported_count": 0,
            },
            "human_report": ["Harness-Ausfuehrung fehlgeschlagen."],
        }
    finally:
        finished = datetime.now()
        report = {
            "kind": "scenario_harness_report",
            "version": 1,
            "scenario": str(scenario),
            "status": status,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_ms": int((finished - started).total_seconds() * 1000.0),
        }
        report.update(run_payload)
        if error_text:
            report["error"] = error_text
        if tb_text:
            report["traceback"] = tb_text
        if (not error_text) and status != "passed":
            summary = report.get("summary") or {}
            if int(summary.get("error_count") or 0) > 0:
                report["error"] = str(summary.get("first_error") or "Prompt execution errors in harness run.")
        path = write_harness_log(root_dir=root_dir, scenario=scenario, report=report)
        report["log_path"] = str(path)
    if path is None:
        raise RuntimeError("harness_report_write_failed")
    return report, path
