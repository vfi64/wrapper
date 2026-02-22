from __future__ import annotations

from types import SimpleNamespace

from Module.compliance_scan import (
    scan_message_compliance_best_effort,
    scan_message_compliance_best_effort_detailed,
)


def _build_route_ctx(_api, user_text: str, _looks_command: bool):
    txt = (user_text or "").strip()
    return {"is_command": txt.startswith("Comm ")}


def test_compliance_scan_detailed_maps_severities_and_keeps_legacy_status():
    api = SimpleNamespace(gov_state=SimpleNamespace(active_profile="Standard", sci_variant="A"))
    gov = SimpleNamespace(
        check_self_debunking=lambda _txt, _prof: "Missing Self-Debunking block",
        check_verification_route_gate=lambda _txt: "Verification Route Gate missing",
    )
    history = [
        {"role": "user", "content": "Bitte beantworte die Frage"},
        {"role": "assistant", "content": "Nur ein kurzer Text ohne QC und ohne Trace-Block"},
    ]
    sample = [history[-1]]

    detailed = scan_message_compliance_best_effort_detailed(
        sample=sample,
        history=history,
        build_route_ctx=_build_route_ctx,
        api=api,
        gov=gov,
    )
    assert len(detailed) == 1
    idx, status, vios = detailed[0]
    assert idx == 1
    assert status.startswith("⚠ ")
    assert len(vios) >= 3

    sev_by_code = {v["code"]: v["severity"] for v in vios}
    assert sev_by_code["missing_sci_trace_block"] == "critical"
    assert sev_by_code["missing_qc_footer"] == "major"
    assert sev_by_code["missing_self_debunking"] == "major"
    assert sev_by_code["verification_route_gate"] == "major"

    legacy = scan_message_compliance_best_effort(
        sample=sample,
        history=history,
        build_route_ctx=_build_route_ctx,
        api=api,
        gov=gov,
    )
    assert legacy == [(idx, status)]


def test_compliance_scan_command_response_skips_qc_sd_sci_checks():
    api = SimpleNamespace(gov_state=SimpleNamespace(active_profile="Standard", sci_variant="A"))
    gov = SimpleNamespace(
        check_self_debunking=lambda _txt, _prof: "Missing Self-Debunking block",
        check_verification_route_gate=lambda _txt: "",
    )
    history = [
        {"role": "user", "content": "Comm State"},
        {"role": "assistant", "content": "Command executed: Comm State"},
    ]
    sample = [history[-1]]

    detailed = scan_message_compliance_best_effort_detailed(
        sample=sample,
        history=history,
        build_route_ctx=_build_route_ctx,
        api=api,
        gov=gov,
    )
    assert detailed[0][1] == "✓ Compliant"
    assert detailed[0][2] == []
