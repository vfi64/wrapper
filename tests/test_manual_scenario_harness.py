from __future__ import annotations

import json

import manual_scenario_harness as sut


def test_build_case_matrix_is_deterministic_and_complete():
    cases = sut.build_case_matrix(
        profiles=("P1", "P2"),
        sci_variants=("off", "A"),
        qc_override_states=(False, True),
        color_states=("on", "off"),
    )
    assert len(cases) == 16
    assert cases[0] == sut.MatrixCase(profile="P1", sci_variant="off", qc_override=False, color="on")
    assert cases[-1] == sut.MatrixCase(profile="P2", sci_variant="A", qc_override=True, color="off")


def test_analyze_response_detects_qc_u_and_color_markers():
    text = (
        "Antwort\n"
        "[GREEN] 🟢 Aussage.\n"
        "Unsicherheit: U1 - Datenluecke.\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)"
    )
    analysis = sut.analyze_response(text, expect_color_on=True)
    assert analysis["qc_footer_complete"] is True
    assert analysis["u_codes"] == ["U1"]
    assert analysis["color_marker_count"] >= 2
    assert analysis["u_marker_position_ok"] is True
    assert analysis["color_marker_position_ok"] is True


def test_run_harness_returns_machine_and_human_sections():
    report = sut.run_harness(driver=sut.SyntheticHarnessDriver())
    assert report["mandatory_prompts"] == list(sut.STANDARD_PROMPTS)
    summary = report["summary"]
    assert summary["case_count"] == len(report["matrix"])
    assert summary["prompt_check_count"] == summary["case_count"] * len(sut.STANDARD_PROMPTS)
    assert isinstance(report["human_report"], list) and report["human_report"]
    influence = {item["name"]: item for item in report["influence_checks"]}
    assert influence["qc_override"]["status"] == "pass"
    assert influence["dynamic_one_shot"]["status"] == "pass"
    assert influence["cgi_feedback"]["status"] == "pass"


def test_run_harness_with_final_log_writes_file_even_on_failure(tmp_path):
    class FailingDriver(sut.SyntheticHarnessDriver):
        def ask(self, prompt: str) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    report, path = sut.run_harness_with_final_log(
        driver=FailingDriver(),
        root_dir=tmp_path,
        scenario="failing_case",
        matrix_cases=[sut.MatrixCase(profile="Standard", sci_variant="off", qc_override=False, color="on")],
    )
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "error" in payload
    assert "log_path" in report

