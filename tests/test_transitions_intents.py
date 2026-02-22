from intents import (
    ComplianceViolation,
    EnterSciRecursion,
    ProcessModelResponse,
    SelectProfile,
    ToggleComm,
    intent_from_command,
)
from state import WrapperState, init_state_from_ruleset
from transitions import apply_intent


def test_intent_from_command_maps_core_tokens():
    assert intent_from_command("Comm Start") == ToggleComm(turn_on=True)
    assert intent_from_command("Comm Stop") == ToggleComm(turn_on=False)
    assert intent_from_command("Profile Expert") == SelectProfile(profile="Expert")
    assert intent_from_command("SCI recurse") == EnterSciRecursion()


def test_init_state_from_ruleset_uses_default_profile_and_profile_overlay():
    ruleset = {
        "default_profile": "Standard",
        "profiles": {
            "Standard": {"mode_overlay": "Strict"},
        },
    }
    state = init_state_from_ruleset(ruleset, answer_language="de", conversation_language="de")
    assert state.active_profile == "Standard"
    assert state.overlay == "Strict"
    assert state.color == "on"


def test_apply_intent_comm_start_enforces_ruleset_default_profile():
    ruleset = {
        "default_profile": "Standard",
        "profiles": {
            "Standard": {"mode_overlay": "Strict"},
            "Expert": {"mode_overlay": "None"},
        },
    }
    before = WrapperState(active_profile="Expert", sci_active=True, sci_pending=True, sci_variant="C")
    out = apply_intent(before, ToggleComm(turn_on=True), ruleset).state
    assert out.comm_active is True
    assert out.active_profile == "Standard"
    assert out.overlay == "Strict"
    assert out.sci_active is False
    assert out.sci_pending is False
    assert out.sci_variant == ""


def test_apply_intent_sci_recurse_honors_max_depth():
    ruleset = {"sci": {"recursive_sci": {"max_depth": 2}}}
    s1 = WrapperState(sci_active=False, sci_variant="", sci_recursion_depth=0)
    s2 = apply_intent(s1, EnterSciRecursion(), ruleset).state
    assert s2.sci_recursion_depth == 1
    assert s2.sci_active is True
    assert s2.sci_variant == "A"

    s3 = apply_intent(s2, EnterSciRecursion(), ruleset).state
    assert s3.sci_recursion_depth == 2

    s4 = apply_intent(s3, EnterSciRecursion(), ruleset).state
    assert s4.sci_recursion_depth == 2


def test_process_model_response_audit_only_keeps_text():
    s = WrapperState(enforcement_policy="audit_only", enforcement_enabled=True)
    intent = ProcessModelResponse(
        raw_text="Antwort",
        violations=(ComplianceViolation(rule="QC-Matrix", severity="major"),),
    )
    out = apply_intent(s, intent, {})
    assert out.command_strings == ["Antwort"]
    assert out.audit_events[-1]["action"] == "audited"


def test_process_model_response_strict_warn_prefixes_warning():
    s = WrapperState(enforcement_policy="strict_warn", enforcement_enabled=True)
    intent = ProcessModelResponse(
        raw_text="Inhalt",
        violations=(ComplianceViolation(rule="SCI Trace", severity="major"),),
    )
    out = apply_intent(s, intent, {})
    assert out.command_strings
    txt = out.command_strings[0]
    assert "Compliance Warnung" in txt
    assert "SCI Trace" in txt
    assert txt.endswith("Inhalt")
    assert out.audit_events[-1]["action"] == "warned"


def test_process_model_response_strict_block_blocks_critical_only_by_default():
    s = WrapperState(enforcement_policy="strict_block", enforcement_enabled=True, blocked_severities=["critical"])
    intent = ProcessModelResponse(
        raw_text="Inhalt",
        violations=(ComplianceViolation(rule="VRG", severity="critical"),),
    )
    out = apply_intent(s, intent, {})
    txt = out.command_strings[0]
    assert txt.startswith("⛔ Antwort blockiert.")
    assert "VRG" in txt
    assert out.audit_events[-1]["action"] == "blocked"


def test_process_model_response_strict_block_does_not_block_major_with_default():
    s = WrapperState(enforcement_policy="strict_block", enforcement_enabled=True, blocked_severities=["critical"])
    intent = ProcessModelResponse(
        raw_text="Inhalt",
        violations=(ComplianceViolation(rule="QC-Matrix", severity="major"),),
    )
    out = apply_intent(s, intent, {})
    assert out.command_strings == ["Inhalt"]
    assert out.audit_events[-1]["action"] == "audited"
