from governance_service import GovernanceService
import types


def test_governance_service_delegates_all_transforms():
    calls = []

    def _norm(x):
        calls.append(("norm", x))
        return f"N:{x}"

    def _sd(x, gov, prof, *, is_command=False, lang="en"):
        calls.append(("sd", x, gov, prof, is_command, lang))
        return f"SD:{x}:{prof}:{lang}:{is_command}"

    def _sci(x, gov):
        calls.append(("sci", x, gov))
        return f"SCI:{x}"

    def _sd_num(x, *, lang="en"):
        calls.append(("sd_num", x, lang))
        return f"SDN:{x}:{lang}"

    svc = GovernanceService(
        normalize_headings_fn=_norm,
        enforce_self_debunking_fn=_sd,
        normalize_sci_trace_fn=_sci,
        normalize_self_debunking_numbering_fn=_sd_num,
    )

    assert svc.normalize_headings("a") == "N:a"
    assert svc.enforce_self_debunking("b", {"g": 1}, "Expert", is_command=False, lang="de") == "SD:b:Expert:de:False"
    assert svc.normalize_sci_trace("c", {"g": 2}) == "SCI:c"
    assert svc.normalize_self_debunking_numbering("d", lang="de") == "SDN:d:de"
    assert calls[0][0] == "norm"
    assert calls[1][0] == "sd"
    assert calls[2][0] == "sci"
    assert calls[3][0] == "sd_num"


def test_governance_service_is_fail_soft_and_returns_input_on_errors():
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    svc = GovernanceService(
        normalize_headings_fn=_boom,
        enforce_self_debunking_fn=_boom,
        normalize_sci_trace_fn=_boom,
        normalize_self_debunking_numbering_fn=_boom,
    )

    assert svc.normalize_headings("x") == "x"
    assert svc.enforce_self_debunking("y", {}, "Standard", lang="de") == "y"
    assert svc.normalize_sci_trace("z", {}) == "z"
    assert svc.normalize_self_debunking_numbering("w", lang="de") == "w"


def test_governance_service_normalize_output_contracts_order_and_fail_soft():
    calls = []

    def _qc(text, gov, profile):
        calls.append(("qc", text, profile))
        return text + "|qc"

    def _ev(text):
        calls.append(("ev", text))
        return text + "|ev"

    def _sd(text, gov, profile, *, is_command=False, lang="en"):
        calls.append(("sd", text, profile, is_command, lang))
        return text + "|sd"

    def _sci(text, gov):
        calls.append(("sci", text))
        return text + "|sci"

    def _ensure_qc(text, gov, profile):
        calls.append(("ensure_qc", text, profile))
        return text + "|ensure_qc"

    svc = GovernanceService(
        enforce_qc_footer_fn=_qc,
        normalize_evidence_tags_fn=_ev,
        enforce_self_debunking_fn=_sd,
        normalize_sci_trace_fn=_sci,
        ensure_qc_footer_present_fn=_ensure_qc,
    )

    out = svc.normalize_output_contracts(
        "x",
        gov_mgr={"g": 1},
        profile_name="Expert",
        governance_enabled=True,
        is_command=False,
        lang="de",
    )
    assert out == "x|qc|ev|sd|sci|ensure_qc"
    assert [c[0] for c in calls] == ["qc", "ev", "sd", "sci", "ensure_qc"]

    calls.clear()
    out2 = svc.normalize_output_contracts(
        "x",
        gov_mgr={"g": 1},
        profile_name="Expert",
        governance_enabled=False,
        is_command=False,
        lang="de",
    )
    assert out2 == "x|ev|sci"
    assert [c[0] for c in calls] == ["ev", "sci"]

    svc2 = GovernanceService(
        enforce_qc_footer_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
        normalize_evidence_tags_fn=lambda s: s + "|ev",
        enforce_self_debunking_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("y")),
        normalize_sci_trace_fn=lambda s, _g: s + "|sci",
        ensure_qc_footer_present_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("z")),
    )
    out3 = svc2.normalize_output_contracts(
        "x",
        gov_mgr={},
        profile_name="Standard",
        governance_enabled=True,
        is_command=False,
        lang="de",
    )
    assert out3 == "x|ev|sci"


def test_governance_service_profile_switch_resets():
    st = types.SimpleNamespace(
        active_profile="Standard",
        qc_overrides={"brevity": 1},
        sci_pending_turns=3,
        sci_active=True,
        sci_pending=True,
        sci_variant="B",
    )
    svc = GovernanceService()
    svc.apply_profile_switch_resets(st, "Briefing")
    assert st.active_profile == "Briefing"
    assert st.qc_overrides == {}
    assert st.sci_pending_turns == 0
    assert st.sci_active is False
    assert st.sci_pending is False
    assert st.sci_variant == ""


def test_governance_service_clear_chat_resets():
    st = types.SimpleNamespace(qc_overrides={"clarity": 2}, sci_pending_turns=7)
    svc = GovernanceService()
    svc.apply_clear_chat_resets(st)
    assert st.qc_overrides == {}
    assert st.sci_pending_turns == 0


def test_governance_service_comm_stop_reset_default_and_custom():
    st = types.SimpleNamespace(comm_active=True)
    svc = GovernanceService()
    svc.apply_comm_stop_resets(st)
    assert st.comm_active is False

    called = {"ok": False}

    def _custom(state):
        called["ok"] = True
        state.comm_active = False
        state.custom = "x"

    st2 = types.SimpleNamespace(comm_active=True)
    svc2 = GovernanceService(apply_comm_stop_fn=_custom)
    svc2.apply_comm_stop_resets(st2)
    assert called["ok"] is True
    assert st2.comm_active is False
    assert st2.custom == "x"


def test_governance_service_apply_legacy_command_variants():
    svc = GovernanceService()
    ruleset = {
        "default_profile": "Standard",
        "profiles": {"Standard": {}, "Expert": {}},
    }

    st = types.SimpleNamespace(
        overlay="",
        color="off",
        sci_pending=False,
        sci_pending_turns=4,
        sci_active=True,
        sci_variant="B",
        comm_active=False,
        active_profile="Expert",
        anchor_auto=True,
        anchor_force_next=True,
        anchor_auto_user_override=False,
        dynamic_one_shot_active=False,
        dynamic_nudge="",
    )

    assert svc.apply_legacy_command(cmd="Strict on", state=st, ruleset_data=ruleset) is True
    assert st.overlay == "Strict"
    assert svc.apply_legacy_command(cmd="Color on", state=st, ruleset_data=ruleset) is True
    assert st.color == "on"
    assert svc.apply_legacy_command(cmd="SCI on", state=st, ruleset_data=ruleset) is True
    assert st.sci_pending is True and st.sci_pending_turns == 0
    assert svc.apply_legacy_command(cmd="SCI off", state=st, ruleset_data=ruleset) is True
    assert st.sci_active is False and st.sci_pending is False and st.sci_variant == ""

    assert svc.apply_legacy_command(cmd="Comm Start", state=st, ruleset_data=ruleset) is True
    assert st.comm_active is True
    assert st.active_profile == "Standard"
    assert svc.apply_legacy_command(cmd="Comm Stop", state=st, ruleset_data=ruleset) is True
    assert st.comm_active is False

    assert svc.apply_legacy_command(cmd="Comm Anchor off", state=st, ruleset_data=ruleset) is True
    assert st.anchor_auto is False and st.anchor_force_next is False and st.anchor_auto_user_override is True
    assert svc.apply_legacy_command(cmd="Comm Anchor on", state=st, ruleset_data=ruleset) is True
    assert st.anchor_auto is True and st.anchor_force_next is False

    assert svc.apply_legacy_command(cmd="Dynamic one-shot on", state=st, ruleset_data=ruleset) is True
    assert st.dynamic_one_shot_active is True and st.dynamic_nudge == "one-shot"
    assert svc.apply_legacy_command(cmd="Unknown", state=st, ruleset_data=ruleset) is False
