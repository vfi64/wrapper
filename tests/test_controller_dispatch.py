from controller import dispatch, dispatch_intent
from intents import SelectProfile
from state import WrapperState


def test_controller_dispatch_comm_start_sets_default_profile_and_effects():
    ruleset = {
        "default_profile": "Standard",
        "profiles": {
            "Standard": {"mode_overlay": "Strict"},
            "Expert": {"mode_overlay": "None"},
        },
    }
    state = WrapperState(active_profile="Expert", comm_active=False, sci_active=True, sci_pending=True, sci_variant="B")
    mirrored = {"called": False}

    def _mirror():
        mirrored["called"] = True

    out = dispatch(cmd="Comm Start", runtime_state=state, ruleset_data=ruleset, mirror_callback=_mirror)
    assert out.applied is True
    assert "recreate_session_with_governance" in out.effects
    assert state.comm_active is True
    assert state.active_profile == "Standard"
    assert state.overlay == "Strict"
    assert state.sci_active is False
    assert state.sci_pending is False
    assert state.sci_variant == ""
    assert mirrored["called"] is True


def test_controller_dispatch_unknown_command_returns_not_applied():
    state = WrapperState()
    out = dispatch(cmd="Comm Unknown", runtime_state=state, ruleset_data={})
    assert out.applied is False
    assert out.effects == []


def test_controller_dispatch_intent_applies_profile_switch_effect():
    ruleset = {"profiles": {"Expert": {"mode_overlay": "None"}}}
    state = WrapperState(active_profile="Standard")
    out = dispatch_intent(
        intent=SelectProfile(profile="Expert"),
        cmd="Profile Expert",
        runtime_state=state,
        ruleset_data=ruleset,
    )
    assert out.applied is True
    assert "profile_switch" in out.effects
    assert state.active_profile == "Expert"
