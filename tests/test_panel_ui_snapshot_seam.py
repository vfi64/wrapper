from __future__ import annotations

import panel_ui_snapshot_seam as sut


def test_panel_ui_default_snapshot_contains_minimum_panel_keys():
    ui = sut.panel_ui_default_snapshot()
    assert isinstance(ui, dict)
    for key in ("providers", "current_provider", "current_model", "available_models", "answer_language", "language_policy_mode", "comm", "profiles", "chat_logs"):
        assert key in ui
    assert ui["current_provider"] == "gemini"
    assert ui["answer_language"] == "de"
    assert ui["language_policy_mode"] == "production"


def test_panel_ui_apply_basic_runtime_normalizes_and_keeps_defaults_when_values_missing():
    ui = sut.panel_ui_default_snapshot()
    out = sut.panel_ui_apply_basic_runtime(
        ui,
        current_provider=" OpenRouter ",
        current_model=" openai/gpt-oss-120b ",
        available_models=["a", "b"],
        answer_language="EN",
    )
    assert out is ui
    assert ui["current_provider"] == "openrouter"
    assert ui["current_model"] == "openai/gpt-oss-120b"
    assert ui["available_models"] == ["a", "b"]
    assert ui["answer_language"] == "en"

    out2 = sut.panel_ui_apply_basic_runtime(ui, current_provider="", current_model="", available_models=None, answer_language="fr")
    assert out2 is ui
    assert ui["current_provider"] == "openrouter"
    assert ui["current_model"] == "openai/gpt-oss-120b"
    assert ui["answer_language"] == "en"


def test_panel_ui_probe_and_apply_basic_runtime_collects_values_from_router_cfg_and_model_loader():
    class _Router:
        def get_active_provider(self):
            return " OpenRouter "

    class _Cfg:
        def get_provider_model(self, provider):
            assert provider == "openrouter"
            return " openrouter/model-1 "

        def get_answer_language(self):
            return "EN"

        def get_language_policy_mode(self):
            return "benchmark"

    seen = {}

    def _get_models(provider):
        seen["provider"] = provider
        return ["m1", "m2"]

    ui = sut.panel_ui_default_snapshot()
    out = sut.panel_ui_probe_and_apply_basic_runtime(
        ui,
        provider_router=_Router(),
        cfg_obj=_Cfg(),
        get_available_models_fn=_get_models,
    )
    assert out is ui
    assert seen["provider"] == "openrouter"
    assert ui["current_provider"] == "openrouter"
    assert ui["current_model"] == "openrouter/model-1"
    assert ui["available_models"] == ["m1", "m2"]
    assert ui["answer_language"] == "en"
    assert ui["language_policy_mode"] == "benchmark"


def test_panel_ui_probe_and_apply_basic_runtime_failsoft_keeps_defaults_on_errors():
    class _Router:
        def get_active_provider(self):
            raise RuntimeError("boom")

    class _Cfg:
        def get_provider_model(self, provider):
            raise RuntimeError("boom")

        def get_answer_language(self):
            raise RuntimeError("boom")

    ui = sut.panel_ui_default_snapshot()
    out = sut.panel_ui_probe_and_apply_basic_runtime(
        ui,
        provider_router=_Router(),
        cfg_obj=_Cfg(),
        get_available_models_fn=lambda provider: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert out is ui
    assert ui["current_provider"] == "gemini"
    assert ui["current_model"] == "gemini-2.0-flash"
    assert ui["answer_language"] == "de"


def test_panel_ui_apply_chat_log_listing_sets_selected_first_entry():
    ui = sut.panel_ui_default_snapshot()
    out = sut.panel_ui_apply_chat_log_listing(ui, logs=["b.json", "a.json"])
    assert out is ui
    assert ui["chat_logs"] == ["b.json", "a.json"]
    assert ui["chat_log_selected"] == "b.json"

    ui2 = sut.panel_ui_default_snapshot()
    sut.panel_ui_apply_chat_log_listing(ui2, logs=[])
    assert ui2["chat_logs"] == []
    assert "chat_log_selected" not in ui2


def test_panel_ui_merge_governance_ui_merges_known_lists_optionals_and_answer_language():
    ui = sut.panel_ui_default_snapshot()
    out = sut.panel_ui_merge_governance_ui(
        ui,
        gov_ui={
            "comm": ["Comm Start"],
            "profiles": ["Standard"],
            "logs": ["Comm State"],
            "current_rule_file": "Comm-SCI-v20.0.3.json",
            "version": "20.0.3",
            "loaded": True,
            "answer_language": "en",
            "ignored": "x",
        },
    )
    assert out is ui
    assert ui["comm"] == ["Comm Start"]
    assert ui["profiles"] == ["Standard"]
    assert ui["logs"] == ["Comm State"]
    assert ui["current_rule_file"] == "Comm-SCI-v20.0.3.json"
    assert ui["version"] == "20.0.3"
    assert ui["loaded"] is True
    assert ui["answer_language"] == "en"
    assert "ignored" not in ui


def test_panel_ui_apply_anchor_toggle_replaces_off_on_pair_with_single_stateful_button():
    ui = sut.panel_ui_default_snapshot()
    ui["comm"] = ["Comm Start", "Comm Anchor off", "Comm Anchor on", "Comm State"]
    out = sut.panel_ui_apply_anchor_toggle(ui, anchor_auto=True)
    assert out is ui
    comm = ui["comm"]
    assert "Comm Anchor off" not in comm and "Comm Anchor on" not in comm
    assert isinstance(comm[1], dict)
    assert comm[1]["cmd"] == "Comm Anchor off"

    ui2 = sut.panel_ui_default_snapshot()
    ui2["comm"] = ["A", "Comm Anchor off", "B", "Comm Anchor on"]
    sut.panel_ui_apply_anchor_toggle(ui2, anchor_auto=False)
    assert any(isinstance(x, dict) and x.get("cmd") == "Comm Anchor on" for x in ui2["comm"])


def test_panel_ui_apply_failsoft_comm_toggle_collapses_comm_start_stop_pair():
    ui = sut.panel_ui_default_snapshot()
    ui["comm"] = ["Comm Start", "Comm Stop", "Comm State"]
    out = sut.panel_ui_apply_failsoft_comm_toggle(ui, comm_active=True)
    assert out is ui
    assert isinstance(ui["comm"][0], dict)
    assert ui["comm"][0]["cmd"] == "Comm Stop"
    assert "Comm Start" not in ui["comm"]
    assert "Comm Stop" not in ui["comm"]

    ui2 = sut.panel_ui_default_snapshot()
    ui2["comm"] = ["Comm Start", "Comm Stop"]
    sut.panel_ui_apply_failsoft_comm_toggle(ui2, comm_active=False)
    assert ui2["comm"][0]["cmd"] == "Comm Start"


def test_panel_ui_apply_comm_visibility_gate_keeps_only_comm_start_when_inactive():
    ui = sut.panel_ui_default_snapshot()
    ui["comm"] = [
        {"name": "Comm ⏻: ON", "cmd": "Comm Start", "desc": "toggle"},
        "Comm Stop",
        "Comm State",
    ]
    ui["profiles"] = ["Standard"]
    ui["sci"] = ["SCI on"]
    ui["overlays"] = ["Strict on"]
    ui["tools"] = ["Tool"]
    ui["logs"] = ["Log"]

    out = sut.panel_ui_apply_comm_visibility_gate(ui, comm_active=False)
    assert out is ui
    assert ui["comm_active"] is False
    assert ui["manual_test_visible"] is False
    assert ui["qc_override_visible"] is False
    assert ui["comm"] == [{"name": "Comm Start", "cmd": "Comm Start", "desc": "toggle"}]
    assert ui["profiles"] == []
    assert ui["sci"] == []
    assert ui["overlays"] == []
    assert ui["tools"] == []
    assert ui["logs"] == []

    ui2 = sut.panel_ui_default_snapshot()
    sut.panel_ui_apply_comm_visibility_gate(ui2, comm_active=True)
    assert ui2["comm_active"] is True
    assert ui2["manual_test_visible"] is True
    assert ui2["qc_override_visible"] is True


def test_panel_ui_apply_legacy_aliases_sets_provider_and_model_alias_keys():
    ui = sut.panel_ui_default_snapshot()
    ui["current_provider"] = "openrouter"
    ui["current_model"] = "x/y"
    out = sut.panel_ui_apply_legacy_aliases(ui)
    assert out is ui
    assert ui["provider"] == "openrouter"
    assert ui["model"] == "x/y"


def test_panel_ui_failopen_snapshot_keeps_minimum_shape_and_comm_visibility_flags():
    class _Gov:
        comm_active = True

    out = sut.panel_ui_failopen_snapshot(gov_state=_Gov())
    assert isinstance(out, dict)
    assert out["current_provider"] == "gemini"
    assert out["current_model"] == "gemini-2.0-flash"
    assert out["comm_active"] is True
    assert out["manual_test_visible"] is True
    assert out["qc_override_visible"] is True
    assert out["provider"] == "gemini"
    assert out["model"] == "gemini-2.0-flash"
    assert isinstance(out.get("comm"), list) and len(out["comm"]) == 1
    assert out["comm"][0]["cmd"] == "Comm Start"


def test_panel_ui_apply_state_postprocess_runs_anchor_normalize_gate_and_aliases():
    class _Gov:
        anchor_auto = True
        comm_active = True
        sci_pending = False
        sci_active = False
        overlay = "Strict"
        color = "on"

    calls = {"normalize": 0, "snapshot": None}

    def _snapshot_ctor(**kwargs):
        calls["snapshot"] = dict(kwargs)
        return {"ok": True, **kwargs}

    def _normalize(ui, snapshot):
        calls["normalize"] += 1
        ui = dict(ui)
        ui["normalized"] = bool(snapshot.get("ok"))
        return ui

    ui = sut.panel_ui_default_snapshot()
    ui["comm"] = ["Comm Start", "Comm Stop", "Comm Anchor off", "Comm Anchor on"]
    out = sut.panel_ui_apply_state_postprocess(
        ui,
        gov_state=_Gov(),
        panel_state_snapshot_ctor=_snapshot_ctor,
        panel_normalize_ui_fn=_normalize,
    )
    assert isinstance(out, dict)
    assert calls["normalize"] == 1
    assert calls["snapshot"]["comm_active"] is True
    assert out["normalized"] is True
    assert out["comm_active"] is True
    assert out["manual_test_visible"] is True
    assert out["qc_override_visible"] is True
    assert out["provider"] == out["current_provider"]
    assert out["model"] == out["current_model"]
    assert any(isinstance(x, dict) and x.get("cmd") == "Comm Anchor off" for x in out.get("comm", []))


def test_panel_ui_apply_state_postprocess_falls_back_to_comm_toggle_and_comm_off_gate():
    class _Gov:
        anchor_auto = False
        comm_active = False
        sci_pending = False
        sci_active = False
        overlay = ""
        color = "off"

    ui = sut.panel_ui_default_snapshot()
    ui["comm"] = ["Comm Start", "Comm Stop", "Comm Anchor off", "Comm Anchor on", "Comm State"]
    ui["profiles"] = ["Standard"]
    out = sut.panel_ui_apply_state_postprocess(
        ui,
        gov_state=_Gov(),
        panel_state_snapshot_ctor=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("ctor boom")),
        panel_normalize_ui_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("normalize boom")),
    )
    assert out["comm_active"] is False
    assert out["manual_test_visible"] is False
    assert out["qc_override_visible"] is False
    assert out["profiles"] == []
    assert out["comm"][0]["cmd"] == "Comm Start"
    assert out["provider"] == out["current_provider"]
    assert out["model"] == out["current_model"]


def test_panel_ui_build_snapshot_composes_runtime_gov_postprocess_and_chat_logs():
    class _Router:
        def get_active_provider(self):
            return "gemini"

    class _Cfg:
        def get_provider_model(self, provider):
            return "gemini-2.0-flash"
        def get_answer_language(self):
            return "de"

    class _GovObj:
        def get_ui_data(self):
            return {
                "comm": ["Comm Start", "Comm Stop", "Comm Anchor off", "Comm Anchor on"],
                "profiles": ["Standard"],
                "answer_language": "de",
            }

    class _GovState:
        anchor_auto = True
        comm_active = False
        sci_pending = False
        sci_active = False
        overlay = "strict"
        color = "on"

    def _state_ctor(**kwargs):
        return kwargs

    def _normalize(ui, snapshot):
        # Identity-ish normalizer to prove hook execution without changing semantics.
        ui = dict(ui)
        ui["normalize_seen"] = bool(snapshot.get("comm_active") is False)
        return ui

    def _list_logs(limit=200):
        assert limit == 200
        return {"ok": True, "logs": ["Log_2.json", "Log_1.json"]}

    out = sut.panel_ui_build_snapshot(
        provider_router=_Router(),
        cfg_obj=_Cfg(),
        get_available_models_fn=lambda provider: ["gemini-2.0-flash"],
        gov_obj=_GovObj(),
        gov_state=_GovState(),
        panel_state_snapshot_ctor=_state_ctor,
        panel_normalize_ui_fn=_normalize,
        list_chat_logs_fn=_list_logs,
        chat_log_limit=200,
    )
    assert out["current_provider"] == "gemini"
    assert out["current_model"] == "gemini-2.0-flash"
    assert out["profiles"] == []
    assert out["comm_active"] is False
    assert out["manual_test_visible"] is False
    assert out["qc_override_visible"] is False
    assert out["chat_logs"] == ["Log_2.json", "Log_1.json"]
    assert out["chat_log_selected"] == "Log_2.json"
    assert out["provider"] == out["current_provider"]
    assert out["model"] == out["current_model"]
