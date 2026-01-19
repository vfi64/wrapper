import json
import types
import importlib.util
from pathlib import Path

"""Unified pytest suite for fix13.

Place this file in the same folder as:
- 03-12-Gemini_JSON-19.6.8-b20-004-fix13-numericguard-tests.py
- Comm-SCI-v19.6.8.json

Run:
  pytest -vv test_comm_sci_fix13_all.py

This suite avoids starting the GUI or doing real model calls.
"""

HERE = Path(__file__).resolve().parent
FIX_PATH = HERE / '03-12-Gemini_JSON-19.6.8-b20-004-fix13-numericguard-tests.py'
JSON_PATH = HERE / 'Comm-SCI-v19.6.8.json'


def load_fix_module():
    spec = importlib.util.spec_from_file_location('comm_sci_fix13_mod', FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_ruleset_data():
    return json.loads(JSON_PATH.read_text(encoding='utf-8'))


def get_any_profile_command(data):
    commands = (data.get('commands') or {})
    all_cmds = []
    for cat in commands.values():
        if isinstance(cat, dict):
            all_cmds.extend(list(cat.keys()))
    # Prefer "Profile Expert" if present; else any Profile * command.
    if 'Profile Expert' in all_cmds:
        return 'Profile Expert'
    for c in all_cmds:
        if isinstance(c, str) and c.startswith('Profile '):
            return c
    return all_cmds[0] if all_cmds else None


def get_numeric_category(data):
    nc = data.get('numeric_codes') or {}
    cats = nc.get('categories') or []
    for cat in cats:
        opts = (cat or {}).get('options') or {}
        if (cat or {}).get('index') is not None and isinstance(opts, dict) and len(opts) > 0:
            return str((cat or {}).get('index')), opts
    return None, None


def make_api_and_state(*, data, sci_pending: bool = False):
    gov = types.SimpleNamespace(data=data)
    api = types.SimpleNamespace(gov=gov)
    state = types.SimpleNamespace(sci_pending=sci_pending)
    return api, state


class DummyResp:
    def __init__(self, text: str):
        self.text = text


class DummySession:
    """A minimal stub that counts send_message calls and returns queued texts."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def send_message(self, msg: str):
        self.calls.append(msg)
        if self._texts:
            return DummyResp(self._texts.pop(0))
        return DummyResp('')


def _extract_text(out):
    """Api.ask() returns either a plain string or a dict like {'html': ..., 'csc': ...}.
    For assertions we normalize to the rendered text/html.
    """
    if isinstance(out, dict):
        return (out.get('html') or '')
    return out or ''


def _prime_module_gov(mod):
    """Inject canonical JSON into the module-level gov so Api() uses the real rules."""
    data = load_ruleset_data()
    mod.gov.data = data
    mod.gov.loaded = True
    try:
        mod.gov.filepath = str(JSON_PATH)
    except Exception:
        pass
    return data


# ------------------------
# Routing / Numeric guard
# ------------------------

def test_mixed_command_is_not_executed():
    mod = load_fix_module()
    data = load_ruleset_data()
    api, state = make_api_and_state(data=data, sci_pending=False)

    cmd = get_any_profile_command(data)
    assert cmd is not None

    r = mod.route_input(f"{cmd}: What is time?", state, api)
    assert r['kind'] == 'chat'
    assert r.get('standalone_only_violation') is True
    assert r.get('standalone_violation_cmd') == cmd


def test_invalid_numeric_code_blocks_only_for_known_index():
    mod = load_fix_module()
    data = load_ruleset_data()
    api, state = make_api_and_state(data=data, sci_pending=False)

    idx, opts = get_numeric_category(data)
    assert idx is not None and opts is not None

    # Choose an option not in the canonical options.
    invalid_opt = '99'
    if invalid_opt in opts:
        invalid_opt = '98'

    r = mod.route_input(f"{idx}-{invalid_opt}", state, api)
    assert r['kind'] == 'error'
    assert 'Invalid numeric code' in r['html']


def test_date_like_input_is_not_blocked_by_numeric_guard():
    mod = load_fix_module()
    data = load_ruleset_data()
    api, state = make_api_and_state(data=data, sci_pending=False)

    r = mod.route_input('2026-01', state, api)
    assert r['kind'] == 'chat'
    assert r.get('is_numeric_code') is not True


# -----------------------------------------
# SCI pending: extension + timeout behavior
# -----------------------------------------

def test_sci_pending_contextual_query_returns_menu_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    # Make SCI pending
    api.gov_state.sci_pending = True
    api.gov_state.sci_pending_turns = 0

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("What is the SCI trace?")

    assert dummy.calls == [], "Contextual SCI query must not call the model while pending"
    text = _extract_text(out)
    assert isinstance(text, str) and text
    assert 'SCI' in text and ('Variants' in text or 'variants' in text), "Expected SCI menu text"


def test_sci_pending_timeout_assumes_variant_A_after_two_non_selections():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    api.gov_state.sci_pending = True
    api.gov_state.sci_pending_turns = 0

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
        "OK2\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
    ])
    api.chat_session = dummy

    out1 = api.ask("Tell me about time.")
    assert api.gov_state.sci_pending is True
    assert api.gov_state.sci_pending_turns == 1

    out2 = api.ask("And what about entropy?")
    assert api.gov_state.sci_pending is False
    assert api.gov_state.sci_variant == 'A'

    # Should have made two model calls total (one per prompt) in this non-contextual path.
    assert len(dummy.calls) == 2
    assert isinstance(_extract_text(out1), str) and isinstance(_extract_text(out2), str)


# -----------------------------
# Exactly one repair pass
# -----------------------------

def test_one_repair_pass_is_applied_once_when_validator_reports_hard_violations():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Stub CSC strict to a no-op so we can focus purely on repair behavior.
    api._apply_csc_strict = lambda text, user_raw=None, is_command=False: (text, None)

    class DummyValidator:
        def __init__(self):
            self.validate_calls = 0

        def _required_trace_steps_for_variant(self, vk: str):
            return []

        def validate(self, *, text, state, expect_menu, expect_trace, is_command, user_prompt):
            self.validate_calls += 1
            return ["Hard violation: missing contract block"], []

        def build_repair_prompt(self, *, user_prompt, raw_response, state, hard_violations, soft_violations):
            return "REPAIR: produce compliant output"

    api.validator = DummyValidator()

    dummy = DummySession([
        "BAD RESPONSE (no required blocks)",
        "REPAIRED RESPONSE\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
    ])
    api.chat_session = dummy

    out = api.ask("Hello")

    assert len(dummy.calls) == 2, "Must call the model exactly twice: original + one repair pass"
    assert api.validator.validate_calls >= 1
    text = _extract_text(out)
    assert isinstance(text, str) and text
    assert 'REPAIRED RESPONSE' in text
