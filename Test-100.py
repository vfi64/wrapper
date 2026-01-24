import json
import os
import types
import importlib.util
from pathlib import Path

"""Unified pytest suite for Wrapper-100.

Expected repo layout:
- Wrapper-100.py
- Test-100.py
- JSON/Comm-SCI-v19.6.8.json

Run:
  python -m pytest -vv -s --tb=long Test-100.py

This suite avoids starting the GUI or doing real model calls.
"""

HERE = Path(__file__).resolve().parent
FIX_PATH = HERE / 'Wrapper-100.py'
# Canonical ruleset lives in JSON/. Fall back to repo root for older layouts.
JSON_PATH = HERE / 'JSON' / 'Comm-SCI-v19.6.8.json'


def load_fix_module():
    spec = importlib.util.spec_from_file_location('Wrapper-098', FIX_PATH)
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


def _extract_html(out):
    """Return the HTML/text payload from Api.ask() output.

    Some command handlers return a dict like {'html': ..., 'csc': ...} while
    other paths may return a plain string. Tests that care about rendered output
    use this helper.
    """
    return _extract_text(out)


def _prime_module_gov(mod):
    """Inject canonical JSON into the module-level gov so Api() uses the real rules.

    Also force provider selection to Gemini for deterministic unit tests (no network calls).
    """
    data = load_ruleset_data()
    mod.gov.data = data
    mod.gov.loaded = True
    try:
        mod.gov.filepath = str(JSON_PATH)
    except Exception:
        pass

    # Deterministic provider for tests: avoid accidental HTTP calls if user's config selects OpenRouter/HF.
    try:
        cfg = getattr(mod, 'cfg', None)
        conf = getattr(cfg, 'config', None)
        if isinstance(conf, dict):
            conf['active_provider'] = 'gemini'
            # Keep legacy 'model' key coherent for Gemini
            if isinstance(conf.get('model'), str) and conf.get('model').strip():
                pass
            else:
                conf['model'] = conf.get('model') or 'gemini-2.0-flash'
    except Exception:
        pass

    # Force deterministic provider for unit tests: avoid accidental real network calls
    try:
        if hasattr(mod, 'cfg') and getattr(mod, 'cfg', None) is not None:
            c = getattr(mod.cfg, 'config', None)
            if isinstance(c, dict):
                c['active_provider'] = 'gemini'
                # keep a sane default model key for gemini path
                c.setdefault('model', c.get('model') or 'gemini-2.0-flash')
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

    r = mod.route_input(f"{cmd}: What is time?", state, api, api.gov)
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



def test_repair_pass_is_rate_limited_counts_as_second_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Stub CSC strict to a no-op so we can focus purely on repair + limiter behavior.
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

    # Rate limit: allow only ONE call per minute -> repair pass must be blocked.
    api.rate_limit_enabled = True
    api.rate_limiter = mod.RateLimiter(per_minute=1, per_hour=100, clock=lambda: 0.0)

    dummy = DummySession([
        "BAD RESPONSE (no required blocks)",
        "REPAIRED RESPONSE SHOULD NOT BE CONSUMED",
    ])
    api.chat_session = dummy

    out = api.ask("Hello")

    # Only the first model call must happen; repair attempt should be blocked before calling the model.
    assert len(dummy.calls) == 1, "Repair pass must count as a second call and be blocked by the limiter"

    text = _extract_text(out)
    assert "CONTROL LAYER BLOCK" in text
    assert "Reason: repair" in text


# -----------------------------
# Dynamic one-shot reset
# -----------------------------

def test_dynamic_one_shot_resets_after_single_answer():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    # Force dynamic one-shot active BEFORE the call.
    api.gov_state.dynamic_one_shot_active = True
    api.gov_state.dynamic_nudge = 'one-shot'

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)"
    ])
    api.chat_session = dummy

    out = api.ask("Hello")

    # Exactly one model call.
    assert len(dummy.calls) == 1

    # One-shot must auto-reset after a single answer.
    assert bool(getattr(api.gov_state, 'dynamic_one_shot_active', False)) is False
    assert (getattr(api.gov_state, 'dynamic_nudge', '') or '') == ''

    text = _extract_text(out)
    assert isinstance(text, str) and text
    assert 'QC-Matrix:' in text


# -----------------------------
# SCI recursion: depth bound + auto-return
# -----------------------------

def test_sci_recursion_depth_increments_and_auto_returns_to_parent_variant():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    # Ensure SCI is active so "SCI recurse" is accepted.
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = 'A'

    # Prevent real session recreation during command handling.
    api._recreate_chat_session = lambda *a, **k: None

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
    ])
    api.chat_session = dummy

    # Command: should not call the model, only arm recursion for the next turn.
    _cmd_out = api.ask("SCI recurse")
    assert len(dummy.calls) == 0
    assert int(getattr(api.gov_state, 'sci_recursion_depth', 0) or 0) == 1
    assert bool(getattr(api.gov_state, 'sci_recursion_one_shot', False)) is True
    assert (getattr(api.gov_state, 'sci_recursion_parent_variant', '') or '') == 'A'

    # Next normal ask: should call the model once and then auto-return to parent.
    _out = api.ask("Subquestion")
    assert len(dummy.calls) == 1
    assert int(getattr(api.gov_state, 'sci_recursion_depth', 0) or 0) == 0
    assert (getattr(api.gov_state, 'sci_recursion_parent_variant', '') or '') == ''
    assert bool(getattr(api.gov_state, 'sci_recursion_one_shot', False)) is False
    assert (getattr(api.gov_state, 'sci_variant', '') or '') == 'A'


# -----------------------------
# QC delta parsing/enforcement
# -----------------------------

def test_qc_delta_corrected_by_python_enforcement_for_at_least_two_dimensions():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    # Ensure a known profile with qc_target corridor exists.
    api.gov_state.active_profile = 'Standard'

    # Provide deliberately wrong deltas. For Standard:
    # - Clarity 3 is within [2..3] => expected Δ0
    # - Brevity 1 is below [2..2] => expected Δ-1
    dummy = DummySession([
        "Answer\n"
        "QC-Matrix: Clarity 3 (Δ+9) · Brevity 1 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ-7)"
    ])
    api.chat_session = dummy

    out = api.ask("Hello")
    assert len(dummy.calls) == 1

    text = _extract_text(out)
    assert 'Clarity 3 (Δ0)' in text
    assert 'Brevity 1 (Δ-1)' in text

    # The originally wrong deltas must not survive.
    assert 'Δ+9' not in text
    assert 'Δ-7' not in text


# -----------------------------
# Evidence tag normalization
# -----------------------------

def test_evidence_tagging_normalizes_origin_suffix_into_brackets_and_strips_trailing_origin_token():
    mod = load_fix_module()
    _prime_module_gov(mod)

    assert hasattr(mod, 'normalize_evidence_tags'), 'Wrapper must expose normalize_evidence_tags()'

    raw = "Alpha [GREEN] 🟢 -TRAIN Beta [RED] 🔴 -DOC Gamma"
    out = mod.normalize_evidence_tags(raw)

    # Origin suffix must move into the bracket tag.
    assert '[GREEN-TRAIN] 🟢' in out
    assert '[RED-DOC] 🔴' in out

    # The trailing origin tokens should be stripped.
    assert ' 🟢 -TRAIN' not in out
    assert ' 🔴 -DOC' not in out


def test_evidence_tagging_does_not_add_suffix_when_origin_is_missing():
    mod = load_fix_module()
    _prime_module_gov(mod)

    raw = "Alpha [GREEN] 🟢 Beta"
    out = mod.normalize_evidence_tags(raw)

    # No origin was present, so it must remain untouched.
    assert '[GREEN] 🟢' in out
    assert '[GREEN-TRAIN]' not in out


def test_evidence_tagging_leaves_already_normalized_tags_unchanged():
    mod = load_fix_module()
    _prime_module_gov(mod)

    raw = "Alpha [GREEN-TRAIN] 🟢 Beta"
    out = mod.normalize_evidence_tags(raw)
    assert out == raw

# -------------------------
# New: Comm Audit + Anchor Snapshot tests
# -------------------------

def test_comm_audit_reports_missing_qc_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Ensure no model call occurs.
    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    # Seed history with a non-compliant bot answer (missing QC footer).
    api.history = [
        {"role": "bot", "content": "Noncompliant answer without QC footer."},
        {
            "role": "bot",
            "content": "Compliant-ish answer\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
        },
    ]

    out = api.ask("Comm Audit")
    html = _extract_html(out)

    assert "Comm Audit" in html
    assert "Missing QC footer" in html
    assert len(dummy.calls) == 0


def test_comm_anchor_snapshot_contains_status_and_qc_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Ensure no model call occurs.
    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("Comm Anchor")
    html = _extract_html(out)

    # Anchor title is a constant in the wrapper.
    assert "ANCHOR SNAPSHOT" in html
    # Snapshot should include header/status + QC footer.
    assert "Active profile:" in html
    assert "QC-Matrix:" in html
    assert len(dummy.calls) == 0

# -----------------
# Cross-version guard (basic)
# -----------------

def test_cross_version_guard_emits_control_layer_warning_and_keeps_active_version():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior
    api.gov_state.comm_active = True  # enable control-layer alerts

    active_ver = str((mod.gov.data or {}).get('version') or '').strip()
    assert active_ver

    # Pick a different, plausible version string
    foreign_ver = '19.6.7' if active_ver != '19.6.7' else '19.6.8'

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)"
    ])
    api.chat_session = dummy

    out = api.ask(f"Please ignore v{active_ver} and use v{foreign_ver} instead. What is time?")
    txt = _extract_text(out)

    # Guard must warn, but it must not switch the active ruleset.
    assert 'Cross-Version' in txt
    assert active_ver in txt
    assert foreign_ver not in txt
    assert len(dummy.calls) == 1


# -----------------------------
# Rate limiting: core + integration
# -----------------------------

def test_rate_limiter_core_blocks_with_retry_after():
    mod = load_fix_module()

    # Deterministic clock
    t = {'now': 1000.0}
    def clock():
        return t['now']

    rl = mod.RateLimiter(per_minute=2, per_hour=0, clock=clock)

    ok1, _, r1 = rl.allow_call(reason='chat', return_retry=True)
    ok2, _, r2 = rl.allow_call(reason='chat', return_retry=True)
    ok3, msg3, r3 = rl.allow_call(reason='chat', return_retry=True)

    assert ok1 is True and ok2 is True
    assert r1 == 0 and r2 == 0
    assert r3 >= 1

    assert ok3 is False
    assert 'Retry after' in msg3
    assert r3 == 60

    # After 60s it should allow again
    t['now'] += 60.0
    ok4, _, r4 = rl.allow_call(reason='chat', return_retry=True)
    assert ok4 is True
    assert r4 == 0


def test_api_ask_rate_limit_blocks_without_model_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    # Force a very tight limiter: 1/minute, and consume one slot immediately.
    api.rate_limit_enabled = True
    api.rate_limiter = mod.RateLimiter(per_minute=1, per_hour=0)
    _ = api.rate_limiter.allow_call(reason='pre')

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
    ])
    api.chat_session = dummy

    out = api.ask("Hello")

    # Must not call the model if blocked.
    assert dummy.calls == []

    text = _extract_text(out)
    assert isinstance(text, str) and text
    assert 'CONTROL LAYER BLOCK' in text
    assert 'Rate limit exceeded' in text
    assert 'Retry after' in text


def test_no_network_calls_via_urllib_urlopen(monkeypatch):
    # Safety net: unit tests must never perform real HTTP calls.
    import urllib.request as _urlreq

    def _boom(*args, **kwargs):
        raise AssertionError("Network call attempted (urllib.request.urlopen)")

    monkeypatch.setattr(_urlreq, "urlopen", _boom, raising=True)

    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    dummy = DummySession([
        "OK\nQC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)",
    ])
    api.chat_session = dummy

    _out = api.ask("Hello")
    assert len(dummy.calls) == 1


from datetime import datetime


def test_export_v2_schema():
    mod = load_fix_module()
    api = mod.Api()
    # create at least one history entry
    api.history.append({'role': 'user', 'content': 'test', 'ts': datetime.now().isoformat()})
    _, audit_path = api.export_audit_v2(audit_only=True)
    data = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    assert data.get('export_version') == '2.0'
    assert 'session_metadata' in data
    assert 'environment' in data
    assert 'provider_config' in data
    assert 'governance_config' in data


def test_export_v2_no_secrets():
    mod = load_fix_module()
    api = mod.Api()
    api.history.append({'role': 'user', 'content': 'test', 'ts': datetime.now().isoformat()})
    _, audit_path = api.export_audit_v2(audit_only=True)
    raw = Path(audit_path).read_text(encoding='utf-8')
    # Must not contain typical key/token prefixes
    assert 'sk-' not in raw
    assert 'hf_' not in raw
    # Allow mentioning env var names as sources
    assert '"api_key"' not in raw.lower()  # should not contain actual key fields


def test_export_v2_timestamps_present():
    mod = load_fix_module()
    api = mod.Api()
    api.history.append({'role': 'user', 'content': 'u', 'ts': datetime.now().isoformat()})
    api.history.append({'role': 'bot', 'content': 'b', 'ts': datetime.now().isoformat()})
    _, audit_path = api.export_audit_v2(audit_only=True)
    data = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    for msg in data.get('conversation', []):
        assert 'ts' in msg


def test_export_v2_provider_config_minimum():
    mod = load_fix_module()
    api = mod.Api()
    api.history.append({'role': 'user', 'content': 'test', 'ts': datetime.now().isoformat()})
    _, audit_path = api.export_audit_v2(audit_only=True)
    data = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    pc = data.get('provider_config') or {}
    assert 'active_provider' in pc
    assert 'model' in pc


def test_export_v2_ruleset_hash_present_or_unknown():
    mod = load_fix_module()
    api = mod.Api()
    api.history.append({'role': 'user', 'content': 'test', 'ts': datetime.now().isoformat()})
    _, audit_path = api.export_audit_v2(audit_only=True)
    data = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    rh = (data.get('governance_config') or {}).get('ruleset_hash', 'unknown')
    assert rh == 'unknown' or str(rh).startswith('sha256:')
def test_b6_set_api_key_persists_without_leaking_to_audit(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # Redirect config + logs to temp
    mod.PROJECT_DIR = str(tmp_path)
    mod.CONFIG_DIR = str(tmp_path / 'Config')
    mod.LOGS_DIR = str(tmp_path / 'Logs')
    mod.AUDIT_LOG_DIR = str(tmp_path / 'Logs' / 'Audit')
    mod.CHAT_LOG_DIR = str(tmp_path / 'Logs' / 'Chats')
    for d in [mod.CONFIG_DIR, mod.LOGS_DIR, mod.AUDIT_LOG_DIR, mod.CHAT_LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    mod.KEYS_PATH = os.path.join(mod.CONFIG_DIR, mod.KEYS_FILENAME)
    mod.KEYS_EXAMPLE_PATH = os.path.join(mod.CONFIG_DIR, mod.KEYS_EXAMPLE_FILENAME)

    secret = "sk-THIS_IS_A_TEST_SECRET_DO_NOT_LEAK"
    res = api.set_api_key_for_provider('openrouter', secret, write_path=mod.KEYS_PATH)
    assert res.get('ok') is True

    # Export audit v2 and ensure secret is NOT present
    _, audit_path = api.export_audit_v2(audit_only=True)
    with open(audit_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    assert secret not in raw
    assert "api_key_source" in raw

def test_b7_load_log_from_path_and_fork(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    legacy_path = tmp_path / "Log_legacy.json"
    legacy = {
        "meta": "x",
        "model": "dummy",
        "history": [
            {"role": "user", "content": "hi", "ts": "2026-01-01T00:00:00"},
            {"role": "bot", "content": "<b>hello</b>", "ts": "2026-01-01T00:00:01"},
        ],
    }
    legacy_path.write_text(json.dumps(legacy), encoding='utf-8')

    sid_before = getattr(api, 'session_id', None)
    res = api.load_log_from_path(str(legacy_path), fork=False)
    assert res.get('ok') is True
    assert len(api.history) == 2
    assert api.history[0]['content'] == "hi"
    assert getattr(api, 'session_id', None) == sid_before

    api2 = mod.Api()
    sid2_before = getattr(api2, 'session_id', None)
    res2 = api2.load_log_from_path(str(legacy_path), fork=True)
    assert res2.get('ok') is True
    assert len(api2.history) == 2
    assert getattr(api2, 'session_id', None) != sid2_before


def test_panel_ping_exists_and_returns_ok():
    mod = load_fix_module()
    api = mod.Api()
    assert hasattr(api, 'ping')
    res = api.ping()
    assert isinstance(res, dict)
    assert res.get('ok') is True

def test_panel_get_ui_returns_minimum_keys():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    ui = api.get_ui()
    assert isinstance(ui, dict)
    assert 'providers' in ui
    assert 'current_provider' in ui
    assert 'current_model' in ui
    assert 'available_models' in ui

    # When the ruleset is primed, UI must include command sections for the Panel buttons.
    assert isinstance(ui.get('comm'), list)
    assert len(ui.get('comm')) > 0
    assert isinstance(ui.get('profiles'), list)
    assert len(ui.get('profiles')) > 0

    # Log list keys must exist (may be empty in tests, but must not crash).
    assert 'chat_logs' in ui


def test_panel_get_ui_safe_without_priming():
    mod = load_fix_module()
    api = mod.Api()
    ui = api.get_ui()
    assert isinstance(ui, dict)
    assert 'providers' in ui
    assert 'current_provider' in ui
    assert 'current_model' in ui
    assert 'available_models' in ui

def test_list_chat_logs_safe_and_returns_list():
    mod = load_fix_module()
    api = mod.Api()
    assert hasattr(api, 'list_chat_logs')
    res = api.list_chat_logs()
    assert isinstance(res, dict)
    assert res.get('ok') is True
    assert isinstance(res.get('logs'), list)


def test_panel_action_list_chat_logs_returns_list():
    mod = load_fix_module()
    api = mod.Api()
    assert hasattr(api, 'panel_action')
    res = api.panel_action('list_chat_logs', {'limit': 10})
    assert isinstance(res, dict)
    assert res.get('ok') is True
    assert isinstance(res.get('logs'), list)


def test_panel_action_cmd_executes_local_command_without_model_call():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    # Guard: if the implementation accidentally tries to call the model, we'd see a send_message call.
    api.chat_session = DummySession(['LLM'])  # type: ignore[attr-defined]
    out = api.panel_action('cmd', {'text': 'Comm State'})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    # Must not call the model for deterministic local commands
    assert api.chat_session.calls == []  # type: ignore[attr-defined]
    # panel_action('cmd') queues into the main UI pipeline; it returns metadata only.
    assert out.get('queued') in (True, None)

# ------------------------
# Panel bridge wiring
# ------------------------

def test_panel_html_uses_panel_action_and_not_remote_cmd():
    mod = load_fix_module()
    assert isinstance(getattr(mod, 'HTML_PANEL', None), str)
    html = mod.HTML_PANEL
    # Panel must not rely on remote_cmd injection (which is backend-unstable)
    assert 'panel_action' in html
    assert 'remote_cmd' not in html


def test_panel_bridge_forwards_ping_get_ui_and_panel_action():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    pb = getattr(api, 'panel_bridge', None)
    assert pb is not None, 'Api must expose panel_bridge'

    r = pb.ping()
    assert isinstance(r, dict)
    assert r.get('ok') is True

    ui = pb.get_ui()
    assert isinstance(ui, dict)
    assert 'providers' in ui and 'current_provider' in ui

    # Local command via panel_action must not call model
    api.chat_session = DummySession(['SHOULD NOT BE USED'])
    out = pb.panel_action('cmd', {'text': 'Comm State'})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    # Must NOT call the model; command is queued into main UI pipeline.
    assert api.chat_session.calls == []


def test_panel_action_cmd_uses_remote_cmd_hook():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    called = {'n': 0, 'last': None}

    def _rc(cmd):
        called['n'] += 1
        called['last'] = cmd

    api.remote_cmd = _rc  # type: ignore[assignment]

    out = api.panel_action('cmd', {'text': 'Comm Start'})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    assert called['n'] == 1
    assert called['last'] == 'Comm Start'


def test_panel_action_refresh_models_accepts_provider_param_without_crash():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    calls = {'set_provider': 0, 'refresh_models': 0}

    def _sp(p):
        calls['set_provider'] += 1
        return {'ok': True, 'provider': p}

    def _rm():
        calls['refresh_models'] += 1
        return {'ok': True}

    api.set_provider = _sp  # type: ignore[assignment]
    api.refresh_models = _rm  # type: ignore[assignment]

    out = api.panel_action('refresh_models', {'provider': 'openrouter'})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    assert calls['set_provider'] == 1
    assert calls['refresh_models'] == 1



def test_refresh_models_hf_populates_cache():
    """When provider is huggingface, refresh_models() must populate _hf_models_cache so the panel dropdown is not empty."""
    mod = load_fix_module()

    class DummyPR:
        def get_active_provider(self):
            return 'huggingface'
        def get_huggingface_models_cached(self, force_refresh=False):
            return (['hf/model-a', 'hf/model-b'], {'source': 'test'})

    api = mod.Api()
    api.provider_router = DummyPR()
    api.main_win = None
    api.panel_win = None

    res = api.refresh_models()
    assert isinstance(res, dict)
    assert res.get('status') is True
    assert res.get('provider') in ('huggingface', 'hf')
    assert getattr(api, '_hf_models_cache', None) == ['hf/model-a', 'hf/model-b']


def test_ui_replay_loaded_history_fallback_incremental():
    """_ui_replay_loaded_history should fall back to incremental replay if resetChatFromHistory is unavailable/fails."""
    mod = load_fix_module()

    class DummyWin:
        def __init__(self):
            self.calls = []
        def evaluate_js(self, js):
            self.calls.append(js)
            # Simulate that the bulk helper is missing/fails
            if 'resetChatFromHistory' in js:
                return 'NOFUNC'
            return 'OK'

    api = mod.Api()
    api.main_win = DummyWin()
    api.history = [
        {'role': 'system', 'content': 'sys msg'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello <b>world</b>'},
    ]
    api._ui_replay_loaded_history(status_msg='Loaded X')

    # Must attempt reset status and then add messages
    joined = '\n'.join(api.main_win.calls)
    assert 'resetChatToStatus' in joined or 'resetChatToStatus' in joined  # status reset call
    assert "addMsg('user'" in joined
    assert "addMsg('bot'" in joined


def test_panel_action_clear_chat_resets_history_and_calls_resetChatToStatus():
    mod = load_fix_module()

    class DummyWin:
        def __init__(self):
            self.calls = []
        def evaluate_js(self, js):
            self.calls.append(js)
            return 'OK'

    api = mod.Api()
    api.main_win = DummyWin()
    api.history = [{'role': 'user', 'content': 'hi'}]

    res = api.panel_action('clear_chat', {})
    assert isinstance(res, dict)
    assert res.get('ok') is True
    assert getattr(api, 'history', None) == []
    joined = '\n'.join(api.main_win.calls)
    assert 'resetChatToStatus' in joined


def test_bind_panel_window_events_attaches_closing_and_closed_handlers():
    mod = load_fix_module()

    class Hook:
        def __init__(self):
            self.handlers = []
        def __iadd__(self, fn):
            self.handlers.append(fn)
            return self

    class Events:
        def __init__(self):
            self.closing = Hook()
            self.closed = Hook()

    class DummyWin:
        def __init__(self):
            self.events = Events()

    api = mod.Api()
    w = DummyWin()

    api._bind_panel_window_events(w)

    assert api.on_panel_closing in w.events.closing.handlers
    assert api.on_panel_closed in w.events.closed.handlers


def test_on_panel_closing_returns_false_and_attempts_hide():
    mod = load_fix_module()
    api = mod.Api()

    called = {"hide": False}
    def fake_hide_panel():
        called["hide"] = True

    api._hide_panel = fake_hide_panel  # type: ignore[assignment]
    res = api.on_panel_closing()

    assert res is False
    assert called["hide"] is True

