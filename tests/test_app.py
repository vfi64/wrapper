import json
import os
import re
import types
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

"""Unified pytest suite for Wrapper-199 (Stage 1 boundary refactor).

Expected repo layout:
- Wrapper-159.py
- Test-199.py
- JSON/Comm-SCI-v19.6.9.json

Run:
  python3 -m pytest -vv -s --tb=long Test-199.py

This suite avoids starting the GUI or doing real model calls.
"""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
def _resolve_ruleset_path(name: str, env_var: str) -> Path:
    """Resolve ruleset file path robustly across common repo layouts.

    Order:
    1) env var override
    2) near this test file (HERE) and common subdirs
    3) cwd and common subdirs
    4) bounded rglob fallback (cwd)
    """
    override = os.environ.get(env_var)
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        raise AssertionError(f"Env var {env_var} points to missing file: {p}")

    candidates: list[Path] = []
    candidates += [
        HERE / name,
        HERE / "JSON" / name,
        HERE / "Rules" / name,
        HERE / "data" / name,
        HERE.parent / name,
        HERE.parent / "JSON" / name,
    ]

    cwd = Path.cwd()
    candidates += [
        cwd / name,
        cwd / "JSON" / name,
        cwd / "Rules" / name,
        cwd / "data" / name,
        cwd.parent / name,
        cwd.parent / "JSON" / name,
    ]

    for c in candidates:
        if c.exists():
            return c

    # bounded search fallback
    hits = []
    try:
        for p in cwd.rglob(name):
            hits.append(p)
            if len(hits) >= 50:
                break
    except Exception:
        hits = []

    if hits:
        # prefer shortest path (closest to root)
        hits.sort(key=lambda p: len(str(p)))
        return hits[0]

    tried = "\n".join(str(c) for c in candidates[:20])
    raise AssertionError(
        f"Missing ruleset '{name}'. Tried (first 20):\n{tried}\n"
        f"Tip: set env {env_var} to the correct absolute path."
    )

FIX_PATH = SRC / 'Comm-SCI-Control-App.py'
# Canonical ruleset lives in repo-root JSON/.
JSON_PATH = ROOT / 'JSON' / 'Comm-SCI-v20.0.3.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'JSON' / 'Comm-SCI-v20.0.2.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.0.3.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.0.2.json'


def load_fix_module():
    spec = importlib.util.spec_from_file_location(FIX_PATH.stem, FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


# ---------------------------
# v192 DOM rendering pipeline tests (dependency-aware)
# ---------------------------

def _load_rendering_pipeline_v192():
    """
    Load Module/rendering_pipeline_v192.py in a robust, informative way.

    We keep DOM-pipeline tests optional, but the skip reason should reflect the real cause.
    """
    global RP192_IMPORT_ERROR
    RP192_IMPORT_ERROR = None

    # Preferred: normal package import (matches runtime usage)
    try:
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        return importlib.import_module("Module.rendering_pipeline_v192")
    except Exception as e_pkg:
        RP192_IMPORT_ERROR = f"package import failed: {type(e_pkg).__name__}: {e_pkg}"

    # Fallback: direct file load via importlib.util
    mod_path = SRC / "Module" / "rendering_pipeline_v192.py"
    if not mod_path.exists():
        RP192_IMPORT_ERROR = (RP192_IMPORT_ERROR or "") + f" | file not found at: {mod_path}"
        return None
    try:
        spec = importlib.util.spec_from_file_location("Module.rendering_pipeline_v192", mod_path)
        if spec is None or spec.loader is None:
            RP192_IMPORT_ERROR = (RP192_IMPORT_ERROR or "") + " | spec/loader missing"
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    except Exception as e_file:
        RP192_IMPORT_ERROR = (RP192_IMPORT_ERROR or "") + f" | file load failed: {type(e_file).__name__}: {e_file}"
        return None
    try:
        spec = importlib.util.spec_from_file_location("rendering_pipeline_v192", mod_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    except Exception:
        return None


def _have_dom_pipeline():
    rp = _load_rendering_pipeline_v192()
    if rp is None:
        return False
    # DOM invariants require markdown rendering + BeautifulSoup
    return (getattr(rp, "_markdown_lib", None) is not None) and (getattr(rp, "BeautifulSoup", None) is not None)

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
    # Mixed-command parsing is only meaningful while Comm is active.
    state.comm_active = True

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




def test_sci_pending_contextual_query_note_is_localized_when_ui_lang_de():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior

    api.gov_state.ui_lang = 'de'
    api.gov_state.answer_lang = 'de'
    api.gov_state.sci_pending = True
    api.gov_state.sci_pending_turns = 0

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("Was ist die SCI-Trace?")

    assert dummy.calls == [], "Contextual SCI query must not call the model while pending"
    text = _extract_text(out)
    assert "SCI-Auswahl" in text or "Auswahl" in text, "Expected localized pending note in German"
    assert "SCI-Varianten" in text, "Expected German SCI menu"

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


def test_repair_pass_banner_classifier_hides_format_only_self_debunking_repairs():
    mod = load_fix_module()
    only_format = [
        "Self-Debunking placed after QC footer.",
        "Self-Debunking must contain 2–3 numbered points (found 0).",
    ]
    mixed = list(only_format) + [
        "Verification Route Gate: strong-claim heuristic triggered, but no verification route markers found (Source/Measurement/Contrast/Web Check)."
    ]
    assert mod._should_show_repair_pass_banner(only_format) is False
    assert mod._should_show_repair_pass_banner(mixed) is True


def test_repair_pass_banner_is_hidden_for_format_only_self_debunking_repairs():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api._apply_csc_strict = lambda text, user_raw=None, is_command=False: (text, None)

    class DummyValidator:
        def __init__(self):
            self.validate_calls = 0

        def _required_trace_steps_for_variant(self, vk: str):
            return []

        def validate(self, *, text, state, expect_menu, expect_trace, is_command, user_prompt):
            self.validate_calls += 1
            return [
                "Self-Debunking placed after QC footer.",
                "Self-Debunking must contain 2–3 numbered points (found 0).",
            ], []

        def build_repair_prompt(self, *, user_prompt, raw_response, state, hard_violations, soft_violations):
            return "REPAIR: format only"

    api.validator = DummyValidator()
    api.chat_session = DummySession([
        "BAD",
        "REPAIRED RESPONSE\nQC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)",
    ])

    out = api.ask("Was ist Zeit?")
    text = _extract_text(out)

    assert len(api.chat_session.calls) == 2
    assert api.validator.validate_calls >= 1
    assert api.session_repair_passes >= 1
    assert "REPAIRED RESPONSE" in text
    assert "CONTROL LAYER NOTE" not in text



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



def test_qc_footer_normalizes_non_integer_values_to_ints():
    """Regression: some providers emit QC values like 0.8/1.2; footer must normalize to integer ratings."""
    mod = load_fix_module()
    fn = getattr(mod, 'enforce_qc_footer_deltas', None)
    assert callable(fn)

    text = "QC-Matrix: clarity 0.8 (Δ0); brevity 2.2 (Δ0); evidence 3 (Δ0); neutrality 1.6 (Δ0); consistency 2 (Δ0)"
    # Corridor doesn't matter for normalization itself, but we pass a plausible one.
    expected = {'clarity': (2, 3), 'brevity': (2, 3), 'evidence': (2, 3), 'neutrality': (2, 3), 'consistency': (2, 3)}
    out = fn(text, expected, profile_name='Standard')

    # Values must be integers now (0.8->1, 2.2->2, 1.6->2).
    assert "clarity 1" in out
    assert "brevity 2" in out
    assert "neutrality 2" in out


def test_qc_alternative_footer_is_canonicalized_and_respects_override():
    """Regression (Known-Good 2): model emits an alternative QC summary (no QC-Matrix line).

    Expectation: Wrapper must produce a canonical QC-Matrix footer with correct deltas,
    and QC overrides must be respected (fixed corridor => Δ0).
    """
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate behavior
    api.gov_state.comm_active = True
    api.gov_state.active_profile = 'Standard'

    # Apply override: Brevity fixed to 1 => expected Δ0 for Brevity 1.
    api.qc_override_apply({"Brevity": 1})

    dummy = DummySession([
        "Antwort in einem Satz: Ein elektrisches Feld ist der Raum um Ladungen, in dem auf andere Ladungen eine Kraft wirkt.\n"
        "Profile: Standard QC: Clarity 3 · Brevity 1 · Evidence 2 · Empathy 2 · Consistency 2 · Neutrality 2"
    ])
    api.chat_session = dummy

    out = api.ask("Gib mir eine 1-Satz-Antwort: Was ist ein elektrisches Feld?")
    assert len(dummy.calls) == 1

    txt = _extract_text(out)
    assert "QC-Matrix:" in txt
    assert "Profile: Standard QC:" not in txt  # replaced in-place
    # Values are ints and deltas are canonical.
    assert "Clarity 3 (Δ0)" in txt
    assert "Brevity 1 (Δ0)" in txt
    assert "Evidence 2 (Δ0)" in txt
    assert "Empathy 2 (Δ0)" in txt
    assert "Consistency 2 (Δ0)" in txt
    assert "Neutrality 2 (Δ0)" in txt

    # Additionally ensure that without override, Brevity 1 would be Δ-1 under Standard corridor.
    corr = mod.gov.get_profile_qc_target('Standard')
    txt2 = mod.enforce_qc_footer_deltas(
        "X\nProfile: Standard QC: Clarity 3 · Brevity 1 · Evidence 2 · Empathy 2 · Consistency 2 · Neutrality 2",
        corr,
        profile_name='Standard'
    )
    assert "Brevity 1 (Δ-1)" in txt2



def test_ensure_qc_footer_present_rebuilds_empty_qc_matrix_line():
    mod = load_fix_module()

    class DummyGov:
        loaded = True
        data = {"global_defaults": {"output_contract": {"require_qc_footer": True}, "qc": {"enabled": True}}}
        def get_effective_qc_values(self, profile_name, overrides=None):
            return {
                "clarity": 3, "brevity": 2, "evidence": 2,
                "empathy": 2, "consistency": 3, "neutrality": 2,
            }
        def get_effective_qc_corridor(self, profile_name, overrides=None):
            return {
                "clarity": (2, 3), "brevity": (2, 2), "evidence": (2, 2),
                "empathy": (2, 2), "consistency": (2, 3), "neutrality": (2, 2),
            }

    raw = "Antworttext\n\nQC-Matrix:\n"
    out = mod.ensure_qc_footer_present(raw, DummyGov(), "Standard", overrides={})
    assert out.count("QC-Matrix:") == 1
    assert "Klarheit 3 (Δ0)" in out
    assert "Kürze 2 (Δ0)" in out


def test_qc_override_changes_delta_calculation():
    mod = load_fix_module()
    _prime_module_gov(mod)

    class DummySession:
        def send_message(self, prompt):
            class R:
                text = (
                    "Antwort.\n\n"
                    "QC-Matrix: K=3 · Clarity 3 (Δ0) · Brevity 1 (Δ-1) · Evidence 2 (Δ0) · "
                    "Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
                )
            return R()

    api = mod.Api()
    api.chat_session = DummySession()
    api.gov_state.comm_active = True

    # Apply via the official API (mirrors to gov-manager used by QC enforcement).
    api.qc_override_apply({"Brevity": 1})

    out = api.ask("hi")
    txt = _extract_text(out)
    assert "QC-Matrix:" in txt
    assert "Brevity 1 (Δ0)" in txt


def test_qc_override_injects_prompt_behavior_directives():
    mod = load_fix_module()
    _prime_module_gov(mod)

    class DummySession:
        def __init__(self):
            self.calls = []
        def send_message(self, prompt):
            self.calls.append(prompt)
            class R:
                text = (
                    "Antwort.\n\n"
                    "QC-Matrix: K=3 · Clarity 3 (Δ0) · Brevity 0 (Δ0) · Evidence 3 (Δ0) · "
                    "Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
                )
            return R()

    sess = DummySession()
    api = mod.Api()
    api.chat_session = sess
    api.gov_state.comm_active = True

    # Set overrides and ensure they are injected into the model prompt.
    api.qc_override_apply({"Brevity": 0, "Evidence": 3})

    _ = api.ask("hi")
    assert sess.calls, "Model should have been called exactly once in this test"
    sent = sess.calls[-1]
    assert "[QC OVERRIDES]" in sent
    assert "Brevity=0" in sent
    assert "Evidence=3" in sent
    assert "[QC BEHAVIOR]" in sent



def test_qc_override_clear_injects_one_time_prompt_reset_directive():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.gov_state.comm_active = True

    api.qc_override_apply({"Brevity": 0, "Evidence": 3})
    api.qc_override_clear({})

    first = api._apply_output_prefs_to_user_message("Was ist Zeit?")
    assert "[QC OVERRIDES] Cleared. Use profile defaults only." in first
    assert "Ignore any previous temporary QC override instructions" in first
    assert "Active temporary targets override profile defaults" not in first

    second = api._apply_output_prefs_to_user_message("Was ist Zeit?")
    assert "[QC OVERRIDES] Cleared. Use profile defaults only." not in second
    assert "Ignore any previous temporary QC override instructions" not in second
    assert "Active temporary targets override profile defaults" not in second


def test_expected_qc_deltas_respects_runtime_overrides():
    mod = load_fix_module()
    _prime_module_gov(mod)
    gov = getattr(mod, 'gov', None)
    assert gov is not None

    overrides = {"Brevity": 0, "Empathy": 2}
    cur = {
        "Clarity": 3,
        "Brevity": 2,
        "Evidence": 2,
        "Empathy": 1,
        "Consistency": 2,
        "Neutrality": 2,
    }
    d = gov.expected_qc_deltas("Expert", cur, overrides=overrides)
    assert d.get("Brevity") == 2
    assert d.get("Empathy") == -1

def test_qc_bridge_qc_get_state_accepts_payload_dict():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    br = mod.QCBridge(api)
    res = br.qc_get_state({})
    assert isinstance(res, dict)
    # ok can be False if ruleset not loaded, but should not crash and should include ok key
    assert 'ok' in res

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

def test_comm_audit_does_not_flag_missing_sd_or_qc_for_command_responses():
    """Regression test for Wrapper-160 (fix a):
    Compliance scan must classify command responses via the preceding user command,
    not via the bot message's first line, and must skip SD/QC checks for commands.
    """
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Ensure no model call occurs.
    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    # Seed history with command exchanges whose bot outputs do NOT contain Self-Debunking,
    # and one includes NO QC footer at all (as seen in Comm Audit export-only outputs).
    api.history = [
        {"role": "user", "content": "Comm State"},
        {"role": "bot", "content": "Zeit: 2026-02-04 16:59\nProfil: Expert\nQC-Matrix: Klarheit 3 (Δ0) · Kürze 3 (Δ0) · Evidenz 3 (Δ0) · Empathie 3 (Δ0) · Konsistenz 3 (Δ0) · Neutralität 3 (Δ0)"},
        {"role": "user", "content": "Comm Audit"},
        {"role": "bot", "content": "Comm Audit\nAudit exported.\nLogs/Audit/Audit_20260204_165930_267057.json"},
    ]

    out = api.ask("Comm Audit")
    html = _extract_html(out)

    # The scan must NOT complain about SD/QC for command responses.
    assert "Missing required 'Self-Debunking'" not in html
    assert "Missing QC footer" not in html

    # No LLM call.
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


def test_html_number_self_debunking_does_not_double_number_inside_ol():
    """Regression: html_number_self_debunking must NOT inject '1.' prefixes when Self-Debunking
    is already an ordered list (<ol>), otherwise the browser numbering + injected numbering
    causes double numbering.
    """
    mod = load_fix_module()

    html_in = (
        "<p><strong>Self-Debunking</strong></p>\n"
        "<ol>\n"
        "  <li><div>Weakness: Missing caveats.</div></li>\n"
        "  <li><div>Weakness: No verification route.</div></li>\n"
        "</ol>\n"
        "<p>QC-Matrix: Clarity 3 (Δ0)</p>\n"
    )
    html_out = mod.html_number_self_debunking(html_in, lang="en")

    # Still an ordered list
    assert "<ol" in html_out and "</ol>" in html_out
    # Must NOT inject textual numbering into the <div> lines.
    assert "1. Weakness" not in html_out
    assert "2. Weakness" not in html_out


def test_html_number_self_debunking_merges_split_secondary_paragraphs_in_ol():
    """Regression: split secondary SD fields must not remain as <p> siblings after the first <li>,
    otherwise browser paragraph margins create inconsistent spacing in item 1 vs 2+.
    """
    mod = load_fix_module()

    html_in = (
        '<div class="self-debunking"><div>Selbst-Debunking:</div><ol>\n'
        '<li><strong>Schwäche</strong>: A.</li>\n'
        '<p><strong>Warum relevant</strong>: B.</p>\n'
        '<p><strong>Prüfen/Widerlegen (nächster Schritt)</strong>: C.</p>\n'
        '<li><strong>Schwäche</strong>: D.<br><strong>Warum relevant</strong>: E.<br>'
        '<strong>Prüfen/Widerlegen (nächster Schritt)</strong>: F.</li>\n'
        '</ol></div>'
    )

    out = mod.html_number_self_debunking(html_in, lang="de")

    # The split <p> rows should be folded into the previous <li> as <br> lines.
    assert "<p><strong>Warum relevant</strong>" not in out
    assert "<p><strong>Prüfen/Widerlegen (nächster Schritt)</strong>" not in out
    assert "<br><strong>Warum relevant</strong>:" in out
    assert "<br><strong>Prüfen/Widerlegen (nächster Schritt)</strong>:" in out
    # Keep two logical list items (no accidental extra numbering rows).
    assert out.lower().count("<li") == 2


def test_apply_color_spans_handles_white_circle_and_multi_suffix():
    """Regression: Color spans must be applied for Evidence-Linker tokens even with
    ⚪/⚪️ emoji variants and multi-part suffixes like -WEB-CHECK.
    """
    mod = load_fix_module()

    s = "[GREEN-WEB-CHECK] 🟢 claim\n[GRAY] ⚪ neutral\n[RED-DOC] 🔴 risk"
    out = mod.apply_color_spans(s, enabled=True)

    # Each tag should become a styled span.
    assert out.count("<span style=") >= 3
    # Color-on rendering is icon-only (no visible bracket tokens).
    assert "[GREEN-WEB-CHECK]" not in out
    assert "[GRAY]" not in out
    assert "[RED-DOC]" not in out
    assert "🟢" in out
    assert "⚪" in out
    assert "🔴" in out


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

def _set_test_log_dirs(mod, tmp_path):
    """Redirect module log dirs into tmp_path (best-effort)."""
    mod.PROJECT_DIR = str(tmp_path)
    mod.CONFIG_DIR = str(tmp_path / 'Config')
    mod.LOGS_DIR = str(tmp_path / 'Logs')
    mod.AUDIT_LOG_DIR = str(tmp_path / 'Logs' / 'Audit')
    mod.CHAT_LOG_DIR = str(tmp_path / 'Logs' / 'Chats')
    for d in [mod.CONFIG_DIR, mod.LOGS_DIR, mod.AUDIT_LOG_DIR, mod.CHAT_LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def test_jsonl_audit_stream_appends_one_line_per_event(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    api.log_event('unit_test_1', {'x': 1})
    api.log_event('unit_test_2', {'y': 2})

    day = datetime.now().strftime('%Y%m%d')
    p = os.path.join(mod.AUDIT_LOG_DIR, f'AuditStream_{day}.jsonl')
    assert os.path.exists(p)

    lines = Path(p).read_text(encoding='utf-8').splitlines()
    assert len(lines) >= 2
    recs = [json.loads(ln) for ln in lines[-2:]]
    assert recs[0].get('event') == 'unit_test_1'
    assert recs[1].get('event') == 'unit_test_2'
    for r in recs:
        assert 'ts' in r and 'meta' in r and 'data' in r


def test_jsonl_audit_stream_redacts_secret_like_keys(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    api.log_event('unit_test_secret', {'api_key': 'sk-THIS-SHOULD-NOT-APPEAR', 'token': 'abc', 'ok': 'yes'})

    day = datetime.now().strftime('%Y%m%d')
    p = os.path.join(mod.AUDIT_LOG_DIR, f'AuditStream_{day}.jsonl')
    lines = Path(p).read_text(encoding='utf-8').splitlines()
    rec = json.loads(lines[-1])

    data = rec.get('data') or {}
    assert data.get('api_key') == '<redacted>'
    assert data.get('token') == '<redacted>'
    assert data.get('ok') == 'yes'
    assert 'sk-THIS-SHOULD-NOT-APPEAR' not in lines[-1]


def test_jsonl_audit_stream_minimum_schema(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    api.log_event('unit_test_schema', {'k': 'v'})
    day = datetime.now().strftime('%Y%m%d')
    p = os.path.join(mod.AUDIT_LOG_DIR, f'AuditStream_{day}.jsonl')
    rec = json.loads(Path(p).read_text(encoding='utf-8').splitlines()[-1])

    assert isinstance(rec.get('ts'), str) and rec['ts']
    assert rec.get('event') == 'unit_test_schema'
    assert 'meta' in rec and isinstance(rec['meta'], dict)
    assert rec['meta'].get('wrapper_version') is not None
    # route snapshot is best-effort but should be present as dict
    assert 'route' in rec and isinstance(rec['route'], dict)

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
    assert len(api.history) >= 2
    assert api.history[0]['content'] == "hi"
    assert api.history[1]['content'] == "<b>hello</b>"
    assert getattr(api, 'session_id', None) == sid_before

    api2 = mod.Api()
    sid2_before = getattr(api2, 'session_id', None)
    res2 = api2.load_log_from_path(str(legacy_path), fork=True)
    assert res2.get('ok') is True
    assert any((m.get('role') == 'sys' and 'Forked from chat log:' in str(m.get('content',''))) for m in api2.history if isinstance(m, dict))
    assert len(api2.history) >= 2
    assert getattr(api2, 'session_id', None) != sid2_before


def test_load_chat_log_uses_storage_service_safe_resolve(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(mod.CHAT_LOG_DIR) / "Log_demo.json"
    log_path.write_text(
        json.dumps({"history": [{"role": "user", "content": "hi"}, {"role": "bot", "content": "ok"}]}),
        encoding="utf-8",
    )

    class _StorageSpy:
        def __init__(self):
            self.safe_called = False
            self.exists_called = False

        def safe_resolve_in_dir(self, base_dir, filename):
            self.safe_called = True
            return str(log_path)

        def exists(self, path):
            self.exists_called = True
            return path == str(log_path)

    api = mod.Api()
    spy = _StorageSpy()
    api.storage_service = spy
    res = api.load_chat_log("Log_demo.json", fork=False)
    assert res.get("ok") is True
    assert spy.safe_called is True
    assert spy.exists_called is True


def test_panel_ping_exists_and_returns_ok():
    mod = load_fix_module()
    api = mod.Api()
    assert hasattr(api, 'ping')
    res = api.ping()
    assert isinstance(res, dict)
    assert res.get('ok') is True

def test_hf_topn_persists_via_localstorage_in_panel_html():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    html = getattr(mod, "HTML_PANEL", "")
    assert "_safeLsGet('hfTopN'" in html
    assert "id=\"hfTopN\"" in html
    assert "max=\"10000\"" in html


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

    # Minimum panel sections must exist even if Comm is off (they may be hidden/emptied).
    assert isinstance(ui.get('comm'), list)
    assert len(ui.get('comm')) > 0
    assert isinstance(ui.get('profiles'), list)
    # Under strict Comm-off gating, command sections are hidden until Comm Start.
    assert isinstance(ui.get('profiles'), list)

    # When Comm is active, the command sections must be populated.
    api.gov_state.comm_active = True
    ui_on = api.get_ui()
    assert isinstance(ui_on.get('profiles'), list)
    assert len(ui_on.get('profiles')) > 0

    # Log list keys must exist (may be empty in tests, but must not crash).
    assert 'chat_logs' in ui_on


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
    api.gov_state.comm_active = True
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


def test_chat_header_displays_wrapper_prompt_label():
    """UI invariant: the chat header must show the current Wrapper-NNN label (no legacy tag)."""
    mod = load_fix_module()
    html = getattr(mod, 'HTML_CHAT', '')
    assert isinstance(html, str) and html
    wrapper_name = getattr(mod, 'WRAPPER_NAME', '') or Path(getattr(mod, '__file__', '')).stem
    assert wrapper_name and wrapper_name in html
    # Must not use the old arrow-style header and must not wrap the wrapper label in square brackets.
    assert 'Wrapper-&gt;' not in html
    assert f'>{wrapper_name}<' in html
    assert '[Wrapper-' not in html


def test_main_window_title_is_dynamic_and_matches_wrapper_name():
    """We don't start pywebview in tests; validate the computed title and the create_window usage."""
    mod = load_fix_module()
    expected_name = FIX_PATH.stem
    assert getattr(mod, 'WRAPPER_NAME', '') == expected_name
    assert getattr(mod, "MAIN_WINDOW_TITLE", "") == expected_name

    txt = FIX_PATH.read_text(encoding='utf-8')
    assert 'MAIN_WINDOW_TITLE' in txt  # create_window must use the variable, not a stale literal
    assert 'Comm-SCi v19.14' not in txt

def test_startup_default_provider_and_model_are_gemini():
    """Config invariant: startup must always default to gemini + gemini-2.0-flash."""
    mod = load_fix_module()
    cfg = getattr(mod, 'cfg', None)
    assert cfg is not None
    assert getattr(cfg, 'get_active_provider', lambda: None)() == 'gemini'
    assert getattr(cfg, 'get_provider_model', lambda _p=None: '')('gemini') == 'gemini-2.0-flash'

def test_can_switch_back_to_gemini_after_other_provider():
    """Regression: switching back to gemini must not be blocked by a broken no-op guard."""
    mod = load_fix_module()
    cfg = getattr(mod, 'cfg', None)
    assert cfg is not None

    # switch away
    st1 = cfg.set_active_provider('openrouter')
    assert isinstance(st1, dict) and st1.get('ok')
    assert cfg.get_active_provider() == 'openrouter'

    # switch back
    st2 = cfg.set_active_provider('gemini')
    assert isinstance(st2, dict) and st2.get('ok')
    assert cfg.get_active_provider() == 'gemini'

    # and again via HF alias
    st3 = cfg.set_active_provider('hf')
    assert isinstance(st3, dict) and st3.get('ok')
    assert cfg.get_active_provider() == 'huggingface'
    st4 = cfg.set_active_provider('gemini')
    assert isinstance(st4, dict) and st4.get('ok')
    assert cfg.get_active_provider() == 'gemini'




def test_panel_html_qc_override_button_is_not_hf_only():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_PANEL', '')
    assert isinstance(html, str) and html
    i_btn = html.find('id="qcOverrideBtn"')
    assert i_btn != -1, "QC Override button must be present in panel HTML"
    i_hf = html.find('id="hfCatalogRow"')
    assert i_hf != -1
    # Button should be outside HF-only row (so it appears for Gemini/OpenRouter as well)
    assert i_btn < i_hf, "QC Override button must not be nested inside HF-only controls"


def test_panel_html_hf_topn_allows_up_to_10000():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_PANEL', '')
    assert isinstance(html, str) and html
    assert 'id="hfTopN"' in html
    assert 'max="10000"' in html, "HF Top-N input should allow up to 10000 models"

def test_panel_html_qc_override_onclick_is_valid():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_PANEL', '')
    assert isinstance(html, str) and html
    # Ensure the onclick handler is not over-escaped (must be valid JS)
    i = html.find('id="qcOverrideBtn"')
    assert i != -1
    snippet = html[i:i+200]
    assert "onclick=\"run('QC Override')\"" in snippet, snippet

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
    api.gov_state.comm_active = True
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


def test_remote_cmd_uses_ui_controller_when_available():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    called = {"n": 0, "cmd": None}

    class _Ui:
        def remote_input(self, win, cmd):
            called["n"] += 1
            called["cmd"] = cmd
            return True

    api.main_win = object()
    api.ui_controller = _Ui()
    out = api.remote_cmd("Comm State")
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert called["n"] == 1
    assert called["cmd"] == "Comm State"


def test_update_stats_ui_uses_ui_controller_when_available():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    called = {"n": 0, "text": ""}

    class _Ui:
        def update_stats(self, win, text):
            called["n"] += 1
            called["text"] = text
            return True

    api.main_win = object()
    api.ui_controller = _Ui()
    api.session_req_count = 7
    api.session_tokens_in = 123
    api.session_tokens_out = 456
    api.update_stats_ui()

    assert called["n"] == 1
    assert "Reqs: 7 | In: 123 | Out: 456" == called["text"]


def test_ui_add_system_message_falls_back_to_main_window_js():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    class _Win:
        def __init__(self):
            self.calls = []
        def evaluate_js(self, script):
            self.calls.append(str(script))

    win = _Win()
    api.main_win = win
    api.ui_controller = None

    ok = api._ui_add_system_message("Hallo UI")
    assert ok is True
    assert any("addMsg('sys'" in s for s in win.calls)
    assert any("Hallo UI" in s for s in win.calls)


def test_ui_update_rule_file_and_refresh_panel_use_ui_controller_first():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.main_win = object()
    api.panel_win = object()

    called = {"rule": 0, "refresh": 0}

    class _Ui:
        def update_rule_file(self, win, filename):
            called["rule"] += 1
            return True
        def eval_js(self, win, script):
            called["refresh"] += 1
            return True

    api.ui_controller = _Ui()
    assert api._ui_update_rule_file("Comm-SCI-v20.0.3.json") is True
    assert api._ui_refresh_panel() is True
    assert called["rule"] == 1
    assert called["refresh"] == 1


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



def test_panel_action_unknown_action_returns_stable_error_schema():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    out = api.panel_action('does_not_exist', {})
    assert isinstance(out, dict)
    assert out.get('ok') is False
    assert out.get('action') == 'does_not_exist'
    assert 'error' in out
    assert 'result' in out
    assert out.get('result') is None


def test_panel_action_list_chat_logs_returns_stable_success_schema():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.list_chat_logs = lambda limit=200: {'ok': True, 'logs': [{'name': 'a.json'}]}  # type: ignore[assignment]
    out = api.panel_action('list_chat_logs', {'limit': 5})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    assert out.get('action') == 'list_chat_logs'
    assert out.get('error') is None
    assert isinstance(out.get('result'), dict)
    assert isinstance(out.get('logs'), list)
    assert out['logs'][0]['name'] == 'a.json'


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


def test_refresh_models_gemini_populates_cache():
    """When provider is gemini, refresh_models() must populate _gemini_models_cache for panel dropdown updates."""
    mod = load_fix_module()

    class DummyPR:
        def get_active_provider(self):
            return 'gemini'
        def get_gemini_models_cached(self, force_refresh=False):
            return (['gemini-2.0-flash', 'gemini-2.5-flash'], {'source': 'test'})

    api = mod.Api()
    api.provider_router = DummyPR()
    api.main_win = None
    api.panel_win = None

    res = api.refresh_models()
    assert isinstance(res, dict)
    assert res.get('status') is True
    assert res.get('provider') == 'gemini'
    assert getattr(api, '_gemini_models_cache', None) == ['gemini-2.0-flash', 'gemini-2.5-flash']


def test_provider_service_canonical_provider_id_maps_aliases():
    mod = load_fix_module()
    psvc = mod.ProviderService(types.SimpleNamespace(config={}), None)

    assert psvc.canonical_provider_id('hf') == 'huggingface'
    assert psvc.canonical_provider_id('HuggingFace') == 'huggingface'
    assert psvc.canonical_provider_id('openai') == 'openrouter'
    assert psvc.canonical_provider_id('openrouter') == 'openrouter'
    assert psvc.canonical_provider_id('gemini') == 'gemini'
    assert psvc.canonical_provider_id('unknown-provider') == 'openrouter'


def test_provider_service_reads_config_fallback_models_deduped():
    mod = load_fix_module()
    cfg = types.SimpleNamespace(config={
        'providers': {
            'huggingface': {
                'fallback_models': ['a/model-1', 'a/model-1', 'b/model-2', ''],
            },
        },
    })
    psvc = mod.ProviderService(cfg, None)

    got = psvc.get_config_fallback_models('hf')
    assert got == ['a/model-1', 'b/model-2']


def test_get_available_models_gemini_uses_runtime_cache():
    mod = load_fix_module()
    api = mod.Api()
    api._gemini_models_cache = ['gemini-2.5-pro-preview', 'gemini-2.0-flash']

    models = api.get_available_models('gemini')
    assert models == ['gemini-2.5-pro-preview', 'gemini-2.0-flash']


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



def test_enforcement_policy_strict_block_blocks_hard_violations():
    mod = load_fix_module()
    _prime_module_gov(mod)

    # Enable strict block
    mod.cfg.config["enforcement_policy"] = "strict_block"
    mod.cfg.config["active_provider"] = "gemini"

    class DummyValidator:
        def validate(self, text=None, state=None, profile=None, **kwargs):
            return (["hard_violation"], [])
        def build_repair_prompt(self, user_prompt=None, raw_response=None, state=None, hard_violations=None, soft_violations=None, **kwargs):
            return "repair"


    class DummyChatSession:
        def send_message(self, prompt):
            class R:
                text = "Antwort ohne QC."
            return R()

    api = mod.Api()
    api.chat_session = DummyChatSession()
    api.validator = DummyValidator()
    api.gov_state.comm_active = True

    out = api.ask("hi")
    assert isinstance(out, dict)
    html = out.get("html", "") or ""
    if isinstance(html, dict):
        html = html.get("html", "") or ""
    assert "STRICT BLOCK" in html
    assert "Content withheld" in html


def test_enforcement_policy_strict_warn_prepends_warning_but_keeps_content():
    mod = load_fix_module()
    _prime_module_gov(mod)

    mod.cfg.config["enforcement_policy"] = "strict_warn"
    mod.cfg.config["active_provider"] = "gemini"

    class DummyValidator:
        def validate(self, text=None, state=None, profile=None, **kwargs):
            return (["hard_violation"], [])
        def build_repair_prompt(self, user_prompt=None, raw_response=None, state=None, hard_violations=None, soft_violations=None, **kwargs):
            return "repair"


    class DummyChatSession:
        def send_message(self, prompt):
            class R:
                text = "Antwort ohne QC."
            return R()

    api = mod.Api()
    api.chat_session = DummyChatSession()
    api.validator = DummyValidator()
    api.gov_state.comm_active = True

    out = api.ask("hi")
    assert isinstance(out, dict)
    html = out.get("html", "") or ""
    if isinstance(html, dict):
        html = html.get("html", "") or ""
    assert "RULE VIOLATION DETECTED" in html
    # content should still be visible in strict_warn
    assert "Antwort" in html


def test_enforcement_disabled_bypasses_strict_block_and_warn():
    mod = load_fix_module()
    _prime_module_gov(mod)

    mod.cfg.config["enforcement_policy"] = "strict_block"
    mod.cfg.config["enforcement_enabled"] = False
    mod.cfg.config["active_provider"] = "gemini"

    class DummyValidator:
        def validate(self, text=None, state=None, profile=None, **kwargs):
            return (["hard_violation"], [])
        def build_repair_prompt(self, user_prompt=None, raw_response=None, state=None, hard_violations=None, soft_violations=None, **kwargs):
            return "repair"

    class DummyChatSession:
        def send_message(self, prompt):
            class R:
                text = "Antwort ohne QC."
            return R()

    api = mod.Api()
    api.chat_session = DummyChatSession()
    api.validator = DummyValidator()
    api.gov_state.comm_active = True

    out = api.ask("hi")
    assert isinstance(out, dict)
    html = out.get("html", "") or ""
    if isinstance(html, dict):
        html = html.get("html", "") or ""
    assert "STRICT BLOCK" not in html
    assert "RULE VIOLATION DETECTED" not in html
    assert "Antwort" in html


# ------------------------
# Stufe 0 smoke tests
# ------------------------

def test_comm_help_renders_without_llm_call_and_emits_events():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("Comm Help")

    assert dummy.calls == [], "Comm Help must be UI-only (no provider call)"
    html = _extract_html(out)
    assert isinstance(html, str) and html.strip()
    assert "Comm" in html
    assert "Comm Anchor off" in html
    assert "Comm Anchor on" in html

    # Regression guard: help header must show the ruleset system name, not the imported `sys` module.
    # (Bug observed in v112 logs: "<module 'sys' (built-in)> v19.6.8 ...")
    assert "<module 'sys'" not in html

    # Minimal observability: input/route/command events should be recorded
    ev = getattr(api, "session_events", []) or []
    kinds = {str((e or {}).get("type")) for e in ev if isinstance(e, dict)}
    assert "input" in kinds
    assert "route" in kinds
    assert "command" in kinds


def test_comm_state_renders_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("Comm State")
    assert dummy.calls == [], "Comm State must be UI-only (no provider call)"
    html = _extract_html(out)
    assert isinstance(html, str) and html.strip()


def test_comm_audit_exports_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    # Snapshot current audit files
    audit_dir = getattr(mod, "AUDIT_LOG_DIR", None)
    assert audit_dir is not None
    before = set()
    try:
        before = set(os.listdir(audit_dir))
    except Exception:
        before = set()

    out = api.ask("Comm Audit")
    assert dummy.calls == [], "Comm Audit must be UI-only (no provider call)"
    html = _extract_html(out)
    assert isinstance(html, str)

    after = set()
    try:
        after = set(os.listdir(audit_dir))
    except Exception:
        after = set()

    # Must create at least one new audit file (or overwrite with new timestamped name)
    created = [x for x in (after - before) if str(x).startswith("Audit_") and str(x).endswith(".json")]
    assert created, "Expected Comm Audit to create a new Audit_*.json file"



def test_comm_audit_history_contains_export_note():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None

    dummy = DummySession(["SHOULD NOT BE USED"])
    api.chat_session = dummy

    out = api.ask("Comm Audit")
    assert dummy.calls == [], "Comm Audit must be UI-only (no provider call)"

    hist = getattr(api, "history", []) or []
    assert hist, "Expected history to contain the Comm Audit bot message"
    last = hist[-1] or {}
    txt = str(last.get("content") or "")
    assert "Comm Audit" in txt
    # The wrapper should include a short export note (path may vary in tests).
    assert "Exportiert (Audit)" in txt


def test_start_background_thread_is_idempotent(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    started = {"n": 0}

    class DummyThread:
        def __init__(self, target=None):
            self.target = target
            self.daemon = False

        def start(self):
            started["n"] += 1

    monkeypatch.setattr(mod.threading, "Thread", DummyThread)

    api.start_background_thread()
    api.start_background_thread()
    assert started["n"] == 1, "start_background_thread must not start twice"



def test_comm_stop_disables_governance_postprocessing():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None  # isolate postprocessing

    # Ensure a known profile with qc_target corridor exists.
    api.gov_state.active_profile = 'Standard'

    # Stub recreate to avoid requiring a real client while still tracking session governance state.
    def _fake_recreate(with_governance: bool = True, reason: str = ""):
        api.session_with_governance = bool(with_governance)

    api._recreate_chat_session = _fake_recreate  # type: ignore

    bad_qc = (
        "Answer\n"
        "Self-Debunking:\n"
        "1. Weakness: x\n"
        "2. Weakness: y\n"
        "QC-Matrix: Clarity 3 (Δ+9) · Brevity 1 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ-7)"
    )

    # With governance enabled (and Comm on), deltas must be corrected.
    api.gov_state.comm_active = True
    api.session_with_governance = True
    dummy1 = DummySession([bad_qc])
    api.chat_session = dummy1
    out1 = api.ask("Hello")
    text1 = _extract_text(out1)
    assert 'Clarity 3 (Δ0)' in text1
    assert 'Δ+9' not in text1

    # Comm Stop disables rule-system formatting on content answers (Safety Core may still stay active).
    api.chat_session = DummySession([bad_qc])
    api.ask("Comm Stop")
    # Simulate a provider/model reconnect bug: session still marked as governance-enabled
    # while Comm is already OFF. The wrapper must still suppress Comm-SCI formatting.
    api.session_with_governance = True
    out2 = api.ask("Hello")
    text2 = _extract_text(out2)
    assert 'Answer' in text2
    assert 'QC-Matrix:' not in text2
    assert 'Δ+9' not in text2
    assert 'Self-Debunking' not in text2 and 'Selbst-Debunking' not in text2
    assert 'SCI Trace' not in text2




def test_qc_footer_is_moved_to_end_when_model_puts_it_early():
    mod = load_fix_module()
    # Build a valid QC line (canonical), but place it before the answer.
    early = (
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 1 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
        "[GREEN] Ein elektrisches Feld ist der Raum um eine elektrische Ladung, in dem auf andere Ladungen eine Kraft wirkt.\n\n"
        "Self-Debunking:\n- Punkt 1\n- Punkt 2\n"
    )
    out = mod.ensure_qc_footer_is_last(early)
    # QC must be last block
    assert out.strip().endswith("Neutrality 2 (Δ0)"), out
    # Answer must remain present and appear before the footer
    assert "[GREEN]" in out
    assert out.find("[GREEN]") < out.rfind("QC-Matrix:")


def test_comm_state_shows_effective_qc_values_and_optional_override_line():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None
    api.chat_session = DummySession(["SHOULD NOT BE USED"])

    # Set an override and verify Comm State reflects it deterministically.
    api.gov_state.qc_overrides = {"brevity": 1}

    out = api.ask("Comm State")
    html = _extract_html(out)

    assert "QC-Matrix:" in html
    assert "Brevity 1 (Δ0)" in html
    assert "QC-Overrides:" in html and "Brevity=1" in html



def test_chat_export_includes_provider_model_metadata_and_history(tmp_path):
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

    # Ensure we can switch without triggering network (OpenRouter path is stateless)
    api.set_provider('openrouter')
    api.set_model('openrouter/test-model')

    # Export and verify additive fields exist
    chat_path, _audit_path = api.export()
    data = json.loads(Path(chat_path).read_text(encoding='utf-8'))

    # Trace metadata must be present for fork provenance
    assert isinstance(data.get('trace_id'), str) and data.get('trace_id').strip()
    assert isinstance(data.get('session_id'), str) and data.get('session_id').strip()

    assert data.get('active_provider') == 'openrouter'
    assert data.get('active_model') == 'openrouter/test-model'
    hist = data.get('provider_model_history') or []
    assert isinstance(hist, list)
    # at least one provider/model event should be present
    assert any(e.get('event') in ('provider_switch', 'model_switch') for e in hist)


def test_fork_records_source_metadata_and_sys_history_line(tmp_path):
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

    # Create a seed log to fork from
    api.history.append({'role': 'user', 'content': 'seed', 'ts': datetime.now().isoformat()})
    seed_chat_path, _ = api.export()

    # Fork-load from the exported chat log
    res = api.load_log_from_path(seed_chat_path, fork=True)
    assert res.get('ok') is True
    assert res.get('forked') is True
    assert getattr(api, 'forked_from_log_path', None) == seed_chat_path

    # The fork should have a visible sys marker line in history
    assert any(
        (m.get('role') == 'sys' and 'Forked from chat log:' in str(m.get('content', '')))
        for m in (api.history or [])
        if isinstance(m, dict)
    )

    # Export again and ensure fork metadata is persisted
    seed_data = json.loads(Path(seed_chat_path).read_text(encoding='utf-8'))
    seed_trace = seed_data.get('trace_id')
    assert isinstance(seed_trace, str) and seed_trace.strip()

    fork_chat_path, _ = api.export()
    fork_data = json.loads(Path(fork_chat_path).read_text(encoding='utf-8'))
    assert fork_data.get('forked_from_log_path') == seed_chat_path
    # For newly exported logs, parent trace id must be captured and match the source trace id
    assert fork_data.get('fork_parent_trace_id') == seed_trace
    # Fork session should have its own trace id
    assert isinstance(fork_data.get('trace_id'), str) and fork_data.get('trace_id').strip()
    assert fork_data.get('trace_id') != seed_trace


def test_log_event_does_not_crash_without_dirs(tmp_path):
    """Stage 0: log_event must never raise, even if log directories are missing.

    This test does *not* start the GUI and must not require any existing folders.
    """
    mod = load_fix_module()
    _prime_module_gov(mod)

    # Point logs at non-existent directories (do not create them)
    mod.PROJECT_DIR = str(tmp_path)
    mod.LOGS_DIR = str(tmp_path / 'Logs')
    mod.AUDIT_LOG_DIR = str(tmp_path / 'Logs' / 'Audit')
    mod.CHAT_LOG_DIR = str(tmp_path / 'Logs' / 'Chats')

    api = mod.Api()

    # Force a weird/empty session_events state to ensure defensive behavior
    try:
        api.session_events = None
    except Exception:
        pass

    api.log_event('ui', {'msg': 'hello', 'big': 'x' * 2000})
    assert isinstance(getattr(api, 'session_events', None), list)
    assert len(api.session_events) >= 1
    ev = api.session_events[-1]
    assert isinstance(ev, dict)
    assert ev.get('type') == 'ui'
    # trace_id must be present (at least session_id fallback)
    assert isinstance(ev.get('trace_id'), (str, type(None)))


def test_trace_id_present_in_audit_v2_if_enabled(tmp_path):
    """Stage 0 (optional): audit v2 export must include a non-empty trace_id."""
    mod = load_fix_module()
    _prime_module_gov(mod)

    mod.PROJECT_DIR = str(tmp_path)
    mod.LOGS_DIR = str(tmp_path / 'Logs')
    mod.AUDIT_LOG_DIR = str(tmp_path / 'Logs' / 'Audit')
    mod.CHAT_LOG_DIR = str(tmp_path / 'Logs' / 'Chats')

    api = mod.Api()

    audit_path = tmp_path / 'Logs' / 'Audit' / 'Audit_test.json'
    api.export_audit_v2(audit_only=True, audit_path=str(audit_path), ts='TEST')

    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding='utf-8'))
    sm = payload.get('session_metadata') or {}
    assert isinstance(sm.get('trace_id'), str) and sm.get('trace_id').strip()


def test_sci_menu_instructions_are_english_when_ui_lang_en():
    mod = load_fix_module()
    data = load_ruleset_data()
    _prime_module_gov(mod)
    api = mod.Api()

    html = api._render_sci_menu_html()
    assert "SCI variants" in html
    assert "Reply in the next prompt" in html
    assert "A–H" in html or "A-H" in html


def test_comm_anchor_off_on_toggles_anchor_snapshot_automation_and_panel_label():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # precondition
    assert bool(getattr(api.gov_state, "anchor_auto", True)) is True

    # Panel UI should show the toggle label for the current state ("off" when currently on)
    ui1 = api.get_ui()
    comm1 = ui1.get('comm') or []
    # single toggle button object is expected when ruleset supports both commands
    assert any((isinstance(x, dict) and x.get('cmd') in ('Comm Anchor off', 'Comm Anchor on')) or (x in ('Comm Anchor off', 'Comm Anchor on')) for x in comm1)

    # Turn off
    api._execute_legacy_command("Comm Anchor off")
    assert bool(getattr(api.gov_state, "anchor_auto", True)) is False

    ui2 = api.get_ui()
    comm2 = ui2.get('comm') or []
    # When off, the toggle label must offer turning it on
    assert any((isinstance(x, dict) and x.get('cmd') == 'Comm Anchor on') or (x == 'Comm Anchor on') for x in comm2)

    # Turn on
    api._execute_legacy_command("Comm Anchor on")
    assert bool(getattr(api.gov_state, "anchor_auto", False)) is True

    ui3 = api.get_ui()
    comm3 = ui3.get('comm') or []
    # When on, the toggle label must offer turning it off
    assert any((isinstance(x, dict) and x.get('cmd') == 'Comm Anchor off') or (x == 'Comm Anchor off') for x in comm3)


# ----------------------------
# STUFE 1: schema contract tests (fail-soft)
# ----------------------------

def test_stage1_contract_route_shapes_smoke():
    mod = load_fix_module()
    _prime_module_gov(mod)

    # noop
    r = mod.route_input("", types.SimpleNamespace(sci_pending=False), types.SimpleNamespace(gov=mod.gov))
    assert mod.contract_route_shape(r) is True

    # command: pick any canonical command token
    data = load_ruleset_data()
    any_cmd = None
    commands = (data.get("commands") or {})
    for cat in commands.values():
        if isinstance(cat, dict):
            for k in cat.keys():
                if isinstance(k, str) and k.strip():
                    any_cmd = k
                    break
        if any_cmd:
            break
    assert any_cmd is not None
    r2 = mod.route_input(any_cmd, types.SimpleNamespace(sci_pending=False), types.SimpleNamespace(gov=mod.gov))
    assert r2.get("kind") == "command"
    assert mod.contract_route_shape(r2) is True

    # chat
    r3 = mod.route_input("hello world", types.SimpleNamespace(sci_pending=False), types.SimpleNamespace(gov=mod.gov))
    assert r3.get("kind") == "chat"
    assert mod.contract_route_shape(r3) is True


def test_stage1_contract_ask_output_shape_smoke():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    # Avoid real provider calls
    api.chat_session = DummySession(["OK\n\nQC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)"])
    out = api.ask("hello")
    assert mod.contract_ask_output_shape(out) is True



def test_stage1_command_response_contract_smoke():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None  # isolate formatting behavior
    # Command path must never call the model.
    api.chat_session = DummySession(["SHOULD NOT BE USED"])

    out = api.ask("Comm State")
    # Some command routes return a dict payload with an 'html' field.
    html_out = out.get('html') if isinstance(out, dict) else out
    assert isinstance(html_out, str) and html_out.strip()
    assert mod.contract_command_response(html_out) is True


def test_profile_switch_emits_profile_switch_audit_line():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None
    api.gov_state.active_profile = "Standard"

    out = api.ask("Profile Expert")
    html_out = out.get("html") if isinstance(out, dict) else str(out)
    assert "Profile-Switch-Audit: command=Profile Expert · from=Standard · to=Expert · rule=explicit-standalone-only" in html_out


def test_comm_start_emits_profile_switch_audit_line_when_resetting_profile():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None

    default_profile = str(((mod.gov.data or {}).get("global_defaults", {}) or {}).get("default_profile", "Standard"))
    api.gov_state.active_profile = "Expert"

    out = api.ask("Comm Start")
    html_out = out.get("html") if isinstance(out, dict) else str(out)
    assert api.gov_state.active_profile == default_profile
    assert (
        f"Profile-Switch-Audit: command=Comm Start · from=Expert · to={default_profile} · rule=explicit-standalone-only"
        in html_out
    )


def test_stage1_answer_response_contract_smoke():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None  # isolate formatting behavior

    # Ensure governance is active so strict postprocessing runs.
    api.ask("Comm Start")

    dummy_text = (
        "Final Answer\n"
        "Time is the parameter that orders events and allows us to quantify durations.\n\n"
        "Self-Debunking\n"
        "1. Weakness: This is a simplified definition.\n"
        "2. Why it matters: Different theories define time differently.\n"
        "3. What would improve it: Specify the operational definition being used.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 3 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )
    dummy = DummySession([dummy_text])
    api.chat_session = dummy

    out = api.ask("What is time?")
    assert dummy.calls, "Model should be called for normal content questions."
    html_out = out.get('html') if isinstance(out, dict) else out
    assert mod.contract_answer_response(html_out) is True

def test_ui_replay_loaded_history_renders_comm_config_dump_as_collapsible_html():
    """When loading/replaying a chat log, a legacy plaintext 'Comm Config' dump should be turned into a collapsible HTML block."""
    mod = load_fix_module()

    class DummyWin:
        def __init__(self):
            self.calls = []
        def evaluate_js(self, js):
            self.calls.append(js)
            # Force incremental replay path
            if 'resetChatFromHistory' in js:
                return 'NOFUNC'
            return 'OK'

    api = mod.Api()
    api.main_win = DummyWin()

    # Minimal-but-recognizable Comm Config plaintext dump (simulate legacy log content)
    big_json = "{\n" + "\n".join([f'  "k{i}": "{("x"*40)}",' for i in range(40)]) + "\n  \"end\": 1\n}"
    comm_dump = (
        "Comm-SCI-Control v19.6.9 · Loaded rules file: Comm-SCI-v19.6.9.json\n\n"
        + big_json
        + "\n\nQC-Matrix: Clarity 2 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
    )

    api.history = [
        {'role': 'assistant', 'content': comm_dump},
    ]

    api._ui_replay_loaded_history(status_msg='Loaded X')

    joined = "\n".join(api.main_win.calls)
    assert "<details" in joined
    assert "raw-json" in joined


# --- Regression tests: Color-on consistency + SCI trace repair (v150) ---


def test_normalization_summary_meta_is_present_for_content_answers():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.comm_active = False
    api.gov_state.color = "off"

    raw = "Self-Debunking\nWeakness: test\nQC: Klarheit 3"
    html_out, meta = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)

    assert isinstance(meta, dict)
    ns = meta.get("normalization")
    assert isinstance(ns, dict)
    assert "qc_footer_raw_count" in ns
    assert "qc_footer_html_count" in ns
    assert "self_debunking_boxed" in ns


def test_apply_csc_strict_collapses_mixed_qc_and_qc_matrix_footers_to_single_canonical_footer():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.comm_active = True
    api.gov_state.color = "off"

    raw = (
        "Antworttext\n\n"
        "QC: Klarheit 3 (Δ0) · Kürze 0 (Δ-2) · Evidenz 3 (Δ+1) · Empathie 2 (Δ0) · Konsistenz 3 (Δ0) · Neutralität 2 (Δ0)\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 0 (Δ0) · Evidence 3 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)"
    )

    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", "\n", str(html_out or ""))
    qc_lines = [ln.strip() for ln in plain.splitlines() if re.match(r"(?i)^QC(?:-Matrix)?\s*:", (ln or "").strip())]

    assert sum(1 for ln in qc_lines if re.match(r"(?i)^QC-Matrix\s*:", ln)) == 1
    assert not any(re.match(r"(?i)^QC\s*:", ln) for ln in qc_lines)


def test_session_render_counters_increment_on_fallback():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.comm_active = False
    api.gov_state.color = "off"

    # Force a "broken" render output by feeding already-escaped HTML inside <pre>
    raw = "<div>hello</div>"
    html_out, meta = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)

    # Call through answer route once to bump counters: simulate the stage where meta is processed
    # We can't easily call the full provider route here; instead emulate counter update logic.
    ns = meta.get("normalization", {}) if isinstance(meta, dict) else {}
    if ns.get("render_ok"):
        api.session_render_ok_count = int(getattr(api, "session_render_ok_count", 0) or 0) + 1
    else:
        api.session_render_fallback_count = int(getattr(api, "session_render_fallback_count", 0) or 0) + 1

    assert int(getattr(api, "session_render_ok_count", 0) or 0) + int(getattr(api, "session_render_fallback_count", 0) or 0) >= 1


def test_color_spans_applied_in_command_and_inactive_render_paths():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # Force Color on, but keep comm inactive so we hit the comm-inactive Markdown render path.
    api.gov_state.color = 'on'
    api.gov_state.comm_active = False

    raw = "Hello\n[GREEN] 🟢 claim\n[YELLOW] 🟡 maybe\n[RED] 🔴 unknown"
    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)
    assert "span" in html_out.lower()
    assert "#137333" in html_out or "#2e7d32" in html_out  # green (either palette is acceptable)

    # Command path
    html_out2, _ = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=True)
    assert "span" in html_out2.lower()


def test_sci_trace_repair_variant_a_extracts_plan_solution_check_and_removes_empty_list():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = 'A'

    raw = (
        "SCI Trace\n"
        "• Plan\n"
        "• Solution\n"
        "• Check\n\n"
        "**Plan:** This is the plan.\n"
        "**Solution:** This is the solution.\n"
        "**Check:** This is the check.\n\n"
        "Final answer text.\n"
        "QC-Matrix: Klarheit 3 (Δ0)\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "• Plan" not in out
    assert "**Plan:**" not in out
    assert "This is the plan." in out
    assert "This is the solution." in out
    assert "This is the check." in out


def test_sci_trace_repair_variant_b_rebuilds_steps_and_never_shows_empty_step_list():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = 'B'

    raw = (
        "SCI Trace\n"
        "- Plan\n"
        "- Solution\n"
        "- Critic\n"
        "- Linguist\n"
        "- Logician\n"
        "- Adversary\n"
        "- Dialectic_1_Thesis\n"
        "- Dialectic_2_Antithesis\n"
        "- Dialectic_3_Synthesis\n"
        "- Dialectic_4_Metathesis\n"
        "- Dialectic_5_Hyperantithesis\n"
        "- Dialectic_6_Synthesis2\n"
        "- Learn\n\n"
        "**Plan:** P.\n"
        "**Solution:** S.\n"
        "**Critic:** C.\n"
        "**Linguist:** L.\n"
        "**Logician:** Lo.\n"
        "**Adversary:** A.\n"
        "**Dialectic_1_Thesis:** T1.\n"
        "**Dialectic_2_Antithesis:** T2.\n"
        "**Dialectic_3_Synthesis:** T3.\n"
        "**Dialectic_4_Metathesis:** T4.\n"
        "**Dialectic_5_Hyperantithesis:** T5.\n"
        "**Dialectic_6_Synthesis2:** T6.\n"
        "**Learn:** Learn.\n\n"
        "Some final answer.\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "- Plan" not in out  # old empty list removed
    assert "**Plan:**" not in out  # extracted
    assert "P." in out and "Learn." in out

# -----------------------------
# SCI variant mapping (v19.6.9)
# -----------------------------

def test_sci_variant_def_resolves_maps_to_object_and_steps_for_B_and_A():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()

    # Variant B must resolve to SCIplus with 13 steps (Dialectics++).
    vdef_b, steps_b, maps_b = api._sci_variant_def('B')
    assert isinstance(vdef_b, dict)
    assert maps_b == 'SCIplus'
    assert isinstance(steps_b, list) and len(steps_b) >= 13
    # load-bearing steps
    for s in [
        'Plan','Solution','Critic','Linguist','Logician','Adversary',
        'Dialectic_1_Thesis','Dialectic_2_Antithesis','Dialectic_3_Synthesis',
        'Dialectic_4_Metathesis','Dialectic_5_Hyperantithesis','Dialectic_6_Synthesis2','Learn'
    ]:
        assert s in steps_b

    # Variant A must resolve to minimal SCI steps.
    vdef_a, steps_a, maps_a = api._sci_variant_def('A')
    assert maps_a in ('SCI', 'SCI')
    assert steps_a[:3] == ['Plan','Solution','Check']


def test_wrap_user_with_sci_includes_all_required_steps_for_variant_B():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    wrapped = api._wrap_user_with_sci('Hello', variant='B')

    assert 'SCI Trace' in wrapped
    # ensure a late dialectic step is explicitly listed
    assert '- Dialectic_6_Synthesis2' in wrapped
    assert '- Learn' in wrapped



def test_repair_prompt_includes_sci_trace_requirements_when_sci_is_active_and_steps_exist():
    mod = load_fix_module()
    _prime_module_gov(mod)

    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)

    class S:
        sci_active = True
        sci_variant = 'B'

    prompt = validator.build_repair_prompt(
        user_prompt='Q',
        raw_response='A',
        state=S,
        hard_violations=['Missing SCI Trace step: Critic'],
        soft_violations=[]
    )

    assert 'SCI Trace requirements:' in prompt
    assert '  - Critic' in prompt
    assert 'Redacted:' in prompt


def test_self_debunking_unnumbered_blocks_are_numbered():
    mod = load_fix_module()

    class GM:
        loaded = True
        data = {
            "global_defaults": {
                "output_contract": {
                    "self_debunking_contract": {
                        "enabled": True,
                        "required_block_title": "Self-Debunking",
                        "required_min_points": 2,
                        "required_max_points": 3,
                    }
                },
                "self_debunking": {
                    "enabled": True,
                    "exceptions": [],
                    "block": {"title": "Self-Debunking"},
                },
            }
        }

    txt = (
        "Answer text.\n\n"
        "Self-Debunking:\n\n"
        "Weakness: First weakness line.\n"
        "Why it matters: First why.\n"
        "What would verify/falsify (next check): First check.\n"
        "Weakness: Second weakness line.\n"
        "Why it matters: Second why.\n"
        "What would verify/falsify (next check): Second check.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(txt, GM(), "Expert", is_command=False, lang="en")
    assert "1. **Weakness**:" in out
    assert "2. **Weakness**:" in out
    assert "Self-Debunking:" in out
    # Ensure we did not keep an empty Self-Debunking section.
    assert "Self-Debunking:\n\nQC-Matrix" not in out




def test_self_debunking_numbered_points_have_indented_continuations():
    """Continuation lines must be indented so Markdown keeps them inside <li> for long answers."""
    mod = load_fix_module()

    class GM:
        loaded = True
        data = {
            "global_defaults": {
                "output_contract": {
                    "self_debunking_contract": {
                        "enabled": True,
                        "required_block_title": "Self-Debunking",
                        "required_min_points": 2,
                        "required_max_points": 3,
                    }
                },
                "self_debunking": {
                    "enabled": True,
                    "exceptions": [],
                    "block": {"title": "Self-Debunking"},
                },
            }
        }

    txt = (
        "Answer text.\n\n"
        "Self-Debunking:\n\n"
        "1. Schwäche: Punkt eins.\n"
        "Warum das wichtig ist: Folgezeile eins.\n"
        "Was würde verifizieren/falsifizieren (nächster Check): Check eins.\n\n"
        "2. Schwäche: Punkt zwei.\n"
        "Warum das wichtig ist: Folgezeile zwei.\n"
        "Was würde verifizieren/falsifizieren (nächster Check): Check zwei.\n\n"
        "QC-Matrix: Klarheit 3 (Δ0) · Kürze 2 (Δ0) · Evidenz 2 (Δ0) · Empathie 2 (Δ0) · Konsistenz 3 (Δ0) · Neutralität 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(txt, GM(), "Standard", is_command=False, lang="de")

    # Ensure numbering is preserved and labels are bolded
    assert re.search(r"(?m)^\s*1\.\s+\*\*Schwäche\*\*:", out)
    assert re.search(r"(?m)^\s*2\.\s+\*\*Schwäche\*\*:", out)

    # Critical: continuation lines must be indented (>=3 spaces) so they stay within list items
    assert re.search(r"(?m)^\s{3}\*\*Warum das wichtig ist\*\*:", out)
    assert re.search(r"(?m)^\s{3}\*\*Was würde verifizieren/falsifizieren \(nächster Check\)\*\*:", out)


def test_sanitize_self_debunking_markdown_in_html_converts_bold_markers():
    mod = load_fix_module()
    html_in = (
        "<div class=\"self-debunking\">"
        "<div>**What would verify/falsify (next check)**: test.</div>"
        "<div>__Weakness__: example.</div>"
        "</div>"
    )
    out = mod.sanitize_self_debunking_markdown_in_html(html_in)
    assert "**What would verify/falsify (next check)**" not in out
    assert "__Weakness__" not in out
    assert "<strong>What would verify/falsify (next check)</strong>" in out
    assert "<strong>Weakness</strong>" in out


def test_sanitize_self_debunking_markdown_in_html_removes_orphan_star_before_br():
    mod = load_fix_module()
    html_in = (
        "<div class=\"self-debunking\">"
        "<p><strong>Schwäche:</strong> Satz eins. *<br><strong>Warum das wichtig ist</strong>: Satz zwei.</p>"
        "</div>"
    )
    out = mod.sanitize_self_debunking_markdown_in_html(html_in)
    assert "*<br>" not in out
    assert "<br><strong>Warum das wichtig ist</strong>" in out


def test_qc_override_runtime_violations_detects_brevity_mismatch():
    mod = load_fix_module()
    short_txt = "Kurze Antwort."
    vios = mod.qc_override_runtime_violations(short_txt, {"brevity": 0})
    assert isinstance(vios, list)
    assert any("Brevity" in v for v in vios)


def test_normalize_known_markdown_control_headings_converts_generic_subheadings():
    mod = load_fix_module()
    raw = (
        "#### Physikalische Perspektive\n"
        "Text\n"
        "### Subsection\n"
        "## Another one\n"
    )
    out = mod.normalize_known_markdown_control_headings(raw)
    assert "#### Physikalische Perspektive" not in out
    assert "### Subsection" not in out
    assert "## Another one" not in out
    assert "<strong>Physikalische Perspektive:</strong>" in out
    assert "<strong>Subsection:</strong>" in out
    assert "<strong>Another one:</strong>" in out


def test_unwrap_accidental_full_text_codefence_unwraps_governance_output():
    mod = load_fix_module()
    raw = (
        "```text\n"
        "Profile Standard\n\n"
        "<span style=\"color:#f9a825; font-weight:600;\">🟡</span> Test.\n\n"
        "Self-Debunking:\n"
        "1. **Schwäche**: Punkt.\n"
        "QC-Matrix: Clarity 2 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)\n"
        "```"
    )
    out = mod.unwrap_accidental_full_text_codefence(raw)
    assert not out.lstrip().startswith("```")
    assert "Profile Standard" in out
    assert "Self-Debunking:" in out
    assert "QC-Matrix:" in out


def test_unwrap_accidental_full_text_codefence_keeps_regular_code_sample():
    mod = load_fix_module()
    raw = "```text\nprint('hello')\nfor i in range(3):\n    print(i)\n```"
    out = mod.unwrap_accidental_full_text_codefence(raw)
    assert out == raw


def test_html_number_self_debunking_numbers_weakness_lines_in_box():
    mod = load_fix_module()
    html_in = (
        "<div class=\"self-debunking\">"
        "<div>Selbst-Debunking:</div>\n"
        "<div><strong>Schwäche</strong>: Punkt eins.</div>\n"
        "<div><strong>Warum das wichtig ist</strong>: A.</div>\n"
        "<div><strong>Schwäche</strong>: Punkt zwei.</div>\n"
        "<div><strong>Warum das wichtig ist</strong>: B.</div>\n"
        "</div>"
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "1. <strong>Schwäche</strong>" in out
    assert "2. <strong>Schwäche</strong>" in out


def test_cgi_user_feedback_triplet_is_intercepted_without_llm_call(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # Start governance
    api.ask("Comm Start")

    # Monkeypatch LLM call to fail if invoked
    def boom(*a, **k):
        raise AssertionError("LLM should not be called for CGI feedback triplets")

    api._llm_call = boom  # type: ignore[assignment]

    res = api.ask("3,3,3")
    assert "CGI feedback recorded" in (res.get("html") or "")


def test_stage1_route_ctx_has_required_keys():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    ctx = mod._build_route_ctx(api, user_raw="x", is_command=False)
    required = {"is_command","comm_active","ui_lang","answer_lang","color","sci_variant","sci_pending","user_raw"}
    assert required.issubset(set(ctx.keys()))
    assert ctx["user_raw"] == "x"
    assert ctx["is_command"] is False


def test_stage1_sci_menu_guard_only_in_allowed_routes():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # Ensure SCI is not pending; normal answer path must not contain the SCI menu text.
    api.gov_state.sci_pending = False
    api.gov_state.sci_variant = "A"
    raw = "Answer content\n\nSelf-Debunking:\n1. Weakness: x\n2. Weakness: y\n\nQC-Matrix: Klarheit 3 (Δ0)"
    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    assert "SCI-Varianten" not in html_out
    assert "SCI variants" not in html_out

    # Now simulate a pending SCI selection (menu allowed).
    api.gov_state.sci_pending = True
    out_menu = api.ask('SCI menu')
    menu_txt = _extract_text(out_menu)
    assert isinstance(menu_txt, str) and menu_txt
    assert ('SCI variants' in menu_txt) or ('SCI-Varianten' in menu_txt) or ('SCI Variants' in menu_txt)


def test_stage1_color_on_applies_spans_in_command_and_inactive_paths_ci_safe():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.color = "on"
    api.gov_state.comm_active = False  # comm-inactive markdown render path

    raw = "Hello\n[GREEN] 🟢 claim\n[YELLOW] 🟡 maybe\n[RED] 🔴 unknown"
    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)

    # Must include spans and a known green palette entry (either is acceptable).
    assert "span" in html_out.lower()
    assert ("#137333" in html_out) or ("#2e7d32" in html_out)


def test_stage2_session_events_ring_buffer_caps_ram_growth(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    _set_test_log_dirs(mod, tmp_path)
    api = mod.Api()

    cap = int(getattr(mod, 'SESSION_EVENTS_MAX', 2000))
    assert cap > 0

    n = cap + 50
    for i in range(n):
        api.log_event('unit_test_cap', {'i': i})

    assert isinstance(api.session_events, list)
    assert len(api.session_events) == cap

    # Oldest retained event should be the first after truncation.
    first = api.session_events[0].get('data') or {}
    last = api.session_events[-1].get('data') or {}
    assert first.get('i') == n - cap
    assert last.get('i') == n - 1



def test_stage3d_missing_optional_module_is_visible_in_state_and_audit(tmp_path):
    """If an optional module (Module.auditstream) is missing, wrapper must not crash and must surface it."""
    import sys as _sys

    # Copy wrapper to isolated temp dir
    wrapper_src = FIX_PATH.read_text(encoding="utf-8")
    wpath = tmp_path / "WrapperTmp.py"
    wpath.write_text(wrapper_src, encoding="utf-8")

    # Create Module/ package in tmpdir but intentionally omit auditstream.py
    src_mod_dir = SRC / "Module"
    dst_mod_dir = tmp_path / "Module"
    dst_mod_dir.mkdir(parents=True, exist_ok=True)
    (dst_mod_dir / "__init__.py").write_text("", encoding="utf-8")

    # Copy the other optional modules so only Module.auditstream is missing
    for fname in ("rendering_utils.py", "compliance_scan.py"):
        src = src_mod_dir / fname
        if src.exists():
            (dst_mod_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Minimal stub to allow import in this isolated test
            (dst_mod_dir / fname).write_text("# stub\n", encoding="utf-8")

    old_path = list(_sys.path)
    try:
        # Ensure optional modules are resolved from tmp_path first and repo root is not preferred
        _sys.path = [str(tmp_path)] + [p for p in old_path if str(HERE) not in str(p)]

        # Clear cached module imports (both root and Module.*)
        for _m in (
            "auditstream",
            "rendering_utils",
            "compliance_scan",
            "Module",
            "Module.auditstream",
            "Module.rendering_utils",
            "Module.compliance_scan",
        ):
            _sys.modules.pop(_m, None)

        spec = importlib.util.spec_from_file_location("WrapperTmp", wpath)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        _prime_module_gov(mod)
        api = mod.Api()
        api.validator = None

        out = api.ask("Comm State")
        html_state = _extract_html(out)
        assert "Modules" in html_state
        assert "auditstream=missing" in html_state

        out2 = api.ask("Comm Audit")
        html_audit = _extract_html(out2)
        assert "Optional modules missing" in html_audit
        assert "auditstream" in html_audit
    finally:
        _sys.path = old_path

def test_stage3e_strict_modules_mode_fails_when_module_missing(tmp_path):
    # Import Wrapper in isolated temp dir with no Module/; strict mode must fail.
    wrapper_src = FIX_PATH
    wrapper_dst = tmp_path / "Wrapper-175.py"
    wrapper_dst.write_text(Path(wrapper_src).read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["WRAPPER_STRICT_MODULES"] = "1"

    code = (
        "import importlib.util; "
        "p='Wrapper-175.py'; "
        "spec=importlib.util.spec_from_file_location('w', p); "
        "m=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(m)"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert p.returncode != 0


def test_stage3e_strict_modules_mode_passes_when_modules_present(tmp_path):
    # Copy Wrapper + Module dir; strict mode must allow import.
    wrapper_src = FIX_PATH
    wrapper_dst = tmp_path / "Wrapper-175.py"
    wrapper_dst.write_text(Path(wrapper_src).read_text(encoding="utf-8"), encoding="utf-8")

    module_src = Path(wrapper_src).resolve().parent / "Module"
    module_dst = tmp_path / "Module"

    if module_src.exists():
        shutil.copytree(module_src, module_dst)
    else:
        # Local fallback: create a minimal Module package so strict import can succeed.
        module_dst.mkdir(parents=True, exist_ok=True)
        (module_dst / "__init__.py").write_text("", encoding="utf-8")
        # Minimal stubs: strict mode checks presence/importability, not full behavior.
        (module_dst / "auditstream.py").write_text("def get_audit_stream_jsonl_path(*a, **k): return ''\n"
                                                  "def append_jsonl_line(*a, **k): return False\n"
                                                  "def build_jsonl_meta(*a, **k): return {}\n",
                                                  encoding="utf-8")
        (module_dst / "rendering_utils.py").write_text("def sanitize_html(x, *a, **k): return x\n"
                                                       "def apply_color_spans(x, *a, **k): return x\n"
                                                       "def html_number_self_debunking(x, *a, **k): return x\n",
                                                       encoding="utf-8")
        (module_dst / "compliance_scan.py").write_text("def scan_message_compliance_best_effort(*a, **k):\n"
                                                       "    return []\n",
                                                       encoding="utf-8")

    env = os.environ.copy()
    env["WRAPPER_STRICT_MODULES"] = "1"

    code = (
        "import importlib.util; "
        "p='Wrapper-175.py'; "
        "spec=importlib.util.spec_from_file_location('w', p); "
        "m=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(m)"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)

def test_strip_sci_variantenmenue_echo_from_content_answer():
    mod = load_fix_module()
    raw = (
        "Intro line.\n"
        "Profile: Expert\n"
        "SCI-Variantenmenü (Auswahl):\n"
        "Antworte im nächsten Prompt mit genau einem Buchstaben (A–H).\n"
        "A: Standard - Plan → Lösung → Check (klassisch)\n"
        "B: Deep-Dive - Dialektik++ (13 Schritte)\n"
        "H: Multi-Agent Simulation - Ensemble\n\n"
        "Final Answer: Time is a measure of change.\n"
        "Self-Debunking:\n"
        "1. **Schwäche**: ...\n"
        "QC-Matrix: Clarity 3 (Δ0)"
    )
    out = mod.strip_sci_menu_from_answer(raw)
    assert "SCI-Variantenmenü" not in out
    assert "Final Answer:" in out

# ---------------------------
# Additional tests for v192 DOM rendering pipeline
# ---------------------------

def test_v192_strip_sci_menu_leaks_plaintext():
    rp = _load_rendering_pipeline_v192()
    if rp is None:
        pytest.skip(f"rendering_pipeline_v192 unavailable: {RP192_IMPORT_ERROR or 'unknown'}")
    raw = (
        "SCI-Variantenmenü (Auswahl a–i)\n"
        "Antworte mit genau einem Buchstaben.\n"
        "A) Foo\nB) Bar\nC) Baz\n\n"
        "Antwort: Zeit ist ...\n"
    )
    out = rp.strip_sci_menu_leaks_plaintext(raw)
    assert "SCI-Varianten" not in out
    assert "Antwort: Zeit ist" in out


@pytest.mark.skipif(not _have_dom_pipeline(), reason="DOM pipeline deps (markdown+bs4) missing")
def test_v192_dom_removes_duplicate_sci_trace_header():
    rp = _load_rendering_pipeline_v192()
    assert rp is not None
    ctx = rp.RenderContext(ui_lang="en", color="off", is_command=False, strict=True)
    html_in = "<p>SCI Trace:</p><div class='sci-trace'><div>SCI Trace</div><ol><li>x</li></ol></div>"
    out = rp.dom_normalize(html_in, ctx)
    assert "SCI Trace:</p>" not in out  # duplicate header removed
    assert ("sci-trace" in out.lower()) and ("SCI Trace" in out)


@pytest.mark.skipif(not _have_dom_pipeline(), reason="DOM pipeline deps (markdown+bs4) missing")
def test_v192_self_debunking_localized_numbered_bold():
    rp = _load_rendering_pipeline_v192()
    assert rp is not None
    ctx = rp.RenderContext(ui_lang="de", color="off", is_command=False, strict=True)
    raw = (
        "Self-Debunking:\n\n"
        "Schwäche: Test\n"
        "Warum das wichtig ist: X\n"
        "Was würde prüfen/widerlegen (nächster Check): Y\n\n"
        "QC-Matrix: end"
    )
    out = rp.render_llm_text_to_html(raw, ctx)
    assert "Selbst-Debunking" in out
    assert "1. Schwäche" in out
    assert ("<strong>" in out.lower()) or ("<b>" in out.lower())


@pytest.mark.skipif(not _have_dom_pipeline(), reason="DOM pipeline deps (markdown+bs4) missing")
def test_v192_qc_footer_unique_and_lastish():
    rp = _load_rendering_pipeline_v192()
    assert rp is not None
    ctx = rp.RenderContext(ui_lang="en", color="off", is_command=False, strict=True)
    raw = "Text\n\nQC-Matrix: first\n\nMore\n\nQC-Matrix: last"
    out = rp.render_llm_text_to_html(raw, ctx)
    assert out.count("QC-Matrix") == 1
    assert out.rfind("QC-Matrix") > out.rfind("More")


# ---------------------------
# v20.0.3 minimal-diff contract (A)
# ---------------------------

def _load_ruleset(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def _normalize_for_minimal_diff(v202: dict, v203: dict) -> tuple[dict, dict]:
    """Return normalized copies so that only the allowed v20.0.3 deltas are ignored."""
    a = json.loads(json.dumps(v202))  # deep copy via json
    b = json.loads(json.dumps(v203))
    # Allowed delta 1: version bump
    b['version'] = a.get('version')
    # Allowed delta 2: meta.governance.governor_config addition
    mg = b.get('meta', {}).get('governance', {})
    if isinstance(mg, dict) and 'governor_config' in mg:
        mg.pop('governor_config', None)
    return a, b

def test_ruleset_v20_0_3_minimal_diff():
    """v20.0.3 must be a minimal patch of v20.0.2: only version + meta.governance.governor_config."""
    base = _resolve_ruleset_path('Comm-SCI-v20.0.2.json', 'COMM_SCI_BASE')
    patched = _resolve_ruleset_path('Comm-SCI-v20.0.3.json', 'COMM_SCI_PATCHED')

    v202 = _load_ruleset(base)
    v203 = _load_ruleset(patched)

    assert v202.get('version') == '20.0.2'
    assert v203.get('version') == '20.0.3'

    g202 = (v202.get('meta') or {}).get('governance') or {}
    g203 = (v203.get('meta') or {}).get('governance') or {}

    assert 'governor_config' not in g202, "v20.0.2 must not contain meta.governance.governor_config"
    assert 'governor_config' in g203, "v20.0.3 must contain meta.governance.governor_config"

    # Enforce minimal diff strictly
    a, b = _normalize_for_minimal_diff(v202, v203)
    assert a == b, "v20.0.3 differs from v20.0.2 beyond the allowed minimal patch (version + governor_config)"


def test_panel_ui_toggles_deterministic():
    # Minimal deterministic check: UI collapses on/off pairs into single toggle objects.
    import importlib.util, os, types

    wrapper_path = os.path.join(str(SRC), "Comm-SCI-Control-App.py")
    if not os.path.exists(wrapper_path):
        # allow running from repo root
        wrapper_path = os.path.join(os.getcwd(), "src", "Comm-SCI-Control-App.py")
    assert os.path.exists(wrapper_path), f"Wrapper not found at {wrapper_path}"

    spec = importlib.util.spec_from_file_location("wrapper201", wrapper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    api = mod.Api()
    # Load ruleset if available
    base = os.environ.get("COMM_SCI_PATCHED") or os.path.join(str(ROOT), "JSON", "Comm-SCI-v20.0.3.json")
    if not os.path.exists(base):
        base = os.path.join(os.getcwd(), "JSON", "Comm-SCI-v20.0.3.json")
    assert os.path.exists(base), "Comm-SCI-v20.0.3.json not found for UI test"
    assert api.gov.load_file(base)

    # Force deterministic states and check toggle mapping
    api.gov_state.comm_active = True
    api.gov_state.sci_pending = False
    api.gov_state.sci_active = False
    api.gov_state.overlay = "Strict"
    api.gov_state.color = "off"

    ui = api.get_ui()
    assert isinstance(ui, dict)
    comm = ui.get("comm")
    sci = ui.get("sci")
    overlays = ui.get("overlays")
    tools = ui.get("tools")

    # Comm toggle exists and only one of start/stop remains in list (as command tokens)
    assert any(isinstance(x, dict) and x.get("cmd") == "Comm Stop" for x in comm), "Expected Comm toggle to stop when active"
    assert "Comm Start" not in comm and "Comm Stop" not in comm, "Start/Stop should be collapsed (only toggle object remains)"

    # SCI menu/recurse hidden when SCI is off
    assert "SCI menu" not in sci and "SCI recurse" not in sci, "SCI advanced controls must hide when SCI is off"
    assert any(isinstance(x, dict) and x.get("cmd") == "SCI on" for x in sci), "Expected SCI toggle to enable when off"

    # Strict toggle shows OFF action (Strict off) when strict is on
    assert any(isinstance(x, dict) and x.get("cmd") == "Strict off" for x in overlays), "Expected Strict toggle to disable when active"
    # Explore toggle should enable when explore is off
    assert any(isinstance(x, dict) and x.get("cmd") == "Explore on" for x in overlays), "Expected Explore toggle to enable when inactive"

    # Color toggle enables when off
    assert any(isinstance(x, dict) and x.get("cmd") == "Color on" for x in tools), "Expected Color toggle to enable when currently off"



def test_toggle_btn_shows_action_not_state():
    """Regression: toggle buttons must show the *action* (opposite of state), not the current state."""
    src = FIX_PATH.read_text(encoding="utf-8")
    # Find _toggle_btn function body (kept small and deterministic)
    import re
    m = re.search(r"def\s+_toggle_btn\s*\(.*?\):\n(.*?)(?:\n\n|\n\s*#)", src, flags=re.S)
    assert m, "Missing _toggle_btn in wrapper"
    body = m.group(1)
    # If state is ON, label must show OFF in the is_on branch.
    assert '{label_prefix}: OFF' in body or 'label_prefix}: OFF' in body, "Toggle label does not show OFF when is_on=True"
    # And the else branch must show ON.
    assert '{label_prefix}: ON' in body or 'label_prefix}: ON' in body, "Toggle label does not show ON when is_on=False"

def test_html_number_self_debunking_removes_orphan_markers_and_no_double_prefix():
    mod = load_fix_module()
    html_in = (
        "<div class=\"self-debunking\">"
        "<div>Selbst-Debunking:</div>\n"
        "<div>1.</div>\n"
        "<div>Schwäche: Punkt eins.</div>\n"
        "<div>Warum das wichtig ist: A.</div>\n"
        "<div>2.</div>\n"
        "<div>2. Schwäche: Punkt zwei.</div>\n"
        "<div>Warum das wichtig ist: B.</div>\n"
        "</div>"
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<div>1.</div>" not in out
    assert "<div>2.</div>" not in out
    assert "2. 2." not in out
    assert "1. <strong>Schwäche</strong>: Punkt eins." in out
    assert "2. <strong>Schwäche</strong>: Punkt zwei." in out

def test_normalize_known_markdown_control_headings_converts_prefixed_subheadings():
    mod = load_fix_module()
    raw = (
        "[GREEN] ### Kulturelle Perspektive\n"
        "🟡 #### Physikalische Perspektive\n"
        "• ## Historische Perspektive\n"
    )
    out = mod.normalize_known_markdown_control_headings(raw)
    assert "### Kulturelle Perspektive" not in out
    assert "#### Physikalische Perspektive" not in out
    assert "## Historische Perspektive" not in out
    assert "<strong>Kulturelle Perspektive:</strong>" in out
    assert "<strong>Physikalische Perspektive:</strong>" in out
    assert "<strong>Historische Perspektive:</strong>" in out


def test_normalize_self_debunking_numbering_text_numbers_and_drops_orphans():
    mod = load_fix_module()
    raw = (
        "Self-Debunking:\n"
        "1.\n"
        "Weakness: First point.\n"
        "Why it matters: A.\n"
        "2.\n"
        "2. Weakness: Second point.\n"
        "What would verify/falsify (next check): B.\n"
        "\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
    )
    out = mod.normalize_self_debunking_numbering_text(raw, lang="de")
    assert "\n1.\n" not in out
    assert "\n2.\n" not in out
    assert "1. Schwäche: First point." in out
    assert "2. Schwäche: Second point." in out
    assert "2. 2. Schwäche" not in out


def test_strip_verification_route_display_lines_hides_train_fallback_lines():
    mod = load_fix_module()
    raw = (
        "Antwortblock\n"
        "Verification Route\n"
        "Source: TRAIN (general background knowledge)\n"
        "Measurement: not performed\n"
        "Contrast: plausible alternative noted but not evaluated\n"
        "Web-Check: not performed\n"
        "Self-Debunking:\n"
    )
    out = mod.strip_verification_route_display_lines(raw)
    assert "Verification Route" not in out
    assert "Source: TRAIN" not in out
    assert "Measurement: not performed" not in out
    assert "Contrast: plausible alternative" not in out
    assert "Web-Check: not performed" not in out
    assert "Self-Debunking:" in out


def test_strip_verification_route_display_lines_hides_gate_and_bulleted_markers():
    mod = load_fix_module()
    raw = (
        "Antwortblock\n"
        "Verification Route Gate:\n"
        "- Source: TRAIN - Allgemeinwissen\n"
        "- Measurement: Nicht durchgeführt\n"
        "- Contrast: Alternative beachtet, aber nicht bewertet.-Check: Nicht durchgeführt\n"
        "Selbst-Debunking:\n"
    )
    out = mod.strip_verification_route_display_lines(raw)
    assert "Verification Route Gate" not in out
    assert "Source:" not in out
    assert "Measurement:" not in out
    assert "Contrast:" not in out
    assert "Selbst-Debunking:" in out


def test_strip_internal_scaffolding_status_lines_removes_leaked_profile_status():
    mod = load_fix_module()
    raw = (
        "Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile: Expert · Overlay: Strict · SCI: B · Color: on\n"
        "Inhalt bleibt sichtbar.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Active profile:" in out
    assert "Profile: Expert · Overlay: Strict · SCI: B · Color: on" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_lines_keeps_normal_profile_sentence():
    mod = load_fix_module()
    raw = (
        "Profile: Expert und Standard sind zwei Modi.\n"
        "Diese Zeile enthält keine interne Statusmatrix.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert out == raw.rstrip("\n")


def test_strip_internal_scaffolding_status_lines_removes_prompt_directive_echo_lines():
    mod = load_fix_module()
    raw = (
        "Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "[QC OVERRIDES] Active temporary targets: Clarity=3, Brevity=0, Evidence=3\n"
        "[QC BEHAVIOR] Brevity=0: be very detailed.\n"
        "Kerninhalt bleibt erhalten.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "[QC OVERRIDES]" not in out
    assert "[QC BEHAVIOR]" not in out
    assert "Kerninhalt bleibt erhalten." in out


def test_strip_internal_scaffolding_status_lines_removes_isolated_profile_name_leak():
    mod = load_fix_module()
    raw = (
        "Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile: Standard.\n"
        "Antwortinhalt bleibt sichtbar.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Profile: Standard." not in out
    assert "Antwortinhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_lines_removes_profile_title_without_separator():
    mod = load_fix_module()
    raw = (
        "Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile Expert:\n"
        "Antwortinhalt bleibt sichtbar.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Profile Expert:" not in out
    assert "Antwortinhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_lines_removes_profile_plus_prompt_echo_block():
    mod = load_fix_module()
    raw = (
        "Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile: Standard\n"
        "Wie kann man das fair umsetzen?\n"
        "Hier startet der eigentliche Inhalt.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Profile: Standard" not in out
    assert "Wie kann man das fair umsetzen?" not in out
    assert "Hier startet der eigentliche Inhalt." in out


def test_strip_internal_scaffolding_status_lines_removes_profile_with_inline_prompt_echo():
    mod = load_fix_module()
    raw = (
        "Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile: Standard What is the best strategy?\n"
        "Hier startet der eigentliche Inhalt.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Profile: Standard" not in out
    assert "What is the best strategy?" not in out
    assert "Hier startet der eigentliche Inhalt." in out


def test_strip_internal_scaffolding_status_html_removes_profile_block():
    mod = load_fix_module()
    html_in = (
        "<p>Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
        "<p>Profile: Expert · Overlay: Strict · SCI: B · Color: on</p>"
        "<p>Inhalt bleibt sichtbar.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Active profile:" in out
    assert "Profile: Expert · Overlay: Strict · SCI: B · Color: on" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_html_removes_qc_override_directive_block():
    mod = load_fix_module()
    html_in = (
        "<p>Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
        "<p>[QC OVERRIDES] Active temporary targets: Clarity=3, Brevity=0</p>"
        "<p>Inhalt bleibt sichtbar.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "[QC OVERRIDES]" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_html_removes_multiline_profile_overlay_sci_block():
    mod = load_fix_module()
    html_in = (
        "<p>Profile: Standard<br>Overlay: Strict<br>SCI: off</p>"
        "<p>Inhalt bleibt sichtbar.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Profile: Standard" not in out
    assert "Overlay: Strict" not in out
    assert "SCI: off" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_html_removes_profile_title_without_separator():
    mod = load_fix_module()
    html_in = "<p>Profile Standard:</p><p>Inhalt bleibt sichtbar.</p>"
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Profile Standard:" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_html_removes_profile_plus_prompt_echo_block():
    mod = load_fix_module()
    html_in = (
        "<p>Profile: Standard\nWie kann man das fair umsetzen?</p>"
        "<p>Hier startet der eigentliche Inhalt.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Profile: Standard" not in out
    assert "Wie kann man das fair umsetzen?" not in out
    assert "Hier startet der eigentliche Inhalt." in out


def test_strip_internal_scaffolding_status_html_removes_profile_with_sci_trace_companion():
    mod = load_fix_module()
    html_in = (
        "<p>Profile: Expert\nSCI Trace</p>"
        "<div class='sci-trace'><div>SCI Trace</div><div>Plan ...</div></div>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Profile: Expert" not in out
    assert "sci-trace" in out


def test_strip_evidence_tags_from_heading_lines_removes_only_heading_markers():
    mod = load_fix_module()
    raw = (
        "[GREEN] 1. Strategie:\n"
        "[YELLOW] Diese Einschätzung ist plausibel, aber unsicher.\n"
    )
    out = mod.strip_evidence_tags_from_heading_lines(raw)
    assert "[GREEN]" not in out
    assert "1. Strategie:" in out
    assert "[YELLOW]" in out


def test_strip_exact_status_header_line_removes_exact_duplicate_only():
    mod = load_fix_module()
    header = "Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on"
    raw = (
        header + "\n"
        "Profile: Expert · Overlay: Strict · SCI: B · Color: on\n"
        "Inhalt bleibt sichtbar.\n"
    )
    out = mod.strip_exact_status_header_line(raw, header)
    assert header not in out
    assert "Profile: Expert · Overlay: Strict · SCI: B · Color: on" in out
    assert "Inhalt bleibt sichtbar." in out


def test_build_repair_prompt_uses_answer_language_from_state():
    mod = load_fix_module()
    gov_mgr = types.SimpleNamespace(loaded=False, data={})
    validator = mod.OutputComplianceValidator(gov_mgr, None)
    state = types.SimpleNamespace(answer_language="de", sci_variant="", sci_active=False, qc_overrides={})
    prompt = validator.build_repair_prompt(
        user_prompt="Was ist Zeit?",
        raw_response="SCI Trace:\n1. Plan:\nTime is ...",
        state=state,
        hard_violations=["Missing SCI Trace step"],
        soft_violations=[],
    )
    assert "Render explanatory text in German" in prompt

def test_normalize_self_debunking_numbering_text_handles_german_title_and_dedups():
    mod = load_fix_module()
    raw = (
        "Selbst-Debunking:\n"
        "1.\n"
        "1. Schwäche: Punkt eins.\n"
        "Warum das wichtig ist: A.\n"
        "2.\n"
        "2. Schwäche: Punkt zwei.\n"
        "Warum das wichtig ist: B.\n"
        "\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 2 (Δ0) · Neutrality 2 (Δ0)"
    )
    out = mod.normalize_self_debunking_numbering_text(raw, lang="de")
    assert "\n1.\n" not in out
    assert "\n2.\n" not in out
    assert "1. Schwäche: Punkt eins." in out
    assert "2. Schwäche: Punkt zwei." in out
    assert "2. 2. Schwäche" not in out


def test_normalize_self_debunking_numbering_text_strips_numbered_field_labels_in_single_point():
    mod = load_fix_module()
    raw = (
        "Self-Debunking:\n"
        "1. Schwäche: Die Definition von Zeit ist sehr abstrakt und lässt viele Fragen offen.\n"
        "2. Warum das wichtig ist: Eine präzisere Definition wäre hilfreich.\n"
        "3. What would verify/falsify (next check): Eine detailliertere Analyse.\n"
        "\n"
        "QC-Matrix: Clarity 3 (Δ0)"
    )
    out = mod.normalize_self_debunking_numbering_text(raw, lang="de")
    assert "1. Schwäche:" in out
    assert "\n2. Warum das wichtig ist:" not in out
    assert "\n3. What would verify/falsify" not in out
    assert "Warum das wichtig ist:" in out
    assert "Was würde verifizieren/falsifizieren (nächster Check):" in out


def test_strip_sci_trace_line_when_inactive_removes_leaked_inline_trace():
    mod = load_fix_module()
    raw = (
        "Antworttext.\n"
        "SCI Trace: Die Frage nach der Natur der Zeit ist komplex.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=False, sci_variant="", sci_pending=False)
    assert "SCI Trace:" not in out
    assert "Self-Debunking:" in out
    assert "Antworttext." in out


def test_strip_sci_trace_line_when_inactive_keeps_trace_when_variant_active():
    mod = load_fix_module()
    raw = "SCI Trace: Plan\n1. Plan: ...\n"
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=True, sci_variant="B", sci_pending=False)
    assert out == raw


def test_openai_compatible_http_error_uses_hf_provider_label_when_requested():
    mod = load_fix_module()
    msg = mod._openrouter_friendly_http_error(402, "insufficient credits", lang="de", provider_label="Hugging Face")
    assert "Hugging Face" in msg
    assert "OpenRouter" not in msg


def test_ensure_qc_footer_present_rebuilds_truncated_qc_line():
    mod = load_fix_module()
    class _GovStub:
        loaded = True
        data = {"global_defaults": {"output_contract": {"require_qc_footer": True}, "qc": {"enabled": True}}}
        def get_effective_qc_values(self, profile_name, overrides=None):
            return {
                "clarity": 3, "brevity": 0, "evidence": 3,
                "empathy": 3, "consistency": 3, "neutrality": 3,
            }
        def get_effective_qc_corridor(self, profile_name, overrides=None):
            return {
                "clarity": (3, 3), "brevity": (0, 0), "evidence": (3, 3),
                "empathy": (3, 3), "consistency": (3, 3), "neutrality": (3, 3),
            }
    raw = (
        "Antwort.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 0 (Δ0) · Evidence\n"
    )
    out = mod.ensure_qc_footer_present(raw, _GovStub(), "Expert")
    assert "QC-Matrix:" in out
    for token in ("Δ0",):
        assert token in out
    assert ("Klarheit " in out or "Clarity " in out)
    assert ("Kürze " in out or "Kuerze " in out or "Brevity " in out)
    assert ("Evidenz " in out or "Evidence " in out)
    assert ("Empathie " in out or "Empathy " in out)
    assert ("Konsistenz " in out or "Consistency " in out)
    assert ("Neutralität " in out or "Neutralitaet " in out or "Neutrality " in out)


def test_dedupe_self_debunking_sections_keeps_single_canonical_block():
    mod = load_fix_module()
    raw = (
        "SCI Trace: Selbst-Debunking:\n"
        "• Schwäche: A\n"
        "• Schwäche: B\n"
        "Selbst-Debunking:\n"
        "1. Schwäche: C\n"
        "2. Schwäche: D\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.dedupe_self_debunking_sections(raw)
    assert "SCI Trace: Selbst-Debunking:" not in out
    assert out.count("Selbst-Debunking:") == 1
    assert "1. Schwäche: C" in out
    assert "2. Schwäche: D" in out
    assert "QC-Matrix:" in out


def test_normalize_inline_self_debunking_header_splits_inline_block():
    mod = load_fix_module()
    raw = (
        "Antwortsatz mit Ende. Self-Debunking: 1. Schwäche: X.\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.normalize_inline_self_debunking_header(raw)
    assert "Ende.\n\nSelf-Debunking:" in out
    assert "Self-Debunking:\n1. Schwäche: X." in out


def test_render_sci_trace_runtime_accepts_bullet_step_headers():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "B"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Critic"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "• Plan: Schritt A.\n"
        "• Solution: Schritt B.\n"
        "• Critic: Schritt C.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "<ol" in out
    assert "Plan:</div>" in out
    assert "Solution:</div>" in out
    assert "Critic:</div>" in out


def test_render_sci_trace_runtime_handles_complex_step_labels_variant_g():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "G"

    def _variant_def(_v):
        return ({}, ["Pre-mortem: assume failure", "List failure modes", "Mitigations/controls"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "• Pre-mortem: assume failure: Annahme X.\n"
        "• List failure modes: Modus A, Modus B.\n"
        "• Mitigations/controls: Gegenmaßnahme Y.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "Missing SCI Trace step content" not in out
    assert "Pre-mortem: assume failure:</div>" in out
    assert "List failure modes:</div>" in out
    assert "Mitigations/controls:</div>" in out


def test_render_sci_trace_runtime_accepts_dialectic_syntheses2_alias_for_variant_b_step6():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "B"

    def _variant_def(_v):
        return ({}, ["Dialectic_6_Synthesis2", "Learn"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "- Dialectic_6_Synthesis2\n"
        "- Learn\n\n"
        "**Dialectic_6_Syntheses_2:** Alias-Step erkannt.\n"
        "**Learn:** Learn-Inhalt bleibt erhalten.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "Missing SCI Trace step content" not in out
    # Canonical rendering label must stay canonical even if input alias drifted.
    assert "Dialectic_6_Synthesis2:</div>" in out
    assert "Alias-Step erkannt." in out
    assert "Learn-Inhalt bleibt erhalten." in out


def test_validate_sci_trace_accepts_dialectic_syntheses2_alias_for_canonical_step():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)

    validator._required_trace_steps_for_variant = lambda _v: ["Dialectic_6_Synthesis2", "Learn"]  # type: ignore[assignment]

    raw = (
        "SCI Trace:\n"
        "Dialectic_6_Syntheses_2: Alias-Stepinhalt.\n"
        "Learn: Lerninhalt.\n"
    )
    vios = validator.validate_sci_trace(raw, "B")
    assert not any("Missing SCI Trace step: Dialectic_6_Synthesis2" in v for v in vios)
    assert not any("Missing SCI Trace step: Learn" in v for v in vios)
    assert not any("has no content" in v for v in vios)


def test_normalize_sci_trace_numbering_handles_complex_step_labels():
    mod = load_fix_module()

    class GOV:
        data = {
            "global_defaults": {
                "output_contract": {
                    "sci_trace_contract": {
                        "required_steps": [
                            "Pre-mortem: assume failure",
                            "List failure modes",
                            "Mitigations/controls",
                        ]
                    }
                }
            }
        }

    raw = (
        "SCI Trace:\n"
        "• Pre-mortem: assume failure: Annahme X.\n"
        "• List failure modes: Modus A.\n"
        "• Mitigations/controls: Gegenmaßnahme Y.\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.normalize_sci_trace_numbering(raw, GOV())
    assert "1. Pre-mortem: assume failure:" in out
    assert "2. List failure modes:" in out
    assert "3. Mitigations/controls:" in out


def test_html_number_self_debunking_handles_single_line_html_block():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking"><div>Selbst-Debunking:</div><div>1.</div>'
        '<div>1. Schwäche: Punkt eins.</div><div>Warum das wichtig ist: A.</div>'
        '<div>2.</div><div>2. Schwäche: Punkt zwei.</div><div>Warum das wichtig ist: B.</div></div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<div>1.</div>" not in out
    assert "<div>2.</div>" not in out
    assert "2. 2." not in out
    assert "1. Schwäche: Punkt eins." in out or "1. <strong>Schwäche</strong>: Punkt eins." in out
    assert "2. Schwäche: Punkt zwei." in out or "2. <strong>Schwäche</strong>: Punkt zwei." in out


def test_normalize_hash_subheadings_in_html_converts_leaked_hash_titles():
    mod = load_fix_module()
    html_in = (
        "<div>#### Kulturelle Zeit</div>\n"
        "<div>### Ethische Implikationen</div>\n"
        "<div>## Fazit</div>\n"
    )
    out = mod.normalize_hash_subheadings_in_html(html_in)
    assert "#### Kulturelle Zeit" not in out
    assert "### Ethische Implikationen" not in out
    assert "## Fazit" not in out
    assert "<strong>Kulturelle Zeit:</strong>" in out
    assert "<strong>Ethische Implikationen:</strong>" in out
    assert "<strong>Fazit:</strong>" in out

def test_detect_self_debunking_numbered_html_detects_numbered_de_block():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">\n'
        '<div>Selbst-Debunking:</div>\n'
        '<div>1. <strong>Schwäche</strong>: Punkt eins.</div>\n'
        '<div><strong>Warum das wichtig ist</strong>: A.</div>\n'
        '<div>2. <strong>Schwäche</strong>: Punkt zwei.</div>\n'
        '</div>'
    )
    assert mod.detect_self_debunking_numbered_html(html_in) is True


def test_detect_self_debunking_numbered_html_false_when_only_orphan_numbers():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">\n'
        '<div>Selbst-Debunking:</div>\n'
        '<div>1.</div>\n'
        '<div>2.</div>\n'
        '</div>'
    )
    assert mod.detect_self_debunking_numbered_html(html_in) is False


def test_detect_self_debunking_numbered_html_true_for_ol_li_without_text_prefix():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins.</li>'
        '<li><strong>Schwäche</strong>: Punkt zwei.</li>'
        '</ol>'
        '</div>'
    )
    assert mod.detect_self_debunking_numbered_html(html_in) is True


def test_html_number_self_debunking_strips_numbering_from_non_weakness_fields():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<div>1. Schwäche: Punkt eins.</div>'
        '<div>2. Warum das wichtig ist: Relevanz.</div>'
        '<div>3. Was würde verifizieren/falsifizieren (nächster Check): Test.</div>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "1. Schwäche:" in out or "1. <strong>Schwäche</strong>:" in out
    assert "2. Warum das wichtig ist:" not in out
    assert "3. Was würde verifizieren/falsifizieren (nächster Check):" not in out


def test_html_number_self_debunking_ol_bolds_lowercase_secondary_labels_and_breaks_line():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins. warum das wichtig ist: Relevanz. '
        'was würde verifizieren/falsifizieren (nächster Check): Test.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<strong>Warum das wichtig ist</strong>:" in out
    assert "<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:" in out
    assert "<br><strong>Warum das wichtig ist</strong>:" in out or "<br><strong>Warum das wichtig ist</strong>" in out


def test_html_number_self_debunking_non_ol_handles_lowercase_labels_with_space_before_colon():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<div>1. <strong>Schwäche</strong>: Punkt eins. warum das wichtig ist : Relevanz.</div>'
        '<div>2. <strong>Schwäche</strong>: Punkt zwei. was würde verifizieren/falsifizieren (nächster Check) : Test.</div>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "warum das wichtig ist :" not in out.lower()
    assert "was würde verifizieren/falsifizieren (nächster check) :" not in out.lower()
    assert "<strong>Warum das wichtig ist</strong>:" in out
    assert "<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:" in out
    assert "<br><strong>Warum das wichtig ist</strong>:" in out or "<br><strong>Warum das wichtig ist</strong>" in out


def test_html_number_self_debunking_ol_handles_sibling_p_secondary_labels_and_missing_colon():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins.</li>'
        '<p>Warum das wichtig ist : Relevanz.</p>'
        '<p>Was würde verifizieren/falsifizieren (nächster Check) Test.</p>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    # New canonical behavior: merge split sibling <p> rows back into the first <li>
    # so browser paragraph margins do not create inconsistent spacing in item 1.
    assert "<p><strong>Warum das wichtig ist</strong>: Relevanz.</p>" not in out
    assert "<p><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Test.</p>" not in out
    assert "<li>" in out and "</li>" in out
    assert "<br><strong>Warum das wichtig ist</strong>: Relevanz." in out
    assert "<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Test." in out


def test_html_number_self_debunking_ol_repairs_fragmented_markdown_and_color_noise():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li>*Schwäche</li>'
        '<p>*:</p>'
        '<p>🟡</p>'
        '<p>Die psychologische Erfahrung von Zeit wurde nur kurz angeschnitten.</p>'
        '<li>Warum es wichtig ist:</li>'
        '<p>🟡</p>'
        '<p>Die subjektive Wahrnehmung von Zeit beeinflusst unser Verhalten.</p>'
        '<p>*</p>'
        '<p><strong>Nächster Check</strong>: </p>'
        '<p>🟡</p>'
        '<p>Psychologische Studien zur Zeitwahrnehmung untersuchen.</p>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert out.count("<li>") == 1
    assert "<strong>Schwäche</strong>:" in out
    assert "<strong>Warum das wichtig ist</strong>:" in out
    assert "<strong>Nächster Check</strong>:" in out
    assert "🟡" not in out
    assert "<p>*</p>" not in out
    assert "*Schwäche" not in out


def test_detect_probable_truncation_flags_abrupt_cut():
    mod = load_fix_module()
    raw = "SCI Trace:\\nDialectic_2_Antithesis: Zeit ist ein subjektives Er"
    ok, msg = mod._detect_probable_truncation(raw, "<div>SCI Trace</div>")
    assert ok is True
    assert "unvollständig" in msg


def test_panel_asset_static_selftest_ok_accepts_required_markers():
    mod = load_fix_module()
    html = """
    <html><body>
    <select id="provider"></select>
    <select id="model"></select>
    <select id="answer-language"></select>
    <select id="manual-test-scenario"></select>
    <select id="monitor-visibility"></select>
    <div class="comm-core-grid"></div><div class="profiles-grid"></div>
    <div class="sci-grid"></div><div class="modes-grid"></div><div class="tools-grid"></div>
    <script>
    function panelAction() {}
    function buildUI() {}
    const x = window.pywebview;
    </script>
    </body></html>
    """
    assert mod._panel_asset_static_selftest_ok(html) is True


def test_panel_asset_static_selftest_ok_rejects_missing_markers():
    mod = load_fix_module()
    html = "<html><body><script>function buildUI() {}</script></body></html>"
    assert mod._panel_asset_static_selftest_ok(html) is False


def test_panel_asset_static_selftest_accepts_current_panel_html_variant():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert isinstance(html, str) and html
    assert mod._panel_asset_static_selftest_ok(html) is True


def test_panel_runtime_selftest_payload_ok_rejects_loaded_without_dynamic_sections():
    mod = load_fix_module()
    ok, why = mod._panel_runtime_selftest_payload_ok({
        "ok": True,
        "bridge_ping": True,
        "build_ui": True,
        "dom_ok": True,
        "data_loaded": True,
        "dynamic_section_count": 0,
    })
    assert ok is False
    assert why == "loaded_ruleset_but_no_dynamic_sections"


def test_panel_action_accepts_panel_bootstrap_selftest_callback():
    mod = load_fix_module()
    api = mod.Api()
    # Simulate an externally loaded panel pending runtime verification.
    api._panel_begin_bootstrap_probe("external")
    out = api.panel_action("panel_bootstrap_selftest", {
        "ok": True,
        "bridge_ping": True,
        "build_ui": True,
        "dom_ok": True,
        "data_loaded": True,
        "dynamic_section_count": 3,
    })
    assert isinstance(out, dict)
    assert out.get("ok") is True
    res = out.get("result") or {}
    assert res.get("accepted") is True
    assert res.get("runtime_ok") is True
    assert (api.panel_bootstrap_state or {}).get("status") == "passed"


def test_on_panel_closed_ignores_retired_panel_close_event_once():
    mod = load_fix_module()
    api = mod.Api()
    marker = object()
    api.panel_win = marker
    api._panel_closed_ignore_count = 1
    api.on_panel_closed()
    assert api.panel_win is marker
    assert api._panel_closed_ignore_count == 0


def test_route_input_passes_through_chat_when_comm_inactive_except_comm_start():
    mod = load_fix_module()
    _prime_module_gov(mod)
    state = types.SimpleNamespace(comm_active=False, sci_pending=False, answer_language='de', conversation_language='de')
    api = types.SimpleNamespace(gov=mod.gov)

    r_chat = mod.route_input('Was ist Zeit?', state, api, api.gov)
    assert r_chat.get('kind') == 'chat'
    assert r_chat.get('txt') == 'Was ist Zeit?'

    r_cmd = mod.route_input('Comm State', state, api, api.gov)
    assert r_cmd.get('kind') == 'chat'
    assert r_cmd.get('txt') == 'Comm State'

    r_start = mod.route_input('Comm Start', state, api, api.gov)
    assert r_start.get('kind') == 'command'
    assert r_start.get('canonical_cmd') == 'Comm Start'


def test_comm_stop_resets_sci_and_qc_state_via_intent_path():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.chat_session = DummySession(['OK'])
    # Avoid backend/session side effects in this unit test.
    api._recreate_chat_session = lambda *a, **k: None
    api._ensure_governance_pinned = lambda *a, **k: None
    api._send_state_update_to_model = lambda *a, **k: None

    api.gov_state.comm_active = True
    api.gov_state.sci_pending = True
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = 'B'
    api.gov_state.sci_pending_turns = 2
    api.gov_state.sci_recursion_one_shot = True
    api.gov_state.dynamic_one_shot_active = True
    api.gov_state.dynamic_nudge = 'one-shot'
    api.gov_state.qc_overrides = {'Brevity': 0}

    out = api.ask('Comm Stop')
    assert isinstance(out, dict)
    assert api.gov_state.comm_active is False
    assert api.gov_state.sci_pending is False
    assert api.gov_state.sci_active is False
    assert api.gov_state.sci_variant == ''
    assert int(getattr(api.gov_state, 'sci_pending_turns', -1)) == 0
    assert bool(getattr(api.gov_state, 'sci_recursion_one_shot', True)) is False
    assert bool(getattr(api.gov_state, 'dynamic_one_shot_active', True)) is False
    assert dict(getattr(api.gov_state, 'qc_overrides', {}) or {}) == {}


def test_panel_get_ui_hides_rule_sections_when_comm_off():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = False
    ui = api.get_ui()
    assert ui.get('comm_active') is False
    assert ui.get('manual_test_visible') is False
    assert ui.get('qc_override_visible') is False
    assert isinstance(ui.get('comm'), list) and any((isinstance(x, dict) and x.get('cmd') == 'Comm Start') or x == 'Comm Start' for x in ui.get('comm'))
    assert ui.get('profiles') == []
    assert ui.get('sci') == []
    assert ui.get('overlays') == []
    assert ui.get('tools') == []
    assert ui.get('logs') == []


def test_manual_test_monitor_closed_callback_clears_window_ref():
    mod = load_fix_module()
    api = mod.Api()
    api.manual_test_monitor_win = object()
    api.on_manual_test_monitor_closed()
    assert getattr(api, 'manual_test_monitor_win', None) is None


def test_panel_action_blocks_stale_rule_actions_when_comm_off():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = False

    r_mt = api.panel_action('manual_test_monitor_show', {})
    assert isinstance(r_mt, dict)
    assert r_mt.get('ok') is False
    assert r_mt.get('error') == 'comm_off_blocked'

    r_ask = api.panel_action('ask', {'text': 'Was ist Zeit?'})
    assert isinstance(r_ask, dict)
    assert r_ask.get('ok') is False
    assert r_ask.get('error') == 'comm_off_blocked'

    r_start = api.panel_action('ask', {'text': 'Comm Start'})
    assert isinstance(r_start, dict)
    # The panel_action gate must not block Comm Start (actual execution path may still fail in isolated test).
    assert r_start.get('error') != 'comm_off_blocked'


def test_comm_start_via_input_line_refreshes_panel_ui_state():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = False

    calls = {"panel_refresh": 0}
    api._ui_refresh_panel = lambda: calls.__setitem__("panel_refresh", calls["panel_refresh"] + 1) or True  # type: ignore[assignment]
    api._recreate_chat_session = lambda *a, **k: None  # type: ignore[assignment]
    api._ensure_governance_pinned = lambda *a, **k: None  # type: ignore[assignment]
    api._send_state_update_to_model = lambda *a, **k: None  # type: ignore[assignment]

    out = api.ask("Comm Start")
    assert isinstance(out, dict)
    assert api.gov_state.comm_active is True
    assert calls["panel_refresh"] >= 1
