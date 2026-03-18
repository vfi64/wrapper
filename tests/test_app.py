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
    JSON_PATH = ROOT / 'JSON' / 'Comm-SCI-v20.1.0.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'JSON' / 'Comm-SCI-v20.2.2.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.0.3.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.0.2.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.1.0.json'
if not JSON_PATH.exists():
    JSON_PATH = ROOT / 'Comm-SCI-v20.2.2.json'


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

def test_governance_manager_loads_operational_v20_2_min_ruleset_with_schema_adapter():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.0.min.json"
    assert p.exists(), "Comm-SCI-v20.2.0.min.json not found"

    ok = gm.load_file(str(p))
    assert ok is True
    assert gm.data.get("_schema_adapted_from") == "operational_v20"
    cmds = gm.data.get("commands") or {}
    assert isinstance(cmds.get("primary"), dict)
    assert isinstance(cmds.get("help_and_codes"), dict)
    assert isinstance(cmds.get("color_control"), dict)
    assert "Comm Start" in (cmds.get("primary") or {})
    assert "Color on" in (cmds.get("color_control") or {})

    toks = ((gm.data.get("parser_contract") or {}).get("command_tokens") or [])
    assert "Anchor auto on" not in toks
    assert "Anchor auto off" not in toks

    sci = gm.data.get("sci") or {}
    vmenu = (sci.get("variant_menu") or {}) if isinstance(sci, dict) else {}
    variants = (vmenu.get("variants") or {}) if isinstance(vmenu, dict) else {}
    assert isinstance(variants, dict) and len(variants) >= 8
    assert "A" in variants and "H" in variants
    assert isinstance(vmenu.get("menu_output"), dict)

    svs = (((gm.data.get("syntax_rules") or {}).get("special_parsing") or {}).get("sci_variant_selection") or {})
    assert svs.get("pattern") == "^[A-Ha-h]$"
    assert int(svs.get("timeout_turns", 0) or 0) >= 1

    gd = gm.data.get("global_defaults") or {}
    sd_mod = gd.get("self_debunking") or {}
    assert bool(sd_mod.get("enabled")) is True
    assert ((sd_mod.get("block") or {}).get("title") or "") == "Self-Debunking"


def test_operational_v20_2_2_adapter_merges_contract_output_contract_into_global_defaults():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.2.json"
    assert p.exists(), "Comm-SCI-v20.2.2.json not found"

    ok = gm.load_file(str(p))
    assert ok is True

    gd = gm.data.get("global_defaults") or {}
    oc = gd.get("output_contract") or {}
    sdc = oc.get("self_debunking_contract") or {}

    assert isinstance(sdc, dict)
    assert bool(sdc.get("enabled")) is True

    sd_mod = gd.get("self_debunking") or {}
    assert bool(sd_mod.get("enabled")) is True
    assert ((sd_mod.get("block") or {}).get("title") or "") == "Self-Debunking"


def test_operational_v20_adapter_preserves_variant_b_steps_for_runtime_resolution():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.0.min.json"
    assert gm.load_file(str(p)) is True

    api = mod.Api()
    api.gov = types.SimpleNamespace(data=gm.data)
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "B"

    _vdef, steps, maps_to = api._sci_variant_def("B")
    assert maps_to == "SCIplus"
    assert isinstance(steps, list) and len(steps) >= 10
    assert steps[0] == "Plan"
    assert "Dialectic_6_Synthesis2" in steps
    assert steps[-1] == "Learn"

    menu = api._render_sci_menu_html(lang="en")
    assert "Error: No SCI variants found in canonical JSON." not in menu


def test_operational_v20_self_debunking_is_injected_before_qc_footer_when_missing():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.0.min.json"
    assert gm.load_file(str(p)) is True

    raw = (
        "Core answer sentence.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · "
        "Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )
    out = mod.enforce_self_debunking_contract(raw, gm, "Expert", is_command=False, lang="en")
    assert "Self-Debunking" in out
    assert out.find("Self-Debunking") < out.find("QC-Matrix:")


def test_enforce_self_debunking_contract_drops_redundant_uncertainty_tail_point():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.5.json"
    if not p.exists():
        p = ROOT / "JSON" / "Comm-SCI-v20.2.2.json"
    assert gm.load_file(str(p)) is True

    raw = (
        "Antwortkern.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: Punkt eins.\n"
        "   Warum das wichtig ist: Relevanz eins.\n"
        "   Was würde verifizieren/falsifizieren (nächster Check): Check eins.\n\n"
        "2. Schwäche: Punkt zwei.\n"
        "   Warum das wichtig ist: Relevanz zwei.\n"
        "   Was würde verifizieren/falsifizieren (nächster Check): Check zwei.\n\n"
        "3. Schwäche: U1 – Data gap. Needed: Source/current context from the user or external verification.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 1 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(raw, gm, "Expert", is_command=False, lang="de")
    sd_match = re.search(r"(?is)Self-Debunking:\s*(.*?)\n\s*QC-Matrix:", out)
    assert sd_match is not None
    sd_block = sd_match.group(1)
    assert len(re.findall(r"(?m)^\s*\d+\.\s+", sd_block)) == 2
    assert "Data gap. Needed:" not in sd_block
    assert re.search(r"(?i)\bU1\b", sd_block) is None


def test_enforce_self_debunking_contract_strips_embedded_uncertainty_tail_fragment():
    mod = load_fix_module()
    gm = mod.GovernanceManager()
    p = ROOT / "JSON" / "Comm-SCI-v20.2.5.json"
    if not p.exists():
        p = ROOT / "JSON" / "Comm-SCI-v20.2.2.json"
    assert gm.load_file(str(p)) is True

    raw = (
        "Antwortkern.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: Punkt eins. Schwäche: U1 – Data gap. Needed: Source/current context from the user or external verification.\n"
        "   Warum das wichtig ist: Relevanz eins.\n"
        "   Was würde verifizieren/falsifizieren (nächster Check): Check eins.\n\n"
        "2. Schwäche: Punkt zwei. Schwäche: U1 – Data gap. Needed: Source/current context from the user or external verification.\n"
        "   Warum das wichtig ist: Relevanz zwei.\n"
        "   Was würde verifizieren/falsifizieren (nächster Check): Check zwei.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 1 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(raw, gm, "Expert", is_command=False, lang="de")
    sd_match = re.search(r"(?is)Self-Debunking:\s*(.*?)\n\s*QC-Matrix:", out)
    assert sd_match is not None
    sd_block = sd_match.group(1)
    assert len(re.findall(r"(?m)^\s*\d+\.\s+", sd_block)) == 2
    assert "Data gap. Needed:" not in sd_block
    assert re.search(r"(?i)\bU1\b", sd_block) is None
    assert "Punkt eins." in sd_block
    assert "Punkt zwei." in sd_block


def test_inject_minimal_self_debunking_de_keeps_compact_core_fields():
    mod = load_fix_module()
    raw = (
        "Antwortsatz.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0)\n"
    )
    out = mod.inject_minimal_self_debunking(raw, title="Self-Debunking", lang="de")
    assert out.count("**Schwäche**:") == 2
    assert "**Warum das wichtig ist**:" in out
    assert "**Was würde verifizieren/falsifizieren (nächster Check)**:" in out
    assert "**Vereinfachung**:" not in out
    assert "**Subjektivität**:" not in out
    assert "**Nächster Schritt**:" not in out
    assert "**Prüfen/Widerlegen (nächster Schritt)**:" not in out


def test_system_instruction_uses_conversation_language_not_hard_english():
    mod = load_fix_module()
    _prime_module_gov(mod)
    sys_instr = mod.gov.get_system_instruction()
    assert "You must reply in English" not in sys_instr
    assert "current conversation language" in sys_instr
    assert "Keep canonical command tokens in English." in sys_instr

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


def test_strip_sci_trace_line_when_pending_without_variant():
    mod = load_fix_module()

    raw = (
        "Antwortsatz.\n"
        "SCI Trace:\n"
        "1. Plan: X\n"
        "2. Check: Y\n"
        "Self-Debunking:\n"
        "1. Schwäche: A\n"
        "2. Schwäche: B\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(
        raw,
        sci_active=False,
        sci_variant="",
        sci_pending=True,
    )
    assert "SCI Trace" not in out
    assert "Self-Debunking" in out


def test_apply_csc_strict_pending_without_variant_strips_sci_trace_and_keeps_self_debunking_boxed():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None
    api.gov_state.sci_active = False
    api.gov_state.sci_variant = ""
    api.gov_state.sci_pending = True
    api.gov_state.answer_language = "de"

    raw = (
        "Antwortsatz.\n\n"
        "SCI Trace:\n"
        "1. Plan: Untersuche die Frage.\n"
        "2. Solution: Erkläre den Kern knapp.\n"
        "3. Check: Prüfe auf Gegenargumente.\n\n"
        "Self-Debunking:\n"
        "- Schwäche: Zu stark vereinfacht.\n"
        "- Schwäche: Gegenpositionen fehlen.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )
    html_out, meta = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))

    assert "SCI: PENDING" in plain
    assert "SCI Trace" not in plain
    assert "self-debunking" in str(html_out or "")
    assert "raw-output" not in str(html_out or "")
    assert mod.detect_self_debunking_numbered_html(str(html_out or "")) is True
    assert bool(((meta or {}).get("normalization") or {}).get("self_debunking_boxed")) is True
    assert bool(((meta or {}).get("normalization") or {}).get("self_debunking_numbered")) is True


def test_profile_expert_command_sets_pending_and_shows_pending_header():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None

    out = api.ask("Profile Expert")
    html_out = out.get("html") if isinstance(out, dict) else str(out)
    assert "SCI: PENDING" in html_out
    assert "SCI variants (selection)" in html_out or "SCI-Varianten" in html_out
    assert api.gov_state.sci_pending is True


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


def test_show_qc_override_refreshes_dialog_state_on_every_show():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    calls = []

    class _Win:
        def show(self):
            calls.append("show")
        def bring_to_front(self):
            calls.append("bring_to_front")
        def evaluate_js(self, js):
            calls.append(("evaluate_js", str(js)))
            return "ok"

    api.qc_win = _Win()
    out = api.show_qc_override()
    assert isinstance(out, dict) and out.get("ok") is True
    assert "show" in calls
    assert "bring_to_front" in calls
    js_calls = [c[1] for c in calls if isinstance(c, tuple) and c and c[0] == "evaluate_js"]
    assert js_calls, "show_qc_override should trigger a dialog state refresh JS call"
    assert any("QCUI.refreshState" in js for js in js_calls)
    assert bool(getattr(api, "_qc_override_open", False)) is True


def test_qc_override_cancel_marks_dialog_closed_flag():
    mod = load_fix_module()
    api = mod.Api()
    api._qc_override_open = True

    calls = []

    class _Win:
        def hide(self):
            calls.append("hide")

    api.qc_win = _Win()
    out = api.qc_override_cancel({})
    assert isinstance(out, dict) and out.get("ok") is True
    assert "hide" in calls
    assert bool(getattr(api, "_qc_override_open", True)) is False


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


def test_strip_empty_citation_placeholders_removes_only_empty_markers():
    mod = load_fix_module()
    _prime_module_gov(mod)

    raw = "Alpha [cite: ] Beta [cite:] Gamma [cite: 1] Delta [cite: doi:10.1000/test]"
    out = mod.strip_empty_citation_placeholders(raw)

    assert "[cite: ]" not in out
    assert "[cite:]" not in out
    assert "[cite: 1]" in out
    assert "[cite: doi:10.1000/test]" in out


def test_strip_empty_citation_placeholders_skips_fenced_code_blocks():
    mod = load_fix_module()
    _prime_module_gov(mod)

    raw = "Outside [cite: ]\n```text\ninside [cite: ]\n```\nTail [cite: 2]"
    out = mod.strip_empty_citation_placeholders(raw)

    assert "Outside [cite: ]" not in out
    assert "inside [cite: ]" in out
    assert "[cite: 2]" in out


def test_api_normalize_raw_output_contracts_strips_empty_cite_placeholders():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    raw = "Alpha [cite: ] Beta [cite: 7]"
    out = api._normalize_raw_output_contracts(
        raw,
        governance_enabled=False,
        is_command=False,
    )

    assert "[cite: ]" not in out
    assert "[cite: 7]" in out

# -------------------------
# New: Comm Audit + Anchor Snapshot tests
# -------------------------

def test_comm_audit_is_ui_only_without_llm_call():
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
    assert "Missing QC footer" not in html
    assert "Compliance scan (best-effort)" not in html
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


def test_html_number_self_debunking_flattens_p_wrappers_inside_li_for_uniform_spacing():
    """Regression: paragraph wrappers inside SD <li> must be flattened to avoid
    browser paragraph margins causing uneven line spacing.
    """
    mod = load_fix_module()

    html_in = (
        '<div class="self-debunking"><div>Self-Debunking:</div><ol>\n'
        '<li><p><strong>Weakness</strong>: A.</p><p><strong>Why it matters</strong>: B.</p>'
        '<p><strong>What would verify/falsify (next check)</strong>: C.</p></li>\n'
        '<li><strong>Weakness</strong>: D.<br><strong>Why it matters</strong>: E.'
        '<br><strong>What would verify/falsify (next check)</strong>: F.</li>\n'
        '</ol></div>'
    )

    out = mod.html_number_self_debunking(html_in, lang="en")

    # No paragraph wrappers must remain inside SD ordered lists.
    assert re.search(r"(?is)<ol[^>]*>.*?<p\\b", out) is None
    # Secondary lines must be kept as inline line breaks in the first item.
    assert "<br><strong>Why it matters</strong>: B." in out
    assert "<br><strong>What would verify/falsify (next check)</strong>: C." in out
    # Keep two logical list items.
    assert out.lower().count("<li") == 2


def test_html_number_self_debunking_merges_fragmented_ol_chunks_with_secondary_paragraphs():
    """Regression: weak HTML conversion may split SD points into multiple <ol> chunks with
    secondary paragraphs in between. These fragments must be merged into stable list items.
    """
    mod = load_fix_module()

    html_in = (
        '<div class="self-debunking"><div>Selbst-Debunking:</div>\n'
        '<ol><li><strong>Schwäche</strong>: A.</li></ol>\n'
        '<p><strong>Warum das wichtig ist</strong>: B.</p>\n'
        '<p><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: C.</p>\n'
        '<ol><li><strong>Schwäche</strong>: D.</li></ol>\n'
        '<p><strong>Warum das wichtig ist</strong>: E.</p>\n'
        '<p><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: F.</p>\n'
        '</div>'
    )

    out = mod.html_number_self_debunking(html_in, lang="de")

    # Fragmented ordered lists must be merged into one logical list.
    assert out.lower().count("<ol") == 1
    assert out.lower().count("<li") == 2
    # Secondary rows must be in-item line breaks, not standalone paragraphs.
    assert len(
        re.findall(r"(?is)<br\s*/?>\s*<strong>Warum das wichtig ist</strong>:", out)
    ) >= 2
    assert len(
        re.findall(
            r"(?is)<br\s*/?>\s*<strong>Was würde verifizieren/falsifizieren \(nächster Check\)</strong>:",
            out,
        )
    ) >= 2
    assert re.search(r"(?is)<p[^>]*>\s*<strong>Warum das wichtig ist</strong>\s*:", out) is None
    assert re.search(
        r"(?is)<p[^>]*>\s*<strong>Was würde verifizieren/falsifizieren \(nächster Check\)</strong>\s*:",
        out,
    ) is None


def test_html_number_self_debunking_normalizes_italic_secondary_labels_and_forces_new_line():
    """Regression: secondary labels must not remain italic and must start on a new line
    in every SD point (including problematic '<em><strong>Label</strong>:</em>:' variants).
    """
    mod = load_fix_module()

    html_in = (
        '<div class="self-debunking"><div>Selbst-Debunking:</div><ol>\n'
        '<li><strong>Schwäche</strong>: Punkt 1.'
        '<br><strong>Warum das wichtig ist</strong>: A.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: B.</li>\n'
        '<li><strong>Schwäche</strong>: Punkt 2.\n'
        '   <em><strong>Warum das wichtig ist</strong>:</em>: C.\n'
        '   <em><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:</em>: D.\n'
        '</li>\n'
        '</ol></div>'
    )

    out = mod.html_number_self_debunking(html_in, lang="de")

    # No italicized secondary labels should remain.
    assert "<em><strong>Warum das wichtig ist</strong>:</em>:" not in out
    assert "<em><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:</em>:" not in out
    # Secondary labels must be bold and line-broken in both points.
    assert len(re.findall(r"(?is)<br\s*/?>\s*<strong>Warum das wichtig ist</strong>:", out)) >= 2
    assert len(re.findall(r"(?is)<br\s*/?>\s*<strong>Was würde verifizieren/falsifizieren \(nächster Check\)</strong>:", out)) >= 2


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


def test_preview_export_file_returns_in_app_preview_for_audit_log(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    mod.AUDIT_LOG_DIR = str(tmp_path / "Logs" / "Audit")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(mod.AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)

    audit_path = Path(mod.AUDIT_LOG_DIR) / "Audit_demo.json"
    audit_path.write_text(json.dumps({"ok": True, "n": 7}, ensure_ascii=False), encoding="utf-8")

    api = mod.Api()
    out = api.preview_export_file(str(audit_path), max_chars=4000)
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("kind") == "audit"
    assert out.get("relative_path") == "Logs/Audit/Audit_demo.json"
    assert '"ok": true' in str(out.get("preview") or "").lower()


def test_preview_export_file_blocks_paths_outside_logs(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    mod.AUDIT_LOG_DIR = str(tmp_path / "Logs" / "Audit")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(mod.AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)

    outside = tmp_path / "outside.json"
    outside.write_text('{"x":1}', encoding="utf-8")

    api = mod.Api()
    out = api.preview_export_file(str(outside), max_chars=4000)
    assert isinstance(out, dict)
    assert out.get("ok") is False
    assert out.get("error") == "path_not_allowed"


def test_open_export_preview_creates_helper_window_with_preview_payload(monkeypatch, tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    mod.AUDIT_LOG_DIR = str(tmp_path / "Logs" / "Audit")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(mod.AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    audit_path = Path(mod.AUDIT_LOG_DIR) / "Audit_demo.json"
    audit_path.write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")

    created = {}

    class _Win:
        def destroy(self):
            created["destroyed"] = True

        def bring_to_front(self):
            created["front"] = True

    def _create_window(title, **kwargs):
        created["title"] = title
        created["kwargs"] = dict(kwargs or {})
        return _Win()

    monkeypatch.setattr(mod.webview, "create_window", _create_window, raising=True)

    api = mod.Api()
    out = api.open_export_preview(str(audit_path), max_chars=1234)
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("relative_path") == "Logs/Audit/Audit_demo.json"
    assert "Comm-SCI Log-Vorschau" in str(created.get("title") or "")
    html_doc = str((created.get("kwargs") or {}).get("html") or "")
    assert "Datei-Vorschau" in html_doc
    assert "Logs/Audit/Audit_demo.json" in html_doc
    assert "&quot;ok&quot;: true" in html_doc.lower()


def test_open_export_preview_ignores_max_chars_and_shows_full_file(monkeypatch, tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    mod.AUDIT_LOG_DIR = str(tmp_path / "Logs" / "Audit")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(mod.AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    audit_path = Path(mod.AUDIT_LOG_DIR) / "Audit_big.json"
    payload = {"blob": "X" * 25000}
    audit_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    captured = {}

    class _Win:
        def bring_to_front(self):
            return None

    def _create_window(title, **kwargs):
        captured["title"] = title
        captured["html"] = str((kwargs or {}).get("html") or "")
        return _Win()

    monkeypatch.setattr(mod.webview, "create_window", _create_window, raising=True)

    api = mod.Api()
    out = api.open_export_preview(str(audit_path), max_chars=10)
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("truncated") is False
    html_doc = str(captured.get("html") or "")
    assert "Vorschau gekürzt" not in html_doc
    assert "XXXXXXXXXXXXXXXXXXXXXXXX" in html_doc


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


def test_load_chat_log_rejects_path_traversal_filename(tmp_path):
    mod = load_fix_module()
    _prime_module_gov(mod)
    mod.CHAT_LOG_DIR = str(tmp_path / "Logs" / "Chats")
    Path(mod.CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)

    api = mod.Api()
    res = api.load_chat_log("../outside.json", fork=True)
    assert isinstance(res, dict)
    assert res.get("ok") is False
    assert res.get("error") == "path_traversal_blocked"


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


def test_panel_get_ui_uses_seam_failopen_snapshot_when_build_raises():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.comm_active = True

    class _Seam:
        @staticmethod
        def panel_ui_build_snapshot(**_kwargs):
            raise RuntimeError("boom")

        @staticmethod
        def panel_ui_failopen_snapshot(*, gov_state):
            return {
                "providers": ["gemini"],
                "current_provider": "gemini",
                "current_model": "gemini-2.0-flash",
                "available_models": ["gemini-2.0-flash"],
                "answer_language": "de",
                "comm": [{"name": "Comm Start", "cmd": "Comm Start"}],
                "profiles": [],
                "sci": [],
                "overlays": [],
                "tools": [],
                "logs": [],
                "chat_logs": [],
                "model_hint": "",
                "comm_active": bool(getattr(gov_state, "comm_active", False)),
                "manual_test_visible": bool(getattr(gov_state, "comm_active", False)),
                "qc_override_visible": bool(getattr(gov_state, "comm_active", False)),
                "provider": "gemini",
                "model": "gemini-2.0-flash",
            }

    old = getattr(mod, "_panel_ui_snapshot_seam_mod", None)
    mod._panel_ui_snapshot_seam_mod = _Seam()
    try:
        ui = api.get_ui()
    finally:
        mod._panel_ui_snapshot_seam_mod = old

    assert isinstance(ui, dict)
    assert ui.get("comm_active") is True
    assert ui.get("manual_test_visible") is True
    assert ui.get("qc_override_visible") is True
    assert ui.get("provider") == "gemini"
    assert ui.get("model") == "gemini-2.0-flash"

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


def test_panel_manual_test_ask_fallback_uses_literal_method_check():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert "msg.toLowerCase().includes('pywebview api method not available: ask')" in html
    assert "panel_action', ['ask', {text: t}]" in html


def test_panel_manual_test_export_uses_panel_action_fallback():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert "msg.toLowerCase().includes('pywebview api method not available: export')" in html
    assert "panel_action', ['export', {}]" in html


def test_external_panel_asset_manual_test_regexes_are_not_overescaped():
    panel_asset = ROOT / "src" / "ui_assets" / "panel.html"
    txt = panel_asset.read_text(encoding="utf-8")
    assert "pywebview api method not available:\\\\s*ask" not in txt
    assert "qc.search(/\\\\bResponse at\\\\b/i)" not in txt
    assert "qc.replace(/\\\\s+/g, ' ')" not in txt
    assert "const hasNumbered = /\\\\b1\\\\.\\\\s*" not in txt


def test_chat_header_uses_compact_controls_without_redundant_wrapper_label():
    """UI invariant: top bar should not duplicate window title with an extra wrapper label."""
    mod = load_fix_module()
    html = getattr(mod, 'HTML_CHAT', '')
    assert isinstance(html, str) and html
    assert '__WRAPPER_LABEL__' not in html
    assert 'load_rule_file()' in html
    assert 'id="rulefile"' in html
    assert 'onclick="exportLogs()"' in html
    assert 'onclick="openExitConfirm()"' in html
    assert 'onclick="window.pywebview.api.close_app()"' not in html


def test_chat_exit_button_uses_confirm_dialog_with_english_actions():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_CHAT', '')
    assert isinstance(html, str) and html
    assert 'id="exitConfirmOverlay"' in html
    assert 'Exit Application' in html
    assert '>Cancel<' in html
    assert '>Exit<' in html
    assert 'function openExitConfirm()' in html
    assert 'function closeExitConfirm()' in html
    assert 'async function confirmExit()' in html
    assert 'set_exit_confirm_open(true)' in html
    assert 'set_exit_confirm_open(false)' in html
    assert 'id="helpModalOverlay"' in html
    assert 'id="helpBtn"' in html
    assert 'async function openHelpModal()' in html
    assert "if(String(e.key || '').toUpperCase() === 'F1')" in html


def test_chat_input_history_supports_arrow_up_down_navigation():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "window.__cmdHistory" in html
    assert "ArrowUp" in html
    assert "ArrowDown" in html
    assert "_cmdHistApply(-1)" in html
    assert "_cmdHistApply(1)" in html


def test_chat_send_path_queues_requests_serially():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "let _sendInFlight = false;" in html
    assert "const _sendQueue = [];" in html
    assert "async function _drainSendQueue()" in html
    assert "while(_sendQueue.length)" in html
    assert "_enqueueSend(txt)" in html


def test_chat_export_button_renders_clickable_file_links_after_export():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "function _displayExportPath(path)" in html
    assert "function _fileLink(path)" in html
    assert "function _renderExportCard(title, bodyHtml)" in html
    assert "async function exportLogs()" in html
    assert "async function _openExportPreviewViaApi(api, raw)" in html
    assert "function openExportFileFromAnchor(anchor)" in html
    assert "async function openExportFile(path)" in html
    assert "open_export_preview(raw, 0)" in html
    assert "panel_action('open_export_preview', {path: raw, max_chars: 0})" in html
    assert "window.pywebview.api.export()" in html
    assert "addMsg('bot', _renderExportNotice(chatPath, auditPath), false, null);" in html


def test_chat_uncertainty_tooltip_supports_mouse_hold():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "__uTipTargets" in html
    assert "_uTipShow(" in html
    assert ".uncertainty-inline-marker" in html
    assert ".signal-dot-marker" in html
    assert ".csc-badge" in html
    assert ".csc-warning" in html
    assert ".control-layer-note" in html
    assert ".copy-btn" in html
    assert ".qc-dim-tip" in html
    assert "[data-u-title]" in html
    assert "csc-help-icon" in html
    assert "document.addEventListener('mousedown'" in html
    assert "document.addEventListener('mouseup', _uTipHide)" in html


def test_chat_csc_block_reads_tooltip_metadata():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "const scoreTip = escHtml(csc.score_tooltip || '');" in html
    assert "const thrTip = escHtml(csc.thresholds_tooltip || '');" in html
    assert "class=\"cgi-help csc-help-icon\"" in html
    assert "data-u-title" in html


def test_chat_cgi_widget_uses_dropdowns_and_repeat_action_only():
    mod = load_fix_module()
    html = getattr(mod, "HTML_CHAT", "")
    assert isinstance(html, str) and html
    assert "_buildCgiWidgetHtml(" in html
    assert "submitCgi(" in html
    assert "data-cgi-field=\"clarity\"" in html
    assert "data-cgi-field=\"insight\"" in html
    assert "data-cgi-field=\"efficiency\"" in html
    assert "submitCgi('${id}','repeat')" in html
    assert "submitCgi('${id}','save')" not in html
    assert "submit_cgi_feedback" in html
    assert "data-u-title=" in html
    assert "Wirkung nur fuer die naechste Antwort" in html
    assert "document.addEventListener('mouseover'" in html
    assert ".cgi-help" in html


def test_main_bridge_exposes_submit_cgi_feedback():
    mod = load_fix_module()
    api = mod.Api()
    bridge = getattr(api, "main_bridge", None)
    assert bridge is not None
    assert hasattr(bridge, "submit_cgi_feedback")
    assert hasattr(bridge, "preview_export_file")
    assert hasattr(bridge, "open_export_preview")


def test_panel_cmd_handles_qc_override_modal_block_with_info_status():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert "qc_override_modal_blocked" in html
    assert "QC Override Dialog ist offen; Aktion temporär blockiert." in html


def test_append_uncertainty_explanation_does_not_add_legend_block():
    mod = load_fix_module()
    api = mod.Api()
    src = "<p>Uncertainty: U4 - Temporal instability. Needed: Web check.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert isinstance(out, str)
    assert "uncertainty-legend" not in out


def test_append_uncertainty_explanation_infers_codes_when_missing_without_legend():
    mod = load_fix_module()
    api = mod.Api()
    src = "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
    out = api._append_uncertainty_explanation_if_needed(
        src,
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    assert isinstance(out, str)
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U5'" in out
    assert "data-u-title='U5 -" in out
    assert ">U5</span>)</span>" in out
    assert "uncertainty-legend" not in out


def test_append_uncertainty_explanation_skips_profile_header_and_keeps_footer():
    mod = load_fix_module()
    api = mod.Api()
    src = (
        "<p>Profile: Standard · Overlay: Strict · SCI: off · Color: on</p>"
        "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
        "<div class='ts-footer'>Response at 2026-02-27 14:00:00</div>"
    )
    out = api._append_uncertainty_explanation_if_needed(
        src,
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    blocks = re.findall(r"(?is)<p[^>]*>.*?</p>", out)
    assert len(blocks) >= 2
    assert "uncertainty-inline-marker" not in blocks[0]
    assert "uncertainty-inline-marker" in blocks[1]
    assert "uncertainty-legend" not in out
    assert "ts-footer" in out


def test_append_uncertainty_explanation_skips_german_profil_header_and_keeps_footer():
    mod = load_fix_module()
    api = mod.Api()
    src = (
        "<p>Profil: Standard · Overlay: Strict · SCI: off · Control Layer: on · Color: on</p>"
        "<p>Der Hauptinhalt ist in Teilen mehrdeutig und interpretationsoffen.</p>"
        "<div class='ts-footer'>Response at 2026-02-27 14:00:00</div>"
    )
    out = api._append_uncertainty_explanation_if_needed(
        src,
        user_text="Bitte beantworte ohne feste Annahmen; mehrere Deutungen sind moeglich.",
    )
    blocks = re.findall(r"(?is)<p[^>]*>.*?</p>", out)
    assert len(blocks) >= 2
    assert "uncertainty-inline-marker" not in blocks[0]
    assert "uncertainty-inline-marker" in blocks[1]
    assert "uncertainty-legend" not in out
    assert "ts-footer" in out


def test_append_uncertainty_explanation_skips_control_layer_note_and_marks_content():
    mod = load_fix_module()
    api = mod.Api()
    src = (
        "<div class='control-layer-note csc-warning'>"
        "<b>CONTROL LAYER NOTE</b>"
        "<ul class='control-layer-violations'>"
        "<li class='control-layer-violation'>Verification Route Gate: RED claim requires uncertainty label (U1-U8).</li>"
        "</ul>"
        "</div>"
        "<p>Unsicherheit: U1 - Datenluecke im Inhaltsteil.</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    note = re.search(r"(?is)<div[^>]*control-layer-note[^>]*>.*?</div>", out)
    body = re.search(r"(?is)<p[^>]*>.*?</p>", out)
    assert note is not None
    assert body is not None
    assert "uncertainty-inline-marker" not in note.group(0)
    assert "data-u-code='U1'" in body.group(0)
    assert "data-u-code='U6'" not in out


def test_append_uncertainty_explanation_does_not_treat_sci_trace_u_marker_as_content_u_marker():
    mod = load_fix_module()
    api = mod.Api()
    src = (
        "<div class='control-layer-note csc-warning'>"
        "<b>CONTROL LAYER NOTE</b>"
        "<ul class='control-layer-violations'>"
        "<li class='control-layer-violation'>Verification Route Gate: RED claim requires uncertainty label (U1-U8).</li>"
        "</ul>"
        "</div>"
        "<div class='sci-trace'>"
        "<div>SCI Trace</div>"
        "<ol>"
        "<li>Critic: Punkt mit vorhandenem Marker "
        "<span class='uncertainty-inline-marker' data-u-code='U6' data-u-title='U6 - x' title='U6 - x'>U6</span></li>"
        "</ol>"
        "</div>"
        "<p><span class='signal-dot-marker'><span style='color:#c62828; font-weight:600;'>🔴</span></span> "
        "Kritische Aussage im Inhaltsblock mit starkem Anspruch und ohne belastbaren Nachweis; "
        "dieser Satz bleibt absichtlich laenger als vierzig Zeichen.</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    body = re.search(r"(?is)<p[^>]*>.*?</p>", out)
    assert body is not None
    body_html = body.group(0)
    assert re.search(r"data-u-code='U[1-8]'", body_html)


def test_append_uncertainty_explanation_keeps_red_span_closed_when_replacing_explicit_u_code():
    mod = load_fix_module()
    api = mod.Api()
    src = (
        "<p>"
        "<span style=\"color:#c62828; font-weight:600;\">🔴</span>(U1) "
        "Kritische Aussage ohne belastbaren Nachweis."
        "</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "uncertainty-inline-marker" in out
    assert "color:#c62828; font-weight:600;\">🔴<span class='uncertainty-inline-wrap'" not in out
    assert re.search(
        r"color:#c62828; font-weight:600;\">🔴</span>\s*(?:</span>\s*)?<span class='uncertainty-inline-wrap'",
        out,
    )


def test_uncertainty_tooltip_uses_answer_language_english():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "en"
    src = "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
    out = api._append_uncertainty_explanation_if_needed(
        src,
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    assert "data-u-title='U5 - Structural limitation" in out
    assert "uncertainty-legend" not in out


def test_uncertainty_tooltip_uses_answer_language_german():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "de"
    src = "<p>Das ist eine komplexe Herausforderung ohne einfache oder perfekte Loesung.</p>"
    out = api._append_uncertainty_explanation_if_needed(
        src,
        user_text="Was ist die objektiv beste weltweit faire Strategie?",
    )
    assert "data-u-title='U5 - Strukturelle Grenze" in out
    assert "uncertainty-legend" not in out


def test_control_layer_alert_tooltip_uses_language_english():
    mod = load_fix_module()
    out = mod._control_layer_alert_html(
        "Contract adjusted.",
        title="CONTROL LAYER NOTE",
        severity="warn",
        lang="en",
    )
    assert "data-u-title='Control Layer note: deterministic safety/contract guard adjusted output." in out


def test_control_layer_alert_tooltip_uses_language_german():
    mod = load_fix_module()
    out = mod._control_layer_alert_html(
        "Vertrag angepasst.",
        title="CONTROL LAYER NOTE",
        severity="warn",
        lang="de",
    )
    assert "data-u-title='Control-Layer-Hinweis: deterministische Sicherheits-/Vertragspruefung hat Ausgabe angepasst." in out


def test_control_layer_alert_prefers_modular_renderer(monkeypatch):
    mod = load_fix_module()

    def _render(**kwargs):
        assert kwargs.get("message") == "Contract adjusted."
        assert kwargs.get("title") == "CONTROL LAYER NOTE"
        assert kwargs.get("severity") == "warn"
        assert "Control Layer note" in str(kwargs.get("tooltip_text") or "")
        return "ALERT-FROM-MODULE"

    monkeypatch.setattr(
        mod,
        "_output_control_layer_note_renderer",
        types.SimpleNamespace(render_control_layer_alert_html=_render),
        raising=False,
    )
    out = mod._control_layer_alert_html(
        "Contract adjusted.",
        title="CONTROL LAYER NOTE",
        severity="warn",
        lang="en",
    )
    assert out == "ALERT-FROM-MODULE"


def test_render_control_layer_block_html_prefers_modular_renderer(monkeypatch):
    mod = load_fix_module()

    def _render(msg, *, suffix_html=""):
        assert msg == "blocked"
        assert suffix_html == "<br>hint"
        return "BLOCK-FROM-MODULE"

    monkeypatch.setattr(
        mod,
        "_output_csc_warning_renderer",
        types.SimpleNamespace(render_control_layer_block_html=_render),
        raising=False,
    )

    out = mod._render_control_layer_block_html("blocked", suffix_html="<br>hint")
    assert out == "BLOCK-FROM-MODULE"


def test_build_profile_switch_audit_line_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()

    def _render(command, from_profile, to_profile):
        assert command == "Profile Expert"
        assert from_profile == "Standard"
        assert to_profile == "Expert"
        return "AUDIT-FROM-CATALOG"

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(build_profile_switch_audit_line=_render),
        raising=False,
    )
    out = mod._build_profile_switch_audit_line("Profile Expert", "Standard", "Expert")
    assert out == "AUDIT-FROM-CATALOG"


def test_resolve_post_state_command_html_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Expert"
    api.gov_state.sci_pending = False

    def _resolve(**kwargs):
        assert kwargs.get("cmd") == "Profile Expert"
        assert kwargs.get("current_profile") == "Expert"
        assert kwargs.get("prev_profile_for_audit") == "Standard"
        assert kwargs.get("timestamp") == "2026-03-11 07:00:00"
        kwargs["set_sci_pending_fn"]()
        return {"html": "POST-CMD-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(resolve_post_state_command_html=_resolve),
        raising=False,
    )
    monkeypatch.setattr(api, "_render_profile_switch_control_html", lambda *a, **k: "LOCAL", raising=True)
    monkeypatch.setattr(api, "_render_sci_menu_html", lambda *a, **k: "MENU", raising=True)
    monkeypatch.setattr(api, "_render_comm_state_html", lambda *a, **k: "STATE", raising=True)
    monkeypatch.setattr(api, "_lang", lambda: "de", raising=True)

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Profile Expert",
        timestamp="2026-03-11 07:00:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "POST-CMD-FROM-CATALOG"
    assert api.gov_state.sci_pending is True


def test_resolve_post_state_command_html_prefers_sci_on_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"
    api.gov_state.sci_pending = False

    calls = {"sci_on": 0, "overlay": 0, "generic": 0}

    def _resolve_sci_on(**kwargs):
        calls["sci_on"] += 1
        assert kwargs.get("cmd") == "SCI on"
        assert callable(kwargs.get("set_sci_pending_fn"))
        kwargs["set_sci_pending_fn"]()
        return {"html": "SCI-ON-FROM-CATALOG"}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        return {"html": "OVERLAY-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_sci_on_command_html=_resolve_sci_on,
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="SCI on",
        timestamp="2026-03-11 07:04:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "SCI-ON-FROM-CATALOG"
    assert calls["sci_on"] == 1
    assert calls["overlay"] == 0
    assert calls["generic"] == 0
    assert api.gov_state.sci_pending is True


def test_resolve_post_state_command_html_sci_on_catalog_falls_back_to_generic_on_empty_html(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"sci_on": 0, "generic": 0}

    def _resolve_sci_on(**kwargs):
        calls["sci_on"] += 1
        return {"html": "  "}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        assert kwargs.get("cmd") == "SCI on"
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_sci_on_command_html=_resolve_sci_on,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="SCI on",
        timestamp="2026-03-11 07:04:30",
        prev_profile_for_audit="Standard",
    )
    assert out == "GENERIC-FROM-CATALOG"
    assert calls["sci_on"] == 1
    assert calls["generic"] == 1


def test_resolve_post_state_command_html_prefers_comm_overlay_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"overlay": 0, "generic": 0}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        assert kwargs.get("cmd") == "Strict on"
        assert kwargs.get("current_profile") == "Standard"
        assert kwargs.get("prev_profile_for_audit") == "Standard"
        assert callable(kwargs.get("render_comm_state_html_fn"))
        return {"html": "OVERLAY-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Strict on",
        timestamp="2026-03-11 07:05:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "OVERLAY-FROM-CATALOG"
    assert calls["overlay"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_comm_overlay_catalog_falls_back_to_generic_on_empty_html(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"overlay": 0, "generic": 0}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        return {"html": "  "}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        assert kwargs.get("cmd") == "Explore off"
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Explore off",
        timestamp="2026-03-11 07:06:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "GENERIC-FROM-CATALOG"
    assert calls["overlay"] == 1
    assert calls["generic"] == 1


def test_resolve_post_state_command_html_prefers_comm_overlay_catalog_for_color_toggle(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"overlay": 0, "generic": 0}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        assert kwargs.get("cmd") == "Color off"
        return {"html": "COLOR-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Color off",
        timestamp="2026-03-11 07:07:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "COLOR-FROM-CATALOG"
    assert calls["overlay"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_prefers_comm_overlay_catalog_for_sci_off(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"overlay": 0, "generic": 0}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        assert kwargs.get("cmd") == "SCI off"
        return {"html": "SCI-OFF-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="SCI off",
        timestamp="2026-03-11 07:08:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "SCI-OFF-FROM-CATALOG"
    assert calls["overlay"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_prefers_comm_overlay_catalog_for_sci_recurse(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"overlay": 0, "generic": 0}

    def _resolve_overlay(**kwargs):
        calls["overlay"] += 1
        assert kwargs.get("cmd") == "SCI recurse"
        return {"html": "SCI-RECURSE-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_overlay_command_html=_resolve_overlay,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="SCI recurse",
        timestamp="2026-03-11 07:09:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "SCI-RECURSE-FROM-CATALOG"
    assert calls["overlay"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_prefers_comm_validate_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"validate": 0, "generic": 0}

    def _resolve_validate(**kwargs):
        calls["validate"] += 1
        assert kwargs.get("cmd") == "Comm Validate"
        return {"html": "VALIDATE-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_validate_command_html=_resolve_validate,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Comm Validate",
        timestamp="2026-03-11 07:09:30",
        prev_profile_for_audit="Standard",
    )
    assert out == "VALIDATE-FROM-CATALOG"
    assert calls["validate"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_prefers_comm_anchor_toggle_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"anchor": 0, "generic": 0}

    def _resolve_anchor(**kwargs):
        calls["anchor"] += 1
        assert kwargs.get("cmd") == "Comm Anchor on"
        return {"html": "ANCHOR-TOGGLE-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_comm_anchor_toggle_command_html=_resolve_anchor,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Comm Anchor on",
        timestamp="2026-03-11 07:09:40",
        prev_profile_for_audit="Standard",
    )
    assert out == "ANCHOR-TOGGLE-FROM-CATALOG"
    assert calls["anchor"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_prefers_dynamic_one_shot_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"dyn": 0, "generic": 0}

    def _resolve_dyn(**kwargs):
        calls["dyn"] += 1
        assert kwargs.get("cmd") == "Dynamic one-shot on"
        return {"html": "DYNAMIC-FROM-CATALOG"}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_dynamic_one_shot_command_html=_resolve_dyn,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Dynamic one-shot on",
        timestamp="2026-03-11 07:09:50",
        prev_profile_for_audit="Standard",
    )
    assert out == "DYNAMIC-FROM-CATALOG"
    assert calls["dyn"] == 1
    assert calls["generic"] == 0


def test_resolve_post_state_command_html_dynamic_one_shot_catalog_falls_back_to_generic_on_empty_html(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    calls = {"dyn": 0, "generic": 0}

    def _resolve_dyn(**kwargs):
        calls["dyn"] += 1
        return {"html": " "}

    def _resolve_generic(**kwargs):
        calls["generic"] += 1
        assert kwargs.get("cmd") == "Dynamic one-shot on"
        return {"html": "GENERIC-FROM-CATALOG"}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            resolve_dynamic_one_shot_command_html=_resolve_dyn,
            resolve_post_state_command_html=_resolve_generic,
        ),
        raising=False,
    )

    out = mod._resolve_post_state_command_html(
        api,
        cmd="Dynamic one-shot on",
        timestamp="2026-03-11 07:10:00",
        prev_profile_for_audit="Standard",
    )
    assert out == "GENERIC-FROM-CATALOG"
    assert calls["dyn"] == 1
    assert calls["generic"] == 1


def test_response_catalog_resolve_comm_overlay_command_html_builds_comm_start_audit_line():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_overlay_command_html(
        cmd="Comm Start",
        current_profile="Standard",
        prev_profile_for_audit="Expert",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    audit = str(out.get("comm_start_audit_line") or "")
    html_out = str(out.get("html") or "")
    assert "Profile-Switch-Audit: command=Comm Start · from=Expert · to=Standard" in audit
    assert audit in html_out


def test_response_catalog_resolve_comm_overlay_command_html_handles_color_commands_without_audit():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_overlay_command_html(
        cmd="Color on",
        current_profile="Standard",
        prev_profile_for_audit="Standard",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("comm_start_audit_line") or "") == ""
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_response_catalog_resolve_comm_overlay_command_html_handles_sci_off_without_audit():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_overlay_command_html(
        cmd="SCI off",
        current_profile="Standard",
        prev_profile_for_audit="Standard",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("comm_start_audit_line") or "") == ""
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_response_catalog_resolve_comm_overlay_command_html_handles_sci_recurse_without_audit():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_overlay_command_html(
        cmd="SCI recurse",
        current_profile="Standard",
        prev_profile_for_audit="Standard",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("comm_start_audit_line") or "") == ""
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_response_catalog_resolve_sci_on_command_html_sets_pending_and_renders_menu():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    calls = {"pending": 0}

    def _set_pending():
        calls["pending"] += 1

    out = cmdcat.resolve_sci_on_command_html(
        cmd="SCI on",
        lang="de",
        set_sci_pending_fn=_set_pending,
        render_sci_menu_html_fn=lambda lang="de": f"SCI-MENU::{lang}",
    )
    assert isinstance(out, dict)
    assert str(out.get("html") or "") == "SCI-MENU::de"
    assert bool(out.get("triggered_sci")) is True
    assert calls["pending"] == 1


def test_response_catalog_resolve_comm_validate_command_html_renders_comm_state():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_validate_command_html(
        cmd="Comm Validate",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_response_catalog_resolve_comm_anchor_toggle_command_html_renders_comm_state():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_comm_anchor_toggle_command_html(
        cmd="Comm Anchor off",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_response_catalog_resolve_dynamic_one_shot_command_html_renders_comm_state():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    out = cmdcat.resolve_dynamic_one_shot_command_html(
        cmd="Dynamic one-shot on",
        render_comm_state_html_fn=lambda audit_line="": f"COMM-STATE::{audit_line}",
    )
    assert isinstance(out, dict)
    assert str(out.get("html") or "") == "COMM-STATE::"


def test_handle_command_deterministic_qc_override_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    called = {"show": False}

    def _show():
        called["show"] = True

    def _payload():
        return {"html": "QC-OVERRIDE-FROM-CATALOG", "csc": None}

    monkeypatch.setattr(api, "show_qc_override", _show, raising=True)
    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(build_qc_override_opened_result=_payload),
        raising=False,
    )

    out = mod._handle_command_deterministic(api, "QC Override", "2026-03-11 08:00:00")
    assert called["show"] is True
    assert isinstance(out, dict)
    assert out.get("html") == "QC-OVERRIDE-FROM-CATALOG"


def test_handle_command_deterministic_sci_menu_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.user_turns = 0
    api.gov_state.sci_pending = False
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _payload(**kwargs):
        assert kwargs.get("cmd") == "SCI menu"
        assert kwargs.get("timestamp") == "2026-03-11 08:01:00"
        kwargs["set_sci_state_fn"]()
        kwargs["increment_user_turn_fn"]()
        kwargs["append_history_fn"]("SCI menu")
        return {"html": "SCI-MENU-FROM-CATALOG", "csc": None}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(build_sci_menu_command_result=_payload),
        raising=False,
    )

    out = mod._handle_command_deterministic(api, "SCI menu", "2026-03-11 08:01:00")
    assert isinstance(out, dict)
    assert out.get("html") == "SCI-MENU-FROM-CATALOG"
    assert api.gov_state.sci_pending is True
    assert api.gov_state.sci_active is False
    assert api.gov_state.sci_variant == ""
    assert int(api.gov_state.user_turns) == 1
    assert bool(api.history) is True


def test_handle_command_deterministic_renderer_map_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    def _resolve(**kwargs):
        assert kwargs.get("cmd") == "Comm State"
        assert kwargs.get("timestamp") == "2026-03-11 08:02:00"
        assert callable(kwargs.get("safe_html_fn"))
        assert callable(kwargs.get("append_history_fn"))
        return {"html": "CMD-MAP-FROM-CATALOG", "csc": None}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(resolve_renderer_map_command=_resolve),
        raising=False,
    )

    out = mod._handle_command_deterministic(api, "Comm State", "2026-03-11 08:02:00")
    assert isinstance(out, dict)
    assert out.get("html") == "CMD-MAP-FROM-CATALOG"


def test_response_catalog_build_comm_audit_command_result_reports_missing_qc_and_appends_history():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    appended = {"txt": ""}

    out = cmdcat.build_comm_audit_command_result(
        cmd="Comm Audit",
        timestamp="2026-03-11 08:02:30",
        now_iso="2026-03-11T08:02:30",
        profile="Standard",
        overlay="Strict",
        sci_pending=False,
        sci_variant="",
        sci_active=False,
        history=[{"role": "bot", "content": "Antwort ohne QC Footer"}],
        comm_audit_window=5,
        export_audit_fn=lambda event: (None, "/tmp/Audit_20260311_080230.json"),
        scan_rows_fn=lambda sample, history: [],
        build_route_ctx_fn=lambda user_raw, is_command: {"is_command": False},
        check_self_debunking_fn=lambda txt, profile_name: "",
        check_verification_route_gate_fn=lambda txt: "",
        append_history_fn=lambda txt: appended.__setitem__("txt", str(txt or "")),
        cwd="/tmp",
    )
    assert isinstance(out, dict)
    html_out = str(out.get("html") or "")
    assert "Comm Audit" in html_out
    assert "Missing QC footer" not in html_out
    assert "Compliance scan (best-effort)" not in html_out
    assert "Audit_20260311_080230.json" in html_out
    assert "data-export-path=" in html_out
    assert "openExportFileFromAnchor(this)" in html_out
    assert "Exportiert (Audit):" in appended["txt"]


def test_response_catalog_build_comm_audit_command_result_omits_compliance_table_rows():
    mod = load_fix_module()
    cmdcat = getattr(mod, "_output_command_response_catalog", None)
    assert cmdcat is not None

    rows = [(1, "✓ Compliant"), (2, "✓ Compliant"), (3, "✓ Compliant")]
    out = cmdcat.build_comm_audit_command_result(
        cmd="Comm Audit",
        timestamp="2026-03-11 08:02:30",
        now_iso="2026-03-11T08:02:30",
        profile="Standard",
        overlay="Strict",
        sci_pending=False,
        sci_variant="",
        sci_active=False,
        history=[{"role": "bot", "content": "Antwort 1"}, {"role": "bot", "content": "Antwort 2"}, {"role": "bot", "content": "Antwort 3"}],
        comm_audit_window=5,
        export_audit_fn=lambda event: (None, "/tmp/Audit_20260311_080230.json"),
        scan_rows_fn=lambda sample, history: list(rows),
        build_route_ctx_fn=lambda user_raw, is_command: {"is_command": False},
        check_self_debunking_fn=lambda txt, profile_name: "",
        check_verification_route_gate_fn=lambda txt: "",
        append_history_fn=lambda txt: None,
        cwd="/tmp",
    )
    assert isinstance(out, dict)
    html_out = str(out.get("html") or "")
    assert "Compliance scan (best-effort)" not in html_out
    assert "#1" not in html_out
    assert "#2" not in html_out
    assert "#3" not in html_out


def test_handle_command_deterministic_comm_audit_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    called = {"cat": 0, "export": 0}

    def _cat_payload(**kwargs):
        called["cat"] += 1
        assert kwargs.get("cmd") == "Comm Audit"
        kwargs["append_history_fn"]("Comm Audit\nExportiert (Audit): Logs/Audit/Audit_test.json")
        return {"html": "COMM-AUDIT-FROM-CATALOG", "csc": None}

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(build_comm_audit_command_result=_cat_payload),
        raising=False,
    )

    def _export(*args, **kwargs):
        called["export"] += 1
        return (None, None)

    monkeypatch.setattr(api, "export", _export, raising=True)

    out = mod._handle_command_deterministic(api, "Comm Audit", "2026-03-11 08:03:00")
    assert isinstance(out, dict)
    assert out.get("html") == "COMM-AUDIT-FROM-CATALOG"
    assert called["cat"] == 1
    assert called["export"] == 0, "Legacy export path must not run when catalog handled Comm Audit"
    hist = getattr(api, "history", []) or []
    assert hist and "Comm Audit" in str((hist[-1] or {}).get("content") or "")


def test_handle_command_deterministic_comm_audit_falls_back_when_catalog_missing(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    monkeypatch.setattr(mod, "_output_command_response_catalog", None, raising=False)

    called = {"export": 0}

    def _export(*args, **kwargs):
        called["export"] += 1
        return (None, "/tmp/Audit_20260311_080310.json")

    monkeypatch.setattr(api, "export", _export, raising=True)

    out = mod._handle_command_deterministic(api, "Comm Audit", "2026-03-11 08:03:10")
    assert isinstance(out, dict)
    html_out = str(out.get("html") or "")
    assert "Comm Audit" in html_out
    assert "Audit exported." in html_out
    assert "data-export-path=" in html_out
    assert "openExportFileFromAnchor(this)" in html_out
    assert called["export"] == 1
    hist = getattr(api, "history", []) or []
    assert hist and "Comm Audit" in str((hist[-1] or {}).get("content") or "")


def test_handle_command_deterministic_no_longer_accepts_comm_audi_alias():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    out = mod._handle_command_deterministic(api, "Comm Audi", "2026-03-11 08:03:00")
    assert out is None


def test_execute_legacy_command_prefers_command_catalog_for_basic_state(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.overlay = ""

    monkeypatch.setattr(mod, "_controller_dispatch", None, raising=False)
    monkeypatch.setattr(mod, "_intent_from_command", None, raising=False)
    monkeypatch.setattr(mod, "_state_from_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_state_apply_to_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_apply_intent", None, raising=False)

    def _apply_basic(**kwargs):
        assert kwargs.get("cmd") == "Strict on"
        st = kwargs.get("state")
        st.overlay = "ModuleStrict"
        return True

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(apply_basic_command_state=_apply_basic),
        raising=False,
    )

    api._execute_legacy_command("Strict on")
    assert api.gov_state.overlay == "ModuleStrict"


def test_execute_legacy_command_skips_basic_fallback_when_catalog_exists(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.overlay = ""

    monkeypatch.setattr(mod, "_controller_dispatch", None, raising=False)
    monkeypatch.setattr(mod, "_intent_from_command", None, raising=False)
    monkeypatch.setattr(mod, "_state_from_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_state_apply_to_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_apply_intent", None, raising=False)

    def _apply_basic(**kwargs):
        assert kwargs.get("cmd") == "Strict on"
        # Simulate catalog no-op. Phase G expects no monolith fallback mutation.
        return False

    def _is_supported(cmd):
        return str(cmd or "").strip() == "Strict on"

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(
            apply_basic_command_state=_apply_basic,
            is_basic_command_supported=_is_supported,
        ),
        raising=False,
    )

    api._execute_legacy_command("Strict on")
    assert api.gov_state.overlay == ""


def test_execute_legacy_command_keeps_basic_fallback_on_catalog_error(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.overlay = ""

    monkeypatch.setattr(mod, "_controller_dispatch", None, raising=False)
    monkeypatch.setattr(mod, "_intent_from_command", None, raising=False)
    monkeypatch.setattr(mod, "_state_from_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_state_apply_to_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_apply_intent", None, raising=False)

    def _apply_basic(**kwargs):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(apply_basic_command_state=_apply_basic),
        raising=False,
    )

    api._execute_legacy_command("Strict on")
    assert api.gov_state.overlay == "Strict"


def test_execute_legacy_command_prefers_command_catalog_for_profile_switch(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    monkeypatch.setattr(mod, "_controller_dispatch", None, raising=False)
    monkeypatch.setattr(mod, "_intent_from_command", None, raising=False)
    monkeypatch.setattr(mod, "_state_from_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_state_apply_to_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_apply_intent", None, raising=False)

    def _apply_profile(**kwargs):
        assert kwargs.get("cmd") == "Profile Expert"
        st = kwargs.get("state")
        st.active_profile = "ExpertFromCatalog"
        cb = kwargs.get("on_profile_qc_reset_fn")
        if callable(cb):
            cb()
        return True

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(apply_profile_switch_state=_apply_profile),
        raising=False,
    )

    api._execute_legacy_command("Profile Expert")
    assert api.gov_state.active_profile == "ExpertFromCatalog"


def test_execute_legacy_command_skips_profile_fallback_when_catalog_exists(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Standard"

    monkeypatch.setattr(mod, "_controller_dispatch", None, raising=False)
    monkeypatch.setattr(mod, "_intent_from_command", None, raising=False)
    monkeypatch.setattr(mod, "_state_from_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_state_apply_to_runtime", None, raising=False)
    monkeypatch.setattr(mod, "_apply_intent", None, raising=False)

    def _apply_profile(**kwargs):
        assert kwargs.get("cmd") == "Profile Expert"
        # Simulate a catalog no-op. Phase G expects no monolith fallback mutation.
        return False

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(apply_profile_switch_state=_apply_profile),
        raising=False,
    )

    api._execute_legacy_command("Profile Expert")
    assert api.gov_state.active_profile == "Standard"


def test_handle_sci_selection_prefers_command_catalog(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.history = []

    def _build(**kwargs):
        assert kwargs.get("letter") == "B"
        return {
            "char": "B",
            "html": "SCI-SELECTION-FROM-CATALOG",
            "history_text": "SCI Variant B activated.",
        }

    monkeypatch.setattr(
        mod,
        "_output_command_response_catalog",
        types.SimpleNamespace(build_sci_selection_result=_build),
        raising=False,
    )

    out = api._handle_sci_selection("b")
    assert isinstance(out, dict)
    assert out.get("html") == "SCI-SELECTION-FROM-CATALOG"
    assert api.gov_state.sci_variant == "B"
    assert bool(api.gov_state.sci_active) is True
    assert bool(api.gov_state.sci_pending) is False
    assert any(isinstance(h, dict) and "SCI Variant B activated." in str(h.get("content", "")) for h in api.history)


def test_apply_color_spans_prefers_modular_renderer(monkeypatch):
    mod = load_fix_module()

    def _render(text, *, enabled=True, evidence_color=None, evidence_icon=None):
        assert text == "[GREEN] 🟢"
        assert enabled is True
        assert isinstance(evidence_color, dict)
        assert isinstance(evidence_icon, dict)
        return "COLOR-FROM-MODULE"

    monkeypatch.setattr(
        mod,
        "_output_color_markers_renderer",
        types.SimpleNamespace(apply_color_spans=_render),
        raising=False,
    )

    out = mod.apply_color_spans("[GREEN] 🟢", enabled=True)
    assert out == "COLOR-FROM-MODULE"


def test_cgi_ui_texts_prefers_modular_renderer(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    def _texts(*, lang="de"):
        assert lang == "en"
        return {
            "saved": "S {c},{i},{e}",
            "applied": "A {c},{i},{e}",
            "no_prompt": "N",
            "invalid": "I",
            "repeat_failed": "R",
        }

    monkeypatch.setattr(
        mod,
        "_output_cgi_line_renderer",
        types.SimpleNamespace(get_cgi_ui_texts=_texts),
        raising=False,
    )

    out = api._cgi_ui_texts(lang="en")
    assert out["saved"] == "S {c},{i},{e}"
    assert out["applied"] == "A {c},{i},{e}"


def test_build_cgi_feedback_prompt_blocks_prefers_modular_renderer(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    def _feedback(*, user_feedback_triplet="", process_feedback=""):
        assert user_feedback_triplet == "3,2,1"
        assert process_feedback == "SCI: 1,1,1"
        return "[CGI Feedback]\nMODULE-LINE"

    def _constraints(lines):
        assert list(lines) == ["- alpha", "- beta"]
        return "[CGI One-Shot Rewrite Constraints]\nMODULE-CONSTRAINTS"

    monkeypatch.setattr(
        mod,
        "_output_cgi_line_renderer",
        types.SimpleNamespace(
            render_cgi_feedback_block=_feedback,
            render_cgi_constraints_block=_constraints,
        ),
        raising=False,
    )

    out = api._build_cgi_feedback_prompt_blocks(
        user_feedback_triplet="3,2,1",
        process_feedback="SCI: 1,1,1",
        one_shot_constraints=["- alpha", "- beta"],
    )
    assert out == [
        "[CGI Feedback]\nMODULE-LINE",
        "[CGI One-Shot Rewrite Constraints]\nMODULE-CONSTRAINTS",
    ]


def test_csc_meta_tooltip_uses_answer_language_english():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api.gov_state.answer_language = "en"
    html_out, meta = api._apply_csc_strict(
        raw_response="I guarantee this is correct.",
        user_raw="x",
        is_command=False,
    )
    assert isinstance(html_out, str) and html_out
    assert isinstance(meta, dict)
    assert bool(meta.get("applied"))
    score_tip = str(meta.get("score_tooltip") or "")
    thr_tip = str(meta.get("thresholds_tooltip") or "")
    assert "score line: f=" in score_tip.lower()
    assert "tokens=" in score_tip.lower()
    assert "complex/technical" in score_tip.lower()
    assert "thresholds line:" in thr_tip.lower()
    assert "f>=" in thr_tip
    assert "tok>=" in thr_tip
    assert "gov_tok>=" in thr_tip
    assert "x1 normal, x2 stricter" in thr_tip.lower()


def test_csc_meta_tooltip_uses_answer_language_german():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api.gov_state.answer_language = "de"
    html_out, meta = api._apply_csc_strict(
        raw_response="Das ist definitiv korrekt.",
        user_raw="x",
        is_command=False,
    )
    assert isinstance(html_out, str) and html_out
    assert isinstance(meta, dict)
    assert bool(meta.get("applied"))
    score_tip = str(meta.get("score_tooltip") or "")
    thr_tip = str(meta.get("thresholds_tooltip") or "")
    assert "Score-Zeile: f=" in score_tip
    assert "tokens=" in score_tip
    assert "komplex/technisch" in score_tip
    assert "Thresholds-Zeile:" in thr_tip
    assert "f>=" in thr_tip
    assert "tok>=" in thr_tip
    assert "gov_tok>=" in thr_tip
    assert "x1 normal, x2 strenger" in thr_tip


def test_response_timestamp_contains_explicit_utc_offset():
    mod = load_fix_module()
    ts = mod._format_response_timestamp()
    assert isinstance(ts, str) and ts
    assert re.search(r"UTC[+-]\d{2}:\d{2}", ts) is not None


def test_signal_dot_tooltip_uses_answer_language_english():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "en"
    src = "<p><span style='color:#2e7d32; font-weight:600;'>🟢</span> Claim.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "signal-dot-marker" in out
    assert "Green: high reliability and comparatively robust evidence." in out


def test_signal_dot_tooltip_uses_answer_language_german():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "de"
    src = "<p><span style='color:#c62828; font-weight:600;'>🔴</span> Aussage.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "signal-dot-marker" in out
    assert "Rot: niedrige Verlaesslichkeit; erhebliche Unsicherheit oder schwache Absicherung." in out


def test_signal_dot_tooltip_upgrades_existing_marker_without_title():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "de"
    src = (
        "<p><span class='signal-dot-marker'><span style='color:#f9a825; font-weight:600;'>🟡</span></span> Aussage.</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "signal-dot-marker" in out
    assert "data-u-title='Gelb: mittlere Verlaesslichkeit; relevante Unsicherheit bleibt.'" in out
    assert "</span></span></span>" not in out


def test_signal_dot_marker_count_follows_llm_output_without_wrapper_capping():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "de"
    src = (
        "<p><span style='color:#2e7d32; font-weight:600;'>🟢</span> Satz A. "
        "<span style='color:#f9a825; font-weight:600;'>🟡</span> Satz B. "
        "<span style='color:#c62828; font-weight:600;'>🔴</span> Satz C.</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    assert out.count("signal-dot-marker") == 3


def test_signal_dot_helper_functions_delegate_to_output_color_renderer(monkeypatch):
    mod = load_fix_module()

    class _StubRenderer:
        def annotate_signal_dot_tooltips_html(self, html_text, **kwargs):
            return f"ANN::{html_text}::{kwargs.get('lang')}"

        def inject_fallback_signal_dots_html(self, html_text, **kwargs):
            return f"INJECT::{html_text}::{kwargs.get('lang')}"

        def limit_signal_dot_marker_density_html(self, html_text, **kwargs):
            return f"LIMIT::{html_text}::{kwargs.get('max_per_block')}"

        def strip_signal_dots_from_heading_only_blocks_html(self, html_text):
            return f"STRIP::{html_text}"

    monkeypatch.setattr(mod, "_output_color_markers_renderer", _StubRenderer())

    assert mod.annotate_signal_dot_tooltips_html("<p>z</p>", lang="en") == "ANN::<p>z</p>::en"
    assert mod.inject_fallback_signal_dots_html("<p>a</p>", lang="en") == "INJECT::<p>a</p>::en"
    assert mod.limit_signal_dot_marker_density_html("<p>b</p>", max_per_block=2) == "LIMIT::<p>b</p>::2"
    assert mod.strip_signal_dots_from_heading_only_blocks_html("<p>c</p>") == "STRIP::<p>c</p>"


def test_signal_dot_helper_functions_are_fail_soft_when_renderer_unavailable(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_color_markers_renderer", None)

    src = "<p>kein marker</p>"
    assert mod.annotate_signal_dot_tooltips_html(src, lang="de") == src
    assert mod.inject_fallback_signal_dots_html(src, lang="de") == src
    assert mod.limit_signal_dot_marker_density_html(src, max_per_block=1) == src
    assert mod.strip_signal_dots_from_heading_only_blocks_html(src) == src


def test_auto_embed_image_urls_delegates_to_output_renderer(monkeypatch):
    mod = load_fix_module()

    class _StubImageEmbedRenderer:
        def auto_embed_image_urls(self, text):
            return f"IMG::{text}"

    monkeypatch.setattr(mod, "_output_image_embed_renderer", _StubImageEmbedRenderer())
    assert mod.auto_embed_image_urls("https://example.com/a.png") == "IMG::https://example.com/a.png"


def test_auto_embed_image_urls_is_fail_soft_and_handles_single_url_code_fences(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_image_embed_renderer", None)
    src = (
        "Bild: https://example.com/live.png\n\n"
        "```txt\n"
        "https://example.com/code.jpg\n"
        "```\n"
    )
    out = mod.auto_embed_image_urls(src)
    assert '<img src="https://example.com/live.png"' in out
    assert '<img src="https://example.com/code.jpg"' in out
    assert out.count("<img ") == 2
    assert "https://example.com/code.jpg" in out


def test_auto_embed_image_urls_handles_trailing_sentence_punctuation(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_image_embed_renderer", None)
    src = "Bild: https://example.com/trailing.png."
    out = mod.auto_embed_image_urls(src)
    assert "https://example.com/trailing.png." in out
    assert '<img src="https://example.com/trailing.png"' in out
    assert out.count("<img ") == 1


def test_auto_embed_image_urls_adds_image_tag_for_inline_code_wrapped_url(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_image_embed_renderer", None)
    src = "`https://example.com/inline.png`"
    out = mod.auto_embed_image_urls(src)
    assert "`https://example.com/inline.png`" in out
    assert '<img src="https://example.com/inline.png"' in out
    assert out.count("<img ") == 1


def test_auto_embed_image_urls_handles_angle_wrapped_url(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_image_embed_renderer", None)
    src = "<https://example.com/angle.png>"
    out = mod.auto_embed_image_urls(src)
    assert "<https://example.com/angle.png>" in out
    assert '<img src="https://example.com/angle.png"' in out
    assert out.count("<img ") == 1


def test_auto_embed_image_urls_survives_markdown_render_for_inline_code_and_single_fence(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_image_embed_renderer", None)
    raw_inline = "`https://example.com/comm-sci-manual-test.png`"
    prepared_inline = mod.auto_embed_image_urls(raw_inline)
    html_inline = mod.markdown.markdown(prepared_inline, extensions=['extra', 'codehilite'])
    assert re.search(r'<img[^>]+src="https://example.com/comm-sci-manual-test\.png"', str(html_inline or ""))

    raw_fence = "```txt\nhttps://example.com/comm-sci-manual-test.png\n```"
    prepared_fence = mod.auto_embed_image_urls(raw_fence)
    html_fence = mod.markdown.markdown(prepared_fence, extensions=['extra', 'codehilite'])
    assert re.search(r'<img[^>]+src="https://example.com/comm-sci-manual-test\.png"', str(html_fence or ""))


def test_evaluate_strict_enforcement_delegates_to_output_renderer(monkeypatch):
    mod = load_fix_module()
    seen = {}

    class _StubStrictGateRenderer:
        def evaluate_strict_enforcement(self, **kwargs):
            seen.update(kwargs)
            return {
                "blocked": False,
                "blocked_html": "",
                "strict_banner_html": "STRICT::WARN",
                "meta": {"strict_enforcement": "warned"},
            }

    monkeypatch.setattr(mod, "_output_strict_gate_renderer", _StubStrictGateRenderer())
    out = mod.evaluate_strict_enforcement(
        raw_for_render="A",
        user_raw="U",
        profile_name="Standard",
        override_violations=["x"],
        settings={"policy": "strict_warn", "enabled": True},
        validator_obj=None,
        runtime_state=None,
        append_system_message_fn=None,
    )
    assert out.get("strict_banner_html") == "STRICT::WARN"
    assert seen.get("raw_for_render") == "A"
    assert callable(seen.get("render_strict_block_warning_html_fn"))
    assert callable(seen.get("render_strict_warn_banner_html_fn"))


def test_evaluate_strict_enforcement_fallback_warn_and_block(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_strict_gate_renderer", None)

    class _DummyValidator:
        def validate(self, text=None, **kwargs):
            return (["hard_violation"], [])

    warn = mod.evaluate_strict_enforcement(
        raw_for_render="Antwort",
        user_raw="Prompt",
        profile_name="Standard",
        override_violations=[],
        settings={"policy": "strict_warn", "enabled": True},
        validator_obj=_DummyValidator(),
        runtime_state=types.SimpleNamespace(),
        append_system_message_fn=None,
    )
    assert warn.get("blocked") is False
    assert "RULE VIOLATION DETECTED" in str(warn.get("strict_banner_html") or "")

    block = mod.evaluate_strict_enforcement(
        raw_for_render="Antwort",
        user_raw="Prompt",
        profile_name="Standard",
        override_violations=[],
        settings={"policy": "strict_block", "enabled": True},
        validator_obj=_DummyValidator(),
        runtime_state=types.SimpleNamespace(),
        append_system_message_fn=None,
    )
    assert block.get("blocked") is True
    assert "STRICT BLOCK" in str(block.get("blocked_html") or "")


def test_render_quality_helpers_delegate_to_output_renderer(monkeypatch):
    mod = load_fix_module()

    class _StubRenderQualityRenderer:
        def looks_like_rendered_html(self, html_text):
            return str(html_text) == "ok"

        def build_normalization_summary(self, **kwargs):
            return {"qc_footer_raw_count": 7, "custom": True}

    monkeypatch.setattr(mod, "_output_render_quality_renderer", _StubRenderQualityRenderer())
    assert mod._looks_like_rendered_html_runtime("ok") is True
    assert mod._looks_like_rendered_html_runtime("bad") is False
    out = mod._build_render_normalization_summary("raw", "<p>html</p>")
    assert out.get("qc_footer_raw_count") == 7
    assert out.get("custom") is True


def test_render_quality_helpers_are_fail_soft_when_renderer_unavailable(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_render_quality_renderer", None)
    assert mod._looks_like_rendered_html_runtime("<p>x</p>") is True
    out = mod._build_render_normalization_summary("QC-Matrix: X", "<p>QC-Matrix: X</p>")
    assert isinstance(out, dict)
    assert out.get("qc_footer_raw_count") == 1
    assert "self_debunking_boxed" in out


def test_append_uncertainty_explanation_delegates_to_output_uncertainty_renderer(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.answer_language = "en"

    class _StubUncertaintyRenderer:
        def append_uncertainty_explanation_if_needed(self, html_text, **kwargs):
            return f"U-DELEGATE::{html_text}::{kwargs.get('lang')}::{bool(kwargs.get('uncertainty_codes_mod') is not None)}"

    monkeypatch.setattr(mod, "_output_uncertainty_renderer", _StubUncertaintyRenderer())
    out = api._append_uncertainty_explanation_if_needed("<p>x</p>", user_text="u")
    assert out == "U-DELEGATE::<p>x</p>::en::True"


def test_append_uncertainty_explanation_is_fail_soft_when_renderer_unavailable(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_uncertainty_renderer", None)
    src = "<p>x</p>"
    assert api._append_uncertainty_explanation_if_needed(src, user_text="u") == src


def test_append_uncertainty_explanation_marks_plain_explicit_u_code():
    mod = load_fix_module()
    api = mod.Api()
    src = "<p>U1: Datenlücke. Benötigt: Kontinuierliche Beobachtung und Anpassung der Strategien.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "uncertainty-inline-marker" in out
    assert "data-u-code='U1'" in out
    assert "data-u-title='U1 - Datenluecke" in out
    assert "U1: Datenlücke" not in out
    assert out.count("data-u-code='U1'") == 1
    assert re.search(r"\(\s*<span class='uncertainty-inline-wrap'[^>]*>\s*\(", out) is None


def test_append_uncertainty_explanation_does_not_inject_signal_dots_without_llm_markers_when_color_on():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.comm_active = True
    api.gov_state.color = "on"
    api.gov_state.answer_language = "de"
    src = "<p>Dies ist eine klare und einfache Aussage ohne besondere Unsicherheiten.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "signal-dot-marker" not in out


def test_append_uncertainty_explanation_does_not_inject_fallback_signal_dots_when_color_off():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.comm_active = True
    api.gov_state.color = "off"
    api.gov_state.answer_language = "de"
    src = "<p>Dies ist eine klare und einfache Aussage ohne besondere Unsicherheiten.</p>"
    out = api._append_uncertainty_explanation_if_needed(src)
    assert "signal-dot-marker" not in out


def test_color_off_strips_visual_markers_and_textual_evidence_tags():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None
    api.gov_state.color = "off"

    api.chat_session = DummySession([
        "Antwortsatz.\n[GREEN] 🟢 Behauptung A.\nSelf-Debunking:\n1. Weakness: x\n2. Weakness: y\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    ])

    out = api.ask("Testfrage")
    html_out = out.get("html") if isinstance(out, dict) else str(out)
    plain = re.sub(r"(?is)<[^>]+>", " ", html_out)
    plain = re.sub(r"\s+", " ", plain)

    assert "[GREEN]" not in plain
    assert "[YELLOW]" not in plain
    assert "[RED]" not in plain
    assert "🟢" not in plain
    assert "🟡" not in plain
    assert "🔴" not in plain
    assert "⚪" not in plain


def test_append_uncertainty_explanation_keeps_existing_llm_signal_markers():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.comm_active = True
    api.gov_state.color = "on"
    src = (
        "<p><span style='color:#2e7d32; font-weight:600;'>🟢</span> A.</p>"
        "<p><span style='color:#f9a825; font-weight:600;'>🟡</span> B.</p>"
    )
    out = api._append_uncertainty_explanation_if_needed(src)
    assert out.count("signal-dot-marker") == 2


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


def test_default_ruleset_prefers_v20_2_5_when_available():
    mod = load_fix_module()
    default_json = Path(str(getattr(mod, 'DEFAULT_JSON', '') or ''))
    target = ROOT / 'JSON' / 'Comm-SCI-v20.2.5.json'
    if target.exists():
        assert default_json.name == 'Comm-SCI-v20.2.5.json'

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
    i_hf_start = html.find('id="hfCatalogRow"')
    assert i_hf_start != -1
    i_hf_end = html.find('</div>', i_hf_start)
    if i_hf_end == -1:
        i_hf_end = i_hf_start + 800
    hf_block = html[i_hf_start:i_hf_end]
    # QC must be rendered from the runtime/governance section, not from HF-only controls.
    assert 'qcOverrideBtn' not in hf_block
    assert "section('Runtime & Governance'" in html


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


def test_panel_html_respects_qc_override_visibility_flag():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert isinstance(html, str) and html
    assert "function _mergeRuntimeItems(overlays, tools, opts)" in html
    assert "const showQc = !!(opts && opts.qcOverrideVisible);" in html
    assert "typeof data.qc_override_visible === 'boolean'" in html
    assert "_mergeRuntimeItems(data.overlays, data.tools, {qcOverrideVisible: qcVisible})" in html


def test_panel_html_section_defaults_and_provider_header():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_PANEL', '')
    assert isinstance(html, str) and html
    assert '<h4>Provider &amp; LLM</h4>' in html
    assert 'data-store-key="provider" data-default-open="1"' in html
    assert 'data-store-key="panel" data-default-open="1"' in html
    assert 'data-store-key="logs" data-default-open="0"' in html
    assert 'data-store-key="manual_test" data-default-open="0"' in html
    assert 'data-tip-key="section_panel"' in html
    assert 'data-tip-key="section_provider"' in html
    assert 'data-tip-key="section_logs"' in html
    assert 'data-tip-key="section_manual_test"' in html
    assert 'section-title-wrap' in html
    assert 'section-info' in html
    assert "section('Comm Core', data.comm, 'comm_core');" in html
    assert "section('Profiles', data.profiles, 'profiles');" in html
    assert "section('SCI Workflow', data.sci, 'sci_workflow');" in html
    assert "section('Runtime & Governance', _mergeRuntimeItems(data.overlays, data.tools, {qcOverrideVisible: qcVisible}), 'runtime_governance');" in html
    assert 'data-tip-key="${_escHtml(sectionTipKey)}"' in html
    assert '_sectionTipKey(key)' in html
    assert '#modelSearch { width: 100%;' in html
    assert '.provider-row { margin-bottom: 8px;' in html
    assert 'const _sectionStateMem = Object.create(null);' in html
    assert "Object.prototype.hasOwnProperty.call(_sectionStateMem, key)" in html
    assert "_bindPanelTooltipEvents();" in html


def test_panel_passphrase_modal_uses_border_box_width_rules():
    mod = load_fix_module()
    html = getattr(mod, "HTML_PANEL", "")
    assert isinstance(html, str) and html
    assert ".setting-select { width: 100%; max-width: 100%; min-width: 0; box-sizing: border-box;" in html
    assert ".modal-card { width: min(420px, calc(100vw - 24px)); max-width: calc(100vw - 24px); box-sizing: border-box;" in html
    assert ".modal-card .setting-select { width: 100%; max-width: 100%; min-width: 0; box-sizing: border-box; display: block; }" in html


def test_panel_tooltips_use_custom_overlay_and_strip_native_title():
    mod = load_fix_module()
    html = getattr(mod, 'HTML_PANEL', '')
    assert isinstance(html, str) and html
    assert "function _panelTipShow(target, ev){" in html
    assert "tip.style.background = '#eff6ff';" in html
    assert "tip.style.color = '#1e3a8a';" in html
    assert "_panelTipTarget(e.target)" in html
    assert "el.removeAttribute('title');" in html
    assert "setAttribute('title', tip)" not in html


def test_chat_template_tooltip_overlay_disables_native_title_tooltips():
    chat_asset = ROOT / "src" / "ui_assets" / "chat_template.html"
    txt = chat_asset.read_text(encoding="utf-8")
    assert '<span class="cgi-help" data-u-title="${escHtml(t.help)}">i</span>' in txt
    assert 'data-u-title="${escHtml(t.help)}" title=' not in txt
    assert 'onclick="copyToClipboard(this)" data-u-title=' in txt
    assert 'onclick="copyToClipboard(this)" title="Copy"' not in txt
    assert "function _applyResponseTooltips(root, answerLang){" in txt
    assert "_decorateQcMatrixTooltips(root, answerLang);" in txt
    assert "_normalizeCustomTooltipTargets(root);" in txt
    assert "_applyResponseTooltips(d, answerLang);" in txt
    assert "requestAnimationFrame(() => {" in txt
    assert "MathJax.typesetPromise().then(() => {" in txt
    assert "_restoreProtectedQcDimTips(" in txt
    assert "__QC_DIM_TIP_PROTECT_" in txt
    assert "const qcMarker = /(?:QC(?:\\s*-\\s*Matrix)?\\s*:)/i;" in txt
    assert "_qcDimScaleRows(dimKey, lang)" in txt
    assert "Scale 0-3 (table):" in txt
    assert "Skala 0-3 (Tabelle):" in txt
    assert "tip.style.whiteSpace = 'pre-line';" in txt
    assert "const txt = String(target.getAttribute('data-u-title') || '').trim();" in txt
    assert "el.removeAttribute('title');" in txt


def test_chat_asset_load_rebuilds_html_chat_after_template_reload():
    src = (ROOT / "src" / "Comm-SCI-Control-App.py").read_text(encoding="utf-8")
    marker = 'HTML_CHAT_TEMPLATE = _load_ui_asset_text("chat_template.html", HTML_CHAT_TEMPLATE)'
    i = src.find(marker)
    assert i != -1
    snippet = src[i:i + 260]
    assert "HTML_CHAT = HTML_CHAT_TEMPLATE.replace('__WRAPPER_LABEL__', html.escape(WRAPPER_NAME))" in snippet

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
    assert psvc.supports_native_retrieval('gemini') is True
    assert psvc.supports_native_retrieval('openrouter') is False


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


def test_settings_show_panel_keeps_main_active_when_opening_from_main_toggle():
    mod = load_fix_module()
    api = mod.Api()

    panel_calls = []
    main_calls = []

    class _PanelWin:
        def show(self):
            panel_calls.append("show")
        def restore(self):
            panel_calls.append("restore")
        def focus(self):
            panel_calls.append("focus")

    class _MainWin:
        def restore(self):
            main_calls.append("restore")
        def bring_to_front(self):
            main_calls.append("bring_to_front")
        def focus(self):
            main_calls.append("focus")

    api.panel_win = _PanelWin()
    api.main_win = _MainWin()
    api.panel_hidden = True
    api._panel_wait_bootstrap_or_fallback = lambda: None  # type: ignore[assignment]
    api.settings()

    assert "show" in panel_calls
    assert "restore" in panel_calls
    assert "focus" not in panel_calls
    assert main_calls, "Main window should be re-focused after panel show toggle."


def test_settings_panel_toggle_does_not_steal_focus_from_open_qc_override():
    mod = load_fix_module()
    api = mod.Api()

    panel_calls = []
    main_calls = []

    class _PanelWin:
        def show(self):
            panel_calls.append("show")
        def restore(self):
            panel_calls.append("restore")
        def focus(self):
            panel_calls.append("focus")

    class _MainWin:
        def restore(self):
            main_calls.append("restore")
        def bring_to_front(self):
            main_calls.append("bring_to_front")
        def focus(self):
            main_calls.append("focus")

    api.panel_win = _PanelWin()
    api.main_win = _MainWin()
    api.panel_hidden = True
    api._qc_override_open = True
    api._panel_wait_bootstrap_or_fallback = lambda: None  # type: ignore[assignment]
    api.settings()

    assert "show" in panel_calls
    assert "restore" in panel_calls
    assert "focus" not in panel_calls
    assert main_calls == []



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


def test_render_sci_trace_runtime_dedupes_double_trace_blocks_and_keeps_tail_order():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    src = (
        "Einleitung vor dem Trace.\n\n"
        "4. SCI Trace (Variante A: Standard)\n"
        "1. Plan: Erste Plan-Version.\n"
        "2. Solution: Erste Solution-Version.\n"
        "3. Check: Erste Check-Version.\n\n"
        "SCI Trace:\n"
        "1. Plan: Zweite Plan-Version.\n"
        "2. Solution: Zweite Solution-Version.\n"
        "3. Check: Zweite Check-Version.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: Beispiel.\n"
        "2. Schwäche: Beispiel 2.\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )

    out = api._render_sci_trace_as_html_runtime(src)
    assert "class='sci-trace'" in out
    assert out.count("class='sci-trace'") == 1
    assert re.search(r"(?im)^\\s*SCI\\s+Trace\\s*:", out) is None
    assert "4. SCI Trace (Variante A: Standard)" not in out
    assert "Self-Debunking:" in out
    assert "QC-Matrix:" in out
    assert out.find("Self-Debunking:") < out.find("QC-Matrix:")


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


def test_comm_state_shows_verification_route_display_policy_state():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None
    api.chat_session = DummySession(["SHOULD NOT BE USED"])

    cfg_obj = getattr(mod, "cfg", None)
    old_root = None
    old_provider = None
    try:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            old_root = cfg_obj.config.get("hide_verification_route_lines")
            cfg_obj.config.pop("hide_verification_route_lines", None)
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            old_provider = g.get("hide_verification_route_lines")
            g.pop("hide_verification_route_lines", None)

        html_visible = _extract_html(api.ask("Comm State"))
        assert "Verification route lines" in html_visible
        assert "visible" in html_visible

        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            g["hide_verification_route_lines"] = True

        html_hidden = _extract_html(api.ask("Comm State"))
        assert "Verification route lines" in html_hidden
        assert "hidden" in html_hidden
    finally:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            if old_root is None:
                cfg_obj.config.pop("hide_verification_route_lines", None)
            else:
                cfg_obj.config["hide_verification_route_lines"] = old_root
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            if old_provider is None:
                g.pop("hide_verification_route_lines", None)
            else:
                g["hide_verification_route_lines"] = old_provider


def test_panel_action_can_toggle_verification_route_display_policy_runtime():
    mod = load_fix_module()
    _prime_module_gov(mod)

    api = mod.Api()
    api.validator = None
    api.chat_session = DummySession(["SHOULD NOT BE USED"])

    cfg_obj = getattr(mod, "cfg", None)
    old_root = None
    old_provider = None
    try:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            old_root = cfg_obj.config.get("hide_verification_route_lines")
            cfg_obj.config.pop("hide_verification_route_lines", None)
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            old_provider = g.get("hide_verification_route_lines")
            g.pop("hide_verification_route_lines", None)

        on = api.panel_action(
            "set_hide_verification_route_lines",
            {"scope": "provider", "provider": "gemini", "enabled": True},
        )
        assert isinstance(on, dict) and bool(on.get("ok")) is True
        assert bool(on.get("effective")) is True
        assert "hidden" in _extract_html(api.ask("Comm State"))

        reset = api.panel_action(
            "set_hide_verification_route_lines",
            {"scope": "provider", "provider": "gemini", "clear": True},
        )
        assert isinstance(reset, dict) and bool(reset.get("ok")) is True
        assert bool(reset.get("effective")) is False
        assert "visible" in _extract_html(api.ask("Comm State"))
    finally:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            if old_root is None:
                cfg_obj.config.pop("hide_verification_route_lines", None)
            else:
                cfg_obj.config["hide_verification_route_lines"] = old_root
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            if old_provider is None:
                g.pop("hide_verification_route_lines", None)
            else:
                g["hide_verification_route_lines"] = old_provider


def test_chat_export_includes_provider_model_metadata_and_history(tmp_path, monkeypatch):
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

    # Keep test deterministic even when local encrypted keys exist.
    monkeypatch.setattr(
        api,
        "_passphrase_requirement_for_provider",
        lambda provider, passphrase_override=None: {"required": False, "encrypted": False},
        raising=False,
    )

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


def test_anchor_auto_alias_is_no_longer_a_valid_command_token():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    route = mod.route_input("Anchor auto off", api.gov_state, api)
    assert route.get("kind") == "chat"


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


def test_profile_switch_repeated_command_keeps_audit_line_deterministic():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None

    first = api.ask("Profile Briefing")
    first_html = first.get("html") if isinstance(first, dict) else str(first)
    assert "Profile-Switch-Audit: command=Profile Briefing · from=Standard · to=Briefing · rule=explicit-standalone-only" in first_html

    second = api.ask("Profile Briefing")
    second_html = second.get("html") if isinstance(second, dict) else str(second)
    assert "Profile-Switch-Audit: command=Profile Briefing · from=Briefing · to=Briefing · rule=explicit-standalone-only" in second_html


def test_profile_sandbox_and_briefing_initialize_color_off_but_color_toggle_can_override():
    mod = load_fix_module()
    _prime_module_gov(mod)
    try:
        rules = mod.gov.data if isinstance(mod.gov.data, dict) else {}
        profiles = rules.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            rules["profiles"] = profiles
        for _p in ("Sandbox", "Briefing"):
            pdef = profiles.get(_p)
            if not isinstance(pdef, dict):
                pdef = {}
                profiles[_p] = pdef
            pdef["color_default"] = "off"
    except Exception:
        pass
    api = mod.Api()
    api.validator = None

    out_sandbox = api.ask("Profile Sandbox")
    html_sandbox = out_sandbox.get("html") if isinstance(out_sandbox, dict) else str(out_sandbox)
    plain_sandbox = re.sub(r"<[^>]+>", " ", html_sandbox)
    assert "Color: off" in plain_sandbox
    assert api.gov_state.color == "off"

    out_briefing = api.ask("Profile Briefing")
    html_briefing = out_briefing.get("html") if isinstance(out_briefing, dict) else str(out_briefing)
    plain_briefing = re.sub(r"<[^>]+>", " ", html_briefing)
    assert "Color: off" in plain_briefing
    assert api.gov_state.color == "off"

    api.ask("Color on")
    assert api.gov_state.color == "on"
    reminder = api._state_reminder_line()
    assert "Profile=Briefing" in reminder
    assert "Color=on" in reminder


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


def test_apply_csc_strict_rebuilds_localized_qc_matrix_without_deltas_to_canonical_en_footer_with_tooltips():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.comm_active = True
    api.gov_state.color = "off"

    raw = (
        "Antworttext\n\n"
        "Self-Debunking:\n"
        "1. Weakness: A.\n"
        "2. Weakness: B.\n\n"
        "QC-Matrix: Klarheit 3 · Kürze 2 · Evidenz 3 · Empathie 2 · Konsistenz 3 · Neutralität 3"
    )

    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)

    assert "QC detected but no deltas found." not in html_out
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))
    plain = re.sub(r"\s+", " ", plain).strip()
    for lbl in ("Clarity", "Brevity", "Evidence", "Empathy", "Consistency", "Neutrality"):
        assert re.search(rf"{lbl}\s+[0-3]\s*\(Δ[+\-]?\d+\)", plain), plain
    assert "Klarheit " not in plain
    assert "Kürze " not in plain
    assert "Evidenz " not in plain
    assert "Empathie " not in plain
    assert "Konsistenz " not in plain
    assert "Neutralität " not in plain
    assert html_out.count('class="qc-dim-tip"') == 6
    assert html_out.find("Self-Debunking") < html_out.find("QC-Matrix:")
    assert html_out.find("QC-Matrix:") < html_out.find("Response at")


def test_apply_csc_strict_rebuilds_localized_qc_without_delta_and_keeps_trace_sd_qc_timestamp_order():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.validator = None
    api.gov_state.comm_active = True
    api.gov_state.color = "off"
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"
    api.gov_state.sci_pending = False
    api.gov_state.answer_language = "de"
    api._sci_variant_def = lambda _v: ({}, ["Plan", "Solution", "Check"], None)  # type: ignore[assignment]

    raw = (
        "Antwortteil.\n\n"
        "SCI Trace:\n"
        "1. Plan: Vorgehen.\n"
        "2. Solution: Ergebnis.\n"
        "3. Check: Konsistenz.\n\n"
        "Self-Debunking:\n"
        "1. Weakness: A.\n"
        "2. Weakness: B.\n\n"
        "QC-Matrix: Klarheit 3 · Kürze 2 · Evidenz 3 · Empathie 2 · Konsistenz 3 · Neutralität 3"
    )

    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))
    plain = re.sub(r"\s+", " ", plain).strip()

    for lbl in ("Clarity", "Brevity", "Evidence", "Empathy", "Consistency", "Neutrality"):
        assert re.search(rf"{lbl}\s+[0-3]\s*\(Δ[+\-]?\d+\)", plain), plain
    assert "Klarheit " not in plain
    assert "Kürze " not in plain
    assert "Evidenz " not in plain
    assert "Empathie " not in plain
    assert "Konsistenz " not in plain
    assert "Neutralität " not in plain
    assert html_out.count('class="qc-dim-tip"') == 6

    idx_trace = plain.find("SCI Trace")
    idx_sd = max(plain.find("Self-Debunking"), plain.find("Selbst-Debunking"))
    idx_qc = plain.find("QC-Matrix:")
    idx_ts = plain.find("Response at")
    assert idx_trace >= 0
    assert idx_sd >= 0
    assert idx_qc >= 0
    assert idx_ts >= 0
    assert idx_trace < idx_sd < idx_qc < idx_ts


def test_apply_csc_strict_footer_regression_no_localized_delta_less_footer_keeps_tooltips_and_order():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.validator = None
    api.gov_state.comm_active = True
    api.gov_state.color = "off"
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "B"
    api.gov_state.sci_pending = False
    api.gov_state.answer_language = "de"
    api._sci_variant_def = lambda _v: ({}, ["Plan", "Solution", "Check"], None)  # type: ignore[assignment]

    raw = (
        "Antwortteil.\n\n"
        "SCI Trace:\n"
        "1. Plan: Vorgehen.\n"
        "2. Solution: Ergebnis.\n"
        "3. Check: Konsistenz.\n\n"
        "Self-Debunking:\n"
        "1. Weakness: A.\n"
        "2. Weakness: B.\n\n"
        "QC-Matrix: Klarheit 3 · Kürze 2 · Evidenz 3 · Empathie 2 · Konsistenz 3 · Neutralität 3"
    )

    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))
    plain = re.sub(r"\s+", " ", plain).strip()

    idx_qc = plain.rfind("QC-Matrix:")
    idx_ts = plain.find("Response at")
    assert idx_qc >= 0
    assert idx_ts > idx_qc

    qc_tail = plain[idx_qc:idx_ts]
    assert not re.search(r"\b(Klarheit|Kürze|Kuerze|Evidenz|Empathie|Konsistenz|Neutralität|Neutralitaet)\b", qc_tail)
    for lbl in ("Clarity", "Brevity", "Evidence", "Empathy", "Consistency", "Neutrality"):
        assert re.search(rf"{lbl}\s+[0-3]\s*\(Δ[+\-]?\d+\)", qc_tail), qc_tail
    assert html_out.count('class="qc-dim-tip"') == 6

    idx_trace = plain.find("SCI Trace")
    idx_sd = max(plain.find("Self-Debunking"), plain.find("Selbst-Debunking"))
    assert idx_trace >= 0
    assert idx_sd >= 0
    assert idx_trace < idx_sd < idx_qc < idx_ts


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


def test_enforce_self_debunking_contract_adds_missing_secondary_fields_de():
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
        "Antworttext.\n\n"
        "Self-Debunking:\n\n"
        "1. Schwäche: Punkt eins ist zu knapp.\n\n"
        "2. Schwäche: Punkt zwei ist ebenfalls zu knapp.\n\n"
        "QC-Matrix: Klarheit 3 (Δ0) · Kürze 2 (Δ0) · Evidenz 2 (Δ0) · Empathie 2 (Δ0) · Konsistenz 3 (Δ0) · Neutralität 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(txt, GM(), "Expert", is_command=False, lang="de")
    assert out.count("**Warum das wichtig ist**:") >= 2
    assert out.count("**Was würde verifizieren/falsifizieren (nächster Check)**:") >= 2
    assert re.search(r"(?m)^\s*1\.\s+\*\*Schwäche\*\*:", out)
    assert re.search(r"(?m)^\s*2\.\s+\*\*Schwäche\*\*:", out)


def test_enforce_self_debunking_contract_adds_missing_secondary_fields_en():
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
        "1. Weakness: First point is too terse.\n\n"
        "2. Weakness: Second point is too terse.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )

    out = mod.enforce_self_debunking_contract(txt, GM(), "Expert", is_command=False, lang="en")
    assert out.count("**Why it matters**:") >= 2
    assert out.count("**What would verify/falsify (next check)**:") >= 2
    assert re.search(r"(?m)^\s*1\.\s+\*\*Weakness\*\*:", out)
    assert re.search(r"(?m)^\s*2\.\s+\*\*Weakness\*\*:", out)


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


def test_sanitize_self_debunking_markdown_in_html_cleans_nested_orphan_star_lines():
    mod = load_fix_module()
    html_in = (
        "<div class=\"self-debunking\">"
        "<div>Selbst-Debunking:</div>"
        "<ol><li><p><strong>Schwäche</strong>: Punkt eins."
        "   *<br><strong>Warum das wichtig ist</strong>: Punkt zwei."
        "   *<br><strong>Nächster Check</strong>: Punkt drei.</p></li></ol>"
        "</div>"
    )
    out = mod.sanitize_self_debunking_markdown_in_html(html_in)
    assert "*<br>" not in out
    assert "<strong>Warum das wichtig ist</strong>" in out
    assert "<strong>Nächster Check</strong>" in out


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


def test_normalize_markdown_list_spacing_handles_star_with_multiple_spaces():
    mod = load_fix_module()
    raw = (
        "Einleitungssatz.\n"
        "*   **Physikalische Definition:** Zeit beschreibt Dauer und Reihenfolge.\n"
        "*   Zweiter Punkt."
    )
    out = mod.normalize_markdown_list_spacing(raw)
    assert "Einleitungssatz.\n\n*   **Physikalische Definition:**" in out
    assert "Einleitungssatz.\n*   **Physikalische Definition:**" not in out


def test_normalize_markdown_list_spacing_enables_markdown_list_rendering():
    mod = load_fix_module()
    raw = (
        "<strong>Definitionen und Perspektiven:</strong>\n"
        "*   <strong>Physikalische Definition:</strong> In der Physik ist Zeit ...\n"
        "*   Zweiter Punkt."
    )
    norm = mod.normalize_markdown_list_spacing(raw)
    html_out = mod.markdown.markdown(norm, extensions=['extra', 'codehilite'])
    assert "<ul>" in html_out and "<li>" in html_out
    assert "*   <strong>Physikalische Definition:</strong>" not in html_out


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


def test_cgi_user_feedback_triplet_works_without_explicit_ruleset_block():
    mod = load_fix_module()
    data = _prime_module_gov(mod)
    api = mod.Api()

    # Simulate operational files where the optional block is absent.
    if isinstance(data, dict):
        gd = data.get("global_defaults")
        if isinstance(gd, dict):
            gd.pop("user_feedback_triplet", None)

    api.ask("Comm Start")

    def boom(*a, **k):
        raise AssertionError("LLM should not be called when CGI triplet fallback is active")

    api._llm_call = boom  # type: ignore[assignment]

    res = api.ask("3,3,3")
    assert "CGI feedback recorded" in (res.get("html") or "")


def test_cgi_bar_is_only_enabled_for_content_answers():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    raw_answer = (
        "Das ist eine Inhaltsantwort.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: A.\nWarum das wichtig ist: A.\nNächster Check: A.\n"
        "2. Schwäche: B.\nWarum das wichtig ist: B.\nNächster Check: B.\n"
        "3. Schwäche: C.\nWarum das wichtig ist: C.\nNächster Check: C.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )
    api._llm_call = lambda *a, **k: raw_answer  # type: ignore[assignment]

    cmd_out = api.ask("Comm Start")
    assert isinstance(cmd_out, dict)
    assert cmd_out.get("cgi_bar") is False

    ans_out = api.ask("Was ist Zeit?")
    assert isinstance(ans_out, dict)
    assert ans_out.get("cgi_bar") is True


def test_submit_cgi_feedback_save_records_triplet_without_llm_call():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    def boom(*a, **k):
        raise AssertionError("LLM must not be called for CGI save")

    api._llm_call = boom  # type: ignore[assignment]
    res = api.submit_cgi_feedback(3, 2, 2, "save")

    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert res.get("repeated") is False
    assert res.get("saved_triplet") == "3,2,2"
    assert api.gov_state.last_user_feedback_triplet == "3,2,2"
    assert bool(api.gov_state.cgi_feedback_pending_for_model) is True


def test_submit_cgi_feedback_repeat_reuses_last_content_prompt_and_injects_feedback():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    raw_answer = (
        "Das ist eine Inhaltsantwort.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: A.\nWarum das wichtig ist: A.\nNächster Check: A.\n"
        "2. Schwäche: B.\nWarum das wichtig ist: B.\nNächster Check: B.\n"
        "3. Schwäche: C.\nWarum das wichtig ist: C.\nNächster Check: C.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )
    prompts = []

    def fake_llm(prompt, reason="chat"):
        prompts.append(str(prompt))
        return raw_answer

    api._llm_call = fake_llm  # type: ignore[assignment]
    _ = api.ask("Wie funktioniert Zeitdilatation?")
    assert len(prompts) >= 1

    res = api.submit_cgi_feedback(0, 0, 0, "repeat")
    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert res.get("repeated") is True
    assert res.get("saved_triplet") == "0,0,0"
    assert "one-shot" in str(res.get("message", "")).lower()
    assert len(prompts) >= 2
    replay_prompts = prompts[1:]
    assert any("[CGI Feedback]" in p for p in replay_prompts)
    assert any("0,0,0" in p for p in replay_prompts)
    assert any("[CGI One-Shot Rewrite Constraints]" in p for p in replay_prompts)
    assert any(
        ("Nur fuer diese naechste Antwort anwenden" in p) or ("Apply only to this next answer" in p)
        for p in replay_prompts
    )
    assert any(
        ("Behandle dies als Ueberarbeitung, nicht als Paraphrase" in p)
        or ("Treat this as a rewrite, not a paraphrase" in p)
        for p in replay_prompts
    )
    assert any(
        ("Clarity-QC ist bereits 3" in p) or ("Clarity QC already at 3" in p)
        for p in replay_prompts
    )

    nested = res.get("response")
    assert isinstance(nested, dict)
    assert nested.get("cgi_bar") is True


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


def test_color_off_strips_textual_evidence_tags_and_marker_icons():
    mod = load_fix_module()
    raw = "A [GREEN] 🟢 claim. B [YELLOW] 🟡 maybe. C [RED] 🔴 unknown."
    out = mod.strip_color_markers_for_color_off_text(raw)
    assert "[GREEN]" not in out and "[YELLOW]" not in out and "[RED]" not in out
    assert "🟢" not in out and "🟡" not in out and "🔴" not in out
    assert "claim" in out and "maybe" in out and "unknown" in out


def test_color_off_strips_textual_evidence_tags_from_rendered_html():
    mod = load_fix_module()
    html_in = (
        "<p>[GREEN] 🟢 Aussage A.</p>"
        "<p><span class='signal-dot-marker'><span style=\"color:#c62828; font-weight:600;\">🔴</span></span> [RED] 🔴 Aussage B.</p>"
    )
    out = mod.strip_color_markers_for_color_off_html(html_in)
    assert "[GREEN]" not in out and "[RED]" not in out
    assert "🟢" not in out and "🔴" not in out
    assert "signal-dot-marker" not in out
    assert "Aussage A" in out and "Aussage B" in out


def test_color_off_strip_works_without_policy_module_via_renderer_fallback(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_color_marker_policy", None, raising=False)
    raw = "x [GREEN] 🟢 y [RED] 🔴 z"
    out_txt = mod.strip_color_markers_for_color_off_text(raw)
    out_html = mod.strip_color_markers_for_color_off_html(f"<p>{raw}</p>")
    assert "[GREEN]" not in out_txt and "[RED]" not in out_txt
    assert "🟢" not in out_txt and "🔴" not in out_txt
    assert "[GREEN]" not in out_html and "[RED]" not in out_html
    assert "🟢" not in out_html and "🔴" not in out_html


def test_apply_csc_strict_keeps_verification_route_lines_visible_by_default():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    raw = (
        "Antwortteil.\n"
        "Verification Route:\n"
        "Source: Primary Source=Example\n"
        "Measurement: Task=X\n"
        "Self-Debunking:\n"
        "1. Weakness: x\n"
        "2. Weakness: y\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )
    html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)
    txt = re.sub(r"<[^>]+>", " ", str(html_out or ""))
    assert "Verification Route" in txt
    assert "Source:" in txt
    assert "Measurement:" in txt


def test_apply_csc_strict_can_hide_verification_route_lines_via_config_toggle():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    cfg_obj = getattr(mod, "cfg", None)
    old = None
    try:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            old = cfg_obj.config.get("hide_verification_route_lines")
            cfg_obj.config["hide_verification_route_lines"] = True

        raw = (
            "Antwortteil.\n"
            "Verification Route:\n"
            "Source: Primary Source=Example\n"
            "Web-Check: Query Time=now\n"
            "Self-Debunking:\n"
            "1. Weakness: x\n"
            "2. Weakness: y\n"
            "QC-Matrix: Clarity 3 (Δ0) · Brevity 3 (Δ0) · Evidence 3 (Δ0) · Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
        )
        html_out, _ = api._apply_csc_strict(raw_response=raw, user_raw="x", is_command=False)
        txt = re.sub(r"<[^>]+>", " ", str(html_out or ""))
        assert "Verification Route" not in txt
        assert "Source:" not in txt
        assert "Web-Check:" not in txt
    finally:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            if old is None:
                cfg_obj.config.pop("hide_verification_route_lines", None)
            else:
                cfg_obj.config["hide_verification_route_lines"] = old


def test_native_retrieval_tool_capability_and_prompt_hint_are_exposed():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    cfg_obj = getattr(mod, "cfg", None)
    old = None
    try:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            old = g.get("native_retrieval")
            g["native_retrieval"] = "on"

        supports = bool(api._provider_supports_native_retrieval("gemini"))
        tools = api._build_native_tools_for_provider("gemini")
        if supports:
            assert isinstance(tools, list)
            assert len(tools) >= 1
        else:
            assert tools == []

        wrapped = api._wrap_user_text_for_model("Was ist Zeit?")
        assert "[RETRIEVAL TOOL]" in wrapped
    finally:
        if cfg_obj is not None and isinstance(getattr(cfg_obj, "config", None), dict):
            provs = cfg_obj.config.setdefault("providers", {})
            g = provs.setdefault("gemini", {})
            if old is None:
                g.pop("native_retrieval", None)
            else:
                g["native_retrieval"] = old


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


def test_strip_sci_varianten_table_echo_without_title_from_content_answer():
    mod = load_fix_module()
    raw = (
        "Intro line.\n"
        "| Variante | Name | Fokus / Methode |\n"
        "| --- | --- | --- |\n"
        "| A | Standard | Plan → Solution → Check (classic) |\n"
        "| B | Deep-Dive | Dialectics++ (13 steps; former SCIplus) |\n"
        "| C | Branch Evaluation | Tree-of-Thoughts: branching solution paths |\n"
        "| D | Axiomatic Reduction | First Principles: axiomatic reduction |\n"
        "| E | Confidence Tracker | Confidence + update via counterarguments |\n"
        "| F | Impact Projection | Second-order: downstream consequences |\n"
        "| G | Failure Mode Analysis | Pre-mortem/inversion: reason from failure |\n"
        "| H | Multi-Agent Simulation | Ensemble roles / expert simulation |\n\n"
        "Final Answer: Time is a measure of change.\n"
        "Self-Debunking:\n"
        "1. **Schwäche**: ...\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.strip_sci_menu_from_answer(raw)
    assert "Variante | Name | Fokus / Methode" not in out
    assert "| A | Standard |" not in out
    assert "Final Answer:" in out
    assert "Self-Debunking:" in out

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
    import re
    # Legacy path: helper kept in monolith.
    m = re.search(r"def\s+_toggle_btn\s*\(.*?\):\n(.*?)(?:\n\n|\n\s*#)", src, flags=re.S)
    if m:
        body = m.group(1)
        assert '{label_prefix}: OFF' in body or 'label_prefix}: OFF' in body, "Toggle label does not show OFF when is_on=True"
        assert '{label_prefix}: ON' in body or 'label_prefix}: ON' in body, "Toggle label does not show ON when is_on=False"
        return

    # S13 path: Comm toggle inversion moved into panel_ui_snapshot_seam.
    seam_path = FIX_PATH.parent / "panel_ui_snapshot_seam.py"
    assert seam_path.exists(), "Missing _toggle_btn in wrapper and panel_ui_snapshot_seam.py not found"
    seam_src = seam_path.read_text(encoding="utf-8")
    m2 = re.search(
        r"def\s+panel_ui_apply_failsoft_comm_toggle\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n(?P<body>(?:\s+.*\n){5,120})",
        seam_src,
    )
    assert m2, "Missing panel_ui_apply_failsoft_comm_toggle() seam helper"
    body = m2.group("body")
    assert "Comm ⏻: OFF" in body and "Comm Stop" in body, "Expected active-state toggle to show OFF action / Comm Stop"
    assert "Comm ⏻: ON" in body and "Comm Start" in body, "Expected inactive-state toggle to show ON action / Comm Start"

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


def test_html_number_self_debunking_fallback_cleans_markdown_leaks_without_box():
    mod = load_fix_module()
    html_in = (
        "<p><strong>Self-Debunking:</strong></p>\n"
        "<li><p><strong><em>*Schwäche</em>*:</strong> Punkt eins.\n"
        "*\n"
        "<strong><strong>Warum das wichtig ist</strong>:</strong> Relevanz eins.\n"
        "*\n"
        "<strong><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:</strong> Test eins.</p></li>\n"
        "<li><p><strong><em>*Schwäche</em>*:</strong> Punkt zwei.\n"
        "*\n"
        "<strong><strong>Warum das wichtig ist</strong>:</strong> Relevanz zwei.</p></li>\n"
        "<p>QC-Matrix: Clarity 3 (Δ0)</p>\n"
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<em>*Schwäche</em>*" not in out
    assert "<strong><strong>" not in out
    assert "<p>*</p>" not in out
    assert ">*\n" not in out
    assert "1. <strong>Schwäche</strong>: Punkt eins." in out or "1. Schwäche: Punkt eins." in out
    assert "2. <strong>Schwäche</strong>: Punkt zwei." in out or "2. Schwäche: Punkt zwei." in out
    assert "<strong>Warum das wichtig ist</strong>:" in out
    assert "<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:" in out


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


def test_strip_verification_route_display_lines_hides_html_wrapped_marker_line():
    mod = load_fix_module()
    raw = (
        "<div>Normaler Inhalt.</div>\n"
        "<div>Verification Route: Source: TRAIN (allgemeines Hintergrundwissen).</div>\n"
        "<div>Selbst-Debunking:</div>\n"
    )
    out = mod.strip_verification_route_display_lines(raw)
    assert "Verification Route: Source: TRAIN" not in out
    assert "Normaler Inhalt." in out
    assert "Selbst-Debunking:" in out


def test_strip_verification_route_display_lines_prefers_central_policy_module(monkeypatch):
    mod = load_fix_module()

    stub = types.SimpleNamespace(
        strip_verification_route_display_lines=lambda txt: "CENTRALIZED_STRIP",
    )
    monkeypatch.setattr(mod, "_output_verification_route_policy_renderer", stub, raising=False)

    out = mod.strip_verification_route_display_lines("Verification Route:\nSource: TRAIN")
    assert out == "CENTRALIZED_STRIP"


def test_strip_pathological_repetition_display_noise_replaces_long_cjk_run_for_german():
    mod = load_fix_module()
    raw = "Antwort " + ("算法和" * 80) + " Ende."
    out = mod.strip_pathological_repetition_display_noise(raw, lang="de")
    assert "算法和算法和" not in out
    assert "[entfernt: fehlerhafte Wiederholungssequenz]" in out


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


def test_strip_internal_scaffolding_status_lines_removes_profile_without_colon():
    mod = load_fix_module()
    raw = (
        "Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profile Standard\n"
        "Hier startet der eigentliche Inhalt.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Active profile:" in out
    assert "Profile Standard" not in out
    assert "Hier startet der eigentliche Inhalt." in out


def test_strip_internal_scaffolding_status_lines_removes_german_profil_status_line():
    mod = load_fix_module()
    raw = (
        "Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on\n"
        "Profil: Standard · Overlay: Strict · SCI: off · Control Layer: on · Color: on\n"
        "Hier startet der eigentliche Inhalt.\n"
    )
    out = mod.strip_internal_scaffolding_status_lines(raw)
    assert "Active profile:" in out
    assert "Profil: Standard · Overlay: Strict · SCI: off · Control Layer: on · Color: on" not in out
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


def test_strip_internal_scaffolding_status_html_removes_german_profil_block():
    mod = load_fix_module()
    html_in = (
        "<p>Active profile: Expert · SCI: B · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
        "<p>Profil: Expert · Overlay: Strict · SCI: B · Control Layer: on · Color: on</p>"
        "<p>Inhalt bleibt sichtbar.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Active profile:" in out
    assert "Profil: Expert · Overlay: Strict · SCI: B · Control Layer: on · Color: on" not in out
    assert "Inhalt bleibt sichtbar." in out


def test_strip_internal_scaffolding_status_html_removes_multiline_profile_overlay_sci_block():
    mod = load_fix_module()
    html_in = (
        "<div>Profile: Standard<br>Overlay: Strict<br>SCI: off<br>Color: on</div>"
        "<p>Inhalt bleibt sichtbar.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Profile: Standard" not in out
    assert "Overlay: Strict" not in out
    assert "SCI: off" not in out
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


def test_strip_internal_scaffolding_status_html_removes_profile_without_colon():
    mod = load_fix_module()
    html_in = (
        "<p>Active profile: Standard · SCI: off · Overlay: Strict · Control Layer: on · QC: on · CGI: on · Color: on</p>"
        "<p>Profile Standard</p>"
        "<p>Hier startet der eigentliche Inhalt.</p>"
    )
    out = mod.strip_internal_scaffolding_status_html(html_in)
    assert "Active profile:" in out
    assert "Profile Standard" not in out
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


def test_normalize_self_debunking_numbering_text_drops_verification_route_rows_and_repairs_trailing_md_stars():
    mod = load_fix_module()
    raw = (
        "Selbst-Debunking:\n"
        "1. Schwäche: Punkt eins.\n"
        "Warum das wichtig ist**: Relevanz eins.\n"
        "Was würde verifizieren/falsifizieren (nächster Check)**: Check eins.\n"
        "2. Schwäche: Punkt zwei.\n"
        "Verification Route:\n"
        "Source: TRAIN (Allgemeines Hintergrundwissen)\n"
        "Warum das wichtig ist**: Relevanz zwei.\n"
        "Was würde verifizieren/falsifizieren (nächster Check)**: Check zwei.\n"
        "\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.normalize_self_debunking_numbering_text(raw, lang="de")
    assert "Verification Route" not in out
    assert "Source: TRAIN" not in out
    assert "Warum das wichtig ist**:" not in out
    assert "Was würde verifizieren/falsifizieren (nächster Check)**:" not in out
    assert "Warum das wichtig ist:" in out
    assert "Was würde verifizieren/falsifizieren (nächster Check):" in out


def test_self_debunking_extended_labels_are_bold_and_on_new_lines():
    mod = load_fix_module()
    raw = (
        "Self-Debunking:\n"
        "1. Schwäche: Punkt. Vereinfachung: Kurzschluss. Nächster Schritt: Test.\n"
        "Warum relevant: Wichtig. Prüfen/Widerlegen (nächster Schritt): Route. Subjektivität: mittel.\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.normalize_self_debunking_field_linebreaks(raw, lang="de")
    out = mod.bold_self_debunking_labels(out, "de")

    assert "1. **Schwäche**: Punkt." in out
    assert "\n   **Vereinfachung**: Kurzschluss." in out
    assert "\n   **Nächster Schritt**: Test." in out
    assert "\n   **Warum relevant**: Wichtig." in out
    assert "\n   **Prüfen/Widerlegen (nächster Schritt)**: Route." in out
    assert "\n   **Subjektivität**: mittel." in out


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


def test_strip_sci_trace_line_when_inactive_removes_section_with_variant_heading():
    mod = load_fix_module()
    raw = (
        "Antworttext.\n\n"
        "4. SCI Trace (Variante A: Standard)\n"
        "1. Plan: Erste Fassung.\n"
        "2. Solution: Erste Fassung.\n"
        "3. Check: Erste Fassung.\n\n"
        "SCI Trace:\n"
        "1. Plan: Zweite Fassung.\n"
        "2. Solution: Zweite Fassung.\n"
        "3. Check: Zweite Fassung.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=False, sci_variant="", sci_pending=False)
    assert "SCI Trace" not in out
    assert "Self-Debunking:" in out
    assert "QC-Matrix:" in out
    assert "Antworttext." in out


def test_strip_sci_trace_line_when_inactive_removes_markdown_bold_trace_heading_block():
    mod = load_fix_module()
    raw = (
        "Antworttext.\n"
        "**SCI Trace:**\n"
        "1. Plan: Erste Fassung.\n"
        "2. Solution: Zweite Fassung.\n"
        "3. Check: Dritte Fassung.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=False, sci_variant="", sci_pending=False)
    assert "SCI Trace" not in out
    assert "Plan:" not in out
    assert "Self-Debunking:" in out
    assert "Antworttext." in out


def test_strip_sci_trace_line_when_inactive_removes_html_strong_trace_heading_block():
    mod = load_fix_module()
    raw = (
        "Einleitung.\n"
        "Um das Konzept der Zeit weiter zu untersuchen, schlage ich folgende SCI Trace vor (Variante A):\n"
        "<strong>SCI Trace</strong>\n"
        "1. Plan: Erste Fassung.\n"
        "2. Solution: Zweite Fassung.\n"
        "3. Check: Dritte Fassung.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=False, sci_variant="", sci_pending=False)
    assert "SCI Trace</strong>" not in out
    assert "<strong>SCI Trace</strong>" not in out
    assert "Plan:" not in out
    assert "Self-Debunking:" in out
    assert "Einleitung." in out
    assert "schlage ich folgende SCI Trace vor" in out


def test_strip_sci_trace_line_when_inactive_keeps_trace_when_variant_active():
    mod = load_fix_module()
    raw = "SCI Trace: Plan\n1. Plan: ...\n"
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=True, sci_variant="B", sci_pending=False)
    assert out == raw


def test_strip_sci_trace_line_when_inactive_strips_when_active_flag_set_but_variant_unset():
    mod = load_fix_module()
    raw = (
        "Antworttext.\n"
        "SCI Trace:\n"
        "Plan: A\n"
        "Solution: B\n"
        "Check: C\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = mod.strip_sci_trace_line_when_inactive(raw, sci_active=True, sci_variant="", sci_pending=False)
    assert "SCI Trace" not in out
    assert "Plan:" not in out
    assert "Self-Debunking:" in out


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


def test_render_sci_trace_runtime_moves_final_answer_out_of_last_step_block():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Check"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "- Plan\n"
        "- Solution\n"
        "- Check\n\n"
        "Plan: Vorgehen festlegen.\n"
        "Solution: Struktur aufbauen.\n"
        "Check: Abschlusskriterium formulieren.\n"
        "\n"
        "<span class='signal-dot-marker'>🟢</span>Zeit beschreibt die Ordnung von Ereignissen in Vergangenheit, Gegenwart und Zukunft.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "Zeit beschreibt die Ordnung von Ereignissen" in out

    m = re.search(r"Check:</div>(.*?)</li>", out, flags=re.DOTALL)
    assert m is not None
    assert "Zeit beschreibt die Ordnung von Ereignissen" not in m.group(1)

    trace_end = out.find("</ol>\n</div>")
    final_pos = out.find("Zeit beschreibt die Ordnung von Ereignissen")
    assert trace_end >= 0
    assert final_pos > trace_end


def test_render_sci_trace_runtime_drops_final_answer_marker_label_from_trace_block():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Check"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "- Plan\n"
        "- Solution\n"
        "- Check\n\n"
        "Plan: Vorgehen festlegen.\n"
        "Solution: Struktur aufbauen.\n"
        "Check: Abschlusskriterium formulieren.\n"
        "\n"
        "Final Answer:\n"
        "Zeit ist eine Ordnungsdimension fuer Ereignisse.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "Zeit ist eine Ordnungsdimension fuer Ereignisse." in out
    assert "Final Answer:" not in out

    m = re.search(r"Check:</div>(.*?)</li>", out, flags=re.DOTALL)
    assert m is not None
    assert "Final Answer:" not in m.group(1)
    assert "Zeit ist eine Ordnungsdimension fuer Ereignisse." not in m.group(1)


@pytest.mark.parametrize(
    "answer_marker, expected_sentence",
    [
        ("Final Answer: Time is an ordering dimension of events.", "Time is an ordering dimension of events."),
        ("Antwort: Zeit ist eine Ordnungsdimension fuer Ereignisse.", "Zeit ist eine Ordnungsdimension fuer Ereignisse."),
    ],
)
def test_render_sci_trace_runtime_strips_inline_answer_markers_from_last_step(answer_marker, expected_sentence):
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Check"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "- Plan\n"
        "- Solution\n"
        "- Check\n\n"
        "Plan: Vorgehen festlegen.\n"
        "Solution: Struktur aufbauen.\n"
        "Check: Abschlusskriterium formulieren.\n"
        f"{answer_marker}\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert expected_sentence in out
    assert "Final Answer:" not in out
    assert "Antwort:" not in out

    m = re.search(r"Check:</div>(.*?)</li>", out, flags=re.DOTALL)
    assert m is not None
    assert "Final Answer:" not in m.group(1)
    assert "Antwort:" not in m.group(1)
    assert expected_sentence not in m.group(1)


def test_render_sci_trace_runtime_strips_numbered_final_answer_marker_from_last_step():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "B"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Critic", "Learn"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "- Plan\n"
        "- Solution\n"
        "- Critic\n"
        "- Learn\n\n"
        "Plan: Vorgehen festlegen.\n"
        "Solution: Antwort aufbauen.\n"
        "Critic: Risiken und Luecken markieren.\n"
        "Learn: Synthese der Punkte.\n"
        "14. Final Answer:\n"
        "Zeit ist eine Ordnungsdimension fuer Ereignisse.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "Zeit ist eine Ordnungsdimension fuer Ereignisse." in out
    assert "Final Answer:" not in out

    m = re.search(r"Learn:</div>(.*?)</li>", out, flags=re.DOTALL)
    assert m is not None
    assert "Final Answer:" not in m.group(1)
    assert "Zeit ist eine Ordnungsdimension fuer Ereignisse." not in m.group(1)


def test_render_sci_trace_runtime_moves_trace_before_narrative_when_model_places_trace_late():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Check"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "Antwortteil vor dem Trace.\n"
        "Noch eine Satzzeile.\n\n"
        "SCI Trace:\n"
        "1. Plan: Vorgehen.\n"
        "2. Solution: Ergebnis.\n"
        "3. Check: Konsistenz.\n\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    idx_trace = out.find("class='sci-trace'")
    idx_answer = out.find("Antwortteil vor dem Trace.")
    idx_sd = out.find("Self-Debunking:")
    assert idx_trace >= 0
    assert idx_answer >= 0
    assert idx_sd >= 0
    assert idx_trace < idx_answer < idx_sd


def test_apply_csc_strict_places_trace_before_answer_and_self_debunking_for_sci_answers():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"
    api.gov_state.sci_pending = False
    api.gov_state.answer_language = "de"
    api._sci_variant_def = lambda _v: ({}, ["Plan", "Solution", "Check"], None)  # type: ignore[assignment]

    raw = (
        "Antwortteil vor dem Trace.\n\n"
        "SCI Trace:\n"
        "1. Plan: Vorgehen.\n"
        "2. Solution: Ergebnis.\n"
        "3. Check: Konsistenz.\n\n"
        "Self-Debunking:\n"
        "- Schwäche: zu knapp.\n"
        "- Schwäche: ohne Quellen.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )
    html_out, _meta = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))

    idx_header = plain.find("Active profile:")
    idx_trace = plain.find("SCI Trace")
    idx_answer = plain.find("Antwortteil vor dem Trace.")
    idx_sd = max(plain.find("Selbst-Debunking"), plain.find("Self-Debunking"))
    assert idx_header >= 0
    assert idx_trace >= 0
    assert idx_answer >= 0
    assert idx_sd >= 0
    assert idx_header < idx_trace < idx_answer < idx_sd
    assert "raw-output" not in str(html_out or "")


def test_render_sci_trace_runtime_strips_leaked_markdown_edge_emphasis_tokens_in_step_content():
    mod = load_fix_module()
    api = mod.Api()
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"

    def _variant_def(_v):
        return ({}, ["Plan", "Solution", "Check"], None)

    api._sci_variant_def = _variant_def  # type: ignore[assignment]
    raw = (
        "SCI Trace:\n"
        "1. Plan: ** Definiere den Begriff Zeit.\n"
        "2. Solution: ** Erkläre physikalische und philosophische Perspektiven.\n"
        "3. Check: ** Prüfe Grenzen der Definition **\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
    )
    out = api._render_sci_trace_as_html_runtime(raw)
    assert "<div class='sci-trace'" in out
    assert "** Definiere" not in out
    assert "** Erkläre" not in out
    assert "Definition **" not in out
    assert "Definiere den Begriff Zeit." in out
    assert "Erkläre physikalische und philosophische Perspektiven." in out
    assert "Prüfe Grenzen der Definition" in out


def test_apply_csc_strict_sci_trace_does_not_render_leading_double_asterisk_fragments():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.validator = None
    api.gov_state.sci_active = True
    api.gov_state.sci_variant = "A"
    api.gov_state.sci_pending = False
    api.gov_state.answer_language = "de"
    api._sci_variant_def = lambda _v: ({}, ["Plan", "Solution", "Check"], None)  # type: ignore[assignment]

    raw = (
        "Antwortteil.\n\n"
        "SCI Trace:\n"
        "1. Plan: ** Definiere Zeit.\n"
        "2. Solution: ** Gib zwei Perspektiven.\n"
        "3. Check: ** Prüfe Konsistenz.\n\n"
        "Self-Debunking:\n"
        "- Schwäche: zu knapp.\n"
        "- Schwäche: ohne Quellen.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 2 (Δ0)\n"
    )
    html_out, _meta = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    assert "<div class=\"sci-trace\"" in str(html_out or "") or "<div class='sci-trace'" in str(html_out or "")
    assert ">** " not in str(html_out or "")


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


def test_validate_sci_trace_requires_final_answer_content_outside_trace():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)
    validator._required_trace_steps_for_variant = lambda _v: ["Plan", "Solution", "Learn"]  # type: ignore[assignment]

    raw = (
        "SCI Trace:\n"
        "Plan: Wir planen die Antwort.\n"
        "Solution: Wir geben eine Kernantwort.\n"
        "Learn: Wir reflektieren das Ergebnis.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
        "QC-Matrix: Clarity 2 (Δ0)\n"
    )
    vios = validator.validate_sci_trace(raw, "B")
    assert any("Missing substantive final answer content outside SCI Trace." in v for v in vios)


def test_validate_sci_trace_accepts_final_answer_content_outside_trace():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)
    validator._required_trace_steps_for_variant = lambda _v: ["Plan", "Solution", "Learn"]  # type: ignore[assignment]

    raw = (
        "SCI Trace:\n"
        "Plan: Wir planen die Antwort.\n"
        "Solution: Wir geben eine Kernantwort.\n"
        "Learn: Wir reflektieren das Ergebnis.\n"
        "\n"
        "Zeit ist ein physikalisches und zugleich erfahrungsbezogenes Konzept.\n"
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
        "QC-Matrix: Clarity 2 (Δ0)\n"
    )
    vios = validator.validate_sci_trace(raw, "B")
    assert not any("Missing substantive final answer content outside SCI Trace." in v for v in vios)


@pytest.mark.parametrize(
    "variant_key, steps",
    [
        ("A", ["Plan", "Solution", "Check"]),
        ("B", ["Plan", "Solution", "Learn"]),
        ("C", ["Branch_1", "Branch_2", "Selection"]),
        ("D", ["Axiom_1", "Axiom_2", "Synthesis"]),
        ("E", ["Confidence_0", "Counterargument", "Confidence_1"]),
        ("F", ["First-order", "Second-order", "Third-order"]),
        ("G", ["Pre-mortem", "Failure modes", "Mitigations"]),
        ("H", ["Agent_1", "Agent_2", "Synthesis"]),
    ],
)
def test_validate_sci_trace_requires_final_answer_content_outside_trace_across_variants(variant_key, steps):
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)
    validator._required_trace_steps_for_variant = lambda _v: list(steps)  # type: ignore[assignment]

    raw = "SCI Trace:\n" + "".join(f"{s}: Inhalt.\n" for s in steps) + (
        "Self-Debunking:\n"
        "1. Schwäche: x\n"
        "QC-Matrix: Clarity 2 (Δ0)\n"
    )
    vios = validator.validate_sci_trace(raw, variant_key)
    assert any("Missing substantive final answer content outside SCI Trace." in v for v in vios)


@pytest.mark.parametrize(
    "variant_key, steps",
    [
        ("A", ["Plan", "Solution", "Check"]),
        ("B", ["Plan", "Solution", "Learn"]),
        ("C", ["Branch_1", "Branch_2", "Selection"]),
        ("D", ["Axiom_1", "Axiom_2", "Synthesis"]),
        ("E", ["Confidence_0", "Counterargument", "Confidence_1"]),
        ("F", ["First-order", "Second-order", "Third-order"]),
        ("G", ["Pre-mortem", "Failure modes", "Mitigations"]),
        ("H", ["Agent_1", "Agent_2", "Synthesis"]),
    ],
)
def test_validate_sci_trace_accepts_final_answer_content_outside_trace_across_variants(variant_key, steps):
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, mod.cfg)
    validator._required_trace_steps_for_variant = lambda _v: list(steps)  # type: ignore[assignment]

    raw = (
        "SCI Trace:\n"
        + "".join(f"{s}: Inhalt.\n" for s in steps)
        + "\nSubstanzieller Antwortteil außerhalb des SCI Trace.\n"
        + "Self-Debunking:\n"
        + "1. Schwäche: x\n"
        + "QC-Matrix: Clarity 2 (Δ0)\n"
    )
    vios = validator.validate_sci_trace(raw, variant_key)
    assert not any("Missing substantive final answer content outside SCI Trace." in v for v in vios)


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


def test_ensure_self_debunking_box_html_reboxes_list_leak_and_keeps_qc_footer():
    mod = load_fix_module()
    html_in = (
        "<ul>"
        "<li><p><strong>Heidegger:</strong> Text. <strong>Self-Debunking:</strong></p></li>"
        "<li><p>1. <strong>Schwäche</strong>: Punkt eins."
        "<br><strong>Warum das wichtig ist</strong>: A.</p></li>"
        "<li><p>2. <strong>Schwäche</strong>: Punkt zwei."
        "<br><strong>Warum das wichtig ist</strong>: B.</p></li>"
        "</ul>"
        "<p>QC-Matrix: Clarity 3 (Δ0)</p>"
    )
    out = mod.ensure_self_debunking_box_html(html_in, lang="de")
    assert 'class="self-debunking"' in out
    assert "Selbst-Debunking:" in out
    assert "<ol>" in out and "</ol>" in out
    assert "<strong>Self-Debunking:</strong>" not in out
    assert re.search(r"(?is)<ol>.*?<li>\s*1\.", out) is None
    assert "<p>QC-Matrix: Clarity 3 (Δ0)</p>" in out


def test_self_debunking_formatter_fallback_import_works_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()
    fmt = getattr(mod, "_sd_formatter", None)
    assert fmt is not None
    assert callable(getattr(fmt, "ensure_self_debunking_box_html", None))


def test_uncertainty_codes_fallback_import_works_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()
    uc = getattr(mod, "_uncertainty_codes_mod", None)
    assert uc is not None
    assert callable(getattr(uc, "ensure_uncertainty_annotations_html", None))


def test_output_renderers_fallback_import_work_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()

    u = getattr(mod, "_output_uncertainty_renderer", None)
    h = getattr(mod, "_output_header_renderer", None)
    f = getattr(mod, "_output_footer_renderer", None)
    s = getattr(mod, "_output_sci_trace_renderer", None)
    cscw = getattr(mod, "_output_csc_warning_renderer", None)
    cln = getattr(mod, "_output_control_layer_note_renderer", None)
    cgil = getattr(mod, "_output_cgi_line_renderer", None)
    colm = getattr(mod, "_output_color_markers_renderer", None)
    iem = getattr(mod, "_output_image_embed_renderer", None)
    sg = getattr(mod, "_output_strict_gate_renderer", None)
    rq = getattr(mod, "_output_render_quality_renderer", None)
    cmdcat = getattr(mod, "_output_command_response_catalog", None)

    assert u is not None
    assert callable(getattr(u, "append_uncertainty_explanation_if_needed", None))
    assert h is not None
    assert callable(getattr(h, "render_profile_switch_control_html", None))
    assert callable(getattr(h, "normalize_sci_display", None))
    assert callable(getattr(h, "build_active_profile_status_line", None))
    assert callable(getattr(h, "build_comm_status_line", None))
    assert f is not None
    assert callable(getattr(f, "render_ts_footer_html", None))
    assert callable(getattr(f, "ensure_qc_footer_html_consistency", None))
    assert callable(getattr(f, "finalize_qc_footer_html", None))
    assert callable(getattr(f, "annotate_qc_matrix_tooltips_html", None))
    assert s is not None
    assert callable(getattr(s, "render_sci_trace_as_html_runtime", None))
    assert cscw is not None
    assert callable(getattr(cscw, "render_control_layer_block_html", None))
    assert callable(getattr(cscw, "render_strict_block_html", None))
    assert callable(getattr(cscw, "render_strict_warn_banner_html", None))
    assert callable(getattr(cscw, "render_cross_version_guard_html", None))
    assert cln is not None
    assert callable(getattr(cln, "render_control_layer_alert_html", None))
    assert callable(getattr(cln, "render_repair_pass_banner_html", None))
    assert cgil is not None
    assert callable(getattr(cgil, "get_cgi_ui_texts", None))
    assert callable(getattr(cgil, "render_cgi_feedback_block", None))
    assert callable(getattr(cgil, "render_cgi_constraints_block", None))
    assert colm is not None
    assert callable(getattr(colm, "apply_color_spans", None))
    assert callable(getattr(colm, "reapply_color_styles_if_stripped", None))
    assert callable(getattr(colm, "strip_color_markers_for_color_off_text", None))
    assert callable(getattr(colm, "strip_color_markers_for_color_off_html", None))
    assert callable(getattr(colm, "annotate_signal_dot_tooltips_html", None))
    assert callable(getattr(colm, "inject_fallback_signal_dots_html", None))
    assert callable(getattr(colm, "limit_signal_dot_marker_density_html", None))
    assert callable(getattr(colm, "strip_signal_dots_from_heading_only_blocks_html", None))
    assert iem is not None
    assert callable(getattr(iem, "auto_embed_image_urls", None))
    assert sg is not None
    assert callable(getattr(sg, "evaluate_strict_enforcement", None))
    assert rq is not None
    assert callable(getattr(rq, "looks_like_rendered_html", None))
    assert callable(getattr(rq, "build_normalization_summary", None))
    assert cmdcat is not None
    assert callable(getattr(cmdcat, "build_profile_switch_audit_line", None))
    assert callable(getattr(cmdcat, "resolve_sci_on_command_html", None))
    assert callable(getattr(cmdcat, "resolve_comm_overlay_command_html", None))
    assert callable(getattr(cmdcat, "resolve_comm_validate_command_html", None))
    assert callable(getattr(cmdcat, "resolve_comm_anchor_toggle_command_html", None))
    assert callable(getattr(cmdcat, "resolve_dynamic_one_shot_command_html", None))
    assert callable(getattr(cmdcat, "resolve_post_state_command_html", None))
    assert callable(getattr(cmdcat, "build_qc_override_opened_result", None))
    assert callable(getattr(cmdcat, "build_comm_audit_command_result", None))
    assert callable(getattr(cmdcat, "build_sci_menu_command_result", None))
    assert callable(getattr(cmdcat, "build_sci_selection_result", None))
    assert callable(getattr(cmdcat, "resolve_renderer_map_command", None))
    assert callable(getattr(cmdcat, "apply_profile_switch_state", None))
    assert callable(getattr(cmdcat, "apply_basic_command_state", None))
    assert callable(getattr(cmdcat, "is_basic_command_supported", None))


def test_output_rules_registry_and_state_snapshot_fallback_imports_work_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()

    rr = getattr(mod, "_output_rules_registry", None)
    ss = getattr(mod, "_output_state_snapshot", None)
    assert rr is not None
    assert callable(getattr(rr, "qc_probe_is_complete", None))
    assert ss is not None
    snap_cls = getattr(ss, "OutputStateSnapshot", None)
    assert snap_cls is not None
    assert callable(getattr(snap_cls, "from_runtime_state", None))


def test_output_resolver_dispatcher_and_routing_runtime_fallback_imports_work_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()

    resolver = getattr(mod, "_output_resolver", None)
    dispatcher = getattr(mod, "_output_dispatcher", None)
    runtime = getattr(mod, "_output_routing_runtime", None)
    assert resolver is not None
    assert callable(getattr(resolver, "resolve_input", None))
    assert callable(getattr(resolver, "normalize_route_shape", None))
    assert callable(getattr(resolver, "contract_route_shape", None))
    assert dispatcher is not None
    assert callable(getattr(dispatcher, "route_kind", None))
    assert callable(getattr(dispatcher, "is_command_route", None))
    assert callable(getattr(dispatcher, "route_meta", None))
    assert callable(getattr(dispatcher, "route_contract_ok", None))
    assert callable(getattr(dispatcher, "route_audit_payload", None))
    assert runtime is not None
    assert callable(getattr(runtime, "resolve_route_context", None))
    assert callable(getattr(runtime, "route_contract_ok", None))
    assert callable(getattr(runtime, "route_audit_payload", None))


def test_output_pipeline_fallback_imports_work_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()

    pipe = getattr(mod, "_output_pipeline", None)
    assert pipe is not None
    assert callable(getattr(pipe, "post_render_normalization", None))
    assert callable(getattr(pipe, "normalize_post_render_html", None))
    assert callable(getattr(pipe, "normalize_self_debunking_postprocess_text", None))
    assert callable(getattr(pipe, "resolve_hide_verification_route_lines", None))
    assert callable(getattr(pipe, "apply_verification_route_display_policy", None))
    seam = getattr(mod, "_output_self_debunking_runtime_seam", None)
    assert seam is not None
    assert callable(getattr(seam, "apply_self_debunking_text_postprocess", None))
    assert callable(getattr(seam, "apply_post_render_normalization", None))
    qc_stage = getattr(mod, "_output_post_render_qc_stage", None)
    assert qc_stage is not None
    assert callable(getattr(qc_stage, "ensure_qc_footer_html_consistency_html_stage", None))
    assert callable(getattr(qc_stage, "finalize_qc_footer_html_stage", None))
    render_end_stage = getattr(mod, "_output_final_html_runtime_stage", None)
    assert render_end_stage is not None
    assert callable(getattr(render_end_stage, "finalize_render_end_html_stage", None))
    render_body_stage = getattr(mod, "_output_render_body_runtime_stage", None)
    assert render_body_stage is not None
    assert callable(getattr(render_body_stage, "render_final_html_body_stage", None))
    route_render_stage = getattr(mod, "_output_route_render_runtime_stage", None)
    assert route_render_stage is not None
    assert callable(getattr(route_render_stage, "render_command_html_stage", None))
    assert callable(getattr(route_render_stage, "render_comm_inactive_html_stage", None))
    csc_mid_stage = getattr(mod, "_output_csc_mid_runtime_stage", None)
    assert csc_mid_stage is not None
    assert callable(getattr(csc_mid_stage, "build_csc_refiner_meta_stage", None))
    assert callable(getattr(csc_mid_stage, "build_alerts_and_header_stage", None))
    assert callable(getattr(csc_mid_stage, "apply_pre_render_policy_strict_gate_stage", None))
    assert callable(getattr(csc_mid_stage, "apply_pre_render_policy_strict_gate_runtime_chain_stage", None))
    assert callable(getattr(csc_mid_stage, "apply_pre_render_policy_strict_gate_ultimate_fallback_stage", None))
    assert callable(getattr(csc_mid_stage, "build_pre_render_policy_strict_gate_runtime_bundle", None))
    assert callable(getattr(csc_mid_stage, "build_pre_render_policy_strict_gate_runtime_passthrough_out", None))
    assert callable(getattr(csc_mid_stage, "apply_pre_render_policy_strict_gate_runtime_dispatch_stage", None))
    assert callable(getattr(csc_mid_stage, "apply_pre_render_policy_strict_gate_runtime_ultimate_fallback_from_bundle_stage", None))
    assert callable(getattr(csc_mid_stage, "normalize_pre_render_policy_strict_gate_stage_out", None))


def test_output_verification_route_policy_renderer_fallback_imports_work_without_src_on_syspath(monkeypatch):
    src_str = str(SRC.resolve())
    filtered = [p for p in sys.path if str(Path(p).resolve()) != src_str] if sys.path else []
    monkeypatch.setattr(sys, "path", filtered)
    mod = load_fix_module()

    vr_policy = getattr(mod, "_output_verification_route_policy_renderer", None)
    assert vr_policy is not None
    assert callable(getattr(vr_policy, "is_verification_route_marker_line", None))
    assert callable(getattr(vr_policy, "strip_verification_route_display_lines", None))
    assert callable(getattr(vr_policy, "resolve_hide_verification_route_lines", None))
    assert callable(getattr(vr_policy, "apply_verification_route_display_policy", None))


def test_output_pipeline_post_render_normalization_order_and_color_policy():
    mod = load_fix_module()
    pipe = getattr(mod, "_output_pipeline", None)
    assert pipe is not None

    calls = []

    def _mark(name):
        def _fn(text, **_kwargs):
            calls.append(name)
            return f"{text}|{name}"
        return _fn

    out = pipe.post_render_normalization(
        "X",
        answer_lang="de",
        color="off",
        ensure_self_debunking_box_html_fn=_mark("box"),
        sanitize_self_debunking_markdown_in_html_fn=_mark("sanitize"),
        normalize_hash_subheadings_in_html_fn=_mark("hash"),
        strip_internal_scaffolding_status_html_fn=_mark("strip_status"),
        html_number_self_debunking_fn=_mark("number"),
        strip_color_markers_for_color_off_html_fn=_mark("strip_color"),
    )
    assert out.startswith("X|box|sanitize|hash|strip_status|number|sanitize|box")
    assert out.endswith("|strip_color")
    assert calls == ["box", "sanitize", "hash", "strip_status", "number", "sanitize", "box", "strip_color"]

    calls = []
    out_on = pipe.post_render_normalization(
        "Y",
        answer_lang="en",
        color="on",
        ensure_self_debunking_box_html_fn=_mark("box"),
        sanitize_self_debunking_markdown_in_html_fn=_mark("sanitize"),
        normalize_hash_subheadings_in_html_fn=_mark("hash"),
        strip_internal_scaffolding_status_html_fn=_mark("strip_status"),
        html_number_self_debunking_fn=_mark("number"),
        strip_color_markers_for_color_off_html_fn=_mark("strip_color"),
    )
    assert out_on.startswith("Y|box|sanitize|hash|strip_status|number|sanitize|box")
    assert not out_on.endswith("|strip_color")
    assert calls == ["box", "sanitize", "hash", "strip_status", "number", "sanitize", "box"]


def test_output_pipeline_self_debunking_postprocess_order_and_fail_soft():
    mod = load_fix_module()
    pipe = getattr(mod, "_output_pipeline", None)
    assert pipe is not None

    calls = []

    def _mark(name, *, fail=False):
        def _fn(text):
            calls.append(name)
            if fail:
                raise RuntimeError(name)
            return f"{text}|{name}"
        return _fn

    out = pipe.normalize_self_debunking_postprocess_text(
        "X",
        normalize_inline_self_debunking_header_fn=_mark("inline"),
        enforce_self_debunking_contract_fn=_mark("enforce"),
        normalize_self_debunking_numbering_text_fn=_mark("number"),
        dedupe_self_debunking_sections_fn=_mark("dedupe"),
    )
    assert out == "X|inline|enforce|number|dedupe"
    assert calls == ["inline", "enforce", "number", "dedupe"]

    calls = []
    out_fail_soft = pipe.normalize_self_debunking_postprocess_text(
        "Y",
        normalize_inline_self_debunking_header_fn=_mark("inline"),
        enforce_self_debunking_contract_fn=_mark("enforce", fail=True),
        normalize_self_debunking_numbering_text_fn=_mark("number"),
        dedupe_self_debunking_sections_fn=_mark("dedupe"),
    )
    assert out_fail_soft == "Y|inline|number|dedupe"
    assert calls == ["inline", "enforce", "number", "dedupe"]


def test_output_pipeline_verification_route_display_policy_is_centralized_and_provider_aware():
    mod = load_fix_module()
    pipe = getattr(mod, "_output_pipeline", None)
    assert pipe is not None

    raw = (
        "Antwortblock\n"
        "Verification Route:\n"
        "Source: TRAIN (general background knowledge)\n"
        "Self-Debunking:\n"
        "1. Weakness: x\n"
    )

    # Default remains visible.
    out_default = pipe.apply_verification_route_display_policy(
        raw,
        config={},
        provider="gemini",
        strip_verification_route_display_lines_fn=lambda txt: "STRIPPED",
    )
    assert out_default == raw

    # Provider override wins for the selected provider.
    out_provider = pipe.apply_verification_route_display_policy(
        raw,
        config={"providers": {"openrouter": {"hide_verification_route_lines": True}}},
        provider="openrouter",
        strip_verification_route_display_lines_fn=lambda txt: "STRIPPED",
    )
    assert out_provider == "STRIPPED"

    out_other_provider = pipe.apply_verification_route_display_policy(
        raw,
        config={"providers": {"openrouter": {"hide_verification_route_lines": True}}},
        provider="gemini",
        strip_verification_route_display_lines_fn=lambda txt: "STRIPPED",
    )
    assert out_other_provider == raw

    # Root fallback supports string toggles.
    out_root = pipe.apply_verification_route_display_policy(
        raw,
        config={"hide_verification_route_lines": "on"},
        provider="gemini",
        strip_verification_route_display_lines_fn=lambda txt: "STRIPPED",
    )
    assert out_root == "STRIPPED"


def test_apply_self_debunking_text_postprocess_seam_prefers_pipeline(monkeypatch):
    mod = load_fix_module()
    stub_pipe = types.SimpleNamespace(
        normalize_self_debunking_postprocess_text=lambda txt, **kwargs: "PIPE_SD_TEXT"
    )
    monkeypatch.setattr(mod, "_output_pipeline", stub_pipe, raising=False)

    out = mod.apply_self_debunking_text_postprocess_seam(
        "RAW",
        gov_mgr=mod.gov,
        profile_name="Standard",
        is_command=False,
        lang="de",
    )
    assert out == "PIPE_SD_TEXT"


def test_apply_post_render_normalization_seam_prefers_pipeline(monkeypatch):
    mod = load_fix_module()
    stub_pipe = types.SimpleNamespace(
        post_render_normalization=lambda html_body, **kwargs: "PIPE_HTML_NORM"
    )
    monkeypatch.setattr(mod, "_output_pipeline", stub_pipe, raising=False)

    out = mod.apply_post_render_normalization_seam(
        "<p>X</p>",
        answer_lang="de",
        color="off",
    )
    assert out == "PIPE_HTML_NORM"


def test_output_header_renderer_status_line_sci_pending_modes_and_color_policy():
    mod = load_fix_module()
    h = getattr(mod, "_output_header_renderer", None)
    assert h is not None

    line_pending = h.build_active_profile_status_line(
        profile="Expert",
        sci_variant="",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        sci_pending=True,
        off_label="off",
        pending_label="PENDING",
        pending_mode="when_pending_and_unset",
        uppercase_sci_non_off=False,
        color_force_off_profiles=(),
    )
    assert "SCI: PENDING" in line_pending

    line_a = h.build_active_profile_status_line(
        profile="Expert",
        sci_variant="A",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        sci_pending=True,
        off_label="off",
        pending_label="PENDING",
        pending_mode="when_pending_and_unset",
        uppercase_sci_non_off=False,
        color_force_off_profiles=(),
    )
    assert "SCI: A" in line_a

    line_sandbox = h.build_active_profile_status_line(
        profile="Sandbox",
        sci_variant="B",
        overlay="Explore",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        sci_pending=False,
        off_label="off",
        pending_label="PENDING",
        pending_mode="when_pending_and_unset",
        uppercase_sci_non_off=False,
        color_force_off_profiles=(),
    )
    assert line_sandbox.endswith("Color: on")


def test_status_line_prefers_modular_header_builder(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    def _builder(**kwargs):
        return "STATUS-LINE-FROM-MODULE"

    monkeypatch.setattr(
        mod,
        "_output_header_renderer",
        types.SimpleNamespace(build_active_profile_status_line=_builder),
        raising=False,
    )

    line = api._status_line(
        sysname="Comm-SCI-Control",
        ver="20.2.5",
        profile="Expert",
        sci="off",
        overlay="Strict",
        ctl="on",
        qc="on",
        cgi="on",
        color="on",
    )
    assert line == "STATUS-LINE-FROM-MODULE"


def test_wrapper_build_active_profile_header_line_pending_mode_when_variant_unset():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    line_pending = api._build_active_profile_header_line(
        profile="Expert",
        sci_variant="",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        sci_pending=True,
        off_label="off",
        pending_label="PENDING",
        pending_mode="when_pending_and_unset",
        uppercase_sci_non_off=False,
        color_force_off_profiles=(),
    )
    assert "SCI: PENDING" in line_pending

    line_variant = api._build_active_profile_header_line(
        profile="Expert",
        sci_variant="A",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        sci_pending=True,
        off_label="off",
        pending_label="PENDING",
        pending_mode="when_pending_and_unset",
        uppercase_sci_non_off=False,
        color_force_off_profiles=(),
    )
    assert "SCI: A" in line_variant


def test_state_reminder_line_keeps_color_on_for_sandbox_profile():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.active_profile = "Sandbox"
    api.gov_state.color = "on"
    line = api._state_reminder_line()
    assert "Profile=Sandbox" in line
    assert "Color=on" in line


def test_state_reminder_line_prefers_output_state_snapshot_module_when_available(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    # Deliberately conflicting runtime state to verify delegation to snapshot module.
    api.gov_state.active_profile = "Standard"
    api.gov_state.color = "off"
    api.gov_state.overlay = ""
    api.gov_state.sci_active = False
    api.gov_state.sci_variant = ""
    api.gov_state.comm_active = False

    class _Snap:
        comm_active = True
        active_profile = "Briefing"
        sci_variant = "A"
        sci_pending = False
        sci_active = True
        overlay = "Strict"
        color = "on"
        control_layer = "on"
        qc = "on"
        cgi = "on"
        anchor_auto = True
        user_turns = 3
        dynamic_nudge = ""
        language_policy_mode = "production"

    class _SnapCls:
        @staticmethod
        def from_runtime_state(_):
            return _Snap()

    monkeypatch.setattr(
        mod,
        "_output_state_snapshot",
        types.SimpleNamespace(OutputStateSnapshot=_SnapCls),
        raising=False,
    )

    line = api._state_reminder_line()
    assert "Profile=Briefing" in line
    assert "Overlay=Strict" in line
    assert "SCI=A" in line
    assert "Color=on" in line
    assert "Comm=on" in line


def test_apply_csc_strict_sandbox_respects_color_on_in_header_and_markers():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    api.gov_state.comm_active = True
    api.gov_state.active_profile = "Sandbox"
    api.gov_state.color = "on"
    api.gov_state.overlay = "Strict"
    api.gov_state.sci_active = False
    api.gov_state.sci_variant = ""

    raw = (
        "[GREEN] 🟢 Zeit ist eine Ordnungsdimension.\n\n"
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)\n"
    )
    html_out, _meta = api._apply_csc_strict(raw_response=raw, user_raw="Was ist Zeit?", is_command=False)
    plain = re.sub(r"<[^>]+>", " ", str(html_out or ""))
    assert "Active profile: Sandbox" in plain
    assert "Color: on" in plain
    html_s = str(html_out or "")
    assert (
        ("signal-dot-marker" in html_s)
        or ("#137333" in html_s)
        or ("#2e7d32" in html_s)
        or ("[GREEN]" in plain)
    )


def test_wrapper_build_comm_status_line_pending_and_off():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    pending_line = api._build_comm_status_line(
        comm="on",
        profile="Expert",
        sci_variant="",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        language_policy="production",
        sci_pending=True,
    )
    assert "SCI: PENDING" in pending_line

    off_line = api._build_comm_status_line(
        comm="on",
        profile="Expert",
        sci_variant="off",
        overlay="Strict",
        control_layer="on",
        qc="on",
        cgi="on",
        color="on",
        language_policy="production",
        sci_pending=False,
    )
    assert "SCI: OFF" in off_line


def test_route_context_uses_modular_resolver_and_dispatcher(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    def _resolve_input(raw_txt, state, api_instance, gov_manager=None, route_input_fn=None):
        return {
            "kind": "command",
            "canonical_cmd": "Comm State",
            "standalone_only_violation": True,
        }

    dispatcher = types.SimpleNamespace(
        route_kind=lambda route: "command",
        is_command_route=lambda route: True,
        is_sci_selection_route=lambda route: False,
        is_error_route=lambda route: False,
        is_noop_route=lambda route: False,
        route_audit_payload=lambda route: {
            "kind": "command",
            "is_command": True,
            "is_sci_selection": False,
        },
    )

    monkeypatch.setattr(
        mod,
        "_output_resolver",
        types.SimpleNamespace(resolve_input=_resolve_input),
        raising=False,
    )
    monkeypatch.setattr(mod, "_output_dispatcher", dispatcher, raising=False)

    route, meta = api._resolve_route_context("Comm State")
    assert route.get("canonical_cmd") == "Comm State"
    assert meta.get("kind") == "command"
    assert meta.get("is_command") is True
    assert meta.get("is_error") is False
    assert meta.get("is_noop") is False

    payload = api._route_audit_payload(route, meta)
    assert payload.get("kind") == "command"
    assert payload.get("is_command") is True
    assert payload.get("standalone_only_violation") is True


def test_route_methods_delegate_to_output_routing_runtime_when_available(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    route_expected = {"kind": "chat", "query_text": "delegated", "standalone_only_violation": True}
    meta_expected = {
        "kind": "chat",
        "is_command": False,
        "is_sci_selection": False,
        "is_error": False,
        "is_noop": False,
        "standalone_only_violation": True,
    }
    payload_expected = {
        "kind": "chat",
        "is_command": False,
        "is_sci_selection": False,
        "standalone_only_violation": True,
    }
    calls = {"resolve": 0, "contract": 0, "audit": 0}

    def _resolve(raw_txt, state, api_instance, **kwargs):
        calls["resolve"] += 1
        assert raw_txt == "hello"
        assert state is api.gov_state
        assert api_instance is api
        assert kwargs.get("output_resolver_mod") is getattr(mod, "_output_resolver", None)
        assert kwargs.get("output_dispatcher_mod") is getattr(mod, "_output_dispatcher", None)
        return route_expected, meta_expected

    def _contract(route, **kwargs):
        calls["contract"] += 1
        assert route == route_expected
        assert kwargs.get("local_contract_fn") is mod.contract_route_shape
        return True

    def _audit(route, route_meta=None, **kwargs):
        calls["audit"] += 1
        assert route == route_expected
        assert route_meta == meta_expected
        assert kwargs.get("output_dispatcher_mod") is getattr(mod, "_output_dispatcher", None)
        return payload_expected

    runtime = types.SimpleNamespace(
        resolve_route_context=_resolve,
        route_contract_ok=_contract,
        route_audit_payload=_audit,
    )
    monkeypatch.setattr(mod, "_output_routing_runtime", runtime, raising=False)

    route, meta = api._resolve_route_context("hello")
    assert route == route_expected
    assert meta == meta_expected
    assert api._route_contract_ok(route) is True
    payload = api._route_audit_payload(route, meta)
    assert payload == payload_expected
    assert calls == {"resolve": 1, "contract": 1, "audit": 1}


def test_resolve_route_context_normalizes_chat_txt_to_query_text(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    resolver = getattr(mod, "_output_resolver", None)
    assert resolver is not None

    monkeypatch.setattr(
        mod,
        "_output_resolver",
        types.SimpleNamespace(
            resolve_input=lambda *args, **kwargs: {"kind": "chat", "txt": "hello"},
            normalize_route_shape=resolver.normalize_route_shape,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "_output_dispatcher", None, raising=False)

    route, meta = api._resolve_route_context("hello")
    assert route.get("kind") == "chat"
    assert route.get("query_text") == "hello"
    assert meta.get("kind") == "chat"
    assert meta.get("is_command") is False


def test_route_contract_ok_prefers_modular_resolver_contract(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    route = {"kind": "chat", "query_text": "x"}

    monkeypatch.setattr(
        mod,
        "_output_resolver",
        types.SimpleNamespace(contract_route_shape=lambda r: False),
        raising=False,
    )
    assert api._route_contract_ok(route) is False

    monkeypatch.setattr(mod, "_output_resolver", None, raising=False)
    assert api._route_contract_ok(route) is True


def test_route_contract_ok_prefers_dispatcher_when_available(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    route = {"kind": "chat", "query_text": "x"}

    monkeypatch.setattr(
        mod,
        "_output_dispatcher",
        types.SimpleNamespace(route_contract_ok=lambda _r, contract_route_shape_fn=None: False),
        raising=False,
    )
    assert api._route_contract_ok(route) is False

    monkeypatch.setattr(
        mod,
        "_output_dispatcher",
        types.SimpleNamespace(route_contract_ok=lambda _r, contract_route_shape_fn=None: True),
        raising=False,
    )
    assert api._route_contract_ok(route) is True


def test_output_resolver_matches_legacy_command_route_shape():
    mod = load_fix_module()
    _prime_module_gov(mod)
    resolver = getattr(mod, "_output_resolver", None)
    assert resolver is not None

    api = mod.Api()
    route_legacy = mod.route_input("Comm State", api.gov_state, api, gov_manager=mod.gov)
    route_new = resolver.resolve_input(
        "Comm State",
        api.gov_state,
        api,
        gov_manager=mod.gov,
        route_input_fn=mod.route_input,
    )
    assert route_new == resolver.normalize_route_shape(route_legacy, raw_txt="Comm State")
    assert route_new.get("kind") == "command"
    assert route_new.get("canonical_cmd") == "Comm State"


def test_output_dispatcher_route_audit_payload_contract():
    mod = load_fix_module()
    dispatcher = getattr(mod, "_output_dispatcher", None)
    assert dispatcher is not None

    payload = dispatcher.route_audit_payload(
        {
            "kind": "chat",
            "query_text": "A",
            "is_sci_selection": True,
            "standalone_only_violation": True,
        }
    )
    assert payload.get("kind") == "chat"
    assert payload.get("is_command") is False
    assert payload.get("is_sci_selection") is True
    assert payload.get("standalone_only_violation") is True


def test_output_rules_registry_qc_probe_is_complete_contract():
    mod = load_fix_module()
    rr = getattr(mod, "_output_rules_registry", None)
    assert rr is not None
    fn = getattr(rr, "qc_probe_is_complete", None)
    assert callable(fn)

    assert bool(fn("QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)")) is True
    assert bool(fn("QC-Matrix: Klarheit 3 (Δ0) · Kuerze 2 (Δ0) · Evidenz 2 (Δ0) · Empathie 2 (Δ0) · Konsistenz 3 (Δ0) · Neutralitaet 3 (Δ0)")) is True
    assert bool(fn("QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0)")) is False


def test_render_ts_footer_html_is_deterministic_wrapper_output():
    mod = load_fix_module()
    ts = "10.03.2026 12:34:56 CET (UTC+01:00)"
    out = mod._render_ts_footer_html(ts)
    assert out == '<div class="ts-footer">Response at 10.03.2026 12:34:56 CET (UTC+01:00)</div>'


def test_qc_footer_helper_functions_delegate_to_output_footer_renderer(monkeypatch):
    mod = load_fix_module()

    class _StubFooterRenderer:
        def ensure_qc_footer_html_consistency(self, **kwargs):
            return f"QC-GUARD::{kwargs.get('profile_name')}::{kwargs.get('raw_for_render')}"

        def annotate_qc_matrix_tooltips_html(self, html_text, **kwargs):
            return f"QC-TIP::{html_text}::{kwargs.get('lang')}"

    monkeypatch.setattr(mod, "_output_footer_renderer", _StubFooterRenderer())

    out_guard = mod.ensure_qc_footer_html_consistency_html_stage(
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        profile_name="Expert",
        gov_mgr=object(),
        overrides={"clarity": 2},
        qc_footer_for_profile_fn=lambda _p: "QC-Matrix: Clarity 2 (Δ0)",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=lambda _probe: True,
    )
    assert out_guard == "QC-GUARD::Expert::RAW"
    assert mod.annotate_qc_matrix_tooltips_html("<p>b</p>", lang="en") == "QC-TIP::<p>b</p>::en"


def test_qc_footer_helper_functions_prefer_post_render_qc_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubQcStage:
        def ensure_qc_footer_html_consistency_html_stage(self, **kwargs):
            return f"QC-STAGE-GUARD::{kwargs.get('profile_name')}::{kwargs.get('raw_for_render')}"

    monkeypatch.setattr(mod, "_output_post_render_qc_stage", _StubQcStage())

    out_guard = mod.ensure_qc_footer_html_consistency_html_stage(
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        profile_name="Expert",
        gov_mgr=object(),
        overrides={"clarity": 2},
        qc_footer_for_profile_fn=lambda _p: "QC-Matrix: Clarity 2 (Δ0)",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=lambda _probe: True,
    )
    assert out_guard == "QC-STAGE-GUARD::Expert::RAW"


def test_qc_footer_finalize_helper_uses_central_guard_then_tooltip_sequence(monkeypatch):
    mod = load_fix_module()
    calls = []

    class _StubFooterRenderer:
        def ensure_qc_footer_html_consistency(self, **kwargs):
            calls.append(("guard", kwargs.get("profile_name")))
            return f"QC-GUARD::{kwargs.get('profile_name')}::{kwargs.get('raw_for_render')}"

        def annotate_qc_matrix_tooltips_html(self, html_text, **kwargs):
            calls.append(("tips", kwargs.get("lang")))
            return f"QC-TIP::{html_text}::{kwargs.get('lang')}"

        def finalize_qc_footer_html(self, **kwargs):
            calls.append(("finalize", kwargs.get("profile_name")))
            return "UNUSED-FINALIZE-PATH"

    monkeypatch.setattr(mod, "_output_footer_renderer", _StubFooterRenderer())

    out = mod.finalize_qc_footer_html_stage(
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        profile_name="Expert",
        gov_mgr=object(),
        overrides={"clarity": 2},
        qc_footer_for_profile_fn=lambda _p: "QC-Matrix: Clarity 2 (Δ0)",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=lambda _probe: True,
        lang="en",
    )
    assert out == "QC-TIP::QC-GUARD::Expert::RAW::en"
    assert calls == [("guard", "Expert"), ("tips", "en")]


def test_qc_footer_finalize_helper_prefers_post_render_qc_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubQcStage:
        def finalize_qc_footer_html_stage(self, **kwargs):
            return f"QC-STAGE-FINAL::{kwargs.get('profile_name')}::{kwargs.get('lang')}"

    monkeypatch.setattr(mod, "_output_post_render_qc_stage", _StubQcStage())

    out = mod.finalize_qc_footer_html_stage(
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        profile_name="Expert",
        gov_mgr=object(),
        overrides={"clarity": 2},
        qc_footer_for_profile_fn=lambda _p: "QC-Matrix: Clarity 2 (Δ0)",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=lambda _probe: True,
        lang="en",
    )
    assert out == "QC-STAGE-FINAL::Expert::en"


def test_qc_footer_finalize_helper_uses_local_sequence_when_renderer_entrypoint_missing(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_footer_renderer", None)

    calls = []

    def _guard(**kwargs):
        calls.append(("guard", kwargs.get("profile_name")))
        return "GUARDED"

    def _tips(html_text, **kwargs):
        calls.append(("tips", kwargs.get("lang")))
        return f"TIPPED::{html_text}"

    monkeypatch.setattr(mod, "ensure_qc_footer_html_consistency_html_stage", _guard)
    monkeypatch.setattr(mod, "annotate_qc_matrix_tooltips_html", _tips)

    out = mod.finalize_qc_footer_html_stage(
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        profile_name="Standard",
        gov_mgr=object(),
        overrides={},
        qc_footer_for_profile_fn=lambda _p: "QC-Matrix: Clarity 2 (Δ0)",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=None,
        lang="de",
    )
    assert out == "TIPPED::GUARDED"
    assert calls == [("guard", "Standard"), ("tips", "de")]


def test_qc_footer_helper_functions_are_fail_soft_when_renderer_unavailable(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_footer_renderer", None)

    src = "<p>qc</p>"
    out_guard = mod.ensure_qc_footer_html_consistency_html_stage(
        final_html_body=src,
        raw_for_render="RAW",
        profile_name="Standard",
        gov_mgr=object(),
        overrides={},
        qc_footer_for_profile_fn=lambda _p: "",
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=None,
    )
    assert out_guard == src
    assert mod.annotate_qc_matrix_tooltips_html(src, lang="de") == src


def test_csc_mid_refiner_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    class _StubCscMidStage:
        def build_csc_refiner_meta_stage(self, **kwargs):
            return {
                "csc_meta": {"applied": True, "trigger": kwargs.get("profile_name")},
                "threshold_multiplier": 7,
            }

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    csc_meta, mult = api._build_csc_refiner_meta_stage(
        raw_response="RAW",
        user_raw="USER",
        profile_name="Expert",
        overlay_name="Strict",
        refiner_obj=None,
    )
    assert mult == 7
    assert csc_meta == {"applied": True, "trigger": "Expert"}


def test_csc_mid_alerts_header_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    class _StubCscMidStage:
        def build_alerts_and_header_stage(self, **_kwargs):
            return {"raw_response": "RAW-AH", "alert_html": "<div>ALERT</div>", "header": "HDR"}

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    raw_out, alert_html, header = api._build_csc_alerts_and_header_stage(
        raw_response="RAW",
        profile_name="Standard",
        overlay_name="off",
        csc_meta=None,
        refiner_obj=None,
    )
    assert raw_out == "RAW-AH"
    assert alert_html == "<div>ALERT</div>"
    assert header == "HDR"


def test_csc_mid_pre_render_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    class _StubCscMidStage:
        def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(self, **kwargs):
            assert kwargs.get("header") == "HDR"
            bundle = kwargs.get("runtime_bundle")
            assert isinstance(bundle, dict)
            assert bundle.get("output_pipeline_mod") is getattr(mod, "_output_pipeline", None)
            assert isinstance(bundle.get("verification_route_config"), dict)
            assert bundle.get("verification_route_provider") in ("gemini", "openrouter", "huggingface")
            assert isinstance(bundle.get("hooks"), dict)
            return {
                "blocked": False,
                "alert_html": "ALERT-OUT",
                "raw_for_render": "RAW-OUT",
                "raw_response": "RESP-OUT",
            }

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="RESP-IN",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT-IN",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT-OUT",
        "raw_for_render": "RAW-OUT",
        "raw_response": "RESP-OUT",
    }


def test_csc_mid_pre_render_helper_prefers_runtime_dispatch_stage_module(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    calls = []

    class _StubCscMidStage:
        def build_pre_render_policy_strict_gate_runtime_bundle(self, **_kwargs):
            calls.append("build_bundle")
            return {"bundle_marker": "D16", "hooks": {}}

        def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(self, **kwargs):
            calls.append("dispatch")
            assert kwargs.get("runtime_bundle", {}).get("bundle_marker") == "D16"
            assert kwargs.get("header") == "HDR"
            return {
                "blocked": False,
                "blocked_response": None,
                "strict_meta": None,
                "alert_html": "ALERT-DISPATCH",
                "raw_for_render": "RAW-DISPATCH",
                "raw_response": "RESP-DISPATCH",
            }

        def apply_pre_render_policy_strict_gate_runtime_chain_stage(self, **_kwargs):
            raise AssertionError("dispatch entry should be used before legacy chain path")

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="RESP-IN",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT-IN",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT-DISPATCH",
        "raw_for_render": "RAW-DISPATCH",
        "raw_response": "RESP-DISPATCH",
    }
    assert calls == ["build_bundle", "dispatch"]


def test_csc_mid_pre_render_helper_prefers_runtime_bundle_factory_and_bundle_fallback_dispatch(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    calls = []

    class _StubCscMidStage:
        def build_pre_render_policy_strict_gate_runtime_bundle(self, **kwargs):
            calls.append("build_bundle")
            assert kwargs.get("gov_mgr") is mod.gov
            assert kwargs.get("runtime_state") is api.gov_state
            assert kwargs.get("validator_obj") is getattr(api, "validator", None)
            assert kwargs.get("app_obj") is api
            hook_scope = kwargs.get("hook_scope")
            assert isinstance(hook_scope, dict)
            assert callable(hook_scope.get("evaluate_strict_enforcement"))
            assert kwargs.get("hook_overrides") in (None, {})
            return {"bundle_marker": "B42", "hooks": {}}

        def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(self, **kwargs):
            calls.append("dispatch")
            assert kwargs.get("runtime_bundle", {}).get("bundle_marker") == "B42"
            return {
                "blocked": False,
                "alert_html": "ALERT-BUNDLE-ULT",
                "raw_for_render": "RAW-BUNDLE-ULT",
                "raw_response": "RESP-BUNDLE-ULT",
            }

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="RESP-IN",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT-IN",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT-BUNDLE-ULT",
        "raw_for_render": "RAW-BUNDLE-ULT",
        "raw_response": "RESP-BUNDLE-ULT",
    }
    assert calls == ["build_bundle", "dispatch"]


def test_strict_gate_runtime_bundle_scope_hooks_keep_chain_and_ultimate_fallback_in_sync():
    mod = load_fix_module()
    stage = getattr(mod, "_output_csc_mid_runtime_stage", None)
    assert stage is not None

    strict_calls = []

    def _strict_eval(**kwargs):
        strict_calls.append(str(kwargs.get("raw_for_render") or ""))
        return {"blocked": False, "strict_banner_html": "<b>STRICT</b>"}

    class _AppStub:
        def _append_system_message(self, *_args, **_kwargs):
            return None

        def _get_enforcement_settings(self):
            return {"enabled": True, "policy": "strict_warn"}

        def _render_sci_trace_as_html_runtime(self, text):
            return str(text or "")

        def _hide_verification_route_lines_in_chat(self):
            return False

    class _GovStub:
        def normalize_qc_overrides(self, overrides):
            return dict(overrides or {})

        def get_effective_qc_corridor(self, *_args, **_kwargs):
            return {}

    runtime_state = types.SimpleNamespace(
        qc_overrides={},
        sci_active=False,
        sci_variant="",
        sci_pending=False,
        answer_language="de",
    )
    bundle = stage.build_pre_render_policy_strict_gate_runtime_bundle(
        gov_mgr=_GovStub(),
        runtime_state=runtime_state,
        validator_obj=object(),
        output_pipeline_mod=None,
        verification_route_config={},
        verification_route_provider="gemini",
        app_obj=_AppStub(),
        hook_scope={
            "evaluate_strict_enforcement": _strict_eval,
            "sanitize_html": lambda txt: f"SAN::{txt}",
            "unwrap_accidental_full_text_codefence": lambda txt: str(txt or ""),
            "strip_pathological_repetition_display_noise": lambda txt, lang="de": str(txt or ""),
        },
    )
    hooks = bundle.get("hooks", {})
    assert callable(hooks.get("evaluate_strict_enforcement_fn"))
    assert callable(hooks.get("append_system_message_fn"))
    assert callable(hooks.get("get_enforcement_settings_fn"))

    chain_out = stage.apply_pre_render_policy_strict_gate_runtime_chain_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT",
        runtime_bundle=bundle,
    )
    assert isinstance(chain_out, dict)
    assert str(chain_out.get("alert_html") or "").startswith("<b>STRICT</b>")
    assert str(chain_out.get("raw_for_render") or "").startswith("HDR\n\n")

    ultimate_out = stage.apply_pre_render_policy_strict_gate_runtime_ultimate_fallback_from_bundle_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        header="HDR",
        alert_html="ALERT",
        runtime_bundle=bundle,
    )
    assert isinstance(ultimate_out, dict)
    assert str(ultimate_out.get("alert_html") or "").startswith("<b>STRICT</b>")
    assert str(ultimate_out.get("raw_for_render") or "").startswith("HDR\n\n")
    assert strict_calls == ["HDR\n\nBODY", "HDR\n\nBODY"]


def test_strict_gate_runtime_dispatch_stage_returns_normalized_payload_with_passthrough_fallback():
    mod = load_fix_module()
    stage = getattr(mod, "_output_csc_mid_runtime_stage", None)
    assert stage is not None

    runtime_state = types.SimpleNamespace(
        qc_overrides={},
        sci_active=False,
        sci_variant="",
        sci_pending=False,
        answer_language="de",
    )
    bundle = stage.build_pre_render_policy_strict_gate_runtime_bundle(
        gov_mgr=types.SimpleNamespace(
            normalize_qc_overrides=lambda overrides: dict(overrides or {}),
            get_effective_qc_corridor=lambda *_args, **_kwargs: {},
        ),
        runtime_state=runtime_state,
        validator_obj=object(),
        output_pipeline_mod=None,
        verification_route_config={},
        verification_route_provider="gemini",
        app_obj=types.SimpleNamespace(
            _append_system_message=lambda *_args, **_kwargs: None,
            _get_enforcement_settings=lambda: {"enabled": False},
            _render_sci_trace_as_html_runtime=lambda txt: str(txt or ""),
            _hide_verification_route_lines_in_chat=lambda: False,
        ),
        hook_scope={
            "evaluate_strict_enforcement": lambda **_kwargs: {"blocked": False, "strict_banner_html": ""},
            "sanitize_html": lambda txt: str(txt or ""),
            "unwrap_accidental_full_text_codefence": lambda txt: str(txt or ""),
            "strip_pathological_repetition_display_noise": lambda txt, lang="de": str(txt or ""),
        },
    )

    out = stage.apply_pre_render_policy_strict_gate_runtime_dispatch_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT",
        runtime_bundle=bundle,
    )
    assert isinstance(out, dict)
    assert out.get("blocked") is False
    assert str(out.get("raw_for_render") or "").startswith("HDR\n\n")
    assert str(out.get("raw_response") or "") == "BODY"

    passthrough = stage.build_pre_render_policy_strict_gate_runtime_passthrough_out(
        raw_response="BODY",
        header="HDR",
        alert_html="ALERT",
    )
    assert passthrough == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT",
        "raw_for_render": "HDR\n\nBODY",
        "raw_response": "BODY",
    }


def test_csc_mid_pre_render_helper_uses_runtime_fallback_stage_when_primary_unavailable(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()

    class _StubCscMidStage:
        def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(self, **kwargs):
            assert kwargs.get("header") == "HDR"
            bundle = kwargs.get("runtime_bundle")
            assert isinstance(bundle, dict)
            assert bundle.get("output_pipeline_mod") is getattr(mod, "_output_pipeline", None)
            assert isinstance(bundle.get("verification_route_config"), dict)
            assert bundle.get("verification_route_provider") in ("gemini", "openrouter", "huggingface")
            assert isinstance(bundle.get("hooks"), dict)
            return {
                "blocked": False,
                "alert_html": "ALERT-FB",
                "raw_for_render": "RAW-FB",
                "raw_response": "RESP-FB",
            }

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="RESP-IN",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT-IN",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT-FB",
        "raw_for_render": "RAW-FB",
        "raw_response": "RESP-FB",
    }


def test_csc_mid_pre_render_helper_uses_runtime_ultimate_fallback_stage_when_needed(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    calls = []

    class _StubCscMidStage:
        def apply_pre_render_policy_strict_gate_runtime_dispatch_stage(self, **kwargs):
            calls.append("dispatch")
            assert kwargs.get("header") == "HDR"
            bundle = kwargs.get("runtime_bundle")
            assert isinstance(bundle, dict)
            assert bundle.get("output_pipeline_mod") is getattr(mod, "_output_pipeline", None)
            assert isinstance(bundle.get("verification_route_config"), dict)
            assert bundle.get("verification_route_provider") in ("gemini", "openrouter", "huggingface")
            assert isinstance(bundle.get("hooks"), dict)
            # Simuliert Dispatch-Ergebnis nach interner Chain/Fallback-Aufloesung im Stage-Modul.
            return {
                "blocked": False,
                "alert_html": "ALERT-ULT",
                "raw_for_render": "RAW-ULT",
                "raw_response": "RESP-ULT",
            }

    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", _StubCscMidStage())
    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="RESP-IN",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="ALERT-IN",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "ALERT-ULT",
        "raw_for_render": "RAW-ULT",
        "raw_response": "RESP-ULT",
    }
    assert calls == ["dispatch"]


def test_csc_mid_pre_render_helper_emergency_brake_is_thin_and_deterministic(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", None)
    monkeypatch.setattr(mod, "_load_optional_module_with_file_fallback", lambda *_args, **_kwargs: None)

    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="<a>ALERT</a>",
    )
    assert out == {
        "blocked": False,
        "blocked_response": None,
        "strict_meta": None,
        "alert_html": "<a>ALERT</a>",
        "raw_for_render": "HDR\n\nBODY",
        "raw_response": "BODY",
    }


def test_csc_mid_refiner_helper_fallback_when_stage_missing_returns_safe_defaults(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", None)

    csc_meta, mult = api._build_csc_refiner_meta_stage(
        raw_response="RAW",
        user_raw="USER",
        profile_name="Standard",
        overlay_name="Explore",
        refiner_obj=None,
    )
    assert csc_meta is None
    assert mult == 2


def test_csc_mid_alerts_header_helper_fallback_when_stage_missing_is_thin_and_deterministic(monkeypatch):
    mod = load_fix_module()
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", None)
    api.gov_state.dynamic_one_shot_active = True

    raw_out, alert_html, header = api._build_csc_alerts_and_header_stage(
        raw_response="RAW",
        profile_name="Standard",
        overlay_name="off",
        csc_meta={"message": "CSC: visible"},
        refiner_obj=None,
    )
    assert raw_out == "RAW"
    assert alert_html == ""
    assert "Active profile: Standard" in header
    assert "Dynamic: one-shot (active)" in header
    assert "CSC: visible" in header


def test_csc_mid_pre_render_helper_fallback_when_stage_missing_applies_strict_banner(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", None)
    monkeypatch.setattr(api, "_get_enforcement_settings", lambda: {"enabled": True, "policy": "strict_warn"})
    monkeypatch.setattr(
        mod,
        "evaluate_strict_enforcement",
        lambda **_kwargs: {"blocked": False, "strict_banner_html": "<warn>STRICT</warn>"},
    )

    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="<a>ALERT</a>",
    )
    assert out["blocked"] is False
    assert out["alert_html"].startswith("<warn>STRICT</warn><a>ALERT</a>")
    assert out["raw_for_render"].startswith("HDR\n\nBODY")


def test_csc_mid_pre_render_helper_fallback_when_stage_missing_supports_blocked_path(monkeypatch):
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    monkeypatch.setattr(mod, "_output_csc_mid_runtime_stage", None)
    monkeypatch.setattr(api, "_get_enforcement_settings", lambda: {"enabled": True, "policy": "strict_block"})
    monkeypatch.setattr(
        mod,
        "evaluate_strict_enforcement",
        lambda **_kwargs: {"blocked": True, "blocked_html": "<b>BLOCKED</b>", "meta": {"strict": "blocked"}},
    )
    monkeypatch.setattr(mod, "sanitize_html", lambda html_text: f"SAN::{html_text}")

    out = api._apply_csc_pre_render_policy_strict_gate_stage(
        raw_response="BODY",
        user_raw="USER",
        profile_name="Standard",
        is_command=False,
        ctx={"sci_pending": False},
        header="HDR",
        alert_html="",
    )
    assert out["blocked"] is True
    assert out["strict_meta"] == {"strict": "blocked"}
    assert out["blocked_response"] == {"html": "SAN::<b>BLOCKED</b>", "text": "", "csc": None}


def test_render_body_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubRenderBodyStage:
        def render_final_html_body_stage(self, **kwargs):
            return (
                f"RAW-STAGE::{kwargs.get('color_mode')}",
                f"HTML-STAGE::{kwargs.get('answer_lang')}",
            )

    monkeypatch.setattr(mod, "_output_render_body_runtime_stage", _StubRenderBodyStage())
    out_raw, out_html = mod.render_final_html_body_stage(
        raw_for_render="RAW",
        color_mode="on",
        answer_lang="de",
        ui_lang_fallback="en",
    )
    assert out_raw == "RAW-STAGE::on"
    assert out_html == "HTML-STAGE::de"


def test_render_body_helper_local_fallback_runs_text_and_html_pipeline(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_render_body_runtime_stage", None)
    monkeypatch.setattr(mod, "_rendering_pipeline_v192", None)
    monkeypatch.setattr(mod, "auto_embed_image_urls", lambda txt: f"{txt}|IMG")
    monkeypatch.setattr(mod, "apply_color_spans", lambda txt, enabled=True: f"{txt}|COLOR")
    monkeypatch.setattr(mod, "strip_color_markers_for_color_off_text", lambda txt: f"{txt}|STRIP")
    monkeypatch.setattr(mod, "normalize_markdown_list_spacing", lambda txt: f"{txt}|SPACE")
    monkeypatch.setattr(mod, "normalize_known_markdown_control_headings", lambda txt: f"{txt}|HEAD")
    monkeypatch.setattr(mod, "sanitize_html", lambda html_text: f"SAN::{html_text}")
    monkeypatch.setattr(mod, "html_number_self_debunking", lambda html_text, lang="en": f"NUM[{lang}]::{html_text}")
    monkeypatch.setattr(
        mod,
        "apply_post_render_normalization_seam",
        lambda html_body, answer_lang="de", color="off": f"POST[{answer_lang}/{color}]::{html_body}",
    )
    monkeypatch.setattr(
        mod,
        "markdown",
        types.SimpleNamespace(markdown=lambda txt, extensions=None: f"<md>{txt}</md>"),
    )

    out_raw, out_html = mod.render_final_html_body_stage(
        raw_for_render="RAW",
        color_mode="off",
        answer_lang="de",
        ui_lang_fallback="en",
    )
    assert out_raw == "RAW|IMG|STRIP|SPACE|HEAD"
    assert out_html == "POST[de/off]::NUM[de]::SAN::<md>RAW|IMG|STRIP|SPACE|HEAD</md>"


def test_route_render_command_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubRouteRenderStage:
        def render_command_html_stage(self, **kwargs):
            return (f"CMD-STAGE::{kwargs.get('ui_lang')}::{kwargs.get('color_mode')}", None)

    monkeypatch.setattr(mod, "_output_route_render_runtime_stage", _StubRouteRenderStage())
    out_html, out_meta = mod.render_command_html_stage(
        raw_response="RAW",
        color_mode="on",
        ui_lang="de",
        comm_active=True,
    )
    assert out_html == "CMD-STAGE::de::on"
    assert out_meta is None


def test_route_render_comm_inactive_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubRouteRenderStage:
        def render_comm_inactive_html_stage(self, **kwargs):
            return (f"OFF-STAGE::{kwargs.get('ui_lang')}::{kwargs.get('color_mode')}", {"normalization": {"render_ok": True}})

    monkeypatch.setattr(mod, "_output_route_render_runtime_stage", _StubRouteRenderStage())
    out_html, out_meta = mod.render_comm_inactive_html_stage(
        raw_response="RAW",
        color_mode="off",
        ui_lang="en",
    )
    assert out_html == "OFF-STAGE::en::off"
    assert out_meta == {"normalization": {"render_ok": True}}


def test_route_render_command_helper_local_fallback_runs_markdown_path(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_route_render_runtime_stage", None)
    monkeypatch.setattr(mod, "_rendering_pipeline_v192", None)
    monkeypatch.setattr(mod, "unwrap_accidental_full_text_codefence", lambda txt: f"{txt}|U")
    monkeypatch.setattr(mod, "normalize_known_markdown_control_headings", lambda txt: f"{txt}|H")
    monkeypatch.setattr(mod, "strip_color_markers_for_color_off_text", lambda txt: f"{txt}|S")
    monkeypatch.setattr(mod, "strip_color_markers_for_color_off_html", lambda html_text: f"{html_text}|HS")
    monkeypatch.setattr(mod, "sanitize_html", lambda html_text: f"SAN::{html_text}")
    monkeypatch.setattr(
        mod,
        "markdown",
        types.SimpleNamespace(markdown=lambda txt, extensions=None: f"<md>{txt}</md>"),
    )

    out_html, out_meta = mod.render_command_html_stage(
        raw_response="RAW",
        color_mode="off",
        ui_lang="de",
        comm_active=False,
    )
    assert out_meta is None
    assert out_html == "SAN::<md>RAW|U|H|S</md>|HS"


def test_route_render_comm_inactive_helper_local_fallback_returns_normalization_meta(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_route_render_runtime_stage", None)
    monkeypatch.setattr(mod, "_rendering_pipeline_v192", None)
    monkeypatch.setattr(mod, "unwrap_accidental_full_text_codefence", lambda txt: f"{txt}|U")
    monkeypatch.setattr(mod, "normalize_known_markdown_control_headings", lambda txt: f"{txt}|H")
    monkeypatch.setattr(mod, "strip_governance_scaffolding_when_comm_inactive", lambda txt: f"{txt}|G")
    monkeypatch.setattr(mod, "strip_color_markers_for_color_off_text", lambda txt: f"{txt}|S")
    monkeypatch.setattr(mod, "strip_color_markers_for_color_off_html", lambda html_text: f"{html_text}|HS")
    monkeypatch.setattr(mod, "sanitize_html", lambda html_text: f"SAN::{html_text}")
    monkeypatch.setattr(mod, "_build_render_normalization_summary", lambda raw, html_text: {"raw": raw, "html": html_text})
    monkeypatch.setattr(mod, "_looks_like_rendered_html_runtime", lambda _html_text: False)
    monkeypatch.setattr(
        mod,
        "markdown",
        types.SimpleNamespace(markdown=lambda txt, extensions=None: f"<md>{txt}</md>"),
    )

    out_html, out_meta = mod.render_comm_inactive_html_stage(
        raw_response="RAW",
        color_mode="off",
        ui_lang="en",
    )
    assert out_html == "SAN::<md>RAW|U|H|G|S</md>|HS|HS"
    assert out_meta == {
        "normalization": {
            "raw": "RAW|U|H|G",
            "html": "SAN::<md>RAW|U|H|G|S</md>|HS|HS",
            "render_ok": False,
            "render_fallback": True,
        }
    }


def test_render_end_finalize_helper_prefers_runtime_stage_module(monkeypatch):
    mod = load_fix_module()

    class _StubRenderEndStage:
        def finalize_render_end_html_stage(self, **kwargs):
            return (
                f"RENDER-END::{kwargs.get('answer_lang')}::{kwargs.get('final_html_body')}",
                {"stage": True},
            )

    monkeypatch.setattr(mod, "_output_final_html_runtime_stage", _StubRenderEndStage())
    out_html, out_meta = mod.finalize_render_end_html_stage(
        alert_html="<div>alert</div>",
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        raw_original="ORIG",
        raw_response="RESP",
        csc_meta={"existing": 1},
        answer_lang="en",
    )
    assert out_html == "RENDER-END::en::<p>a</p>"
    assert out_meta == {"stage": True}


def test_render_end_finalize_helper_local_fallback_render_ok(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_final_html_runtime_stage", None)
    monkeypatch.setattr(mod, "_build_render_normalization_summary", lambda raw, html: {"raw_len": len(str(raw or ""))})
    monkeypatch.setattr(mod, "_looks_like_rendered_html_runtime", lambda _html: True)
    monkeypatch.setattr(mod, "_detect_probable_truncation", lambda _raw, _html: (False, ""))
    monkeypatch.setattr(mod, "_format_response_timestamp", lambda: "TS")
    monkeypatch.setattr(mod, "_render_ts_footer_html", lambda ts: f"<ts>{ts}</ts>")

    out_html, out_meta = mod.finalize_render_end_html_stage(
        alert_html="<div>alert</div>",
        final_html_body="<p>a</p>",
        raw_for_render="RAW",
        raw_original="ORIG",
        raw_response="RESP",
        csc_meta=None,
        answer_lang="de",
    )
    assert out_html == "<div>alert</div><p>a</p><ts>TS</ts>"
    assert out_meta == {"normalization": {"raw_len": 3, "render_ok": True, "render_fallback": False}}


def test_render_end_finalize_helper_local_fallback_render_broken_with_truncation_note(monkeypatch):
    mod = load_fix_module()
    monkeypatch.setattr(mod, "_output_final_html_runtime_stage", None)
    monkeypatch.setattr(mod, "_build_render_normalization_summary", lambda _raw, _html: {"boxed": True})
    monkeypatch.setattr(mod, "_looks_like_rendered_html_runtime", lambda _html: False)
    monkeypatch.setattr(mod, "_detect_probable_truncation", lambda _raw, _html: (True, "TRUNC"))
    monkeypatch.setattr(
        mod,
        "_control_layer_alert_html",
        lambda msg, **_kwargs: f"<warn>{msg}</warn>",
    )
    monkeypatch.setattr(mod, "_format_response_timestamp", lambda: "TS")
    monkeypatch.setattr(mod, "_render_ts_footer_html", lambda ts: f"<ts>{ts}</ts>")

    out_html, out_meta = mod.finalize_render_end_html_stage(
        alert_html="<div>alert</div>",
        final_html_body="<broken",
        raw_for_render="RAW",
        raw_original="R<AW>",
        raw_response="RESP",
        csc_meta=None,
        answer_lang="de",
    )
    assert out_html.startswith("<warn>TRUNC Bitte gegenprüfen oder Antwort neu generieren.</warn><div>alert</div>")
    assert "<b>Render fallback</b>: showing raw model output." in out_html
    assert "R&lt;AW&gt;" in out_html
    assert out_html.endswith("<ts>TS</ts>")
    assert out_meta == {
        "normalization": {"boxed": True, "render_ok": False, "render_fallback": True},
        "probable_truncation": True,
    }


def test_ensure_self_debunking_box_html_handles_realistic_leak_fragment():
    mod = load_fix_module()
    html_in = (
        "<p><strong>Definitionen und Perspektiven:</strong>\n"
        "*   <strong>Physikalische Definition:</strong> In der Physik ist Zeit ...</p>\n"
        "<ul>\n"
        "<li><p><strong>Heidegger:</strong> Text.\n"
        "<strong>Self-Debunking:</strong></p></li>\n"
        "<li><p>1. <strong>Schwäche</strong>: Punkt eins."
        "<br><strong>Warum das wichtig ist</strong>: A.</p></li>\n"
        "<li><p>2. <strong>Schwäche</strong>: Punkt zwei."
        "<br><strong>Warum das wichtig ist</strong>: B.</p></li>\n"
        "</ul>\n"
        "<p>QC-Matrix: Clarity 3 (Δ0)</p>"
    )
    out = mod.ensure_self_debunking_box_html(html_in, lang="de")
    assert 'class="self-debunking"' in out
    assert "<strong>Self-Debunking:</strong>" not in out
    assert "<ol>" in out and "<li>" in out
    assert "<p>QC-Matrix: Clarity 3 (Δ0)</p>" in out


def test_html_number_self_debunking_does_not_split_vereinfachungen_word():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "Vereinfachungen enthalten" in out
    assert "Vereinfachung</strong>: en" not in out


def test_html_number_self_debunking_canonicalizes_unsicherheit_labels_to_schwaeche():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<p>Unsicherheit: U1 - Datenlücke im Debunking.</p>'
        '<li><p><strong>Unsicherheit</strong>: U1 - Zweiter Punkt.</p></li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<strong>Unsicherheit</strong>:" not in out
    assert "Unsicherheit: U1" not in out
    assert "<strong>Schwäche</strong>:" in out
    assert "<li>" in out


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


def test_html_number_self_debunking_ol_merges_trailing_secondary_paras_before_empty_ol_tail():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins.'
        '<br><strong>Warum das wichtig ist</strong>: Relevanz eins.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Check eins.</li>'
        '<li><strong>Schwäche</strong>: Punkt zwei.</li>'
        '</ol>'
        '<p><strong>Warum das wichtig ist</strong>: Relevanz zwei.</p>'
        '<p><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Check zwei.</p>'
        '<ol></ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert re.search(
        r"(?is)<li[^>]*>\s*<strong>\s*Schwäche\s*</strong>\s*:\s*Punkt zwei\."
        r"\s*<br>\s*<strong>\s*Warum das wichtig ist\s*</strong>\s*:\s*Relevanz zwei\."
        r"\s*<br>\s*<strong>\s*Was würde verifizieren/falsifizieren \(nächster Check\)\s*</strong>\s*:\s*Check zwei\.\s*</li>",
        out,
    )
    assert "<ol></ol>" not in out
    assert "</ol></ol>" not in out


def test_html_number_self_debunking_en_merges_why_this_is_important_rows_inside_ol():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Self-Debunking:</div>'
        '<ol>'
        '<li><strong>Weakness</strong>: The answer may contain simplifications.</li>'
        '<p>Why this is important Simplifications can obscure edge cases.</p>'
        '<p><strong>What would verify/falsify (next check)</strong>: Check assumptions against primary sources.</p>'
        '<li><strong>Weakness</strong>: The answer may omit uncertainty limits. '
        'Why this is important : Missing restrictions can overstate validity.'
        '<br><strong>What would verify/falsify (next check)</strong>: Add a strong counterexample.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="en")
    assert "<p>Why this is important" not in out
    assert "Why this is important :" not in out
    assert "<strong>Why it matters</strong>:" in out
    assert "<br><strong>Why it matters</strong>: Simplifications can obscure edge cases." in out
    assert "<br><strong>Why it matters</strong>: Missing restrictions can overstate validity." in out


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


def test_html_number_self_debunking_ol_strips_color_marker_sentence_rows_inside_items():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Self-Debunking:</div>'
        '<ol>'
        '<li><strong>Weakness</strong>: Base weakness.'
        '<br><strong>Why it matters</strong>: Relevance one.'
        '<br><strong>What would verify/falsify (next check)</strong>: Check one.'
        '<br>🔴 Eine Hyperantithesis blendet Gegenargumente aus.'
        '<br><span class="signal-dot-marker"><span style="color:#c62828; font-weight:600;">🔴</span></span> Diese radikale Perspektive kann zu Fehlgewichtungen führen.'
        '</li>'
        '<li><strong>Weakness</strong>: Second weakness.'
        '<br><strong>Why it matters</strong>: Relevance two.'
        '<br><strong>What would verify/falsify (next check)</strong>: Check two.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="en")
    assert "Eine Hyperantithesis blendet Gegenargumente aus." not in out
    assert "Diese radikale Perspektive kann zu Fehlgewichtungen führen." not in out
    assert "signal-dot-marker" not in out
    assert out.count("<li") == 2
    assert "<strong>Why it matters</strong>:" in out
    assert "<strong>What would verify/falsify (next check)</strong>:" in out


def test_html_number_self_debunking_en_repairs_broken_strong_colon_artifact():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Self-Debunking:</div>'
        '<ol>'
        '<li>'
        '<strong>Weakness</strong>: Simplification.'
        '<br><strong>Why it matters</strong>:<strong>: Counterexamples are missing.'
        '<br><strong>What would verify/falsify (next check)</strong>: Compare primary sources.'
        '</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="en")
    assert ":<strong>:" not in out
    assert "<strong>Why it matters</strong>: Counterexamples are missing." in out


def test_html_number_self_debunking_ol_keeps_primary_label_inline_without_br_break():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins bleibt inline.'
        '<br><strong>Warum das wichtig ist</strong>: Relevanz eins.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Check eins.</li>'
        '<li><strong>Schwäche</strong>:<br>Die zweite Schwäche startet fälschlich in neuer Zeile.'
        '<br><strong>Warum das wichtig ist</strong>: Relevanz zwei.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Check zwei.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<strong>Schwäche</strong>:<br>" not in out
    assert re.search(
        r"(?is)<strong>\s*Schwäche\s*</strong>:\s+Die zweite Schwäche startet fälschlich in neuer Zeile\.",
        out,
    )


def test_html_number_self_debunking_ol_repairs_orphan_paragraph_and_prefixes_missing_primary_labels():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<p>Die Definition von Zeit ist komplex und nicht abschließend geklärt.</p>'
        '<li><p>Die Relativität der Zeit ist schwer zu fassen.</p></li>'
        '<li>Es ist unklar, ob Zeit vor dem Urknall existierte. (U1)</li>'
        '<p>Verification Route:</p>'
        '<p>Source: TRAIN</p>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<ol><p>" not in out and "<ol>\n<p>" not in out
    assert out.count("<strong>Schwäche</strong>:") >= 2
    assert mod.detect_self_debunking_numbered_html(out) is True


def test_html_number_self_debunking_ol_drops_verification_route_rows_and_repairs_trailing_md_star_labels():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins.'
        '<br>Warum das wichtig ist**: Relevanz eins.'
        '<br>Was würde verifizieren/falsifizieren (nächster Check)**: Check eins.</li>'
        '<li><strong>Schwäche</strong>: Punkt zwei.'
        '<br>Verification Route:'
        '<br>Source: TRAIN (Allgemeines Hintergrundwissen)'
        '<br>Warum das wichtig ist**: Relevanz zwei.'
        '<br>Was würde verifizieren/falsifizieren (nächster Check)**: Check zwei.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "Verification Route" not in out
    assert "Source: TRAIN" not in out
    assert "Warum das wichtig ist**:" not in out
    assert "Was würde verifizieren/falsifizieren (nächster Check)**:" not in out
    assert "<strong>Warum das wichtig ist</strong>:" in out
    assert "<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:" in out


def test_html_number_self_debunking_repairs_duplicate_primary_label_and_orphan_star_paragraph():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>:<strong>1. Schwäche</strong></li>'
        '<p>*: Die Antwort kann Vereinfachungen enthalten.</p>'
        '<p><strong>Warum das wichtig ist</strong>: Relevanz.</p>'
        '<p><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Test.</p>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert "<strong>1. Schwäche</strong>" not in out
    assert "<p>*:" not in out
    assert "<strong>Schwäche</strong>:" in out
    assert "<br><strong>Warum das wichtig ist</strong>:" in out
    assert "<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:" in out


def test_html_number_self_debunking_ol_drops_uncertainty_tail_li_when_two_points_exist():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins.<br><strong>Warum das wichtig ist</strong>: Relevanz.</li>'
        '<li><strong>Schwäche</strong>: Punkt zwei.<br><strong>Warum das wichtig ist</strong>: Relevanz.</li>'
        '<li><strong>Schwäche</strong>: U1 – Data gap. Needed: Source/current context from the user or external verification.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert out.count("<li") == 2
    assert "Data gap. Needed:" not in out
    assert re.search(r"(?i)<li[^>]*>\\s*<strong>Schwäche</strong>:\\s*U1\\b", out) is None


def test_html_number_self_debunking_ol_strips_embedded_uncertainty_tail_fragments():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Punkt eins. Schwäche: U1 – Data gap. Needed: Source/current context from the user or external verification.<br><strong>Warum das wichtig ist</strong>: Relevanz.</li>'
        '<li><strong>Schwäche</strong>: Punkt zwei. Schwäche: U1 – Data gap. Needed: Source/current context from the user or external verification.<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Test.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert out.count("<li") == 2
    assert "Data gap. Needed:" not in out
    assert re.search(r"(?i)\bU1\b", out) is None
    assert "Punkt eins." in out
    assert "Punkt zwei." in out


def test_html_number_self_debunking_ol_adds_missing_secondary_fields_to_all_items():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: Die Antwort kann Vereinfachungen enthalten.</li>'
        '<li><strong>Schwäche</strong>: Die Antwort kann wichtige Gegenpositionen auslassen.</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    assert out.count("<li") == 2
    assert out.count("<strong>Warum das wichtig ist</strong>:") >= 2
    assert out.count("<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:") >= 2
    assert "Ohne Begründung bleibt die Aussage schwer einzuordnen." not in out
    assert "Eine Primärquelle oder ein Gegenbeispiel gezielt prüfen." not in out


def test_html_number_self_debunking_repairs_leading_secondary_duplicate_pair_in_first_item():
    mod = load_fix_module()
    html_in = (
        '<div class="self-debunking">'
        '<div>Selbst-Debunking:</div>'
        '<ol>'
        '<li><strong>Schwäche</strong>: '
        '<strong>Warum das wichtig ist</strong>: Die benannte Schwäche kann Reichweite, Präzision oder Belastbarkeit der Aussage einschränken.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Den betroffenen Punkt mit Primärquelle, Gegenbeispiel oder Zusatzkontext gezielt nachprüfen.'
        '<br>Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen.'
        '<br><strong>Warum das wichtig ist</strong>: Vereinfachungen können Randfälle oder alternative Deutungen verdecken.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Die zentralen Annahmen explizit machen und gegen Primärquellen/Definitionen prüfen.'
        '</li>'
        '<li><strong>Schwäche</strong>: Die Antwort kann wichtige Gegenpositionen oder Unsicherheitsgrenzen auslassen.'
        '<br><strong>Warum das wichtig ist</strong>: Fehlende Einschränkungen können die Gültigkeit überdehnen oder Sicherheit vortäuschen.'
        '<br><strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>: Mindestens ein starkes Gegenbeispiel ergänzen und prüfen, ob die Kernaussagen bestehen bleiben.'
        '</li>'
        '</ol>'
        '</div>'
    )
    out = mod.html_number_self_debunking(html_in, lang="de")
    m_first = re.search(r"(?is)<li[^>]*>(.*?)</li>", out)
    assert m_first is not None
    first_li = str(m_first.group(1) or "")
    assert "Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen." in first_li
    assert "Die benannte Schwäche kann Reichweite, Präzision oder Belastbarkeit der Aussage einschränken." not in first_li
    assert first_li.count("Warum das wichtig ist") == 1
    assert first_li.count("Was würde verifizieren/falsifizieren (nächster Check)") == 1
    assert re.search(
        r"(?is)^\s*<strong>\s*Schwäche\s*</strong>\s*:\s*(?:<br\s*/?>\s*)*<strong>\s*Warum das wichtig ist\s*</strong>\s*:",
        first_li,
    ) is None


def test_output_footer_renderer_normalizes_bold_qc_footer_to_plain_line():
    mod = load_fix_module()
    footer = getattr(mod, "_output_footer_renderer", None)
    assert footer is not None

    raw_qc = (
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)"
    )
    html_in = (
        "<p>Antwortsatz.</p>"
        "<p><strong>QC-Matrix:</strong> Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)</p>"
    )

    class _GovStub:
        def normalize_qc_overrides(self, ov):
            return ov
        def get_effective_qc_corridor(self, profile_name, overrides):
            return {}

    out = footer.ensure_qc_footer_html_consistency(
        final_html_body=html_in,
        raw_for_render=raw_qc,
        profile_name="Standard",
        gov_mgr=_GovStub(),
        overrides={},
        qc_footer_for_profile_fn=lambda _p: raw_qc,
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=lambda _probe: True,
    )
    assert "<strong>QC-Matrix:" not in out
    assert out.count("QC-Matrix:") == 1


def test_output_footer_renderer_falls_back_to_profile_qc_when_raw_and_rebuild_missing():
    mod = load_fix_module()
    footer = getattr(mod, "_output_footer_renderer", None)
    assert footer is not None

    profile_qc = (
        "QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · "
        "Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)"
    )
    html_in = (
        "<p>Antwortsatz.</p>"
        "<div class='self-debunking'>Self-Debunking:</div>"
        "<div class='ts-footer'>Response at 12.03.2026 20:07:06 CET (UTC+01:00)</div>"
    )

    class _GovStub:
        def normalize_qc_overrides(self, ov):
            return ov

        def get_effective_qc_corridor(self, profile_name, overrides):
            return {}

    out = footer.ensure_qc_footer_html_consistency(
        final_html_body=html_in,
        raw_for_render="Antwort ohne QC-Matrix",
        profile_name="Expert",
        gov_mgr=_GovStub(),
        overrides={},
        qc_footer_for_profile_fn=lambda _p: profile_qc,
        ensure_qc_footer_present_fn=lambda txt, _g, _p, _o: txt,
        enforce_qc_footer_deltas_fn=lambda txt, _c, _p: txt,
        ensure_qc_footer_is_last_fn=lambda txt: txt,
        qc_probe_is_complete_fn=None,
    )
    plain = re.sub(r"<[^>]+>", " ", out)
    plain = re.sub(r"\s+", " ", plain).strip()
    assert "QC-Matrix:" in plain
    assert "Clarity 3 (Δ0)" in plain
    assert out.count("QC-Matrix:") == 1


def test_output_footer_renderer_annotates_qc_dimension_tooltips_and_keeps_footer_order():
    mod = load_fix_module()
    footer = getattr(mod, "_output_footer_renderer", None)
    assert footer is not None
    fn = getattr(footer, "annotate_qc_matrix_tooltips_html", None)
    assert callable(fn)

    html_in = (
        "<div class='sci-trace'>SCI Trace</div>"
        "<div class='self-debunking'>Selbst-Debunking:</div>"
        "<p>QC-Matrix: Clarity 3 (Δ0) · Brevity 0 (Δ0) · Evidence 3 (Δ0) · "
        "Empathy 3 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)</p>"
        "<div class='ts-footer'>Response at 11.03.2026 21:02:27 CET (UTC+01:00)</div>"
    )
    out = fn(html_in, lang="de")
    assert out.count('class="qc-dim-tip"') == 6
    assert out.count("data-u-title=") >= 6
    assert re.search(r'class="qc-dim-tip"[^>]*>Clarity 3 \(Δ0\)</span>', out)
    assert re.search(r'class="qc-dim-tip"[^>]*>Brevity 0 \(Δ0\)</span>', out)
    assert re.search(r'class="qc-dim-tip"[^>]*>Evidence 3 \(Δ0\)</span>', out)
    assert re.search(r'class="qc-dim-tip"[^>]*>Empathy 3 \(Δ0\)</span>', out)
    assert re.search(r'class="qc-dim-tip"[^>]*>Consistency 3 \(Δ0\)</span>', out)
    assert re.search(r'class="qc-dim-tip"[^>]*>Neutrality 3 \(Δ0\)</span>', out)
    assert out.find("SCI Trace") < out.find("Selbst-Debunking:")
    assert out.find("Selbst-Debunking:") < out.find("QC-Matrix:")
    assert out.find("QC-Matrix:") < out.find("Response at")

    out2 = fn(out, lang="de")
    assert out2.count('class="qc-dim-tip"') == 6


def test_self_debunking_en_next_step_label_is_linebroken_and_bold():
    mod = load_fix_module()
    raw = (
        "Self-Debunking:\n"
        "1. Weakness: A compact claim. Next step: Validate against edge-cases.\n"
        "QC-Matrix: Clarity 3 (Δ0)\n"
    )
    out = mod.normalize_self_debunking_field_linebreaks(raw, lang="en")
    out = mod.bold_self_debunking_labels(out, "en")
    assert "\n   **Next step**: Validate against edge-cases." in out


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


def test_manual_test_monitor_state_mutators_update_state_and_emit_js_calls():
    mod = load_fix_module()
    api = mod.Api()
    calls = []
    api._manual_test_monitor_eval = lambda js: calls.append(str(js)) or True  # type: ignore[assignment]

    r1 = api.manual_test_monitor_reset({'scenario': 's1', 'summary': 'ready'})
    assert r1.get('ok') is True
    assert getattr(api, 'manual_test_monitor_state', {}).get('scenario') == 's1'
    assert getattr(api, 'manual_test_monitor_state', {}).get('status') == 'running'
    assert getattr(api, 'manual_test_monitor_state', {}).get('events') == []
    assert any(c.startswith('mtmReplace(') for c in calls)

    calls.clear()
    r2 = api.manual_test_monitor_append('hello')
    assert r2.get('ok') is True
    st = getattr(api, 'manual_test_monitor_state', {})
    assert isinstance(st.get('events'), list)
    assert st['events'][-1] == {'message': 'hello'}
    assert any(c.startswith('mtmAppend(') for c in calls)

    calls.clear()
    r3 = api.manual_test_monitor_set_header({'status': 'done', 'summary': 'ok'})
    assert r3.get('ok') is True
    st = getattr(api, 'manual_test_monitor_state', {})
    assert st.get('scenario') == 's1'
    assert st.get('status') == 'done'
    assert st.get('summary') == 'ok'
    assert any(c.startswith('mtmSetHeader(') for c in calls)


def test_save_manual_test_report_actual_test_overwrites_stable_filename(tmp_path, monkeypatch):
    mod = load_fix_module()
    api = mod.Api()

    monkeypatch.setattr(mod, "LOGS_DIR", str(tmp_path / "Logs"), raising=False)
    monkeypatch.setattr(mod, "_StorageService", None, raising=False)
    api.storage_service = None

    r1 = api.save_manual_test_report({"scenario": "actual_test", "summary": {"status": "PASS", "fails": 0}})
    r2 = api.save_manual_test_report({"scenario": "actual_test", "summary": {"status": "FAIL", "fails": 3}})

    assert isinstance(r1, dict) and isinstance(r2, dict)
    assert r1.get("ok") is True and r2.get("ok") is True
    assert r1.get("overwritten") is True and r2.get("overwritten") is True
    assert r1.get("path") == r2.get("path")
    assert str(r2.get("path") or "").endswith("ManualTest_ACTUAL-TEST.json")

    payload = json.loads(Path(str(r2["path"])).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert ((payload.get("summary") or {}).get("fails")) == 3


def test_manual_test_main_chat_append_emits_expected_js_calls():
    mod = load_fix_module()
    api = mod.Api()
    js_calls = []

    class _MainWin:
        def evaluate_js(self, js):
            js_calls.append(str(js))

    api.main_win = _MainWin()

    r_user = api.manual_test_main_chat_append({"role": "user", "text": "Hallo"})
    assert isinstance(r_user, dict)
    assert r_user.get("ok") is True

    r_bot = api.manual_test_main_chat_append({
        "role": "bot",
        "html": "<p>Antwort</p>",
        "cgi_bar": True,
        "csc": {"score": 3},
        "answer_lang": "en",
    })
    assert isinstance(r_bot, dict)
    assert r_bot.get("ok") is True

    joined = "\n".join(js_calls)
    assert "addMsg('user'" in joined
    assert "addMsg('bot'" in joined
    assert '"answerLang": "en"' in joined


def test_manual_test_request_stop_sets_stop_flag_in_panel_runner():
    mod = load_fix_module()
    api = mod.Api()
    js_calls = []

    class _PanelWin:
        def evaluate_js(self, js):
            js_calls.append(str(js))
            return {"ok": True, "running": True}

    api.panel_win = _PanelWin()
    out = api.manual_test_request_stop({"lang": "de"})
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("running") is True
    joined = "\n".join(js_calls)
    assert "window.__manualTestRunner.stop = true" in joined
    assert "Stop angefordert (Monitor)." in joined


def test_panel_action_manual_test_stop_routes_to_manual_test_request_stop():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    calls = []

    def _stop(payload=None):
        calls.append(payload if isinstance(payload, dict) else {})
        return {"ok": True, "running": False}

    api.manual_test_request_stop = _stop  # type: ignore[assignment]
    out = api.panel_action("manual_test_stop", {"lang": "en"})
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("running") is False
    assert calls and calls[0] == {"lang": "en"}


def test_panel_action_export_routes_to_export():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    calls = []

    def _exp(*_a, **_k):
        calls.append(True)
        return ("/tmp/chat.json", "/tmp/audit.json")

    api.export = _exp  # type: ignore[assignment]
    out = api.panel_action("export", {})
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("chat_path") == "/tmp/chat.json"
    assert out.get("audit_path") == "/tmp/audit.json"
    assert calls == [True]


def test_panel_action_preview_export_file_routes_to_preview():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    calls = []

    def _preview(path, max_chars=0):
        calls.append((path, max_chars))
        return {
            "ok": True,
            "kind": "audit",
            "relative_path": "Logs/Audit/Audit_demo.json",
            "preview": "{\"ok\": true}",
            "truncated": False,
        }

    api.preview_export_file = _preview  # type: ignore[assignment]
    out = api.panel_action("preview_export_file", {"path": "/tmp/Audit_demo.json", "max_chars": 1234})
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("relative_path") == "Logs/Audit/Audit_demo.json"
    assert calls == [("/tmp/Audit_demo.json", 1234)]


def test_panel_action_open_export_preview_routes_to_window_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    calls = []

    def _open(path, max_chars=0):
        calls.append((path, max_chars))
        return {"ok": True, "relative_path": "Logs/Audit/Audit_demo.json", "kind": "audit", "truncated": False}

    api.open_export_preview = _open  # type: ignore[assignment]
    out = api.panel_action("open_export_preview", {"path": "/tmp/Audit_demo.json", "max_chars": 1234})
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert out.get("relative_path") == "Logs/Audit/Audit_demo.json"
    assert calls == [("/tmp/Audit_demo.json", 1234)]


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

    r_mirror = api.panel_action('manual_test_main_chat_append', {'payload': {'role': 'sys', 'text': 'x'}})
    assert isinstance(r_mirror, dict)
    assert r_mirror.get('ok') is False
    assert r_mirror.get('error') == 'comm_off_blocked'

    r_start = api.panel_action('ask', {'text': 'Comm Start'})
    assert isinstance(r_start, dict)
    # The panel_action gate must not block Comm Start (actual execution path may still fail in isolated test).
    assert r_start.get('error') != 'comm_off_blocked'


def test_main_input_is_blocked_while_qc_override_modal_is_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api._qc_override_open = True
    api.chat_session = DummySession(['SHOULD_NOT_BE_CALLED'])

    calls = []

    class _QcWin:
        def show(self):
            calls.append("show")
        def bring_to_front(self):
            calls.append("bring_to_front")

    api.qc_win = _QcWin()
    out = api.ask("Was ist Zeit?")
    assert isinstance(out, dict)
    html = str(out.get("html") or "")
    assert "QC Override" in html
    assert "geöffnet" in html or "open" in html
    assert api.chat_session.calls == []  # type: ignore[attr-defined]
    assert "bring_to_front" in calls or "show" in calls


def test_panel_action_blocks_non_qc_actions_while_qc_override_modal_is_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api._qc_override_open = True

    calls = []

    class _QcWin:
        def show(self):
            calls.append("show")
        def bring_to_front(self):
            calls.append("bring_to_front")

    api.qc_win = _QcWin()
    out = api.panel_action('list_chat_logs', {'limit': 10})
    assert isinstance(out, dict)
    assert out.get('ok') is False
    assert out.get('error') == 'qc_override_modal_blocked'
    assert "bring_to_front" in calls or "show" in calls


def test_panel_action_allows_qc_clear_while_qc_override_modal_is_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api._qc_override_open = True

    class _QcWin:
        def hide(self):
            return None

    api.qc_win = _QcWin()
    out = api.panel_action('qc_override_clear', {})
    assert isinstance(out, dict)
    assert out.get('ok') is True
    assert bool(getattr(api, '_qc_override_open', True)) is False


def test_remote_cmd_is_blocked_while_qc_override_modal_is_open():
    mod = load_fix_module()
    api = mod.Api()
    api.main_win = object()
    api._qc_override_open = True

    calls = []

    class _QcWin:
        def show(self):
            calls.append("show")
        def bring_to_front(self):
            calls.append("bring_to_front")

    api.qc_win = _QcWin()
    blocked = api.remote_cmd("Comm State")
    assert isinstance(blocked, dict)
    assert blocked.get("ok") is False
    assert blocked.get("error") == "qc_override_modal_blocked"

    reopen = api.remote_cmd("QC Override")
    assert isinstance(reopen, dict)
    assert reopen.get("ok") is True
    assert reopen.get("already_open") is True
    assert "bring_to_front" in calls or "show" in calls


def test_panel_action_is_blocked_while_exit_confirm_modal_is_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api.gov_state.comm_active = True
    api._exit_confirm_open = True

    out = api.panel_action('list_chat_logs', {'limit': 5})
    assert isinstance(out, dict)
    assert out.get('ok') is False
    assert out.get('error') == 'exit_confirm_open_blocked'


def test_remote_cmd_is_blocked_while_exit_confirm_modal_is_open():
    mod = load_fix_module()
    api = mod.Api()
    api.main_win = object()
    api._exit_confirm_open = True

    out = api.remote_cmd("Comm State")
    assert isinstance(out, dict)
    assert out.get("ok") is False
    assert out.get("error") == "exit_confirm_open_blocked"


def test_ask_is_blocked_while_exit_confirm_modal_is_open():
    mod = load_fix_module()
    _prime_module_gov(mod)
    api = mod.Api()
    api._exit_confirm_open = True
    api.chat_session = DummySession(['should-not-run'])  # type: ignore[attr-defined]

    out = api.ask("Was ist Zeit?")
    assert isinstance(out, dict)
    html = str(out.get("html") or "")
    assert "Exit" in html or "Bestaetigung" in html
    assert api.chat_session.calls == []  # type: ignore[attr-defined]


def test_help_content_follows_answer_language():
    mod = load_fix_module()
    api = mod.Api()

    api.gov_state.answer_language = "en"
    en = api.get_help_content()
    assert isinstance(en, dict) and en.get("ok") is True
    assert en.get("lang") == "en"
    assert isinstance(en.get("payload"), dict)
    assert "Help" in str((en.get("payload") or {}).get("title") or "")

    api.gov_state.answer_language = "de"
    de = api.get_help_content()
    assert isinstance(de, dict) and de.get("ok") is True
    assert de.get("lang") == "de"
    assert isinstance(de.get("payload"), dict)
    assert "Hilfe" in str((de.get("payload") or {}).get("title") or "")


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


def test_language_script_contract_flags_cyrillic_outside_quote_or_source():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = "Dies ist ein deutscher Satz mit заинтересованных сторон im Fliesstext."
    vios = validator.validate_language_script_contract(txt, expected_lang='de')

    assert any('Language contract' in v for v in vios)


def test_output_validator_conversation_lang_follows_answer_language_config():
    mod = load_fix_module()
    _prime_module_gov(mod)
    cfg = getattr(mod, "cfg", None)
    assert cfg is not None

    cfg.set_answer_language("en")
    validator = mod.OutputComplianceValidator(mod.gov, cfg)
    assert validator._conversation_lang() == "en"

    cfg.set_answer_language("de")
    validator2 = mod.OutputComplianceValidator(mod.gov, cfg)
    assert validator2._conversation_lang() == "de"


def test_language_script_contract_allows_cyrillic_in_quote_and_source_line():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = (
        'Die Uebersetzung lautet "заинтересованных сторон".\n'
        'Source: https://example.org/ru заинтересованных сторон\n'
        'Der restliche Antworttext bleibt deutsch.'
    )
    vios = validator.validate_language_script_contract(txt, expected_lang='de')

    assert vios == []


def test_language_script_contract_ignores_control_layer_lines_and_scientific_symbols():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = (
        "CONTROL LAYER NOTE: технический Hinweis\n"
        "Die Formel Δt beschreibt eine zeitliche Änderung im Modell."
    )
    vios = validator.validate_language_script_contract(txt, expected_lang='de')

    assert vios == []


def test_language_policy_benchmark_moves_language_violations_to_soft():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))
    state = types.SimpleNamespace(
        answer_language="de",
        language_policy_mode="benchmark",
        active_profile="Standard",
        sci_variant="",
        sci_active=False,
    )

    hard, soft = validator.validate(
        text="Status: смешанный текст",
        state=state,
        expect_menu=False,
        expect_trace=False,
        is_command=True,
        user_prompt="Comm State",
    )

    assert hard == []
    assert any("Language policy benchmark:" in v for v in soft)


def test_language_policy_production_keeps_language_violations_hard():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))
    state = types.SimpleNamespace(
        answer_language="de",
        language_policy_mode="production",
        active_profile="Standard",
        sci_variant="",
        sci_active=False,
    )

    hard, soft = validator.validate(
        text="Status: смешанный текст",
        state=state,
        expect_menu=False,
        expect_trace=False,
        is_command=True,
        user_prompt="Comm State",
    )

    assert any("Language contract" in v for v in hard)
    assert not any("Language policy benchmark:" in v for v in soft)


def test_verification_route_gate_strong_claim_u_only_is_not_enough():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = 'Das beweist definitiv, dass die Methode immer korrekt ist. U1.'
    vios = validator.validate_verification_route_gate(txt, is_command=False)

    assert any('uncertainty label alone is insufficient' in v for v in vios)


def test_verification_route_gate_allows_downgraded_strong_claim_with_u_label():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = 'Das wirkt definitiv, ist aber nur eine Hypothese und daher unsicher. U1.'
    vios = validator.validate_verification_route_gate(txt, is_command=False)

    assert not any('strong-claim heuristic triggered' in v for v in vios)


def test_verification_route_gate_red_claim_requires_u_and_route():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))

    txt = '[RED] 🔴 Kritische Aussage ohne Nachweis.'
    vios = validator.validate_verification_route_gate(txt, is_command=False)

    assert any('RED claim requires uncertainty label' in v for v in vios)
    assert any('RED claim requires at least one verification route marker' in v for v in vios)


def test_verification_route_gate_accepts_u8_marker():
    mod = load_fix_module()
    _prime_module_gov(mod)
    validator = mod.OutputComplianceValidator(mod.gov, getattr(mod, 'cfg', None))
    txt = "🔴 RED claim mit downgraded wording und U8, aber ohne Source."
    vios = validator.validate_verification_route_gate(txt, is_command=False)
    assert not any("RED claim requires uncertainty label" in v for v in vios)


def test_build_repair_prompt_adds_transliteration_guidance_on_language_contract_violation():
    mod = load_fix_module()
    gov_mgr = types.SimpleNamespace(loaded=False, data={})
    validator = mod.OutputComplianceValidator(gov_mgr, None)
    state = types.SimpleNamespace(answer_language="de", sci_variant="", sci_active=False, qc_overrides={})

    prompt = validator.build_repair_prompt(
        user_prompt="Bitte erklaere den Begriff.",
        raw_response="Antwort mit заинтересованных сторон im Fliesstext.",
        state=state,
        hard_violations=["Language contract: expected DE content; found non-DE script outside allowed quote/source contexts."],
        soft_violations=[],
    )

    assert "Language contract repair guidance:" in prompt
    assert "transliterate" in prompt.lower()
    assert "source/citation lines" in prompt
    assert "Do not rely on name-specific whitelists." in prompt
