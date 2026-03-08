import os
import json
import re
import html
import sys
import traceback
import importlib
import difflib
import hashlib
import base64
from collections import deque, defaultdict
from pathlib import Path
import math
try:
    import markdown  # type: ignore
except Exception:
    # Optional dependency. Fallback keeps UI functional and tests runnable.
    class _MarkdownShim:
        @staticmethod
        def markdown(text, extensions=None):
            try:
                return "<pre>" + html.escape(str(text)) + "</pre>"
            except Exception:
                return "<pre>" + str(text) + "</pre>"
    markdown = _MarkdownShim()  # type: ignore

try:
    import Module.compliance_scan as _compliance_scan  # type: ignore
except Exception:
    _compliance_scan = None  # type: ignore

try:
    import Module.auditstream as _auditstream  # type: ignore
except Exception:
    _auditstream = None  # type: ignore

try:
    import Module.rendering_utils as _rendering_utils  # type: ignore
except Exception:
    _rendering_utils = None  # type: ignore

try:
    import Module.rendering_pipeline_v192 as _rendering_pipeline_v192  # type: ignore
except Exception:
    _rendering_pipeline_v192 = None  # type: ignore

try:
    from ui_panel_model import StateSnapshot as _PanelStateSnapshot, normalize_panel_ui as _panel_normalize_ui  # type: ignore
except Exception:
    _PanelStateSnapshot = None  # type: ignore
    _panel_normalize_ui = None  # type: ignore

try:
    from intents import intent_from_command as _intent_from_command, ProcessModelResponse as _ProcessModelResponse, ComplianceViolation as _ComplianceViolation  # type: ignore
    from state import state_from_runtime as _state_from_runtime, apply_state_to_runtime as _state_apply_to_runtime, init_state_from_ruleset as _state_init_from_ruleset  # type: ignore
    from transitions import apply_intent as _apply_intent  # type: ignore
except Exception:
    _intent_from_command = None  # type: ignore
    _ProcessModelResponse = None  # type: ignore
    _ComplianceViolation = None  # type: ignore
    _state_from_runtime = None  # type: ignore
    _state_apply_to_runtime = None  # type: ignore
    _state_init_from_ruleset = None  # type: ignore
    _apply_intent = None  # type: ignore

try:
    from controller import dispatch as _controller_dispatch  # type: ignore
except Exception:
    _controller_dispatch = None  # type: ignore

try:
    from controller import dispatch_intent as _controller_dispatch_intent  # type: ignore
except Exception:
    _controller_dispatch_intent = None  # type: ignore

try:
    from governance_service import GovernanceService as _GovernanceService  # type: ignore
except Exception:
    _GovernanceService = None  # type: ignore

try:
    from ui_controller import UIController as _UIController  # type: ignore
except Exception:
    _UIController = None  # type: ignore

try:
    from storage_service import StorageService as _StorageService  # type: ignore
except Exception:
    _StorageService = None  # type: ignore

try:
    import app_bootstrap as _app_bootstrap  # type: ignore
except Exception:
    _app_bootstrap = None  # type: ignore

try:
    import panel_asset_loader as _panel_asset_loader  # type: ignore
except Exception:
    _panel_asset_loader = None  # type: ignore

try:
    import panel_bootstrap_state as _panel_bootstrap_state_mod  # type: ignore
except Exception:
    _panel_bootstrap_state_mod = None  # type: ignore

try:
    import panel_window_fallback as _panel_window_fallback_mod  # type: ignore
except Exception:
    _panel_window_fallback_mod = None  # type: ignore

try:
    import panel_html_source as _panel_html_source_mod  # type: ignore
except Exception:
    _panel_html_source_mod = None  # type: ignore

try:
    import panel_lifecycle_seam as _panel_lifecycle_seam_mod  # type: ignore
except Exception:
    _panel_lifecycle_seam_mod = None  # type: ignore

try:
    import qc_override_window_seam as _qc_override_window_seam_mod  # type: ignore
except Exception:
    _qc_override_window_seam_mod = None  # type: ignore

try:
    import panel_ui_snapshot_seam as _panel_ui_snapshot_seam_mod  # type: ignore
except Exception:
    _panel_ui_snapshot_seam_mod = None  # type: ignore

try:
    import manual_test_monitor_seam as _manual_test_monitor_seam_mod  # type: ignore
except Exception:
    _manual_test_monitor_seam_mod = None  # type: ignore

try:
    import uncertainty_codes as _uncertainty_codes_mod  # type: ignore
except Exception:
    _uncertainty_codes_mod = None  # type: ignore

try:
    import help_i18n as _help_i18n_mod  # type: ignore
except Exception:
    _help_i18n_mod = None  # type: ignore

try:
    from qc_bridge import QCBridge as _QCBridge  # type: ignore
except Exception:
    _QCBridge = None  # type: ignore

try:
    from panel_bridge import PanelBridge as _PanelBridge  # type: ignore
except Exception:
    _PanelBridge = None  # type: ignore

try:
    from main_bridge import MainBridge as _MainBridge  # type: ignore
except Exception:
    _MainBridge = None  # type: ignore

# Stage 3e (CI): strict module mode. If enabled, missing extracted modules is a hard error.
_STRICT_MODULES = (os.environ.get('WRAPPER_STRICT_MODULES', '') or '').strip().lower() in ('1', 'true', 'yes', 'on')

def _strict_local_module_loaded(mod_obj, module_dir_name: str = 'Module') -> bool:
    """Return True only when module file is loaded from the local wrapper tree."""
    try:
        if mod_obj is None:
            return False
        mod_file = Path(getattr(mod_obj, '__file__', '') or '').resolve()
        if not mod_file:
            return False
        local_module_dir = (Path(__file__).resolve().parent / module_dir_name).resolve()
        return str(mod_file).startswith(str(local_module_dir) + os.sep)
    except Exception:
        return False

if _STRICT_MODULES:
    missing = []
    if not _strict_local_module_loaded(_auditstream):
        missing.append('Module.auditstream')
    if not _strict_local_module_loaded(_rendering_utils):
        missing.append('Module.rendering_utils')
    if not _strict_local_module_loaded(_compliance_scan):
        missing.append('Module.compliance_scan')
    if missing:
        raise SystemExit("WRAPPER_STRICT_MODULES=1: missing required modules: " + ", ".join(missing))

try:
    import bleach  # type: ignore
except Exception:
    bleach = None  # type: ignore

_css_sanitizer_cached = None

def _get_css_sanitizer():
    """Return a Bleach CSSSanitizer allowing only the minimal inline styles we inject.

    We keep this extremely narrow to avoid changing rendering semantics or expanding attack surface.
    """
    global _css_sanitizer_cached
    if _css_sanitizer_cached is not None:
        return _css_sanitizer_cached
    try:
        from bleach.css_sanitizer import CSSSanitizer  # type: ignore
        _css_sanitizer_cached = CSSSanitizer(allowed_css_properties=['color','font-weight'])
    except Exception:
        _css_sanitizer_cached = None
    return _css_sanitizer_cached

def sanitize_html(html_text: str) -> str:
    # Defensive HTML sanitization for model outputs before injecting into pywebview.
    # Keeps a conservative allow-list while preserving our own injected spans/images.
    if not html_text:
        return ''
    if bleach is None:
        return html_text

    allowed_tags = [
        'p','br','b','strong','i','em','u','code','pre','blockquote','details','summary',
        'ul','ol','li','table','thead','tbody','tr','th','td','hr',
        'div','span','img','a',
        'h1','h2','h3','h4','h5','h6'
    ]
    allowed_attrs = {
        '*': ['class','style'],
        'a': ['href','title','target','rel','class','style'],
        'img': ['src','alt','title','style','loading','class'],
        'code': ['class'],
        'pre': ['class'],
        'details': ['open','class','style'],
        'summary': ['class','style'],
        'th': ['colspan','rowspan','class','style'],
        'td': ['colspan','rowspan','class','style'],
    }
    try:
        css_sanitizer = _get_css_sanitizer()
        _attrs = allowed_attrs
        _kwargs = {}
        if css_sanitizer is None:
            # Avoid bleach NoCssSanitizerWarning by not allowing 'style' through bleach when no CSS sanitizer exists.
            _attrs = {k: [a for a in v if a != 'style'] for k, v in allowed_attrs.items()}
        else:
            _kwargs['css_sanitizer'] = css_sanitizer
        cleaned = bleach.clean(
            html_text,
            tags=allowed_tags,
            attributes=_attrs,
            protocols=['http','https','mailto'],
            strip=True,
            **_kwargs,
        )
        if css_sanitizer is None:
            cleaned = _reapply_color_styles_if_stripped(cleaned)
        return cleaned
    except Exception:
        return html_text

# ============================================
# STUFE 1 — IN-FILE BOUNDARY REFACTOR (v141)
# ============================================
# Goal: Introduce explicit boundary markers + minimal schema contracts
# WITHOUT changing runtime behavior, UI rendering, provider handling,
# or governance semantics. This is a "seams + contracts" step only.
#
# Boundaries (conceptual; still single-file by plan):
#   A) Identity / Versioning
#   B) Safety utilities (sanitization, hashing, previews)
#   C) Routing / Parsing (standalone commands, SCI pending A–H, numeric code guard)
#   D) Governance loading & token registry
#   E) Rendering (HTML/CSS, QC/SCI blocks)
#   F) Providers (Gemini/OpenRouter/HF) — untouched in this stage
#   G) UI glue (pywebview API) + Panel wiring
#   H) Main bootstrap
#
# Note: Contracts are fail-soft: they never raise, they only return bool
# and optionally emit an internal log_event() on violation.

# ----------------------------
# WRAPPER IDENTITY (dynamic, derived from filename)
# ----------------------------

def _detect_wrapper_identity() -> tuple[str, str]:
    """Return stable app identity independent of historical Wrapper-<NNN> filenames."""
    try:
        stem = Path(__file__).stem
        if stem:
            return "Comm-SCI-Control-App", ""
    except Exception:
        pass
    return "Comm-SCI-Control-App", ""

WRAPPER_NAME, WRAPPER_VERSION = _detect_wrapper_identity()
MAIN_WINDOW_TITLE = WRAPPER_NAME
PANEL_WINDOW_TITLE = f"{WRAPPER_NAME} Panel"


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class RateLimiter:
    """Simple sliding-window rate limiter (minute/hour) with optional scopes.

    Global scope is always enforced. Provider/model scopes are enforced only if configured
    in `scopes` during initialization.

    Interface for tests:
      - allow_call(..., return_retry=True) -> (ok, msg, retry_after_s)
      - allow_call(..., return_retry=False) -> (ok, msg)
    """

    def __init__(
        self,
        per_minute: int = 0,
        per_hour: int = 0,
        *,
        scopes=None,
        clock=None,
    ):
        from collections import defaultdict, deque
        import threading

        self._clock = clock or time.time
        self._lock = threading.Lock()

        self._limits = {
            "global": {"per_minute": int(per_minute or 0), "per_hour": int(per_hour or 0)}
        }

        # Optional per-scope overrides
        if isinstance(scopes, dict):
            for sc, lim in scopes.items():
                try:
                    self._limits[str(sc)] = {
                        "per_minute": int((lim or {}).get("per_minute", 0) or 0),
                        "per_hour": int((lim or {}).get("per_hour", 0) or 0),
                    }
                except Exception:
                    continue

        self._buckets = defaultdict(deque)

    def _prune(self, dq, now: float):
        cutoff = now - 3600.0
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _count_last_minute(self, dq, now: float):
        cutoff = now - 60.0
        c = 0
        for t in reversed(dq):
            if t >= cutoff:
                c += 1
            else:
                break
        return c

    def _check_one_scope(self, scope: str, now: float):
        lim = self._limits.get(scope) or {"per_minute": 0, "per_hour": 0}
        per_h = int(lim.get("per_hour", 0) or 0)
        per_m = int(lim.get("per_minute", 0) or 0)

        dq = self._buckets[scope]
        self._prune(dq, now)

        worst_retry = 0.0
        worst_msg = ""

        if per_h > 0 and len(dq) >= per_h:
            earliest = dq[0]
            retry = max(0.0, (earliest + 3600.0) - now)
            worst_retry = retry
            worst_msg = f"Rate limit exceeded (hourly) for scope '{scope}': {per_h}/hour"

        if per_m > 0:
            used = self._count_last_minute(dq, now)
            if used >= per_m:
                cutoff = now - 60.0
                oldest_in_window = None
                for t in dq:
                    if t >= cutoff:
                        oldest_in_window = t
                        break
                if oldest_in_window is None:
                    oldest_in_window = dq[-1] if dq else now
                retry = max(0.0, (oldest_in_window + 60.0) - now)
                if retry >= worst_retry:
                    worst_retry = retry
                    worst_msg = f"Rate limit exceeded (per-minute) for scope '{scope}': {per_m}/min"

        ok = (worst_retry <= 0.0)
        return ok, worst_retry, worst_msg

    def allow_call(
        self,
        *,
        provider: str = "",
        model: str = "",
        reason: str = "",
        consume: bool = True,
        return_retry: bool = False,
    ):
        with self._lock:
            now = float(self._clock())

            scopes = ["global"]
            p = (provider or "").strip().lower()
            m = (model or "").strip()

            # optional scopes only if configured
            if p and f"provider:{p}" in self._limits:
                scopes.append(f"provider:{p}")
            if p and m and f"model:{p}:{m}" in self._limits:
                scopes.append(f"model:{p}:{m}")

            worst_retry = 0.0
            worst_msg = ""

            for sc in scopes:
                ok, retry, msg = self._check_one_scope(sc, now)
                if not ok and retry >= worst_retry:
                    worst_retry = retry
                    worst_msg = msg or worst_msg

            if worst_retry > 0.0:
                retry_s = int(worst_retry + 0.999)  # ceil
                msg = worst_msg or "Rate limit exceeded"
                # Unit-test contract: message includes a Retry-after hint
                msg = f"{msg} | Retry after {retry_s}s"
                if reason:
                    msg = f"{msg} | Reason: {reason}"
                if return_retry:
                    return False, msg, retry_s
                return False, msg

            if consume:
                for sc in scopes:
                    self._buckets[sc].append(now)

            if return_retry:
                return True, "", 0
            return True, ""

    # Backward-compatible API
    def allow(self):
        return self.allow_call()
def _derive_fernet_key(passphrase: str, salt_b64: str) -> bytes:
    """Derive a Fernet key from passphrase + salt (urlsafe base64)."""
    salt = base64.urlsafe_b64decode((salt_b64 or '').encode('utf-8'))
    key = hashlib.pbkdf2_hmac('sha256', passphrase.encode('utf-8'), salt, 200_000, dklen=32)
    return base64.urlsafe_b64encode(key)


def _try_decrypt_api_key(enc_b64: str, *, passphrase: str, salt_b64: str):
    """Best-effort decrypt of encrypted API key using cryptography. Returns None on failure."""
    if not enc_b64 or not passphrase or not salt_b64:
        return None
    try:
        from cryptography.fernet import Fernet  # type: ignore
    except Exception:
        return None
    try:
        fkey = _derive_fernet_key(passphrase, salt_b64)
        f = Fernet(fkey)
        plain = f.decrypt(enc_b64.encode('utf-8'))
        return plain.decode('utf-8').strip()
    except Exception:
        return None


def _try_encrypt_api_key(plain_text: str, *, passphrase: str):
    """Best-effort encrypt helper (Fernet). Returns (enc_b64, salt_b64) or (None, None)."""
    if plain_text is None:
        plain_text = ""
    if not str(plain_text).strip() or not passphrase:
        return None, None
    try:
        from cryptography.fernet import Fernet  # type: ignore
    except Exception:
        return None, None
    try:
        salt_b64 = base64.urlsafe_b64encode(os.urandom(16)).decode("utf-8")
        fkey = _derive_fernet_key(passphrase, salt_b64)
        f = Fernet(fkey)
        enc_b64 = f.encrypt(str(plain_text).encode("utf-8")).decode("utf-8")
        return enc_b64, salt_b64
    except Exception:
        return None, None

try:
    import webview  # type: ignore
except Exception:
    webview = None  # type: ignore
import pathlib
import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore

# ----------------------------
# PATHS & DEFAULT FILES
# ----------------------------
# Project paths (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_ASSETS_DIR = os.path.join(SCRIPT_DIR, 'ui_assets')

PROJECT_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR

JSON_DIR = os.path.join(PROJECT_DIR, 'JSON')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'Config')

LOGS_DIR = os.path.join(PROJECT_DIR, 'Logs')
AUDIT_LOG_DIR = os.path.join(LOGS_DIR, 'Audit')
CHAT_LOG_DIR = os.path.join(LOGS_DIR, 'Chats')
HISTORY_LOG_DIR = os.path.join(LOGS_DIR, 'History')
CACHE_LOG_DIR = os.path.join(LOGS_DIR, 'Cache')
SESSION_EVENTS_MAX = 2000  # cap in-memory session_events (JSONL is full history)
USAGE_LOG_DIR = os.path.join(LOGS_DIR, 'Usage_statistics')

# Create directories (idempotent)
for _d in (JSON_DIR, CONFIG_DIR, LOGS_DIR, AUDIT_LOG_DIR, CHAT_LOG_DIR, HISTORY_LOG_DIR, CACHE_LOG_DIR, USAGE_LOG_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass


def _load_ui_asset_text(filename: str, fallback_text: str) -> str:
    """Load optional external UI asset text with deterministic fail-open fallback.

    S7 goal: move large embedded UI templates out of the monolith without changing
    runtime behavior. If the file is missing/unreadable, keep the embedded string.
    """
    if _panel_asset_loader is not None:
        try:
            return _panel_asset_loader.load_ui_asset_text(UI_ASSETS_DIR, filename, fallback_text)
        except Exception:
            pass
    try:
        path = os.path.join(UI_ASSETS_DIR, filename)
        if not os.path.isfile(path):
            return fallback_text
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return txt if txt else fallback_text
    except Exception:
        return fallback_text


def _env_flag_enabled(name: str) -> bool:
    v = (os.environ.get(name) or '').strip().lower()
    return v in {'1', 'true', 'yes', 'on'}


def _panel_asset_static_selftest_report(panel_html: str) -> dict:
    """Static sanity checks for external panel.html across legacy/current marker variants."""
    if _panel_asset_loader is not None:
        try:
            return _panel_asset_loader.panel_asset_static_selftest_report(panel_html)
        except Exception:
            pass
    txt = panel_html or ''

    # Accept both historical and current marker names so the self-test checks
    # structure/behavioral anchors instead of one exact HTML snapshot.
    marker_groups = {
        "action_fn": ('function panelAction(', 'function run('),
        "build_ui_fn": ('function buildUI(',),
        "pywebview_bridge": ('window.pywebview',),
        "provider_select": ('id="provider"',),
        "model_select": ('id="model"',),
        "answer_language": ('id="answer-language"', 'id="anslang"'),
        "manual_test_scenario": ('id="manual-test-scenario"', 'id="manualTestScenario"'),
        "monitor_visibility": ('id="monitor-visibility"', 'id="manualTestMonitorMode"'),
        "dynamic_sections": (
            'comm-core-grid',
            "section('Comm Core'",
            'section("Comm Core"',
        ),
    }

    missing = []
    matched = {}
    for key, variants in marker_groups.items():
        hit = None
        for marker in variants:
            if marker in txt:
                hit = marker
                break
        matched[key] = hit
        if hit is None:
            missing.append(key)

    return {
        "ok": (len(missing) == 0),
        "missing": missing,
        "matched": matched,
    }


def _panel_asset_static_selftest_ok(panel_html: str) -> bool:
    if _panel_asset_loader is not None:
        try:
            return bool(_panel_asset_loader.panel_asset_static_selftest_ok(panel_html))
        except Exception:
            pass
    try:
        return bool((_panel_asset_static_selftest_report(panel_html) or {}).get("ok"))
    except Exception:
        return False


def _panel_runtime_selftest_payload_ok(payload) -> tuple[bool, str]:
    """Validate panel runtime-selftest callback payload from external panel.html."""
    if _panel_asset_loader is not None:
        try:
            return _panel_asset_loader.panel_runtime_selftest_payload_ok(payload)
        except Exception:
            pass
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    if payload.get("ok") is not True:
        return False, "js_report_not_ok"
    if payload.get("bridge_ping") is not True:
        return False, "bridge_ping_missing"
    if payload.get("build_ui") is not True:
        return False, "build_ui_missing"
    if payload.get("dom_ok") is not True:
        return False, "dom_markers_missing"

    try:
        loaded = (payload.get("data_loaded") is True)
    except Exception:
        loaded = False
    try:
        dyn_count = int(payload.get("dynamic_section_count", 0) or 0)
    except Exception:
        dyn_count = 0
    if loaded and dyn_count <= 0:
        return False, "loaded_ruleset_but_no_dynamic_sections"
    return True, ""


def _load_panel_asset_text_s7(fallback_text: str):
    """Prefer external panel.html when static self-test passes; runtime self-test happens pre-show."""
    if _panel_asset_loader is not None:
        try:
            return _panel_asset_loader.load_panel_asset_text_s7(UI_ASSETS_DIR, fallback_text, print_fn=print)
        except Exception:
            pass
    txt = _load_ui_asset_text('panel.html', '')
    if not txt:
        print('[S7] panel.html asset missing/unreadable; using embedded panel.')
        return fallback_text, {"source": "embedded", "reason": "missing_asset", "static_ok": False}

    report = _panel_asset_static_selftest_report(txt)
    if not report.get("ok"):
        miss = ",".join(report.get("missing") or []) or "unknown"
        print(f"[S7] panel.html static self-test failed ({miss}); using embedded panel.")
        return fallback_text, {"source": "embedded", "reason": "static_selftest_failed", "static_ok": False, "report": report}

    print('[S7] panel.html external asset enabled (static self-test passed; runtime self-test pending).')
    return txt, {"source": "external", "reason": "static_selftest_passed", "static_ok": True, "report": report}

# ----------------------------
# STUFE 0: Golden Run Checklist (manual, non-network)
# ----------------------------
# This is a compact, in-code reference so regressions can be spotted quickly.
# It is intentionally non-normative (the JSON ruleset remains the Source of Truth).
GOLDEN_RUN_STUFE0 = [
    "Comm Start -> Comm Help -> Comm State",
    "Profile switch (e.g., Expert/Sparring) -> SCI menu appears -> choose A -> ask a real question",
    "Color on/off toggles evidence tag rendering",
    "Comm Audit exports (no provider call)",
    "Comm Stop -> no governance pinned -> UI remains responsive",
    "Clear Chat resets runtime state (incl. QC overrides if present)",
]

# === Stage 3d: Dependency health (soft-import visibility) ===
# We keep soft-import fallbacks, but we also surface module availability deterministically
# in Comm State / Comm Audit so missing modules never silently change behavior.
def _check_optional_module(name: str, *, strict: bool = False):
    """Return (ok: bool, err: str). If strict=True, raises on failure."""
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as e:
        if strict:
            raise
        return False, f"{type(e).__name__}: {e}"

def _strict_modules_enabled() -> bool:
    v = os.environ.get('WRAPPER_STRICT_MODULES', '')
    v = (v or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')





def _safe_preview_text(s: str, limit: int = 160) -> str:
    try:
        s = str(s or "")
    except Exception:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > limit:
        return s[:limit] + "…"
    return s


def _safe_sha256(s: str) -> str:
    try:
        b = (s or "").encode("utf-8", errors="replace")
        return hashlib.sha256(b).hexdigest()[:16]
    except Exception:
        return ""

# Default ruleset location preference (latest first):
#   1) ./JSON/Comm-SCI-v20.2.1.json
#   2) ./JSON/Comm-SCI-v20.2.0.json
#   3) ./JSON/Comm-SCI-v20.1.0.json
#   4) legacy fallbacks (v20.0.3 / v20.0.2)
_DEFAULT_RULESET_CANDIDATES = [
    'Comm-SCI-v20.2.1.json',
    'Comm-SCI-v20.2.0.json',
    'Comm-SCI-v20.1.0.json',
    'Comm-SCI-v20.0.3.json',
    'Comm-SCI-v20.0.2.json',
]
DEFAULT_JSON = ''
for _name in _DEFAULT_RULESET_CANDIDATES:
    _p = os.path.join(JSON_DIR, _name)
    if os.path.exists(_p):
        DEFAULT_JSON = _p
        break
if not DEFAULT_JSON:
    for _name in _DEFAULT_RULESET_CANDIDATES:
        _p = os.path.join(PROJECT_DIR, _name)
        if os.path.exists(_p):
            DEFAULT_JSON = _p
            break
if not DEFAULT_JSON:
    # Final fallback path (may still fail later with clear file-not-found log).
    DEFAULT_JSON = os.path.join(JSON_DIR, 'Comm-SCI-v20.2.1.json')

# Config/keys location: ./Config/
CONFIG_FILENAME = 'Comm-SCI-Config.json'
KEYS_FILENAME = 'Comm-SCI-API-Keys.json'

CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_FILENAME)
KEYS_PATH = os.path.join(CONFIG_DIR, KEYS_FILENAME)
KEYS_EXAMPLE_FILENAME = 'Comm-SCI-API-Keys.example.json'
KEYS_EXAMPLE_PATH = os.path.join(CONFIG_DIR, KEYS_EXAMPLE_FILENAME)

# --- Keys file override (to keep original files untouched) ---
KEYS_OVERRIDE_ENV = 'COMM_SCI_KEYS_FILE'
KEYS_OVERRIDE_FILENAME = 'Comm-SCI-API-Keys.override.json'
KEYS_OVERRIDE_PATH = os.path.join(CONFIG_DIR, KEYS_OVERRIDE_FILENAME)

def _iter_keys_paths():
    """Yield key file candidates in priority order.

    Order:
      1) ENV override: COMM_SCI_KEYS_PATH (if set)
      2) Local override file in project root: Comm-SCI-API-Keys.local.json (if present)
      3) Standard file: Config/Comm-SCI-API-Keys.json (if present)
      4) Example file:  Config/Comm-SCI-API-Keys.example.json (if present)
    """
    p = os.environ.get('COMM_SCI_KEYS_PATH', '').strip()
    if p:
        yield p

    # optional per-user local override (kept out of git)
    local_override = os.path.join(PROJECT_DIR, 'Comm-SCI-API-Keys.local.json')
    if os.path.exists(local_override):
        yield local_override

    # prefer real keys if present; otherwise fall back to example keys
    if os.path.exists(KEYS_PATH):
        yield KEYS_PATH
    elif os.path.exists(KEYS_EXAMPLE_PATH):
        yield KEYS_EXAMPLE_PATH


def _load_keys_json():
    """Try to load a keys JSON from candidate paths.

    Returns: (data_dict, used_path, error_str_or_empty)
    """
    last_err = ''
    for p in _iter_keys_paths():
        try:
            if not p or not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f) or {}
            if isinstance(data, dict):
                return data, p, ''
            last_err = f'Not a JSON object: {p}'
        except Exception as e:
            last_err = f'{p}: {e}'
    return {}, '', last_err

# Usage stats location: ./Logs/Usage_statistics/Comm-SCI-Use.txt
STATS_FILENAME = os.path.join(USAGE_LOG_DIR, 'Comm-SCI-Use.txt')
INPUT_HISTORY_FILENAME = 'InputLineHistory.json'
INPUT_HISTORY_PATH = os.path.join(HISTORY_LOG_DIR, INPUT_HISTORY_FILENAME)

# Backwards-compatible alias (some forks used STATS_PATH)
STATS_PATH = STATS_FILENAME

# ----------------------------
# UI (English-only)
# ----------------------------
UI_LANG = "en"  # hard-fixed; UI/system language (menus/help) independent of answer_language

QC_LABELS = {
    "clarity": "Clarity",
    "brevity": "Brevity",
    "evidence": "Evidence",
    "empathy": "Empathy",
    "consistency": "Consistency",
    "neutrality": "Neutrality",
}

CONTROL_LAYER_ALERTS_TITLE = "CONTROL LAYER ALERTS (Python)"
CSC_WARNING_TEXT = (
    "Warning: This answer contains complex claims/uncertainty. "
    "A cross-check is recommended under strict rules."
)

ANCHOR_TITLE = "ANCHOR SNAPSHOT"
ANCHOR_SUBTITLE = "Deterministic checkpoint (no LLM)."
ANCHOR_CHECKPOINT = "Checkpoint created."


def ui_onoff(v: str) -> str:
    return "on" if (v or "").strip().lower() == "on" else "off"


def ui_overlay(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("", "none", "off"):
        return "off"
    if v == "strict":
        return "Strict"
    if v == "explore":
        return "Explore"
    return v

# ----------------------------
# INPUT ROUTER (deterministic)
# ----------------------------
# NOTE: This must NOT assume a global 'gov' exists. Always resolve safely.

def route_input(raw_txt: str, state, api_instance, gov_manager=None) -> dict:
    """Deterministically route raw user input.

    Returns dict with keys:
      - kind: 'noop' | 'command' | 'chat' | 'error'
      - canonical_cmd (if command)
      - query_text (if chat)
      - is_sci_selection (optional)
      - html (if error)

    This mirrors the legacy behavior but is hardened against missing globals.
    """
    txt = (raw_txt or "").strip()
    if not txt:
        return {"kind": "noop"}

    # Resolve ruleset / commands safely (no assumptions about globals).
    # Prefer explicit injection; otherwise use the Api instance.
    gov_obj = gov_manager or getattr(api_instance, 'gov', None)
    commands = {}
    try:
        commands = (getattr(gov_obj, 'data', {}) or {}).get('commands', {}) or {}
    except Exception:
        commands = {}

    all_cmds = []
    try:
        for cat in (commands or {}).values():
            if isinstance(cat, dict):
                all_cmds.extend(list(cat.keys()))
    except Exception:
        all_cmds = []


    # Wrapper-local commands (not part of the ruleset JSON)
    for _c in ("QC Override",):
        try:
            if _c not in all_cmds:
                all_cmds.append(_c)
        except Exception:
            pass
    sci_pending = False
    try:
        sci_pending = bool(getattr(state, 'sci_pending', False))
    except Exception:
        sci_pending = False

    # Comm-off gate (strict parsing disable): only "Comm Start" is interpreted locally.
    # Everything else is passed through as plain chat so the LLM sees it as normal content.
    try:
        comm_active_now = bool(getattr(state, 'comm_active', True))
    except Exception:
        comm_active_now = False
    if not comm_active_now:
        if txt == 'Comm Start':
            return {'kind': 'command', 'canonical_cmd': 'Comm Start'}
        return {'kind': 'chat', 'txt': txt}

    # Standalone-only: exact command tokens only.
    # If a command token is mixed with additional text (e.g. "Profile Expert what is time?"),
    # we MUST NOT interpret it as a command.
    if txt in all_cmds:
        return {"kind": "command", "canonical_cmd": txt}

    # Mixed-command detection: if the input starts with a known command token followed by
    # whitespace and additional content, treat it as chat and report a deterministic violation.
    # Example: "Profile Expert what is time?" must NOT execute "Profile Expert".
    try:
        # Prefer the longest match (e.g. "Profile Expert" over "Profile")
        for cmd_tok in sorted(set(all_cmds), key=lambda s: len(str(s)), reverse=True):
            c = str(cmd_tok)
            if not c:
                continue
            if txt.startswith(c + " ") or txt.startswith(c + ":") or txt.startswith(c + " :"):
                return {
                    "kind": "chat",
                    "query_text": txt,
                    "standalone_only_violation": True,
                    "standalone_violation_cmd": c,
                }
    except Exception:
        pass

    # CGI feedback triplets (optional user feedback; should not trigger an LLM call).
    # Canonical JSON (v19.6.9+): global_defaults.user_feedback_triplet and global_defaults.process_cgi_feedback
    try:
        gd = (getattr(gov_obj, 'data', {}) or {}).get('global_defaults', {}) or {}
        # user_feedback_triplet: e.g., "3,3,3"
        uft = (gd.get('user_feedback_triplet') or {}) if isinstance(gd, dict) else {}
        if isinstance(uft, dict) and bool(uft.get('enabled', False)):
            uft_pat = str(uft.get('regex') or uft.get('pattern') or '').strip() or r'^\s*([0-3])\s*,\s*([0-3])\s*,\s*([0-3])\s*$'
            if re.fullmatch(uft_pat, txt):
                return {"kind": "chat", "query_text": txt, "is_user_feedback_triplet": True}

        # process_cgi_feedback: e.g., "SCI: 3,2,1"
        pcf = (gd.get('process_cgi_feedback') or {}) if isinstance(gd, dict) else {}
        if isinstance(pcf, dict) and bool(pcf.get('enabled', False)):
            pcf_pat = str(pcf.get('regex') or pcf.get('pattern') or '').strip()
            if pcf_pat and re.fullmatch(pcf_pat, txt):
                return {"kind": "chat", "query_text": txt, "is_process_cgi_feedback": True}
    except Exception:
        pass

    # SCI pending: single-letter variant selection (A–H).
    # This must be detected deterministically so a standalone letter does NOT call the model.
    if sci_pending and re.fullmatch(r'[A-Ha-h]', txt):
        return {"kind": "chat", "query_text": txt, "is_sci_selection": True}


    # Numeric codes (best-effort validation against canonical JSON).
    # We only treat short forms like "1-2" (1–2 digits each) as numeric codes AND only
    # if the INDEX matches a known category index. This avoids false positives like dates
    # ("2026-01") or ranges ("10-12").
    try:
        nc = ((getattr(gov_obj, 'data', {}) or {}).get('numeric_codes') or {}) if gov_obj is not None else {}
        if isinstance(nc, dict):
            cats = nc.get('categories') or []
            idx_set = set()
            if isinstance(cats, list):
                for cat in cats:
                    try:
                        idx_set.add(str((cat or {}).get('index')))
                    except Exception:
                        pass
            m = re.fullmatch(r'([0-9]{1,2})-([0-9]{1,2})', txt)
            if m:
                idx, opt = m.group(1), m.group(2)
                # Only enforce if the index exists in the canonical categories.
                if idx in idx_set:
                    valid = False
                    if isinstance(cats, list):
                        for cat in cats:
                            if str((cat or {}).get('index')) == idx:
                                options = (cat or {}).get('options') or {}
                                if isinstance(options, dict) and str(opt) in options:
                                    valid = True
                                break
                    if not valid:
                        err_html = (
                            '<div class="csc-warning" style="background:#fee; border-color:#c00; color:#a00;">'
                            '<b>CONTROL LAYER BLOCK:</b><br>'
                            + 'Invalid numeric code: ' + html.escape(txt)
                            + '<br>Valid format: INDEX-OPTION (e.g., 1-2).'
                            + '</div>'
                        )
                        return {"kind": "error", "html": err_html}
                    # Valid numeric code → treat as chat (the model decides how to use it)
                    return {"kind": "chat", "query_text": txt, "is_numeric_code": True}
                # Unknown index → treat as normal chat (no enforcement)
    except Exception:
        pass


    # Verification Gate (Control Layer)
    prof = "Standard"
    try:
        prof = getattr(state, 'active_profile', 'Standard') or 'Standard'
    except Exception:
        prof = 'Standard'

    if prof != "Sandbox":
        try:
            gate_error = api_instance.check_verification_route_gate(txt)
        except AttributeError:
            gate_error = None
        except Exception:
            gate_error = None

        if gate_error:
            err_html = (
                '<div class="csc-warning" style="background:#fee; border-color:#c00; color:#a00;">'
                '<b>CONTROL LAYER BLOCK:</b><br>' + html.escape(str(gate_error)) +
                '</div>'
            )
            return {"kind": "error", "html": err_html}

    return {"kind": "chat", "query_text": txt}



# ----------------------------
# STUFE 1: SCHEMA CONTRACTS (fail-soft; never raises)
# ----------------------------
_ALLOWED_ROUTE_KINDS = {"noop", "command", "chat", "error"}

def contract_route_shape(route: dict) -> bool:
    """Best-effort contract for route_input() outputs.

    This is intentionally conservative to avoid false positives:
    - only checks presence/type of the *core* fields for each route kind
    - does NOT enforce optional keys
    - never raises (returns False on exceptions)
    """
    try:
        if not isinstance(route, dict):
            return False
        kind = route.get("kind")
        if kind not in _ALLOWED_ROUTE_KINDS:
            return False
        if kind == "command":
            return isinstance(route.get("canonical_cmd"), str) and bool(route.get("canonical_cmd"))
        if kind == "chat":
            return isinstance(route.get("query_text"), str) and bool(route.get("query_text"))
        if kind == "error":
            return isinstance(route.get("html"), str) and bool(route.get("html"))
        return True
    except Exception:
        return False


def contract_ask_output_shape(out) -> bool:
    """Best-effort contract for Api.ask() outputs (permissive; never raises)."""
    try:
        if isinstance(out, str):
            return True
        if isinstance(out, dict):
            if "html" in out:
                return isinstance(out.get("html"), str) or out.get("html") is None
            if "text" in out:
                return isinstance(out.get("text"), str) or out.get("text") is None
            return True
        return False
    except Exception:
        return False

def get_api_key():
    """Return Gemini API key.

    Lookup order:
      1) ENV: GEMINI_API_KEY (preferred), then GOOGLE_API_KEY (legacy)
      2) Config/Comm-SCI-API-Keys.json (KEYS_PATH):
         - provider-structured: providers.gemini.api_key_plain / api_key_enc
         - legacy: GOOGLE_API_KEY / GOOGLE_API_KEY_ENC
    """
    # key file candidates may be overridden via ENV/override file

    # 1) Prefer env var (simplest + safest)
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env_key = os.environ.get(env_name)
        if env_key:
            print("[System] API key loaded from environment variable.")
            return env_key.strip()

    # 2) Fallback: key file (supports optional encryption)
    data, used_path, err = _load_keys_json()
    if err and not data:
        print(f"[System] Error reading key file: {err}")
    if data:
        try:

            # --- provider-structured keys (recommended for multi-provider builds) ---
            provs = data.get('providers') if isinstance(data, dict) else None
            if isinstance(provs, dict):
                g = provs.get('gemini') or provs.get('google') or {}
                if isinstance(g, dict):
                    # encrypted
                    enc = (g.get('api_key_enc') or '').strip()
                    salt = (g.get('api_key_salt') or '').strip()
                    scheme = (g.get('enc_scheme') or '').strip().lower()
                    if enc and salt and (scheme in {"fernet", ""}):
                        passphrase = (os.environ.get("COMM_SCI_KEY_PASSPHRASE") or "").strip()
                        key = _try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt)
                        if key:
                            print(f"[System] API key loaded from encrypted {KEYS_FILENAME} (providers.gemini).")
                            return key

                    # plaintext
                    key = (g.get('api_key_plain') or g.get('api_key') or '').strip()
                    if key:
                        print(f"[System] API key loaded from {KEYS_FILENAME} (providers.gemini plaintext).")
                        return key

            # --- legacy encrypted form ---
            enc = (data.get("GOOGLE_API_KEY_ENC") or "").strip()
            salt = (data.get("GOOGLE_API_KEY_SALT") or "").strip()
            scheme = (data.get("ENC_SCHEME") or "").strip().lower()
            if enc and salt and (scheme in {"fernet", ""}):
                passphrase = (os.environ.get("COMM_SCI_KEY_PASSPHRASE") or "").strip()
                key = _try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt)
                if key:
                    print(f"[System] API key loaded from encrypted {KEYS_FILENAME}.")
                    return key
                else:
                    print("[System] Encrypted key present, but decryption failed (missing passphrase or cryptography).")

            # --- legacy plaintext fallback ---
            key = (data.get("GOOGLE_API_KEY") or "").strip()
            if key:
                print(f"[System] API key loaded from {KEYS_FILENAME} (plaintext).")
                return key

        except Exception as e:
            # Keep going; caller may still proceed with other providers
            print(f"[System] Error processing keys from {used_path or KEYS_FILENAME}: {e}")

    return "" 

# --- CONFIG MANAGER ---
class ConfigManager:
    def __init__(self):
        # English-only build: no UI language switching or persisted language state.
        self.config = {
            "model": "gemini-2.0-flash",
            "active_provider": "gemini",
            "enforcement_policy": "audit_only",  # audit_only | strict_warn | strict_block
            "providers": {
              "gemini": {
                "default_model": "gemini-2.0-flash"
              },
              "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "api_key_plain": "",
                "api_key_enc": "",
                "default_model": "openai/gpt-4.1-mini",
                "app_referrer": "",
                "app_title": "Comm-SCI Desktop"
              }
            },
            "answer_language": "de",
            "language_policy_mode": "production",  # production | benchmark
            "rate_limit_enabled": True,
            "rate_limit_per_minute": 30,
            "rate_limit_per_hour": 120
        }
        # Warn only once per instance for config parse issues (avoid log spam)
        self._warned_load_error = False
        self._warned_save_error = False
        self.load()
        # Compatibility flag: some older code paths may check this.
        self.loaded = True

    def load(self):
        path = CONFIG_PATH
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.config.update(data)
                # Ignore/remove any persisted language key from older builds.
                self.config.pop("language", None)
            except Exception as e:
                if not getattr(self, "_warned_load_error", False):
                    print(f"[Config] Error: {e}")
                    self._warned_load_error = True
        else:
            self.save()

        # Startup defaults (requested): always start with Gemini + gemini-2.0-flash.
        # This is applied after loading (or creating) the config and will override any
        # previously persisted provider/model selection.
        try:
            changed = False
            if (self.config.get('active_provider') or '').strip().lower() != 'gemini':
                self.config['active_provider'] = 'gemini'
                changed = True

            provs = self.config.get('providers')
            if not isinstance(provs, dict):
                provs = {}
                self.config['providers'] = provs
                changed = True
            g = provs.get('gemini')
            if not isinstance(g, dict):
                g = {}
                provs['gemini'] = g
                changed = True
            if (g.get('default_model') or '').strip() != 'gemini-2.0-flash':
                g['default_model'] = 'gemini-2.0-flash'
                changed = True
            # Back-compat key for Gemini
            if (self.config.get('model') or '').strip() != 'gemini-2.0-flash':
                self.config['model'] = 'gemini-2.0-flash'
                changed = True
            mode = (self.config.get("language_policy_mode", "production") or "production").strip().lower()
            if mode not in ("production", "benchmark"):
                self.config["language_policy_mode"] = "production"
                changed = True

            if changed:
                self.save()
        except Exception:
            pass

    def save(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            if not getattr(self, "_warned_save_error", False):
                print(f"[Config] Save Error: {e}")
                self._warned_save_error = True
    def get_active_provider(self) -> str:
        """Return the currently active provider name."""
        try:
            p = (self.config.get('active_provider', 'gemini') or 'gemini').strip().lower()
            if p in ('hf', 'huggingface'):
                return 'huggingface'
            if p in ('gemini', 'openrouter', 'huggingface'):
                return p
            return 'gemini'
        except Exception:
            return 'gemini'

    def _config_path(self) -> str:
        """Return the current config path (dynamic; honors runtime CONFIG_DIR/CONFIG_FILENAME overrides)."""
        try:
            return os.path.join(CONFIG_DIR, CONFIG_FILENAME)
        except Exception:
            return CONFIG_PATH

    def _write_to_disk(self, path: str, payload: dict) -> None:
        """Atomic JSON write. Raises on failure."""
        # If callers pass the module-level CONFIG_PATH (which can become stale in tests),
        # prefer the dynamic path derived from current CONFIG_DIR/CONFIG_FILENAME.
        if not path or path == CONFIG_PATH:
            path = self._config_path()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


    def set_active_provider(self, provider: str):
        """Persist the currently active provider in the config.

        Must be safe to call repeatedly (panel refreshes can call it often).
        """
        provider = (provider or 'gemini').strip().lower()
        # Accept common aliases from the UI
        if provider in ('hf',):
            provider = 'huggingface'

        # No-op guard: avoid duplicate reconnect paths when provider already active
        cur = (self.get_active_provider() or 'gemini').strip().lower()
        if provider == cur:
            return {'ok': True, 'skipped': True, 'reason': 'already_active'}

        try:
            self.config['active_provider'] = provider
            self._write_to_disk(CONFIG_PATH, self.config)
            return {'ok': True, 'skipped': False}
        except Exception as e:
            # Keep UI resilient: persistence errors must not crash the app.
            return {'ok': False, 'error': str(e)}

    def _merged_provider_conf(self, provider: str) -> dict:
        """Merge provider config from Comm-SCI-Config.json and Comm-SCI-API-Keys.json (best-effort).

        Precedence: config.json overrides api-keys.json for overlapping keys.
        """
        provider = (provider or '').strip().lower()
        conf_cfg = {}
        try:
            provs = (self.cfg.config or {}).get('providers') or {}
            if isinstance(provs, dict):
                conf_cfg = (provs.get(provider) or {}) if isinstance(provs.get(provider) or {}, dict) else {}
        except Exception:
            conf_cfg = {}
        conf_keys = {}
        try:
            data, used_path, err = _load_keys_json()
            provs2 = (data.get('providers') or {}) if isinstance(data, dict) else {}
            if isinstance(provs2, dict):
                conf_keys = (provs2.get(provider) or {}) if isinstance(provs2.get(provider) or {}, dict) else {}
        except Exception:
            conf_keys = {}
        merged = {}
        if isinstance(conf_keys, dict):
            merged.update(conf_keys)
        if isinstance(conf_cfg, dict):
            merged.update(conf_cfg)
        return merged

    def get_provider_model(self, provider: str = '') -> str:
        """Get the default/selected model for a given provider (or active provider)."""
        provider = (provider or self.get_active_provider() or 'gemini').strip().lower()
        if provider in ('hf', 'huggingface'):
            provider = 'huggingface'
        try:
            provs = self.config.get('providers') or {}
            if isinstance(provs, dict):
                pconf = provs.get(provider) or {}
                if isinstance(pconf, dict):
                    m = (pconf.get('default_model') or '').strip()
                    if m:
                        return m
        except Exception:
            pass
        # Back-compat for Gemini
        try:
            if provider == 'gemini':
                return (self.config.get('model') or 'gemini-2.0-flash').strip()
        except Exception:
            pass
        return ''

    def set_provider_model(self, provider: str, model: str):
        try:
            provider = (provider or 'gemini').strip().lower()
            model = (model or '').strip()
            if not model:
                return
            provs = self.config.get('providers')
            if not isinstance(provs, dict):
                provs = {}
                self.config['providers'] = provs
            pconf = provs.get(provider)
            if not isinstance(pconf, dict):
                pconf = {}
                provs[provider] = pconf
            pconf['default_model'] = model
            # Back-compat key for Gemini
            if provider == 'gemini':
                self.config['model'] = model
            self.save()
        except Exception:
            pass

    def get_model(self):
        """Back-compat: return active provider model."""
        return self.get_provider_model(self.get_active_provider()) or self.config.get('model', 'gemini-2.0-flash')

    def set_model(self, model):
        """Back-compat: set model for the active provider."""
        self.set_provider_model(self.get_active_provider(), model)

    def get_answer_language(self):
        try:
            return (self.config.get("answer_language", "de") or "de").strip().lower()
        except Exception:
            return "de"

    def set_answer_language(self, lang: str):
        try:
            lang = (lang or "de").strip().lower()
            if lang not in ("en", "de"):
                lang = "de"
            self.config["answer_language"] = lang
            self.save()
        except Exception:
            pass

    def get_language_policy_mode(self) -> str:
        try:
            mode = (self.config.get("language_policy_mode", "production") or "production").strip().lower()
            if mode in ("production", "benchmark"):
                return mode
        except Exception:
            pass
        return "production"

    def set_language_policy_mode(self, mode: str):
        try:
            m = (mode or "production").strip().lower()
            if m not in ("production", "benchmark"):
                m = "production"
            self.config["language_policy_mode"] = m
            self.save()
        except Exception:
            pass


    # --- Window geometry persistence (optional) ---
    def get_panel_geom(self):
        return self.config.get("panel_geom", {}) or {}

    def set_panel_geom(self, geom: dict):
        if isinstance(geom, dict) and geom:
            self.config["panel_geom"] = geom
            self.save()

    def get_main_geom(self):
        return self.config.get("main_geom", {}) or {}

    def set_main_geom(self, geom: dict):
        if isinstance(geom, dict) and geom:
            self.config["main_geom"] = geom
            self.save()


    # Backward-compatible helpers used by older code paths
    def get_model(self) -> str:
        """Return the currently selected model for the active provider (backward-compat)."""
        try:
            return self.get_provider_model(self.get_active_provider())
        except Exception:
            return (self.config.get("model") or "gemini-2.0-flash")

    def set_model(self, model: str):
        """Set the model for the active provider (backward-compat)."""
        try:
            self.set_provider_model(self.get_active_provider(), model)
        except Exception:
            try:
                self.config["model"] = str(model or "").strip()
                self.save()
            except Exception:
                pass

cfg = ConfigManager()


def cfg_get_model() -> str:
    """Robustly get the current model from config (works even if ConfigManager lacks get_model)."""
    c = getattr(cfg, "config", {}) or {}
    try:
        p = (c.get("active_provider") or "gemini").strip().lower()
    except Exception:
        p = "gemini"
    prov = (c.get("providers") or {}).get(p, {}) if isinstance(c, dict) else {}
    return str((prov.get("default_model") or c.get("model") or "")).strip()

# --- GOVERNANCE MANAGER ---
class GovernanceManager:
    def __init__(self):
        self.raw_json = ""
        self.data = {}
        self.loaded = False
        self.logs = []
        self.current_filename = DEFAULT_JSON # Stores the current filename

    def log(self, msg):
        print(f"[System] {msg}")
        self.logs.append(msg)


    def _adapt_operational_rules_json(self, data: dict):
        """Best-effort adapter for operational v20.x rulesets to wrapper canonical shape.

        The runtime is historically built for canonical files with top-level keys like
        `commands`, `global_defaults`, and `version`. Operational files use
        `command_model`, `contracts`, and `source.version`.
        """
        try:
            if not isinstance(data, dict):
                return data, ""

            # Already canonical enough for the wrapper.
            if isinstance(data.get("commands"), dict):
                return data, ""

            cmd_model = data.get("command_model") or {}
            groups = cmd_model.get("groups") if isinstance(cmd_model, dict) else {}
            profiles = data.get("profiles")
            if not isinstance(groups, dict) or not isinstance(profiles, dict):
                return data, ""

            out = dict(data)

            # Map operational contracts to legacy/global_defaults access paths.
            if "global_defaults" not in out and isinstance(out.get("contracts"), dict):
                out["global_defaults"] = out.get("contracts") or {}

            # Bridge self-debunking module fields expected by legacy runtime helpers.
            # Operational v20 keeps the normative contract at:
            #   contracts.output_contract.self_debunking_contract
            # but legacy enforcement also requires:
            #   global_defaults.self_debunking.enabled + block.title
            try:
                gd = out.get("global_defaults")
                if isinstance(gd, dict):
                    oc_gd = gd.get("output_contract")
                    if isinstance(oc_gd, dict):
                        sdc = oc_gd.get("self_debunking_contract")
                        if isinstance(sdc, dict) and not isinstance(gd.get("self_debunking"), dict):
                            req_title = str(sdc.get("required_block_title") or "Self-Debunking").strip() or "Self-Debunking"
                            gd["self_debunking"] = {
                                "enabled": bool(sdc.get("enabled", False)),
                                "exceptions": [],
                                "block": {"title": req_title},
                            }
                            out["global_defaults"] = gd
            except Exception:
                pass

            src = out.get("source") or {}
            if not isinstance(src, dict):
                src = {}
            if "version" not in out:
                ver = (
                    src.get("version")
                    or src.get("operational_profile_version")
                    or src.get("canonical_version")
                    or ""
                )
                if ver:
                    out["version"] = str(ver)

            state_defaults = ((out.get("state_model") or {}).get("defaults") or {})
            if isinstance(state_defaults, dict):
                if "default_profile" not in out and state_defaults.get("default_profile"):
                    out["default_profile"] = state_defaults.get("default_profile")
                if "default_code" not in out and state_defaults.get("default_code"):
                    out["default_code"] = state_defaults.get("default_code")

            cmd_desc = cmd_model.get("command_descriptions") if isinstance(cmd_model, dict) else {}
            if not isinstance(cmd_desc, dict):
                cmd_desc = {}
            cmd_rc = cmd_model.get("command_output_contract_overrides") if isinstance(cmd_model, dict) else {}
            if not isinstance(cmd_rc, dict):
                cmd_rc = {}

            commands = {}
            for group_name, raw_tokens in groups.items():
                gname = str(group_name or "").strip()
                if not gname:
                    continue
                items = {}
                tokens = []
                if isinstance(raw_tokens, list):
                    tokens = [str(t).strip() for t in raw_tokens if str(t).strip()]
                elif isinstance(raw_tokens, dict):
                    tokens = [str(t).strip() for t in raw_tokens.keys() if str(t).strip()]
                for token in tokens:
                    row = {}
                    desc = cmd_desc.get(token)
                    if isinstance(desc, str) and desc.strip():
                        row["function"] = desc.strip()
                    rc = cmd_rc.get(token)
                    row["response_contract"] = rc if isinstance(rc, str) and rc.strip() else "status_and_qc_only"
                    items[token] = row
                commands[gname] = items

            # Ensure required command groups exist for the schema guard.
            for req in ("primary", "help_and_codes", "sci_control", "profile_control", "mode_control", "color_control"):
                if not isinstance(commands.get(req), dict):
                    commands[req] = {}
            out["commands"] = commands

            # Normalize operational SCI layout to the legacy wrapper shape.
            # v20 operational files expose `sci.variants` / `sci.menu_output`,
            # while wrapper runtime paths expect `sci.variant_menu.*`.
            try:
                sci = out.get("sci")
                if isinstance(sci, dict):
                    variant_menu = sci.get("variant_menu")
                    if not isinstance(variant_menu, dict):
                        variant_menu = {}

                    if (
                        ("variants" not in variant_menu)
                        and isinstance(sci.get("variants"), dict)
                    ):
                        variant_menu["variants"] = sci.get("variants")

                    if (
                        ("menu_output" not in variant_menu)
                        and isinstance(sci.get("menu_output"), dict)
                    ):
                        variant_menu["menu_output"] = sci.get("menu_output")

                    if variant_menu:
                        sci["variant_menu"] = variant_menu

                    # Bridge operational timeout config into legacy syntax_rules path.
                    vsel = sci.get("variant_selection")
                    if isinstance(vsel, dict):
                        syntax_rules = out.get("syntax_rules")
                        if not isinstance(syntax_rules, dict):
                            syntax_rules = {}
                        sp = syntax_rules.get("special_parsing")
                        if not isinstance(sp, dict):
                            sp = {}
                        if not isinstance(sp.get("sci_variant_selection"), dict):
                            timeout_turns = 2
                            timeout_turns_ext = 3
                            try:
                                timeout_turns = int(vsel.get("timeout_turns", 2) or 2)
                            except Exception:
                                timeout_turns = 2
                            try:
                                timeout_turns_ext = int(vsel.get("timeout_turns_extended", 3) or 3)
                            except Exception:
                                timeout_turns_ext = 3
                            sp["sci_variant_selection"] = {
                                "pattern": "^[A-Ha-h]$",
                                "on_match": "set_sci_variant_if_pending",
                                "only_when_sci_pending": True,
                                "timeout_turns": timeout_turns,
                                "timeout_turns_extended": timeout_turns_ext,
                                "extension_condition": str(vsel.get("extension_condition") or ""),
                                "default_variant_if_no_selection": str(vsel.get("default_variant_if_no_selection") or "A"),
                            }
                        syntax_rules["special_parsing"] = sp
                        out["syntax_rules"] = syntax_rules

                    out["sci"] = sci
            except Exception:
                pass

            # Strip deprecated Anchor auto aliases from operational transitions/tokens.
            try:
                cm = out.get("command_model") or {}
                dts = cm.get("deterministic_transitions") if isinstance(cm, dict) else None
                if isinstance(dts, list):
                    for tr in dts:
                        if not isinstance(tr, dict):
                            continue
                        on = tr.get("on")
                        if isinstance(on, list):
                            tr["on"] = [x for x in on if str(x) not in ("Anchor auto on", "Anchor auto off")]
                pc = out.get("parser_contract")
                if isinstance(pc, dict) and isinstance(pc.get("command_tokens"), list):
                    pc["command_tokens"] = [x for x in pc["command_tokens"] if str(x) not in ("Anchor auto on", "Anchor auto off")]
            except Exception:
                pass

            out["_schema_adapted_from"] = "operational_v20"
            return out, "Schema adapter: operational ruleset mapped to wrapper runtime schema."
        except Exception as e:
            return data, f"Schema adapter skipped (error): {e}"

    def _is_valid_rules_json(self, data: dict):
        """Minimal schema guard to prevent accidentally loading non-rule JSON (e.g., Comm-SCI-Config.json)."""
        try:
            if not isinstance(data, dict):
                return False, "Root is not an object."
            if not isinstance(data.get("commands"), dict):
                return False, "Missing/invalid key 'commands'."
            if not isinstance(data.get("profiles"), dict):
                return False, "Missing/invalid key 'profiles'."
            if "version" not in data:
                return False, "Missing key 'version'."

            cmds = data.get("commands") or {}
            required_groups = ["primary", "help_and_codes", "sci_control", "profile_control", "mode_control", "color_control"]
            missing = [g for g in required_groups if not isinstance(cmds.get(g), dict)]
            if missing:
                return False, f"Missing/invalid commands groups: {', '.join(missing)}."

            prim = cmds.get("primary") or {}
            if not ("Comm Start" in prim and "Comm Stop" in prim):
                return False, "Primary commands missing 'Comm Start'/'Comm Stop'."
            return True, ""
        except Exception as e:
            return False, f"Schema check crashed: {e}"

    def load_file(self, filename=None):
        """Lädt eine spezifische JSON Datei oder den Standard.
        Enthält einen Schema-Guard, damit nicht versehentlich Config-JSONs als Ruleset loaded werden."""
        target_file = filename if filename else self.current_filename

        self.log(f"Loading ruleset: {os.path.basename(target_file)}...")
        # Pfad auflösen (Absolut oder Relativ)
        resolved = target_file

        # 1) Wenn relativ: zuerst relativ zum Skriptverzeichnis auflösen (unterstützt z.B. ./JSON/...)
        if not os.path.isabs(resolved):
            candidate = os.path.join(SCRIPT_DIR, resolved)
            if os.path.exists(candidate):
                resolved = candidate

        # 2) Backwards-Fallback: alte Logik (nur basename im Skriptverzeichnis)
        if not os.path.exists(resolved):
            candidate = os.path.join(SCRIPT_DIR, os.path.basename(str(target_file)))
            if os.path.exists(candidate):
                resolved = candidate

        if not os.path.exists(resolved):
            self.log(f"ERROR: File {resolved} not found!")
            return False

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            data, adapt_msg = self._adapt_operational_rules_json(data)
            if isinstance(adapt_msg, str) and adapt_msg.strip():
                self.log(adapt_msg)

            ok, why = self._is_valid_rules_json(data)
            if not ok:
                self.log(f"JSON ERROR: File is not a Comm-SCI ruleset ({why})")
                return False

            # Commit only after successful validation
            self.raw_json = raw
            self.data = data
            self.loaded = True
            self.current_filename = resolved
            self.log(f"JSON OK: {len(self.data.get('profiles', {}))} profiles.")
            return True

        except Exception as e:
            self.log(f"JSON ERROR: {e}")
            return False

    def get_system_instruction(self):
        if not self.loaded: return "System Error."

        lang_instruction = (
            "IMPORTANT: Reply in the current conversation language for explanations and content answers. "
            "Keep canonical command tokens in English."
        )

        version_info = f"loaded_file: {os.path.basename(self.current_filename)}"
        return f"GOVERNANCE RULES ({WRAPPER_NAME} - {version_info}):\n{self.raw_json}\n\n--- LANGUAGE SETTING ---\n{lang_instruction}\n\nAdhere strictly to these rules."

    def get_ui_data(self):
        """Return UI data in the schema expected by the original HTML_PANEL.

        HTML_PANEL's buildUI() expects each section items to be either:
          - a string (used as both label and command), or
          - an object with keys: name, cmd, desc
        """

        def profile_desc(pname: str) -> str:
            try:
                return (self.data.get("profiles", {}) or {}).get(pname, {}).get("description", "") or ""
            except Exception:
                return ""

        def overlay_desc(token: str) -> str:
            try:
                mo = (self.data.get("components", {}) or {}).get("mode_overlays", {}) or {}
                if token.startswith("Strict"):
                    return mo.get("Strict", "") or ""
                if token.startswith("Explore"):
                    return mo.get("Explore", "") or ""
                return mo.get("None", "") or ""
            except Exception:
                return ""

        if not self.loaded:
            return {
                "loaded": False,
                "current_rule_file": os.path.basename(self.current_filename),
                "current_model": cfg_get_model(),
                "answer_language": getattr(cfg, "get_answer_language", lambda: "de")(),
                "language_policy_mode": getattr(cfg, "get_language_policy_mode", lambda: "production")(),
                "comm": [],
                "profiles": [],
                "sci": [],
                "overlays": [],
                "tools": [],
                "logs": self.logs,
            }

        commands = self.data.get("commands", {}) or {}

        # --- Comm core (strings) ---
        comm_cmds = []
        comm_cmds += list((commands.get("primary", {}) or {}).keys())
        comm_cmds += list((commands.get("help_and_codes", {}) or {}).keys())

        # Keep deterministic ordering, remove duplicates while preserving order
        seen = set()
        comm_cmds = [c for c in comm_cmds if not (c in seen or seen.add(c))]

        # --- Profiles (objects: name/cmd/desc) ---
        prof_keys = []
        if isinstance(self.data.get("profiles", None), dict):
            prof_keys = list(self.data["profiles"].keys())

        if not prof_keys:
            # Fallback: derive from profile_control commands ("Profile X")
            prof_tokens = list((commands.get("profile_control", {}) or {}).keys())
            for t in prof_tokens:
                if t.startswith("Profile "):
                    prof_keys.append(t.split(" ", 1)[1])

        if not prof_keys:
            prof_keys = ["Standard", "Expert"]

        profiles = [{
            "name": p,
            "cmd": f"Profile {p}",
            "desc": profile_desc(p)
        } for p in prof_keys]

        # --- SCI workflow (strings) ---
        sci_cmds = list((commands.get("sci_control", {}) or {}).keys())

        # --- Modes & overlays (objects: name/cmd/desc) ---
        overlay_tokens = list((commands.get("mode_control", {}) or {}).keys())
        overlays = [{
            "name": t,
            "cmd": t,
            "desc": overlay_desc(t) or ((commands.get("mode_control", {}) or {}).get(t, {}).get("function", "") or "")
        } for t in overlay_tokens]

        # --- Tools (strings) ---
        tools = []
        tools += list((commands.get("color_control", {}) or {}).keys())
        tools += list((commands.get("dynamic_control", {}) or {}).keys())

        return {
            "loaded": True,
            "version": self.data.get("version"),
            "current_rule_file": os.path.basename(self.current_filename),
            "current_model": cfg_get_model(),
            "answer_language": getattr(cfg, "get_answer_language", lambda: "de")(),
            "language_policy_mode": getattr(cfg, "get_language_policy_mode", lambda: "production")(),
            "comm": comm_cmds,
            "profiles": profiles,
            "sci": sci_cmds,
            "overlays": overlays,
            "tools": tools,
            "logs": self.logs,
        }

    def all_command_tokens(self):
        """Return the set of all canonical command tokens from commands.* (exact match)."""
        if not self.loaded:
            return set()
        cmds = self.data.get("commands", {}) or {}
        tokens = set()
        for group_obj in cmds.values():
            if isinstance(group_obj, dict):
                tokens.update(group_obj.keys())
        return tokens

    def suggest_nearest_command(self, user_input: str, cutoff: float = 0.84):
        """Return the nearest canonical token (or None) using a deterministic string similarity."""
        tokens = sorted(self.all_command_tokens())
        if not tokens:
            return None
        best = difflib.get_close_matches(user_input, tokens, n=1, cutoff=cutoff)
        return best[0] if best else None

    def validate_standalone_command(self, user_input: str):
        """Validate command tokens in standalone prompts (exact match). Returns (ok, canonical_or_none, error_msg)."""
        if not self.loaded:
            return True, None, ""
        txt = (user_input or "").strip()
        if not txt:
            return True, None, ""
        tokens = self.all_command_tokens()
        if txt in tokens:
            return True, txt, ""
        # Only enforce if it *looks* like the user attempted a command
        first = txt.split()[0]
        if first in {"Comm", "Profile", "SCI", "Strict", "Explore", "Color", "Dynamic", "Anchor"}:
            suggestion = self.suggest_nearest_command(txt)
            if suggestion:
                return False, None, f"Invalid command token: '{txt}'. Nearest canonical token: '{suggestion}'. Command tokens must match exactly."
            return False, None, f"Invalid command token: '{txt}'. This token is not defined in the canonical command set."
        return True, None, ""

    def get_profile_qc_target(self, profile_name: str):
        if not self.loaded:
            return {}
        prof = (self.data.get("profiles", {}) or {}).get(profile_name, {}) or {}
        return prof.get("qc_target", {}) or {}

    def _normalize_qc_key(self, k) -> str:
        s = ("" if k is None else str(k)).strip().lower()
        m = {
            "clarity":"clarity","brevity":"brevity","evidence":"evidence","empathy":"empathy","consistency":"consistency","neutrality":"neutrality",
            "bravity":"brevity",
            "klarheit":"clarity","kürze":"brevity","kuerze":"brevity","evidenz":"evidence","empathie":"empathy","konsistenz":"consistency","neutralität":"neutrality","neutralitaet":"neutrality",
        }
        return m.get(s, s)

    def normalize_qc_overrides(self, overrides: dict | None) -> dict:
        """Return overrides as canonical lowercase keys -> int (clamped 0..3)."""
        ov = overrides if isinstance(overrides, dict) else {}
        out = {}
        for k, v in (ov or {}).items():
            key = self._normalize_qc_key(k)
            try:
                iv = int(float(str(v).replace(",", ".").strip()))
            except Exception:
                continue
            if iv < 0:
                iv = 0
            if iv > 3:
                iv = 3
            out[key] = iv
        return out

    def get_effective_qc_corridor(self, profile_name: str, overrides: dict | None = None) -> dict:
        """Single Source of Truth: effective corridor dim->(mn,mx), incl. overrides as fixed [v..v]."""
        base = self.get_profile_qc_target(profile_name) or {}
        eff = {}
        if isinstance(base, dict):
            for k, v in base.items():
                key = self._normalize_qc_key(k)
                try:
                    lo, hi = v
                    eff[key] = (int(lo), int(hi))
                except Exception:
                    continue
        ov = self.normalize_qc_overrides(overrides)
        for k, iv in ov.items():
            eff[k] = (iv, iv)
        return eff

    def get_effective_qc_values(self, profile_name: str, overrides: dict | None = None) -> dict:
        """Effective target values using upper bound (override becomes that fixed value)."""
        corr = self.get_effective_qc_corridor(profile_name, overrides)
        return {k: int(hi) for k, (lo, hi) in (corr or {}).items()}


    def expected_qc_deltas(self, profile_name: str, current_values: dict, overrides: dict = None):
        """Compute expected deltas against the *effective* corridor (profile + overrides)."""
        corr = self.get_effective_qc_corridor(profile_name, overrides)
        out = {}
        for dim, c in (current_values or {}).items():
            key = self._normalize_qc_key(dim)
            corridor = corr.get(key)
            if not corridor:
                continue
            mn, mx = corridor
            try:
                c_int = int(c)
            except Exception:
                continue
            if c_int < mn:
                out[QC_LABELS.get(key, key)] = c_int - mn
            elif c_int > mx:
                out[QC_LABELS.get(key, key)] = c_int - mx
            else:
                out[QC_LABELS.get(key, key)] = 0
        return out


    def parse_qc_footer(self, text: str):
        """Extract QC current values and reported deltas from a model response."""
        if not text:
            return {}, {}
        # Try both "QC:" and "QC-Matrix:" lines (DE/EN labels)
        qc_line = None
        for line in text.splitlines()[::-1]:
            if line.strip().startswith("QC:") or line.strip().startswith("QC-Matrix:"):
                qc_line = line.strip()
                break
        if not qc_line:
            # sometimes embedded in a paragraph; try a regex
            m = re.search(r"(QC(?:-Matrix)?:\s*.+)$", text, re.M)
            qc_line = m.group(1).strip() if m else None
        if not qc_line:
            return {}, {}

        # Normalize separators
        parts = [p.strip() for p in qc_line.split("·")]
        label_map = {
            # EN
            "Clarity": "clarity",
            "Brevity": "brevity",
            "Bravity": "brevity",
            "Evidence": "evidence",
            "Empathy": "empathy",
            "Consistency": "consistency",
            "Neutrality": "neutrality",
            # DE
            "Klarheit": "clarity",
            "Kürze": "brevity",
            "Evidenz": "evidence",
            "Empathie": "empathy",
            "Konsistenz": "consistency",
            "Neutralität": "neutrality",
        }

        cur = {}
        delta = {}
        for p in parts:
            # strip leading QC:
            p2 = p.replace("QC:", "").replace("QC-Matrix:", "").strip()
            # match: Label <int> (Δ<int>)
            # Be tolerant to different delta glyphs (Δ U+0394 vs ∆ U+2206) and multi-digit numbers.
            m = re.match(r"^([A-Za-zÄÖÜäöüß]+)\s+(\d+)\s*(?:\((?:Δ|∆|delta)\s*([+-]?\d+)\))?\s*$", p2)
            if not m:
                # sometimes parentheses use unicode delta only: (Δ0) / (∆0)
                m = re.match(r"^([A-Za-zÄÖÜäöüß]+)\s+(\d+)\s*\((?:Δ|∆)\s*([+-]?\d+)\)\s*$", p2)
            if not m:
                continue
            lbl, v, d = m.group(1), int(m.group(2)), m.group(3)
            dim = label_map.get(lbl)
            if not dim:
                continue
            cur[dim] = v
            if d is not None:
                delta[dim] = int(d)
        return cur, delta

    # Compatibility: expose QC-delta enforcement as a method for tests/tools.
    # Core implementation stays a standalone helper (enforce_qc_footer_deltas).
    def enforce_qc_footer_deltas(self, text: str, profile_name: str) -> str:
        try:
            return enforce_qc_footer_deltas(text, self, profile_name)
        except Exception:
            return text

    def check_self_debunking(self, text: str, profile_name: str):
        if not self.loaded or not text:
            return None
        sd = (self.data.get("global_defaults", {}) or {}).get("self_debunking", {}) or {}
        if not sd.get("enabled", False):
            return None
        if profile_name in (sd.get("exceptions", []) or []):
            return None
        title = (sd.get("block", {}) or {}).get("title", "Self-Debunking")
        if title not in text:
            return f"Missing required '{title}' block."
        return None


gov = GovernanceManager()

# --- POST-PROCESSING HELPERS (deterministic rendering-only) ---

_EVIDENCE_COLOR = {
    "GREEN": "#2e7d32",
    "YELLOW": "#f9a825",
    "RED": "#c62828",
    "GRAY": "#616161",
}
_EVIDENCE_ICON = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
    "GRAY": "⚪",
}

def dedupe_qc_lines(text: str) -> str:
    """Remove redundant QC header line if a QC-Matrix footer is present."""
    if not text:
        return text
    lines = text.splitlines()
    has_footer = any(l.strip().startswith("QC-Matrix:") for l in lines)
    if not has_footer:
        return text
    out = []
    for l in lines:
        s = l.strip()
        if s.startswith("QC:") and not s.startswith("QC-Matrix:"):
            continue
        out.append(l)
    return "\n".join(out)



def enforce_qc_footer_deltas(text: str, gov_mgr_or_expected, profile_name: str = 'Standard') -> str:
    """Normalize QC footer values to ints and correct deltas against the expected corridor.

    Fix (Wrapper-132):
      - If no canonical 'QC-Matrix:' line exists but a QC summary line exists (e.g. "QC:" or "Profile: ... QC: ..."),
        we deterministically convert it into a canonical QC-Matrix footer with computed deltas.

    Args:
        text: The text that may contain a QC footer line.
        gov_mgr_or_expected: Either a corridor dict (dim->(min,max)) or a GovernanceManager-like object.
        profile_name: Profile name used when resolving corridor via the manager.
    """
    if not text:
        return text

    # Resolve expected corridor dict.
    expected = {}
    if isinstance(gov_mgr_or_expected, dict):
        expected = gov_mgr_or_expected
    else:
        obj = gov_mgr_or_expected
        if hasattr(obj, 'get_profile_qc_target'):
            try:
                expected = obj.get_profile_qc_target(profile_name) or {}
            except Exception:
                expected = {}
        elif hasattr(obj, 'profile_get_qc_target'):
            try:
                expected = obj.profile_get_qc_target(profile_name) or {}
            except Exception:
                expected = {}

    expected_norm = {}
    if isinstance(expected, dict):
        for k, v in expected.items():
            try:
                lo, hi = v
                expected_norm[str(k).strip().lower()] = (int(lo), int(hi))
            except Exception:
                continue

    # Map known labels (case-insensitive) to normalized keys.
    label_map = {
        'clarity': 'clarity',
        'brevity': 'brevity',
        'bravity': 'brevity',
        'evidence': 'evidence',
        'neutrality': 'neutrality',
        'consistency': 'consistency',
        'empathy': 'empathy',
        # DE
        'klarheit': 'clarity',
        'kürze': 'brevity',
        'kuerze': 'brevity',
        'evidenz': 'evidence',
        'empathie': 'empathy',
        'konsistenz': 'consistency',
        'neutralität': 'neutrality',
        'neutralitaet': 'neutrality',
    }

    # Canonical render labels (stable order)
    canon_order = [
        ('clarity', 'Clarity'),
        ('brevity', 'Brevity'),
        ('evidence', 'Evidence'),
        ('empathy', 'Empathy'),
        ('consistency', 'Consistency'),
        ('neutrality', 'Neutrality'),
    ]

    def _to_int_rating(value_raw: str):
        s = (value_raw or '').replace(',', '.').strip()
        try:
            f = float(s)
        except Exception:
            return None
        # Round half-up for positive ratings.
        iv = int(f + 0.5) if f >= 0 else int(f - 0.5)
        if iv < 0:
            iv = 0
        if iv > 3:
            iv = 3
        return iv

    def _expected_delta(val_int: int, corridor):
        if not corridor:
            return None
        lo, hi = corridor
        if val_int < lo:
            return val_int - lo
        if val_int > hi:
            return val_int - hi
        return 0

    # ----------------------------
    # Case 1: canonical QC-Matrix present -> normalize values + deltas in-place (existing behavior).
    # ----------------------------
    if 'QC-Matrix:' in text:
        entry_re = re.compile(
            r'(?P<label>[A-Za-zÄÖÜäöüß]+)\s*'
            r'(?P<value>\d+(?:[\.,]\d+)?)\s*'
            r'\(\s*Δ\s*(?P<delta>[+-]?\d+(?:[\.,]\d+)?)\s*\)',
            re.UNICODE,
        )

        def _repl(m: re.Match):
            label_raw = m.group('label')
            value_raw = m.group('value')
            delta_raw = m.group('delta')

            key = label_map.get(label_raw.strip().lower(), label_raw.strip().lower())
            val_int = _to_int_rating(value_raw)
            if val_int is None:
                return m.group(0)

            corr = expected_norm.get(key)
            d_corr = _expected_delta(val_int, corr)

            # Always normalize the numeric value; correct delta only if we have a corridor.
            if d_corr is None:
                d = delta_raw.replace(',', '.')
                if d.startswith('+'):
                    d = d[1:]
                return f"{label_raw} {val_int} (Δ{d})"

            sign = '+' if d_corr > 0 else ''
            return f"{label_raw} {val_int} (Δ{sign}{d_corr})"

        return entry_re.sub(_repl, text)

    # ----------------------------
    # Case 2: No QC-Matrix present -> try to canonicalize an alternative QC summary line.
    # ----------------------------
    # Find the *last* QC summary line of either form:
    #   - "QC: Clarity 3 · Brevity 1 · ..."
    #   - "Profile: Standard QC: Clarity 3 · Brevity 1 · ..."
    pat_profile = re.compile(r'(?im)^(?P<indent>\s*)Profile:\s*[^\n]*?\bQC\s*:\s*(?P<body>.*)\s*$', re.UNICODE)
    pat_qc = re.compile(r'(?im)^(?P<indent>\s*)QC\s*:\s*(?P<body>.*)\s*$', re.UNICODE)

    def _last_match(pat):
        ms = list(pat.finditer(text))
        return ms[-1] if ms else None

    m_prof = _last_match(pat_profile)
    m_qc = _last_match(pat_qc)
    if m_prof and m_qc:
        m_alt = m_prof if m_prof.start() >= m_qc.start() else m_qc
    else:
        m_alt = m_prof or m_qc

    if not m_alt:
        return text

    body = (m_alt.group('body') or '').strip()
    if not body:
        return text

    # Split items on common separators (providers often use "·" or ";" or "|").
    items = [p.strip() for p in re.split(r'[·;\|]+', body) if p.strip()]
    vals = {}
    for it in items:
        # tolerate "Label=2" / "Label: 2" / "Label 2"
        m_it = re.match(r'^\s*([A-Za-zÄÖÜäöüß]+)\s*(?:=|:)?\s*(\d+(?:[\.,]\d+)?)\s*$', it, re.UNICODE)
        if not m_it:
            continue
        lbl = (m_it.group(1) or '').strip()
        num = (m_it.group(2) or '').strip()
        key = label_map.get(lbl.lower())
        if not key:
            continue
        iv = _to_int_rating(num)
        if iv is None:
            continue
        vals[key] = iv

    # Only canonicalize when we have all canonical dimensions; otherwise do not invent anything.
    if not all(k in vals for k, _ in canon_order):
        return text

    # Build canonical QC-Matrix line with computed deltas.
    parts = []
    for k, disp in canon_order:
        iv = int(vals.get(k))
        d = _expected_delta(iv, expected_norm.get(k))
        if d is None:
            d = 0
        sign = '+' if d > 0 else ''
        parts.append(f"{disp} {iv} (Δ{sign}{d})")
    qc_line = "QC-Matrix: " + " · ".join(parts)

    indent = m_alt.group('indent') or ''
    qc_line = indent + qc_line

    # Replace the alternative summary line in-place so the footer is always present and canonical.
    new_text = text[:m_alt.start()] + qc_line + text[m_alt.end():]
    return new_text


def ensure_qc_footer_is_last(text: str) -> str:
    """Ensure QC-Matrix footer is the last block.

    Moves the last QC-Matrix line to the end. If the same line also contains a
    color-tagged answer marker like [GREEN], we keep that part in place and move
    only the QC portion.
    """
    if not text or 'QC-Matrix:' not in text:
        return text

    lines = text.splitlines(True)
    qc_idx = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*QC-Matrix\s*:', ln):
            qc_idx = i
    if qc_idx is None:
        return text

    ln = lines[qc_idx]
    m = re.search(r'\[(GREEN|YELLOW|RED)\]', ln)
    qc_part = ln
    keep_part = ''
    if m:
        pos = m.start()
        qc_part = ln[:pos].rstrip()
        keep_part = ln[pos:].lstrip()
    if keep_part:
        lines[qc_idx] = keep_part if keep_part.endswith('\n') else keep_part + '\n'
    else:
        lines.pop(qc_idx)

    base = ''.join(lines).rstrip()
    qc_part = (qc_part or '').strip()
    if not qc_part:
        return base + ('\n' if text.endswith('\n') else '')
    sep = '\n\n' if base and not base.endswith('\n\n') else ''
    out = base + sep + qc_part
    out = out.rstrip() + ('\n' if text.endswith('\n') else '')
    return out


def format_sci_menu(text: str) -> str:
    """
    Deterministic readability formatter for SCI selection menus.

    Goal: If the model prints options a) ... b) ... inline in one line or using separators,
    we render them as a vertical list (one option per line). This is *rendering-only*:
    - no command tokens are changed
    - no options are invented/removed
    """
    if not text:
        return text


def inject_minimal_self_debunking(text: str, *, title: str = "Self-Debunking", lang: str = "en") -> str:
    """Deterministically inject a minimal compliant Self-Debunking block (2 points).

    This is a last-resort guard used only when the ruleset requires Self-Debunking but
    the model output omitted it (and a single repair pass didn't fix it).
    The injected content avoids new factual claims; it only states generic limitations
    and next checks.
    """
    if not text:
        return text
    if title in text:
        return text

    block = ""
    lang_norm = (lang or "en").lower().strip()
    if lang_norm.startswith("de"):
        block = (
            f"\n\n{title}:\n\n"
            "1. **Schwäche**: Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen.\n"
            "   **Warum relevant**: Vereinfachungen können Randfälle oder alternative Deutungen verdecken.\n"
            "   **Prüfen/Widerlegen (nächster Schritt)**: Die zentralen Annahmen explizit machen und gegen Primärquellen/Definitionen prüfen.\n\n"
            "2. **Schwäche**: Die Antwort kann wichtige Gegenpositionen oder Unsicherheitsgrenzen auslassen.\n"
            "   **Warum relevant**: Fehlende Einschränkungen können die Gültigkeit überdehnen oder Sicherheit vortäuschen.\n"
            "   **Prüfen/Widerlegen (nächster Schritt)**: Mindestens ein starkes Gegenbeispiel ergänzen und prüfen, ob die Kernaussagen bestehen bleiben.\n"
        )
    else:
        block = (
            f"\n\n{title}:\n\n"
            "1. **Weakness**: The answer may rely on simplified framing or implicit assumptions.\n"
            "   **Why it matters**: Simplifications can hide edge-cases or alternative interpretations.\n"
            "   **What would verify/falsify (next check)**: Identify key assumptions and test them against primary sources or formal definitions.\n\n"
            "2. **Weakness**: The answer may omit important counter-perspectives or uncertainty boundaries.\n"
            "   **Why it matters**: Missing caveats can overstate confidence or applicability.\n"
            "   **What would verify/falsify (next check)**: Add at least one strong counter-example and check whether conclusions still hold.\n"
        )
    # Place block BEFORE QC-Matrix if present, else append.
    m = re.search(r"(?im)^\s*QC-Matrix:\s*.*$", text)
    if not m:
        return text.rstrip() + block
    insert_at = m.start()
    return text[:insert_at].rstrip() + block + "\n\n" + text[insert_at:].lstrip()

    # Quick pre-check: only attempt if the response likely contains an SCI menu or option run.
    # (This keeps accidental reformatting extremely unlikely.)
    menu_hint = re.search(r"\bSCI\b", text, re.IGNORECASE)
    if not menu_hint:
        return text

    # Find option markers like "a)" "b:" "c -" "d –"
    opt_pat = re.compile(r"(?im)(?:^|\s)([a-hA-H])\s*[\)\.:\-–]\s*")
    hits = list(opt_pat.finditer(text))
    if len(hits) < 3:
        return text  # not a menu (or too little signal)

    # Build segments from first option onwards, keep header prefix as-is
    first = hits[0].start(1)
    prefix = text[:first].rstrip()

    items = []
    for i, h in enumerate(hits):
        letter = h.group(1).lower()
        start = h.end()
        end = hits[i + 1].start(1) if i + 1 < len(hits) else len(text)
        body = text[start:end].strip()

        # If body begins with separator artifacts, clean lightly
        body = re.sub(r"^[·\|\-–\s]+", "", body).strip()
        # Collapse excessive internal whitespace
        body = re.sub(r"[ \t]{2,}", " ", body)

        # Keep empty bodies (rare) but still show the option line
        items.append((letter, body))

    # Render as markdown-like list; HTML_CHAT already formats lists nicely.
    rendered_lines = []
    if prefix:
        rendered_lines.append(prefix)
    rendered_lines.append("")  # blank line before list for readability

    for letter, body in items:
        if body:
            rendered_lines.append(f"- {letter}) {body}")
        else:
            rendered_lines.append(f"- {letter})")

    return "\n".join(rendered_lines)


def normalize_evidence_tags(text: str) -> str:
    """Normalize Evidence-Linker provenance formatting without inventing new information.

    Goal:
    - If the model outputs a bare tag like [GREEN] followed by an origin token (-TRAIN/-WEB/-DOC),
      collapse it deterministically into the canonical bracket form: [GREEN-TRAIN].
    - Remove the redundant standalone origin token afterwards.

    IMPORTANT: We do *not* add a provenance suffix if none was provided.
    """
    if not text:
        return text

    # Collapse patterns like: "[GREEN] 🟢 -TRAIN" or "[RED] -WEB" into "[GREEN-TRAIN] 🟢"
    pat = re.compile(
        r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)\]\s*(?P<emoji>[🟢🟡🔴⚪️])?\s*(?:[·•\-–—]+\s*)?(?P<orig>TRAIN|WEB|DOC)\b",
        re.IGNORECASE,
    )

    def _repl(m: re.Match) -> str:
        tag = m.group('tag').upper()
        orig = m.group('orig').upper()
        emoji = (m.group('emoji') or '').strip()
        out = f"[{tag}-{orig}]"
        if emoji:
            out += f" {emoji}"
        return out

    out = pat.sub(_repl, text)

    # Clean up common leftovers like "[GREEN-TRAIN] •" or stray "-TRAIN" tokens after bullets.
    out = re.sub(r"(?i)\b(?:\-\s*)?(TRAIN|WEB|DOC)\b", lambda m: m.group(0), out)  # noop placeholder (keeps case)
    out = re.sub(r"(?im)(\[(?:GREEN|YELLOW|RED|GRAY)\-[A-Z]+\])\s*[·•]\s*", r"\1 ", out)
    out = re.sub(r"(?im)\s+[·•]\s*(?=\-?(?:TRAIN|WEB|DOC)\b)", " ", out)
    out = re.sub(r"(?im)\s+\-\s*(TRAIN|WEB|DOC)\b", "", out)
    return out


def normalize_known_markdown_control_headings(text: str) -> str:
    """Normalize leaked Markdown heading markers for control and common subheadings.

    Deterministic scope:
    - Keep code fences untouched.
    - Normalize known control headings (SCI Trace, Self-Debunking, ...).
    - Normalize common content subheadings with ##/###/#### into bold lines.
    """
    if not text:
        return text

    try:
        parts = str(text).split("```")

        # Normalize strict control-heading whitelist first.
        pat_control = re.compile(
            r"(?im)^\s{0,3}#{1,6}\s*(Final Answer|SCI Trace|Self-Debunking|Selbst[- ]?Debunking)\s*:?\s*$"
        )

        def _repl_control(m: re.Match) -> str:
            title = (m.group(1) or "").strip()
            return f"<strong>{html.escape(title)}:</strong>"

        # Then normalize generic content subheadings (##/###/####) to bold text.
        # Keep this conservative: no empty headings, no very long lines.
        # Accept optional evidence/linker prefixes and bullets before hashes.
        pat_sub = re.compile(
            r"(?im)^\s{0,3}"
            r"(?:\[(?:GREEN|YELLOW|RED|GRAY)(?:-[A-Z0-9]+)*\]\s*)?"
            r"(?:[🟢🟡🔴⚪⚪️]\s*)?"
            r"(?:[•*\-]\s*)?"
            r"#{2,4}\s*([^\n#][^\n]{0,120}?)\s*$"
        )

        def _repl_sub(m: re.Match) -> str:
            title = (m.group(1) or "").strip()
            if not title:
                return m.group(0)
            # Avoid double conversion if a control heading already matched.
            if re.fullmatch(r"(?i)(Final Answer|SCI Trace|Self-Debunking|Selbst[- ]?Debunking)\s*:?", title):
                return f"<strong>{html.escape(title.rstrip(':'))}:</strong>"
            return f"<strong>{html.escape(title.rstrip(':'))}:</strong>"

        for i in range(0, len(parts), 2):
            parts[i] = pat_control.sub(_repl_control, parts[i])
            parts[i] = pat_sub.sub(_repl_sub, parts[i])

        return "```".join(parts)
    except Exception:
        return text


def unwrap_accidental_full_text_codefence(text: str) -> str:
    """Unwrap a full-response fenced code block when it clearly contains governance chat output.

    Some weaker models occasionally wrap normal prose/governance responses in ```text ... ```
    which causes the Markdown renderer to escape color spans and keep markdown markers visible.
    We only unwrap when the fenced payload contains strong wrapper-specific markers.
    """
    try:
        if not text:
            return text
        s = str(text).strip()
        m = re.match(r"(?is)^```(?:\s*(?:text|txt|markdown|md))?\s*\n(?P<body>.*?)(?:\n)?```\s*$", s)
        if not m:
            return text
        body = (m.group("body") or "").strip("\n")
        if not body:
            return text

        probe = str(body)
        probe_low = probe.lower()
        governance_markers = (
            "qc-matrix:" in probe_low
            or "self-debunking" in probe_low
            or "selbst-debunking" in probe_low
            or "sci trace" in probe_low
            or "active profile:" in probe_low
            or "profile standard" in probe_low
            or "<span style=" in probe_low
        )
        if not governance_markers:
            return text
        return body
    except Exception:
        return text


def strip_verification_route_display_lines(text: str) -> str:
    """Hide noisy verification-route marker lines from UI display.

    Contract markers may be useful for validation/audit but reduce readability in chat.
    This function is display-only and deterministic.
    """
    try:
        if not text:
            return text
        out_lines = []
        pats = [
            # Header variants
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Verification\s+Route(?:\s+Gate)?\s*:?.*$"),
            # Marker lines (EN/DE, with/without bullets)
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Source\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Measurement\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Contrast\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Web[\s\-]*Check\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Quelle\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Messung\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Kontrast\s*:.*$"),
            re.compile(r"(?im)^\s*(?:[-*•]\s*)?Web[\s\-]*Prüfung\s*:.*$"),
        ]
        for ln in str(text).splitlines():
            raw_ln = str(ln or "")
            plain_ln = re.sub(r"(?is)<[^>]+>", " ", raw_ln)
            plain_ln = html.unescape(plain_ln or "")
            plain_ln = re.sub(r"\s+", " ", plain_ln).strip()
            if any(p.match(plain_ln) for p in pats):
                continue
            out_lines.append(raw_ln)
        return "\n".join(out_lines)
    except Exception:
        return text


def strip_pathological_repetition_display_noise(text: str, *, lang: str = "de") -> str:
    """Best-effort display cleanup for obvious malformed long repetition sequences."""
    try:
        src = str(text or "")
        if not src:
            return src
        ll = str(lang or "").strip().lower()
        if ll.startswith(("zh", "ja", "ko")):
            return src

        pat = re.compile(r"[\u3400-\u9fff]{120,}")
        replacement = (
            "[removed: malformed repetition sequence]"
            if ll.startswith("en")
            else "[entfernt: fehlerhafte Wiederholungssequenz]"
        )
        return pat.sub(replacement, src)
    except Exception:
        return text


def strip_internal_scaffolding_status_lines(text: str) -> str:
    """Remove leaked internal status scaffold lines from model output (display-only).

    Deterministic and conservative:
    - removes explicit prompt-directive echoes (`[QC OVERRIDES]`, etc.)
    - removes clear status matrix lines (`Profile: ... · Overlay: ...`)
    - removes compact multi-line status blocks (`Profile:/Overlay:/SCI:`)
    - removes isolated `Profile: <known profile>` leak lines
    - code fences remain untouched
    """
    try:
        if not text:
            return text

        directive_pat = re.compile(
            r"(?i)^\s*(?:[-*•]\s*)?\[(?:output language|answer length|qc overrides|qc behavior|sci trace detail)\]"
        )
        status_line_pat = re.compile(
            r"(?i)^\s*(?:active profile|profile|overlay|sci|comm|control layer|qc|cgi|color)\s*:\s*.+$"
        )
        profile_only_pat = re.compile(
            r"(?i)^\s*profile\s*:\s*(?:standard|briefing|sandbox|sparring|expert)\s*\.?\s*$"
        )
        profile_plain_pat = re.compile(
            r"(?i)^\s*profile\s+(?:standard|briefing|sandbox|sparring|expert)\s*\.?\s*$"
        )
        profile_title_pat = re.compile(
            r"(?i)^\s*profile\s+(?:standard|briefing|sandbox|sparring|expert)\s*:\s*$"
        )
        profile_with_tail_pat = re.compile(
            r"(?i)^\s*profile\s*:\s*(?:standard|briefing|sandbox|sparring|expert)\s+(?P<tail>.+?)\s*$"
        )
        question_like_pat = re.compile(
            r"(?i)^(?:was|wie|warum|wieso|wer|wo|wann|welche?|what|why|how|who|where|when|which|is|are|can|could|should|do|does)\b"
        )
        compact_keys_pat = re.compile(r"(?i)^\s*(?:profile|overlay|sci)\s*:\s*.+$")

        def _is_prompt_echo_line(sline: str) -> bool:
            t = (sline or "").strip()
            if not t:
                return False
            tl = t.lower()
            if "?" in t:
                return True
            if question_like_pat.match(tl):
                return True
            return False

        def _is_block_scaffold(block_lines):
            if not block_lines:
                return False
            cleaned = [re.sub(r"\s+", " ", str(x or "")).strip() for x in block_lines if str(x or "").strip()]
            if len(cleaned) < 2:
                return False
            key_hits = sum(1 for ln in cleaned if compact_keys_pat.match(ln))
            if key_hits < 2:
                return False
            return all(status_line_pat.match(ln or "") for ln in cleaned)

        lines = str(text).splitlines()
        out_lines = []
        in_code = False
        i = 0
        n = len(lines)
        while i < n:
            ln = lines[i]
            s = (ln or "").strip()
            if s.startswith("```"):
                in_code = not in_code
                out_lines.append(ln)
                i += 1
                continue
            if in_code:
                out_lines.append(ln)
                i += 1
                continue

            # Remove compact leaked status blocks like:
            # Profile: ...
            # Overlay: ...
            # SCI: ...
            if status_line_pat.match(s or ""):
                j = i
                block = []
                while j < n:
                    cur = (lines[j] or "").strip()
                    if not cur:
                        break
                    if cur.startswith("```"):
                        break
                    if not status_line_pat.match(cur):
                        break
                    block.append(lines[j])
                    j += 1
                if _is_block_scaffold(block):
                    i = j
                    continue

            low = s.lower()
            if directive_pat.match(s):
                i += 1
                continue
            m_tail = profile_with_tail_pat.match(s)
            if m_tail:
                tail = (m_tail.group("tail") or "").strip()
                if _is_prompt_echo_line(tail):
                    i += 1
                    continue
            if profile_only_pat.match(s) or profile_plain_pat.match(s):
                # Optionally consume a leaked prompt-echo line directly below.
                j = i + 1
                while j < n and not (lines[j] or "").strip():
                    j += 1
                if j < n:
                    nxt = (lines[j] or "").strip()
                    if (
                        _is_prompt_echo_line(nxt)
                        and not status_line_pat.match(nxt)
                        and len(nxt) <= 600
                    ):
                        i = j + 1
                        continue
                i += 1
                continue
            if profile_title_pat.match(s):
                j = i + 1
                while j < n and not (lines[j] or "").strip():
                    j += 1
                if j < n:
                    nxt = (lines[j] or "").strip()
                    if (
                        _is_prompt_echo_line(nxt)
                        and not status_line_pat.match(nxt)
                        and len(nxt) <= 600
                    ):
                        i = j + 1
                        continue
                i += 1
                continue
            starts_status = low.startswith("profile:")
            starts_comm = low.startswith("comm:")
            has_sep = any(ch in s for ch in ("·", "•", "|"))
            token_count = sum(
                1
                for t in ("overlay", "sci", "color", "control layer", "qc", "cgi")
                if t in low
            )
            has_active_profile = "active profile" in low

            if starts_status and has_sep and token_count >= 3:
                i += 1
                continue
            if starts_comm and has_sep and has_active_profile and token_count >= 4:
                i += 1
                continue

            out_lines.append(ln)
            i += 1

        return "\n".join(out_lines)
    except Exception:
        return text


def strip_sci_trace_line_when_inactive(
    text: str,
    *,
    sci_active: bool = False,
    sci_variant: str = "",
    sci_pending: bool = False,
) -> str:
    """Remove leaked SCI Trace lines/sections when SCI is inactive.

    This is display cleanup only and intentionally conservative:
    - Runs only when SCI is effectively OFF (no active variant, not pending).
    - Keeps code fences untouched.
    """
    try:
        if not text:
            return text
        if bool(sci_active) or bool((sci_variant or "").strip()) or bool(sci_pending):
            return text

        src = str(text)

        # First remove section-style SCI Trace blocks (including numbered heading variants
        # like "4. SCI Trace (Variante A: Standard)") up to Self-Debunking/QC/Final Answer or end.
        src = re.sub(
            r"(?is)(?:^|\n)\s*(?:#+\s*)?(?:\d+[\.\)]\s*)?SCI\s*Trace(?:\s*\([^\n]*\))?\s*:?\s*\n.*?"
            r"(?=\n\s*(?:#+\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking|QC(?:-Matrix)?\s*:|Final\s+Answer)\b|\Z)",
            "\n",
            src,
        )

        out_lines = []
        in_code = False
        for ln in src.splitlines():
            s = (ln or "").strip()
            if s.startswith("```"):
                in_code = not in_code
                out_lines.append(ln)
                continue
            if in_code:
                out_lines.append(ln)
                continue

            if re.match(
                r"(?im)^\s*(?:#+\s*)?(?:\d+[\.\)]\s*)?SCI\s*Trace(?:\s*\([^\n]*\))?\s*:?\s*.*$",
                s,
            ):
                continue
            out_lines.append(ln)

        return "\n".join(out_lines)
    except Exception:
        return text


def strip_governance_scaffolding_when_comm_inactive(text: str) -> str:
    """Remove visible governance formatting blocks from normal answers when Comm is OFF.

    Rule intent for `Comm Stop`: rule system is inactive (Safety Core may stay active), so
    content answers should not display Comm-SCI governance formatting such as SCI Trace,
    Self-Debunking, QC footer lines, or wrapper status headers.

    This is a deterministic display cleanup on raw model text before Markdown/HTML rendering.
    """
    try:
        if not text:
            return text
        out = str(text)

        # Remove explicit wrapper/governance status header lines if they leak into model output.
        out = re.sub(
            r"(?im)^\s*Active profile\s*:\s*.*(?:Control Layer\s*:\s*on|QC\s*:\s*on|CGI\s*:\s*on).*$\n?",
            "",
            out,
        )

        # Remove plain-text SCI Trace blocks entirely (section style).
        out = re.sub(
            r"(?is)(?:^|\n)\s*(?:#+\s*)?SCI\s*Trace\s*:?\s*\n.*?(?=\n\s*(?:#+\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking|QC(?:-Matrix)?\s*:)\b|\Z)",
            "\n",
            out,
        )
        # Also remove single leaked SCI Trace lines (conservative fallback).
        out = re.sub(r"(?im)^\s*SCI\s*Trace\s*:\s*.*$\n?", "", out)

        # Remove Self-Debunking sections (EN/DE) up to the next QC footer or end.
        out = re.sub(
            r"(?is)(?:^|\n)\s*(?:#+\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:?\s*\n.*?(?=\n\s*QC(?:-Matrix)?\s*:|\Z)",
            "\n",
            out,
        )

        # Remove QC footer / summary lines.
        out = re.sub(r"(?im)^\s*QC(?:-Matrix)?\s*:.*$\n?", "", out)

        # Collapse excessive blank lines created by removals.
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out
    except Exception:
        return text


def strip_internal_scaffolding_status_html(html_text: str) -> str:
    """Fallback cleanup: remove leaked scaffold status lines from rendered HTML blocks.

    This runs post-render and only removes full block nodes (<p>/<div>/<li>) that
    look like internal status scaffolding.
    """
    try:
        if not html_text:
            return html_text

        block_pat = re.compile(r"(?is)<(p|div|li)\b[^>]*>(.*?)</\1>")

        def _is_scaffold(inner_html: str) -> bool:
            txt_lines = re.sub(r"(?is)<br\s*/?>", "\n", inner_html or "")
            txt_lines = re.sub(r"(?is)</(?:p|div|li|tr)>", "\n", txt_lines)
            txt_lines = re.sub(r"(?is)<[^>]+>", " ", txt_lines)
            txt_lines = html.unescape(txt_lines or "")
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in str(txt_lines).splitlines() if ln.strip()]
            txt = re.sub(r"\s+", " ", " ".join(lines)).strip()
            low = txt.lower()
            if re.match(
                r"(?i)^\s*(?:[-*•]\s*)?\[(?:output language|answer length|qc overrides|qc behavior|sci trace detail)\]",
                txt,
            ):
                return True

            status_line_pat = re.compile(
                r"(?i)^\s*(?:active profile|profile|overlay|sci|comm|control layer|qc|cgi|color)\s*:\s*.+$"
            )
            profile_only_pat = re.compile(
                r"(?i)^\s*profile\s*:\s*(?:standard|briefing|sandbox|sparring|expert)\s*\.?\s*$"
            )
            profile_plain_pat = re.compile(
                r"(?i)^\s*profile\s+(?:standard|briefing|sandbox|sparring|expert)\s*\.?\s*$"
            )
            profile_title_pat = re.compile(
                r"(?i)^\s*profile\s+(?:standard|briefing|sandbox|sparring|expert)\s*:\s*$"
            )
            profile_with_tail_pat = re.compile(
                r"(?i)^\s*profile\s*:\s*(?:standard|briefing|sandbox|sparring|expert)\s+(?P<tail>.+?)\s*$"
            )
            compact_keys_pat = re.compile(r"(?i)^\s*(?:profile|overlay|sci)\s*:\s*.+$")
            question_like_pat = re.compile(
                r"(?i)^(?:was|wie|warum|wieso|wer|wo|wann|welche?|what|why|how|who|where|when|which|is|are|can|could|should|do|does)\b"
            )

            def _is_prompt_echo_line(sline: str) -> bool:
                t = (sline or "").strip()
                if not t:
                    return False
                tl = t.lower()
                if "?" in t:
                    return True
                if question_like_pat.match(tl):
                    return True
                return False

            if profile_only_pat.match(txt or ""):
                return True
            if profile_plain_pat.match(txt or ""):
                return True
            if profile_title_pat.match(txt or ""):
                return True
            m_tail = profile_with_tail_pat.match(txt or "")
            if m_tail and _is_prompt_echo_line((m_tail.group("tail") or "").strip()):
                return True
            if len(lines) >= 2 and (
                profile_only_pat.match(lines[0] or "")
                or profile_plain_pat.match(lines[0] or "")
                or profile_title_pat.match(lines[0] or "")
            ):
                nxt = (lines[1] or "").strip()
                if (
                    _is_prompt_echo_line(nxt)
                    and not status_line_pat.match(nxt)
                    and len(nxt) <= 600
                ):
                    return True
                if re.match(r"(?i)^\s*SCI\s*Trace\s*:?\s*$", nxt or ""):
                    return True
            if len(lines) >= 2:
                key_hits = sum(1 for ln in lines if compact_keys_pat.match(ln or ""))
                if key_hits >= 2 and all(status_line_pat.match(ln or "") for ln in lines):
                    return True

            starts_status = low.startswith("profile:")
            starts_comm = low.startswith("comm:")
            has_sep = any(ch in txt for ch in ("·", "•", "|"))
            token_count = sum(
                1
                for t in ("overlay", "sci", "color", "control layer", "qc", "cgi")
                if t in low
            )
            has_active_profile = "active profile" in low
            if starts_status and has_sep and token_count >= 3:
                return True
            if starts_comm and has_sep and has_active_profile and token_count >= 4:
                return True
            return False

        def _repl(m: re.Match) -> str:
            inner = m.group(2) or ""
            return "" if _is_scaffold(inner) else m.group(0)

        out = block_pat.sub(_repl, str(html_text))
        return out
    except Exception:
        return html_text


def strip_exact_status_header_line(text: str, header_line: str) -> str:
    """Remove exact copies of the canonical header line from model output.

    This ensures the wrapper can prepend exactly one authoritative header line.
    """
    try:
        if not text or not header_line:
            return text
        out = []
        target = str(header_line).strip()
        for ln in str(text).splitlines():
            if (ln or "").strip() == target:
                continue
            out.append(ln)
        return "\n".join(out)
    except Exception:
        return text


def sanitize_self_debunking_markdown_in_html(html_text: str) -> str:
    """Normalize leaked markdown emphasis inside already-rendered Self-Debunking HTML blocks.

    Deterministic scope:
    - only runs if a self-debunking block exists
    - converts `**label**` / `__label__` to `<strong>label</strong>`
    - formatting only (no semantic rewrites)
    """
    try:
        if not html_text:
            return html_text
        if re.search(r"(?is)class=(?:\"|')[^\"']*self-debunking[^\"']*(?:\"|')", html_text) is None:
            return html_text
        out = re.sub(r"\*\*([^*\n][^*\n]*?)\*\*", r"<strong>\1</strong>", html_text)
        out = re.sub(r"__([^_\n][^_\n]*?)__", r"<strong>\1</strong>", out)
        # Remove orphan markdown bullet artifacts like "*<br>" inside full Self-Debunking blocks.
        sd_block_re = re.compile(
            r'(?is)<div[^>]*class=(?:"|\')[^"\']*self-debunking[^"\']*(?:"|\')[^>]*>.*?</div>(?=\s*<(?:p|div)\b|\s*\Z)'
        )

        def _clean_sd_block(m: re.Match) -> str:
            block = str(m.group(0) or "")
            block = re.sub(r"(?im)\s*\*\s*(?=<br\s*/?>)", "", block)
            block = re.sub(r"(?im)(^|>\s*)\*\s*(?=(?:<strong>|[A-Za-zÄÖÜäöü]))", r"\1", block)
            block = re.sub(r"(?im)\n[ \t]*\*[ \t]*(?=\n)", "\n", block)
            return block

        out = sd_block_re.sub(_clean_sd_block, out)
        return out
    except Exception:
        return html_text


def qc_override_runtime_violations(text: str, overrides: dict | None) -> list[str]:
    """Deterministic best-effort checks for active QC overrides against runtime output."""
    try:
        ov = overrides if isinstance(overrides, dict) else {}
        if not ov:
            return []

        canon = {
            "clarity": "clarity",
            "brevity": "brevity",
            "evidence": "evidence",
            "empathy": "empathy",
            "consistency": "consistency",
            "neutrality": "neutrality",
            "klarheit": "clarity",
            "kürze": "brevity",
            "kuerze": "brevity",
            "evidenz": "evidence",
            "empathie": "empathy",
            "konsistenz": "consistency",
            "neutralität": "neutrality",
            "neutralitaet": "neutrality",
        }
        ov_clean = {}
        for k, v in ov.items():
            kk = canon.get(str(k or "").strip().lower())
            if not kk:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            ov_clean[kk] = max(0, min(3, iv))
        if not ov_clean:
            return []

        raw = str(text or "")
        plain = re.sub(r"<[^>]+>", " ", raw)
        plain = re.sub(r"[ \t]+", " ", plain)
        filtered = []
        for ln in plain.splitlines():
            s = ln.strip()
            if not s:
                continue
            if re.match(r"(?i)^(QC(?:-Matrix)?|Self-?Debunking|Selbst-?Debunking|SCI Trace)\b", s):
                continue
            if re.match(r"(?i)^(Plan|Solution|Check|Critic|Linguist|Logician|Adversary|Learn|Dialectic_[A-Za-z0-9_]+)\s*:?\s*$", s):
                continue
            filtered.append(s)
        probe = " ".join(filtered).strip() or plain.strip()

        words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", probe)
        wc = len(words)
        sc = max(1, len(re.findall(r"[.!?](?:\s|$)", probe)))
        avg_sent = (float(wc) / float(sc)) if sc else float(wc)

        def _obs_brevity() -> int:
            if wc >= 260:
                return 0
            if wc >= 170:
                return 1
            if wc >= 90:
                return 2
            return 3

        def _obs_clarity() -> int:
            has_structure = bool(re.search(r"(?im)^\s*(?:[-*•]|\d+\.)\s+", raw))
            if has_structure and 8 <= avg_sent <= 24 and wc >= 80:
                return 3
            if wc >= 60 and 7 <= avg_sent <= 28:
                return 2
            if wc >= 30:
                return 1
            return 0

        def _obs_evidence() -> int:
            c = 0
            c += len(re.findall(r"(?i)\b(Source|Measurement|Contrast|Web-?Check)\s*:", raw))
            c += len(re.findall(r"\[(?:GREEN|YELLOW|RED|GRAY)(?:-(?:TRAIN|WEB|DOC))?\]", raw))
            c += len(re.findall(r"https?://", raw))
            if c >= 6:
                return 3
            if c >= 3:
                return 2
            if c >= 1:
                return 1
            return 0

        def _obs_empathy() -> int:
            c = len(re.findall(r"(?i)\b(Ich verstehe|I understand|gerne|helpful|hilfreich|danke|thanks)\b", probe))
            if c >= 3:
                return 3
            if c >= 1:
                return 2
            return 1

        def _obs_consistency() -> int:
            contradictions = [
                r"(?i)\b(always|immer)\b.*\b(never|niemals)\b",
                r"(?i)\bist\b.*\bist nicht\b",
                r"(?i)\bis\b.*\bis not\b",
            ]
            return 1 if any(re.search(p, probe) for p in contradictions) else 3

        def _obs_neutrality() -> int:
            loaded = len(re.findall(r"(?i)\b(unglaublich|katastrophal|lächerlich|idiotisch|ridiculous|disaster|obviously|clearly)\b", probe))
            if loaded == 0:
                return 3
            if loaded <= 2:
                return 2
            if loaded <= 4:
                return 1
            return 0

        obs_map = {
            "brevity": _obs_brevity(),
            "clarity": _obs_clarity(),
            "evidence": _obs_evidence(),
            "empathy": _obs_empathy(),
            "consistency": _obs_consistency(),
            "neutrality": _obs_neutrality(),
        }
        labels = {
            "brevity": "Brevity",
            "clarity": "Clarity",
            "evidence": "Evidence",
            "empathy": "Empathy",
            "consistency": "Consistency",
            "neutrality": "Neutrality",
        }
        out = []
        for k, target in ov_clean.items():
            observed = int(obs_map.get(k, target))
            if observed != target:
                out.append(
                    f"QC-Override mismatch ({labels.get(k,k)}): target={target}, observed={observed} (deterministic runtime check)."
                )
        return out
    except Exception:
        return []


def normalize_self_debunking_language(text: str, lang: str) -> str:
    """Translate Self-Debunking label tokens into the target language (currently DE),
    without changing the required header 'Self-Debunking' or adding new factual claims.

    This is a deterministic post-processing step for models that keep English label words
    (e.g., 'Weakness', 'Why it matters', 'What would verify/falsify (next check)') even
    when answer_language=de.
    """
    try:
        if not text or not lang:
            return text
        if not str(lang).lower().startswith("de"):
            return text

        # Isolate the Self-Debunking block up to the QC footer (or end of text).
        # We keep the section header unchanged but translate common label tokens inside.
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text

        block = m.group(1)
        tail_marker = m.group(2)  # either QC footer marker or end

        # Translate label phrases (keep punctuation/colon style flexible).
        repl = [
            (r"(?i)\bWeakness\b\s*:", "Schwäche:"),
            (r"(?i)\bWhy\s+it\s+matters\b\s*:", "Warum das wichtig ist:"),
            (r"(?i)\bWhat\s+would\s+verify\s*/\s*falsify\s*\(next\s+check\)\s*:",
             "Was würde verifizieren/falsifizieren (nächster Check):"),
            (r"(?i)\bWhat\s+would\s+verify\s+or\s+falsify\s*\(next\s+check\)\s*:",
             "Was würde verifizieren oder falsifizieren (nächster Check):"),
            (r"(?i)\bNext\s+check\b\s*:", "Nächster Check:"),
        ]
        for pat, rep in repl:
            block = re.sub(pat, rep, block)

        # Reassemble
        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def bold_self_debunking_labels(text: str, lang: str) -> str:
    """Bold the label token before the first colon inside Self-Debunking points.

    Deterministic post-processing. Formatting only.
    """
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text
        block = m.group(1)

        # 1) Numbered point heads: "1. Weakness:" -> "1. **Weakness**:"
        def _bold_head(m2):
            lead = m2.group(1)
            label = (m2.group(2) or "").strip()
            if not label:
                return m2.group(0)
            if label.startswith("**") and label.endswith("**"):
                return m2.group(0)
            return f"{lead}**{label}**:"
        block = re.sub(r"(?m)^(\s*\d+\.\s*)([^\n:<]{1,80}?)(\s*):", lambda m2: _bold_head(m2), block)

        # 2) Field labels (possibly indented): "Why it matters:" -> "**Why it matters**:"
        labels = [
            "Weakness", "Schwäche",
            "Why it matters", "Warum relevant", "Warum das wichtig ist",
            "What would verify/falsify (next check)", "What would verify or falsify (next check)",
            "Was würde verifizieren/falsifizieren (nächster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)",
            "Next check", "Nächster Check",
            "Prüfen/Widerlegen (nächster Schritt)",
        ]
        for lab in labels:
            block = re.sub(
                rf"(?m)^(\s*)(?!\*\*){re.escape(lab)}\s*:(?!\*)",
                rf"\1**{lab}**:",
                block
            )

        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def normalize_self_debunking_field_linebreaks(text: str, *, lang: str = "en") -> str:
    """Ensure secondary Self-Debunking field labels start on a new line (no extra numbering).

    This keeps formatting stable when weaker models place
    "Warum das wichtig ist:" / "What would verify..." inline behind the Weakness sentence.
    """
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text
        block = m.group(1)

        labels = [
            "Why it matters", "Warum relevant", "Warum das wichtig ist",
            "What would verify/falsify (next check)", "What would verify or falsify (next check)",
            "Was würde verifizieren/falsifizieren (nächster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)",
            "Next check", "Nächster Check", "Nächste Prüfung",
            "Prüfen/Widerlegen (nächster Schritt)",
        ]
        labels_rx = "|".join(re.escape(x) for x in labels)
        # Insert a line break before secondary field labels if they leak inline.
        block = re.sub(
            rf"(?i)([^\n])\s+(?=(?:\*\*|__)?(?:{labels_rx})(?:\*\*|__)?\s*:)",
            r"\1\n   ",
            block,
        )

        start, end = m.span(1)
        return text[:start] + block + text[end:]
    except Exception:
        return text


def normalize_self_debunking_numbering_text(text: str, *, lang: str = "en") -> str:
    """Ensure stable numbered Self-Debunking points in plain text before HTML rendering."""
    try:
        if not text:
            return text
        m = re.search(r"(?is)(\b(?:Self-Debunking|Selbst[- ]?Debunking)\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text

        block = m.group(1)
        lines = block.splitlines()
        if not lines:
            return text

        out = []
        n = 0
        # Keep title line unchanged
        out.append(lines[0])
        for ln in lines[1:]:
            s = (ln or "").strip()
            # Remove orphan marker lines that cause visible double numbering.
            if re.fullmatch(r"\d+\.", s or ""):
                continue

            # Strip leaked list prefixes in front of known Self-Debunking labels.
            # Weakness/Schwäche lines get renumbered deterministically below.
            ln = re.sub(
                r"(?im)^\s*\d+\.\s*(?=(?:\*\*|__)?(?:Weakness|Schwäche|Why it matters|Warum relevant|Warum das wichtig ist|What would verify/falsify \(next check\)|What would verify or falsify \(next check\)|Was würde verifizieren/falsifizieren \(nächster Check\)|Was würde verifizieren oder falsifizieren \(nächster Check\)|Next check|Nächster Check|Prüfen/Widerlegen \(nächster Schritt\))(?:\*\*|__)?\s*:)",
                "",
                ln,
                count=1,
            )

            if lang.lower().startswith("de"):
                ln = re.sub(r"(?i)\bWeakness\b\s*:", "Schwäche:", ln)
                ln = re.sub(r"(?i)\bWhy\s+it\s+matters\b\s*:", "Warum das wichtig ist:", ln)
                ln = re.sub(
                    r"(?i)\bWhat\s+would\s+verify\s*/\s*falsify\s*\(next\s+check\)\s*:",
                    "Was würde verifizieren/falsifizieren (nächster Check):",
                    ln,
                )

            # Number only the Weakness/Schwäche lead lines.
            if re.match(
                r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*|__)?(?:Weakness|Schwäche)(?:\*\*|__)?\s*:",
                ln or "",
            ):
                n += 1
                ln = re.sub(r"(?im)^\s*\d+\.\s*", "", ln, count=1)
                ln = f"{n}. {ln.lstrip()}"
            out.append(ln)

        start, end = m.span(1)
        return text[:start] + "\n".join(out) + text[end:]
    except Exception:
        return text


def normalize_inline_self_debunking_header(text: str) -> str:
    """Ensure Self-Debunking header starts on its own line for deterministic boxing."""
    try:
        if not text:
            return text
        out = str(text)
        # If title leaks inline after a sentence, split into a new block.
        out = re.sub(
            r"([^\n])\s+((?:SCI\s*Trace\s*:\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:)",
            r"\1\n\n\2",
            out,
            flags=re.IGNORECASE,
        )
        # If title line has immediate inline body, push the body to the next line.
        out = re.sub(
            r"(?im)^(\s*(?:SCI\s*Trace\s*:\s*)?(?:Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:)\s+(?=\S)",
            r"\1\n",
            out,
        )
        return out
    except Exception:
        return text


def dedupe_self_debunking_sections(text: str) -> str:
    """Keep exactly one Self-Debunking section when duplicates leak from weaker models.

    Rules (deterministic):
    - Detect section headers:
      - "Self-Debunking:" / "Selbst-Debunking:"
      - "SCI Trace: Self-Debunking:" / "SCI Trace: Selbst-Debunking:"
    - If multiple are present, keep the last *pure* Self-Debunking section.
      If none is pure, keep the last detected section.
    - Section ends at next SD header, QC-Matrix header, or end of text.
    """
    try:
        if not text:
            return text

        lines = str(text).splitlines()
        if not lines:
            return text

        re_sd_header = re.compile(
            r"(?i)^\s*(?:(?P<sci>SCI\s*Trace)\s*:\s*)?(?P<sd>Self[- ]?Debunking|Selbst[- ]?Debunking)\s*:\s*$"
        )
        re_qc = re.compile(r"(?i)^\s*QC(?:-Matrix)?\s*:")

        starts = []
        for i, ln in enumerate(lines):
            m = re_sd_header.match((ln or "").strip())
            if m:
                starts.append((i, bool(m.group("sci"))))

        if len(starts) <= 1:
            return text

        # Build section ranges [start, end)
        ranges = []
        for idx, (start_i, is_sci_prefixed) in enumerate(starts):
            end_i = len(lines)
            for j in range(start_i + 1, len(lines)):
                s = (lines[j] or "").strip()
                if re_sd_header.match(s) or re_qc.match(s):
                    end_i = j
                    break
            ranges.append((start_i, end_i, is_sci_prefixed))

        # Keep last pure SD section if available; else last section.
        keep = None
        for r in ranges:
            if not r[2]:
                keep = r
        if keep is None:
            keep = ranges[-1]

        # Remove all non-kept SD ranges.
        remove_mask = [False] * len(lines)
        for r in ranges:
            if r is keep:
                continue
            for k in range(r[0], r[1]):
                remove_mask[k] = True

        out_lines = [ln for i, ln in enumerate(lines) if not remove_mask[i]]
        return "\n".join(out_lines)
    except Exception:
        return text


def html_number_self_debunking(html_text: str, *, lang: str = "en") -> str:
    """Best-effort: add stable 1./2./3. numbering to Self-Debunking in rendered HTML.

    Safety goals:
    - Never inject numbering inside existing <ol> lists.
    - Remove orphan list-marker lines like "1." that can appear after Markdown conversion.
    - Keep exactly one numeric prefix on each Weakness/Schwäche line.
    """
    try:
        if not html_text:
            return html_text
        # Normalize compact one-line HTML so line-wise processing can see block rows.
        html_text = str(html_text).replace("><", ">\n<")
        if re.search(r"(?i)Self-Debunking|Selbst[- ]?Debunking", html_text) is None:
            return html_text

        def _normalize_block(body: str) -> str:
            if '<ol' in body.lower():
                # Keep ordered lists intact, but clean known leak artifacts:
                # - orphan marker rows like "<p>1.</p>"
                # - accidental numbering on non-Weakness field labels
                cleaned = body
                cleaned = re.sub(r"(?is)<p>\s*\d+\.\s*</p>", "", cleaned)
                cleaned = re.sub(r"(?is)<li>\s*\d+\.\s*</li>", "", cleaned)
                non_weak_labels = (
                    "Why it matters",
                    "Warum relevant",
                    "Warum es wichtig ist",
                    "Warum das wichtig ist",
                    "What would verify/falsify (next check)",
                    "What would verify or falsify (next check)",
                    "Was würde verifizieren/falsifizieren (nächster Check)",
                    "Was würde verifizieren oder falsifizieren (nächster Check)",
                    "Next check",
                    "Nächster Check",
                    "Nächste Prüfung",
                    "Prüfen/Widerlegen (nächster Schritt)",
                )
                labels_rx = "|".join(re.escape(x) for x in non_weak_labels)
                cleaned = re.sub(
                    rf"(?is)(<li[^>]*>\s*)(\d+\.\s*)(?=(?:<strong>\s*)?(?:{labels_rx})\s*:)",
                    r"\1",
                    cleaned,
                )
                # Keep secondary field labels on a new visual line inside list items.
                secondary_labels = (
                    "Why it matters", "Warum relevant", "Warum es wichtig ist", "Warum das wichtig ist",
                    "What would verify/falsify (next check)", "What would verify or falsify (next check)",
                    "Was würde verifizieren/falsifizieren (nächster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)",
                    "Next check", "Nächster Check", "Nächste Prüfung",
                    "Prüfen/Widerlegen (nächster Schritt)",
                )
                sec_rx = "|".join(re.escape(x) for x in secondary_labels)
                # Self-Debunking boxes should not show CGI color bullets or isolated markdown debris.
                cleaned = re.sub(r"(?is)<span[^>]*>\s*[🟢🟡🔴]\s*</span>", "", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*[🟢🟡🔴]\s*</\1>", "", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</\1>", "", cleaned)
                cleaned = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", cleaned)
                cleaned = re.sub(r"(?is)>\s*\*\s*(?=<br)", ">", cleaned)
                cleaned = re.sub(
                    rf"(?is)([^>\n])\s+(?=(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:?)",
                    r"\1<br>",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                # Canonicalize + bold secondary labels even when models emit lowercase variants.
                canonical_sec = [
                    ("Why it matters", "Why it matters"),
                    ("Warum relevant", "Warum relevant"),
                    ("Warum es wichtig ist", "Warum das wichtig ist"),
                    ("Warum das wichtig ist", "Warum das wichtig ist"),
                    ("What would verify/falsify (next check)", "What would verify/falsify (next check)"),
                    ("What would verify or falsify (next check)", "What would verify or falsify (next check)"),
                    ("Was würde verifizieren/falsifizieren (nächster Check)", "Was würde verifizieren/falsifizieren (nächster Check)"),
                    ("Was würde verifizieren oder falsifizieren (nächster Check)", "Was würde verifizieren oder falsifizieren (nächster Check)"),
                    ("Next check", "Next check"),
                    ("Nächster Check", "Nächster Check"),
                    ("Nächste Prüfung", "Nächste Prüfung"),
                    ("Prüfen/Widerlegen (nächster Schritt)", "Prüfen/Widerlegen (nächster Schritt)"),
                ]
                sec_canon_map = {str(_pat).lower(): _canon for _pat, _canon in canonical_sec}
                for _pat, _canon in canonical_sec:
                    cleaned = re.sub(
                        rf"(?is)(?<!\()(?:(?:<strong>\s*)?{re.escape(_pat)}(?:\s*</strong>)?)\s*:?\s*",
                        f"<strong>{_canon}</strong>: ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                cleaned = re.sub(r"(?is)</strong>:\s+<", "</strong>:<", cleaned)
                # Some weak models/HTML conversions split a single logical <li> item into
                # "<li>...Weakness...</li><p>Why...</p><p>What would...</p>" inside the same <ol>.
                # Normalize those sibling <p> rows as well (bold + canonical colon).
                for _pat, _canon in canonical_sec:
                    cleaned = re.sub(
                        rf"(?is)(<p[^>]*>\s*)(?:<strong>\s*)?{re.escape(_pat)}(?:\s*</strong>)?\s*:?\s*",
                        rf"\1<strong>{_canon}</strong>: ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                # Handle fragmented primary items from weak markdown conversion:
                # "<li>*Schwäche</li><p>*:</p><p>🟡</p><p>Text...</p>" -> one canonical <li>.
                split_primary_li = re.compile(
                    rf"(?is)<li[^>]*>\s*(?:<strong>\s*)?\*?\s*(Schwäche|Weakness)\s*\*?(?:\s*</strong>)?\s*</li>"
                    rf"(?:\s*<p[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</p>)*"
                    rf"(?:\s*<p[^>]*>\s*[🟢🟡🔴]\s*</p>)*"
                    rf"\s*<p[^>]*>\s*(?!\s*(?:<strong>\s*)?(?:{sec_rx})\b)(.*?)\s*</p>"
                )

                def _merge_split_primary_li(mm: re.Match) -> str:
                    _label = (mm.group(1) or "").strip()
                    _txt = (mm.group(2) or "").strip()
                    if not _txt:
                        return mm.group(0)
                    return f"<li><strong>{_label}</strong>: {_txt}</li>"

                cleaned = split_primary_li.sub(_merge_split_primary_li, cleaned)
                # Merge sibling secondary rows back into the preceding item:
                #   </li><li>Warum ...</li><p>Text</p>  ->  <br><strong>Warum ...</strong>: Text</li>
                #   </li><p><strong>Nächster Check</strong>:</p><p>Text</p> -> same
                split_secondary_rows = re.compile(
                    rf"(?is)</li>\s*<(?:li|p)[^>]*>\s*(?:<strong>\s*)?(?P<label>{sec_rx})(?:\s*</strong>)?\s*:?\s*</(?:li|p)>"
                    rf"(?:\s*<p[^>]*>\s*(?:\*+\s*:?\s*|:\s*)</p>)*"
                    rf"(?:\s*<p[^>]*>\s*[🟢🟡🔴]\s*</p>)*"
                    rf"\s*<p[^>]*>\s*(?P<txt>.*?)\s*</p>"
                )

                def _merge_split_secondary_rows(mm: re.Match) -> str:
                    _lab_raw = re.sub(r"(?is)<[^>]+>", "", mm.group("label") or "").strip()
                    _lab = sec_canon_map.get(_lab_raw.lower(), _lab_raw)
                    _txt = (mm.group("txt") or "").strip()
                    if not _lab or not _txt:
                        return mm.group(0)
                    return f"<br><strong>{_lab}</strong>: {_txt}</li>"

                for _ in range(8):
                    _new = split_secondary_rows.sub(_merge_split_secondary_rows, cleaned)
                    if _new == cleaned:
                        break
                    cleaned = _new
                # If a secondary label became a separate <li>, merge it into the previous weakness item.
                split_secondary_li = re.compile(
                    rf"(?is)</li>\s*<li([^>]*)>\s*((?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?)(?=</li>)</li>"
                )

                def _merge_split_secondary_li(mm: re.Match) -> str:
                    _inner = (mm.group(2) or "").strip()
                    if not _inner:
                        return mm.group(0)
                    return f"<br>{_inner}</li>"

                for _ in range(8):
                    _new = split_secondary_li.sub(_merge_split_secondary_li, cleaned)
                    if _new == cleaned:
                        break
                    cleaned = _new
                cleaned = re.sub(r"(?is)(<strong>[^<]+</strong>:)\s*</strong>", r"\1", cleaned)
                cleaned = re.sub(r"(?is)<(p|li)[^>]*>\s*\*\s*</\1>", "", cleaned)
                # If a logical list item was split into "</li><p>Why...</p><p>Next check...</p>",
                # merge those secondary paragraphs back into the preceding <li> using <br>.
                split_li_paras = re.compile(
                    rf"(?is)</li>((?:\s*<p[^>]*>\s*(?:<strong>\s*)?(?:{sec_rx})(?:\s*</strong>)?\s*:.*?</p>)+)"
                )

                def _merge_split_li_paras(mm: re.Match) -> str:
                    paras_blob = mm.group(1) or ""
                    parts = []
                    for pm in re.finditer(r"(?is)<p[^>]*>\s*(.*?)\s*</p>", paras_blob):
                        inner = (pm.group(1) or "").strip()
                        if inner:
                            parts.append(inner)
                    if not parts:
                        return mm.group(0)
                    return "".join(f"<br>{p}" for p in parts) + "</li>"

                cleaned = split_li_paras.sub(_merge_split_li_paras, cleaned)
                return cleaned

            # Drop standalone marker lines ("1.", "2.") that cause visible double numbering.
            # Also normalize common markdown-leak patterns from weak model output where
            # labels appear as "*Schwäche*" and nested <strong> tags.
            body = re.sub(
                r"(?is)<strong>\s*<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?\s*:\s*</strong>",
                r"<strong>\1</strong>:",
                body,
            )
            body = re.sub(
                r"(?is)<strong>\s*<strong>\s*([^<]+?)\s*</strong>\s*:\s*</strong>",
                r"<strong>\1</strong>:",
                body,
            )
            body = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", body)
            body = re.sub(r"(?im)^\s*(?:<[^>]+>\s*)*\*\s*(?:</[^>]+>\s*)*$", "", body)
            body = re.sub(r"(?is)>\s*\*\s*(?=<br\s*/?>|</(?:p|div|li)>)", ">", body)
            body = re.sub(
                r'(?im)^\s*<div[^>]*>\s*\d+\.\s*</div>\s*$',
                '',
                body,
            )
            body = re.sub(
                r'(?im)^\s*\d+\.\s*(?:<br\s*/?>\s*)?$',
                '',
                body,
            )

            n = 0
            out_lines = []
            for ln in body.splitlines():
                # Localize the three canonical labels inside the block.
                if lang.lower().startswith('de'):
                    ln = ln.replace('Weakness:', 'Schwäche:')
                    ln = ln.replace('Why it matters:', 'Warum das wichtig ist:')
                    ln = ln.replace('What would verify/falsify (next check):', 'Was würde verifizieren/falsifizieren (nächster Check):')
                    ln = ln.replace('<strong>Weakness</strong>:', '<strong>Schwäche</strong>:')
                    ln = ln.replace('<strong>Why it matters</strong>:', '<strong>Warum das wichtig ist</strong>:')
                    ln = ln.replace(
                        '<strong>What would verify/falsify (next check)</strong>:',
                        '<strong>Was würde verifizieren/falsifizieren (nächster Check)</strong>:'
                    )
                else:
                    ln = ln.replace('Schwäche:', 'Weakness:')
                    ln = ln.replace('Warum das wichtig ist:', 'Why it matters:')
                    ln = ln.replace('Was würde verifizieren/falsifizieren (nächster Check):', 'What would verify/falsify (next check):')

                ln = re.sub(r"(?is)<em>\s*\*?\s*(Schwäche|Weakness)\s*\*?\s*</em>\s*\*?", r"\1", ln)
                ln = re.sub(r"(?i)\*+\s*(Schwäche|Weakness)\s*\*+\s*:", r"\1:", ln)
                ln = re.sub(r"(?is)<strong>\s*<strong>\s*", "<strong>", ln)
                ln = re.sub(r"(?is)</strong>\s*</strong>", "</strong>", ln)
                ln = re.sub(r"(?is)(<strong>[^<]+</strong>:)\s*</strong>", r"\1", ln)

                # Bold known labels (formatting only).
                for lab in (
                    'Weakness', 'Schwäche',
                    'Why it matters', 'Warum relevant', 'Warum es wichtig ist', 'Warum das wichtig ist',
                    'What would verify/falsify (next check)', 'What would verify or falsify (next check)',
                    'Was würde verifizieren/falsifizieren (nächster Check)', 'Was würde verifizieren oder falsifizieren (nächster Check)',
                    'Next check', 'Nächster Check',
                ):
                    ln = re.sub(
                        rf"(?i)\b{re.escape(lab)}\s*:",
                        rf"<strong>{lab}</strong>:",
                        ln,
                    )

                # Force secondary field labels onto a new visual line if they leaked inline after
                # the Weakness sentence (common with weaker models / compact HTML rendering).
                secondary_labels = (
                    'Why it matters', 'Warum relevant', 'Warum es wichtig ist', 'Warum das wichtig ist',
                    'What would verify/falsify (next check)', 'What would verify or falsify (next check)',
                    'Was würde verifizieren/falsifizieren (nächster Check)', 'Was würde verifizieren oder falsifizieren (nächster Check)',
                    'Next check', 'Nächster Check', 'Nächste Prüfung',
                    'Prüfen/Widerlegen (nächster Schritt)',
                )
                _sec_rx = "|".join(re.escape(x) for x in secondary_labels)
                ln = re.sub(
                    rf"(?is)([^>\n])\s+(?=(?:<strong>\s*)?(?:{_sec_rx})(?:\s*</strong>)?\s*:?)",
                    r"\1<br>",
                    ln,
                    flags=re.IGNORECASE,
                )

                # Count/number only Weakness lines.
                plain = re.sub(r'<[^>]+>', '', ln).strip()
                plain_norm = re.sub(r"\*+", "", plain).strip()
                # Non-Weakness field labels must not carry list numbering.
                if re.match(
                    r"(?i)^(?:\d+\.\s*)(Why it matters|Warum relevant|Warum es wichtig ist|Warum das wichtig ist|What would verify/falsify \(next check\)|What would verify or falsify \(next check\)|Was würde verifizieren/falsifizieren \(nächster Check\)|Was würde verifizieren oder falsifizieren \(nächster Check\)|Next check|Nächster Check|Nächste Prüfung|Prüfen/Widerlegen \(nächster Schritt\))\s*:",
                    plain_norm,
                ):
                    ln = re.sub(r'(?i)(<div[^>]*>\s*)\d+\.\s*', r'\1', ln, count=1)
                    ln = re.sub(r'(?i)^\s*\d+\.\s*', '', ln, count=1)
                    plain = re.sub(r'<[^>]+>', '', ln).strip()
                    plain_norm = re.sub(r"\*+", "", plain).strip()
                if re.match(r'(?i)^(?:\d+\.\s*)?(Weakness|Schwäche)\b\s*:', plain_norm):
                    n += 1
                    # Remove any existing numeric prefix immediately after opening div/start.
                    ln = re.sub(r'(?i)(<div[^>]*>\s*)\d+\.\s*', r'\1', ln, count=1)
                    ln = re.sub(r'(?i)^\s*\d+\.\s*', '', ln, count=1)
                    if '<div' in ln:
                        ln = re.sub(r'(<div[^>]*>\s*)', lambda m: f"{m.group(1)}{n}. ", ln, count=1)
                    else:
                        ln = f'{n}. ' + ln

                out_lines.append(ln)

            return '\n'.join([x for x in out_lines if x is not None])

        # Primary path: normalize explicit self-debunking boxes.
        block_re = re.compile(
            r'(?is)(<div[^>]*class=(?:"|\')[^"\']*self-debunking[^"\']*(?:"|\')[^>]*>)(.*?)(</div>)'
        )

        def _block_sub(mb: re.Match) -> str:
            return mb.group(1) + _normalize_block(mb.group(2)) + mb.group(3)

        out_text = block_re.sub(_block_sub, html_text)

        # Fallback path: line-wise region between Self-Debunking header and QC footer.
        lines = out_text.splitlines()
        out = []
        in_sd = False
        chunk = []

        def _flush_chunk() -> None:
            nonlocal chunk
            if chunk:
                out.extend(_normalize_block('\n'.join(chunk)).splitlines())
                chunk = []

        for ln in lines:
            plain = re.sub(r'<[^>]+>', '', ln)
            if (not in_sd) and re.search(r'(?i)Self-Debunking|Selbst[- ]?Debunking', plain):
                in_sd = True
                out.append(ln)
                continue
            if in_sd and re.search(r'(?im)^\s*QC(?:-Matrix)?\s*:', plain):
                _flush_chunk()
                in_sd = False
                out.append(ln)
                continue
            if in_sd:
                chunk.append(ln)
            else:
                out.append(ln)

        _flush_chunk()
        return '\n'.join(out)
    except Exception:
        return html_text


def normalize_hash_subheadings_in_html(html_text: str) -> str:
    """Final safety net: convert leaked ##/###/#### headings in rendered HTML to bold lines."""
    try:
        if not html_text:
            return html_text
        out = str(html_text)
        out = out.replace("><", ">\n<")
        pat = re.compile(
            r"(?im)^(\s*(?:\[(?:GREEN|YELLOW|RED|GRAY)(?:-[A-Z0-9]+)*\]\s*)?(?:[🟢🟡🔴⚪⚪️]\s*)?(?:[•*\-]\s*)?)#{2,4}\s*([^\n<]{1,120})\s*$"
        )
        pat_tag = re.compile(
            r"(?is)^(\s*<(?P<tag>div|p|li)\b[^>]*>\s*)"
            r"((?:\[(?:GREEN|YELLOW|RED|GRAY)(?:-[A-Z0-9]+)*\]\s*)?(?:[🟢🟡🔴⚪⚪️]\s*)?(?:[•*\-]\s*)?)"
            r"#{2,4}\s*([^\n<]{1,120})\s*"
            r"(</(?P=tag)>\s*)$"
        )
        lines = []
        for ln in out.splitlines():
            mt = pat_tag.match(ln or "")
            if mt:
                pre = mt.group(1) or ""
                lead = mt.group(3) or ""
                title = (mt.group(4) or "").strip().rstrip(":")
                post = mt.group(5) or ""
                if title:
                    lines.append(f"{pre}{lead}<strong>{html.escape(title)}:</strong>{post}")
                    continue
            m = pat.match(ln or "")
            if m:
                lead = m.group(1) or ""
                title = (m.group(2) or "").strip().rstrip(":")
                if title:
                    lines.append(f"{lead}<strong>{html.escape(title)}:</strong>")
                    continue
            lines.append(ln)
        return "\n".join(lines)
    except Exception:
        return html_text


def _repair_violation_is_format_only(vio: str) -> bool:
    """True for cosmetic/ordering-only repair violations (banner can stay hidden)."""
    try:
        s = re.sub(r"\s+", " ", str(vio or "")).strip().lower()
        if not s:
            return False
        if s in {
            "self-debunking placed after qc footer.",
            "self-debunking placed after qc footer",
            "selbst-debunking nach qc-footer platziert.",
            "selbst-debunking nach qc-footer platziert",
        }:
            return True
        if re.match(r"^self-?debunking must contain 2(?:-|–|—)3 numbered points \(found \d+\)\.?$", s):
            return True
        if re.match(r"^selbst-?debunking muss 2(?:-|–|—)3 nummerierte punkte enthalten \(gefunden \d+\)\.?$", s):
            return True
        return False
    except Exception:
        return False


def _should_show_repair_pass_banner(violations: list[str] | None) -> bool:
    """Show UI banner only if at least one repair violation is not format-only."""
    try:
        vios = [str(v).strip() for v in (violations or []) if str(v or "").strip()]
        if not vios:
            return False
        return any(not _repair_violation_is_format_only(v) for v in vios)
    except Exception:
        return True


def detect_self_debunking_numbered_html(html_text: str) -> bool:
    """Audit helper: detect numbered Self-Debunking points in rendered HTML."""
    try:
        if not html_text:
            return False
        raw = str(html_text)
        if re.search(r"(?i)self[- ]?debunking|selbst[- ]?debunking", raw) is None:
            return False
        plain = re.sub(r"<[^>]+>", "\n", raw)
        m = re.search(
            r"(?is)(Self[- ]?Debunking|Selbst[- ]?Debunking).*?(?:\n\s*QC(?:-Matrix)?\s*:|\Z)",
            plain,
        )
        block = m.group(0) if m else plain
        if re.search(r"(?im)^\s*\d+\.\s*(Schwäche|Weakness)\b", block):
            return True
        # HTML <ol>/<li> numbering is visual and may not appear as text prefixes.
        if re.search(
            r"(?is)<ol[^>]*>.*?<li[^>]*>.*?(?:<strong>\s*)?(Schwäche|Weakness)(?:\s*</strong>)?\s*:.*?</li>",
            raw,
        ):
            return True
        return False
    except Exception:
        return False


def enforce_self_debunking_contract(text: str, gov_mgr, profile_name: str, *, is_command: bool = False, lang: str = "en") -> str:
    """Deterministically enforce the Self-Debunking contract (2–3 numbered points) when required.

    Note: is_command disables enforcement for command-only responses.

    - If missing: inject a minimal compliant block (no new factual claims).
    - If too many points: keep the first 3.
    - If too few points: add generic points to reach the minimum.
    - Ensure placement BEFORE the QC footer.
    """
    try:
        # Commands must NEVER trigger Self-Debunking enforcement/injection.
        if is_command:
            return text
        if not text or not gov_mgr or not getattr(gov_mgr, 'loaded', False):
            return text

        gd = (gov_mgr.data.get('global_defaults', {}) or {})
        oc = (gd.get('output_contract', {}) or {})
        contract = (oc.get('self_debunking_contract', {}) or {})
        if not contract.get('enabled', False):
            return text

        module = (gd.get('self_debunking', {}) or {})
        if not module.get('enabled', False):
            return text

        exceptions = set(module.get('exceptions') or [])
        if (profile_name or '') in exceptions:
            return text

        title = (contract.get('required_block_title') or (module.get('block', {}) or {}).get('title') or 'Self-Debunking').strip() or 'Self-Debunking'

        min_p = int(contract.get('required_min_points', 2) or 2)
        max_p = int(contract.get('required_max_points', 3) or 3)
        min_p = 2 if min_p < 2 else min_p
        max_p = 3 if max_p > 6 else max_p  # safety cap

        def _finalize_sd(t: str) -> str:
            # Apply language normalization first (for DE) then bold labels (formatting only).
            t2 = normalize_self_debunking_language(t, lang)
            t2 = normalize_self_debunking_field_linebreaks(t2, lang=lang)
            t2 = bold_self_debunking_labels(t2, lang)
            t2 = normalize_self_debunking_numbering_text(t2, lang=lang)
            return t2

        # Locate QC footer (insertion anchor)
        qc_m = re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$", text)
        qc_pos = qc_m.start() if qc_m else None

        # Locate Self-Debunking title line
        title_re = re.compile(rf"(?im)^\s*(?:#+\s*)?\*{{0,2}}{re.escape(title)}\*{{0,2}}\s*:?\s*$")
        m = title_re.search(text)
        if not m:
            # Missing entirely -> inject minimal
            out = inject_minimal_self_debunking(text, title=title, lang=lang)
            return _finalize_sd(out)
        # If title occurs after QC, remove that trailing block and inject at the correct place.
        if qc_pos is not None and m.start() > qc_pos:
            trimmed = text[:m.start()].rstrip()
            out = inject_minimal_self_debunking(trimmed, title=title, lang=lang)
            return _finalize_sd(out)

        # Extract the block region: from title line to QC footer (or end)
        end = qc_pos if qc_pos is not None else len(text)
        before = text[:m.end()]
        block = text[m.end():end]
        after = text[end:] if end < len(text) else ""

        # Split numbered points (keep their multi-line bodies)
        # We detect lines starting with "<n>." or "<n>)".
        point_iter = list(re.finditer(r"(?m)^\s*(\d+)\s*[\.)]\s+", block))
        points = []
        for i, pm in enumerate(point_iter):
            p_start = pm.start()
            p_end = point_iter[i + 1].start() if i + 1 < len(point_iter) else len(block)
            points.append(block[p_start:p_end].rstrip())

        # If there are no numbered points at all, attempt to convert common unnumbered formats
        # (e.g., repeated 'Weakness:' blocks) into a numbered list without inventing new facts.
        if not points:
            # Try to extract unnumbered points from repeated label blocks inside this Self-Debunking section.
            extracted = []
            try:
                # Accept both plain labels and already-bolded markdown/HTML labels.
                # This avoids a failure mode where an earlier pass bolded labels (e.g. "**Schwäche**:")
                # and this extractor would no longer recognize unnumbered point starts.
                label_iter = list(
                    re.finditer(
                        r"(?im)^\s*(?:\*\*|<strong>)?\s*(Weakness|Schwäche)\b\s*(?:\*\*|</strong>)?\s*:\s*",
                        block,
                    )
                )
                if label_iter:
                    for i, lm in enumerate(label_iter):
                        p_start = lm.start()
                        p_end = label_iter[i + 1].start() if i + 1 < len(label_iter) else len(block)
                        chunk = block[p_start:p_end].strip()
                        if chunk:
                            extracted.append(chunk)
            except Exception:
                extracted = []

            if extracted:
                # Trim to contract max and normalize to 1..k numbering.
                extracted = extracted[:max_p]
                normalized = []
                for i, chunk in enumerate(extracted, 1):
                    lines = chunk.splitlines()
                    if not lines:
                        continue
                    # Ensure the first line contains the label (Weakness/Schwäche) as in the model output.
                    first = lines[0].strip()
                    rest = [ln.rstrip() for ln in lines[1:]]
                    body = first
                    if rest:
                        body += "\n" + "\n".join("   " + ln.lstrip() for ln in rest if ln.strip() != "")
                    normalized.append(f"{i}. {body.strip()}")
                if normalized:
                    new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"
                    out = before.rstrip() + new_block + after.lstrip()
                    return _finalize_sd(out)

            # If the model used bullet points, convert the first 2–3 bullets into numbered points
            # without inventing content.
            try:
                bullet_lines = [ln.rstrip() for ln in block.splitlines()]
                bullets = []
                for ln in bullet_lines:
                    m_b = re.match(r"^\s*(?:[-*+]|•)\s+(.+?)\s*$", ln)
                    if m_b:
                        b = m_b.group(1).strip()
                        if b:
                            bullets.append(b)
                if bullets:
                    bullets = bullets[:max_p]
                    normalized = [f"{i}. {b}" for i, b in enumerate(bullets, 1)]
                    new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"
                    out = before.rstrip() + new_block + after.lstrip()
                    return _finalize_sd(out)
            except Exception:
                pass

            # Fallback: remove the broken/empty block and inject a minimal compliant numbered block.
            try:
                base = text[:m.start()].rstrip() + "\n\n" + text[end:].lstrip()
            except Exception:
                base = before.rstrip() + "\n\n" + after.lstrip()
            injected = inject_minimal_self_debunking(base, title=title, lang=lang)
            return _finalize_sd(injected)
        # Normalize number of points to the contract window.
        if len(points) > max_p:
            points = points[:max_p]
        while len(points) < min_p:
            n = len(points) + 1
            if str(lang).lower().startswith("de"):
                points.append(
                    f"{n}. Schwäche: Die Antwort könnte wichtige Einschränkungen oder Randbedingungen auslassen.\n"
                    f"   Warum das wichtig ist: Fehlende Vorbehalte können Sicherheit oder Gültigkeit überzeichnen.\n"
                    f"   Was würde verifizieren/falsifizieren (nächster Check): Formuliere ein konkretes Gegenbeispiel und prüfe, ob die Schlussfolgerung dann noch gilt."
                )
            else:
                points.append(
                    f"{n}. Weakness: The answer may omit important limitations or boundary conditions.\n"
                    f"   **Why it matters**: Missing caveats can overstate confidence or applicability.\n"
                    f"   **What would verify/falsify (next check)**: Identify a concrete counterexample and test whether the conclusion still holds."
                )
        # Re-number points sequentially (1..k)
        normalized = []
        for i, p in enumerate(points, 1):
            # Keep continuation lines inside the numbered item for stable Markdown rendering.
            lines = [ln.rstrip() for ln in str(p).splitlines()]
            if not lines:
                continue
            first = re.sub(r"^\s*\d+\s*[\.)]\s+", f"{i}. ", lines[0].strip(), count=1)
            rest = []
            for ln in lines[1:]:
                if ln.strip():
                    rest.append("   " + ln.lstrip())
            item = first if not rest else (first + "\n" + "\n".join(rest))
            normalized.append(item.strip())

        new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"

        # Reassemble
        out = before.rstrip() + new_block + after.lstrip()
        out = normalize_self_debunking_language(out, lang)
        out = normalize_self_debunking_field_linebreaks(out, lang=lang)
        out = bold_self_debunking_labels(out, lang)
        return out

    except Exception:
        return text


# === ROUTE CONTEXT / CONTRACTS (Stage 1) =====================================

def _build_route_ctx(api, user_raw: str, is_command: bool) -> dict:
    """Build a minimal, stable route context dict (pure, deterministic).

    Stage 1: Internal contract used to avoid scattered gov_state reads.
    Must not change runtime behavior; it only centralizes already-used flags.
    """
    try:
        gs = getattr(api, 'gov_state', None)
        ui_lang = (getattr(gs, 'ui_lang', None) or 'en')
        answer_lang = (getattr(gs, 'answer_lang', None) or ui_lang or 'en')
        color = (getattr(gs, 'color', None) or 'off')
        sci_variant = (getattr(gs, 'sci_variant', None) or 'A')
        sci_pending = bool(getattr(gs, 'sci_pending', False))
        comm_active = bool(getattr(gs, 'comm_active', False))
    except Exception:
        ui_lang = 'en'
        answer_lang = 'en'
        color = 'off'
        sci_variant = 'A'
        sci_pending = False
        comm_active = False

    return {
        'is_command': bool(is_command),
        'comm_active': bool(comm_active),
        'ui_lang': ui_lang,
        'answer_lang': answer_lang,
        'color': color,
        'sci_variant': sci_variant,
        'sci_pending': bool(sci_pending),
        'user_raw': user_raw or '',
    }


def contract_command_response(html_text: str) -> bool:
    """Best-effort contract: command responses must not be required to have SD/QC.

    Used in tests only (Stage 1). Must stay non-blocking in runtime.
    """
    txt = str(html_text or '')
    # Self-Debunking is not required for command-only responses.
    if re.search(r"\bSelf-?Debunking\b", txt, flags=re.IGNORECASE):
        return False
    return True


def contract_answer_response(html_text: str) -> bool:
    """Best-effort contract for a normal answer response: QC footer is expected."""
    txt = str(html_text or '')
    return ('QC-Matrix:' in txt) or ('QC:' in txt)


_SIGNAL_DOT_SPAN_RE = re.compile(r"(?is)<span(?P<attrs>[^>]*)>(?P<body>\s*[🟢🟡🔴]\s*)</span>")
_SIGNAL_DOT_BLOCK_RE = re.compile(r"(?is)<(?P<tag>p|li)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>")
_SIGNAL_DOT_MARKER_SPAN_RE = re.compile(
    r"(?is)<span(?P<attrs1>[^>]*)class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')(?P<attrs2>[^>]*)>\s*"
    r"(?P<body>(?:<span\b[^>]*>\s*[🟢🟡🔴]\s*</span>|[🟢🟡🔴]))\s*</span>"
)
_SIGNAL_DOT_MARKER_RE = re.compile(
    r"(?is)<span\b[^>]*class=(?:\"|')[^\"']*\bsignal-dot-marker\b[^\"']*(?:\"|')[^>]*>\s*"
    r"(?:<span\b[^>]*>\s*[🟢🟡🔴]\s*</span>|[🟢🟡🔴])\s*</span>\s*"
)
_SIGNAL_DOT_STATUS_PREFIX_RE = re.compile(
    r"(?i)^(?:active profile|profile|overlay|sci|control layer|qc|cgi|color|comm)\s*:"
)
_SIGNAL_DOT_COLOR = {
    "🟢": "#2e7d32",
    "🟡": "#f9a825",
    "🔴": "#c62828",
}


def _signal_dot_tooltip_text(icon: str, *, lang: str = "de") -> str:
    use_en = str(lang or "").strip().lower().startswith("en")
    ic = str(icon or "").strip()
    if ic == "🟢":
        return (
            "Green: high reliability and comparatively robust evidence."
            if use_en
            else "Gruen: hohe Verlaesslichkeit und vergleichsweise robuste Evidenz."
        )
    if ic == "🟡":
        return (
            "Yellow: medium reliability; relevant uncertainty remains."
            if use_en
            else "Gelb: mittlere Verlaesslichkeit; relevante Unsicherheit bleibt."
        )
    if ic == "🔴":
        return (
            "Red: low reliability; substantial uncertainty or weak support."
            if use_en
            else "Rot: niedrige Verlaesslichkeit; erhebliche Unsicherheit oder schwache Absicherung."
        )
    return ""


def _tooltip_lang(lang: str = "de") -> str:
    l = str(lang or "").strip().lower()
    return "en" if l.startswith("en") else "de"


def _format_response_timestamp(dt_obj=None) -> str:
    """Return deterministic chat timestamp with explicit local timezone."""
    try:
        dt = dt_obj if dt_obj is not None else datetime.now()
        try:
            dt = dt.astimezone()
        except Exception:
            pass
        base = dt.strftime("%d.%m.%Y %H:%M:%S")
        tz_name = str(dt.tzname() or "").strip()
        off = dt.strftime("%z")
        off_fmt = ""
        if len(off) == 5 and off[0] in "+-":
            off_fmt = off[:3] + ":" + off[3:]
        elif off:
            off_fmt = off

        if tz_name and off_fmt:
            return f"{base} {tz_name} (UTC{off_fmt})"
        if off_fmt:
            return f"{base} (UTC{off_fmt})"
        if tz_name:
            return f"{base} {tz_name}"
        return base
    except Exception:
        return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def _csc_score_tooltip_text(*, lang: str = "de", f_score: int = 0, token_count: int = 0) -> str:
    if _tooltip_lang(lang) == "en":
        return (
            f"Score line: f={int(f_score)}, tokens={int(token_count)}. "
            "f = how complex/technical your prompt is (based on code/math patterns). "
            "tokens = approximate prompt length in words."
        )
    return (
        f"Score-Zeile: f={int(f_score)}, tokens={int(token_count)}. "
        "f = wie komplex/technisch dein Prompt ist (anhand von Code-/Mathe-Mustern). "
        "tokens = ungefaehre Prompt-Laenge in Woertern."
    )


def _csc_thresholds_tooltip_text(
    *,
    lang: str = "de",
    thr_fs: int = 0,
    thr_tok: int = 0,
    gov_min_tok: int = 0,
    mult: int = 1,
) -> str:
    if _tooltip_lang(lang) == "en":
        return (
            f"Thresholds line: f>={int(thr_fs)}, tok>={int(thr_tok)}, gov_tok>={int(gov_min_tok)}, x{int(mult or 1)}. "
            "x = threshold multiplier (x1 normal, x2 stricter). "
            "f = minimum complexity needed for CSC complexity trigger. "
            "tok = minimum prompt length for the complexity trigger. "
            "gov_tok = minimum prompt length when a governance trigger is active."
        )
    return (
        f"Thresholds-Zeile: f>={int(thr_fs)}, tok>={int(thr_tok)}, gov_tok>={int(gov_min_tok)}, x{int(mult or 1)}. "
        "x = Schwellen-Multiplikator (x1 normal, x2 strenger). "
        "f = minimale Komplexitaet fuer den CSC-Komplexitaets-Trigger. "
        "tok = minimale Prompt-Laenge fuer den Komplexitaets-Trigger. "
        "gov_tok = minimale Prompt-Laenge, wenn ein Governance-Trigger aktiv ist."
    )


def _control_layer_tooltip_text(*, lang: str = "de", severity: str = "warn") -> str:
    use_en = (_tooltip_lang(lang) == "en")
    sev = str(severity or "").strip().lower()
    if sev == "error":
        return (
            "Control Layer error: deterministic safety/contract guard stopped or corrected output."
            if use_en
            else "Control-Layer-Fehler: deterministische Sicherheits-/Vertragspruefung hat Ausgabe gestoppt oder korrigiert."
        )
    if sev == "warn":
        return (
            "Control Layer note: deterministic safety/contract guard adjusted output. "
            "Please cross-check critical parts."
            if use_en
            else "Control-Layer-Hinweis: deterministische Sicherheits-/Vertragspruefung hat Ausgabe angepasst. "
                 "Kritische Teile bitte gegenpruefen."
        )
    return (
        "Control Layer info: deterministic wrapper guard information."
        if use_en
        else "Control-Layer-Info: deterministische Schutzinformation des Wrappers."
    )


def annotate_signal_dot_tooltips_html(html_text: str, *, lang: str = "de") -> str:
    """Wrap color signal dots with hold-tooltip metadata in answer language."""
    src = str(html_text or "")
    if not src:
        return src

    protected: list[str] = []

    def _protect_existing_marker(m: re.Match) -> str:
        attrs = f"{m.group('attrs1') or ''}{m.group('attrs2') or ''}"
        body = str(m.group("body") or "")
        block = m.group(0)
        if re.search(r"(?i)\b(?:data-u-title|title)\s*=", attrs) is None:
            im = re.search(r"[🟢🟡🔴]", body)
            icon = str(im.group(0) if im else "")
            tip = _signal_dot_tooltip_text(icon, lang=lang)
            if tip:
                esc = html.escape(tip)
                if re.search(r"(?i)\bstyle\s*=", block):
                    block = re.sub(
                        r"(?is)<span\b",
                        f"<span data-u-title='{esc}' title='{esc}'",
                        block,
                        count=1,
                    )
                else:
                    block = re.sub(
                        r"(?is)<span\b",
                        f"<span data-u-title='{esc}' title='{esc}' style='cursor:help;'",
                        block,
                        count=1,
                    )
        token = f"__SIGNAL_DOT_MARKER_PROTECT_{len(protected)}__"
        protected.append(block)
        return token

    stage = _SIGNAL_DOT_MARKER_SPAN_RE.sub(_protect_existing_marker, src)

    def _repl(m: re.Match) -> str:
        body = str(m.group("body") or "")
        icon = re.sub(r"\s+", "", body)
        tip = _signal_dot_tooltip_text(icon, lang=lang)
        if not tip:
            return m.group(0)
        esc = html.escape(tip)
        return (
            "<span class='signal-dot-marker' "
            f"data-u-title='{esc}' title='{esc}' style='cursor:help;'>"
            f"{m.group(0)}"
            "</span>"
        )

    stage = _SIGNAL_DOT_SPAN_RE.sub(_repl, stage)
    for idx, block in enumerate(protected):
        stage = stage.replace(f"__SIGNAL_DOT_MARKER_PROTECT_{idx}__", block)
    return stage


def _fallback_signal_dot_icon_for_text(text: str) -> str:
    """Map block uncertainty signal to a deterministic fallback dot icon."""
    norm = html.unescape(re.sub(r"(?is)<[^>]+>", " ", str(text or "")))
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return "🟢"

    codes: list[str] = []
    try:
        if _uncertainty_codes_mod is not None and hasattr(_uncertainty_codes_mod, "infer_uncertainty_codes"):
            codes = list(_uncertainty_codes_mod.infer_uncertainty_codes(norm, user_text="") or [])
    except Exception:
        codes = []
    code_set = {str(c or "").strip().upper() for c in codes}
    if "U1" in code_set or "U4" in code_set:
        return "🔴"
    if code_set:
        return "🟡"
    return "🟢"


def inject_fallback_signal_dots_html(html_text: str, *, lang: str = "de") -> str:
    """Insert deterministic signal dots when Color=on but model emitted no evidence dots."""
    src = str(html_text or "")
    if not src:
        return src
    if "signal-dot-marker" in src:
        return src

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        tag = str(m.group("tag") or "")
        attrs = str(m.group("attrs") or "")
        body = str(m.group("body") or "")
        out.append(src[cursor:start])

        plain = html.unescape(re.sub(r"(?is)<[^>]+>", " ", body))
        plain = re.sub(r"\s+", " ", plain).strip()
        low = plain.lower()

        skip = False
        if not plain or len(plain) < 24:
            skip = True
        elif _SIGNAL_DOT_STATUS_PREFIX_RE.match(plain):
            skip = True
        elif low.startswith("qc-matrix:") or low.startswith("verification route"):
            skip = True
        elif "response at " in low or "selbst-debunking" in low or "self-debunking" in low:
            skip = True
        elif "uncertainty-auto-marker" in attrs:
            skip = True

        if skip:
            out.append(src[start:end])
        else:
            icon = _fallback_signal_dot_icon_for_text(plain)
            tip = _signal_dot_tooltip_text(icon, lang=lang)
            esc_tip = html.escape(tip) if tip else ""
            color = _SIGNAL_DOT_COLOR.get(icon, "#5f6368")
            dot_html = (
                "<span class='signal-dot-marker' "
                f"data-u-title='{esc_tip}' title='{esc_tip}' style='cursor:help;'>"
                f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"
                "</span> "
            )
            marked_body = re.sub(r"^(\s*)", r"\1" + dot_html, body, count=1)
            out.append(f"<{tag}{attrs}>{marked_body}</{tag}>")
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def limit_signal_dot_marker_density_html(html_text: str, *, max_per_block: int = 1) -> str:
    """Keep at most N signal-dot markers per content block to avoid visual marker overload."""
    src = str(html_text or "")
    cap = max(1, int(max_per_block or 1))
    if not src or "signal-dot-marker" not in src:
        return src

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        tag = str(m.group("tag") or "")
        attrs = str(m.group("attrs") or "")
        body = str(m.group("body") or "")
        out.append(src[cursor:start])

        if "signal-dot-marker" not in body:
            out.append(src[start:end])
            cursor = end
            continue

        kept = 0

        def _trim(mm: re.Match) -> str:
            nonlocal kept
            kept += 1
            if kept <= cap:
                return mm.group(0)
            return ""

        cleaned = _SIGNAL_DOT_MARKER_RE.sub(_trim, body)
        out.append(f"<{tag}{attrs}>{cleaned}</{tag}>")
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def strip_signal_dots_from_heading_only_blocks_html(html_text: str) -> str:
    """Remove signal-dot markers from pure heading blocks (<p>/<li> with only <strong>/<hN>)."""
    src = str(html_text or "")
    if not src or "signal-dot-marker" not in src:
        return src

    def _is_heading_only(inner_html: str) -> bool:
        content = re.sub(r"(?is)^\s*(?:<br\s*/?>|\s|&nbsp;)+", "", str(inner_html or ""))
        content = re.sub(r"(?is)(?:<br\s*/?>|\s|&nbsp;)+\s*$", "", content)
        if not content:
            return False
        if re.fullmatch(r"(?is)<strong\b[^>]*>.*?</strong>", content):
            return True
        if re.fullmatch(r"(?is)<h[1-6]\b[^>]*>.*?</h[1-6]>", content):
            return True
        return False

    out = []
    cursor = 0
    for m in _SIGNAL_DOT_BLOCK_RE.finditer(src):
        start, end = m.span()
        block = str(m.group(0) or "")
        out.append(src[cursor:start])
        if "signal-dot-marker" not in block:
            out.append(block)
        else:
            block_wo_markers = _SIGNAL_DOT_MARKER_RE.sub("", block)
            mm = re.match(r"(?is)<(?P<tag>p|li)\b[^>]*>(?P<body>.*?)</(?P=tag)>", block_wo_markers)
            body = str(mm.group("body") if mm else "")
            if _is_heading_only(body):
                out.append(block_wo_markers)
            else:
                out.append(block)
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def apply_color_spans(text: str, enabled: bool = True) -> str:
    """Render Evidence-Linker tags with actual HTML colors (does not invent tags)."""
    if not enabled or not text:
        return text

    def repl(m: re.Match) -> str:
        tag = (m.group("tag") or "").upper()
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag, "#616161")
        icon = emoji or _EVIDENCE_ICON.get(tag, "⚪")
        return f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"

    # Patterns like: [GREEN] 🟢  or [GREEN-WEB] 🟢
    pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?")
    return pat.sub(repl, text)


def _reapply_color_styles_if_stripped(html_text: str) -> str:
    """If Bleach stripped inline CSS (style=""), re-apply our own safe color styles.

    This is a defensive fallback for environments where Bleach cannot load CSSSanitizer (e.g., missing tinycss2).
    We only touch spans that contain our Evidence-Linker tokens, and we only inject the fixed palette colors.
    """
    if not html_text:
        return html_text

    # Replace empty style="" on our evidence spans.
    def repl(m: re.Match) -> str:
        tag = (m.group("tag") or "").upper()
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag, "#616161")
        icon = emoji or _EVIDENCE_ICON.get(tag, "⚪")
        return f"<span style=\"color:{color}; font-weight:600;\">{icon}</span>"

    # Match: <span style="">[GREEN-WEB-CHECK] 🟢</span>  (or without emoji)
    pat = re.compile(
        r"<span\s+style=\"\"\s*>\s*\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?\s*</span>",
        flags=re.IGNORECASE,
    )
    return pat.sub(repl, html_text)

@dataclass
class GovernanceRuntimeState:
    comm_active: bool = False
    active_profile: str = "Standard"
    overlay: str = ""
    color: str = "on"
    conversation_language: str = ""
    answer_language: str = "de"
    language_policy_mode: str = "production"
    sci_pending: bool = False
    sci_variant: str = ""
    sci_active: bool = False

    # Anchor snapshot automation (session-level)
    user_turns: int = 0
    anchor_auto: bool = True
    anchor_force_next: bool = False
    last_anchor: str = ""
    anchor_auto_user_override: bool = False

    qc_overrides: dict = field(default_factory=dict)
    # CGI feedback (optional): last captured feedback strings (not a code change)
    last_user_feedback_triplet: str = ""
    last_process_cgi_feedback: str = ""
    cgi_feedback_pending_for_model: bool = False
def try_enter_sci_recursion(state, *, max_depth: int = 2) -> bool:
    """Deterministically enter SCI recursion if depth allows."""
    try:
        cur = int(getattr(state, 'sci_recursion_depth', 0) or 0)
    except Exception:
        cur = 0
    if cur >= int(max_depth or 0):
        return False

    try:
        state.sci_recursion_parent_variant = getattr(state, 'sci_variant', '') or ''
        state.sci_recursion_depth = cur + 1
        state.sci_recursion_one_shot = True
    except Exception:
        return False

    # Ensure SCI has a defined trace context
    try:
        if not bool(getattr(state, 'sci_active', False)):
            state.sci_active = True
        if not (getattr(state, 'sci_variant', '') or '').strip():
            state.sci_variant = 'A'
    except Exception:
        pass

    return True


    # SCI pending selection timeout tracking (canonical JSON: syntax_rules.special_parsing.sci_variant_selection)
    sci_pending_turns: int = 0

    # SCI recursion (canonical JSON: sci.recursive_sci)
    sci_recursion_depth: int = 0
    sci_recursion_parent_variant: str = ""
    sci_recursion_one_shot: bool = False
    sci_recursion_scope: str = ""

    # Dynamic prompting one-shot (canonical JSON: global_defaults.dynamic_prompting.one_shot_override)
    dynamic_one_shot_active: bool = False

    # Dynamic prompting auto-trigger tracking (best-effort; driven by JSON thresholds)
    dynamic_consecutive_turns: int = 0

    # Last observed QC (from model output) + Python-derived deltas (used for dynamic prompting)
    last_qc: dict = field(default_factory=dict)
    # Cross-version leak guard (ignore foreign Comm-SCI versions referenced in user input)
    active_ruleset_version: str = ""
    cross_version_guard_hits: list = field(default_factory=list)
    user_turns: int = 0
    anchor_auto: bool = True
    anchor_force_next: bool = False
    last_anchor: str = ""
    dynamic_nudge: str = ""

# --- SCI Trace normalization (ordered-step numbering) ---
# The model sometimes turns the SCI Trace into a line-by-line ordered list (1..N), which is not desired.
# This normalizer rewrites the SCI Trace section so that ONLY the required SCI steps are numbered
# (1..len(required_steps)), while step contents remain unnumbered paragraphs.


def _strip_basic_html_for_enforcement(t: str) -> str:
    """Convert simple HTML-ish outputs into plain text so regex-based contracts can be enforced."""
    if not t:
        return t
    if '<' not in t or '>' not in t:
        return t
    # Replace common block/line breaks with newlines
    t2 = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', t)
    t2 = re.sub(r'(?i)</\s*p\s*>', '\n', t2)
    t2 = re.sub(r'(?i)<\s*p[^>]*>', '', t2)
    t2 = re.sub(r'(?i)</\s*div\s*>', '\n', t2)
    t2 = re.sub(r'(?i)<\s*div[^>]*>', '', t2)
    t2 = re.sub(r'(?i)</\s*li\s*>', '\n', t2)
    t2 = re.sub(r'(?i)<\s*li[^>]*>', '', t2)
    t2 = re.sub(r'(?i)</\s*h[1-6]\s*>', '\n', t2)
    t2 = re.sub(r'(?i)<\s*h[1-6][^>]*>', '', t2)
    # Strip any remaining tags
    t2 = re.sub(r'<[^>]+>', '', t2)
    # Unescape a few common entities
    t2 = (t2.replace('&nbsp;', ' ')
              .replace('&amp;', '&')
              .replace('&lt;', '<')
              .replace('&gt;', '>')
              .replace('&#39;', "'")
              .replace('&quot;', '"'))
    return t2

def ensure_qc_footer_present(text: str, gov_mgr, profile_name: str, overrides: dict | None = None) -> str:
    """If QC is enabled and no QC-Matrix footer exists, append a canonical QC-Matrix line.
    Uses effective QC values (upper bounds) and Δ computed against effective corridor (thus usually Δ0).
    """
    try:
        if not text or not gov_mgr or not getattr(gov_mgr, 'loaded', False):
            return text
        gd = (gov_mgr.data.get('global_defaults', {}) or {})
        oc = (gd.get('output_contract', {}) or {})
        if not (oc.get('require_qc_footer', False) or (gd.get('qc', {}) or {}).get('enabled', False)):
            return text
        if re.search(r'(?im)^\s*QC(?:-Matrix)?\s*:\s*', text):
            qc_matches = list(re.finditer(r'(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$', text))
            qc_line = qc_matches[-1].group(0) if qc_matches else ""
            qc_probe = qc_line or text
            labels = [
                "Clarity", "Brevity", "Evidence", "Empathy", "Consistency", "Neutrality",
                "Klarheit", "Kürze", "Kuerze", "Evidenz", "Empathie", "Konsistenz", "Neutralität", "Neutralitaet",
            ]
            found_metric_labels = 0
            for lbl in labels:
                if re.search(rf'(?i)\b{re.escape(lbl)}\b\s+\d+\s*\(\s*Δ', qc_probe):
                    found_metric_labels += 1
            # EN or DE complete footer => 6 dimensions
            if found_metric_labels >= 6:
                return text
            # Remove malformed/empty QC lines and rebuild canonical footer below.
            text = re.sub(r'(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()

        vals = {}
        try:
            vals = gov_mgr.get_effective_qc_values(profile_name, overrides or {})
        except Exception:
            vals = {}
        # Need the full canonical set; otherwise do not invent.
        canon_order = [
            ('clarity', 'Klarheit'),
            ('brevity', 'Kürze'),
            ('evidence', 'Evidenz'),
            ('empathy', 'Empathie'),
            ('consistency', 'Konsistenz'),
            ('neutrality', 'Neutralität'),
        ]
        if not all(k in vals for k, _ in canon_order):
            return text

        parts = []
        for k, disp in canon_order:
            iv = int(vals.get(k))
            # expected delta relative to corridor
            d = 0
            try:
                corr = gov_mgr.get_effective_qc_corridor(profile_name, overrides or {})
                if corr and k in corr:
                    lo, hi = corr[k]
                    if iv < lo: d = iv - lo
                    elif iv > hi: d = iv - hi
                    else: d = 0
            except Exception:
                d = 0
            sign = '+' if d > 0 else ''
            parts.append(f"{disp} {iv} (Δ{sign}{d})")
        qc_line = "QC-Matrix: " + " · ".join(parts)

        return (text.rstrip() + "\n\n" + qc_line + "\n")
    except Exception:
        return text


def strip_sci_menu_from_answer(text: str) -> str:
    """Remove an erroneously echoed SCI variant menu from a normal answer.

    The SCI menu must only appear on: 'Profile Expert', 'Profile Sparring', 'SCI menu', 'SCI on' command outputs.
    Some models echo the menu at the beginning of a content answer; we strip it deterministically.
    """
    try:
        if not text:
            return text
        # Work on plain text (not HTML); if HTML is already present, strip tags first for detection only.
        probe = re.sub(r"<[^>]+>", "", text)
        if ("SCI variants" not in probe) and ("SCI-Varianten" not in probe):
            return text

        # Identify the menu region heuristically: from the first menu title to the first of:
        # - 'SCI Trace' line
        # - 'Final Answer' line
        # - 'Self-Debunking' line
        # - QC footer line
        # We remove only if it looks like the standard A–H listing.
        start_m = re.search(r"(?im)^\s*(?:Profile\s*:\s*\w+\s*)?(SCI variants \(selection\)|SCI-Varianten(?:menü)? \(Auswahl\))\s*:?\s*$", probe)
        if not start_m:
            # Sometimes the title is on the same line as 'Profile: ...'
            start_m = re.search(r"(?im)\b(SCI variants \(selection\)|SCI-Varianten(?:menü)? \(Auswahl\))\b", probe)
        if not start_m:
            return text
        start = start_m.start()

        end_m = re.search(r"(?im)^\s*(SCI Trace|Final Answer|Self-Debunking|QC(?:-Matrix)?)\b", probe[start:])
        if end_m:
            end = start + end_m.start()
        else:
            # Fallback: cut at first blank line after H: entry
            h_m = re.search(r"(?im)^\s*H\s*:\s+.*$", probe[start:])
            if h_m:
                tail = probe[start + h_m.end():]
                blank = re.search(r"\n\s*\n", tail)
                end = start + h_m.end() + (blank.start() if blank else 0)
            else:
                return text

        # If the region doesn't contain at least A..H option markers, don't touch.
        # Accept both "A: ..." style and "a) ..." style menus.
        region = probe[start:end]
        has_AH_colon = bool(re.search(r"(?im)^\s*A\s*:\s+", region) and re.search(r"(?im)^\s*H\s*:\s+", region))
        has_ah_paren = bool(re.search(r"(?im)^\s*a\s*[\)\.]\s+", region) and re.search(r"(?im)^\s*h\s*[\)\.]\s+", region))
        if not (has_AH_colon or has_ah_paren):
            return text

        # Now remove corresponding slice from original text by mapping via length in probe.
        # Best-effort: remove by finding the same region in the original (possibly with HTML tags).
        # We try two strategies: exact probe substring, then a regex-based removal.
        if probe[start:end] in re.sub(r"<[^>]+>", "", text):
            # Remove on probe basis: use regex over original to delete menu block.
            text = re.sub(r"(?is)(?:^|\n)\s*(?:Profile\s*:\s*[^\n]*\n)?\s*(SCI variants \(selection\)|SCI-Varianten(?:menü)? \(Auswahl\))\s*:?.*?(?=\n\s*(SCI Trace|Final Answer|Self-Debunking|QC(?:-Matrix)?)\b)", "\n", text, count=1)
        return text
    except Exception:
        return text

def match_required_sci_step_header(line: str, required_steps: list[str]):
    """Return (step, rest) when line starts with a required SCI step header.

    Supports optional bullet markers/list numbering and bold wrappers around the step label.
    Step labels are matched against ruleset labels as-is (including spaces, '/', '+', ':', '-').
    """
    try:
        s = str(line or "")
        if not s or not isinstance(required_steps, list):
            return None, None
        prefix = r"^\s*(?:[*+-]|•)?\s*(?:\d+\.)?\s*"
        def _norm_step_label(v: str) -> str:
            t = re.sub(r"<[^>]+>", "", str(v or ""))
            t = t.strip().strip("*_").strip().lower()
            # Common model drift in SCI-B step 6: "Syntheses_2" vs canonical "Synthesis2"
            t = t.replace("syntheses", "synthesis")
            t = re.sub(r"[^a-z0-9]+", "", t)
            return t
        steps = sorted(
            [str(x).strip() for x in required_steps if str(x or "").strip()],
            key=len,
            reverse=True,
        )
        for step in steps:
            esc = re.escape(step)
            pat = re.compile(
                prefix + rf"(?:\*\*|__)?{esc}(?:\*\*|__)?\s*:\s*(?P<rest>.*)$",
                flags=re.IGNORECASE,
            )
            m = pat.match(s)
            if m:
                return step, (m.group("rest") or "").strip()
        # Fallback: tolerate minor label-variant drift (e.g. Synthesis2 vs Syntheses_2)
        m_any = re.match(
            prefix + r"(?:\*\*|__)?(?P<label>[^:\n]{1,200}?)(?:\*\*|__)?\s*:\s*(?P<rest>.*)$",
            s,
            flags=re.IGNORECASE,
        )
        if m_any:
            got = _norm_step_label(m_any.group("label"))
            if got:
                for step in steps:
                    if got == _norm_step_label(step):
                        return step, (m_any.group("rest") or "").strip()
        return None, None
    except Exception:
        return None, None

def normalize_sci_trace_numbering(text: str, gov) -> str:
    try:
        if not text or 'SCI Trace' not in text:
            return text

        # Get required steps from ruleset (Source of Truth)
        data = getattr(gov, 'data', None) or {}
        gd = (data.get('global_defaults') or {})
        oc = (gd.get('output_contract') or {})
        stc = (oc.get('sci_trace_contract') or {})
        required_steps = stc.get('required_steps') or []
        if not isinstance(required_steps, list) or not required_steps:
            return text

        lines = text.splitlines()

        # Locate SCI Trace header line
        sci_idx = None
        for i, ln in enumerate(lines):
            if re.match(r"^\s*SCI\s+Trace\s*:?.*$", ln):
                sci_idx = i
                break
        if sci_idx is None:
            return text

        # Determine end of SCI Trace section
        end_idx = len(lines)
        end_pat = re.compile(r"^\s*(Final\s+Answer\s*:|Self-?Debunking\s*:|QC-?Matrix\s*:)")
        for j in range(sci_idx + 1, len(lines)):
            if end_pat.match(lines[j]):
                end_idx = j
                break

        pre = lines[:sci_idx]
        sci_header = lines[sci_idx].strip()
        body = lines[sci_idx + 1:end_idx]
        post = lines[end_idx:]

        # Parse step blocks
        blocks = {}
        cur = None
        buf = []

        def flush():
            nonlocal cur, buf
            if cur is not None:
                # Strip accidental ordered-list prefixes inside the step content
                cleaned = []
                for x in buf:
                    cleaned.append(re.sub(r"^\s*\d+\.\s+", "", x))

                # Trim leading/trailing empty lines
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()

                # CRITICAL: Ensure all step-body lines are indented so markdown keeps them
                # inside the numbered step item (prevents 1..31 runaway numbering).
                indented = []
                for x in cleaned:
                    if not x.strip():
                        indented.append("")
                        continue
                    m2 = re.match(r"^\s*([*+-])\s+(.*)$", x)
                    if m2:
                        indented.append("    * " + m2.group(2).strip())
                    else:
                        indented.append("    " + x.strip())

                blocks[cur] = indented
            cur = None
            buf = []

        # Count how many headers we actually recognize; if none -> do nothing
        recognized = 0
        for ln in body:
            step_name, rest = match_required_sci_step_header(ln, required_steps)
            if step_name:
                flush()
                cur = step_name
                recognized += 1
                if rest:
                    buf.append(rest)
                continue
            if cur is not None:
                buf.append(ln)
        flush()

        # Only rewrite if we recognized at least 2 step headers (avoid harming non-standard outputs)
        if recognized < 2:
            return text

        out = []
        out.extend(pre)
        out.append('SCI Trace:')

        for k, step in enumerate(required_steps, start=1):
            step = str(step)
            if step in blocks:
                out.append(f"{k}. {step}:")
                out.extend(blocks[step] if blocks[step] else [""])
                out.append("")

        # Remove trailing blank
        while out and out[-1] == "":
            out.pop()

        out.extend(post)
        return "\n".join(out)
    except Exception:
        return text


# --- SCI Trace hard-render as HTML (prevents Markdown list runaway numbering) ---
# Python-Markdown can accidentally treat step-body lines as additional <ol><li> items,
# producing 1..31 numbering although the ruleset requires exactly len(required_steps) steps.
# This function replaces the SCI Trace section with an HTML <ol> whose <li> count is fixed,
# so numbering can never exceed the number of required steps.

def render_sci_trace_as_html(text: str, gov) -> str:
    try:
        if not text or 'SCI Trace' not in text:
            return text

        data = getattr(gov, 'data', None) or {}
        gd = (data.get('global_defaults') or {})
        oc = (gd.get('output_contract') or {})
        stc = (oc.get('sci_trace_contract') or {})
        required_steps = stc.get('required_steps') or []
        if not isinstance(required_steps, list) or not required_steps:
            return text

        lines = text.splitlines()
        sci_idx = None
        for i, ln in enumerate(lines):
            if re.match(r"^\s*SCI\s+Trace\s*:?.*$", ln):
                sci_idx = i
                break
        if sci_idx is None:
            return text

        end_idx = len(lines)
        end_pat = re.compile(r"^\s*(Final\s+Answer\s*:|Self-?Debunking\s*:|QC-?Matrix\s*:)")
        for j in range(sci_idx + 1, len(lines)):
            if end_pat.match(lines[j]):
                end_idx = j
                break

        pre = lines[:sci_idx]
        body = lines[sci_idx + 1:end_idx]
        post = lines[end_idx:]

        blocks: dict[str, list[str]] = {}
        cur = None
        buf: list[str] = []

        def flush():
            nonlocal cur, buf
            if cur is not None:
                cleaned: list[str] = []
                for x in buf:
                    # remove any line-level numbering artifacts
                    cleaned.append(re.sub(r"^\s*\d+\.\s+", "", x))
                # trim empties
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                blocks[cur] = cleaned
            cur = None
            buf = []

        recognized = 0
        for ln in body:
            step_name, rest = match_required_sci_step_header(ln, required_steps)
            if step_name:
                flush()
                cur = step_name
                recognized += 1
                if rest:
                    buf.append(rest)
                continue
            if cur is not None:
                buf.append(ln)
        flush()
        if recognized < 2:
            return text

        # Build deterministic HTML
        # Keep styling minimal and consistent with existing CSS; rely on browser defaults.
        html_parts = [
            "<!-- SCI Trace: -->",
            "<div class='sci-trace' style='margin:10px 0; padding:10px; border:1px solid #ddd; border-radius:12px;'>",
            "<div style='font-weight:700; margin-bottom:6px;'>SCI Trace</div>",
            "<ol style='margin:0 0 0 22px; padding:0;'>"
        ]

        for step in required_steps:
            step = str(step)
            if step not in blocks:
                continue
            html_parts.append("<li style='margin:4px 0 10px 0;'>")
            html_parts.append(f"<div style='font-weight:700; margin:0 0 4px 0;'>{html.escape(step)}:</div>")

            # Step body: render as simple lines; convert list markers to bullets, preserve paragraphs.
            for ln in blocks[step]:
                t = ln.rstrip("\n")
                if not t.strip():
                    html_parts.append("<div style='height:6px'></div>")
                    continue
                m2 = re.match(r"^\s*([*+-]|•)\s+(.*)$", t)
                if m2:
                    html_parts.append(f"<div style='margin-left:14px;'>• {html.escape(m2.group(2).strip())}</div>")
                else:
                    html_parts.append(f"<div>{html.escape(t.strip())}</div>")

            html_parts.append("</li>")

        html_parts.extend(["</ol>", "</div>"])

        # Replace SCI trace section with HTML block. Keep a plain 'SCI Trace:' marker line for logs if needed.
        out_lines = []
        out_lines.extend(pre)
        out_lines.append("\n".join(html_parts))
        out_lines.extend(post)
        return "\n".join(out_lines)
    except Exception:
        return text


def _init_state_from_rules():
    if not gov.loaded:
        return GovernanceRuntimeState()
    try:
        if _state_init_from_ruleset is not None:
            dom = _state_init_from_ruleset(
                getattr(gov, "data", {}) or {},
                answer_language=(getattr(cfg, "get_answer_language", lambda: "de")() or "de"),
                conversation_language=(UI_LANG or "").lower() or "de",
                language_policy_mode=(getattr(cfg, "get_language_policy_mode", lambda: "production")() or "production"),
            )
            return GovernanceRuntimeState(
                comm_active=dom.comm_active,
                active_profile=dom.active_profile,
                overlay=dom.overlay,
                color=dom.color,
                conversation_language=dom.conversation_language,
                answer_language=dom.answer_language,
                language_policy_mode=getattr(dom, "language_policy_mode", "production"),
                sci_pending=dom.sci_pending,
                sci_variant=dom.sci_variant,
                sci_active=dom.sci_active,
            )
    except Exception:
        pass
    ui = gov.get_ui_data()
    prof = ui.get("defaults", {}).get("profile", "Standard") or "Standard"
    ov = ui.get("defaults", {}).get("overlay", "") or ""
    col = ui.get("defaults", {}).get("color_default", "on") or "on"
    return GovernanceRuntimeState(
        comm_active=True,
        active_profile=prof,
        overlay=ov,
        color=col,
        conversation_language=(UI_LANG or '').lower(),
        answer_language=(getattr(cfg, "get_answer_language", lambda: "de")() or "de"),
        language_policy_mode=(getattr(cfg, "get_language_policy_mode", lambda: "production")() or "production"),
        sci_pending=False,
        sci_variant="",
        sci_active=False,
    )


# --- HTML TEMPLATES ---

HTML_CHAT_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #f0f2f5; display: flex; flex-direction: column; height: 100vh; margin:0; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
  
  .msg { 
    padding: 12px 16px; 
    border-radius: 12px; 
    background: white; 
    border: 1px solid #ddd; 
    max-width: 85%; 
    line-height: 1.6; 
    position: relative; 
    user-select: text; 
  }
  
  .user { align-self: flex-end; background: #e8f0fe; border-right: 5px solid #1a73e8; }
  .bot { align-self: flex-start; border-left: 5px solid #34a853; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
  .sys { background: #333; color: #fff; font-family: monospace; font-size: 11px; align-self: center; width:auto; border:none; }
  .err { background: #fee; color: #c00; border: 1px solid #fcc; align-self: center; }

  .copy-btn {
    position: absolute;
    top: 5px;
    right: 5px;
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 14px;
    opacity: 0.3;
    transition: opacity 0.2s;
    padding: 2px;
    height: auto;
    width: auto;
  }
  .msg:hover .copy-btn { opacity: 1.0; }

  .ts-footer { display: block; width: 100%; border-top: 1px solid #eee; margin-top: 8px; padding-top: 4px; font-size: 10px; color: #888; text-align: right; }

  .raw-output-pre { background: #f8f9fa; padding: 10px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  details.raw-output { margin: 8px 0; }
  .note-box { background: #fff7ed; border-left: 4px solid #fb923c; padding: 8px 10px; border-radius: 6px; margin: 8px 0; }

  ul, ol { margin: 5px 0 5px 20px; padding: 0; }
  li { margin-bottom: 5px; }
  p { margin: 0 0 10px 0; }
  pre { background: #f8f9fa; padding: 10px; border-radius: 6px; overflow-x: auto; border: 1px solid #e1e4e8; }

  .input-area { padding: 15px; background: white; border-top: 1px solid #ccc; display: flex; gap: 10px; align-items: center; }
  textarea { flex: 1; height: 50px; padding: 10px; border-radius: 6px; border: 1px solid #ccc; font-family: inherit; resize: none; }
  button { padding: 0 20px; height: 50px; background: #1a73e8; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
  button:disabled { background: #ccc; cursor: not-allowed; }
  
  .top { background: #202124; color: white; padding: 8px 15px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
  .top-stats { margin-left: 10px; font-family: monospace; color: #aaa; border-left: 1px solid #555; padding-left: 10px;}
  .exit-btn { background: #a50e0e; font-size: 10px; height: auto; padding: 5px 10px; margin-left: 10px;}
  .menu-btn { font-size: 10px; height: auto; padding: 5px 10px; background: #444; }
  .load-btn { cursor: pointer; font-size: 12px; margin-right: 5px; background: transparent; border: none; color: white;}
  .exit-confirm-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.45); display: none; align-items: center; justify-content: center; z-index: 120000; }
  .exit-confirm-overlay.show { display: flex; }
  .exit-confirm-dialog { width: min(360px, calc(100vw - 24px)); background: #fff; border: 1px solid #cbd5e1; border-radius: 12px; box-shadow: 0 12px 30px rgba(0,0,0,0.25); padding: 14px; color: #0f172a; }
  .exit-confirm-title { margin: 0 0 8px 0; font-size: 14px; font-weight: 700; }
  .exit-confirm-text { margin: 0; font-size: 12px; color: #334155; }
  .exit-confirm-actions { margin-top: 14px; display: flex; justify-content: flex-end; gap: 8px; }
  .exit-confirm-actions button { height: auto; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #cbd5e1; }
  .exit-confirm-cancel { background: #f8fafc; color: #1f2937; }
  .exit-confirm-cancel:hover { background: #e2e8f0; }
  .exit-confirm-exit { background: #b91c1c; border-color: #b91c1c !important; color: #fff; }
  .exit-confirm-exit:hover { background: #991b1b; }

  .qc-bar { margin-top: 8px; border-top: 1px solid #eee; padding-top: 5px; font-size: 11px; color:#555; }
  .qc-btn { cursor: pointer; margin-right: 8px; color: #1a73e8; background:#f1f3f4; padding:2px 6px; border-radius:4px; }
  .qc-btn:hover { background:#1a73e8; color:white; }
  .cgi-widget { display:flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; }
  .cgi-title { font-weight: 700; color: #334155; margin-right: 2px; }
  .cgi-field { display:inline-flex; align-items:center; gap:4px; color:#475569; }
  .cgi-select { height:24px; border:1px solid #cbd5e1; border-radius:4px; background:#fff; color:#0f172a; font-size:11px; padding:0 4px; }
  .cgi-sep { color:#94a3b8; margin: 0 1px; }
  .cgi-action { height:24px !important; padding:0 8px !important; border-radius:4px !important; border:1px solid #93c5fd !important; background:#e8f0fe !important; color:#1a73e8 !important; font-size:11px !important; font-weight:600 !important; line-height:22px !important; cursor:pointer; }
  .cgi-action:hover { background:#1a73e8 !important; color:#fff !important; }
  .cgi-action.repeat { border-color:#60a5fa !important; }
  .cgi-help { display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:50%; background:#e2e8f0; color:#334155; font-weight:700; cursor:help; font-size:11px; }
  .cgi-status { min-height:16px; color:#475569; font-size:11px; }
  .cgi-hint-line { width:100%; color:#64748b; font-size:10px; margin-top:2px; }

  .csc-badge {
    position: absolute;
    top: 6px;
    left: 10px;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid #f9ab00;
    background: #fff7e0;
    color: #8a4f00;
  }

  .csc-warning {
    border: 1px solid #f9ab00;
    background: #fff7e0;
    padding: 10px;
    border-radius: 10px;
    margin: 0 0 8px 0;
    color: #3c2b00;
  }

  .csc-warning summary { cursor: pointer; font-weight: 600; }
  .csc-warning .csc-details { margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }


  /* --- Comm Help (deterministic HTML renderer) --- */
  .comm-help { font-size: inherit; line-height: inherit; }
  .comm-help .help-status { font-size: inherit; color:#444; margin: 0 0 10px 0; }
  .comm-help h3 { margin: 14px 0 8px 0; font-size: 1em; }
  .comm-help .didactic { margin: 8px 0 12px 0; font-style: italic; color:#555; }
  .comm-help .cmd-dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; margin: 0; }
  .comm-help .cmd-dl dt { margin: 0; }
  .comm-help .cmd-dl dd { margin: 0; }
  .comm-help code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background:#f6f8fa; padding: 1px 5px; border-radius: 6px; }
  .comm-help code.nowrap { white-space: nowrap; }
  .comm-help .section-note { color:#666; font-size: 12px; margin: 0 0 8px 0; }
  .comm-help .variants { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; margin: 0; }
  .comm-help .variants .vkey { font-weight: 700; }
  .comm-help .numcodes { margin: 0; }
  .comm-help .numcodes-table { border-collapse: collapse; width: 100%; font-size: 12px; }
  .comm-help .numcodes-table th, .comm-help .numcodes-table td { border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }
  .comm-help .numcodes-table th { background: #f9fafb; text-align: left; }
  .comm-help .muted { color:#666; font-size: 11px; }
  .comm-help .opts code { padding: 0 4px; }
  .comm-help .dash-note { margin-top: 8px; color:#444; font-size: 12px; }

  /* Comm State / Comm Config (scoped) */
  .comm-help .state-table { border-collapse: collapse; width: 100%; font-size: 12px; }
  .comm-help .state-table th, .comm-help .state-table td { border: 1px solid #e0e0e0; padding: 6px 8px; vertical-align: top; }
  .comm-help .state-table th { text-align: left; background: #fafafa; width: 220px; }
  .comm-help pre.raw-json { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; box-sizing: border-box; overflow-x: auto; background: #f6f8fa; padding: 10px; border-radius: 10px; font-size: 11px; border: 1px solid #e0e0e0; }
  .comm-help details.config-details > summary { cursor: pointer; font-weight: 600; margin: 8px 0; }
  .comm-help .minor { color:#666; font-size: 12px; }


  /* SCI Menu (deterministic HTML) */
  .comm-help .sci-table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
  .comm-help .sci-table th, .comm-help .sci-table td { border: 1px solid #e0e0e0; padding: 6px 8px; vertical-align: top; }
  .comm-help .sci-table th { text-align: left; background: #fafafa; width: 80px; }

  /* Anchor Snapshot (deterministic HTML) */
  .comm-help .anchor-box { background:#f3f4f6; border: 1px solid #d1d5db; border-radius: 12px; padding: 10px 12px; }
  .comm-help .anchor-box .anchor-badge { display:inline-block; font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; background:#e5e7eb; margin-bottom:8px; }
  .comm-help .anchor-box pre { margin: 8px 0 0 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }

</style>
</head>
<body>
  <div class="top">
    <div style="display:flex; align-items:center;">
        <button class="load-btn" onclick="window.pywebview.api.load_rule_file()" title="Load ruleset">📂</button>
        <span id="rulefile" style="font-size:11px; color:#8ab4f8; margin-right:10px;"></span>
        
        <span id="stats" class="top-stats">Session: loading...</span>
    </div>
    <div>
       <button class="menu-btn" onclick="window.pywebview.api.export()">💾 EXPORT</button>
       <button class="menu-btn" onclick="window.pywebview.api.settings()">⚙️ PANEL</button>
       <button class="exit-btn" onclick="openExitConfirm()">❌ EXIT</button>
    </div>
  </div>

  <div id="exitConfirmOverlay" class="exit-confirm-overlay" aria-hidden="true" role="dialog" aria-modal="true" aria-labelledby="exitConfirmTitle">
    <div class="exit-confirm-dialog" onclick="event.stopPropagation()">
      <h3 id="exitConfirmTitle" class="exit-confirm-title">Exit Application</h3>
      <p class="exit-confirm-text">Do you really want to close Comm-SCI-Control-App?</p>
      <div class="exit-confirm-actions">
        <button id="exitConfirmCancelBtn" class="exit-confirm-cancel" type="button" onclick="closeExitConfirm()">Cancel</button>
        <button class="exit-confirm-exit" type="button" onclick="confirmExit()">Exit</button>
      </div>
    </div>
  </div>
  
  <div id="chat">
    <div class="msg sys" id="status">System initialized...</div>
  </div>
  
  <div class="input-area">
    <textarea id="inp" placeholder="Please wait..." disabled></textarea>
    <button id="btn" onclick="send()" disabled>...</button>
  </div>

<script>
  let __readyChecks = 0;
  let checkInterval = setInterval(async () => {
      const res = await window.pywebview.api.is_ready();
      __readyChecks++;
      // Auto-open the panel even if the system is not ready yet (e.g., missing API keys).
      try {
          const msg = (res && res.msg) ? String(res.msg).toLowerCase() : '';
          if(!window.__panel_auto_shown && (__readyChecks >= 3 || msg.includes('key missing') || msg.includes('api key') || msg.includes('openrouter'))) {
              window.__panel_auto_shown = true;
              window.pywebview.api.ensure_panel_visible();
          }
      } catch(e) {}
      if(res.status === true) {
          clearInterval(checkInterval);
          document.getElementById('status').innerText = "System ready: " + res.msg;
          
          if(res.filename) {
             // Nur den Dateinamen anzeigen, nicht den ganzen Pfad
             const parts = res.filename.split(/[\\\\/]/);
             document.getElementById('rulefile').innerText = "[" + parts.pop() + "]";
          }

          document.getElementById('inp').disabled = false;
          document.getElementById('inp').placeholder = "Command or message...";
          document.getElementById('btn').disabled = false;
          document.getElementById('btn').innerText = "Send";
          document.getElementById('inp').focus();
          window.pywebview.api.update_stats_ui();
          // Auto-show panel once after rules are loaded
          if(!window.__panel_auto_shown){ window.__panel_auto_shown = true; window.pywebview.api.ensure_panel_visible(); }

      } else {
          document.getElementById('status').innerText = res.msg;
          if(res.msg.includes("ERROR")) document.getElementById('status').className = "msg err";
      }
  }, 500);

  function updateStats(text) {
      document.getElementById('stats').innerText = text;
  }
  
  function updateRuleFile(name) {
      const parts = name.split(/[\\\\/]/);
      document.getElementById('rulefile').innerText = "[" + parts.pop() + "]";
  }

  function openExitConfirm(){
      const ov = document.getElementById('exitConfirmOverlay');
      if(!ov) return;
      ov.classList.add('show');
      ov.setAttribute('aria-hidden', 'false');
      try {
          const btn = document.getElementById('exitConfirmCancelBtn');
          if(btn) btn.focus();
      } catch(e) {}
  }

  function closeExitConfirm(){
      const ov = document.getElementById('exitConfirmOverlay');
      if(!ov) return;
      ov.classList.remove('show');
      ov.setAttribute('aria-hidden', 'true');
  }

  async function confirmExit(){
      closeExitConfirm();
      try {
          await window.pywebview.api.close_app();
      } catch(e) {
          console.error('close_app failed', e);
      }
  }

  function _bindExitConfirm(){
      const ov = document.getElementById('exitConfirmOverlay');
      if(!ov) return;
      ov.addEventListener('click', (e)=>{
          if(e.target === ov) closeExitConfirm();
      });
      document.addEventListener('keydown', (e)=>{
          if(String(e.key || '') !== 'Escape') return;
          if(!ov.classList.contains('show')) return;
          e.preventDefault();
          closeExitConfirm();
      });
  }
  _bindExitConfirm();

  function escHtml(s){
      return (''+s).replace(/[&<>"']/g, (c)=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  }

  function renderCscBlock(csc){
      if(!csc || !csc.applied) return '';
      const msg = escHtml(csc.message || 'CSC applied.');
      const trig = escHtml(csc.trigger || '');
      const mode = escHtml(csc.mode || '');
      const ov = escHtml(csc.overlay || '');
      const prof = escHtml(csc.profile || '');
      const fs = escHtml(csc.f_score ?? '');
      const tok = escHtml(csc.token_count ?? '');
      const mult = escHtml(csc.threshold_multiplier ?? '');
      const thrFs = escHtml(csc.threshold_f_score ?? '');
      const thrTok = escHtml((csc.min_token_count ?? csc.threshold_token_count) ?? '');
      const thrGov = escHtml(csc.min_token_count_governance ?? '');
      const scoreTip = escHtml(csc.score_tooltip || '');
      const thrTip = escHtml(csc.thresholds_tooltip || '');
      const scoreInfo = scoreTip ? ` <span class="cgi-help csc-help-icon" data-u-title="${scoreTip}">i</span>` : '';
      const thrInfo = thrTip ? ` <span class="cgi-help csc-help-icon" data-u-title="${thrTip}">i</span>` : '';

      let details = '';
      details += `<details><summary>Details</summary>`;
      details += `<div class="csc-details">`;
      details += `Trigger: ${trig || '—'}<br>`;
      details += `Mode: ${mode || '—'}${csc.governance_triggered ? ' (governance)' : ''}<br>`;
      details += `Overlay: ${ov || 'off'} · Profile: ${prof || '—'}<br>`;
      details += `Score: f=${fs} · tokens=${tok}${scoreInfo}<br>`;
      details += `Thresholds (x${mult || 1}): f>=${thrFs}, tok>=${thrTok}, gov_tok>=${thrGov}${thrInfo}`;
      details += `</div></details>`;

      return `<span class="csc-badge">CSC applied</span>` +
             `<div class="csc-warning"><b>CSC</b>: ${msg}${details}</div>`;
  }

  function _cgiLang(lang){
      const l = String(lang || '').trim().toLowerCase();
      if(l.startsWith('en')) return 'en';
      return 'de';
  }

  function _cgiTexts(lang){
      const useEn = _cgiLang(lang) === 'en';
      if(useEn){
          return {
              title: 'CGI Feedback:',
              clarity: 'Clarity gain',
              insight: 'Insight gain',
              efficiency: 'Efficiency gain',
              repeat: 'Repeat with adjustments',
              help: 'Optional user feedback (0-3) for clarity, insight, and efficiency. Repeat re-runs the last content question with this feedback.',
              hint: '0 = weak (stronger rewrite), 3 = good (lighter rewrite). Effect applies to the next answer only.',
              repeating: 'Applying CGI feedback and repeating last content answer...',
              action_failed: 'CGI feedback action could not be executed.',
              invalid: 'Please select values from 0 to 3 for all three criteria.'
          };
      }
      return {
          title: 'CGI-Feedback:',
          clarity: 'Klarheitsgewinn',
          insight: 'Erkenntnisgewinn',
          efficiency: 'Effizienzgewinn',
          repeat: 'Antwort mit Anpassungen wiederholen',
          help: 'Optionales Nutzerfeedback (0-3): 0 bedeutet schwach und erzwingt eine staerkere Ueberarbeitung, 3 bedeutet gut und nur leichte Anpassung. Die Wirkung gilt nur fuer die naechste Antwort.',
          hint: '0 = schlecht (staerkere Anpassung), 3 = gut (geringere Anpassung). Wirkung nur fuer die naechste Antwort.',
          repeating: 'CGI-Feedback wird angewendet, letzte Inhaltsfrage wird neu beantwortet...',
          action_failed: 'CGI-Aktion konnte nicht ausgeführt werden.',
          invalid: 'Bitte für alle drei Kriterien Werte von 0 bis 3 wählen.'
      };
  }

  function _cgiOptions(){
      return '<option value="0">0</option><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option>';
  }

  let __cgiCounter = 0;
  function _buildCgiWidgetHtml(answerLang){
      const lang = _cgiLang(answerLang);
      const t = _cgiTexts(lang);
      __cgiCounter += 1;
      const id = 'cgi-widget-' + String(__cgiCounter);
      return (
          `<div class="qc-bar cgi-widget" id="${id}" data-cgi-lang="${lang}">` +
          `<span class="cgi-title">${escHtml(t.title)}</span>` +
          `<label class="cgi-field">${escHtml(t.clarity)} <select class="cgi-select" data-cgi-field="clarity">${_cgiOptions()}</select></label>` +
          `<span class="cgi-sep">/</span>` +
          `<label class="cgi-field">${escHtml(t.insight)} <select class="cgi-select" data-cgi-field="insight">${_cgiOptions()}</select></label>` +
          `<span class="cgi-sep">/</span>` +
          `<label class="cgi-field">${escHtml(t.efficiency)} <select class="cgi-select" data-cgi-field="efficiency">${_cgiOptions()}</select></label>` +
          `<button type="button" class="cgi-action repeat" onclick="submitCgi('${id}','repeat')">${escHtml(t.repeat)}</button>` +
          `<span class="cgi-help" data-u-title="${escHtml(t.help)}">i</span>` +
          `<span class="cgi-status"></span>` +
          `<div class="cgi-hint-line">${escHtml(t.hint)}</div>` +
          `</div>`
      );
  }

  function _copyTip(answerLang){
      return _cgiLang(answerLang) === 'en'
          ? 'Copy message to clipboard.'
          : 'Nachricht in die Zwischenablage kopieren.';
  }

  function _qcDimKey(label){
      const raw = String(label || '').trim().toLowerCase();
      const map = {
          'clarity': 'clarity',
          'brevity': 'brevity',
          'evidence': 'evidence',
          'empathy': 'empathy',
          'consistency': 'consistency',
          'neutrality': 'neutrality',
          'klarheit': 'clarity',
          'kürze': 'brevity',
          'kuerze': 'brevity',
          'evidenz': 'evidence',
          'empathie': 'empathy',
          'konsistenz': 'consistency',
          'neutralität': 'neutrality',
          'neutralitaet': 'neutrality',
      };
      return map[raw] || '';
  }

  function _qcDimScaleRows(dimKey, lang){
      const de = {
          clarity: ['unklar / schwer lesbar', 'grundlegend klar', 'klar und gut strukturiert', 'sehr klar, didaktisch stark'],
          brevity: ['sehr ausfuehrlich / lang', 'eher ausfuehrlich', 'ausgewogen', 'sehr knapp / stark verdichtet'],
          evidence: ['kaum Belege', 'einige Begruendungen', 'solide belegt', 'stark belegt, gut nachverfolgbar'],
          empathy: ['sehr sachlich / distanziert', 'hoeflich, eher distanziert', 'ruecksichtsvoll', 'sehr unterstuetzend'],
          consistency: ['Widersprueche moeglich', 'ueberwiegend konsistent', 'konsistente Logik', 'sehr strikt konsistent'],
          neutrality: ['deutlich wertend', 'leichte Tendenz moeglich', 'weitgehend neutral', 'streng neutral / ausbalanciert'],
      };
      const en = {
          clarity: ['unclear / hard to follow', 'basically clear', 'clear and well structured', 'very clear, highly didactic'],
          brevity: ['very detailed / long', 'rather detailed', 'balanced length', 'very concise / compressed'],
          evidence: ['little support', 'some justification', 'solid support', 'strongly supported and traceable'],
          empathy: ['very factual / distant', 'polite but somewhat distant', 'considerate tone', 'highly supportive tone'],
          consistency: ['contradictions possible', 'mostly consistent', 'consistent logic', 'very strict consistency'],
          neutrality: ['clearly opinionated', 'slight bias possible', 'mostly neutral', 'strictly neutral / balanced'],
      };
      const rows = (lang === 'en' ? en : de)[dimKey] || (lang === 'en' ? ['low', 'basic', 'good', 'high'] : ['niedrig', 'grundlegend', 'gut', 'hoch']);
      return rows.map((txt, idx) => `${idx} | ${txt}`);
  }

  function _qcDimTipText(dimKey, value, delta, answerLang){
      const lang = _cgiLang(answerLang);
      const v = Number.isFinite(Number(value)) ? Number(value) : null;
      const d = String(delta || '').replace('−', '-');
      const baseEN = {
          clarity: 'Clarity: readability and structure quality.',
          brevity: 'Brevity: conciseness versus detail depth.',
          evidence: 'Evidence: support and traceability of claims.',
          empathy: 'Empathy: considerate and supportive tone.',
          consistency: 'Consistency: internal logic and contradiction control.',
          neutrality: 'Neutrality: unbiased and balanced wording.',
      };
      const baseDE = {
          clarity: 'Clarity: Verstaendlichkeit und Struktur der Antwort.',
          brevity: 'Brevity: Kuerze im Verhaeltnis zur Detailtiefe.',
          evidence: 'Evidence: Belegbarkeit und Nachvollziehbarkeit der Aussagen.',
          empathy: 'Empathy: ruecksichtsvolle, unterstuetzende Tonalitaet.',
          consistency: 'Consistency: innere Logik und Widerspruchsfreiheit.',
          neutrality: 'Neutrality: neutrale, ausgewogene Formulierung.',
      };
      const base = (lang === 'en' ? baseEN : baseDE)[dimKey] || (lang === 'en' ? 'QC dimension.' : 'QC-Dimension.');
      const vTxt = (v === null)
          ? (lang === 'en' ? 'Current value: n/a.' : 'Aktueller Wert: n/a.')
          : (lang === 'en' ? `Current value: ${v} of 3.` : `Aktueller Wert: ${v} von 3.`);
      const dTxt = d
          ? (lang === 'en' ? `Delta ${d}: offset to profile target.` : `Delta ${d}: Abweichung zum Profilziel.`)
          : (lang === 'en' ? 'Delta n/a.' : 'Delta n/a.');
      const hdr = (lang === 'en') ? 'Scale 0-3 (table):' : 'Skala 0-3 (Tabelle):';
      const rows = _qcDimScaleRows(dimKey, lang).join('\n');
      return `${base}\n${vTxt} ${dTxt}\n${hdr}\n${rows}`;
  }

  function _decorateQcMatrixTooltips(root, answerLang){
      if(!root) return;
      const src = String(root.innerHTML || '');
      if(src.indexOf('QC-Matrix:') < 0) return;
      if(src.indexOf('qc-dim-tip') >= 0) return;
      const re = /(Clarity|Brevity|Evidence|Empathy|Consistency|Neutrality|Klarheit|Kürze|Kuerze|Evidenz|Empathie|Konsistenz|Neutralität|Neutralitaet)\\s+([0-3])\\s*\\(\\s*Δ\\s*([+\\-−]?\\d+)\\s*\\)/g;
      const out = src.replace(re, (full, label, value, delta) => {
          const key = _qcDimKey(label);
          if(!key) return full;
          const tip = _qcDimTipText(key, value, delta, answerLang);
          return `<span class="qc-dim-tip" data-u-title="${escHtml(tip)}">${full}</span>`;
      });
      if(out !== src) root.innerHTML = out;
  }

  function _normalizeCustomTooltipTargets(root){
      if(!root || !root.querySelectorAll) return;
      root.querySelectorAll('.uncertainty-inline-marker, .signal-dot-marker, .cgi-help, .csc-badge, .csc-warning, .control-layer-note, .copy-btn, .qc-dim-tip').forEach((el)=>{
          try {
              const tip = String(el.getAttribute('data-u-title') || '').trim();
              const nativeTitle = String(el.getAttribute('title') || '').trim();
              if(!tip && nativeTitle){
                  el.setAttribute('data-u-title', nativeTitle);
              }
              if(nativeTitle){
                  el.removeAttribute('title');
              }
          } catch(e) {}
      });
  }

  async function submitCgi(widgetId, mode){
      const root = document.getElementById(String(widgetId || ''));
      if(!root) return;
      const lang = _cgiLang(root.getAttribute('data-cgi-lang') || '');
      const t = _cgiTexts(lang);
      const statusEl = root.querySelector('.cgi-status');
      const kEl = root.querySelector('select[data-cgi-field="clarity"]');
      const iEl = root.querySelector('select[data-cgi-field="insight"]');
      const eEl = root.querySelector('select[data-cgi-field="efficiency"]');
      const actButtons = root.querySelectorAll('button.cgi-action');
      const sel = root.querySelectorAll('select.cgi-select');
      const k = parseInt(kEl ? kEl.value : '', 10);
      const i = parseInt(iEl ? iEl.value : '', 10);
      const e = parseInt(eEl ? eEl.value : '', 10);
      if([k,i,e].some(v => Number.isNaN(v) || v < 0 || v > 3)){
          if(statusEl){
              statusEl.style.color = '#b91c1c';
              statusEl.textContent = t.invalid;
          }
          return;
      }
      const busy = t.repeating;
      if(statusEl){
          statusEl.style.color = '#475569';
          statusEl.textContent = busy;
      }
      actButtons.forEach(b => { b.disabled = true; });
      sel.forEach(s => { s.disabled = true; });
      try{
          const res = await window.pywebview.api.submit_cgi_feedback(k, i, e, String(mode || 'repeat'));
          if(!res || !res.ok){
              const err = (res && res.error) ? String(res.error) : t.action_failed;
              if(statusEl){
                  statusEl.style.color = '#b91c1c';
                  statusEl.textContent = err;
              }
              return;
          }
          if(statusEl){
              statusEl.style.color = '#166534';
              statusEl.textContent = String(res.message || '');
          }
          if(res.repeated && res.response){
              const rr = res.response || {};
              const rrLang = String(rr.answer_lang || lang || '');
              addMsg('bot', String(rr.html || ''), !!rr.cgi_bar, rr.csc || null, {answerLang: rrLang});
          }
      } catch(e2){
          if(statusEl){
              statusEl.style.color = '#b91c1c';
              statusEl.textContent = t.action_failed + ' ' + String(e2);
          }
      } finally {
          actButtons.forEach(b => { b.disabled = false; });
          sel.forEach(s => { s.disabled = false; });
      }
  }

  function addMsg(role, text, qc=false, csc=null, opts={}) {
      const d = document.createElement('div');
      d.className = 'msg ' + role;
      const copyTip = escHtml(_copyTip((opts && opts.answerLang) ? opts.answerLang : ''));
      let html = `<button class="copy-btn" onclick="copyToClipboard(this)" data-u-title="${copyTip}">📋</button>`;
      if(role === 'bot') html += renderCscBlock(csc);
      html += text;
      if(qc && role === 'bot') html += _buildCgiWidgetHtml((opts && opts.answerLang) ? opts.answerLang : '');
      d.innerHTML = html;
      _decorateQcMatrixTooltips(d, (opts && opts.answerLang) ? opts.answerLang : '');
      _normalizeCustomTooltipTargets(d);
      document.getElementById('chat').appendChild(d);
      document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
      if(window.MathJax) MathJax.typesetPromise();
  }

  function copyToClipboard(btn) {
      const msgDiv = btn.parentElement;
      const clone = msgDiv.cloneNode(true);
      const unwanted = clone.querySelectorAll('.copy-btn, .qc-bar, .ts-footer, .csc-warning, .csc-badge');
      unwanted.forEach(el => el.remove());
      const textToCopy = clone.innerText.trim();
      
      const textArea = document.createElement("textarea");
      textArea.value = textToCopy;
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      textArea.style.top = "0";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      
      try {
          const successful = document.execCommand('copy');
          if(successful) {
              const originalText = btn.innerText;
              btn.innerText = "✅";
              setTimeout(() => btn.innerText = originalText, 1500);
          } else {
              btn.innerText = "❌";
          }
      } catch (err) {
          console.error('Fallback copy failed', err);
          btn.innerText = "❌";
      }
      document.body.removeChild(textArea);
  }

  let __uTipEl = null;
  function _uTipHide(){
      if(!__uTipEl) return;
      try { __uTipEl.remove(); } catch(e) {}
      __uTipEl = null;
  }

  function _uTipShow(target, ev){
      if(!target) return;
      const txt = String(target.getAttribute('data-u-title') || '').trim();
      if(!txt) return;
      _uTipHide();
      const tip = document.createElement('div');
      tip.className = 'uncertainty-tooltip';
      tip.textContent = txt;
      tip.style.position = 'fixed';
      tip.style.zIndex = '99999';
      tip.style.maxWidth = '420px';
      tip.style.padding = '8px 10px';
      tip.style.border = '1px solid #93c5fd';
      tip.style.borderRadius = '8px';
      tip.style.background = '#eff6ff';
      tip.style.color = '#1e3a8a';
      tip.style.fontSize = '12px';
      tip.style.lineHeight = '1.35';
      tip.style.whiteSpace = 'pre-line';
      tip.style.boxShadow = '0 6px 16px rgba(0,0,0,0.18)';
      tip.style.pointerEvents = 'none';
      document.body.appendChild(tip);
      __uTipEl = tip;
      try {
          const rect = tip.getBoundingClientRect();
          const vw = window.innerWidth || document.documentElement.clientWidth || 1024;
          const vh = window.innerHeight || document.documentElement.clientHeight || 768;
          const cx = (ev && typeof ev.clientX === 'number') ? ev.clientX : Math.round(vw / 2);
          const cy = (ev && typeof ev.clientY === 'number') ? ev.clientY : Math.round(vh / 2);
          let left = cx + 12;
          let top = cy + 12;
          if(left + rect.width + 8 > vw) left = Math.max(8, vw - rect.width - 8);
          if(top + rect.height + 8 > vh) top = Math.max(8, cy - rect.height - 12);
          tip.style.left = left + 'px';
          tip.style.top = top + 'px';
      } catch(e) {}
  }

  const __uTipTargets = '.uncertainty-inline-marker, .signal-dot-marker, .cgi-help, .csc-badge, .csc-warning, .control-layer-note, .copy-btn, .qc-dim-tip, [data-u-title]';
  document.addEventListener('mousedown', (e)=>{
      if(typeof e.button === 'number' && e.button !== 0) return;
      const t = (e.target && e.target.closest)
          ? e.target.closest(__uTipTargets)
          : null;
      if(!t) return;
      _uTipShow(t, e);
  });
  document.addEventListener('mouseover', (e)=>{
      const t = (e.target && e.target.closest)
          ? e.target.closest(__uTipTargets)
          : null;
      if(!t) return;
      _uTipShow(t, e);
  });
  document.addEventListener('mouseout', (e)=>{
      const t = (e.target && e.target.closest)
          ? e.target.closest(__uTipTargets)
          : null;
      if(!t) return;
      _uTipHide();
  });
  document.addEventListener('mouseup', _uTipHide);
  document.addEventListener('dragstart', _uTipHide);
  document.addEventListener('scroll', _uTipHide, true);
  window.addEventListener('blur', _uTipHide);

  window.__cmdHistory = window.__cmdHistory || {
      entries: [],
      index: -1,
      draft: '',
      maxEntries: 200
  };

  function _cmdHistNormalizeEntries(entries, maxEntries){
      const arr = Array.isArray(entries) ? entries : [];
      const out = [];
      for(const v of arr){
          const txt = String(v || '').trim();
          if(!txt) continue;
          if(out.length && out[out.length - 1] === txt) continue;
          out.push(txt);
      }
      const lim = Math.max(20, parseInt(maxEntries || 200, 10) || 200);
      while(out.length > lim) out.shift();
      return out;
  }

  async function _cmdHistLoadFromBackend(){
      const h = window.__cmdHistory;
      if(!h) return;
      try{
          if(!window.pywebview || !window.pywebview.api || !window.pywebview.api.get_input_history) return;
          const payload = await window.pywebview.api.get_input_history(parseInt(h.maxEntries || 200, 10) || 200);
          if(!payload || payload.ok !== true) return;
          if(typeof payload.max_entries === 'number' && Number.isFinite(payload.max_entries)){
              h.maxEntries = Math.max(20, parseInt(payload.max_entries, 10) || 200);
          }
          h.entries = _cmdHistNormalizeEntries(payload.entries || [], h.maxEntries);
          _cmdHistResetBrowse();
      } catch(_e){
          // silent fail-open
      }
  }

  function _cmdHistResetBrowse() {
      const h = window.__cmdHistory;
      if(!h) return;
      h.index = -1;
      h.draft = '';
  }

  function _cmdHistPush(raw) {
      const txt = String(raw || '').trim();
      if(!txt) return;
      const h = window.__cmdHistory;
      if(!h) return;
      const arr = Array.isArray(h.entries) ? h.entries : [];
      const last = arr.length ? String(arr[arr.length - 1] || '') : '';
      if(last !== txt) arr.push(txt);
      const maxN = Math.max(20, parseInt(h.maxEntries || 200, 10) || 200);
      while(arr.length > maxN) arr.shift();
      h.entries = arr;
      _cmdHistResetBrowse();
  }

  function _cmdHistBrowse(currentValue, step) {
      const h = window.__cmdHistory;
      const cur = String(currentValue || '');
      if(!h || !Array.isArray(h.entries) || !h.entries.length) return cur;
      const dir = (step < 0) ? -1 : 1;
      if(parseInt(h.index || -1, 10) < 0){
          if(dir > 0) return cur;
          h.draft = cur;
          h.index = h.entries.length - 1;
          return String(h.entries[h.index] || '');
      }
      let next = parseInt(h.index || -1, 10) + dir;
      if(next < 0){
          h.index = -1;
          return String(h.draft || '');
      }
      if(next >= h.entries.length){
          h.index = -1;
          return String(h.draft || '');
      }
      h.index = next;
      return String(h.entries[h.index] || '');
  }

  function _cmdHistApply(step){
      const inp = document.getElementById('inp');
      if(!inp) return;
      inp.value = _cmdHistBrowse(inp.value, step);
  }

  let _sendInFlight = false;
  const _sendQueue = [];
  const _sendQueueMax = 20;

  function _enqueueSend(txt) {
      const s = String(txt || '').trim();
      if(!s) return false;
      if(_sendQueue.length >= _sendQueueMax){
          // Keep bounded memory; prefer latest user intent.
          _sendQueue.shift();
      }
      _sendQueue.push(s);
      return true;
  }

  async function _drainSendQueue() {
      if(_sendInFlight) return;
      const inp = document.getElementById('inp');
      const btn = document.getElementById('btn');
      _sendInFlight = true;
      try {
          while(_sendQueue.length){
              const txt = String(_sendQueue.shift() || '').trim();
              if(!txt) continue;
              _cmdHistPush(txt);
              _cmdHistResetBrowse();
              try {
                  if(window.pywebview && window.pywebview.api && window.pywebview.api.append_input_history){
                      await window.pywebview.api.append_input_history(txt);
                  }
              } catch(_e) {
                  // fail-open: local history still works
              }
              addMsg('user', txt);
              if(btn) btn.disabled = true;
              try {
                  const res = await window.pywebview.api.ask(txt);
                  const qcEnabled = await window.pywebview.api.ui_qc_bar_enabled();
                  if(typeof res === 'string') {
                      addMsg('bot', res, false);
                  } else {
                      const showCgi = !!qcEnabled && !!(res && res.cgi_bar);
                      const answerLang = (res && typeof res.answer_lang === 'string') ? res.answer_lang : '';
                      addMsg('bot', res.html || '', showCgi, res.csc || null, {answerLang: answerLang});
                  }
              } catch(e) {
                  addMsg('bot', '<span style="color:red">Error: '+e+'</span>');
              }
          }
      } finally {
          _sendInFlight = false;
          if(btn) btn.disabled = false;
          if(inp) inp.focus();
      }
  }

  async function send() {
      const inp = document.getElementById('inp');
      if(!inp) return;
      const txt = (inp.value || '').trim();
      if(!txt) return;
      inp.value = '';
      if(!_enqueueSend(txt)) return;
      await _drainSendQueue();
  }

  function remoteInput(txt) {
      if(!_enqueueSend(txt)) return;
      const inp = document.getElementById('inp');
      if(inp) inp.value = '';
      _cmdHistResetBrowse();
      _drainSendQueue();
  }
  
  document.getElementById('inp').addEventListener('keydown', (e)=>{
      if(e.key==='Enter' && !e.shiftKey){e.preventDefault(); send(); return;}
      if(e.key==='ArrowUp' && !e.shiftKey){e.preventDefault(); _cmdHistApply(-1); return;}
      if(e.key==='ArrowDown' && !e.shiftKey){e.preventDefault(); _cmdHistApply(1); return;}
  });
  document.getElementById('inp').addEventListener('input', ()=>{
      const h = window.__cmdHistory;
      if(!h) return;
      if(parseInt(h.index || -1, 10) >= 0){
          h.index = -1;
      }
  });
  _cmdHistLoadFromBackend();
  window.addEventListener('pywebviewready', _cmdHistLoadFromBackend);
  setTimeout(_cmdHistLoadFromBackend, 350);

// --- Panel helpers: allow Python to replay a loaded chat log into the main UI (no model call).
function resetChatToStatus(msg) {
  const chat = document.getElementById('chat');
  if (!chat) return;
  chat.innerHTML = '';
  const st = document.createElement('div');
  st.className = 'msg sys';
  st.id = 'status';
  st.textContent = msg || 'System initialized...';
  chat.appendChild(st);
}

window.resetChatFromHistory = function(history, statusMsg) {
  try {
    resetChatToStatus(statusMsg || 'Loaded chat log.');
    if (!Array.isArray(history)) return;
    for (const m of history) {
      if (!m) continue;
      const role = (m.role === 'bot' || m.role === 'assistant') ? 'bot' : (m.role === 'sys' || m.role === 'system' ? 'sys' : 'user');
      const text = (m.html != null) ? String(m.html) : (m.content != null ? String(m.content) : '');
      if (role === 'bot') addMsg('bot', text, false, null);
      else addMsg(role, escHtml(text));
    }
    chat.scrollTop = chat.scrollHeight;
  } catch (e) {
    addMsg('sys', 'resetChatFromHistory failed: ' + String(e));
  }
};
</script>
</body>
</html>
"""
HTML_CHAT = HTML_CHAT_TEMPLATE.replace('__WRAPPER_LABEL__', html.escape(WRAPPER_NAME))

HTML_PANEL = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body { font-family: sans-serif; padding: 10px; background: #f8f9fa; user-select: none; }
  h4 { margin: 12px 0 6px 0; color: #1a73e8; border-bottom: 2px solid #dae0e5; font-size: 11px; text-transform: uppercase; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
  button { padding: 8px 5px; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-size: 11px; text-align:center; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
  button:hover { background: #e8f0fe; border-color: #1a73e8; color: #1a73e8; }
  .setting-select { width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #999; margin-bottom: 5px; font-weight: bold; font-size: 12px;}
  .card { background: white; padding: 8px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #ddd; }
  .log-box { font-family: monospace; font-size: 9px; color: #333; margin-bottom: 8px; max-height: 70px; overflow-y: auto; background: #eee; padding: 6px; border-radius: 6px; border: 1px solid #ddd; }
  .test-log-box { font-family: monospace; font-size: 10px; color: #222; margin-top: 6px; max-height: 160px; overflow-y: auto; background: #f4f6f8; padding: 6px; border-radius: 6px; border: 1px solid #ddd; white-space: pre-wrap; }
  .status-box { font-family: monospace; font-size: 10px; color: #111; margin-bottom: 8px; background: #fff; padding: 6px; border-radius: 6px; border: 1px solid #ddd; }
  .hint { font-size: 9px; color: #666; margin-bottom: 8px; text-align: center; }
  .row { display:flex; gap:6px; }
  .row > * { flex: 1; }
  .smallbtn { padding: 7px 6px; font-size: 10px; }
  .err { color: #b00020; }
  .ok { color: #0b6b0b; }
  .warn { color: #a05a00; }
</style>
</head>
<body>

<div class="card">
  <div class="row">
    <select id="provider" class="setting-select" onchange="changeProvider()">
      <option value="gemini">Provider: Gemini</option>
      <option value="openrouter">Provider: OpenRouter</option>
      <option value="huggingface">Provider: Hugging Face</option>
    </select>
    <button id="refreshModelsBtn" class="smallbtn" onclick="refreshModels()" title="Fetch provider models and refresh cache (Gemini/OpenRouter/HF)">Refresh Models</button>
    <button class="smallbtn" id="qcOverrideBtn" onclick="run('QC Override')" title="QC Override">⚙ QC</button>
</div>

  <div id="hfCatalogRow" class="row" style="display:none;">
    <select id="hfProviderFilter" class="setting-select" onchange="onHFProviderFilterChange()">
      <option value="all">HF Provider: all</option>
    </select>
    <input id="hfTopN" class="setting-select" type="number" min="1" max="10000" value="200" />
    <button id="hfCatalogBtn" class="smallbtn" onclick="fetchHFCatalog()" title="Fetch Hugging Face Hub catalog (Top N) and cache it">HF Catalog (Top N)</button>
  </div>

  <select id="model" class="setting-select" onchange="changeModel()">
    <option value="">Model: (offline)</option>
  </select>

  <input id="modelSearch" class="setting-select" type="text" placeholder="Model search…" oninput="onModelSearch()" />
  <div id="modelHint" class="hint" style="display:none; margin-top:4px;"></div>

  <div class="row" id="freeOnlyRow" style="display:none; margin-top:6px;">
    <label style="font-size:13px; user-select:none;">
      <input type="checkbox" id="freeOnly" onchange="toggleFreeOnly()" /> Nur kostenlose Modelle anzeigen (:free)
    </label>
  </div>

  <select id="anslang" class="setting-select" onchange="changeAnswerLanguage()">
    <option value="en">Answer language (LLM): English</option>
    <option value="de">Answer language (LLM): Deutsch</option>
  </select>

  <div class="hint">Panel runs fail-open: it stays usable even if the bridge is not ready yet.</div>
</div>

<div class="card" id="logLoader" style="margin-top:8px;">
  <h4>Logs</h4>
  <select id="chatlog" class="setting-select"></select>
  <div class="row" style="margin-top:6px;">
    <button class="smallbtn" onclick="refreshLogList()">Refresh list</button>
    <button class="smallbtn" onclick="loadSelectedLog(true)">Load &amp; fork</button>
  </div>
  <div class="row" style="margin-top:6px;">
    <button class="smallbtn" onclick="loadSelectedLog(false)">Load (no fork)</button>
    <button class="smallbtn" onclick="clearChat()">Clear</button>
  </div>
  <div class="hint" id="logHint" style="margin-top:6px; display:none;"></div>
</div>

<div class="card" id="manualTestCard" style="margin-top:8px;">
  <h4>Manual Test</h4>
  <select id="manualTestMonitorMode" class="setting-select" onchange="setManualTestMonitorMode()" title="Manual-Test-Monitor Fenster">
    <option value="show">Monitorfenster: anzeigen</option>
    <option value="hide">Monitorfenster: ausblenden</option>
  </select>
  <select id="manualTestScenario" class="setting-select">
    <option value="smoke_short">Kurztest (A+C+D+F ohne HF)</option>
    <option value="provider_switch">Providerwechsel (Gemini/OpenRouter/HF optional)</option>
    <option value="sci_format">SCI-Format (A/B)</option>
    <option value="qc_override_footer">QC-Override + Footer (SCI B, Gemini-Referenz)</option>
    <option value="komplexttest">Komplexttest (Matrix + Pflichtprompts + Influence-Checks)</option>
    <option value="full_regression_light">A-F (leicht, HF optional)</option>
  </select>
  <div class="row" style="margin-top:6px;">
    <button class="smallbtn" id="manualTestStartBtn" onclick="startManualTestRunner()">Start Test</button>
    <button class="smallbtn" id="manualTestStopBtn" onclick="stopManualTestRunner()">Stop</button>
  </div>
  <div class="hint" id="manualTestHint" style="margin-top:6px;">Fuehrt einen blockierenden GUI-Testablauf ueber `api.ask(...)` aus und prueft Basis-Formatregeln.</div>
  <div id="manualTestLog" class="test-log-box" style="display:none;"></div>
</div>

<div id="status" class="status-box">Panel boot…</div>
<div id="logs" class="log-box"></div>
<div id="ui"></div>

<script>
/* ---------- helpers ---------- */
function _api(){
  if (window.pywebview && window.pywebview.api) return window.pywebview.api;
  if (typeof pywebview !== "undefined" && pywebview.api) return pywebview.api;
  return null;
}

function _now(){
  try { return new Date().toISOString().replace('T',' ').replace('Z',''); } catch(e){ return ''; }
}

function _log(msg){
  try { console.log('[panel]', msg); } catch(e) {}
  const el = document.getElementById('logs');
  if(!el) return;
  const line = document.createElement('div');
  line.textContent = (_now() ? (_now() + ' · ') : '') + String(msg);
  el.appendChild(line);
  // keep last ~80 lines
  while(el.children.length > 80) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

function _setStatus(msg, cls){
  const el = document.getElementById('status');
  if(!el) return;
  el.textContent = String(msg || '');
  el.classList.remove('ok'); el.classList.remove('err');
  if(cls) el.classList.add(cls);
}

function _setSelectOptions(sel, options, selectedValue) {
  const el = document.getElementById(sel);
  if(!el) return;
  el.innerHTML = '';
  (options || []).forEach(o => {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    el.appendChild(opt);
  });
  if(selectedValue !== undefined && selectedValue !== null) {
    try { el.value = selectedValue; } catch(e) {}
  }
}

function _fallbackData(){
  return {
    providers: ['gemini','openrouter','huggingface'],
    current_provider: 'gemini',
    current_model: '',
    available_models: [],
    answer_language: 'de',
    comm: [],
    profiles: [],
    sci: [],
    overlays: [],
    tools: [],
    logs: [],
    chat_logs: [],
    chat_log_selected: ''
  };
}

async function _apiCall(name, args, timeoutMs) {
  timeoutMs = timeoutMs || 2000;
  const api = _api();
  if (!api) throw new Error('bridge missing (pywebview.api not injected)');
  if (typeof api[name] !== 'function') throw new Error('pywebview api method not available: ' + name);

  const p = api[name].apply(api, args || []);
  const t = new Promise((_, rej) => setTimeout(() => rej(new Error('timeout ' + timeoutMs + 'ms: ' + name)), timeoutMs));
  return await Promise.race([p, t]);
}

/* ---------- UI rendering ---------- */
function _storageKeyForModelQuery(provider){
  return `model_query_${provider||'unknown'}`;
}

function _safeLsGet(key, fallback){
  try {
    if(typeof window !== 'undefined' && window.localStorage){
      const v = window.localStorage.getItem(String(key || ''));
      return (v === null || v === undefined) ? fallback : v;
    }
  } catch(e) {}
  return fallback;
}

function _safeLsSet(key, value){
  try {
    if(typeof window !== 'undefined' && window.localStorage){
      window.localStorage.setItem(String(key || ''), String(value || ''));
      return true;
    }
  } catch(e) {}
  return false;
}

function buildUIFromData(raw){
  const base = _fallbackData();
  const data = Object.assign({}, base, (raw || {}));

  // Status/logs
  try {
    if(Array.isArray(data.logs) && data.logs.length){
      document.getElementById('logs').innerHTML = data.logs.map(x => String(x)).join('<br>');
    }
  } catch(e) {}

  // Provider select
  const providers = (data.providers || ['gemini']);
  _setSelectOptions('provider', providers.map(p => ({value:p, label:`Provider: ${p}`})), data.current_provider || 'gemini');

  const p = (data.current_provider || 'gemini');
  const btn = document.getElementById('refreshModelsBtn');
  if(btn) btn.style.display = 'block';

  // HF catalog controls
  const hfRow = document.getElementById('hfCatalogRow');
  if(hfRow) hfRow.style.display = (p === 'huggingface') ? 'flex' : 'none';
  if(p === 'huggingface'){
    const opts = (data.hf_provider_filter_options || ['all']);
    let savedPF = (data.hf_catalog_default_provider_filter || 'all');
    let savedTopN = String(_safeLsGet('hfTopN', (data.hf_catalog_default_top_n || 200)));
    const topInp = document.getElementById('hfTopN');
    savedPF = (_safeLsGet('hf_provider_filter', savedPF) || savedPF);
    savedTopN = (_safeLsGet('hf_catalog_topn', savedTopN) || savedTopN);
    _setSelectOptions('hfProviderFilter', opts.map(x => ({value:x, label:`HF Provider: ${x}`})), savedPF);
    try {
      if(topInp && !topInp.dataset.bound){
        topInp.addEventListener('input', ()=>{
          _safeLsSet('hfTopN', String(topInp.value||''));
        });
        topInp.dataset.bound = '1';
      }
    } catch(e) {}
    if(topInp) topInp.value = savedTopN;
  }

  // Models list (filter + search client-side)
  const allModels = data.available_models || [];
  window._allModels = allModels;

  const freeRow = document.getElementById('freeOnlyRow');
  const freeCb  = document.getElementById('freeOnly');
  const isOpenRouter = (p === 'openrouter');
  if(freeRow) freeRow.style.display = isOpenRouter ? 'block' : 'none';

  // Restore free-only (OpenRouter)
  let freeOnly = false;
  try {
    freeOnly = isOpenRouter && (_safeLsGet('openrouter_free_only', '0') === '1');
    if(freeCb) freeCb.checked = freeOnly;
  } catch(e) {}

  // Restore model search query per provider
  try {
    const key = _storageKeyForModelQuery(p);
    const savedQ = _safeLsGet(key, '') || '';
    const inp = document.getElementById('modelSearch');
    if(inp) inp.value = savedQ;
  } catch(e) {}

  // Apply filters and select model
  applyModelFilters(data.current_model || '');

  // Provider/model hint
  try {
    const hint = (data.model_hint || '').trim();
    const box = document.getElementById('modelHint');
    if(box){
      box.style.display = hint ? 'block' : 'none';
      box.textContent = hint;
    }
  } catch(e) {}

  // Answer language
  try { document.getElementById('anslang').value = (data.answer_language || 'de'); } catch(e) {}

  // Chat logs
  try {
    const logs = Array.isArray(data.chat_logs) ? data.chat_logs : [];
    _setSelectOptions('chatlog', logs.map(x => ({value:String(x), label:String(x)})), data.chat_log_selected || (logs[0] || ''));
    const hintEl = document.getElementById('logHint');
    if(hintEl){
      if(!logs.length){
        hintEl.style.display = 'block';
        hintEl.textContent = 'No chat logs found in Logs/Chats.';
      } else {
        hintEl.style.display = 'none';
        hintEl.textContent = '';
      }
    }
  } catch(e) {}

  // Comm-off static UI gating (dynamic sections are filtered server-side).
  try {
    const commOn = !(data && data.comm_active === false);
    const qcBtn = document.getElementById('qcOverrideBtn');
    if(qcBtn) qcBtn.style.display = commOn ? '' : 'none';
    const mtCard = document.getElementById('manualTestCard');
    if(mtCard) mtCard.style.display = (commOn && !(data && data.manual_test_visible === false)) ? 'block' : 'none';
  } catch(e) {}

  // Buttons sections
  let html = '';
  const section = (title, items) => {
    if(!items || !items.length) return;
    html += `<div class="card"><h4>${title}</h4><div class="grid">`;
    items.forEach(i => {
      let cmd, lbl, tip;
      if (typeof i === 'string') { cmd = i; lbl = i; tip = ""; }
      else { cmd = i.cmd ? i.cmd : i.name; lbl = i.name; tip = i.desc || ""; }
      html += `<button title="${tip}" onclick="run('${cmd}')">${lbl}</button>`;
    });
    html += '</div></div>';
  };

  section('Comm Core', data.comm);
  section('Profiles', data.profiles);
  section('SCI Workflow', data.sci);
  section('Modes & Overlays', data.overlays);
  section('Tools', data.tools);

  document.getElementById('ui').innerHTML = html;
}

async function buildUI(){
  let data = {};
  try {
    data = await _apiCall('get_ui', [], 2500) || {};
  } catch(e){
    _log('get_ui failed: ' + (e && e.message ? e.message : String(e)));
    data = {};
  }
  buildUIFromData(data);
}

/* ---------- fail-open bootstrap with retry ---------- */
let _retryTimer = null;
let _retryN = 0;
const _MAX_RETRIES = 40;
const _RETRY_MS = 250;

async function _tryBringOnline(){
  const api = _api();
  if(!api) throw new Error('bridge not ready');
  // Eindeutig: nicht nur Objekt existiert, sondern Call funktioniert:
  const pong = await _apiCall('ping', [], 800);
  if(!pong || pong.ok !== true) throw new Error('ping failed');
  _log('bridge ok (ping)');
  await buildUI();
  _setStatus('Panel ready (online)', 'ok');
  return true;
}

function _startRetryLoop(){
  if(_retryTimer) return;
  _retryTimer = setInterval(async () => {
    if(_retryN >= _MAX_RETRIES){
      clearInterval(_retryTimer); _retryTimer = null;
      _log('retry loop stopped (max retries)');
      return;
    }
    _retryN += 1;
    _log('retry #' + _retryN);
    try {
      await _tryBringOnline();
      clearInterval(_retryTimer); _retryTimer = null;
    } catch(e) {
      _setStatus('Panel ready (offline) · ' + (e && e.message ? e.message : String(e)), 'err');
    }
  }, _RETRY_MS);
}

function initPanelFailOpen(){
  _log('boot');
  // Always render immediately with fallback defaults
  buildUIFromData(_fallbackData());
  _setStatus('Panel ready (offline) · bridge not ready', 'err');

  // Immediate attempt, then retry loop
  (async () => {
    try {
      await _tryBringOnline();
      // success -> no retry loop needed
    } catch(e) {
      _setStatus('Panel ready (offline) · ' + (e && e.message ? e.message : String(e)), 'err');
      _startRetryLoop();
    }
  })();
}

document.addEventListener('DOMContentLoaded', initPanelFailOpen);
document.addEventListener('DOMContentLoaded', () => {
  try {
    const saved = _safeLsGet('manual_test_monitor_mode', 'show');
    const el = document.getElementById('manualTestMonitorMode');
    if(el) el.value = (saved === 'hide' ? 'hide' : 'show');
    setManualTestMonitorMode();
  } catch(e) {}
});
window.addEventListener('pywebviewready', () => {
  _log('pywebviewready');
  // If still offline, kick retry loop.
  if(!_retryTimer) _startRetryLoop();
});

// Visible, fail-loud error surfacing
window.addEventListener('error', function(ev){
  const msg = (ev && ev.message) ? ev.message : String(ev);
  _log('JS error: ' + msg);
  _setStatus('Panel JS error: ' + msg, 'err');
});
window.addEventListener('unhandledrejection', function(ev){
  const msg = (ev && ev.reason && ev.reason.message) ? ev.reason.message : String(ev && ev.reason ? ev.reason : ev);
  _log('Unhandled rejection: ' + msg);
  _setStatus('Panel JS error: ' + msg, 'err');
});

// Public hook (Python evaluate_js calls use this)
window.refresh_panel = buildUI;

/* ---------- actions ---------- */
async function run(c) {
  const cmd = (c || '').trim();
  if(!cmd) return;
  try {
    const r = await _apiCall('panel_action', ['cmd', {text: cmd}], 30000);
    if(r && r.ok === false) {
      const err = String(r.error || 'unknown error');
      if(err === 'qc_override_modal_blocked'){
        _setStatus('QC Override Dialog ist offen; Aktion temporär blockiert.', 'info');
      } else {
        _setStatus('cmd failed: ' + err, 'err');
      }
    }
    // Refresh panel UI after commands that change state-dependent labels (e.g., Comm Anchor on/off)
    try { await buildUI(); } catch(e) {}
    try { setTimeout(() => { try { buildUI(); } catch(e) {} }, 200); } catch(e) {}
    try { setTimeout(() => { try { buildUI(); } catch(e) {} }, 800); } catch(e) {}
  } catch(e) {
    _setStatus('cmd failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
}

async function changeProvider() {
  const provider = document.getElementById('provider').value;
  // Immediately align provider-specific controls to avoid stale UI states.
  try {
    const freeRow = document.getElementById('freeOnlyRow');
    const freeCb = document.getElementById('freeOnly');
    const isOpenRouter = (provider === 'openrouter');
    if(freeRow) freeRow.style.display = isOpenRouter ? 'block' : 'none';
    if(!isOpenRouter && freeCb) freeCb.checked = false;
    const hfRow = document.getElementById('hfCatalogRow');
    if(hfRow) hfRow.style.display = (provider === 'huggingface') ? 'flex' : 'none';
  } catch(e) {}
  try {
    await _apiCall('panel_action', ['set_provider', {provider: provider}], 8000);
    await _apiCall('panel_action', ['refresh_models', {provider: provider}], 15000);
  } catch(e) {
    _setStatus('set_provider failed: ' + (e && e.message ? e.message : String(e)), 'err');
    return;
  }
  await buildUI();
  try { setTimeout(() => { try { buildUI(); } catch(e) {} }, 250); } catch(e) {}
}

function changeModel() {
  const model = document.getElementById('model').value;
  const m = (model || '').trim();
  if(!m) return;
  (async () => {
    try {
      await _apiCall('panel_action', ['set_model', {model: m}], 15000);
    } catch(e) {
      _setStatus('set_model failed: ' + (e && e.message ? e.message : String(e)), 'err');
    }
  })();
}

function changeAnswerLanguage() {
  const lang = document.getElementById('anslang').value;
  const l = (lang || '').trim();
  if(!l) return;
  (async () => {
    try {
      await _apiCall('panel_action', ['set_answer_language', {lang: l}], 8000);
    } catch(e) {
      _setStatus('set_answer_language failed: ' + (e && e.message ? e.message : String(e)), 'err');
    }
  })();
}

async function refreshLogList(){
  try {
    const data = await _apiCall('panel_action', ['list_chat_logs', {limit: 200}], 1500);
    const logs = (data && Array.isArray(data.logs)) ? data.logs : [];
    _setSelectOptions('chatlog', logs.map(x => ({value:String(x), label:String(x)})), logs[0] || '');
    const hintEl = document.getElementById('logHint');
    if(hintEl){
      if(!logs.length){
        hintEl.style.display = 'block';
        hintEl.textContent = 'No chat logs found in Logs/Chats.';
      } else {
        hintEl.style.display = 'none';
        hintEl.textContent = '';
      }
    }
  } catch(e){
    _setStatus('list_chat_logs failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
}

async function loadSelectedLog(fork){
  const sel = document.getElementById('chatlog');
  const name = sel ? (sel.value || '') : '';
  if(!name){
    const hintEl = document.getElementById('logHint');
    if(hintEl){ hintEl.style.display = 'block'; hintEl.textContent = 'Select a log first.'; }
    return;
  }
  try {
    const res = await _apiCall('panel_action', ['load_chat_log', {name: name, fork: !!fork}], 8000);
    if(res && res.ok === true){
      const hintEl = document.getElementById('logHint');
      if(hintEl){ hintEl.style.display = 'block'; hintEl.textContent = `Loaded: ${name} (messages: ${res.history_len||'?'}${res.forked?' · forked':''})`; }
      _setStatus('Panel ready (online)', 'ok');
      // Inform main UI (optional) by issuing a refresh.
      try { await _apiCall('panel_action', ['cmd', {text: 'Comm State'}], 8000); } catch(e) {}
    } else {
      const err = res && res.error ? res.error : 'unknown error';
      _setStatus('load_chat_log failed: ' + err, 'err');
    }
  } catch(e){
    _setStatus('load_chat_log failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
}

function clearLogHint(){
  try {
    const hintEl = document.getElementById('logHint');
    if(hintEl){ hintEl.style.display = 'none'; hintEl.textContent = ''; }
  } catch(e) {}
}


function clearChat(){
  // Clear the main chat UI + in-memory history via backend action
  clearLogHint();
  try { _setStatus('Clearing chat…', 'info'); } catch(e) {}
  _apiCall('panel_action', ['clear_chat', {}], 8000).then((res) => {
    try {
      if(res && res.ok === true){
        const hintEl = document.getElementById('logHint');
        if(hintEl){
          hintEl.style.display = 'block';
          hintEl.textContent = 'Chat cleared.';
        }
        _setStatus('Panel ready (online)', 'ok');
      } else {
        const err = res && res.error ? res.error : 'unknown error';
        _setStatus('clear_chat failed: ' + err, 'err');
      }
    } catch(e){
      _setStatus('clear_chat failed: ' + (e && e.message ? e.message : String(e)), 'err');
    }
  }).catch((e) => {
    try {
      _setStatus('clear_chat failed: ' + (e && e.message ? e.message : String(e)), 'err');
    } catch(_e) {}
  });
}


function onModelSearch(){
  const p = (document.getElementById('provider') || {}).value || 'gemini';
  const q = (document.getElementById('modelSearch') || {}).value || '';
  _safeLsSet(_storageKeyForModelQuery(p), q);
  applyModelFilters();
}

function applyModelFilters(desiredModel){
  const p = (document.getElementById('provider') || {}).value || 'gemini';
  const allModels = window._allModels || [];
  let models = Array.isArray(allModels) ? allModels.slice() : [];

  // Free-only filter (OpenRouter)
  let freeOnly = false;
  const freeCb = document.getElementById('freeOnly');
  if(p === 'openrouter' && freeCb && freeCb.checked) freeOnly = true;
  if(freeOnly) models = models.filter(m => (m || '').includes(':free'));

  // Search filter
  const q = ((document.getElementById('modelSearch') || {}).value || '').trim().toLowerCase();
  if(q) models = models.filter(m => String(m || '').toLowerCase().includes(q));

  // Build options
  const modelOptions = (models || []).map(m => ({value:m, label:`Model: ${m}`}));
  if(!modelOptions.length){
    let label = 'Model: (offline / no models)';
    if(p === 'openrouter' && freeOnly) label = 'Model: (keine :free Modelle)';
    else if(q) label = 'Model: (keine Treffer)';
    modelOptions.push({value:'', label: label});
  }

  const current = (document.getElementById('model') || {}).value || '';
  const desired = (desiredModel || '').trim();
  const visible = new Set((models || []).map(x => String(x)));
  const selected = (desired && visible.has(String(desired))) ? desired
                  : (current && visible.has(String(current))) ? current
                  : (models[0] || '');

  _setSelectOptions('model', modelOptions, selected);

  // Only push to backend if online
  const api = _api();
  if(api && selected && selected !== current){
    (async () => { try { await _apiCall('panel_action', ['set_model', {model: selected}], 15000); } catch(e) {} })();
  }
}

function toggleFreeOnly() {
  const p = (document.getElementById('provider') || {}).value || '';
  const cb = document.getElementById('freeOnly');
  if(p !== 'openrouter') return;
  const v = cb && cb.checked;
  _safeLsSet('openrouter_free_only', (v ? '1' : '0'));
  applyModelFilters();
}

async function refreshModels() {
  const p = (document.getElementById('provider') || {}).value || 'gemini';
  try {
    await _apiCall('panel_action', ['refresh_models', {provider: p}], 15000);
  } catch(e) {
    _setStatus('refresh_models failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
  await buildUI();
}

/* ---------- HF catalog hooks (optional; backend may ignore) ---------- */
function onHFProviderFilterChange(){
  const v = (document.getElementById('hfProviderFilter') || {}).value || 'all';
  _safeLsSet('hf_provider_filter', v);
}

async function fetchHFCatalog(){
  const api = _api();
  if(!api){ _setStatus('offline: cannot fetch HF catalog', 'err'); return; }
  let topN = 200;
  let pf = 'all';
  try {
    topN = parseInt((document.getElementById('hfTopN') || {}).value || '200', 10);
    if(!isFinite(topN) || topN < 1) topN = 200;
    _safeLsSet('hfTopN', String(topN));
    pf = (document.getElementById('hfProviderFilter') || {}).value || 'all';
    _safeLsSet('hf_catalog_topn', String(topN));
  } catch(e) {}
  try {
    await _apiCall('panel_action', ['hf_catalog', {top_n: topN, provider_filter: pf, force_refresh: true}], 20000);
  } catch(e) {
    _setStatus('hf_catalog failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
  await buildUI();
}

/* ---------- Manual test runner (blocking on answers) ---------- */
window.__manualTestRunner = window.__manualTestRunner || { running:false, stop:false, runId:0, events:[], summary:null, scenario:'', askFallbackNoted:false };

function _mtMonitorEnabled(){
  try {
    const el = document.getElementById('manualTestMonitorMode');
    return !!(el && el.value === 'show');
  } catch(e) { return true; }
}

async function setManualTestMonitorMode(){
  try {
    const mode = _mtMonitorEnabled() ? 'show' : 'hide';
    _safeLsSet('manual_test_monitor_mode', mode);
    if(mode === 'show'){
      await _apiCall('panel_action', ['manual_test_monitor_show', {}], 8000);
    } else {
      await _apiCall('panel_action', ['manual_test_monitor_hide', {}], 8000);
    }
  } catch(e) {
    _mtWarn('Monitor-Umschaltung fehlgeschlagen: ' + String(e && e.message ? e.message : e));
  }
}

function _mtLog(msg, cls){
  const el = document.getElementById('manualTestLog');
  if(!el) return;
  let stamp = '';
  try { stamp = _now() || ''; } catch(e) { stamp = ''; }
  const line = document.createElement('div');
  if(cls) line.className = cls;
  line.textContent = (stamp ? (stamp + ' · ') : '') + String(msg || '');
  el.appendChild(line);
  while(el.children.length > 200) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
  try {
    const mt = window.__manualTestRunner;
    if(mt && mt.running){
      if(!Array.isArray(mt.events)) mt.events = [];
      const entry = {
        ts: stamp || null,
        level: cls || 'info',
        message: String(msg || ''),
      };
      mt.events.push(entry);
      try {
        if(mt.monitorEnabled){
          _apiCall('panel_action', ['manual_test_monitor_append', {entry: entry}], 4000).catch(()=>{});
        }
      } catch(e) {}
    }
  } catch(e) {}
}

function _mtClearLog(){
  const el = document.getElementById('manualTestLog');
  if(el) el.innerHTML = '';
}

async function _mtSaveReport(extra){
  try {
    const mt = window.__manualTestRunner || {};
    const report = {
      kind: 'manual_test_runner_report',
      version: 1,
      scenario: String(mt.scenario || ''),
      started_at: mt.startedAt || null,
      finished_at: (extra && extra.finished_at) || (_now ? _now() : null),
      duration_ms: (extra && extra.duration_ms) || null,
      summary: Object.assign({}, (mt.summary || {}), (extra && extra.summary ? extra.summary : {})),
      events: Array.isArray(mt.events) ? mt.events.slice() : [],
      ui_state: {
        provider: ((document.getElementById('provider') || {}).value || ''),
        model: ((document.getElementById('model') || {}).value || ''),
        answer_language: ((document.getElementById('anslang') || {}).value || ''),
      }
    };
    const res = await _apiCall('panel_action', ['save_manual_test_report', {report: report}], 20000);
    if(res && res.ok !== false){
      const path = (res.path || (res.result && res.result.path) || '');
      if(path) _mtLog('REPORT saved: ' + path, 'ok');
      else _mtLog('REPORT saved.', 'ok');
      return res;
    }
    _mtWarn('REPORT save failed: ' + String((res && res.error) || 'unknown'));
    return res;
  } catch(e) {
    _mtWarn('REPORT save failed: ' + String(e && e.message ? e.message : e));
    return null;
  }
}

async function _mtExportChatAudit(label){
  const suffix = String(label || '').trim();
  const tag = suffix ? (' [' + suffix + ']') : '';
  _mtLog('EXPORT > Chat/Audit' + tag);
  try {
    await _apiCall('export', [], 25000);
    _mtLog('PASS: Export ausgefuehrt', 'ok');
    return true;
  } catch(e) {
    const msg = String(e && e.message ? e.message : e);
    if(msg.toLowerCase().includes('pywebview api method not available: export')){
      try {
        const res = await _apiCall('panel_action', ['export', {}], 25000);
        if(!res || res.ok === false){
          throw new Error((res && res.error) ? res.error : 'panel_action export failed');
        }
        _mtLog('PASS: Export ausgefuehrt', 'ok');
        return true;
      } catch(e2) {
        _mtWarn('Export nicht ausgefuehrt: ' + String(e2 && e2.message ? e2.message : e2));
        return false;
      }
    }
    _mtWarn('Export nicht ausgefuehrt: ' + msg);
    return false;
  }
}

function _mtSetButtons(running){
  const a = document.getElementById('manualTestStartBtn');
  const b = document.getElementById('manualTestStopBtn');
  if(a) a.disabled = !!running;
  if(b) b.disabled = !running;
}

function _mtStripHtml(html){
  try{
    const d = document.createElement('div');
    d.innerHTML = String(html || '');
    return (d.textContent || d.innerText || '').replace(/\\u00a0/g, ' ');
  }catch(e){ return String(html || ''); }
}

function _mtHasCompleteQcFooter(html){
  const txt = _mtStripHtml(html);
  const idxMatrix = txt.lastIndexOf('QC-Matrix:');
  const idxQC = txt.lastIndexOf('QC:');
  const idx = Math.max(idxMatrix, idxQC);
  if(idx < 0) return false;
  let qc = txt.slice(idx);
  const tsIdx = qc.search(/\\bResponse at\\b/i);
  if(tsIdx >= 0) qc = qc.slice(0, tsIdx);
  qc = qc.replace(/\\s+/g, ' ').trim();
  const en = ['Clarity','Brevity','Evidence','Empathy','Consistency','Neutrality'];
  const de = ['Klarheit','Evidenz','Empathie','Konsistenz'];
  const hasEN = en.every(k => qc.includes(k + ' '));
  const hasDE = de.every(k => qc.includes(k + ' ')) &&
                (qc.includes('Kürze ') || qc.includes('Kuerze ')) &&
                (qc.includes('Neutralität ') || qc.includes('Neutralitaet '));
  return !!(hasEN || hasDE);
}

function _mtHasSelfDebunkingBox(html){
  const h = String(html || '').toLowerCase();
  const hasDebunkLabel = h.includes('self-debunking') || h.includes('selbst-debunking');
  const hasDebunkClass = h.includes('class=\"self-debunk') || h.includes("class='self-debunk");
  const hasBoxStyle = (h.includes('background') || h.includes('background-color')) &&
                      (h.includes('border-left') || h.includes('border-radius'));
  return hasDebunkLabel && (hasDebunkClass || hasBoxStyle);
}

function _mtHasVerificationRouteMarkers(html){
  const txt = _mtStripHtml(html || '');
  return /\\b(?:Verification Route|Verification Route Gate|Source|Measurement|Contrast|Web[- ]Check|Retrieval-Check|Quelle|Messung|Kontrast)\\s*:/i.test(txt);
}

function _mtHasSciTraceStructure(html){
  const txt = _mtStripHtml(html);
  if(!/SCI Trace/i.test(txt)) return false;
  // Accept numbered or bullet-style step labels (models/renderers vary).
  const hasNumbered = /\\b1\\.\\s*(Plan|Check|Solution|Critic|Linguist|Logician|Adversary|Dialectic)/i.test(txt);
  const hasBulletLabels = /(?:^|\\n)\\s*(?:[-•*]\\s*)?(Plan|Check|Solution|Critic|Linguist|Logician|Adversary|Dialectic(?:_[0-9]+_[A-Za-z]+)?)\\s*:/im.test(txt);
  return !!(hasNumbered || hasBulletLabels);
}

function _mtIncludesProviderLimitLabel(htmlOrText, providerLabel){
  const t = _mtStripHtml(htmlOrText);
  return t.toLowerCase().includes(String(providerLabel || '').toLowerCase()) &&
         /(limit|guthaben|credits|rate)/i.test(t);
}

function _mtEnsureRunning(){
  if(!window.__manualTestRunner || !window.__manualTestRunner.running) throw new Error('manual_test_not_running');
  if(window.__manualTestRunner.stop) throw new Error('manual_test_stopped');
}

async function _mtSleep(ms){
  await new Promise(r => setTimeout(r, ms||0));
}

function _mtAskTimeoutForCurrentProvider(timeoutMs){
  const base = Number(timeoutMs || 180000) || 180000;
  // In manual GUI regression runs, even command-like asks (profile/SCI menu selections)
  // may trigger provider roundtrips and exceed 60s on OpenRouter/HF. Use a robust floor.
  if(base < 180000){
    return Math.max(base, 180000);
  }
  return base;
}

async function _mtAsk(text, timeoutMs){
  _mtEnsureRunning();
  const t = String(text || '').trim();
  if(!t) return {html:'', csc:null};
  const askTimeout = _mtAskTimeoutForCurrentProvider(timeoutMs);
  _mtLog('ASK > ' + t);
  let res = null;
  try {
    res = await _apiCall('ask', [t], askTimeout);
  } catch(e) {
    const msg = String(e && e.message ? e.message : e);
    if(msg.toLowerCase().includes('pywebview api method not available: ask')){
      try {
        const mt = window.__manualTestRunner || {};
        if(!mt.askFallbackNoted){
          mt.askFallbackNoted = true;
          _mtLog('INFO: api.ask nicht verfuegbar -> Fallback auf panel_action(ask)', 'info');
        }
      } catch(_e) {}
      res = await _apiCall('panel_action', ['ask', {text: t}], askTimeout);
      if(res && res.result && typeof res.result === 'object'){
        res = res.result;
      }
    } else {
      throw e;
    }
  }
  _mtEnsureRunning();
  const html = (res && res.html) ? String(res.html) : '';
  _mtLog('ASK < done (' + (html.length||0) + ' html chars)');
  await _mtSleep(150);
  try { await buildUI(); } catch(e) {}
  return res || {html:'', csc:null};
}

async function _mtPanelAction(action, payload, timeoutMs){
  _mtEnsureRunning();
  const res = await _apiCall('panel_action', [action, payload || {}], timeoutMs || 120000);
  _mtEnsureRunning();
  if(!res || res.ok === false){
    throw new Error((res && res.error) ? res.error : ('panel_action failed: ' + action));
  }
  return res;
}

async function _mtSetProvider(provider){
  const p = String(provider || '').trim().toLowerCase();
  _mtLog('SET provider = ' + p);
  // OpenRouter/Hugging Face can be slower due provider API latency / remote catalogs.
  const _setProviderTimeout = (p === 'openrouter' || p === 'huggingface') ? 45000 : 30000;
  const _refreshTimeout = (p === 'openrouter' || p === 'huggingface') ? 120000 : 60000;
  await _mtPanelAction('set_provider', {provider:p}, _setProviderTimeout);
  await _mtPanelAction('refresh_models', {provider:p}, _refreshTimeout);
  await _mtSleep(250);
  await buildUI();
}

async function _mtSetAnswerLanguage(lang){
  const l = String(lang || 'de').trim().toLowerCase();
  _mtLog('SET answer_language = ' + l);
  await _mtPanelAction('set_answer_language', {lang:l}, 10000);
  await _mtSleep(100);
}

async function _mtApplyQcOverride(values){
  _mtEnsureRunning();
  _mtLog('QC Override apply: ' + JSON.stringify(values || {}));
  const api = _api();
  if(api && typeof api.qc_override_apply === 'function'){
    const res = await api.qc_override_apply(values || {});
    if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_apply failed');
    await _mtSleep(120);
    return;
  }
  _mtWarn('qc_override_apply nicht direkt verfuegbar -> Fallback auf panel_action');
  const res = await _apiCall('panel_action', ['qc_override_apply', {values: values || {}}], 10000);
  if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_apply failed');
  await _mtSleep(120);
}

async function _mtClearQcOverride(){
  _mtEnsureRunning();
  _mtLog('QC Override clear');
  const api = _api();
  if(api && typeof api.qc_override_clear === 'function'){
    const res = await api.qc_override_clear({});
    if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_clear failed');
    await _mtSleep(120);
    return;
  }
  _mtWarn('qc_override_clear nicht direkt verfuegbar -> Fallback auf panel_action');
  const res = await _apiCall('panel_action', ['qc_override_clear', {}], 10000);
  if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_clear failed');
  await _mtSleep(120);
}

function _mtCheck(cond, okMsg, failMsg){
  if(cond){
    _mtLog('PASS: ' + okMsg, 'ok');
    return true;
  }
  _mtLog('FAIL: ' + failMsg, 'err');
  return false;
}

function _mtWarn(msg){
  _mtLog('WARN: ' + msg, 'warn');
}

function _mtParseErrorTextFromAskResult(res){
  try{
    const txt = _mtStripHtml((res && res.html) || '');
    if(/(error|limit|guthaben|credits|rate)/i.test(txt)) return txt;
  }catch(e){}
  return '';
}

async function _mtScenarioSmokeShort(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetAnswerLanguage('de');
  await _mtAsk('Standard', 30000);
  await _mtAsk('SCI off', 30000);
  const r1 = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(_mtHasCompleteQcFooter(r1.html), 'QC-Footer bei Basisantwort vollstaendig', 'QC-Footer bei Basisantwort fehlt/ist unvollstaendig')) fails++;
  if(!_mtCheck(_mtStripHtml(r1.html).includes('Self-Debunking') || _mtStripHtml(r1.html).includes('Selbst-Debunking'),
      'Self-Debunking vorhanden', 'Self-Debunking fehlt')) fails++;
  return {fails};
}

async function _mtScenarioProviderSwitch(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetAnswerLanguage('de');
  await _mtSetProvider('gemini');
  await _mtAsk('Standard', 30000);
  let rg = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(_mtHasCompleteQcFooter(rg.html), 'Gemini: QC-Footer ok', 'Gemini: QC-Footer fehlt/unvollstaendig')) fails++;

  await _mtSetProvider('openrouter');
  let ro = await _mtAsk('Was ist Zeit?', 180000);
  const txtOR = _mtStripHtml(ro.html);
  if(/(insufficient credits|Nicht genügend Guthaben|rate limit|Limit erreicht)/i.test(txtOR)){
    _mtWarn('OpenRouter lieferte Limit/Credit-Fehler (Test trotzdem ok, Providerpfad verifiziert).');
    if(!_mtCheck(_mtIncludesProviderLimitLabel(ro.html, 'OpenRouter'), 'OpenRouter-Fehlerlabel korrekt', 'OpenRouter-Fehlerlabel unklar/falsch')) fails++;
  } else {
    if(!_mtCheck(_mtHasCompleteQcFooter(ro.html), 'OpenRouter: QC-Footer ok', 'OpenRouter: QC-Footer fehlt/unvollstaendig')) fails++;
    if(!_mtCheck(_mtHasSelfDebunkingBox(ro.html), 'OpenRouter: Self-Debunking-Box erkannt', 'OpenRouter: Self-Debunking nicht als Box erkannt')) fails++;
  }

  await _mtSetProvider('huggingface');
  let rhf = await _mtAsk('Was ist Zeit?', 180000);
  const txtHF = _mtStripHtml(rhf.html);
  if(/(insufficient credits|Nicht genügend Guthaben|rate limit|Limit erreicht|quota)/i.test(txtHF)){
    _mtWarn('Hugging Face lieferte Limit/Credit-Fehler (optional).');
    if(!_mtCheck(_mtIncludesProviderLimitLabel(rhf.html, 'Hugging Face'), 'HF-Fehlerlabel korrekt', 'HF-Fehler wurde nicht als Hugging Face gekennzeichnet')) fails++;
  } else {
    if(!_mtCheck(_mtHasCompleteQcFooter(rhf.html), 'HF: QC-Footer ok', 'HF: QC-Footer fehlt/unvollstaendig')) fails++;
  }
  return {fails};
}

async function _mtScenarioSciFormat(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtAsk('Expert', 30000);
  await _mtAsk('SCI menu', 30000);
  await _mtAsk('A', 30000);
  const rA = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(_mtHasSciTraceStructure(rA.html), 'SCI A: SCI-Trace-Struktur erkannt', 'SCI A: SCI-Trace-Struktur fehlt/unsauber')) fails++;
  if(!_mtCheck(_mtHasCompleteQcFooter(rA.html), 'SCI A: QC-Footer ok', 'SCI A: QC-Footer fehlt/unvollstaendig')) fails++;

  await _mtAsk('SCI menu', 30000);
  await _mtAsk('B', 30000);
  const rB = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(_mtHasSciTraceStructure(rB.html), 'SCI B: SCI-Trace-Struktur erkannt', 'SCI B: SCI-Trace-Struktur fehlt/unsauber')) fails++;
  if(!_mtCheck(_mtHasCompleteQcFooter(rB.html), 'SCI B: QC-Footer ok', 'SCI B: QC-Footer fehlt/unvollstaendig')) fails++;
  return {fails};
}

async function _mtScenarioQcOverrideFooter(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  // Deterministic reference provider for QC/SCI contract checks (credits/limits on
  // optional providers must not turn feature-contract tests into false FAILs).
  await _mtSetProvider('gemini');
  await _mtAsk('Expert', 30000);
  await _mtAsk('SCI menu', 30000);
  await _mtAsk('B', 30000);
  await _mtApplyQcOverride({Clarity:3,Brevity:0,Evidence:3,Empathy:3,Consistency:3,Neutrality:3});
  const r = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(_mtHasCompleteQcFooter(r.html), 'SCI B + QC-Override: QC-Footer vollstaendig', 'SCI B + QC-Override: QC-Footer fehlt/unvollstaendig')) fails++;
  if(!_mtCheck(_mtHasSelfDebunkingBox(r.html), 'SCI B + QC-Override: Self-Debunking-Box erkannt', 'SCI B + QC-Override: Self-Debunking nicht als Box erkannt')) fails++;
  try { await _mtClearQcOverride(); } catch(e) { _mtWarn('QC-Override clear fehlgeschlagen: ' + String(e && e.message ? e.message : e)); }
  return {fails};
}

async function _mtScenarioFullRegressionLight(){
  let fails = 0;
  for(const fn of [_mtScenarioSmokeShort, _mtScenarioProviderSwitch, _mtScenarioSciFormat, _mtScenarioQcOverrideFooter]){
    _mtEnsureRunning();
    const r = await fn();
    fails += (r && r.fails) ? r.fails : 0;
  }
  await _mtExportChatAudit('full_regression_light');
  return {fails};
}

async function _mtScenarioKomplexttest(){
  let fails = 0;
  const prompts = [
    'Was ist Zeit?',
    'Was ist die objektiv beste und dauerhaft faire Strategie, um ab heute weltweit ein einheitliches KI-Regelwerk verbindlich durchzusetzen, sodass alle LLMs in jeder Sprache, Kultur und Rechtsordnung identische Antworten liefern, ohne negative Folgen fuer Datenschutz, Demokratie, Kreativitaet, Wissenschaft und Arbeitsmarkt?',
  ];
  const profiles = ['Standard', 'Expert'];
  const sciVariants = ['off', 'A', 'B'];
  const qcStates = [false, true];
  const colorStates = ['on', 'off'];
  const totalCases = profiles.length * sciVariants.length * qcStates.length * colorStates.length;
  let caseIdx = 0;

  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage('de');

  for(const profile of profiles){
    for(const sci of sciVariants){
      for(const qcOn of qcStates){
        for(const color of colorStates){
          _mtEnsureRunning();
          caseIdx += 1;
          _mtLog(
            'CASE ' + caseIdx + '/' + totalCases
            + ' > profile=' + profile
            + ' · sci=' + sci
            + ' · qc_override=' + (qcOn ? 'on' : 'off')
            + ' · color=' + color
          );

          if(caseIdx > 1){
            await _mtExportChatAudit('case_checkpoint_before_clear_chat_' + caseIdx);
          }
          await _mtPanelAction('clear_chat', {}, 8000);
          await _mtAsk(profile, 30000);
          if(color === 'on') await _mtAsk('Color on', 30000);
          else await _mtAsk('Color off', 30000);

          if(sci === 'off'){
            await _mtAsk('SCI off', 30000);
          } else {
            await _mtAsk('SCI menu', 30000);
            await _mtAsk(sci, 30000);
          }

          if(qcOn){
            await _mtApplyQcOverride({Clarity:3, Brevity:0, Evidence:3, Empathy:3, Consistency:3, Neutrality:3});
          } else {
            try { await _mtClearQcOverride(); } catch(e) {}
          }

          for(let i = 0; i < prompts.length; i += 1){
            _mtEnsureRunning();
            const prompt = prompts[i];
            const res = await _mtAsk(prompt, 180000);
            const txt = _mtStripHtml(res.html || '');
            const caseLabel = 'CASE ' + caseIdx + ' P' + String(i + 1);

            if(!_mtCheck(
              _mtHasCompleteQcFooter(res.html),
              caseLabel + ': QC-Footer vollstaendig',
              caseLabel + ': QC-Footer fehlt/unvollstaendig'
            )) fails++;

            if(!_mtCheck(
              /\bU[1-6]\b/.test(txt),
              caseLabel + ': U-Marker vorhanden',
              caseLabel + ': kein U-Marker gefunden'
            )) fails++;

            if(!_mtCheck(
              _mtHasSelfDebunkingBox(res.html) || /(Self-Debunking|Selbst-Debunking)/i.test(txt),
              caseLabel + ': Self-Debunking vorhanden',
              caseLabel + ': Self-Debunking fehlt'
            )) fails++;

            const hasTagMarker = /\\[(GREEN|YELLOW|RED|GRAY|WHITE)(-[A-Z0-9]+)*\\]/i.test(txt);
            const hasEmojiMarker = txt.indexOf('🟢') >= 0 || txt.indexOf('🟡') >= 0 || txt.indexOf('🔴') >= 0 || txt.indexOf('⚪') >= 0;
            const hasColorMarker = hasTagMarker || hasEmojiMarker;
            if(color === 'on'){
              if(!_mtCheck(
                hasColorMarker,
                caseLabel + ': Farbmarker vorhanden (Color on)',
                caseLabel + ': Farbmarker fehlen (Color on)'
              )) fails++;
            } else if(hasColorMarker){
              _mtWarn(caseLabel + ': Farbmarker trotz Color off erkannt (modell-/renderpfadabhaengig).');
            }
          }

          if(qcOn){
            try { await _mtClearQcOverride(); } catch(e) { _mtWarn('QC-Override clear fehlgeschlagen: ' + String(e && e.message ? e.message : e)); }
          }
        }
      }
    }
  }

  _mtEnsureRunning();
  await _mtExportChatAudit('before_influence_checks');
  _mtLog('INFLUENCE-CHECKS > CGI / QC-Override / Dynamic one-shot');
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtAsk('Standard', 30000);
  await _mtAsk('SCI off', 30000);
  await _mtAsk('Color on', 30000);

  const baseline = await _mtAsk('Was ist Zeit?', 180000);
  const baselineTxt = _mtStripHtml(baseline.html || '');

  const cgiNote = await _mtAsk('3,3,3', 30000);
  if(!_mtCheck(
    /cgi/i.test(_mtStripHtml(cgiNote.html || '')),
    'CGI-Feedback wurde vom Wrapper registriert',
    'CGI-Feedback wurde nicht registriert'
  )) fails++;
  const afterCgi = await _mtAsk('Was ist Zeit?', 180000);
  const afterCgiTxt = _mtStripHtml(afterCgi.html || '');
  if(!_mtCheck(
    baselineTxt !== afterCgiTxt,
    'CGI-Feedback beeinflusst Folgeresponse',
    'CGI-Feedback beeinflusst Folgeresponse nicht sichtbar'
  )) fails++;

  const baselineQc = await _mtAsk('Was ist Zeit?', 180000);
  await _mtApplyQcOverride({Clarity:3, Brevity:0, Evidence:3, Empathy:3, Consistency:3, Neutrality:3});
  const afterQc = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(
    _mtStripHtml(baselineQc.html || '') !== _mtStripHtml(afterQc.html || ''),
    'QC-Override beeinflusst Folgeresponse',
    'QC-Override beeinflusst Folgeresponse nicht sichtbar'
  )) fails++;
  try { await _mtClearQcOverride(); } catch(e) { _mtWarn('QC-Override clear fehlgeschlagen: ' + String(e && e.message ? e.message : e)); }

  const baselineDyn = await _mtAsk('Was ist Zeit?', 180000);
  await _mtAsk('Dynamic one-shot on', 30000);
  const afterDyn = await _mtAsk('Was ist Zeit?', 180000);
  if(!_mtCheck(
    _mtStripHtml(baselineDyn.html || '') !== _mtStripHtml(afterDyn.html || ''),
    'Dynamic one-shot beeinflusst Folgeresponse',
    'Dynamic one-shot beeinflusst Folgeresponse nicht sichtbar'
  )) fails++;

  _mtLog('VERIFICATION-ROUTE VISIBILITY CHECK > explicit marker prompt');
  const vrProbe = await _mtAsk(
    'Gib eine kurze Antwort mit einem expliziten Verification Route Block. Verwende genau diese Marker: Source:, Measurement:, Contrast:, Web-Check:.',
    180000
  );
  if(!_mtCheck(
    _mtHasVerificationRouteMarkers(vrProbe.html),
    'Verification-Route Marker sichtbar',
    'Verification-Route Marker fehlen oder werden gefiltert'
  )) fails++;

  await _mtExportChatAudit('komplexttest_final');
  return {fails};
}

function stopManualTestRunner(){
  if(!window.__manualTestRunner) return;
  window.__manualTestRunner.stop = true;
  _mtLog('Stop angefordert.');
}

async function startManualTestRunner(){
  const mt = window.__manualTestRunner || (window.__manualTestRunner = { running:false, stop:false, runId:0 });
  if(mt.running){
    _mtWarn('Ein Testlauf laeuft bereits.');
    return;
  }
  mt.running = true;
  mt.stop = false;
  mt.runId = (mt.runId || 0) + 1;
  mt.events = [];
  mt.summary = null;
  mt.monitorEnabled = _mtMonitorEnabled();
  mt.askFallbackNoted = false;
  mt.monitorLang = (String(((document.getElementById('anslang') || {}).value || 'de')).trim().toLowerCase() === 'en') ? 'en' : 'de';
  const myRun = mt.runId;
  _mtSetButtons(true);
  _mtClearLog();
  const sel = document.getElementById('manualTestScenario');
  const scenario = (sel && sel.value) ? sel.value : 'smoke_short';
  mt.scenario = scenario;
  mt.startedAt = (_now ? _now() : null);
  try {
    if(mt.monitorEnabled){
      await _apiCall('panel_action', ['manual_test_monitor_show', {}], 8000);
      await _apiCall('panel_action', ['manual_test_monitor_reset', {
        scenario: scenario,
        status: 'running',
        summary: '-',
        lang: mt.monitorLang,
      }], 8000);
    }
  } catch(e) {
    _mtWarn('Manual-Test-Monitor konnte nicht initialisiert werden: ' + String(e && e.message ? e.message : e));
  }
  _mtLog('Manual-Test gestartet: ' + scenario);
  const t0 = Date.now();
  try {
    let result = {fails: 0};
    if(scenario === 'smoke_short') result = await _mtScenarioSmokeShort();
    else if(scenario === 'provider_switch') result = await _mtScenarioProviderSwitch();
    else if(scenario === 'sci_format') result = await _mtScenarioSciFormat();
    else if(scenario === 'qc_override_footer') result = await _mtScenarioQcOverrideFooter();
    else if(scenario === 'komplexttest') result = await _mtScenarioKomplexttest();
    else if(scenario === 'full_regression_light') result = await _mtScenarioFullRegressionLight();
    else throw new Error('unknown scenario: ' + scenario);
    _mtEnsureRunning();
    const ms = Date.now() - t0;
    if((result && result.fails) > 0){
      mt.summary = { status: 'FAIL', fails: Number(result.fails||0) };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'FAIL', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog('SUMMARY: FAILS=' + result.fails + ' · Dauer=' + ms + 'ms', 'err');
      _setStatus('Manual Test fertig: ' + result.fails + ' Fehler', 'err');
    } else {
      mt.summary = { status: 'PASS', fails: 0 };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'PASS', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog('SUMMARY: PASS · Dauer=' + ms + 'ms', 'ok');
      _setStatus('Manual Test fertig: PASS', 'ok');
    }
    await _mtSaveReport({ duration_ms: ms, summary: mt.summary, finished_at: (_now ? _now() : null) });
  } catch(e) {
    const stopped = (String(e && e.message ? e.message : e).indexOf('manual_test_stopped') >= 0);
    if(stopped){
      mt.summary = { status: 'STOPPED' };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'STOPPED', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog('SUMMARY: STOPPED', 'warn');
      _setStatus('Manual Test gestoppt', 'err');
      await _mtExportChatAudit('manual_test_stopped_partial');
      await _mtSaveReport({ duration_ms: (Date.now()-t0), summary: mt.summary, finished_at: (_now ? _now() : null) });
    } else {
      mt.summary = { status: 'ERROR', error: String(e && e.message ? e.message : e) };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'ERROR', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog('ERROR: ' + String(e && e.message ? e.message : e), 'err');
      _setStatus('Manual Test error: ' + String(e && e.message ? e.message : e), 'err');
      await _mtExportChatAudit('manual_test_error_partial');
      await _mtSaveReport({ duration_ms: (Date.now()-t0), summary: mt.summary, finished_at: (_now ? _now() : null) });
    }
  } finally {
    if(window.__manualTestRunner && window.__manualTestRunner.runId === myRun){
      window.__manualTestRunner.running = false;
      window.__manualTestRunner.stop = false;
    }
    _mtSetButtons(false);
    try { await buildUI(); } catch(e) {}
  }
}
</script>
</body>
</html>
"""
# Inject dynamic wrapper name into embedded HTML templates (no f-strings → avoids brace issues)
try:
    HTML_CHAT = (HTML_CHAT or '').replace('Wrapper-115', WRAPPER_NAME)
    HTML_PANEL = (HTML_PANEL or '').replace('Wrapper-115', WRAPPER_NAME)
except Exception:
    pass



HTML_QC_OVERRIDE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>QC Override</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 16px; color: #111; }
  h2 { margin: 0 0 10px 0; font-size: 18px; }
  .sub { margin: 0 0 14px 0; color: #444; font-size: 13px; }
  .row { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
  .lbl { width: 120px; font-weight: 600; font-size: 13px; }
  input[type=range] { flex: 1; }
  .val { width: 26px; text-align: right; font-variant-numeric: tabular-nums; }
  .status { margin-top: 10px; font-size: 12px; color: #444; }
  .err { color: #b00020; white-space: pre-wrap; font-size: 12px; }
  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 10px 0; }
  .presets button { padding: 6px 10px; font-size: 12px; }
  .actions { display: flex; gap: 10px; margin-top: 16px; }
  .actions button { flex: 1; padding: 10px 12px; font-size: 13px; }
</style>
<script>
(function(){
  function qs(sel){ return document.querySelector(sel); }
  function setStatus(txt, isErr){
    const el = qs('#status');
    if(!el) return;
    el.className = isErr ? 'err' : 'status';
    el.textContent = txt || '';
  }
  window.addEventListener('error', function(ev){
    try{ setStatus('JS Error: ' + (ev && ev.message ? ev.message : String(ev)), true); }catch(e){}
  });

  const DIMENSIONS = [
    ['Clarity','clarity'],
    ['Brevity','brevity'],
    ['Evidence','evidence'],
    ['Empathy','empathy'],
    ['Consistency','consistency'],
    ['Neutrality','neutrality']
  ];
  let _attached = false;

  function readValues(){
    const out = {};
    for(const [label, key] of DIMENSIONS){
      const inp = qs('#sl-' + key);
      if(inp) out[label] = parseInt(inp.value, 10);
    }
    return out;
  }

  function setValues(vals){
    for(const [label, key] of DIMENSIONS){
      const v = vals && (vals[label] !== undefined ? vals[label] : vals[key]);
      if(v === undefined || v === null) continue;
      const inp = qs('#sl-' + key);
      const sp = qs('#v-' + key);
      if(inp){ inp.value = String(v); }
      if(sp){ sp.textContent = String(v); }
    }
  }

  function attach(){
    for(const [label, key] of DIMENSIONS){
      const inp = qs('#sl-' + key);
      const sp = qs('#v-' + key);
      if(inp && sp){
        inp.addEventListener('input', function(){ sp.textContent = String(inp.value); });
      }
    }
  }

  function applyState(st){
    if(!(st && st.ok)){
      setStatus('Online, aber qc_get_state fehlgeschlagen.', true);
      return false;
    }
    const prof = st.profile || 'Standard';
    document.title = 'Temporary QC override – Profile: ' + prof;
    const h = qs('#title');
    if(h) h.textContent = 'Temporary QC override – Profile: ' + prof;

    const defaults = st.defaults || {};
    const ovs = st.overrides || {};
    const base = {};
    for(const [label, key] of DIMENSIONS){
      const d = defaults[key];
      let v = 2;
      if(Array.isArray(d) && d.length>=2){
        v = parseInt(d[1],10);
      }else if(typeof d === 'number'){
        v = d;
      }
      base[label] = v;
    }
    setValues(base);
    setValues(ovs);
    setStatus(st.note || 'Online.', false);
    return true;
  }

  async function callApi(fn, payload){
    if(!window.pywebview || !pywebview.api || !pywebview.api[fn]){
      throw new Error('Bridge not ready: ' + fn);
    }
    return await pywebview.api[fn](payload || {});
  }

  async function refreshState(){
    const st = await callApi('qc_get_state', {});
    return applyState(st);
  }

  async function boot(){
    setStatus('Offline (bridge not ready). Trying…', false);
    if(!_attached){
      attach();
      _attached = true;
    }

    for(let i=0;i<40;i++){
      try{
        if(window.pywebview && pywebview.api && pywebview.api.ping){
          const pong = await pywebview.api.ping();
          if(pong && pong.ok){
            await refreshState();
            return;
          }
        }
      }catch(e){}
      await new Promise(r => setTimeout(r, 250));
    }
    setStatus('Bridge not ready (offline). You can still adjust sliders, then Apply will retry.', false);
  }

  async function onApply(){
    try{
      const vals = readValues();
      const res = await callApi('qc_override_apply', vals);
      if(res && res.ok){
        setStatus('Applied.', false);
      }else{
        setStatus('Apply failed: ' + (res && res.error ? res.error : 'unknown'), true);
      }
    }catch(e){
      setStatus('Apply error: ' + String(e), true);
    }
  }

  async function onClear(){
    try{
      const res = await callApi('qc_override_clear', {});
      if(res && res.ok){
        try{
          await refreshState();
        }catch(e){}
        setStatus('Cleared.', false);
      }else{
        setStatus('Clear failed: ' + (res && res.error ? res.error : 'unknown'), true);
      }
    }catch(e){
      setStatus('Clear error: ' + String(e), true);
    }
  }

  async function onCancel(){
    try{
      await callApi('qc_override_cancel', {});
    }catch(e){
      try{ window.close(); }catch(_){}
    }
  }

  function preset(kind){
    const vals = readValues();
    if(kind==='verbose'){ vals['Brevity']=0; vals['Clarity']=3; }
    if(kind==='short'){ vals['Brevity']=3; vals['Clarity']=2; }
    if(kind==='evidence'){ vals['Evidence']=3; vals['Clarity']=3; }
    if(kind==='neutral'){ vals['Neutrality']=3; vals['Empathy']=1; }
    setValues(vals);
  }

  window.QCUI = { boot, refreshState, onApply, onClear, onCancel, preset };
})();
</script>
</head>
<body onload="QCUI.boot()">
  <h2 id="title">Temporary QC override – Profile: ?</h2>
  <p class="sub">Temporary QC adjustment (active until profile switch / Clear)</p>

  <div class="row"><div class="lbl">Clarity</div><input id="sl-clarity" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-clarity">2</div></div>
  <div class="row"><div class="lbl">Brevity</div><input id="sl-brevity" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-brevity">2</div></div>
  <div class="row"><div class="lbl">Evidence</div><input id="sl-evidence" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-evidence">2</div></div>
  <div class="row"><div class="lbl">Empathy</div><input id="sl-empathy" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-empathy">2</div></div>
  <div class="row"><div class="lbl">Consistency</div><input id="sl-consistency" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-consistency">2</div></div>
  <div class="row"><div class="lbl">Neutrality</div><input id="sl-neutrality" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-neutrality">2</div></div>

  <div class="presets">
    <button onclick="QCUI.preset('verbose')">More verbose</button>
    <button onclick="QCUI.preset('short')">Shorter</button>
    <button onclick="QCUI.preset('evidence')">More evidence</button>
    <button onclick="QCUI.preset('neutral')">More neutral</button>
  </div>

  <div class="actions">
    <button onclick="QCUI.onApply()">Apply</button>
    <button onclick="QCUI.onClear()">Clear Overrides</button>
    <button onclick="QCUI.onCancel()">Cancel</button>
  </div>

  <div id="status" class="status"></div>
</body>
</html>
"""

# S7 (UI assets): prefer externalized templates, keep embedded strings as deterministic fallback.
# panel.html is loaded from the external asset when its static self-test passes; the first
# visible show still requires a runtime self-test callback, otherwise the app falls back to
# the embedded panel before showing it.
HTML_CHAT_TEMPLATE = _load_ui_asset_text("chat_template.html", HTML_CHAT_TEMPLATE)
HTML_CHAT = HTML_CHAT_TEMPLATE.replace('__WRAPPER_LABEL__', html.escape(WRAPPER_NAME))
HTML_PANEL_EMBEDDED = HTML_PANEL
HTML_PANEL, PANEL_HTML_ASSET_META = _load_panel_asset_text_s7(HTML_PANEL)
try:
    # Keep fallback panel content in sync with the externally loaded panel asset when available.
    # This avoids feature drift between primary panel and runtime embedded fallback.
    if isinstance(PANEL_HTML_ASSET_META, dict) and str(PANEL_HTML_ASSET_META.get("source") or "") == "external":
        if str(HTML_PANEL or "").strip():
            HTML_PANEL_EMBEDDED = HTML_PANEL
except Exception:
    pass
HTML_QC_OVERRIDE = _load_ui_asset_text("qc_override.html", HTML_QC_OVERRIDE)

HTML_MANUAL_TEST_MONITOR = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Manual Test Monitor</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 10px; background:#f7f8fa; color:#222; }
  .toolbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
  .actions { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
  .pill { border:1px solid #ccc; border-radius: 10px; padding: 3px 8px; background:#fff; font-size: 12px; }
  .btn { border:1px solid #b5b5b5; border-radius: 6px; padding: 4px 10px; background:#fff; font-size: 12px; cursor: pointer; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .ok { color: #0b6b0b; }
  .warn { color: #a05a00; }
  .err { color: #b00020; }
  .box { background:#fff; border:1px solid #ddd; border-radius:6px; padding:8px; }
  #log { height: 520px; overflow-y: auto; white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
  .line { margin: 0 0 2px 0; }
  .line.ok { color:#0b6b0b; }
  .line.warn { color:#8a5a00; }
  .line.err { color:#b00020; }
  .muted { color:#666; }
</style>
</head>
<body>
  <div class="toolbar">
    <div id="scenario" class="pill">Scenario: -</div>
    <div id="status" class="pill">Status: idle</div>
    <div id="summary" class="pill muted">Summary: -</div>
  </div>
  <div class="actions">
    <button id="stopBtn" class="btn" onclick="mtmRequestStop()">Stop Test</button>
    <div id="stopState" class="muted"></div>
  </div>
  <div class="box">
    <div id="log"></div>
  </div>
<script>
window.__mtm = window.__mtm || {events:[], status:'idle', lang:'en'};
const _MTM_I18N = {
  de: {
    scenario: 'Szenario',
    status: 'Status',
    summary: 'Zusammenfassung',
    idle: 'bereit',
    stop_button: 'Test stoppen',
    stop_requesting: 'Stop wird angefordert ...',
    stop_requested: 'Stop angefordert.',
    stop_failed: 'Stop fehlgeschlagen: ',
  },
  en: {
    scenario: 'Scenario',
    status: 'Status',
    summary: 'Summary',
    idle: 'idle',
    stop_button: 'Stop Test',
    stop_requesting: 'Requesting stop ...',
    stop_requested: 'Stop requested.',
    stop_failed: 'Stop failed: ',
  },
};
function _mtmLang(){
  return (window.__mtm && window.__mtm.lang === 'de') ? 'de' : 'en';
}
function _mtmT(key){
  const lang = _mtmLang();
  const dict = _MTM_I18N[lang] || _MTM_I18N.en;
  return (dict && dict[key]) ? dict[key] : String(key || '');
}
function _mtmApplyLabels(){
  try {
    const btn = document.getElementById('stopBtn');
    if(btn) btn.textContent = _mtmT('stop_button');
  } catch(e) {}
}
function _mtmUpdateStopButton(){
  try {
    const btn = document.getElementById('stopBtn');
    if(!btn) return;
    const status = String((window.__mtm && window.__mtm.status) || 'idle').toLowerCase();
    btn.disabled = (status !== 'running');
  } catch(e) {}
}
function mtmClear(){
  const el = document.getElementById('log');
  if(el) el.innerHTML = '';
  window.__mtm.events = [];
  try {
    const stopState = document.getElementById('stopState');
    if(stopState) stopState.textContent = '';
  } catch(e) {}
}
function mtmSetHeader(data){
  data = data || {};
  try {
    const lang = String(data.lang || '').trim().toLowerCase();
    if(lang === 'de' || lang === 'en') window.__mtm.lang = lang;
  } catch(e) {}
  _mtmApplyLabels();
  const statusValue = String(data.status || 'idle');
  window.__mtm.status = statusValue;
  try { document.getElementById('scenario').textContent = _mtmT('scenario') + ': ' + String(data.scenario || '-'); } catch(e) {}
  try { document.getElementById('status').textContent = _mtmT('status') + ': ' + statusValue; } catch(e) {}
  try {
    const s = data.summary || '-';
    document.getElementById('summary').textContent = _mtmT('summary') + ': ' + (typeof s === 'string' ? s : JSON.stringify(s));
  } catch(e) {}
  _mtmUpdateStopButton();
}
function mtmAppend(entry){
  try {
    const el = document.getElementById('log');
    if(!el) return false;
    const d = document.createElement('div');
    const lvl = String((entry && entry.level) || 'info');
    d.className = 'line ' + lvl;
    const ts = (entry && entry.ts) ? String(entry.ts) + ' · ' : '';
    d.textContent = ts + String((entry && entry.message) || '');
    el.appendChild(d);
    while(el.children.length > 500) el.removeChild(el.firstChild);
    el.scrollTop = el.scrollHeight;
    return true;
  } catch(e) { return false; }
}
async function mtmRequestStop(){
  const btn = document.getElementById('stopBtn');
  const stopState = document.getElementById('stopState');
  try {
    if(btn) btn.disabled = true;
    if(stopState){
      stopState.className = 'muted';
      stopState.textContent = _mtmT('stop_requesting');
    }
    const api = window.pywebview && window.pywebview.api;
    if(!api) throw new Error('pywebview api unavailable');
    let res = null;
    if(typeof api.panel_action === 'function'){
      res = await api.panel_action('manual_test_stop', {lang: _mtmLang()});
    } else if(typeof api.manual_test_request_stop === 'function'){
      res = await api.manual_test_request_stop({lang: _mtmLang()});
    } else {
      throw new Error('manual_test_stop unavailable');
    }
    if(!res || res.ok === false){
      throw new Error(String((res && res.error) || 'manual_test_stop failed'));
    }
    if(stopState){
      stopState.className = 'ok';
      stopState.textContent = _mtmT('stop_requested');
    }
    return true;
  } catch(e) {
    if(stopState){
      stopState.className = 'err';
      stopState.textContent = _mtmT('stop_failed') + String(e && e.message ? e.message : e);
    }
    _mtmUpdateStopButton();
    return false;
  }
}
function mtmReplace(data){
  try {
    mtmClear();
    mtmSetHeader(data || {});
    const ev = (data && Array.isArray(data.events)) ? data.events : [];
    for(const x of ev) mtmAppend(x);
    return true;
  } catch(e) { return false; }
}
_mtmApplyLabels();
_mtmUpdateStopButton();
</script>
</body>
</html>
"""
HTML_MANUAL_TEST_MONITOR = _load_ui_asset_text("manual_test_monitor.html", HTML_MANUAL_TEST_MONITOR)


# --- API BACKEND ---

# ==============================================================================
# Output-Compliance Validator (b9): hard output-contract checks + optional one-pass repair
#   - No HTML injection into model text
#   - Alerts are rendered separately (UI-only)
#   - Validators are derived from the active Comm-SCI rules JSON where available
# ==============================================================================
class OutputComplianceValidator:
    def __init__(self, gov_manager, cfg_obj):
        self.gov = gov_manager
        self.cfg = cfg_obj

        # Ready-status must exist very early (UI may call is_ready during init errors)
        self.ready_status = {"status": False, "msg": "Not connected."}

    # -------- JSON path helper --------
    def _get_path(self, root, path, default=None):
        try:
            cur = root
            for part in path.split('.'):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    return default
            return default if cur is None else cur
        except Exception:
            return default

    def _as_dict(self, x, default=None):
        if isinstance(x, dict):
            return x
        return {} if default is None else default

    def _as_list(self, x):
        return x if isinstance(x, list) else []

    def _as_str(self, x, default=""):
        return x if isinstance(x, str) else default

    def _conversation_lang(self):
        try:
            lang = str(getattr(self.cfg, "get_answer_language", lambda: "de")() or "").strip().lower()
        except Exception:
            lang = ""
        if lang.startswith("de"):
            return "de"
        if lang.startswith("en"):
            return "en"
        return "de"

    # -------- SCI menu validation --------
    def validate_sci_menu(self, text: str):
        """Returns list of violations (strings) for SCI menu contract."""
        vios = []
        if not self.gov.loaded:
            return vios

        gd = self._as_dict(self.gov.data.get("global_defaults", {}))
        oc = self._as_dict(gd.get("output_contract", {}))
        contract = self._as_dict(oc.get("sci_variant_menu_contract", {}))
        if not contract.get("enabled", False):
            return vios

        required = contract.get("required_variant_keys") or ["A","B","C","D","E","F","G","H"]
        found = []
        for key in required:
            k = re.escape(str(key))
            pat = re.compile(rf"(?im)^\s*{k}\s*[\)\:\-–—]\s+\S+")
            pat2 = re.compile(rf"(?im)^\s*{k.lower()}\s*[\)\:\-–—]\s+\S+")
            if pat.search(text) or pat2.search(text):
                found.append(key)
        missing = [k for k in required if k not in found]
        if missing:
            vios.append(f"SCI menu missing variants: {', '.join(missing)}")

        # Menu title + instructions: accept canonical OR localized for conversation lang
        menu_output = self._as_dict(self._get_path(self.gov.data, "sci.variant_menu.menu_output", {}))
        title_canon = (menu_output.get("title") or "").strip()
        instr_canon = (menu_output.get("instructions") or "").strip()
        localized = (menu_output.get("localized") or {})
        lang = self._conversation_lang()
        title_loc = ""
        instr_loc = ""
        if isinstance(localized, dict) and lang in localized and isinstance(localized[lang], dict):
            title_loc = (localized[lang].get("title") or "").strip()
            instr_loc = (localized[lang].get("instructions") or "").strip()

        if title_canon or title_loc:
            if not ((title_loc and title_loc in text) or (title_canon and title_canon in text)):
                vios.append("SCI menu title line missing (canonical or localized).")

        if instr_canon or instr_loc:
            def _prefix(s, n=35): 
                return s[:n] if s else ""
            pref_c = _prefix(instr_canon)
            pref_l = _prefix(instr_loc)
            if (pref_l and pref_l not in text) and (pref_c and pref_c not in text):
                vios.append("SCI menu instruction line missing (canonical or localized).")

        return vios

    # -------- SCI trace validation --------
    def _required_trace_steps_for_variant(self, variant_key: str):
        if not self.gov.loaded:
            return []
        gd = self._as_dict(self.gov.data.get("global_defaults", {}))
        oc = self._as_dict(gd.get("output_contract", {}))
        stc = self._as_dict(oc.get("sci_trace_contract", {}))
        if not stc.get("enabled", False):
            return []

        variants = self._get_path(self.gov.data, "sci.variant_menu.variants", {}) or {}
        vk = (variant_key or "").upper()
        vdef = variants.get(vk, {}) if isinstance(variants, dict) else {}

        # Primary: variant-defined trace_steps (deterministic list)
        try:
            ts = vdef.get("trace_steps") if isinstance(vdef, dict) else None
            if isinstance(ts, list) and ts:
                return [str(s) for s in ts]
        except Exception:
            pass

        # Secondary: maps_to sci_mode
        maps = (vdef.get("maps_to") or {}) if isinstance(vdef, dict) else {}
        mtype = maps.get("type") if isinstance(maps, dict) else None
        mval = maps.get("value") if isinstance(maps, dict) else None

        if mtype == "sci_mode" and isinstance(mval, str) and mval.strip():
            steps = self._get_path(self.gov.data, f"sci.modes.{mval}.steps", []) or []
            if isinstance(steps, list):
                return [str(s) for s in steps]

        # Fallback (method_tag or unknown): minimal SCI steps (Plan/Solution/Check)
        steps = self._get_path(self.gov.data, "sci.modes.SCI.steps", []) or []
        if isinstance(steps, list):
            return [str(s) for s in steps]
        return []


    def _label_regex(self, label: str):
        def _word_pat(w: str) -> str:
            wl = str(w or "").strip().lower()
            if wl == "synthesis":
                # Allow common pluralized model drift: "Syntheses" for canonical "Synthesis"
                return r"(?:synthesis|syntheses)"
            return re.escape(str(w or ""))

        def _part_pat(p: str) -> str:
            p = str(p or "").strip()
            if not p:
                return ""
            m = re.fullmatch(r"([A-Za-z]+)(\d+)", p)
            if m:
                # Accept optional separators before trailing digits (Synthesis2 vs Synthesis_2)
                return _word_pat(m.group(1)) + r"[_\s\-]*" + re.escape(m.group(2))
            return _word_pat(p)

        parts = re.split(r"[_\s]+", label.strip())
        core = r"[_\s\-]*".join([_part_pat(p) for p in parts if p])
        return re.compile(
            rf"(?im)^(?:\s*(?:[-*]|\d+\.)\s+)?(?:\s*#+\s+)?\s*\*{{0,2}}{core}\*{{0,2}}\s*(?:[:\-–—]|$)"
        )

    def validate_sci_trace(self, text: str, variant_key: str):
        vios = []
        if not self.gov.loaded:
            return vios
        gd = self._as_dict(self.gov.data.get("global_defaults", {}))
        oc = self._as_dict(gd.get("output_contract", {}))
        stc = self._as_dict(oc.get("sci_trace_contract", {}))
        if not stc.get("enabled", False):
            return vios

        block_title = stc.get("block_title") or "SCI Trace"
        if not re.search(rf"(?i)\b{re.escape(block_title)}\b", text):
            vios.append(f"Missing '{block_title}' block title.")

        steps = self._required_trace_steps_for_variant(variant_key)
        if not steps:
            return vios

        positions = []
        for s in steps:
            m = self._label_regex(s).search(text)
            if not m:
                vios.append(f"Missing SCI Trace step: {s}")
                positions.append(None)
            else:
                positions.append(m.start())

        # Content check: steps must not be empty (at least one substantive line)
        try:
            # Determine the SCI Trace section span (best-effort)
            m_title = re.search(rf"(?im)^\s*{re.escape(block_title)}\s*:?\s*$", text)
            section_text = text[m_title.start():] if m_title else text
            for s in steps:
                m = self._label_regex(s).search(section_text)
                if not m:
                    continue
                start = m.end()
                line_end = section_text.find("\n", m.start())
                if line_end == -1:
                    line_end = len(section_text)
                header_line = section_text[m.start():line_end]
                _mh_step, _mh_rest = match_required_sci_step_header(header_line, [s])
                if _mh_step:
                    inline = str(_mh_rest or "").strip()
                else:
                    inline = re.sub(rf"(?im)^\s*(?:[-*]|\d+\.)?\s*\*{{0,2}}{re.escape(s)}\*{{0,2}}\s*[:\-–—]\s*", "", header_line).strip()
                if inline:
                    continue
                nxt = len(section_text)
                boundary = re.search(r"(?im)^\s*(Self-?Debunking\b|QC-?Matrix\b)", section_text[start:])
                if boundary:
                    nxt = start + boundary.start()
                for s2 in steps:
                    if s2 == s:
                        continue
                    m2 = self._label_regex(s2).search(section_text[start:nxt])
                    if m2:
                        nxt = start + m2.start()
                        break
                body = section_text[start:nxt]
                body_clean = re.sub(r"(?im)^\s*(?:[*+-]|•|\d+\.)\s*", "", body)
                body_clean = re.sub(r"\s+", " ", body_clean).strip()
                if not re.search(r"[A-Za-z0-9ÄÖÜäöüß]", body_clean):
                    vios.append(f"SCI Trace step '{s}' has no content.")
        except Exception:
            pass

        if all(p is not None for p in positions):
            if any(positions[i] >= positions[i+1] for i in range(len(positions)-1)):
                vios.append("SCI Trace steps not in required order.")

        # Require a substantive final-answer body outside the SCI Trace block.
        # Otherwise a repair pass may satisfy only the trace protocol and still drop the actual answer.
        try:
            m_title = re.search(rf"(?im)^\s*{re.escape(block_title)}\s*:?\s*$", text)
            if m_title and any(p is not None for p in positions):
                tail = text[m_title.end():]
                m_boundary = re.search(r"(?im)^\s*(Self-?Debunking\b|QC(?:-Matrix)?\b)", tail)
                trace_plus_answer = tail[:m_boundary.start()] if m_boundary else tail

                lines = trace_plus_answer.splitlines()
                last_step_line_idx = -1
                for idx, line in enumerate(lines):
                    if any(self._label_regex(s).search(line) for s in steps):
                        last_step_line_idx = idx

                if last_step_line_idx >= 0:
                    post_trace_lines = lines[last_step_line_idx + 1:]
                    post_trace_txt = "\n".join(post_trace_lines)
                    post_trace_txt = re.sub(r"<[^>]+>", " ", post_trace_txt or "")
                    post_trace_txt = re.sub(r"(?im)^\s*(?:Final Answer)\s*:?\s*$", "", post_trace_txt)
                    post_trace_txt = re.sub(r"\s+", " ", post_trace_txt).strip()
                    if not re.search(r"[A-Za-z0-9ÄÖÜäöüß]", post_trace_txt):
                        vios.append("Missing substantive final answer content outside SCI Trace.")
        except Exception:
            pass
        return vios


    # -------- Self-Debunking contract (hard) --------
    def validate_self_debunking(self, text: str, profile_name: str, *, is_command: bool):
        """Returns list of violations for Self-Debunking output contract.

        Skips enforcement for command-like prompts and for exception profiles (e.g., Sandbox).
        """
        vios = []
        if is_command:
            return vios
        if not self.gov.loaded:
            return vios

        # Contract + module config
        gd = self._as_dict(self.gov.data.get("global_defaults", {}))
        oc = (gd.get("output_contract", {}) or {})
        contract = self._as_dict(oc.get("self_debunking_contract", {}))
        if not contract.get("enabled", False):
            return vios

        module = self._as_dict(gd.get("self_debunking", {}))
        if not module.get("enabled", False):
            return vios

        exceptions = set(module.get("exceptions") or [])
        if profile_name in exceptions:
            return vios

        title = (contract.get("required_block_title") or (module.get("block") or {}).get("title") or "Self-Debunking").strip()
        if not title:
            title = "Self-Debunking"

        # Presence
        title_re = re.compile(rf"(?im)^\s*(?:#+\s*)?\*{{0,2}}{re.escape(title)}\*{{0,2}}\s*:?\s*$")
        m = title_re.search(text)
        if not m:
            vios.append(f"Missing '{title}' block.")
            return vios

        # Placement: must be before QC footer
        qc_pos = None
        qc_m = re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:", text)
        if qc_m:
            qc_pos = qc_m.start()

        if qc_pos is not None and m.start() > qc_pos:
            vios.append("Self-Debunking placed after QC footer.")

        # Extract block region for counting points: from title line to QC footer (or end)
        end = qc_pos if qc_pos is not None else len(text)
        block = text[m.end():end]

        # Count numbered points (1., 2., 3.) in the block
        points = re.findall(r"(?m)^\s*\d+\s*[\.)]\s+.+$", block)
        n = len(points)

        min_p = int(contract.get("required_min_points", 2) or 2)
        max_p = int(contract.get("required_max_points", 3) or 3)

        if n < min_p or n > max_p:
            vios.append(f"Self-Debunking must contain {min_p}–{max_p} numbered points (found {n}).")

        return vios


    # -------- Language + Verification gates --------
    _NON_LATIN_SCRIPT_RE = re.compile(
        r"[Ѐ-ԯⷠ-ⷿꙀ-ꚟ"  # Cyrillic
        r"֐-׿"  # Hebrew
        r"؀-ۿݐ-ݿࢠ-ࣿ"  # Arabic
        r"ऀ-ॿ"  # Devanagari
        r"぀-ヿㇰ-ㇿ"  # Kana
        r"㐀-䶿一-鿿"  # CJK
        r"가-힯"  # Hangul
        r"]"
    )

    def _norm_language_policy_mode(self, value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode in ("production", "benchmark"):
            return mode
        return "production"

    def _language_policy_mode(self, state) -> str:
        try:
            mode = self._norm_language_policy_mode(getattr(state, "language_policy_mode", "production"))
            if mode in ("production", "benchmark"):
                return mode
        except Exception:
            pass
        try:
            if self.cfg is not None and hasattr(self.cfg, "get_language_policy_mode"):
                return self._norm_language_policy_mode(self.cfg.get_language_policy_mode())
        except Exception:
            pass
        return "production"

    def _extract_language_contract_scope(self, text: str, *, is_command: bool) -> str:
        """Best-effort extraction of the content scope for language checks."""
        src = str(text or "")
        if not src:
            return src
        # Language guard should evaluate readable text, not structural HTML.
        src = re.sub(r"<[^>]+>", " ", src)
        if is_command:
            return src

        # Content scope ends before post-content blocks.
        end_m = re.search(
            r"(?im)^\s*(?:Self-?Debunking|QC(?:-Matrix)?|CGI-Feedback|QC\s*\(CGI\)|Response\s+at)\b",
            src,
        )
        if end_m:
            src = src[:end_m.start()]

        # Exclude wrapper/system lines from content-language checks.
        ctl_re = re.compile(
            r"(?i)^\s*(?:CONTROL\s+LAYER(?:\s+(?:NOTE|ALERT|BLOCK|WARN(?:ING)?))?|CSC(?:\s|:)|COMM\s+(?:STATE|CONFIG|ANCHOR|AUDIT|VALIDATE|HELP)\b)"
        )
        header_re = re.compile(r"(?i)^\s*(?:Reqs\s*:|In\s*:|Out\s*:|Profile\s*:|\[[^\]]*Comm-SCI[^\]]*\])")
        out_lines = []
        for ln in src.splitlines():
            if ctl_re.search(ln) or header_re.search(ln):
                continue
            out_lines.append(ln)
        return "\n".join(out_lines)

    def _has_verification_route_marker(self, text: str, markers: dict) -> bool:
        text_l = str(text or "").lower()
        found = False
        for _rtype, spec in (markers.items() if isinstance(markers, dict) else []):
            if not isinstance(spec, dict):
                continue
            any_of = spec.get("any_of", []) or []
            for mk in any_of:
                if mk and (str(mk).lower() in text_l):
                    found = True
                    break
            if found:
                break

        if not found and re.search(r"\[(GREEN|YELLOW|RED|GRAY)-(?:WEB|DOC|TRAIN)\]", text):
            found = True

        # Deterministic fallback for localized/handwritten marker lines.
        if not found and re.search(r"(?im)^\s*(?:[-*]\s*)?(?:source|quelle|measurement|messung|contrast|kontrast|web-?check)\s*:", text):
            found = True

        return found

    def _has_claim_downgrade_marker(self, text_l: str) -> bool:
        return any(
            kw in text_l
            for kw in (
                "hypothesis",
                "hypothetical",
                "unclear",
                "uncertain",
                "unverified",
                "preliminary",
                "speculat",
                "pr claim",
                "annahme",
                "hypothese",
                "unklar",
                "unsicher",
                "unverifiziert",
                "vorlaeufig",
                "spekulativ",
                "nicht gesichert",
            )
        )

    # -------- Verification Route Gate (hard) --------
    def validate_verification_route_gate(self, text: str, *, is_command: bool):
        """Hard checks for strong/RED claims against verification-route + uncertainty contracts."""
        vios = []
        if is_command or (not self.gov.loaded) or (not text):
            return vios

        gate = (self.gov.data.get("global_defaults", {}) or {}).get("verification_route_gate", {}) or {}
        if not gate.get("enabled", False):
            return vios

        heur = gate.get("strong_claim_heuristics", {}) or {}
        if not heur.get("enabled", False):
            return vios

        text_l = text.lower()
        markers = gate.get("route_presence_markers", {}) or {}
        has_route = self._has_verification_route_marker(text, markers)
        has_uncertainty = bool(
            re.search(r"\bU[1-8]\b", text)
            or re.search(r"(?i)data-u-code\s*=\s*(?:\"|')?(U[1-8])(?:\"|')?", text)
        )
        has_downgrade = self._has_claim_downgrade_marker(text_l)

        kw = []
        kw += list(heur.get("keywords_de", []) or [])
        kw += list(heur.get("keywords_en", []) or [])
        has_strong = any(str(k).lower() in text_l for k in kw) if kw else False

        # RED-claim guard: RED requires uncertainty + route (or explicit downgrade fallback path).
        has_red_claim = bool(
            re.search(
                r"\[(?:RED|ROT)(?:-[A-Z0-9]+)*\]|\b(?:RED|ROT)\b|🔴|niedrige verlaesslichkeit|low reliability",
                text,
                re.IGNORECASE,
            )
        )

        if has_strong and (not has_route):
            if has_uncertainty and has_downgrade:
                pass
            elif has_uncertainty and (not has_downgrade):
                vios.append(
                    "Verification Route Gate: strong-claim heuristic triggered; uncertainty label alone is insufficient. "
                    "Add a verification route marker (Source/Measurement/Contrast/Web Check) or explicit downgrade wording."
                )
            elif has_downgrade and (not has_uncertainty):
                vios.append(
                    "Verification Route Gate: strong-claim heuristic triggered; downgraded wording without uncertainty label (U1-U8)."
                )
            else:
                vios.append(
                    "Verification Route Gate: strong-claim heuristic triggered, but no verification route markers found "
                    "(Source/Measurement/Contrast/Web Check)."
                )

        if has_red_claim:
            if not has_uncertainty:
                vios.append("Verification Route Gate: RED claim requires uncertainty label (U1-U8).")
            if (not has_route) and not (has_uncertainty and has_downgrade):
                vios.append(
                    "Verification Route Gate: RED claim requires at least one verification route marker "
                    "(Source/Measurement/Contrast/Web Check)."
                )

        return vios

    # -------- Epistemic provenance (soft) --------
    def validate_epistemic_provenance(self, text: str):
        """Soft check: if Evidence-Linker tags are used without an origin suffix, warn (default origin is usually TRAIN)."""
        vios = []
        if (not self.gov.loaded) or (not text):
            return vios
        prov = (self.gov.data.get("global_defaults", {}) or {}).get("epistemic_provenance", {}) or {}
        if not prov.get("enabled", False):
            return vios
        bare = re.findall(r"\[(GREEN|YELLOW|RED|GRAY)\](?!-)", text)
        if bare:
            default_origin = prov.get("default_origin_when_unknown", "TRAIN")
            vios.append(f"Epistemic provenance: Evidence-Linker tags without origin suffix detected. Consider adding '-{default_origin}' (e.g. [GREEN-{default_origin}]).")
        return vios

    def _strip_language_exception_spans(self, text: str) -> str:
        src = str(text or "")

        # Technical spans: code + URLs.
        src = re.sub(r"```[\s\S]*?```", " ", src)
        src = re.sub(r"`[^`]*`", " ", src)
        src = re.sub(r"https?://\S+", " ", src)

        # Citation/source lines are allowed to keep original language.
        cleaned_lines = []
        cite_re = re.compile(
            r"(?i)^\s*(?:>|[-*]\s*)?(?:source|quelle|reference|references|citation|zitat|quote|doi|url|link|arxiv|isbn)\s*[:\-]"
        )
        for ln in src.splitlines():
            if cite_re.search(ln):
                cleaned_lines.append(" ")
            else:
                cleaned_lines.append(ln)
        src = "\n".join(cleaned_lines)

        # Quotes are an explicit exception.
        src = re.sub(r"\"(?:[^\"\\\\]|\\\\.)*\"", " ", src)
        src = re.sub(r"'(?:[^'\\]|\\.)*'", " ", src)
        src = re.sub(r"„[^“]*“", " ", src)
        src = re.sub(r"‚[^‘]*‘", " ", src)
        src = re.sub(r"»[^«]*«", " ", src)
        src = re.sub(r"«[^»]*»", " ", src)

        # Drop HTML tags before script checks.
        src = re.sub(r"<[^>]+>", " ", src)
        return src

    def validate_language_script_contract(self, text: str, *, expected_lang: str, is_command: bool = False):
        """Hard check: no foreign scripts outside explicit exceptions (quotes/sources/code)."""
        vios = []
        tgt = str(expected_lang or "").strip().lower()
        if tgt not in ("de", "en"):
            return vios

        scoped = self._extract_language_contract_scope(text, is_command=is_command)
        cleaned = self._strip_language_exception_spans(scoped)
        m = self._NON_LATIN_SCRIPT_RE.search(cleaned)
        if m:
            vios.append(
                f"Language contract: expected {tgt.upper()} content; found non-{tgt.upper()} script outside allowed quote/source contexts."
            )
        return vios

    def validate_language(self, text: str, *, expected_lang: str = "", is_command: bool = False):
        """Heuristic DE/EN drift detection on explanatory text (excluding code spans)."""
        vios = []
        tgt = str(expected_lang or "").strip().lower() or self._conversation_lang()
        if tgt not in ("de", "en"):
            return vios

        scoped = self._extract_language_contract_scope(text, is_command=is_command)

        # Strip code blocks and inline code to avoid bias from snippets.
        sample = re.sub(r"```[\s\S]*?```", " ", str(scoped or ""))
        sample = re.sub(r"`[^`]*`", " ", sample)

        # Allowed protocol tokens should not count as EN drift in DE answers.
        sample = re.sub(
            r"(?i)\b(?:SCI|Trace|Self-?Debunking|QC-?Matrix|Verification|Route|Source|Measurement|Contrast|Web-?Check|Profile|Overlay|Control|Layer)\b",
            " ",
            sample,
        )
        sample = sample[:1200]

        tokens = re.findall(r"[A-Za-zÄÖÜäöüß]+", sample.lower())
        if not tokens:
            return vios

        de_markers = {
            "der", "die", "das", "und", "ist", "sind", "wird", "wurde", "mit", "fuer", "für", "nicht",
            "dass", "ich", "du", "wir", "sie", "es", "ein", "eine", "als", "auch", "bei", "auf",
            "im", "in", "zu", "von", "oder", "wenn", "dann", "weil", "den", "dem", "des",
        }
        en_markers = {
            "the", "and", "is", "are", "will", "was", "were", "with", "for", "not", "that", "this",
            "you", "your", "we", "they", "a", "an", "as", "also", "in", "to", "of", "or", "if",
            "then", "because", "which", "who", "from",
        }

        de_score = sum(1 for t in tokens if t in de_markers)
        en_score = sum(1 for t in tokens if t in en_markers)

        # Keep threshold robust against short snippets.
        if tgt == "de":
            if en_score >= 6 and en_score > max(2, de_score) * 2:
                vios.append("Language drift: expected DE, output appears predominantly EN.")
        elif tgt == "en":
            if de_score >= 6 and de_score > max(2, en_score) * 2:
                vios.append("Language drift: expected EN, output appears predominantly DE.")
        return vios

    def validate(self, *, text: str, state, expect_menu: bool, expect_trace: bool, is_command: bool, user_prompt: str):
        hard = []
        soft = []

        profile_name = getattr(state, "active_profile", "Standard") or "Standard"

        # Comm Start: harden against hallucinated profile claims (forbid inferred switching).
        # Canonical intent: Comm Start re-initializes to the default profile.
        try:
            if is_command and (user_prompt or "").strip() == "Comm Start":
                default_prof = (getattr(self.gov, 'data', {}) or {}).get('default_profile') or 'Standard'
                m = re.search(r"(?im)\bProfile\s+([A-Z][A-Za-z0-9_-]+)\b", text)
                if m:
                    mentioned = m.group(1)
                    if mentioned != str(default_prof) and ("Profile-Switch-Audit" not in text):
                        hard.append(
                            f"Comm Start: response mentions non-default profile '{mentioned}' without Profile-Switch-Audit."
                        )
        except Exception:
            pass


        if expect_menu:
            hard += self.validate_sci_menu(text)
        if expect_trace:
            hard += self.validate_sci_trace(text, getattr(state, "sci_variant", "") or "")

        
        # Verification Route Gate is a hard contract for content answers
        hard += self.validate_verification_route_gate(text, is_command=is_command)
        
        # Epistemic provenance is a soft warning
        soft += self.validate_epistemic_provenance(text)


        # Self-Debunking is a hard contract when module is active (skip for commands)
        hard += self.validate_self_debunking(text, profile_name, is_command=is_command)

        # Language contract:
        # - Commands/UI: conversation language.
        # - Content answers: configured answer_language.
        try:
            expected_lang = ""
            if is_command:
                expected_lang = str(self._conversation_lang() or "").strip().lower()
            else:
                expected_lang = str(getattr(state, "answer_language", "") or "").strip().lower()
                if expected_lang not in ("de", "en"):
                    expected_lang = str(self._conversation_lang() or "").strip().lower()

            language_mode = self._language_policy_mode(state)

            script_vios = self.validate_language_script_contract(
                text,
                expected_lang=expected_lang,
                is_command=is_command,
            )
            lang_vios = self.validate_language(
                text,
                expected_lang=expected_lang,
                is_command=is_command,
            )
            language_vios = list(script_vios or []) + list(lang_vios or [])
            if language_vios:
                if language_mode == "benchmark":
                    soft += [f"Language policy benchmark: {v}" for v in language_vios]
                else:
                    hard += language_vios
        except Exception:
            pass

        return hard, soft


    def build_repair_prompt(self, *, user_prompt: str, raw_response: str, state, hard_violations: list, soft_violations: list):
        lang = (getattr(state, "answer_language", "") or "").strip().lower()
        if lang not in ("de", "en"):
            lang = self._conversation_lang()
        parts = []
        parts.append("CONTROL LAYER REPAIR REQUEST (one pass).")
        parts.append("You MUST output a corrected assistant message that complies with the active Comm-SCI ruleset.")
        parts.append("Constraints:")
        parts.append("- Do NOT mention this repair request.")
        parts.append("- Keep the meaning and content as unchanged as possible; ONLY add missing required protocol blocks or formatting.")
        # Language contract: keep command tokens English, but render explanatory text in the conversation language.
        if (lang or "").lower() == "de":
            parts.append("- Render explanatory text in German. Keep command tokens in English.")
        else:
            parts.append("- Render explanatory text in English. Keep command tokens in English.")
        parts.append("")
        parts.append("Detected hard contract violations:")
        for v in hard_violations:
            parts.append(f"- {v}")
        if soft_violations:
            parts.append("")
            parts.append("Additional warnings:")
            for v in soft_violations:
                parts.append(f"- {v}")

        vk = getattr(state, "sci_variant", "") or ""
        steps = self._required_trace_steps_for_variant(vk) if vk else []
        if steps:
            parts.append("")
        
        parts.append("")
        parts.append("Repair guidance (minimal/surgical):")
        parts.append("- Do NOT rewrite the whole answer. Only add the missing protocol elements.")
        parts.append("- If you add blocks, place them at the required position (typically after the final answer and before QC).")
        parts.append("")

        need_vrg = any("Verification Route Gate" in v for v in hard_violations)
        if need_vrg:
            parts.append("Verification Route Gate repair guidance:")
            parts.append("- Add at least ONE verification route marker line.")
            parts.append("- Prefer safe/transparent markers (do NOT fabricate web checks):")
            parts.append("  - Source: TRAIN (general background knowledge)")
            parts.append("  - Measurement: not performed")
            parts.append("  - Contrast: plausible alternative noted but not evaluated")
            parts.append("  - Web-Check: not performed")
            parts.append("- If you cannot support the strong claim, downgrade it and include an uncertainty label U1–U8.")
            parts.append("")

        all_hard_vios = [str(v) for v in (hard_violations or [])]
        need_lang_contract = any(("Language contract" in v) or ("Language drift" in v) for v in all_hard_vios)
        if need_lang_contract:
            parts.append("Language contract repair guidance:")
            if (lang or "").lower() == "de":
                parts.append("- Keep answer content in German (Latin script).")
            else:
                parts.append("- Keep answer content in English (Latin script).")
            parts.append("- If foreign-script names/terms appear in running text, transliterate them into the answer-language script.")
            parts.append("- Keep original-script forms only in explicit quotes, source/citation lines, code blocks, and URLs.")
            parts.append("- Do not rely on name-specific whitelists.")
            parts.append("")

        # SCI Trace guidance: required whenever SCI is active and the ruleset defines steps for the chosen variant.
        try:
            sci_active = bool(getattr(state, "sci_active", False))
        except Exception:
            sci_active = False

        sci_vios = any(("SCI Trace" in v) or ("Missing SCI Trace" in v) or ("SCI Trace step" in v) for v in hard_violations)
        if sci_active and steps:
            parts.append("SCI Trace requirements:")
            parts.append("- Include a visible block titled 'SCI Trace'.")
            parts.append("- Include ALL step labels exactly (underscores/spaces/hyphens allowed), in this order:")
            for s in steps:
                parts.append(f"  - {s}")
            parts.append("- Each step must contain at least one substantive sentence; do NOT output empty step headers.")
            parts.append("- If content must be withheld, keep the step label and write: 'Redacted: <reason>'.")
            parts.append("- After the SCI Trace, include a substantive final answer body OUTSIDE the SCI Trace (before Self-Debunking/QC).")
            try:
                color_on = str(getattr(state, "color", "off") or "off").strip().lower() == "on"
            except Exception:
                color_on = False
            if color_on:
                parts.append("- Color=on: apply Evidence-Linker tags only in the final answer body (not in SCI Trace / Self-Debunking / QC).")
            try:
                brev = int((getattr(state, "qc_overrides", {}) or {}).get("brevity", 2))
            except Exception:
                brev = 2
            if brev <= 0:
                parts.append("- Because Brevity=0 is active: each SCI Trace step must contain at least TWO substantive sentences.")
            parts.append("")


        parts.append("Original user prompt:")
        parts.append(user_prompt.strip())
        parts.append("")
        parts.append("Your previous assistant response (to be repaired):")
        parts.append(raw_response.strip())
        parts.append("")
        parts.append("Now output ONLY the corrected assistant message (Markdown).")
        return "\\n".join(parts)


# ----------------------------
# CSC Refiner (deterministic, wrapper-enforced)
# ----------------------------
@dataclass
class CSCDecision:
    apply: bool
    governance_triggered: bool
    trigger_source: str
    token_count: int
    f_score: int
    mode: str  # "none" | "refine" | "refine_governance"


@dataclass
class CSCMetadata:
    applied: bool
    message: str
    trigger: str
    mode: str
    governance_triggered: bool
    token_count: int
    f_score: int
    overlay: str = ""
    profile: str = ""
    threshold_multiplier: int = 1
    threshold_f_score: int = 0
    threshold_token_count: int = 0
    min_token_count_governance: int = 0
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

class CSCRefiner:
    """Deterministic CSC trigger evaluation + strict injection.

    Key principle (v19.6.9 intent): refinement_only.
    The wrapper enforces the decision (when/where to apply) deterministically.

    NOTE: We do NOT let CSC mutate protocol/meta blocks (SCI menu/trace, QC, Control Layer).
    We inject a short counter-perspective + marker before Self-Debunking/QC.
    """

    def __init__(self, gov_manager, cfg_obj):
        self.gov = gov_manager
        self.cfg = cfg_obj

        # Fail-safe defaults
        self.max_len_increase_pct = 15
        self.marker = "CSC-Refine: applied"

        try:
            cl = (self.gov.data.get("control_layer", {}) or {})
            bridge = (cl.get("csc_trigger_bridge", {}) or {})
            constraints = (bridge.get("constraints", {}) or {})
            tm = (constraints.get("transparency_marker", {}) or {})
            self.marker = tm.get("marker", self.marker) or self.marker

            csc = ((cl.get("subsystems", {}) or {}).get("csc_engine", {}) or {})
            policy = (csc.get("policy", {}) or {})
            brev = (policy.get("brevity_cap", {}) or {})
            self.max_len_increase_pct = int(brev.get("max_relative_length_increase_percent", self.max_len_increase_pct) or self.max_len_increase_pct)

            det = ((csc.get("metrics_engine", {}) or {}).get("feature_detectors", {}) or {})
            self._re_code = re.compile(det.get("count_code", r"```[a-z]*"), re.MULTILINE)
            self._re_math = re.compile(det.get("count_math", r"(\d+\s*[+\-*/=^%<>]\s*\d+|[\w\d]+\^)"))

            pipe = ((csc.get("operational_workflow", {}) or {}).get("pipeline", []) or [])
            self._refine_params = (pipe[1].get("parameters", {}) if len(pipe) > 1 else {})
            self._gov_params = (pipe[2].get("parameters", {}) if len(pipe) > 2 else {})
        except Exception:
            # keep fail-safe defaults
            self._re_code = re.compile(r"```[a-z]*", re.MULTILINE)
            self._re_math = re.compile(r"(\d+\s*[+\-*/=^%<>]\s*\d+|[\w\d]+\^)" )
            self._refine_params = {}
            self._gov_params = {}

    @staticmethod
    def count_ws_tokens(s: str) -> int:
        return len((s or "").split())

    def f_score(self, s: str) -> int:
        s = s or ""
        code = len(self._re_code.findall(s))
        math_ = len(self._re_math.findall(s))
        return code * 5 + math_ * 4

    def decide(
        self,
        *,
        comm_active: bool,
        active_profile: str,
        input_raw: str,
        uncertainty_U4_active: bool,
        web_check_hook_active: bool,
        strong_claim_detected: bool,
        neutrality_delta_negative: bool,
        threshold_multiplier: int = 1,
    ) -> CSCDecision:

        tok = self.count_ws_tokens(input_raw)
        fs = self.f_score(input_raw)

        if not comm_active:
            return CSCDecision(False, False, "", tok, fs, "none")

        # Profile constraints (strict per v19.6.9 intent)
        disallowed = {"Briefing", "Sandbox"}
        if (active_profile or "Standard") in disallowed:
            return CSCDecision(False, False, "", tok, fs, "none")

        thr_fs = int(self._refine_params.get("threshold_f_score", 8) or 8)
        thr_tok = int(self._refine_params.get("min_token_count", 80) or 80)
        gov_min_tok = int(self._gov_params.get("min_token_count_governance", 40) or 40)

        mul = int(threshold_multiplier or 1)
        if mul < 1:
            mul = 1
        if mul != 1:
            thr_fs *= mul
            thr_tok *= mul
            gov_min_tok *= mul

        csc_complexity_threshold = (fs >= thr_fs and tok >= thr_tok)
        governance_triggered = bool(
            uncertainty_U4_active or web_check_hook_active or strong_claim_detected or neutrality_delta_negative
        )

        flags = {
            "uncertainty_U4_active": uncertainty_U4_active,
            "web_check_hook_active": web_check_hook_active,
            "strong_claim_detected": strong_claim_detected,
            "neutrality_delta_negative": neutrality_delta_negative,
            "csc_complexity_threshold": csc_complexity_threshold,
        }

        trigger_source = ""
        for k in ("uncertainty_U4_active", "web_check_hook_active", "strong_claim_detected", "neutrality_delta_negative", "csc_complexity_threshold"):
            if flags.get(k):
                trigger_source = k
                break

        apply = any(flags.values())
        if not apply:
            return CSCDecision(False, governance_triggered, "", tok, fs, "none")

        if governance_triggered and tok >= gov_min_tok:
            mode = "refine_governance"
        elif (not governance_triggered) and csc_complexity_threshold:
            mode = "refine"
        else:
            mode = "none"

        return CSCDecision(True, governance_triggered, trigger_source, tok, fs, mode)

    def _find_insertion_index(self, text: str) -> int:
        """Insert before Self-Debunking or QC footer (whichever comes first)."""
        if not text:
            return 0

        candidates = []
        m = re.search(r"(?im)^\s*Self-Debunking\b", text)
        if m:
            candidates.append(m.start())
        m = re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:", text)
        if m:
            candidates.append(m.start())

        return min(candidates) if candidates else len(text)
    def _lang(self) -> str:
        """Return UI language (de/en) for deterministic UI strings."""
        try:
            lang = getattr(self.gov_state, 'ui_lang', None) or UI_LANG
        except Exception:
            lang = UI_LANG
        lang_s = str(lang or '').strip().lower()
        if lang_s.startswith('de'):
            return 'de'
        return 'en'

    # ----------------------------
    # SCI prompt helpers (deterministic, zero-LLM)
    # ----------------------------
    def _sci_variant_def(self, letter: str):
        '''Return (variant_def, required_steps, mapped_mode).

        Canonical v19.6.9 semantics:
        - If the active variant defines trace_steps: use them exactly.
        - Else, if maps_to.type == 'sci_mode': use sci.modes.<value>.steps.
        - Else (method_tag or unknown): fall back to sci.modes.SCI.steps (Plan/Solution/Check).

        Returns fail-soft defaults on malformed rules.
        '''
        L = (letter or '').strip().upper()
        sci = self.gov.data.get('sci', {}) if getattr(self, 'gov', None) else {}
        variant_menu = sci.get('variant_menu', {}) if isinstance(sci, dict) else {}
        variants = variant_menu.get('variants', {}) if isinstance(variant_menu, dict) else {}
        vdef = variants.get(L, {}) if isinstance(variants, dict) else {}

        # 1) Variant-provided trace_steps (takes precedence)
        steps = []
        try:
            ts = vdef.get('trace_steps') if isinstance(vdef, dict) else None
            if isinstance(ts, list) and ts:
                steps = [str(s) for s in ts if s is not None]
        except Exception:
            steps = []

        # 2) maps_to (usually an object {type,value})
        maps_to = 'SCI'
        maps_type = None
        maps_val = None
        try:
            maps = vdef.get('maps_to') if isinstance(vdef, dict) else None
            if isinstance(maps, dict):
                maps_type = maps.get('type')
                maps_val = maps.get('value')
            elif isinstance(maps, str):
                # legacy/older rulesets
                maps_val = maps
                maps_type = 'sci_mode'
        except Exception:
            maps_type = None
            maps_val = None

        if maps_type == 'sci_mode' and isinstance(maps_val, str) and maps_val.strip():
            maps_to = maps_val.strip()
        else:
            maps_to = 'SCI'

        # 3) If no explicit trace_steps, resolve steps via mode
        if not steps:
            try:
                modes = sci.get('modes', {}) if isinstance(sci, dict) else {}
                mode_obj = modes.get(maps_to, {}) if isinstance(modes, dict) else {}
                mode_steps = mode_obj.get('steps', []) if isinstance(mode_obj, dict) else []
                if isinstance(mode_steps, list) and mode_steps:
                    steps = [str(s) for s in mode_steps if s is not None]
            except Exception:
                steps = []

        # 4) Ultimate fallback
        if not steps:
            try:
                modes = sci.get('modes', {}) if isinstance(sci, dict) else {}
                mode_obj = modes.get('SCI', {}) if isinstance(modes, dict) else {}
                mode_steps = mode_obj.get('steps', []) if isinstance(mode_obj, dict) else []
                if isinstance(mode_steps, list):
                    steps = [str(s) for s in mode_steps if s is not None]
            except Exception:
                steps = []

        return vdef if isinstance(vdef, dict) else {}, steps, maps_to


    def _wrap_user_with_sci(self, user_text: str, *, variant: str) -> str:
        """Prefix the user message with SCI instructions so the model actually follows the selected variant."""
        vdef, steps, maps_to = self._sci_variant_def(variant)
        name = str(vdef.get("name", "")).strip()
        focus = str(vdef.get("focus", "")).strip()

        # Keep internal control tokens/instructions in English (not user-facing).
        hdr = f"[SCI MODE ACTIVE] Variant {variant}"

        if name:
            hdr += f" — {name}"
        if focus:
            hdr += f"\nFocus: {focus}"
        hdr += f"\nMapped mode: {maps_to}\n"

        step_lines = "\n".join([f"- {s}" for s in steps]) if steps else "- (no steps configured)"

        instr = (
            hdr +
            "\nYou MUST follow the Comm-SCI SCI Trace protocol for this answer.\n"
            "Output requirements:\n"
            "1) Include a visible section titled exactly: 'SCI Trace'\n"
            "2) For EACH required step, write ONE line in this exact format: '<Step>: <content>'\n"
            "   - <content> must be at least one substantive sentence (no empty placeholders).\n"
            "   - Keep steps in the same order as listed.\n"
            "3) After the SCI Trace section, provide the final answer.\n"
            "4) Do not rename steps, do not invent extra steps, do not output empty step headers.\n\n"
            "Required SCI Trace steps (use exactly these labels):\n" + step_lines +
            "\n\nUser request:\n" + (user_text or "")
        )
        return instr

    # ----------------------------
    # Deterministic Comm Help renderer (no LLM call)
    # ----------------------------

    def _ui_onoff(self, v: str) -> str:
        lang = self._lang()
        v = (v or "").strip().lower()
        return ui_onoff(v)

    def _ui_overlay(self, overlay: str) -> str:
        lang = self._lang()
        ov = (overlay or "").strip()
        ov_l = ov.lower()
        if ov_l in {"", "off", "none"}:
            return 'off'
        if ov_l == "strict":
            return 'Strict'
        if ov_l == "explore":
            return 'Explore'
        return ov

    def _status_line(
        self,
        *,
        sysname: str,
        ver: str,
        profile: str,
        sci: str,
        overlay: str,
        ctl: str,
        qc: str,
        cgi: str,
        color: str,
        lang_override: str = None,
    ) -> str:
        lang = (lang_override or self._lang())
        sci_norm = (sci or "OFF").strip()
        sci_l = sci_norm.lower()
        sci_out = "OFF" if sci_l in {"", "off", "none"} else sci_norm.upper()
        # IMPORTANT: avoid accidental shadowing by the imported `sys` module.
        # The status line must use the system name from the ruleset (sysname).
        return (
            f"Active profile: {profile} · SCI: {sci_out} · Overlay: {overlay} · "
            f"Control Layer: {ctl} · QC: {qc} · CGI: {cgi} · Color: {color}"
        )
    def _qc_footer_for_profile(self, profile_name: str) -> str:
        """Create a deterministic QC-Matrix line based on the active profile's qc_target (use upper bounds).

        UI language (DE/EN) is applied deterministically. Command tokens are unaffected.
        """
        lang = self._lang()
        try:
            overrides = getattr(self.gov_state, 'qc_overrides', {}) if hasattr(self, 'gov_state') else {}
            eff = gov.get_effective_qc_values(profile_name, overrides)
            order = ["clarity", "brevity", "evidence", "empathy", "consistency", "neutrality"]

            parts = []
            for key in order:
                if key in eff:
                    val = int(eff[key])
                else:
                    val = 2 if key not in {"clarity", "consistency"} else 3
                parts.append(f"{QC_LABELS.get(key, key)} {val} (Δ0)")

            return f"{'QC-Matrix'}: " + " · ".join(parts)
        except Exception:
            # fallback in case ruleset is missing/incomplete
            return f"{'QC-Matrix'}: " + " · ".join([
                f"{'Clarity'} 3 (Δ0)",
                f"{'Brevity'} 2 (Δ0)",
                f"{'Evidence'} 2 (Δ0)",
                f"{'Empathy'} 2 (Δ0)",
                f"{'Consistency'} 3 (Δ0)",
                f"{'Neutrality'} 2 (Δ0)",
            ])


    def _render_comm_help(self) -> str:
        """Render 'Comm Help' locally from the loaded ruleset to avoid LLM reflow/hallucinated token text."""
        if not gov.loaded:
            return "Comm Help: No ruleset loaded."

        sysname = gov.data.get("system_name", "Comm-SCI-Control")
        ver = gov.data.get("version", "")

        ui_lang = self._lang()
        note_html = ''
        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"
        overlay = getattr(self.gov_state, "overlay", "") or "off"
        sci = getattr(self.gov_state, "sci_variant", "") or "off"
        color = getattr(self.gov_state, "color", "off") or "off"

        # Header line (matches your existing style)
        out = []
        out.append(self._status_line(sysname=sysname, ver=ver, profile=prof, sci=sci, overlay=overlay, ctl="on", qc="on", cgi="on", color=color))

        cmds = gov.data.get("commands", {}) or {}

        def _render_cmd_group(title: str, group_key: str):
            grp = (cmds.get(group_key, {}) or {})
            out.append(f"\n{title}")
            if not isinstance(grp, dict) or not grp:
                out.append("(none)")
                return
            for token in sorted(grp.keys(), key=lambda x: x.lower()):
                fn = (grp.get(token, {}) or {}).get("function", "").strip()
                fn = re.sub(r"\s+", " ", fn)
                out.append(f"- {token}: {fn}" if fn else f"- {token}")

        out.append("\n1) Comm core (commands.primary + commands.help_and_codes)")
        _render_cmd_group("", "primary")
        _render_cmd_group("", "help_and_codes")

        out.append("\n2) Profiles (commands.profile_control)")
        _render_cmd_group("", "profile_control")

        out.append("\n3) Modes & Overlays (commands.mode_control)")
        _render_cmd_group("", "mode_control")

        out.append("\n4) SCI control (commands.sci_control)")
        _render_cmd_group("", "sci_control")

        out.append("\n5) Color tools (commands.color_control)")
        _render_cmd_group("", "color_control")

        # SCI variants
        out.append("\n6) SCI variants (A–H)")
        sci_root = gov.data.get("sci", {}) or {}
        vmenu = (sci_root.get("variant_menu", {}) or {})
        variants = (vmenu.get("variants", {}) or {})
        if isinstance(variants, dict) and variants:
            for key in sorted(variants.keys(), key=lambda x: str(x).upper()):
                entry = variants.get(key, {}) or {}
                name = (entry.get("name") or "").strip()
                focus = (entry.get("focus") or entry.get("short_focus") or "").strip()
                line = f"- {str(key).upper()}) {name}"
                if focus:
                    focus = re.sub(r"\s+", " ", focus)
                    line += f" — {focus}"
                out.append(line)
        else:
            out.append("(no variants defined)")

        # Numeric codes
        out.append("\n7) Numeric codes (numeric_codes)")
        nc = gov.data.get("numeric_codes", {}) or {}
        cats = nc.get("categories", []) or []
        sv = (nc.get("special_values", {}) or {}).get("dash", "").strip()
        if isinstance(cats, list) and cats:
            for c in cats:
                if not isinstance(c, dict):
                    continue
                nm = c.get("name", "")
                idx = c.get("index", "")
                out.append(f"- {nm} (Index {idx})")
                opts = c.get("options", {}) or {}
                for k in sorted(opts.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                    out.append(f"  - {k}: {opts.get(k)}")
        if sv:
            out.append(f"- Dash: {sv}")

        # Modules
        out.append("\n8) Quality/control modules")
        out.append("- QC: Rating footer (Clarity/Brevity/Evidence/Empathy/Consistency/Neutrality). Active while QC=on (profile-dependent).") 
        out.append("- CGI: Optional user feedback (cognitive gain), if CGI=on.")
        out.append("- Control Layer: deterministic token/output contracts (no silent adjustments).")

        # Parsing rule (SCI pending)
        out.append("\n9) Parsing rules (SCI pending)")
        out.append("- If SCI selection is pending: a single letter A–H selects the variant.")
        out.append("- Otherwise, a standalone letter is treated as normal input text (not a command).")

        # Deterministic QC footer (keeps toolchain stable)
        out.append("\n" + self._qc_footer_for_profile(prof))

        return "\n".join(out).strip()
 
    def _render_comm_help_html(self, lang=None):
        """Deterministic HTML help using the loaded governance JSON as source of truth."""
        ui_lang = str(lang or self._lang() or "en").strip().lower()
        ui_lang = "de" if ui_lang.startswith("de") else "en"

        txt = {
            "en": {
                "title": "Comm Help",
                "ruleset": "Active ruleset",
                "version": "Version",
                "status": "Runtime status",
                "intro": "Commands are parsed standalone-only. Keep command tokens exact.",
                "cmd_col": "Command",
                "desc_col": "Description / Function",
                "sec_primary": "Primary Commands",
                "sec_comm_tools": "Comm Tools",
                "sec_profiles": "Profiles",
                "sec_modes": "Modes and Overlays",
                "sec_sci": "SCI Control",
                "sec_color": "Color Control",
                "sec_variants": "SCI Variants (A-H)",
                "sec_codes": "Numeric Codes",
                "sec_modules": "Governance Modules",
                "sec_panel": "Panel Operations and Effect",
                "sec_parsing": "Parsing Rules",
                "sec_citation": "DOI and Citation",
                "no_commands": "No commands found in the active ruleset.",
                "variant_col": "Variant",
                "name_col": "Name",
                "focus_col": "Focus",
                "map_col": "Mapped mode",
                "step_col": "Trace steps",
                "cat_col": "Category",
                "opt_col": "Option",
                "meaning_col": "Meaning",
                "default_code": "Default code",
                "module_col": "Module",
                "notes_col": "Notes",
                "mod_qc": "QC footer and target corridor enforcement for content answers.",
                "mod_cgi": "Optional user feedback loop for cognitive gain adjustment.",
                "mod_ctl": "Deterministic command routing, contracts and compliance checks.",
                "panel_provider": "Provider / Model",
                "panel_provider_note": "Select runtime backend and model. This directly defines execution context and output characteristics.",
                "panel_answer_lang": "Answer language",
                "panel_answer_lang_note": "Sets default language for model answers and F1 help payload localization.",
                "panel_logs": "Logs and audits",
                "panel_logs_note": "Load/fork chats and inspect audits for reproducibility and incident analysis.",
                "panel_manual": "Manual tests",
                "panel_manual_note": "Runs scenario checks for wrapper contracts (format, routing, QC footer, SCI behavior).",
                "parse_standalone": "Standalone command mode",
                "parse_standalone_note": "Commands are interpreted only as standalone input.",
                "parse_sci_pending": "SCI selection pending",
                "parse_sci_pending_note": "A single letter A-H selects the variant while pending.",
                "parse_default": "Default SCI variant",
                "parse_default_none": "not defined",
                "res_col": "Resource",
                "doi_col": "DOI / Link",
                "wrapper_concept": "Wrapper concept DOI",
                "ruleset_concept": "Ruleset concept DOI",
                "wrapper_site": "Public wrapper website",
                "qc_overrides": "QC-Overrides",
                "public_site": "Public ruleset website",
                "ruleset_repo": "Ruleset GitHub repository",
                "wrapper_repo": "Wrapper GitHub repository",
                "license": "License",
                "maintainer": "Maintainer",
                "license_text": "Apache-2.0 (see LICENSE)",
            },
            "de": {
                "title": "Comm Hilfe",
                "ruleset": "Aktives Regelwerk",
                "version": "Version",
                "status": "Runtime-Status",
                "intro": "Kommandos werden nur standalone geparst. Tokens exakt senden.",
                "cmd_col": "Kommando",
                "desc_col": "Beschreibung / Funktion",
                "sec_primary": "Primaere Kommandos",
                "sec_comm_tools": "Comm Tools",
                "sec_profiles": "Profile",
                "sec_modes": "Modi und Overlays",
                "sec_sci": "SCI-Steuerung",
                "sec_color": "Color-Steuerung",
                "sec_variants": "SCI-Varianten (A-H)",
                "sec_codes": "Numerische Codes",
                "sec_modules": "Governance-Module",
                "sec_panel": "Panel-Bedienung und Wirkung",
                "sec_parsing": "Parsing-Regeln",
                "sec_citation": "DOI und Zitation",
                "no_commands": "Im aktiven Regelwerk wurden keine Kommandos gefunden.",
                "variant_col": "Variante",
                "name_col": "Name",
                "focus_col": "Fokus",
                "map_col": "Abbildung",
                "step_col": "Trace-Schritte",
                "cat_col": "Kategorie",
                "opt_col": "Option",
                "meaning_col": "Bedeutung",
                "default_code": "Default-Code",
                "module_col": "Modul",
                "notes_col": "Hinweis",
                "mod_qc": "QC-Footer und Zielkorridor werden fuer Inhaltsantworten durchgesetzt.",
                "mod_cgi": "Optionaler Feedback-Loop fuer kognitiven Gewinn.",
                "mod_ctl": "Deterministisches Command-Routing, Contracts und Compliance-Checks.",
                "panel_provider": "Provider / Modell",
                "panel_provider_note": "Waehlt Backend und Modell. Das bestimmt direkt den Ausfuehrungskontext und Antwortcharakter.",
                "panel_answer_lang": "Antwortsprache",
                "panel_answer_lang_note": "Setzt die Standardsprache fuer Modellantworten und die Lokalisierung der F1-Hilfe.",
                "panel_logs": "Logs und Audits",
                "panel_logs_note": "Laedt/forkt Chats und prueft Audits fuer Reproduzierbarkeit und Fehleranalyse.",
                "panel_manual": "Manual Tests",
                "panel_manual_note": "Fuehrt Szenario-Checks fuer Wrapper-Contracts aus (Format, Routing, QC-Footer, SCI-Verhalten).",
                "parse_standalone": "Standalone-Command-Modus",
                "parse_standalone_note": "Kommandos werden nur als eigenstaendiger Input interpretiert.",
                "parse_sci_pending": "SCI-Auswahl ausstehend",
                "parse_sci_pending_note": "Ein einzelner Buchstabe A-H waehlt die Variante solange pending.",
                "parse_default": "Default-SCI-Variante",
                "parse_default_none": "nicht definiert",
                "res_col": "Ressource",
                "doi_col": "DOI / Link",
                "wrapper_concept": "Wrapper Concept DOI",
                "ruleset_concept": "Regelwerk Concept DOI",
                "wrapper_site": "Oeffentliche Wrapper-Webseite",
                "qc_overrides": "QC-Overrides",
                "public_site": "Oeffentliche Regelwerk-Webseite",
                "ruleset_repo": "Regelwerk GitHub-Repository",
                "wrapper_repo": "Wrapper GitHub-Repository",
                "license": "Lizenz",
                "maintainer": "Maintainer",
                "license_text": "Apache-2.0 (siehe LICENSE)",
            },
        }[ui_lang]

        gov_obj = getattr(self, "gov", None) or globals().get("gov")
        data = getattr(gov_obj, "data", None) if gov_obj else None
        if not isinstance(data, dict):
            err = "Governance JSON not available." if ui_lang == "en" else "Governance-JSON nicht verfuegbar."
            return f"<div class='comm-help' style='color:red'>Error: {html.escape(err)}</div>"

        commands = data.get("commands") or {}
        if not isinstance(commands, dict):
            commands = {}

        profile = getattr(getattr(self, "gov_state", object()), "active_profile", "Standard") or "Standard"
        overlay = getattr(getattr(self, "gov_state", object()), "overlay", "off") or "off"
        color = getattr(getattr(self, "gov_state", object()), "color", "off") or "off"
        comm = "on" if bool(getattr(getattr(self, "gov_state", object()), "comm_active", False)) else "off"
        sci_pending = bool(getattr(getattr(self, "gov_state", object()), "sci_pending", False))
        sci_variant = str(getattr(getattr(self, "gov_state", object()), "sci_variant", "") or "").strip().upper()
        sci = "PENDING" if sci_pending else (sci_variant or "OFF")
        language_policy = self._language_policy_mode()

        source = data.get("source") if isinstance(data.get("source"), dict) else {}
        system_name = str(source.get("system_name") or data.get("system_name") or "Comm-SCI-Control")
        version = str(source.get("version") or data.get("version") or "").strip() or "n/a"
        rules_file = os.path.basename(str(getattr(gov_obj, "current_filename", "") or ""))

        def _spec_desc(spec_obj) -> str:
            if isinstance(spec_obj, dict):
                return re.sub(r"\s+", " ", str(spec_obj.get("function") or "").strip())
            if spec_obj is None:
                return ""
            return re.sub(r"\s+", " ", str(spec_obj).strip())

        def _render_command_group(title: str, group_key: str) -> str:
            group = commands.get(group_key) or {}
            if not isinstance(group, dict) or not group:
                return ""
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(title)}</h3>")
            out.append(
                "<table><thead><tr>"
                f"<th>{html.escape(txt['cmd_col'])}</th>"
                f"<th>{html.escape(txt['desc_col'])}</th>"
                "</tr></thead><tbody>"
            )
            for token in sorted(group.keys(), key=lambda x: str(x).lower()):
                desc = _spec_desc(group.get(token))
                out.append(
                    "<tr><td class='cmd'>%s</td><td>%s</td></tr>"
                    % (html.escape(str(token)), html.escape(desc))
                )
            out.append("</tbody></table></div>")
            return "".join(out)

        parts = []
        parts.append("<div class='comm-help'>")
        parts.append(f"<div class='help-status'><b>{html.escape(txt['title'])}</b> · {html.escape(system_name)}</div>")
        parts.append(
            f"<div class='minor'>{html.escape(txt['ruleset'])}: <code class='nowrap'>{html.escape(rules_file or 'n/a')}</code> · "
            f"{html.escape(txt['version'])}: <b>{html.escape(version)}</b></div>"
        )
        parts.append(
            "<div class='minor'>%s: Comm=%s · Profile=%s · SCI=%s · Overlay=%s · Color=%s · Language policy=%s</div>"
            % (
                html.escape(txt["status"]),
                html.escape(comm),
                html.escape(str(profile)),
                html.escape(sci),
                html.escape(str(overlay)),
                html.escape(str(color)),
                html.escape(str(language_policy)),
            )
        )
        parts.append(f"<p><i>{html.escape(txt['intro'])}</i></p>")

        group_specs = [
            (txt["sec_primary"], "primary"),
            (txt["sec_comm_tools"], "help_and_codes"),
            (txt["sec_profiles"], "profile_control"),
            (txt["sec_modes"], "mode_control"),
            (txt["sec_sci"], "sci_control"),
            (txt["sec_color"], "color_control"),
        ]
        rendered_any_group = False
        for title, key in group_specs:
            chunk = _render_command_group(title, key)
            if chunk:
                rendered_any_group = True
                parts.append(chunk)
        if not rendered_any_group:
            parts.append(f"<div class='minor'>{html.escape(txt['no_commands'])}</div>")

        sci = data.get("sci") if isinstance(data.get("sci"), dict) else {}
        variant_menu = sci.get("variant_menu") if isinstance(sci.get("variant_menu"), dict) else {}
        variants = variant_menu.get("variants") if isinstance(variant_menu.get("variants"), dict) else {}
        if isinstance(variants, dict) and variants:
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(txt['sec_variants'])}</h3>")
            out.append(
                "<table class='sci-table'><thead><tr>"
                f"<th>{html.escape(txt['variant_col'])}</th>"
                f"<th>{html.escape(txt['name_col'])}</th>"
                f"<th>{html.escape(txt['focus_col'])}</th>"
                f"<th>{html.escape(txt['map_col'])}</th>"
                f"<th>{html.escape(txt['step_col'])}</th>"
                "</tr></thead><tbody>"
            )
            for letter in sorted(variants.keys(), key=lambda x: str(x).upper()):
                vdef = variants.get(letter) if isinstance(variants.get(letter), dict) else {}
                name = str(vdef.get("name") or "").strip()
                focus = str(vdef.get("focus") or "").strip()
                maps_to = vdef.get("maps_to")
                if isinstance(maps_to, dict):
                    mapped = str(maps_to.get("value") or maps_to.get("type") or "")
                else:
                    mapped = str(maps_to or "")
                trace_steps = vdef.get("trace_steps") if isinstance(vdef.get("trace_steps"), list) else []
                step_count = str(len(trace_steps)) if trace_steps else "-"
                out.append(
                    "<tr><td class='cmd'><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (
                        html.escape(str(letter).upper()),
                        html.escape(name),
                        html.escape(focus),
                        html.escape(mapped),
                        html.escape(step_count),
                    )
                )
            out.append("</tbody></table></div>")
            parts.append("".join(out))

        numeric_codes = data.get("numeric_codes") if isinstance(data.get("numeric_codes"), dict) else {}
        categories = numeric_codes.get("categories") if isinstance(numeric_codes.get("categories"), list) else []
        dash_meaning = str((numeric_codes.get("special_values") or {}).get("dash") or numeric_codes.get("dash_meaning") or "").strip()
        default_code = str(numeric_codes.get("default") or "").strip()
        if categories:
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(txt['sec_codes'])}</h3>")
            if default_code:
                out.append(f"<div class='minor'>{html.escape(txt['default_code'])}: <b>{html.escape(default_code)}</b></div>")
            out.append(
                "<table class='numcodes-table'><thead><tr>"
                f"<th>{html.escape(txt['cat_col'])}</th>"
                f"<th>{html.escape(txt['opt_col'])}</th>"
                f"<th>{html.escape(txt['meaning_col'])}</th>"
                "</tr></thead><tbody>"
            )
            for cat in categories:
                if not isinstance(cat, dict):
                    continue
                cat_name = str(cat.get("name") or "").strip()
                cat_idx = str(cat.get("index") or "").strip()
                opts = cat.get("options") if isinstance(cat.get("options"), dict) else {}
                opt_keys = list(opts.keys())
                opt_keys.sort(key=lambda x: int(x) if str(x).isdigit() else str(x))
                first = True
                for opt_key in opt_keys:
                    meaning = str(opts.get(opt_key) or "")
                    if first:
                        out.append(
                            "<tr><td><b>%s</b> (Index %s)</td><td><b>%s</b></td><td>%s</td></tr>"
                            % (
                                html.escape(cat_name),
                                html.escape(cat_idx),
                                html.escape(str(opt_key)),
                                html.escape(meaning),
                            )
                        )
                        first = False
                    else:
                        out.append(
                            "<tr><td></td><td><b>%s</b></td><td>%s</td></tr>"
                            % (html.escape(str(opt_key)), html.escape(meaning))
                        )
            if dash_meaning:
                out.append(
                    "<tr><td><b>Dash</b></td><td><b>-</b></td><td>%s</td></tr>"
                    % html.escape(dash_meaning)
                )
            out.append("</tbody></table></div>")
            parts.append("".join(out))

        out = []
        out.append('<div class="help-cat">')
        out.append(f"<h3>{html.escape(txt['sec_modules'])}</h3>")
        out.append(
            "<table><thead><tr>"
            f"<th>{html.escape(txt['module_col'])}</th>"
            f"<th>{html.escape(txt['notes_col'])}</th>"
            "</tr></thead><tbody>"
        )
        out.append("<tr><td class='cmd'>QC</td><td>%s</td></tr>" % html.escape(txt["mod_qc"]))
        out.append("<tr><td class='cmd'>CGI</td><td>%s</td></tr>" % html.escape(txt["mod_cgi"]))
        out.append("<tr><td class='cmd'>Control Layer</td><td>%s</td></tr>" % html.escape(txt["mod_ctl"]))
        out.append("</tbody></table></div>")
        parts.append("".join(out))

        out = []
        out.append('<div class="help-cat">')
        out.append(f"<h3>{html.escape(txt['sec_panel'])}</h3>")
        out.append(
            "<table><thead><tr>"
            f"<th>{html.escape(txt['module_col'])}</th>"
            f"<th>{html.escape(txt['notes_col'])}</th>"
            "</tr></thead><tbody>"
        )
        out.append("<tr><td class='cmd'>%s</td><td>%s</td></tr>" % (html.escape(txt["panel_provider"]), html.escape(txt["panel_provider_note"])))
        out.append("<tr><td class='cmd'>%s</td><td>%s</td></tr>" % (html.escape(txt["panel_answer_lang"]), html.escape(txt["panel_answer_lang_note"])))
        out.append("<tr><td class='cmd'>%s</td><td>%s</td></tr>" % (html.escape(txt["panel_logs"]), html.escape(txt["panel_logs_note"])))
        out.append("<tr><td class='cmd'>%s</td><td>%s</td></tr>" % (html.escape(txt["panel_manual"]), html.escape(txt["panel_manual_note"])))
        out.append("</tbody></table></div>")
        parts.append("".join(out))

        parser_contract = data.get("parser_contract") if isinstance(data.get("parser_contract"), dict) else {}
        standalone_only = bool(parser_contract.get("commands_standalone_only"))
        special_parsing = parser_contract.get("special_parsing") if isinstance(parser_contract.get("special_parsing"), dict) else {}
        sci_select = special_parsing.get("sci_variant_selection") if isinstance(special_parsing.get("sci_variant_selection"), dict) else {}
        default_variant = str(sci_select.get("default_variant_if_no_selection") or "").strip() or txt["parse_default_none"]
        out = []
        out.append('<div class="help-cat">')
        out.append(f"<h3>{html.escape(txt['sec_parsing'])}</h3>")
        out.append(
            "<table><thead><tr>"
            f"<th>{html.escape(txt['module_col'])}</th>"
            f"<th>{html.escape(txt['notes_col'])}</th>"
            "</tr></thead><tbody>"
        )
        out.append(
            "<tr><td class='cmd'>%s</td><td>%s (%s)</td></tr>"
            % (
                html.escape(txt["parse_standalone"]),
                html.escape(txt["parse_standalone_note"]),
                html.escape("enabled" if standalone_only else "disabled"),
            )
        )
        out.append(
            "<tr><td class='cmd'>%s</td><td>%s</td></tr>"
            % (html.escape(txt["parse_sci_pending"]), html.escape(txt["parse_sci_pending_note"]))
        )
        out.append(
            "<tr><td class='cmd'>%s</td><td>%s</td></tr>"
            % (html.escape(txt["parse_default"]), html.escape(default_variant))
        )
        out.append("</tbody></table></div>")
        parts.append("".join(out))

        citation_rows = [
            (txt["wrapper_concept"], "https://doi.org/10.5281/zenodo.18445672", "10.5281/zenodo.18445672"),
            (txt["ruleset_concept"], "https://doi.org/10.5281/zenodo.17928357", "10.5281/zenodo.17928357"),
            (txt["wrapper_site"], "https://vfi64.github.io/wrapper/", "vfi64.github.io/wrapper"),
            (txt["public_site"], "https://vfi64.github.io/Comm-SCI-Control/", "vfi64.github.io/Comm-SCI-Control"),
            (txt["ruleset_repo"], "https://github.com/vfi64/Comm-SCI-Control", "github.com/vfi64/Comm-SCI-Control"),
            (txt["wrapper_repo"], "https://github.com/vfi64/wrapper", "github.com/vfi64/wrapper"),
            (txt["license"], "https://github.com/vfi64/Comm-SCI-Control-private/blob/main/LICENSE", txt["license_text"]),
            (txt["maintainer"], "", "Volker Fickert"),
        ]
        out = []
        out.append('<div class="help-cat">')
        out.append(f"<h3>{html.escape(txt['sec_citation'])}</h3>")
        out.append(
            "<table><thead><tr>"
            f"<th>{html.escape(txt['res_col'])}</th>"
            f"<th>{html.escape(txt['doi_col'])}</th>"
            "</tr></thead><tbody>"
        )
        for label, url, ref in citation_rows:
            if url:
                out.append(
                    "<tr><td>%s</td><td><a href='%s' target='_blank' rel='noreferrer'>%s</a></td></tr>"
                    % (html.escape(label), html.escape(url), html.escape(ref))
                )
            else:
                out.append("<tr><td>%s</td><td>%s</td></tr>" % (html.escape(label), html.escape(ref)))
        out.append("</tbody></table></div>")
        parts.append("".join(out))

        try:
            parts.append(f"<div class='minor' style='margin-top:10px'>{html.escape(self._qc_footer_for_profile(profile))}</div>")
            ovs = gov.normalize_qc_overrides(getattr(self.gov_state, "qc_overrides", {}) or {})
            if ovs:
                disp = {
                    "clarity": "Clarity",
                    "brevity": "Brevity",
                    "evidence": "Evidence",
                    "empathy": "Empathy",
                    "consistency": "Consistency",
                    "neutrality": "Neutrality",
                }
                parts2 = [f"{disp.get(k, k)}={v}" for k, v in ovs.items()]
                parts.append(f"<div class='minor'>{html.escape(txt['qc_overrides'])}: {html.escape(' · '.join(parts2))}</div>")
        except Exception:
            pass

        parts.append("</div>")
        return "".join(parts)



    def _render_comm_state(self) -> str:
        """Deterministic plaintext renderer for 'Comm State' (no LLM)."""
        if not gov.loaded:
            return "Comm State: No ruleset loaded."

        sysname = gov.data.get("system_name", "Comm-SCI-Control")
        ver = gov.data.get("version", "")

        ui_lang = self._lang()
        note_html = ''

        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"
        comm = "on" if getattr(self.gov_state, "comm_active", False) else "off"
        overlay = getattr(self.gov_state, "overlay", "") or "off"
        color = getattr(self.gov_state, "color", "off") or "off"
        language_policy = self._language_policy_mode()
        ctl = getattr(self.gov_state, "control_layer", "on") or "on"
        qc = getattr(self.gov_state, "qc", "on") or "on"
        cgi = getattr(self.gov_state, "cgi", "on") or "on"

        sci_pending = bool(getattr(self.gov_state, "sci_pending", False))
        sci_variant = getattr(self.gov_state, "sci_variant", "") or ""
        if sci_pending:
            sci = "PENDING"
        else:
            sci = sci_variant.upper() if sci_variant else "OFF"

        anchor_auto = "on" if bool(getattr(self.gov_state, "anchor_auto", True)) else "off"
        user_turns = int(getattr(self.gov_state, "user_turns", 0) or 0)
        dyn = getattr(self.gov_state, "dynamic_nudge", "") or ""

        out = []
        out.append(
            f"Comm: {comm} · Active profile: {prof} · SCI: {sci} · Overlay: {overlay} · "
            f"Control Layer: {ctl} · QC: {qc} · CGI: {cgi} · Color: {color} · Language policy: {language_policy}"
        )
        try:
            ds = getattr(self, 'deps_status', {}) or {}
            parts = []
            for k in ('auditstream','rendering_utils','compliance_scan'):
                ok, _err = ds.get(k, (False, ''))
                parts.append(f"{k}={'ok' if ok else 'missing'}")
            out.append('Modules: ' + ', '.join(parts))
        except Exception:
            pass

        out.append(f"Anchor auto: {anchor_auto} · User turns: {user_turns}")
        if dyn:
            out.append(f"Dynamic nudge: {dyn}")

        out.append(self._qc_footer_for_profile(prof))
        try:
            ovs = gov.normalize_qc_overrides(getattr(self.gov_state, 'qc_overrides', {}) or {})
            if ovs:
                disp = {'clarity':'Clarity','brevity':'Brevity','evidence':'Evidence','empathy':'Empathy','consistency':'Consistency','neutrality':'Neutrality'}
                parts = [f"{disp.get(k,k)}={v}" for k, v in ovs.items()]
                out.append("QC-Overrides: " + " · ".join(parts))
        except Exception:
            pass
        return "\n".join(out).strip()


    def _render_comm_state_html(self, audit_line: str = "") -> str:
        """Deterministic, stable HTML renderer for 'Comm State' (no Markdown reflow)."""
        if not gov.loaded:
            return '<div class="comm-help comm-state">Comm State: No ruleset loaded.</div>'

        sysname = gov.data.get("system_name", "Comm-SCI-Control")
        ver = gov.data.get("version", "")

        ui_lang = self._lang()
        note_html = ''

        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"
        comm = "on" if getattr(self.gov_state, "comm_active", False) else "off"
        overlay = getattr(self.gov_state, "overlay", "") or "off"
        color = getattr(self.gov_state, "color", "off") or "off"
        language_policy = self._language_policy_mode()
        ctl = getattr(self.gov_state, "control_layer", "on") or "on"
        qc = getattr(self.gov_state, "qc", "on") or "on"
        cgi = getattr(self.gov_state, "cgi", "on") or "on"

        sci_pending = bool(getattr(self.gov_state, "sci_pending", False))
        sci_variant = getattr(self.gov_state, "sci_variant", "") or ""
        sci = "PENDING" if sci_pending else (sci_variant.upper() if sci_variant else "OFF")

        anchor_auto = "on" if bool(getattr(self.gov_state, "anchor_auto", True)) else "off"
        user_turns = int(getattr(self.gov_state, "user_turns", 0) or 0)
        dyn = getattr(self.gov_state, "dynamic_nudge", "") or ""

        status = (
            f"Comm: {comm} · Active profile: {prof} · SCI: {sci} · Overlay: {overlay} · "
            f"Control Layer: {ctl} · QC: {qc} · CGI: {cgi} · Color: {color} · Language policy: {language_policy}"
        )

        rows = [
            ("Comm active", comm),
            ("Active profile", prof),
            ("Overlay", overlay),
            ("SCI", sci),
            ("Control Layer", ctl),
            ("QC", qc),
            ("CGI", cgi),
            ("Color", color),
            ("Language policy", language_policy),
            ("Anchor auto", anchor_auto),
            ("User turns", str(user_turns)),
        ]
        if dyn:
            rows.append(("Dynamic nudge", dyn))
        # Module availability (Stage 3d)
        try:
            ds = getattr(self, 'deps_status', {}) or {}
            parts = []
            for k in ('auditstream','rendering_utils','compliance_scan'):
                ok, _err = ds.get(k, (False, ''))
                parts.append(f"{k}={'ok' if ok else 'missing'}")
            rows.append(("Modules", ", ".join(parts)))
        except Exception:
            pass

        out = []
        out.append('<div class="comm-help comm-state">')
        if note_html:
            out.append(note_html)
        out.append(f'<div class="help-status">{html.escape(status)}</div>')
        if audit_line:
            out.append(f"<div style='margin-top:8px'>{html.escape(str(audit_line))}</div>")
        out.append('<table class="state-table">')
        out.append('<tbody>')
        for k, v in rows:
            out.append(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>")
        out.append('</tbody></table>')
        out.append(f"<div style='margin-top:10px'>{html.escape(self._qc_footer_for_profile(prof))}</div>")
        try:
            ovs = gov.normalize_qc_overrides(getattr(self.gov_state, 'qc_overrides', {}) or {})
            if ovs:
                disp = {'clarity':'Clarity','brevity':'Brevity','evidence':'Evidence','empathy':'Empathy','consistency':'Consistency','neutrality':'Neutrality'}
                parts2 = [f"{disp.get(k,k)}={v}" for k, v in ovs.items()]
                out.append(f"<div class='minor'>QC-Overrides: {html.escape(' · '.join(parts2))}</div>")
        except Exception:
            pass

        out.append('</div>')
        return "\n".join(out)


    def _render_comm_config(self) -> str:
        """Deterministic plaintext renderer for 'Comm Config' (no LLM)."""
        if not gov.loaded:
            return "Comm Config: No ruleset loaded."

        sysname = gov.data.get("system_name", "Comm-SCI-Control")
        ver = gov.data.get("version", "")
        fname = getattr(gov, "current_filename", "") or ""
        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"

        out = []
        out.append(f"Loaded rules file: {fname}")
        out.append("")
        # Prefer raw_json (exact), fallback to pretty dump
        raw = getattr(gov, "raw_json", "") or ""
        if raw.strip():
            out.append(raw.strip())
        else:
            out.append(json.dumps(gov.data, ensure_ascii=False, indent=2, sort_keys=True))
        out.append("")
        out.append(self._qc_footer_for_profile(prof))
        return "\n".join(out).strip()


    def _render_comm_config_html(self) -> str:
        """Deterministic, stable HTML renderer for 'Comm Config' (no Markdown reflow)."""
        if not gov.loaded:
            return '<div class="comm-help comm-config">Comm Config: No ruleset loaded.</div>'

        sysname = gov.data.get("system_name", "Comm-SCI-Control")
        ver = gov.data.get("version", "")

        ui_lang = self._lang()
        note_html = ''
        fname = getattr(gov, "current_filename", "") or ""
        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"

        status = f"Loaded rules file: {fname}"

        # Prefer raw_json (exact), fallback to pretty dump
        raw = getattr(gov, "raw_json", "") or ""
        if not raw.strip():
            raw = json.dumps(gov.data, ensure_ascii=False, indent=2, sort_keys=True)

        out = []
        out.append('<div class="comm-help comm-config">')
        if note_html:
            out.append(note_html)
        out.append(f'<div class="help-status">{html.escape(status)}</div>')
        out.append('<div class="minor">Read-only view of the full governance configuration (deterministic from JSON, no LLM).</div>')
        out.append('<details class="config-details">')
        out.append('<summary>Raw JSON anzeigen</summary>')
        out.append(f'<pre class="raw-json">{html.escape(raw)}</pre>')
        out.append('</details>')
        out.append(f"<div style='margin-top:10px'>{html.escape(self._qc_footer_for_profile(prof))}</div>")
        out.append('</div>')
        return "\n".join(out)
    
    def _render_sci_menu_html(self, lang=None):
        """SCI menu: renders in current conversation language (de/en), keeps styling,
        uses JSON as Source of Truth, optional I18N strings only if present.
        """
        ui_lang = (lang or getattr(self.gov_state, 'ui_lang', None) or UI_LANG)
        ui_lang = 'de' if str(ui_lang).strip().lower().startswith('de') else 'en'

        def tr(key: str, fallback: str = "") -> str:
            try:
                s = key
                return s if s and s != key else fallback
            except Exception:
                return fallback

        gov_obj = getattr(self, "gov", None) or globals().get("gov")
        data = getattr(gov_obj, "data", None) if gov_obj else None
        if not isinstance(data, dict):
            return "<div class='comm-help' style='color:red'>Error: Governance JSON not available.</div>"

        sci = data.get("sci") or {}
        variant_menu = (sci.get("variant_menu") or {}) if isinstance(sci, dict) else {}
        menu_output = (variant_menu.get("menu_output") or {}) if isinstance(variant_menu, dict) else {}

        localized = (menu_output.get("localized") or {}) if isinstance(menu_output, dict) else {}
        loc_block = (localized.get(ui_lang) or {}) if isinstance(localized, dict) else {}

        # Title/instructions: JSON localized first; I18N as fallback
        title = loc_block.get("title") or menu_output.get("title") or ""
        instructions = loc_block.get("instructions") or menu_output.get("instructions") or ""

        if not title:
            title = tr("sci_menu_title", "SCI variants (selection)")
        hint = tr("sci_menu_hint", "")

        variants = (variant_menu.get("variants") or {}) if isinstance(variant_menu, dict) else {}
        if not isinstance(variants, dict) or not variants:
            return "<div class='comm-help' style='color:red'>Error: No SCI variants found in canonical JSON.</div>"

        col_var = tr("sci_menu_col_var", "Variant")
        col_name = "Name"
        col_focus = tr("sci_menu_col_focus", "Focus / Method")

        parts = []
        parts.append("<div class='sci-menu-container'>")
        # Keep a stable English anchor phrase for deterministic parsing and tests.
        # (The visible title may still be localized.)
        parts.append("<div class='sci-menu-caption'>SCI Variants</div>")
        parts.append(f"<h3>{html.escape(str(title))}</h3>")
        if hint:
            parts.append(f"<p><i>{html.escape(str(hint))}</i></p>")
        if instructions:
            parts.append(f"<p><i>{html.escape(str(instructions))}</i></p>")

        parts.append("<table class='sci-table'>")
        parts.append(
            "<thead><tr>"
            f"<th>{html.escape(str(col_var))}</th>"
            f"<th>{html.escape(str(col_name))}</th>"
            f"<th>{html.escape(str(col_focus))}</th>"
            "</tr></thead><tbody>"
        )

        for letter in "ABCDEFGH":
            v = variants.get(letter) or {}
            if not isinstance(v, dict):
                v = {}

            name_json = v.get("name") or ""
            focus_json = v.get("focus") or ""

            # Optional I18N overrides (only if present in your I18N table)
            name_i18n = tr(f"sci_name_{letter}", "") or tr(f"sci_var_{letter}", "")
            focus_i18n = tr(f"sci_focus_{letter}", "")

            name = name_i18n or name_json
            focus = focus_i18n or focus_json

            parts.append(
                "<tr style='cursor:pointer' onclick=\"remoteInput('%s')\">"
                "<td class='cmd'><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                % (
                    html.escape(letter),
                    html.escape(letter),
                    html.escape(str(name)),
                    html.escape(str(focus)),
                )
            )

        parts.append("</tbody></table></div>")
        return "".join(parts)
    
    def _render_anchor_snapshot_html(self) -> str:
        """Deterministic Anchor Snapshot rendered as distinct HTML block (no LLM)."""
        ui_lang = self._lang()
        title = ANCHOR_TITLE
        badge = ANCHOR_CHECKPOINT

        # Build a compact snapshot using the SAME sources as status_line/QC (no hallucination)
        sysname = gov.data.get("system_name", "Comm-SCI-Control") if gov.loaded else "Comm-SCI-Control"
        ver = gov.data.get("version", "") if gov.loaded else ""
        prof = getattr(self.gov_state, "active_profile", "Standard") or "Standard"
        sci = getattr(self.gov_state, "sci_variant", "") or ("OFF" if not getattr(self.gov_state, "sci_pending", False) else "PENDING")
        overlay = getattr(self.gov_state, "overlay", "") or "off"
        ctl = getattr(self.gov_state, "control_layer", "on") if hasattr(self.gov_state, "control_layer") else "on"
        qc = "on" if getattr(self.gov_state, "qc_on", True) else "off"
        cgi = "on" if getattr(self.gov_state, "cgi_on", True) else "off"
        color = getattr(self.gov_state, "color", "on") or "on"

        status = self._status_line(sysname=sysname, ver=ver, profile=prof, sci=sci, overlay=overlay, ctl=ctl, qc=qc, cgi=cgi, color=color)

        qc_footer = self._qc_footer_for_profile(prof)

        snapshot_lines = []
        snapshot_lines.append(status)
        snapshot_lines.append(qc_footer)
        # Optional: include numeric code in state if available
        try:
            code = getattr(self.gov_state, "numeric_code", "") or ""
            if code:
                snapshot_lines.append(f"Code: {code}")
        except Exception:
            pass

        out = []
        out.append('<div class="comm-help comm-anchor">')
        out.append(f'<div class="help-status">{html.escape(title)}</div>')
        out.append('<div class="anchor-box">')
        out.append(f'<div class="anchor-badge">{html.escape(badge)}</div>')
        joined = "\n".join(snapshot_lines)
        out.append(f'<pre>{html.escape(joined)}</pre>')
        out.append('</div>')
        out.append('</div>')
        return "\n".join(out)

    def start_background_thread(self):
        # Idempotent: pywebview can (depending on backend/window lifecycle) call the start callback more than once.
        try:
            if getattr(self, "_bg_started", False):
                return
            setattr(self, "_bg_started", True)
        except Exception:
            pass
        t = threading.Thread(target=self._init_process)
        t.daemon = True
        t.start()

    def _init_process(self):
        time.sleep(0.5) 
        gov.load_file() # Lädt Standard-Datei
        self.gov_state = _init_state_from_rules()

        # Session counters for renderer stability (structural-only diagnostics)
        self.session_render_ok_count = int(getattr(self, "session_render_ok_count", 0) or 0)
        self.session_render_fallback_count = int(getattr(self, "session_render_fallback_count", 0) or 0)


        
        
        
        
        try:
            _g = globals().get('gov')
            if _g is not None:
                gov.runtime_state = self.gov_state
                setattr(_g, 'runtime_state', self.gov_state)
        except Exception:
            pass

        # ENONLY startup defaults (requested):
        # 1) Comm Start automatically after successful ruleset load
        # 2) Color on by default
        # 3) Strict on by default
        try:
            self.gov_state.color = "on"
        except Exception:
            pass
        try:
            self.gov_state.overlay = "Strict"
        except Exception:
            pass
        try:
            self.gov_state.comm_active = True
        except Exception:
            pass

        self._connect_api(reason="startup")

        # Visible system notice (does not send a message to the model).
        try:
            if getattr(self, "main_win", None):
                self.main_win.evaluate_js("addMsg('sys', 'Auto: Comm Start · Strict on · Color on.')")
        except Exception:
            pass


    def _get_plain_system_instruction(self) -> str:
        """Minimal system instruction used when Comm-SCI is stopped."""
        return "You are a helpful assistant. Answer in English."

    def _hide_verification_route_lines_in_chat(self) -> bool:
        """Display policy for verification-route marker lines.

        Default is visible (False) so users can audit verification provenance directly.
        Optional config:
          - root: hide_verification_route_lines
          - provider scoped: providers.<id>.hide_verification_route_lines
        """
        try:
            conf = (getattr(getattr(self, 'cfg_mgr', None), 'config', None) or getattr(cfg, 'config', {}) or {})
            provider = (self._active_provider() or 'gemini').strip().lower()
            provs = conf.get('providers') or {}
            pconf = provs.get(provider) if isinstance(provs, dict) else {}
            raw = None
            if isinstance(pconf, dict):
                raw = pconf.get('hide_verification_route_lines')
            if raw is None:
                raw = conf.get('hide_verification_route_lines')
            if raw is None:
                return False
            if isinstance(raw, bool):
                return raw
            s = str(raw).strip().lower()
            if s in ('1', 'true', 'on', 'yes', 'y'):
                return True
            if s in ('0', 'false', 'off', 'no', 'n'):
                return False
        except Exception:
            pass
        return False

    def _native_retrieval_mode_for_provider(self, provider: str) -> str:
        """Return native retrieval mode: off|auto|on."""
        p = str(provider or '').strip().lower() or 'gemini'
        raw = None
        try:
            conf = (getattr(getattr(self, 'cfg_mgr', None), 'config', None) or getattr(cfg, 'config', {}) or {})
            provs = conf.get('providers') or {}
            pconf = provs.get(p) if isinstance(provs, dict) else {}
            if isinstance(pconf, dict):
                raw = pconf.get('native_retrieval')
            if raw is None:
                raw = conf.get('native_retrieval')
        except Exception:
            raw = None
        if raw is None:
            raw = os.environ.get('COMM_SCI_NATIVE_RETRIEVAL', 'auto')
        s = str(raw or '').strip().lower()
        if s in ('1', 'true', 'on', 'yes', 'enabled'):
            return 'on'
        if s in ('0', 'false', 'off', 'no', 'disabled'):
            return 'off'
        if s in ('auto',):
            return 'auto'
        return 'auto'

    def _provider_supports_native_retrieval(self, provider: str) -> bool:
        """Capability check for native retrieval/web-search tool wiring."""
        try:
            psvc = getattr(self, 'provider_service', None)
            if psvc is not None and hasattr(psvc, 'supports_native_retrieval'):
                return bool(psvc.supports_native_retrieval(provider))
        except Exception:
            pass
        pid = str(provider or '').strip().lower()
        if pid in ('openai', 'openai_compat'):
            pid = 'openrouter'
        if pid == 'hf':
            pid = 'huggingface'
        if pid != 'gemini':
            return False
        return bool(types is not None and hasattr(types, 'Tool') and (hasattr(types, 'GoogleSearch') or hasattr(types, 'GoogleSearchRetrieval')))

    def _build_native_tools_for_provider(self, provider: str):
        """Build provider-native tool list (best-effort, fail-soft)."""
        pid = str(provider or '').strip().lower() or 'gemini'
        mode = self._native_retrieval_mode_for_provider(pid)
        if mode == 'off':
            return []
        if not self._provider_supports_native_retrieval(pid):
            return []
        # Currently the wrapper has a concrete native-tool path for Gemini.
        if pid != 'gemini':
            return []
        if types is None:
            return []
        try:
            if hasattr(types, 'Tool') and hasattr(types, 'GoogleSearch'):
                return [types.Tool(google_search=types.GoogleSearch())]
        except Exception:
            pass
        try:
            if hasattr(types, 'Tool') and hasattr(types, 'GoogleSearchRetrieval'):
                return [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
        except Exception:
            pass
        return []

    def _recreate_chat_session(self, with_governance: bool, reason: str = "") -> bool:
        """Hard-reset the underlying model chat session.

        - with_governance=True: uses self._get_governed_system_instruction() (minimal + runtime state; canonical JSON is injected once as a pinned message)
        - with_governance=False: uses minimal/plain system instruction
        """
        if not getattr(self, "client", None):
            return False

        current_model = cfg_get_model()

        try:
            if with_governance:
                sys_instr = self._get_governed_system_instruction()
            else:
                sys_instr = self._get_plain_system_instruction()
        except Exception:
            sys_instr = gov.get_system_instruction() if with_governance else self._get_plain_system_instruction()

        try:
            gov.log(f"New chat session created ({'with' if with_governance else 'without'} ruleset) · reason: {reason or 'n/a'}")
        except Exception:
            pass

        config_kwargs = dict(
            system_instruction=sys_instr,
            temperature=0.0,
            top_p=0.1,
            candidate_count=1,
            max_output_tokens=65536,
        )
        provider = (self._active_provider() or 'gemini').strip().lower()
        native_tools = []
        try:
            native_tools = self._build_native_tools_for_provider(provider)
        except Exception:
            native_tools = []
        if native_tools:
            config_kwargs["tools"] = native_tools

        try:
            self.chat_session = self.client.chats.create(
                model=current_model,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            self._native_retrieval_active = bool(native_tools)
        except Exception as e:
            # Fail-soft fallback: if native tools are unsupported by the selected model/runtime,
            # retry without tools so the wrapper remains usable for all providers/models.
            if native_tools:
                try:
                    gov.log(f"Native retrieval tool setup failed ({type(e).__name__}). Retrying without native tools.")
                except Exception:
                    pass
                config_kwargs.pop("tools", None)
                self.chat_session = self.client.chats.create(
                    model=current_model,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                self._native_retrieval_active = False
            else:
                raise

        self.session_with_governance = bool(with_governance)
        # reset pinned-governance injection flags on session resets
        try:
            self._gov_pinned_sent = False
            self._gov_pinned_fp = ''
        except Exception:
            pass

        try:
            self.gov_state.user_turns = 0
            self.gov_state.anchor_force_next = False
            self.gov_state.last_anchor = ""
        except Exception:
            pass

        return True
    
    def _connect_api(self, reason: str = "connect"):
        """Connect provider backend.

        - Gemini: requires GOOGLE_API_KEY (via Comm-SCI-API-Keys.json or ENV)
        - OpenRouter: requires OPENROUTER_API_KEY (ENV by default; can fall back to api_key_plain in config)

        This method sets self.ready_status for the UI. For stateless providers we do not create a chat session.
        """
        # Prevent duplicate concurrent connects (startup + panel can trigger twice)
        if getattr(self, '_connect_inflight', False):
            return
        self._connect_inflight = True
        try:
            provider = (self._active_provider() or 'gemini').strip().lower()

            # Defensive: connect can be triggered more than once during startup/window lifecycle.
            try:
                sig_model = (cfg_get_model() or '').strip()
            except Exception:
                sig_model = ''
            try:
                sig_lang = str(UI_LANG or '').strip().lower()
            except Exception:
                sig_lang = ''
            sig = f"{provider}:{sig_model}:{sig_lang}"
            try:
                if getattr(self, "_last_connect_sig", None) == sig:
                    rs = getattr(self, "ready_status", {}) or {}
                    if bool(rs.get("status")) and (provider != "gemini" or getattr(self, "chat_session", None)):
                        return
                setattr(self, "_last_connect_sig", sig)
            except Exception:
                pass

            # Encrypted-key gate: require passphrase before provider connect/reconnect.
            try:
                gate = self._passphrase_requirement_for_provider(provider)
            except Exception:
                gate = {'required': False}
            if bool(gate.get('required')):
                gate_reason = str(gate.get('reason') or 'missing_passphrase')
                try:
                    self._queue_passphrase_request(provider, reason=reason or 'connect')
                except Exception:
                    pass
                self.ready_status = {
                    "status": False,
                    "msg": f"Passphrase required for encrypted key ({provider}).",
                }
                try:
                    self.log_event('passphrase_required', {'provider': provider, 'reason': gate_reason, 'scope': reason or 'connect'})
                except Exception:
                    pass
                try:
                    self._ui_add_system_message(
                        f"Encrypted API key detected for {provider}. Please enter passphrase to continue."
                    )
                except Exception:
                    pass
                try:
                    self._ui_refresh_panel()
                except Exception:
                    pass
                return
            else:
                try:
                    self._clear_passphrase_request(provider)
                except Exception:
                    pass


            # --- Stateless provider (OpenRouter) ---
            if provider in ('openrouter', 'huggingface', 'hf', 'openai', 'openai_compat'):
                pr = getattr(self, 'provider_router', None)
                psvc = getattr(self, 'provider_service', None)
                if psvc is not None and hasattr(psvc, 'router') and pr is not None:
                    psvc.router = pr
                client = None
                try:
                    if psvc is not None:
                        client = psvc.get_openai_client(provider)
                    elif pr is not None:
                        if provider in ('huggingface', 'hf') and hasattr(pr, 'build_huggingface_client'):
                            client = pr.build_huggingface_client()
                        elif hasattr(pr, 'build_openrouter_client'):
                            client = pr.build_openrouter_client()
                except Exception:
                    client = None

                # Validate key presence (best-effort)
                ok = False
                try:
                    ok = bool(getattr(client, 'api_key', '') or '')
                except Exception:
                    ok = False

                model = ''
                try:
                    model = (
                        psvc.get_provider_model(provider, fallback_model='')
                        if psvc is not None
                        else (getattr(cfg, 'get_provider_model', lambda _p: '')(provider) or '')
                    ).strip()
                except Exception:
                    model = ''
                if not model:
                    try:
                        model = (cfg_get_model() or '').strip()
                    except Exception:
                        model = ''

                # Do not touch Gemini client/session in this path.
                if ok:
                    try:
                        gov.log(f"Provider ready ({provider_name}) · model: {model or 'n/a'}")
                    except Exception:
                        pass
                    self.ready_status = {"status": True, "msg": f"Ready [openrouter:{model or 'n/a'}]", "filename": gov.current_filename}
                    return
                else:
                    # Do NOT hard-exit on missing OpenRouter key.
                    # Keep the UI alive and fall back to Gemini if possible.
                    try:
                        gov.log("OpenRouter API key missing. You can set it in the PANEL. Falling back to Gemini if available.")
                    except Exception:
                        pass
                    try:
                        setattr(self, "_openrouter_key_missing", True)
                    except Exception:
                        pass
                    # If OpenRouter was selected as active provider, switch to Gemini so the app can start.
                    try:
                        if hasattr(cfg, "set_active_provider"):
                            cfg.set_active_provider("gemini")
                    except Exception:
                        pass
                    try:
                        if getattr(self, "gov_state", None) is not None:
                            setattr(self.gov_state, "active_provider", "gemini")
                    except Exception:
                        pass
                    # Fall through to Gemini connect below (may still fail if Gemini key is missing).

            # --- Gemini provider (stateful chat_session) ---
            api_key = get_api_key()
            current_model = cfg_get_model()
            try:
                lang = self._answer_lang()
            except Exception:
                lang = str(getattr(cfg, 'get_answer_language', lambda: 'de')() or 'de').strip().lower() or "de"

            if api_key:
                try:
                    gov.log(f"Connecting model ({current_model}, language: {lang})...")
                    self.client = genai.Client(api_key=api_key)

                    # create initial chat session with active governance
                    self._recreate_chat_session(with_governance=True, reason="connect")
                    gov.log("Connected.")

                    self.ready_status = {
                        "status": True,
                        "msg": f"Ready [{current_model}] ({lang.upper()})",
                        "filename": gov.current_filename
                    }

                except Exception as e:
                    gov.log(f"API CRASH: {e}")
                    self.ready_status = {"status": False, "msg": f"API ERROR: {e}"}
            else:
                gov.log("API key missing.")
                self.ready_status = {"status": False, "msg": "API key missing (check JSON or ENV)!"}


        finally:
            self._connect_inflight = False
    def _auto_comm_start(self, reason="startup"):
        """Sendet deterministisch 'Comm Start' nach Connect/Reload (optional sichtbar als Systemmeldung)."""
        try:
            if not getattr(self, "chat", None):
                return
            # deterministischer State-Mirror
            self.gov_state.comm_active = True
            self.gov_state.sci_pending = False
            self.gov_state.sci_variant = ""
            # 'Comm Start' an das Modell senden (silent; wir zeigen nur eine Systemmeldung)
            _ = self.chat.send_message("Comm Start")
            if getattr(self, "main_win", None):
                self.main_win.evaluate_js(f"addMsg('sys', 'Auto: Comm Start ({reason}).')")
        except Exception as e:
            if getattr(self, "main_win", None):
                safe = str(e).replace("'", "'").replace('"', '\"')
                self.main_win.evaluate_js(f"addMsg('sys', 'Auto Comm Start failed: {safe}')")



    def is_ready(self):
        return getattr(self, 'ready_status', {"status": False, "msg": "Not connected."})

    def ui_qc_bar_enabled(self):
        """UI helper: show QC/CGI rating bar only when Comm-SCI is active."""
        try:
            return bool(getattr(self.gov_state, 'comm_active', False))
        except Exception:
            return False

    def _normalize_input_history_entries(self, entries, *, max_entries: int | None = None) -> list[str]:
        """Normalize persisted command-line history (trim, drop empties, dedupe adjacent)."""
        try:
            lim = int(max_entries if max_entries is not None else getattr(self, "input_history_max_entries", 200))
        except Exception:
            lim = 200
        lim = max(20, lim)
        out: list[str] = []
        if not isinstance(entries, list):
            return out
        for raw in entries:
            txt = str(raw or "").strip()
            if not txt:
                continue
            if out and out[-1] == txt:
                continue
            out.append(txt)
        if len(out) > lim:
            out = out[-lim:]
        return out

    def _load_input_history_entries(self) -> list[str]:
        """Best-effort load of input history from Logs/History/InputLineHistory.json."""
        payload = None
        try:
            st = getattr(self, "storage_service", None)
            if st is not None and hasattr(st, "read_json"):
                payload = st.read_json(INPUT_HISTORY_PATH)
            else:
                with open(INPUT_HISTORY_PATH, "r", encoding="utf-8") as f:
                    payload = json.load(f)
        except Exception:
            payload = None

        entries = []
        if isinstance(payload, dict):
            entries = payload.get("entries") or []
        elif isinstance(payload, list):
            entries = payload
        return self._normalize_input_history_entries(entries)

    def _save_input_history_entries(self, *, reason: str = "manual") -> bool:
        """Persist current in-memory input history to Logs/History/InputLineHistory.json."""
        entries = self._normalize_input_history_entries(getattr(self, "input_cmd_history", []) or [])
        try:
            self.input_cmd_history = list(entries)
        except Exception:
            pass
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "reason": str(reason or "manual"),
            "max_entries": int(getattr(self, "input_history_max_entries", 200) or 200),
            "entries": entries,
        }
        ok = False
        try:
            st = getattr(self, "storage_service", None)
            if st is not None and hasattr(st, "write_json"):
                ok = bool(st.write_json(INPUT_HISTORY_PATH, payload, indent=2, ensure_ascii=False))
            else:
                os.makedirs(HISTORY_LOG_DIR, exist_ok=True)
                with open(INPUT_HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                ok = True
        except Exception:
            ok = False
        return ok

    def get_input_history(self, max_entries: int = 200):
        """JS bridge: return normalized command-line history for the input box."""
        try:
            lim = int(max_entries or 200)
        except Exception:
            lim = 200
        lim = max(20, min(lim, 1000))
        entries = self._normalize_input_history_entries(getattr(self, "input_cmd_history", []) or [])
        if len(entries) > lim:
            entries = entries[-lim:]
        return {
            "ok": True,
            "entries": entries,
            "max_entries": int(getattr(self, "input_history_max_entries", 200) or 200),
        }

    def append_input_history(self, raw):
        """JS bridge: append one user input line to in-memory command history."""
        txt = str(raw or "").strip()
        if not txt:
            return {"ok": True, "added": False, "size": len(getattr(self, "input_cmd_history", []) or [])}
        cur = list(getattr(self, "input_cmd_history", []) or [])
        if cur and str(cur[-1] or "") == txt:
            return {"ok": True, "added": False, "size": len(cur)}
        cur.append(txt)
        cur = self._normalize_input_history_entries(cur)
        self.input_cmd_history = cur
        return {"ok": True, "added": True, "size": len(cur)}

    def _cgi_ui_texts(self, *, lang: str = "") -> dict:
        use_en = str(lang or self._answer_lang() or "").strip().lower().startswith("en")
        if use_en:
            return {
                "saved": "CGI saved: C={c} · I={i} · E={e}",
                "applied": "CGI applied (one-shot): C={c} · I={i} · E={e}",
                "no_prompt": "No previous content question available for repeat.",
                "invalid": "Please provide values from 0 to 3 for all three CGI criteria.",
                "repeat_failed": "CGI repeat could not be executed.",
            }
        return {
            "saved": "CGI gespeichert: K={c} · E={i} · F={e}",
            "applied": "CGI angewendet (one-shot): K={c} · E={i} · F={e}",
            "no_prompt": "Keine vorherige Inhaltsfrage für die Wiederholung vorhanden.",
            "invalid": "Bitte für alle drei CGI-Kriterien Werte von 0 bis 3 angeben.",
            "repeat_failed": "CGI-Wiederholung konnte nicht ausgeführt werden.",
        }

    def _parse_cgi_triplet(self, triplet: str):
        m = re.fullmatch(r"\s*([0-3])\s*,\s*([0-3])\s*,\s*([0-3])\s*", str(triplet or ""))
        if not m:
            return None
        try:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        except Exception:
            return None

    def _qc_current_value(self, key: str, default: int = 2) -> int:
        try:
            qc = getattr(self.gov_state, "last_qc", {}) or {}
            if not isinstance(qc, dict):
                return int(default)
            key_l = str(key or "").strip().lower()
            for cand in (key_l, key_l.capitalize(), key_l.upper()):
                if cand in qc:
                    return int(qc.get(cand))
        except Exception:
            pass
        return int(default)

    def _build_cgi_one_shot_constraints(self, triplet: str, *, lang: str = "") -> list[str]:
        vals = self._parse_cgi_triplet(triplet)
        if not vals:
            return []
        c, i, e = vals
        use_en = str(lang or self._answer_lang() or "").strip().lower().startswith("en")

        lines = []
        if use_en:
            lines.append("- Apply only to this next answer.")
            lines.append("- Keep hard contracts unchanged (language contract, uncertainty markers, verification gates, self-debunking, QC footer).")
        else:
            lines.append("- Nur fuer diese naechste Antwort anwenden.")
            lines.append("- Harte Vertraege unveraendert einhalten (Sprachvertrag, Unsicherheitsmarker, Verification-Gates, Self-Debunking, QC-Footer).")

        if c <= 1 or i <= 1 or e <= 1:
            if use_en:
                lines.append("- Treat this as a rewrite, not a paraphrase: use a different opening sentence and a different section order.")
                lines.append("- Include at least 3 concrete additions that were not present in the previous answer.")
            else:
                lines.append("- Behandle dies als Ueberarbeitung, nicht als Paraphrase: nutze einen anderen Einstiegssatz und eine andere Abschnittsreihenfolge.")
                lines.append("- Fuege mindestens 3 konkrete Ergaenzungen ein, die in der vorherigen Antwort noch nicht enthalten waren.")

        if c <= 1:
            cur_clarity = self._qc_current_value("clarity", default=2)
            if use_en:
                if cur_clarity >= 3:
                    lines.append("- Clarity QC already at 3 (numeric cap): enforce qualitative rewrite (clearer structure, simpler sentence flow).")
                else:
                    lines.append("- QC direction for this turn: Clarity >= 3.")
                lines.append("- Add a short summary first, then at least 4 clearly titled sections.")
                lines.append("- Define key terms and include at least 1 concrete example.")
            else:
                if cur_clarity >= 3:
                    lines.append("- Clarity-QC ist bereits 3 (numerisches Limit): erzwinge qualitative Ueberarbeitung (klarere Struktur, einfacherer Satzfluss).")
                else:
                    lines.append("- QC-Richtung fuer diesen Turn: Clarity >= 3.")
                lines.append("- Starte mit einer Kurzfassung, danach mindestens 4 klar betitelte Abschnitte.")
                lines.append("- Definiere Schluesselbegriffe und fuege mindestens 1 konkretes Beispiel ein.")

        if i <= 1:
            cur_evidence = self._qc_current_value("evidence", default=2)
            cur_consistency = self._qc_current_value("consistency", default=2)
            if use_en:
                if cur_evidence >= 3 and cur_consistency >= 3:
                    lines.append("- Evidence/Consistency already at QC 3: enforce qualitative insight delta instead of numeric increase.")
                else:
                    lines.append("- QC direction for this turn: Evidence >= 3 and Consistency >= 3.")
                lines.append("- Add at least 2 new concrete insights (not rephrased duplicates).")
                lines.append("- Add 1 counterpoint and 1 explicit verification route.")
            else:
                if cur_evidence >= 3 and cur_consistency >= 3:
                    lines.append("- Evidence/Consistency sind bereits bei QC 3: erzwinge qualitative Erkenntnis-Differenz statt numerischer Erhoehung.")
                else:
                    lines.append("- QC-Richtung fuer diesen Turn: Evidence >= 3 und Consistency >= 3.")
                lines.append("- Fuege mindestens 2 neue konkrete Erkenntnisse ein (keine reinen Umformulierungen).")
                lines.append("- Fuege 1 Gegenperspektive und 1 explizite Verification-Route hinzu.")

        if e <= 1:
            cur_brevity = self._qc_current_value("brevity", default=2)
            if use_en:
                if cur_brevity <= 0:
                    lines.append("- Brevity already at depth cap (0): enforce qualitative depth expansion.")
                else:
                    lines.append("- QC direction for this turn: Brevity -> deeper style (towards 0), not shorter.")
                lines.append("- Prefer depth over brevity for this turn: target about >=300 words unless the user explicitly requested concise output.")
            else:
                if cur_brevity <= 0:
                    lines.append("- Brevity ist bereits am Tiefen-Limit (0): erzwinge qualitative Tiefen-Erweiterung.")
                else:
                    lines.append("- QC-Richtung fuer diesen Turn: Brevity -> tiefere Ausfuehrung (Richtung 0), nicht kuerzer.")
                lines.append("- Fuer diesen Turn Tiefe vor Kuerze: Ziel grob >=300 Woerter, ausser der Nutzer fordert explizit eine kurze Antwort.")

        if use_en:
            lines.append("- Ensure the revised answer differs substantially from the previous answer (structure + concrete detail delta).")
        else:
            lines.append("- Stelle sicher, dass die ueberarbeitete Antwort deutlich von der vorherigen abweicht (Struktur + konkrete Detaildifferenz).")
        return lines

    def submit_cgi_feedback(self, clarity, insight, efficiency, mode: str = "repeat"):
        """Record CGI feedback via UI widgets; optionally re-answer the last content question."""
        texts = self._cgi_ui_texts()
        try:
            c = int(str(clarity).strip())
            i = int(str(insight).strip())
            e = int(str(efficiency).strip())
        except Exception:
            return {"ok": False, "error": texts["invalid"]}

        if any(v < 0 or v > 3 for v in (c, i, e)):
            return {"ok": False, "error": texts["invalid"]}

        triplet = f"{c},{i},{e}"
        try:
            self.gov_state.last_user_feedback_triplet = triplet
            self.gov_state.cgi_feedback_pending_for_model = True
        except Exception:
            pass

        mode_s = str(mode or "save").strip().lower()
        try:
            self.log_event("cgi_feedback", {"value": triplet, "kind": "widget_triplet", "mode": mode_s})
        except Exception:
            pass

        if mode_s not in {"repeat", "rerun", "reanswer"}:
            return {
                "ok": True,
                "repeated": False,
                "saved_triplet": triplet,
                "message": texts["saved"].format(c=c, i=i, e=e),
            }

        last_prompt = str(getattr(self, "_last_content_user_prompt", "") or "").strip()
        if not last_prompt:
            return {"ok": False, "error": texts["no_prompt"]}

        try:
            res = self.ask(last_prompt)
        except Exception:
            return {"ok": False, "error": texts["repeat_failed"]}

        if isinstance(res, dict):
            payload = dict(res)
        else:
            payload = {"html": str(res or ""), "csc": None}
        payload.setdefault("cgi_bar", bool(payload.get("cgi_bar", False)))
        payload.setdefault("answer_lang", self._answer_lang())

        return {
            "ok": True,
            "repeated": True,
            "saved_triplet": triplet,
            "message": texts["applied"].format(c=c, i=i, e=e),
            "response": payload,
        }

    def load_rule_file(self):
        """Öffnet Dateidialog und lädt neues JSON (robust gegen versehentliches Laden von Comm-SCI-Config.json).
        Wichtig: Bei ungültiger Auswahl bleibt das aktuell aktive Ruleset unverändert."""
        # PyWebView: neuer Enum (FileDialog.OPEN), fallback auf OPEN_DIALOG (alt)
        dlg_open = None
        try:
            dlg_open = webview.FileDialog.OPEN  # type: ignore[attr-defined]
        except Exception:
            dlg_open = getattr(webview, "OPEN_DIALOG", None)

        # Filter: macOS ignoriert Filter teils → wir validieren zusätzlich deterministisch.
        # Filter: pywebview erwartet Strings im Format "Description (*.ext;*.ext)".
        # Viele Backends ignorieren Filter ohnehin → wir validieren danach deterministisch (Schema-Guard).
        file_types = (
            'JSON Files (*.json)',
        )

        start_dir = os.path.dirname(gov.current_filename) if getattr(gov, 'current_filename', None) else os.path.dirname(os.path.abspath(__file__))

        # Einmaliger Retry-Loop: verhindert "falsche Datei gewählt" ohne den Nutzer zu nerven.
        for attempt in range(2):
            try:
                result = self.main_win.create_file_dialog(
                    dlg_open,
                    allow_multiple=False,
                    directory=start_dir,
                    file_types=file_types
                )
            except ValueError:
                # Manche pywebview-Versionen sind sehr strikt beim Filter-Format -> notfalls ohne Filter öffnen.
                result = self.main_win.create_file_dialog(
                    dlg_open,
                    allow_multiple=False,
                    directory=start_dir,
                )

            if not result or len(result) == 0:
                return  # cancelled

            new_file = result[0]
            base = os.path.basename(new_file)

            # Sichtbares Echo, damit klar ist, WAS wirklich ausgewählt wurde.
            self._ui_add_system_message(f"Selected: {base}")

            # Harte Sperre: Config-Datei ist KEIN Ruleset.
            if base.lower() == "comm-sci-config.json" or base.lower().endswith("-config.json") or base.lower().endswith("config.json"):
                self._ui_add_system_message(
                    "JSON ERROR: You selected the configuration file. Please choose a ruleset file (e.g., Comm-SCI-v20.2.1.json)."
                )
                start_dir = os.path.dirname(new_file)
                continue  # retry once

            # Versuch laden (Schema-Guard im GovernanceManager)
            success = gov.load_file(new_file)
            if not success:
                self._ui_add_system_message("Ruleset NOT loaded (invalid Comm-SCI ruleset).")
                start_dir = os.path.dirname(new_file)
                if attempt == 0:
                    continue
                return

            # Ab hier: erfolgreich geladen → Session deterministisch neu setzen
            self.gov_state = _init_state_from_rules()

            # 1) API reconnecten (damit System Instructions neu gesetzt werden)
            self._ui_add_system_message(f"Loading new ruleset: {os.path.basename(gov.current_filename)}...")
            self._connect_api(reason="config_reload")
            self._auto_comm_start('rules-reload')

            # 2) UI im Chatfenster updaten (Dateiname oben)
            self._ui_update_rule_file(os.path.basename(gov.current_filename))

            # 3) Panel robust neu aufbauen: altes Panel zerstören & neu erzeugen
            self._rebuild_panel(reason='rules-reload')

            self._ui_add_system_message("Ruleset loaded and panel updated.")
            return



    def _render_sci_trace_as_html_runtime(self, text_in: str) -> str:
        """Repair + render SCI Trace deterministically for display.

        Goals:
        - If the model emits an empty 'SCI Trace' (only step names), rebuild it from step-labeled content
          elsewhere in the response.
        - Ensure all required steps for the active SCI variant are shown, and never as empty bullets.
        - Avoid duplicate content: extracted step sections are removed from the final-answer body.
        """
        try:
            if not text_in or 'SCI Trace' not in text_in:
                return text_in

            variant = (getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip().upper()
            sci_active = bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False))
            if not sci_active or not variant:
                return text_in

            # Determine required steps from ruleset mapping
            try:
                _vdef, steps, _maps_to = self._sci_variant_def(variant)
            except Exception:
                steps = []
            required_steps = [str(s) for s in (steps or []) if str(s).strip()]
            if not required_steps:
                return text_in

            lines = text_in.splitlines()

            def _is_sci_trace_heading_line(plain_line: str) -> bool:
                p = re.sub(r"\s+", " ", str(plain_line or "")).strip()
                p = p.strip("*").strip()
                return bool(
                    re.match(
                        r"^(?:#+\s*)?(?:\d+[\.\)]\s*)?SCI\s+Trace\b",
                        p,
                        flags=re.IGNORECASE,
                    )
                )

            # Find the earliest SCI Trace marker line (also accepts forms like
            # "4. SCI Trace (Variante A: Standard)").
            sci_idx = None
            for i, ln in enumerate(lines):
                ln_plain = re.sub(r"<[^>]+>", "", ln).strip()
                ln_plain = ln_plain.strip().strip("*").strip()
                if _is_sci_trace_heading_line(ln_plain):
                    sci_idx = i
                    break
            if sci_idx is None:
                return text_in

            # Identify the end of the immediate list after 'SCI Trace' (bullets/numbering only)
            list_pat = re.compile(r"^\s*(?:[*+-]|•|\d+\.)\s+")
            k = sci_idx + 1
            while k < len(lines) and (not lines[k].strip() or list_pat.match(re.sub(r"<[^>]+>", "", lines[k]))):
                k += 1
            trace_list_end = k  # exclusive

            # If this immediate list already contains SCI step headers (e.g. "• Plan: ..."),
            # keep it as content; otherwise it is typically an empty placeholder list.
            try:
                _has_step_header = False
                for _ln in lines[sci_idx + 1:trace_list_end]:
                    _plain = re.sub(r"<[^>]+>", "", _ln or "")
                    _step, _rest = match_required_sci_step_header(_plain, required_steps)
                    if _step:
                        _has_step_header = True
                        break
                if _has_step_header:
                    trace_list_end = sci_idx + 1
            except Exception:
                pass

            # Build a working copy without the original (often empty) trace list block
            pre = lines[:sci_idx]
            rest = lines[trace_list_end:]

            # Split off trailing governance blocks we must keep in place (Self-Debunking, QC-Matrix)
            boundary_pat = re.compile(r"^\s*(Self-?Debunking\s*:|QC-?Matrix\s*:)", re.IGNORECASE)
            tail_start = None
            for i, ln in enumerate(rest):
                plain_ln = re.sub(r"<[^>]+>", "", ln or "")
                if (
                    boundary_pat.match(plain_ln)
                    or re.search(r"(?i)\bself[- ]?debunking\b", plain_ln)
                    or re.search(r"(?i)\bQC-?Matrix\s*:", plain_ln)
                    or re.search(r"(?is)class=(?:\"|')[^\"']*self-debunking[^\"']*(?:\"|')", ln or "")
                ):
                    tail_start = i
                    break
            if tail_start is None:
                main = rest
                tail = []
            else:
                main = rest[:tail_start]
                tail = rest[tail_start:]

            # Normalize basic HTML bold headers that may leak into raw text
            def _strip_basic_tags(s: str) -> str:
                s = re.sub(r"</?(strong|b)>", "", s, flags=re.IGNORECASE)
                s = re.sub(r"</?p>", "", s, flags=re.IGNORECASE)
                return s

            blocks = {}
            out_main = []
            cur_step = None
            buf = []
            last_step = required_steps[-1] if required_steps else ""
            after_last_step_break = False

            def _looks_like_final_answer_start(plain_line: str, raw_line: str = "") -> bool:
                s = str(plain_line or "").strip()
                if not s:
                    return False
                if re.match(r"^(?:[*+-]|•|\d+\.)\s+", s):
                    return False
                if re.match(r"(?i)^(?:Self[- ]?Debunking|Selbst[- ]?Debunking|QC(?:-Matrix)?|Final\s+Answer)\b", s):
                    return False
                # Evidence/uncertainty markers in raw HTML are a strong indicator that
                # we have entered the narrative final-answer body.
                raw = str(raw_line or "")
                if ("signal-dot-marker" in raw) or ("uncertainty-inline-marker" in raw):
                    return True
                words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", s)
                return (len(words) >= 8) or (len(s) >= 80)

            def flush():
                nonlocal cur_step, buf
                if cur_step is None:
                    return
                cleaned = []
                for x in buf:
                    cleaned.append(re.sub(r"^\s*\d+\.\s+", "", x))
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                # Keep the first non-empty capture for a step when duplicated SCI Trace
                # sections leak into the same answer.
                if (cur_step not in blocks) or (not blocks.get(cur_step)):
                    blocks[cur_step] = cleaned
                cur_step = None
                buf = []

            recognized_steps = 0
            for ln in main:
                ln2 = _strip_basic_tags(re.sub(r"<[^>]+>", "", ln))
                if _is_sci_trace_heading_line(ln2):
                    # Drop duplicate SCI Trace headings from the body once we rebuild.
                    continue
                step_name, rest = match_required_sci_step_header(ln2, required_steps)
                if step_name:
                    flush()
                    cur_step = step_name
                    recognized_steps += 1
                    after_last_step_break = False
                    if rest:
                        buf.append(rest)
                    continue
                if cur_step is not None:
                    if cur_step == last_step:
                        if not ln2.strip():
                            after_last_step_break = True
                            buf.append(ln2)
                            continue
                        if after_last_step_break and _looks_like_final_answer_start(ln2, ln):
                            # Split here: remaining narrative belongs to final answer,
                            # not to the last SCI step.
                            flush()
                            after_last_step_break = False
                            out_main.append(ln)
                            continue
                    buf.append(ln2)
                else:
                    out_main.append(ln)

            flush()            # If we didn't recognize any step content, do not fabricate trace items.
            if recognized_steps == 0:
                return text_in

            # Render only steps that have real content; never emit empty steps.
            missing = []
            for s in required_steps:
                if s in missing:
                    continue
                if not blocks.get(s):
                    missing.append(s)


            # Optional deterministic alert if step content is missing
            alert_html = ""
            if missing:
                safe = ", ".join([html.escape(x) for x in missing])
                alert_html = (
                    "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                    "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                    "<b>CONTROL LAYER ALERT (SCI)</b><br>"
                    "Missing SCI Trace step content for: " + safe +
                    "</div>"
                )

            # Render deterministic HTML trace block
            html_parts = [
                "<!-- SCI Trace: -->",
                "<div class='sci-trace' style='margin:10px 0; padding:10px; border:1px solid #ddd; border-radius:12px;'>",
                "<div style='font-weight:700; margin-bottom:6px;'>SCI Trace</div>",
                "<ol style='margin:0 0 0 22px; padding:0;'>",
            ]
            for s in required_steps:
                if s in missing:
                    continue
                html_parts.append("<li style='margin:4px 0 10px 0;'>")
                html_parts.append(f"<div style='font-weight:700; margin:0 0 4px 0;'>{html.escape(s)}:</div>")
                for ln in (blocks.get(s) or []):
                    t = (ln or "").rstrip("\n")
                    if not t.strip():
                        html_parts.append("<div style='height:6px'></div>")
                        continue
                    m2 = re.match(r"^\s*([*+-]|•)\s+(.*)$", t)
                    if m2:
                        html_parts.append(f"<div style='margin-left:14px;'>• {html.escape(m2.group(2).strip())}</div>")
                    else:
                        html_parts.append(f"<div>{html.escape(t.strip())}</div>")
                html_parts.append("</li>")
            html_parts.extend(["</ol>", "</div>"])

            out_lines = []
            out_lines.extend(pre)
            if alert_html:
                out_lines.append(alert_html)
            out_lines.append("\n".join(html_parts))
            out_lines.extend(out_main)
            out_lines.extend(tail)
            return "\n".join(out_lines)
        except Exception:
            return text_in


    def _apply_csc_strict(self, raw_response: str, *, user_raw: str, is_command: bool):
        """Wrapper-enforced CSC (strict) with Full Rendering (Ported from Fix7c5-Plus)."""
        
        ctx = _build_route_ctx(self, user_raw=user_raw, is_command=is_command)
        # --- Nested Helper: Color Spans (Logic from Fix7c5-Plus) ---
        def _apply_color_spans_local(text):
            if not text: return text
            # Fallback falls global _EVIDENCE_COLOR fehlt
            ev_colors = globals().get('_EVIDENCE_COLOR', {
                "GREEN": "#137333", "YELLOW": "#f9ab00", "RED": "#d93025", "GRAY": "#5f6368"
            })
            def repl(m):
                tag = m.group("tag")
                suffix = m.group("suffix") or ""
                emoji = m.group("emoji") or ""
                color = ev_colors.get(tag, "#616161")
                token = f"[{tag}{suffix}]"
                if emoji: token = f"{token} {emoji}"
                return f'<span style="color:{color}; font-weight:600;">{token}</span>'
            
            # Regex wie in der alten Version
            pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z0-9]+)*)\]\s*(?P<emoji>[🟢🟡🔴⚪⚪️])?")
            return pat.sub(repl, text)

        # --- Nested Helper: Image Embedding ---
        def _auto_embed_image_urls(text):
            if not text or 'http' not in text: return text
            parts = text.split('```')
            url_re = re.compile(r"(https?://[^\s<>()\]\[]+?\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s<>()\]\[]*)?(?:/[^\s<>()\]\[]+)*)", re.IGNORECASE)
            
            for i in range(0, len(parts), 2):
                seg = parts[i]
                def repl_img(m):
                    url = m.group(0)
                    safe_url = html.escape(url, quote=True)
                    img = f'\n\n<img src="{safe_url}" style="max-width:100%; height:auto; border-radius:10px; margin:6px 0;" loading="lazy" />\n'
                    return url + img
                parts[i] = url_re.sub(repl_img, seg)
            return '```'.join(parts)

        try:
            # 1. Command? -> Render via v192 pipeline if available (deterministic-ish), else legacy Markdown
            if is_command:
                raw_response = unwrap_accidental_full_text_codefence(raw_response or "")
                raw_response = normalize_known_markdown_control_headings(raw_response or "")
                if _rendering_pipeline_v192 is not None:
                    try:
                        rctx = _rendering_pipeline_v192.RenderContext(
                            ui_lang=(self._lang() or 'en'),
                            color=str(ctx.get('color', 'off') or 'off'),
                            is_command=True,
                            comm_active=bool(getattr(self.gov_state, 'comm_active', False)),
                            strict=False,
                        )
                        return _rendering_pipeline_v192.render_llm_text_to_html(raw_response or "", rctx), None
                    except Exception:
                        pass

                # Legacy fallback (kept for safety)
                if ctx.get('color', 'off') == 'on':
                    raw_response = apply_color_spans(raw_response, enabled=True)
                _h = markdown.markdown(raw_response, extensions=['extra', 'codehilite'])
                return sanitize_html(_h), None

            # 2. Comm Inactive? -> Render via v192 pipeline if available (comm-inactive Markdown render path), else legacy Markdown
            if not getattr(self.gov_state, 'comm_active', False):
                html_out = ""
                try:
                    raw_response = unwrap_accidental_full_text_codefence(raw_response or "")
                    raw_response = normalize_known_markdown_control_headings(raw_response or "")
                    try:
                        raw_response = strip_governance_scaffolding_when_comm_inactive(raw_response or "")
                    except Exception:
                        pass
                    if _rendering_pipeline_v192 is not None:
                        try:
                            rctx = _rendering_pipeline_v192.RenderContext(
                                ui_lang=(self._lang() or 'en'),
                                color=str(ctx.get('color', 'off') or 'off'),
                                is_command=False,
                                comm_active=False,
                                strict=False,
                            )
                            html_out = _rendering_pipeline_v192.render_llm_text_to_html(raw_response or "", rctx)
                        except Exception:
                            html_out = ""
            
                    if not html_out:
                        # Legacy fallback (kept for safety)
                        _raw = raw_response
                        if ctx.get('color', 'off') == 'on':
                            _raw = apply_color_spans(_raw, enabled=True)
                        _h = markdown.markdown(_raw, extensions=['extra', 'codehilite'])
                        html_out = sanitize_html(_h)
                except Exception:
                    # Ultimate fallback: show escaped raw in <pre> (never crash).
                    try:
                        html_out = "<pre>" + html.escape(str(raw_response or "")) + "</pre>"
                    except Exception:
                        html_out = "<pre></pre>"
            
                # Provide deterministic normalization meta even in comm-inactive path (tests rely on it).
                try:
                    _qc_pat = re.compile(r"(?im)^\s*QC(?:-Matrix)?\s*:")
                    _raw_qc_count = len(_qc_pat.findall(str(raw_response or "")))
                    _html_qc_count = len(_qc_pat.findall(re.sub(r"<[^>]+>", "", str(html_out or ""))))
                    _sd_boxed = ("self-debunking" in str(html_out or "").lower()) or ("selbst-debunking" in str(html_out or "").lower())
                    _sd_numbered = detect_self_debunking_numbered_html(str(html_out or ""))
            
                    def _looks_like_rendered_html(h: str) -> bool:
                        if not h:
                            return False
                        hl = h.lstrip().lower()
                        if hl.startswith("<pre") and "&lt;" in hl:
                            return False
                        if h.count("&lt;") > 10 and ("<p" not in hl and "<div" not in hl and "<ol" not in hl):
                            return False
                        return any(t in hl for t in ("<p", "<div", "<ol", "<ul", "<table", "<pre", "<blockquote"))
            
                    render_ok = _looks_like_rendered_html(str(html_out or ""))
                    meta = {
                        "normalization": {
                            "qc_footer_raw_count": _raw_qc_count,
                            "qc_footer_html_count": _html_qc_count,
                            "qc_footer_deduped": (_raw_qc_count > 1 and _html_qc_count == 1),
                            "self_debunking_boxed": bool(_sd_boxed),
                            "self_debunking_numbered": bool(_sd_numbered),
                            "render_ok": bool(render_ok),
                            "render_fallback": (not bool(render_ok)),
                        }
                    }
                except Exception:
                    meta = {"normalization": {"render_ok": True, "render_fallback": False}}
            
                return html_out, meta
            # 3. Refiner Logic (Erhalten für csc_meta)
            refiner = getattr(gov, 'csc_refiner', None)
            csc_meta = None
            
            # Trigger-Analyse (wie bisher)
            prof = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
            overlay = getattr(self.gov_state, 'overlay', '') or ''
            mult = 2 if overlay == "Explore" else 1
            txt_l = (raw_response or "").lower()
            uncertainty_U4 = bool(re.search(r"\bU[4-6]\b", raw_response or ""))
            web_check = bool(re.search(r"\bweb\s*[- ]\s*check\b", txt_l))
            strong_claim = any(x in txt_l for x in ["immer", "niemals", "definitiv", "guarantee", "prove"])
            
            if refiner:
                dec = refiner.decide(
                    comm_active=True, active_profile=prof, input_raw=user_raw or "",
                    uncertainty_U4_active=uncertainty_U4, web_check_hook_active=web_check,
                    strong_claim_detected=strong_claim, neutrality_delta_negative=False,
                    threshold_multiplier=mult
                )
                
                # Metadata bauen (für Badge)
                if dec.apply:
                    ans_lang = self._answer_lang()
                    msg = CSC_WARNING_TEXT
                    thr_fs = int(getattr(refiner, '_refine_params', {}).get('threshold_f_score', 8) or 8)
                    thr_tok = int(getattr(refiner, '_refine_params', {}).get('min_token_count', 80) or 80)
                    gov_min_tok = int(getattr(refiner, '_gov_params', {}).get('min_token_count_governance', 40) or 40)
                    if mult != 1:
                        thr_fs *= mult
                        thr_tok *= mult
                        gov_min_tok *= mult
                    csc_meta = {
                        "applied": True, "message": msg,
                        "trigger": str(getattr(dec, 'trigger_source', '')),
                        "mode": str(getattr(dec, 'mode', '')),
                        "governance_triggered": bool(getattr(dec, 'governance_triggered', False)),
                        "token_count": int(getattr(dec, 'token_count', 0)),
                        "f_score": int(getattr(dec, 'f_score', 0)),
                        "score_tooltip": _csc_score_tooltip_text(
                            lang=ans_lang,
                            f_score=int(getattr(dec, 'f_score', 0)),
                            token_count=int(getattr(dec, 'token_count', 0)),
                        ),
                        "thresholds_tooltip": _csc_thresholds_tooltip_text(
                            lang=ans_lang,
                            thr_fs=thr_fs,
                            thr_tok=thr_tok,
                            gov_min_tok=gov_min_tok,
                            mult=mult,
                        ),
                        "overlay": overlay,
                        "profile": str(prof or ''),
                        "threshold_multiplier": int(mult or 1),
                        "threshold_f_score": int(thr_fs),
                        "threshold_token_count": int(thr_tok),
                        "min_token_count": int(thr_tok),
                        "min_token_count_governance": int(gov_min_tok),
                        "schema_version": "1.0",
                    }

            # 4. Alerts generieren (Wiederhergestellt aus alter Version)
            alerts = []
            csc_visible_marker = ""

            # CSC transparency marker: render deterministically in the visible header when the
            # ruleset requires visibility, instead of emitting a noisy false-positive alert based
            # on raw model text (the marker is prompt-side and may be stripped during normalization).
            try:
                if csc_meta and csc_meta.get('applied'):
                    cl = (getattr(gov, 'data', {}) or {}).get('control_layer', {}) or {}
                    bridge = (cl.get('components', {}) or {}).get('csc_trigger_bridge', {}) or {}
                    constraints = (bridge.get('constraints', {}) or {})
                    tm = (constraints.get('transparency_marker', {}) or {})
                    marker = str(tm.get('marker', getattr(refiner, 'marker', '') or '') or '').strip()
                    marker_enabled = bool(tm.get('enabled', True))
                    marker_visibility = str(tm.get('visibility', '') or '').strip().lower()
                    if marker_enabled and marker and marker_visibility == 'always_visible_if_applied':
                        csc_visible_marker = marker
            except Exception:
                pass
            
            # VR Gate Check
            try:
                vr_msg = gov.check_verification_route_gate(raw_response)
                if vr_msg: alerts.append(("Verification Route Gate", vr_msg))
            except: pass
            
            # QC Matrix Check (deterministic delta enforcement before alerting)
            try:
                # IMPORTANT: use the same profile as shown in the header/rendering.
                # This avoids false "expected Δ..." alerts if the runtime state changes mid-turn.
                _prof = (prof or getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard')
                _ovr_raw = getattr(self.gov_state, 'qc_overrides', {}) or {}
                _ovr = gov.normalize_qc_overrides(_ovr_raw)
                _corr = gov.get_effective_qc_corridor(_prof, _ovr)
                
                raw_response = _strip_basic_html_for_enforcement(raw_response)
                raw_response = ensure_qc_footer_present(raw_response, gov, _prof, _ovr)
                enforced_txt = enforce_qc_footer_deltas(raw_response, _corr, _prof)
                enforced_txt = ensure_qc_footer_is_last(enforced_txt)
                cur_qc, rep_delta = gov.parse_qc_footer(enforced_txt)
                if cur_qc:
                    exp_delta = gov.expected_qc_deltas(_prof, cur_qc, overrides=_ovr)
                    if rep_delta:
                        mism = [f"{k}: expected Δ{v}, got Δ{rep_delta[k]}" for k, v in exp_delta.items() if rep_delta.get(k) != v]
                        if mism:
                            alerts.append(("QC-Matrix", "Delta mismatch: " + "; ".join(mism)))
                    else:
                        alerts.append(("QC-Matrix", "QC detected but no deltas found."))
            except Exception:
                pass

            
            # Cross-version leak guard alerts
            try:
                hits = list(getattr(self.gov_state, 'cross_version_guard_hits', []) or [])
                active_v = str(getattr(self.gov_state, 'active_ruleset_version', '') or str((self.gov.data or {}).get('version', '') or '')).strip()
                if hits:
                    alerts.append(("Cross-Version Guard", f"Ignored foreign version token(s) in user input (active {active_v})."))
            except Exception:
                pass

            # Render Alerts HTML
            alert_html = ""
            if alerts:
                items = "".join([f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>" for k,v in alerts])
                alert_html = (
                    "<div style='border:1px solid #b00; background:#fff5f5; padding:10px; "
                    "border-radius:10px; margin:8px 0;'><b>CONTROL LAYER ALERTS (Python)</b>"
                    f"<ul style='margin:6px 0 0 18px; padding:0;'>{items}</ul></div>"
                )

            # 5. Header generieren (Manuell erzwingen wie in alter Version)
            header = ""
            try:
                ver = gov.data.get("version", "")
                sysname = gov.data.get("system_name", "Comm-SCI-Control")
                sci = getattr(self.gov_state, "sci_variant", "") or ""
                color = getattr(self.gov_state, "color", "off")
                # Falls Profil Sandbox/Briefing -> Color off im Header anzeigen
                disp_color = "off" if prof in {"Sandbox", "Briefing"} else color
                
                header = (
                    f"Active profile: {prof} · SCI: {sci or 'off'} · Overlay: {overlay or 'off'} · "
                    f"Control Layer: on · QC: on · CGI: on · Color: {disp_color}"
                )

                # Dynamic one-shot marker (canonical JSON requires a visible marker)
                try:
                    if bool(getattr(self.gov_state, 'dynamic_one_shot_active', False)):
                        header += " · Dynamic: one-shot (active)"
                except Exception:
                    pass
                try:
                    if csc_visible_marker and (csc_visible_marker not in header):
                        header += f" · {csc_visible_marker}"
                except Exception:
                    pass
            except: header = ""

            # 6. Finales Assembly (Rendering Pipeline)
            # Apply deterministic QC delta enforcement before any further rendering.
            # This keeps the QC footer stable even if the model's deltas drift.
            try:
                _ovr_raw = getattr(self.gov_state, 'qc_overrides', {}) or {}
                _ovr = gov.normalize_qc_overrides(_ovr_raw)
                corr = gov.get_effective_qc_corridor(prof, _ovr)
                
                raw_response = _strip_basic_html_for_enforcement(raw_response)
                raw_response = ensure_qc_footer_present(raw_response, gov, prof, _ovr)
                raw_for_render = enforce_qc_footer_deltas(raw_response, corr, prof)
                raw_for_render = ensure_qc_footer_is_last(raw_for_render)
            except Exception:
                raw_for_render = raw_response

            # Persist last observed QC + Python-computed deltas for dynamic one-shot prompting
            try:
                cur_qc, _rep = gov.parse_qc_footer(raw_for_render)
                if cur_qc:
                    exp_delta = gov.expected_qc_deltas(prof, cur_qc, overrides=getattr(self.gov_state, "qc_overrides", {}))
                    self.gov_state.last_qc = dict(cur_qc)
                    self.gov_state.last_qc_deltas = dict(exp_delta or {})
            except Exception:
                pass

            # Normalize Evidence-Linker provenance formatting (without inventing provenance).
            try:
                raw_for_render = normalize_evidence_tags(raw_for_render)
            except Exception:
                pass

                        # Strip erroneous SCI menu echoes from normal answers (menu is command-only).
            try:
                if (not is_command) and (not bool(ctx.get('sci_pending'))):
                    raw_for_render = strip_sci_menu_from_answer(raw_for_render)
            except Exception:
                pass

            # If SCI is off, remove leaked inline "SCI Trace: ..." lines from model output.
            try:
                raw_for_render = strip_sci_trace_line_when_inactive(
                    raw_for_render,
                    sci_active=bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False)),
                    sci_variant=(getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or ''),
                    sci_pending=bool(getattr(getattr(self, 'gov_state', None), 'sci_pending', False)),
                )
            except Exception:
                pass

            # Normalize inline "Self-Debunking: ..." leaks so the block parser can box it deterministically.
            try:
                raw_for_render = normalize_inline_self_debunking_header(raw_for_render)
            except Exception:
                pass

            # Enforce Self-Debunking contract deterministically (when required by JSON).
            try:
                raw_for_render = enforce_self_debunking_contract(raw_for_render, gov, prof, lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'))
            except Exception:
                pass
            try:
                raw_for_render = normalize_self_debunking_numbering_text(
                    raw_for_render,
                    lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'),
                )
            except Exception:
                pass
            try:
                raw_for_render = dedupe_self_debunking_sections(raw_for_render)
            except Exception:
                pass

            # Normalize SCI Trace numbering (only step headers numbered)
            try:
                raw_for_render = normalize_sci_trace_numbering(raw_for_render, gov)
            except Exception:
                pass

            # Hard-render SCI Trace as HTML to prevent Markdown list runaway numbering (1..31)
            try:
                raw_for_render = self._render_sci_trace_as_html_runtime(raw_for_render)
            except Exception:
                pass

            # Deterministic QC-override runtime checks (best-effort): detect obvious target/output mismatches.
            override_vios = []
            try:
                _ovr_runtime = getattr(self.gov_state, 'qc_overrides', {}) or {}
                override_vios = qc_override_runtime_violations(raw_for_render, _ovr_runtime)
                if override_vios:
                    alerts.append(("QC-Override", "; ".join(override_vios)))
            except Exception:
                override_vios = []

            # Remove internal status scaffolding leaked by weaker models.
            try:
                raw_for_render = strip_internal_scaffolding_status_lines(raw_for_render or "")
            except Exception:
                pass
            try:
                raw_for_render = strip_exact_status_header_line(raw_for_render or "", header or "")
            except Exception:
                pass
            
            # A: Header voranstellen
            if header:
                raw_for_render = header + "\n\n" + raw_for_render
            # Strict enforcement gate (optional): validate final text (pre-render) and optionally warn/block.
            strict_banner_html = ""
            try:
                _ens = self._get_enforcement_settings()
                pol = str((_ens or {}).get("policy") or "audit_only")
                ens_enabled = bool((_ens or {}).get("enabled", True))
                ens_blocked = list((_ens or {}).get("blocked_severities") or ["critical", "major"])
            except Exception:
                pol = "audit_only"
                ens_enabled = True
                ens_blocked = ["critical", "major"]
            if ens_enabled and pol in ("strict_warn", "strict_block"):
                try:
                    hv2, sv2 = self.validator.validate(
                        raw_for_render,
                        state=self.gov_state,
                        profile=prof,
                        expect_menu=False,
                        expect_trace=False,
                        is_command=False,
                        user_prompt=user_raw,
                        raw_response=raw_for_render,
                    )
                    if override_vios:
                        hv2 = list(hv2 or []) + list(override_vios)
                except Exception as e:
                    hv2, sv2 = [], []
                    # Fail-soft: show a warning in chat, but never crash.
                    try:
                        self._append_system_message(f"⚠️ QC/Validator error in strict enforcement: {e}")
                    except Exception:
                        pass
                if hv2:
                    # Build structured violations (code/message/severity) for incremental policy handling.
                    vios_struct = []
                    try:
                        if _compliance_scan is not None and hasattr(_compliance_scan, "classify_violation_messages_best_effort"):
                            vios_struct = _compliance_scan.classify_violation_messages_best_effort(
                                [str(x) for x in (hv2 or [])],
                                default_severity="critical",  # keep strict_block backward-compatible for current hard-validator path
                            ) or []
                    except Exception:
                        vios_struct = []

                    # New incremental path: evaluate strict action via modular transition intent.
                    # Fallback to legacy behavior if modules are unavailable.
                    strict_action = None
                    try:
                        if (
                            _state_from_runtime is not None
                            and _apply_intent is not None
                            and _ProcessModelResponse is not None
                            and _ComplianceViolation is not None
                        ):
                            dom_state = _state_from_runtime(self.gov_state)
                            try:
                                dom_state.enforcement_policy = pol
                            except Exception:
                                pass
                            try:
                                dom_state.enforcement_enabled = bool(ens_enabled)
                            except Exception:
                                dom_state.enforcement_enabled = True
                            try:
                                _bsl = [str(x).strip().lower() for x in (ens_blocked or []) if str(x).strip()]
                                dom_state.blocked_severities = _bsl or ["critical", "major"]
                            except Exception:
                                dom_state.blocked_severities = ["critical", "major"]

                            vio_objs = []
                            if isinstance(vios_struct, list) and vios_struct:
                                for vv in vios_struct:
                                    if not isinstance(vv, dict):
                                        continue
                                    vio_objs.append(
                                        _ComplianceViolation(
                                            rule=str(vv.get("code") or "hard_violation"),
                                            severity=str(vv.get("severity") or "critical"),
                                            message=str(vv.get("message") or ""),
                                        )
                                    )
                            else:
                                for msg in hv2 or []:
                                    vio_objs.append(
                                        _ComplianceViolation(
                                            rule="hard_violation",
                                            severity="critical",
                                            message=str(msg),
                                        )
                                    )

                            tr = _apply_intent(
                                dom_state,
                                _ProcessModelResponse(raw_text=raw_for_render, violations=tuple(vio_objs)),
                                {},
                            )
                            evs = list(getattr(tr, "audit_events", []) or [])
                            if evs:
                                strict_action = str((evs[-1] or {}).get("action") or "").strip().lower() or None
                    except Exception:
                        strict_action = None

                    if strict_action is None:
                        strict_action = "blocked" if pol == "strict_block" else "warned"

                    if strict_action == "blocked":
                        vio_lines = []
                        if isinstance(vios_struct, list) and vios_struct:
                            for vv in vios_struct:
                                sev = str((vv or {}).get("severity") or "").strip().upper()
                                msg = str((vv or {}).get("message") or "").strip()
                                code = str((vv or {}).get("code") or "").strip()
                                if msg:
                                    vio_lines.append(f"<li>[{html.escape(sev)}] {html.escape(msg)}</li>")
                                else:
                                    vio_lines.append(f"<li>[{html.escape(sev)}] {html.escape(code)}</li>")
                        else:
                            vio_lines = [f"<li>{html.escape(str(x))}</li>" for x in hv2]
                        blocked_html = (
                            "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
                            "<summary>⛔ STRICT BLOCK (hard violations)</summary>"
                            "<div class='csc-details'>"
                            "<p>The model response was blocked by the wrapper because hard rule violations remained after repair/enforcement.</p>"
                            "<ul>"
                            + "".join(vio_lines)
                            + "</ul>"
                            "<p><i>(Content withheld by wrapper)</i></p>"
                            "</div></details>"
                        )
                        try:
                            return (
                                {"html": sanitize_html(blocked_html), "text": "", "csc": None},
                                {"strict_enforcement": "blocked", "hard_violations": hv2, "violations_struct": vios_struct},
                            )
                        except Exception:
                            return (
                                {"html": blocked_html, "text": "", "csc": None},
                                {"strict_enforcement": "blocked", "hard_violations": hv2, "violations_struct": vios_struct},
                            )
                    else:
                        if strict_action == "warned":
                            # strict_warn
                            vio_lines = []
                            if isinstance(vios_struct, list) and vios_struct:
                                for vv in vios_struct:
                                    sev = str((vv or {}).get("severity") or "").strip().upper()
                                    msg = str((vv or {}).get("message") or "").strip()
                                    code = str((vv or {}).get("code") or "").strip()
                                    if msg:
                                        vio_lines.append(f"<li>[{html.escape(sev)}] {html.escape(msg)}</li>")
                                    else:
                                        vio_lines.append(f"<li>[{html.escape(sev)}] {html.escape(code)}</li>")
                            else:
                                vio_lines = [f"<li>{html.escape(str(x))}</li>" for x in hv2]

                            strict_banner_html = (
                                "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
                                "<summary>⚠️ RULE VIOLATION DETECTED (strict_warn)</summary>"
                                "<div class='csc-details'>"
                                "<p>The following response still contains hard rule violations after repair/enforcement:</p>"
                                "<ul>"
                                + "".join(vio_lines)
                                + "</ul>"
                                "</div></details><hr>"
                            )
            # Strict warn: show banner above the normal alerts/content
            if strict_banner_html:
                alert_html = strict_banner_html + alert_html

            # Display policy for verification-route markers:
            # default = visible (auditable); optional legacy hide via config switch.
            try:
                if self._hide_verification_route_lines_in_chat():
                    raw_for_render = strip_verification_route_display_lines(raw_for_render or "")
            except Exception:
                pass

            try:
                raw_for_render = unwrap_accidental_full_text_codefence(raw_for_render or "")
            except Exception:
                pass
            try:
                raw_for_render = strip_pathological_repetition_display_noise(
                    raw_for_render or "",
                    lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'),
                )
            except Exception:
                pass

            
            # B: Bilder einbetten
            raw_for_render = _auto_embed_image_urls(raw_for_render)
            
            # C: Farben anwenden
            if ctx.get('color', 'off') == 'on':
                raw_for_render = apply_color_spans(raw_for_render, enabled=True)
            
            # D: Markdown Cleanup (Abstände)
            raw_for_render = re.sub(r'(?<!\n)\n([*-]|\d+\.) ', r'\n\n\1 ', raw_for_render)
            raw_for_render = re.sub(r'(?<!\n)\nQC-Matrix:', r'\n\nQC-Matrix:', raw_for_render)
            
            # E: Render (prefer v192 pipeline if available, else legacy Markdown+Sanitize+SD numbering)
            raw_for_render = normalize_known_markdown_control_headings(raw_for_render or "")
            if _rendering_pipeline_v192 is not None:
                try:
                    # For SD/labels we follow Answer Language (not UI language) because SD must be in Answer Language.
                    ans_lang = getattr(getattr(self, 'gov_state', None), 'answer_language', None) or self._lang() or 'en'
                    ans_lang = 'de' if str(ans_lang).lower().startswith('de') else 'en'
                    rctx = _rendering_pipeline_v192.RenderContext(
                        ui_lang=ans_lang,
                        color=str(ctx.get('color', 'off') or 'off'),
                        is_command=False,
                        comm_active=True,
                        strict=True,
                    )
                    final_html_body = _rendering_pipeline_v192.render_llm_text_to_html(raw_for_render or "", rctx)
                except Exception:
                    final_html_body = ""
            else:
                # Legacy path (kept as fallback)
                try:
                    final_html_body = markdown.markdown(raw_for_render, extensions=['extra', 'codehilite'])
                except Exception:
                    final_html_body = markdown.markdown(raw_for_render, extensions=['fenced_code', 'tables'])
                final_html_body = sanitize_html(final_html_body)
                try:
                    final_html_body = html_number_self_debunking(final_html_body, lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'))
                except Exception:
                    pass

            # After HTML rendering, normalize leaked markdown emphasis inside self-debunking blocks.
            try:
                final_html_body = sanitize_self_debunking_markdown_in_html(final_html_body or "")
            except Exception:
                pass
            try:
                final_html_body = normalize_hash_subheadings_in_html(final_html_body or "")
            except Exception:
                pass
            try:
                final_html_body = strip_internal_scaffolding_status_html(final_html_body or "")
            except Exception:
                pass

            # Ensure Self-Debunking points are visibly numbered in HTML on all render paths (v192 + legacy).
            try:
                final_html_body = html_number_self_debunking(
                    final_html_body or "",
                    lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'),
                )
            except Exception:
                pass
            try:
                # Second pass: html_number_self_debunking may surface orphan "*" lines from weak markdown output.
                final_html_body = sanitize_self_debunking_markdown_in_html(final_html_body or "")
            except Exception:
                pass

            # HTML-stage QC footer guard:
            # If a canonical QC footer exists in raw text but is missing or truncated after HTML rendering,
            # restore the raw canonical line at the end of the HTML body.
            try:
                _raw_qc_match = None
                for _m in re.finditer(r"(?im)^\s*QC-Matrix:\s*.*$", str(raw_for_render or "")):
                    _raw_qc_match = _m
                _raw_qc_line = (_raw_qc_match.group(0).strip() if _raw_qc_match else "")
                # Fallback: rebuild a canonical footer from parsed raw text if line extraction failed.
                if not _raw_qc_line:
                    try:
                        _ovr2_raw = getattr(self.gov_state, 'qc_overrides', {}) or {}
                        _ovr2 = gov.normalize_qc_overrides(_ovr2_raw)
                        _corr2 = gov.get_effective_qc_corridor(prof, _ovr2)
                        _seed = ensure_qc_footer_present(str(raw_for_render or ""), gov, prof, _ovr2)
                        _rebuilt = enforce_qc_footer_deltas(str(_seed or ""), _corr2, prof)
                        _rebuilt = ensure_qc_footer_is_last(_rebuilt or "")
                        for _m2 in re.finditer(r"(?im)^\s*QC-Matrix:\s*.*$", str(_rebuilt or "")):
                            _raw_qc_match = _m2
                        _raw_qc_line = (_raw_qc_match.group(0).strip() if _raw_qc_match else "")
                    except Exception:
                        pass
                # If a raw footer line exists but is itself incomplete/truncated, replace it with a
                # deterministic canonical footer based on the active profile + current overrides.
                try:
                    _raw_probe = re.sub(r"\s+", " ", str(_raw_qc_line or "")).strip()
                    _raw_complete = (
                        ("Clarity " in _raw_probe and "Brevity " in _raw_probe and "Evidence " in _raw_probe and
                         "Empathy " in _raw_probe and "Consistency " in _raw_probe and "Neutrality " in _raw_probe)
                        or
                        ("Klarheit " in _raw_probe and ("Kürze " in _raw_probe or "Kuerze " in _raw_probe) and "Evidenz " in _raw_probe and
                         "Empathie " in _raw_probe and "Konsistenz " in _raw_probe and ("Neutralität " in _raw_probe or "Neutralitaet " in _raw_probe))
                    )
                    if (not _raw_complete):
                        try:
                            _raw_qc_line = str(self._qc_footer_for_profile(prof) or "").strip()
                        except Exception:
                            pass
                except Exception:
                    pass
                if _raw_qc_line:
                    _plain_html = re.sub(r"<[^>]+>", "", str(final_html_body or ""))
                    _html_any_qc_lines = re.findall(r"(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$", _plain_html)
                    _html_qc_lines = re.findall(r"(?im)^\s*QC-Matrix:\s*.*$", _plain_html)
                    _html_qc_last = (_html_qc_lines[-1].strip() if _html_qc_lines else "")
                    # Use the whole suffix from the last footer marker because renderers can fragment
                    # the QC footer across wrappers/newlines.
                    _qc_probe = _html_qc_last
                    try:
                        _idx = _plain_html.rfind("QC-Matrix:")
                        if _idx >= 0:
                            _qc_probe = _plain_html[_idx:]
                    except Exception:
                        pass
                    try:
                        _ts_idx = str(_qc_probe or "").lower().find("response at")
                        if _ts_idx >= 0:
                            _qc_probe = str(_qc_probe or "")[:_ts_idx]
                    except Exception:
                        pass
                    _qc_probe = re.sub(r"\s+", " ", str(_qc_probe or "")).strip()
                    _qc_complete = (
                        ("Clarity " in _qc_probe and "Brevity " in _qc_probe and "Evidence " in _qc_probe and
                         "Empathy " in _qc_probe and "Consistency " in _qc_probe and "Neutrality " in _qc_probe)
                        or
                        ("Klarheit " in _qc_probe and ("Kürze " in _qc_probe or "Kuerze " in _qc_probe) and "Evidenz " in _qc_probe and
                         "Empathie " in _qc_probe and "Konsistenz " in _qc_probe and ("Neutralität " in _qc_probe or "Neutralitaet " in _qc_probe))
                    )
                    # If the renderer kept multiple footer variants (e.g. model-emitted localized "QC:" plus
                    # the wrapper's canonical "QC-Matrix:"), collapse them to exactly one canonical footer.
                    _has_duplicate_qc_footer_lines = len(_html_any_qc_lines) > 1
                    if (not _qc_complete) or _has_duplicate_qc_footer_lines:
                        final_html_body = re.sub(
                            r"(?is)<p>\s*QC(?:-Matrix)?\s*:.*?</p>\s*",
                            "",
                            str(final_html_body or ""),
                        ).rstrip()
                        # Also remove common fragmented QC footer wrappers if present.
                        final_html_body = re.sub(
                            r"(?is)<div[^>]*>\s*QC(?:-Matrix)?\s*:.*?</div>\s*",
                            "",
                            str(final_html_body or ""),
                        ).rstrip()
                        final_html_body += "<p>" + html.escape(_raw_qc_line) + "</p>"
            except Exception:
                pass
            
            # F: Alerts + Body + Timestamp zusammenbauen
            # Render-failure behavior (Variant D = Auto):
            # - If the rendered HTML looks valid, show it and include raw provider output in a collapsible <details>.
            # - If rendering looks broken/escaped, show the raw output (escaped) instead (no double output).
            def _looks_like_rendered_html(h: str) -> bool:
                if not h:
                    return False
                hl = h.lstrip().lower()
                # Common failure: escaped HTML inside <pre> or heavy entity-escaping.
                if hl.startswith("<pre") and "&lt;" in hl:
                    return False
                if h.count("&lt;") > 10 and ("<p" not in hl and "<div" not in hl and "<ol" not in hl):
                    return False
                # Must contain at least one typical HTML block tag.
                return any(t in hl for t in ("<p", "<div", "<ol", "<ul", "<table", "<pre", "<blockquote"))

            
            # --- Normalization / rendering summary (structural-only; used for audits & debugging) ---
            try:
                _qc_pat = re.compile(r"(?im)^\s*QC(?:-Matrix)?\s*:")
                _raw_qc_count = len(_qc_pat.findall(str(raw_for_render or "")))
                _html_qc_count = len(_qc_pat.findall(re.sub(r"<[^>]+>", "", str(final_html_body or ""))))
                _sd_boxed = ("self-debunking" in str(final_html_body or "").lower()) or ("selbst-debunking" in str(final_html_body or "").lower())
                _sd_numbered = detect_self_debunking_numbered_html(str(final_html_body or ""))
                _norm_summary = {
                    "qc_footer_raw_count": _raw_qc_count,
                    "qc_footer_html_count": _html_qc_count,
                    "qc_footer_deduped": (_raw_qc_count > 1 and _html_qc_count == 1),
                    "self_debunking_boxed": bool(_sd_boxed),
                    "self_debunking_numbered": bool(_sd_numbered),
                }
                if csc_meta is None:
                    csc_meta = {}
                if isinstance(csc_meta, dict):
                    csc_meta["normalization"] = _norm_summary
            except Exception:
                pass

            render_ok = _looks_like_rendered_html(final_html_body or "")

            # Provider outputs can occasionally end abruptly (especially with weaker/free models).
            # Warn deterministically, but do not alter the model text.
            try:
                _raw_probe = (raw_original or raw_response or "")
                _trunc, _trunc_msg = _detect_probable_truncation(_raw_probe, final_html_body or "")
                if _trunc and _trunc_msg:
                    alert_html = _control_layer_alert_html(
                        _trunc_msg + " Bitte gegenprüfen oder Antwort neu generieren.",
                        title="CONTROL LAYER NOTE",
                        severity="warn",
                        lang=self._answer_lang(),
                    ) + (alert_html or "")
                    try:
                        if csc_meta is None:
                            csc_meta = {}
                        if isinstance(csc_meta, dict):
                            csc_meta["probable_truncation"] = True
                    except Exception:
                        pass
            except Exception:
                pass

            # Record render outcome in normalization summary (if present)
            try:
                if csc_meta is None:
                    csc_meta = {}
                if isinstance(csc_meta, dict):
                    ns = csc_meta.get("normalization")
                    if isinstance(ns, dict):
                        ns["render_ok"] = bool(render_ok)
                        ns["render_fallback"] = (not bool(render_ok))
            except Exception:
                pass

            timestamp = _format_response_timestamp()

            if render_ok:
                raw_details = ""
                try:
                    _raw = (raw_original or raw_response or "")
                    # Keep it readable but safe.
                    _raw_esc = html.escape(str(_raw))
                    raw_details = (
                        '<details class="raw-output"><summary>Raw model output</summary>'
                        '<pre class="raw-output-pre">' + _raw_esc + '</pre></details>'
                    )
                except Exception:
                    raw_details = ""

                final_html = alert_html + final_html_body + raw_details + f'<div class="ts-footer">Response at {timestamp}</div>'
                return final_html, csc_meta

            # Render looks broken: show raw (escaped) as a last-resort, but do not duplicate it.
            try:
                _raw = (raw_original or raw_response or "")
                _raw_esc = html.escape(str(_raw))
                fallback_note = '<div class="note-box"><b>Render fallback</b>: showing raw model output.</div>'
                final_html = alert_html + fallback_note + '<pre class="raw-output-pre">' + _raw_esc + '</pre>' + f'<div class="ts-footer">Response at {timestamp}</div>'
                return final_html, csc_meta
            except Exception:
                final_html = alert_html + f'<div class="ts-footer">Response at {timestamp}</div>'
                return final_html, csc_meta


        except Exception as e:
            # Fallback bei schwerem Error
            return f"<span style='color:red'>Runtime Error in Renderer: {e}</span>", None

    def _normalize_raw_output_contracts(self, text: str, *, governance_enabled: bool, is_command: bool = False) -> str:
        """Apply raw governance/output contract normalizations in deterministic order."""
        repaired = text
        lang = getattr(getattr(self, 'gov_state', None), 'answer_language', 'de')
        profile_name = getattr(getattr(self, 'gov_state', None), 'active_profile', 'Standard') or 'Standard'

        svc = getattr(self, 'governance_service', None)
        if svc is not None and hasattr(svc, 'normalize_output_contracts'):
            try:
                return svc.normalize_output_contracts(
                    repaired,
                    gov_mgr=gov,
                    profile_name=profile_name,
                    governance_enabled=bool(governance_enabled),
                    is_command=bool(is_command),
                    lang=lang,
                )
            except Exception:
                pass

        try:
            if not governance_enabled:
                raise RuntimeError('governance disabled')
            repaired = enforce_qc_footer_deltas(repaired, gov, profile_name)
        except Exception:
            pass
        try:
            repaired = normalize_evidence_tags(repaired)
        except Exception:
            pass
        try:
            if not governance_enabled:
                raise RuntimeError('governance disabled')
            repaired = enforce_self_debunking_contract(
                repaired,
                gov,
                profile_name,
                is_command=is_command,
                lang=lang,
            )
        except Exception:
            pass
        try:
            repaired = normalize_sci_trace_numbering(repaired, gov)
        except Exception:
            pass
        return repaired

    def _apply_output_prefs_to_user_message(self, user_raw: str) -> str:
        """Apply wrapper-level preferences to the USER message only.

        - Answer language (LLM content only): en/de
        - Slightly increase answer length (modest, deterministic)

        All UI/help/state/config/SCI/header/footer/QC remain English because they are deterministic renderers.
        The model is also instructed to keep scaffolding labels in English.
        """
        try:
            raw = user_raw or ""
            # Guard: avoid double-wrapping
            if raw.lstrip().startswith('[OUTPUT LANGUAGE]'):
                return raw

            # Resolve desired answer language
            lang = None
            try:
                lang = getattr(getattr(self, 'gov_state', None), 'answer_language', None)
            except Exception:
                lang = None
            if not lang:
                try:
                    lang = getattr(cfg, 'get_answer_language', lambda: 'de')()
                except Exception:
                    lang = 'de'
            lang = (lang or 'de').strip().lower()
            if lang not in ('en', 'de'):
                lang = 'de'

            lang_name = 'English' if lang == 'en' else 'German'

            # Small, explicit wrapper directives.
            lines = []
            lines.append(
                f"[OUTPUT LANGUAGE] Final answer and SCI Trace content in {lang_name} ({lang}). "
                "Keep fixed protocol tokens/step labels unchanged."
            )
            lines.append(
                "[ANSWER LENGTH] Prefer substantive depth (+20-40% vs minimal). "
                "Use concrete mechanisms/examples where useful; avoid one-liners."
            )

            # QC overrides (session-local): these should influence BOTH
            # - delta calculation / enforcement (Python side) and
            # - the model's writing behavior (prompt side)
            # without touching any other governance logic.
            try:
                ovs_raw = getattr(getattr(self, 'gov_state', None), 'qc_overrides', None)
            except Exception:
                ovs_raw = None
            ovs = ovs_raw if isinstance(ovs_raw, dict) else {}
            if ovs:
                # Normalize keys + clamp values.
                canon = {
                    'clarity': 'Clarity',
                    'brevity': 'Brevity',
                    'evidence': 'Evidence',
                    'empathy': 'Empathy',
                    'consistency': 'Consistency',
                    'neutrality': 'Neutrality',
                }
                clean = {}
                for k, v in ovs.items():
                    try:
                        kk = (str(k) or '').strip().lower()
                        kk = canon.get(kk, None)
                        if not kk:
                            continue
                        iv = int(v)
                        if iv < 0:
                            iv = 0
                        if iv > 3:
                            iv = 3
                        clean[kk] = iv
                    except Exception:
                        continue

                if clean:
                    # Let a Brevity override take precedence over the generic answer-length hint.
                    b = clean.get('Brevity')
                    if isinstance(b, int):
                        if b <= 1:
                            lines[1] = (
                                "[ANSWER LENGTH] Be detailed and thorough. "
                                "Do not compress; cover key dimensions explicitly and include concrete steps/examples."
                            )
                        elif b >= 3:
                            lines[1] = "[ANSWER LENGTH] Be concise. Use short sentences; minimize background; prefer bullets."

                    parts = [f"{k}={v}" for k, v in clean.items()]
                    lines.append(f"[QC OVERRIDES] Active temporary targets override profile defaults: {', '.join(parts)}")

                    # Minimal, deterministic behavior hints. Note: in this QC scale,
                    # higher Brevity => more concise; lower Brevity => more detailed.
                    hints = []
                    for k, v in clean.items():
                        if k == 'Brevity':
                            if v <= 0:
                                hints.append("Brevity=0: be very detailed; include steps/examples; avoid compressing.")
                            elif v == 1:
                                hints.append("Brevity=1: be detailed (but not endless); include key steps.")
                            elif v == 2:
                                hints.append("Brevity=2: moderate length; balance detail and concision.")
                            else:
                                hints.append("Brevity=3: be as concise as possible; short answer, minimal extras.")
                        elif k == 'Evidence':
                            if v >= 3:
                                hints.append("Evidence=3: make claims traceable; cite sources/assumptions; mark uncertainty.")
                            elif v == 2:
                                hints.append("Evidence=2: support key claims with reasoning; state assumptions.")
                            elif v == 1:
                                hints.append("Evidence=1: light justification; avoid over-claiming.")
                            else:
                                hints.append("Evidence=0: minimal justification; keep it practical.")
                        elif k == 'Clarity':
                            if v >= 3:
                                hints.append("Clarity=3: be extremely clear; structure with headings/bullets; define terms.")
                            elif v == 2:
                                hints.append("Clarity=2: clear structure; avoid ambiguity.")
                            else:
                                hints.append("Clarity<=1: keep it understandable; skip extra pedagogy.")
                        elif k == 'Empathy':
                            if v >= 3:
                                hints.append("Empathy=3: warm and supportive tone.")
                            elif v == 2:
                                hints.append("Empathy=2: considerate tone.")
                            else:
                                hints.append("Empathy<=1: neutral-professional tone.")
                        elif k == 'Consistency':
                            if v >= 3:
                                hints.append("Consistency=3: self-check; keep internal logic tight; avoid contradictions.")
                            elif v == 2:
                                hints.append("Consistency=2: keep reasoning consistent.")
                            else:
                                hints.append("Consistency<=1: keep it simple; avoid conflicting statements.")
                        elif k == 'Neutrality':
                            if v >= 3:
                                hints.append("Neutrality=3: strictly neutral wording; avoid loaded language.")
                            elif v == 2:
                                hints.append("Neutrality=2: mostly neutral tone.")
                            else:
                                hints.append("Neutrality<=1: neutral by default; avoid polarizing phrasing.")

                    if hints:
                        lines.append("[QC BEHAVIOR] " + " ".join(hints))
                    try:
                        sci_now = bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False))
                    except Exception:
                        sci_now = False
                    if sci_now and isinstance(b, int) and b <= 0:
                        lines.append(
                            "[SCI TRACE DETAIL] Brevity=0 is active: write at least two substantive sentences per SCI Trace step "
                            "(ideally three when needed for clarity)."
                        )
            else:
                try:
                    _qc_reset_pending = bool(getattr(self, '_qc_override_prompt_reset_pending', False))
                except Exception:
                    _qc_reset_pending = False
                if _qc_reset_pending:
                    lines.append("[QC OVERRIDES] Cleared. Use profile defaults only.")
                    lines.append("[QC BEHAVIOR] Ignore any previous temporary QC override instructions from earlier turns.")
                    try:
                        self._qc_override_prompt_reset_pending = False
                    except Exception:
                        pass
            lines.append("")
            return "\n".join(lines) + raw
        except Exception:
            return user_raw

    def _csc_wrap_user_message(self, user_raw: str):
        """Deterministic CSC enforcement on the prompt side.

        Returns (text_to_send, pre_csc_meta or None).
        We only use strings/configs from the active Comm-SCI ruleset.
        """
        try:
            # Guard rails
            if not getattr(self.gov_state, 'comm_active', False):
                return user_raw, None
            prof = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
            if prof in {'Briefing', 'Sandbox'}:
                return user_raw, None

            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            if not getattr(gov_obj, 'loaded', False):
                return user_raw, None

            refiner = getattr(gov_obj, 'csc_refiner', None)
            if not refiner:
                return user_raw, None

            overlay = getattr(self.gov_state, 'overlay', '') or ''
            mult = 2 if overlay == 'Explore' else 1

            # Governance trigger heuristics (deterministic, conservative)
            txt_l = (user_raw or '').lower()
            uncertainty_U4 = bool(re.search(r"\bU[4-8]\b", user_raw or ""))
            web_check = bool(re.search(r"\bweb\s*[- ]\s*check\b", txt_l))
            strong_claim = any(x in txt_l for x in ["always", "never", "definitely", "guarantee", "prove", "immer", "niemals", "definitiv"])

            dec = refiner.decide(
                comm_active=True,
                active_profile=prof,
                input_raw=user_raw or "",
                uncertainty_U4_active=uncertainty_U4,
                web_check_hook_active=web_check,
                strong_claim_detected=strong_claim,
                neutrality_delta_negative=False,
                threshold_multiplier=mult,
            )

            if not getattr(dec, 'apply', False):
                return user_raw, None

            # Pull all user-visible instructions strictly from JSON
            cl = (getattr(gov_obj, 'data', {}) or {}).get('control_layer', {}) or {}
            bridge = (cl.get('components', {}) or {}).get('csc_trigger_bridge', {}) or {}
            constraints = (bridge.get('constraints', {}) or {})

            tm = (constraints.get('transparency_marker', {}) or {})
            marker = tm.get('marker', 'CSC-Refine: applied') or 'CSC-Refine: applied'

            brev = (constraints.get('brevity_guard', {}) or {})
            brev_fallback = brev.get('fallback', '') or ''
            if not brev_fallback:
                # policy fallback (also JSON)
                csc = ((cl.get('subsystems', {}) or {}).get('csc_engine', {}) or {})
                brev_fallback = ((csc.get('policy', {}) or {}).get('brevity_cap', {}) or {}).get('fallback', '') or ''

            dyn = (getattr(gov_obj, 'data', {}) or {}).get('global_defaults', {})
            dyn_neut = (((dyn.get('dynamic_prompting', {}) or {}).get('actions', {}) or {}).get('neutrality', {}) or {})
            add_instr_neutrality = dyn_neut.get('delta_negative', '') or ''

            # We must not inject German UI; but this is internal instruction text (JSON is English).
            # Use only the configured marker + the configured additional-instruction string.
            add_lines = []
            add_lines.append(f"{marker}")
            # Neutrality instruction is the strongest JSON-defined general-purpose counter-perspective requirement.
            if add_instr_neutrality:
                add_lines.append(add_instr_neutrality)
            if brev_fallback:
                add_lines.append(brev_fallback)

            # Keep it compact and clearly separated.
            injected = "\n".join([l for l in add_lines if l.strip()])
            wrapped = user_raw + "\n\n" + injected

            thr_fs = int(getattr(refiner, '_refine_params', {}).get('threshold_f_score', 8) or 8)
            thr_tok = int(getattr(refiner, '_refine_params', {}).get('min_token_count', 80) or 80)
            gov_min_tok = int(getattr(refiner, '_gov_params', {}).get('min_token_count_governance', 40) or 40)
            if mult != 1:
                thr_fs *= mult
                thr_tok *= mult
                gov_min_tok *= mult
            pre_meta = {
                'applied': True,
                'message': CSC_WARNING_TEXT,
                'trigger': str(getattr(dec, 'trigger_source', '')),
                'mode': str(getattr(dec, 'mode', '')),
                'governance_triggered': bool(getattr(dec, 'governance_triggered', False)),
                'token_count': int(getattr(dec, 'token_count', 0)),
                'f_score': int(getattr(dec, 'f_score', 0)),
                'score_tooltip': _csc_score_tooltip_text(
                    lang=self._answer_lang(),
                    f_score=int(getattr(dec, 'f_score', 0)),
                    token_count=int(getattr(dec, 'token_count', 0)),
                ),
                'thresholds_tooltip': _csc_thresholds_tooltip_text(
                    lang=self._answer_lang(),
                    thr_fs=thr_fs,
                    thr_tok=thr_tok,
                    gov_min_tok=gov_min_tok,
                    mult=mult,
                ),
                'overlay': overlay,
                'profile': str(prof or ''),
                'threshold_multiplier': int(mult or 1),
                'threshold_f_score': int(thr_fs),
                'threshold_token_count': int(thr_tok),
                'min_token_count': int(thr_tok),
                'min_token_count_governance': int(gov_min_tok),
                'schema_version': '1.0',
            }
            return wrapped, pre_meta
        except Exception:
            return user_raw, None

    def check_verification_route_gate(self, text: str):
        """Deterministically detect strong claims without a verification-route marker."""
        gov_obj = getattr(self, 'gov', None) or globals().get('gov')
        if not getattr(gov_obj, 'loaded', False):
            return None

        gate = (getattr(gov_obj, 'data', {}) or {}).get("global_defaults", {})
        gate = (gate or {}).get("verification_route_gate", {})
        if not (isinstance(gate, dict) and gate.get("enabled", False)):
            return None

        heur = gate.get("strong_claim_heuristics", {}) if isinstance(gate, dict) else {}
        kw = []
        if isinstance(heur, dict):
            kw = (heur.get("keywords_de", []) or []) + (heur.get("keywords_en", []) or [])
        if not kw:
            kw = ["immer", "niemals", "definitiv", "guarantee", "prove"]

        text_l = (text or "").lower()
        if not any(str(k).lower() in text_l for k in kw):
            return None

        rpm = gate.get("route_presence_markers", {}) if isinstance(gate, dict) else {}
        markers = []
        if isinstance(rpm, dict):
            markers = rpm.get("markers", []) or []
        if not markers:
            markers = ["Source", "Measurement", "Contrast", "Web Check", "Quelle", "Messung"]

        has_linker = bool(re.search(r"\[(GREEN|YELLOW|RED|GRAY)-", text or ""))
        if not (any(str(m).lower() in text_l for m in markers) or has_linker):
            return "Verification Route Gate: Strong claim detected, but no verification-route marker found."
        return None
    
    def _execute_legacy_command(self, cmd: str):
        """Apply deterministic state changes for Profiles, Modes, and Core commands."""
        gov_obj = getattr(self, 'gov', None) or globals().get('gov')
        data = getattr(gov_obj, 'data', {}) if gov_obj is not None else {}

        # Phase 3: controller dispatch path (application layer). Legacy path stays as fallback.
        try:
            if _controller_dispatch is not None:
                def _mirror():
                    gov_obj2 = getattr(self, 'gov', None) or globals().get('gov')
                    if gov_obj2 is not None:
                        if hasattr(self.gov_state, 'qc_overrides'):
                            setattr(gov_obj2, 'qc_overrides', dict(getattr(self.gov_state, 'qc_overrides', {}) or {}))
                        gov.runtime_state = self.gov_state
                        setattr(gov_obj2, 'runtime_state', self.gov_state)
                outcome = _controller_dispatch(
                    cmd=cmd,
                    runtime_state=self.gov_state,
                    ruleset_data=data if isinstance(data, dict) else {},
                    mirror_callback=_mirror,
                )
                if outcome.applied:
                    return
        except Exception:
            pass

        # Phase 2 fallback: central state transition path (intents + reducer).
        try:
            if (
                _intent_from_command is not None
                and _state_from_runtime is not None
                and _state_apply_to_runtime is not None
                and _apply_intent is not None
            ):
                intent = _intent_from_command(cmd)
                if intent is not None:
                    dom_state = _state_from_runtime(self.gov_state)
                    result = _apply_intent(dom_state, intent, data if isinstance(data, dict) else {})
                    _state_apply_to_runtime(result.state, self.gov_state)

                    # Keep manager mirrors aligned for deterministic QC/state behavior.
                    try:
                        gov_obj2 = getattr(self, 'gov', None) or globals().get('gov')
                        if gov_obj2 is not None:
                            if hasattr(self.gov_state, 'qc_overrides'):
                                setattr(gov_obj2, 'qc_overrides', dict(getattr(self.gov_state, 'qc_overrides', {}) or {}))
                            gov.runtime_state = self.gov_state
                            setattr(gov_obj2, 'runtime_state', self.gov_state)
                    except Exception:
                        pass
                    return
        except Exception:
            pass

        # 1) Profile switching
        if cmd.startswith("Profile "):
            pname = cmd.split(" ", 1)[1].strip()
            profiles = (data or {}).get("profiles", {}) if isinstance(data, dict) else {}
            if isinstance(profiles, dict) and pname in profiles:
                self.gov_state.active_profile = pname
                # QC overrides are session-local and must reset on profile switch
                try:
                    self.gov_state.qc_overrides = {}
                except Exception:
                    pass
                try:
                    gov_obj2 = getattr(self, 'gov', None) or globals().get('gov')
                    if gov_obj2 is not None:
                        setattr(gov_obj2, 'qc_overrides', {})
                        gov.runtime_state = self.gov_state
                        setattr(gov_obj2, 'runtime_state', self.gov_state)
                except Exception:
                    pass

                # Reset pending counters on any explicit profile switch
                try:
                    self.gov_state.sci_pending_turns = 0
                except Exception:
                    pass
                # Reset SCI on profile switch (except Expert/Sparring)
                if pname not in ["Expert", "Sparring"]:
                    self.gov_state.sci_active = False
                    self.gov_state.sci_pending = False
                    try:
                        self.gov_state.sci_variant = ""
                    except Exception:
                        pass
            return

        # 2. Mode Overlays
        if cmd == "Strict on": self.gov_state.overlay = "Strict"
        elif cmd == "Strict off": self.gov_state.overlay = ""
        
        elif cmd == "Explore on": self.gov_state.overlay = "Explore"
        elif cmd == "Explore off": self.gov_state.overlay = ""

        # 3. Color Mode
        elif cmd == "Color on": self.gov_state.color = "on"
        elif cmd == "Color off": self.gov_state.color = "off"

        # 4. SCI Control
        elif cmd == "SCI on": 
            self.gov_state.sci_pending = True
            try:
                self.gov_state.sci_pending_turns = 0
            except Exception:
                pass
        elif cmd == "SCI off":
            self.gov_state.sci_pending = False
            self.gov_state.sci_active = False
            self.gov_state.sci_variant = ""
            try:
                self.gov_state.sci_pending_turns = 0
            except Exception:
                pass

        elif cmd == "SCI recurse":
            # Canonical JSON: sci.recursive_sci
            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            data = getattr(gov_obj, 'data', {}) if gov_obj is not None else {}

            max_depth = 2
            try:
                max_depth = int(((data.get('sci') or {}).get('recursive_sci') or {}).get('max_depth', 2))
            except Exception:
                max_depth = 2

            try:
                cur = int(getattr(self.gov_state, 'sci_recursion_depth', 0) or 0)
            except Exception:
                cur = 0

            ok = try_enter_sci_recursion(self.gov_state, max_depth=max_depth)
            if not ok:
                return

        # 5. Comm Core
        elif cmd == "Comm Stop":
            self.gov_state.comm_active = False
        elif cmd == "Comm Start":
            self.gov_state.comm_active = True
            # Canonical JSON: comm_start_initialization.enforce_default_profile_on_comm_start
            try:
                default_prof = (data.get('default_profile') or 'Standard')
                profiles = (data.get('profiles') or {}) if isinstance(data, dict) else {}
                if isinstance(profiles, dict) and default_prof in profiles:
                    self.gov_state.active_profile = default_prof
                    self.gov_state.sci_pending_turns = 0
                    if default_prof not in ['Expert', 'Sparring']:
                        self.gov_state.sci_active = False
                        self.gov_state.sci_pending = False
                        self.gov_state.sci_variant = ''
            except Exception:
                pass
        elif cmd == "Comm Anchor off":
            # Disable periodic Anchor Snapshot automation for this session
            try:
                self.gov_state.anchor_auto = False
                self.gov_state.anchor_force_next = False
                self.gov_state.anchor_auto_user_override = True
            except Exception:
                pass

        elif cmd == "Comm Anchor on":
            # Enable periodic Anchor Snapshot automation for this session
            try:
                self.gov_state.anchor_auto = True
                self.gov_state.anchor_auto_user_override = True
            except Exception:
                pass
            # Do not force an immediate anchor unless explicitly requested; keep previous behavior conservative.
            try:
                self.gov_state.anchor_force_next = False
            except Exception:
                pass


        
        # 6. Dynamic
        elif cmd == "Dynamic one-shot on":
            # Canonical JSON: global_defaults.dynamic_prompting.one_shot_override
            try:
                self.gov_state.dynamic_one_shot_active = True
            except Exception:
                pass
            # Keep legacy flag for UI/state rendering compatibility
            try:
                self.gov_state.dynamic_nudge = "one-shot"
            except Exception:
                pass
  
    def _handle_sci_selection(self, letter: str):
        """Activate SCI variant A–H based strictly on canonical JSON; UI strings in current language.
        Also refreshes the underlying chat session so the model actually uses the selected SCI state.
        """
        ui_lang = (self._lang() if hasattr(self, "_lang") else "en") or "de"
        ui_lang = UI_LANG

        def tr(key: str, fallback: str = "") -> str:
            try:
                s = key
                return s if s and s != key else fallback
            except Exception:
                return fallback

        char = (letter or "").strip().upper()

        # Update state
        try:
            self.gov_state.sci_variant = char
            self.gov_state.sci_active = True
            self.gov_state.sci_pending = False
            self.gov_state.sci_pending_turns = 0
        except Exception:
            pass

        # Canonical JSON lookup
        gov_obj = getattr(self, "gov", None) or globals().get("gov")
        data = getattr(gov_obj, "data", None) if gov_obj else None

        vname = ""
        vfocus = ""
        if isinstance(data, dict):
            variants = (((data.get("sci") or {}).get("variant_menu") or {}).get("variants") or {})
            v = (variants.get(char) or {}) if isinstance(variants, dict) else {}
            if isinstance(v, dict):
                vname = v.get("name") or ""
                vfocus = v.get("focus") or ""

        # Display (optional I18N overrides if present; otherwise JSON)
        title = tr(f"sci_name_{char}", "") or tr(f"sci_var_{char}", "") or (vname or f"Variant {char}")
        desc = tr(f"sci_focus_{char}", "") or (vfocus or "")

        footer = "SCI activated"
        proto = "Protocol"

        html_out = f"""
        <div style="border: 2px solid #1a73e8; background: #f0f7ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="font-weight: bold; color: #1a73e8; font-size: 14px; margin-bottom: 5px;">SCI ACTIVE: {html.escape(char)}</div>
            <div style="font-size: 18px; font-weight: bold; color: #333; margin-bottom: 8px;">{html.escape(title)}</div>
        """

        if desc:
            html_out += f"""
            <div style="font-size: 14px; color: #444; line-height: 1.4;">
                <i>"{html.escape(desc)}"</i>
            </div>
            """

        html_out += f"""
            <hr style="border: 0; border-top: 1px solid #ccd; margin: 10px 0;">
            <div style="font-size: 11px; color: #666;">
                <b>{proto}:</b> Plan &rarr; Solution &rarr; Check.<br>
                Control Layer strictly monitors compliance with this role.
            </div>
        </div>
        <div class="ts-footer">{html.escape(footer)}</div>
        """
        # Update model state (no session recreation)
        try:
            self._ensure_governance_pinned(reason=f'SCI select {char}')
            self._send_state_update_to_model(reason=f'SCI select {char}')
            self._last_session_stamp = (
                getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard',
                getattr(self.gov_state, 'overlay', '') or '',
                getattr(self.gov_state, 'color', 'off') or 'off',
                bool(getattr(self.gov_state, 'sci_active', False)),
                getattr(self.gov_state, 'sci_variant', '') or '',
                getattr(self.gov_state, 'conversation_language', '') or '',
                bool(getattr(self.gov_state, 'comm_active', False)),
                bool(getattr(self.gov_state, 'dynamic_one_shot_active', False)),
            )
        except Exception:
            pass

        try:
            self.history.append({"role": "bot", "content": f"SCI Variant {char} activated.", "ts": datetime.now().isoformat(), "csc": None})
        except Exception:
            pass

        return {"html": html_out, "csc": None}

    def _state_reminder_line(self) -> str:
        # Compact runtime-state reminder for the model (low token overhead).
        try:
            prof = getattr(getattr(self, 'gov_state', None), 'active_profile', 'Standard') or 'Standard'
            overlay = getattr(getattr(self, 'gov_state', None), 'overlay', '') or ''
            sci = getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or ''
            sci_active = bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False))
            color = getattr(getattr(self, 'gov_state', None), 'color', 'off') or 'off'
            comm = bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False))
            if prof in {'Sandbox', 'Briefing'}:
                color = 'off'
            sci_show = (sci or 'off') if sci_active else 'off'
            return f"[CURRENT STATE] Profile={prof} | Overlay={(overlay or 'off')} | SCI={sci_show} | Color={color} | Comm={'on' if comm else 'off'}"
        except Exception:
            return "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"

    def _ruleset_fingerprint(self) -> str:
        # Fingerprint based on raw canonical JSON so we can inject it once per session/ruleset.
        try:
            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            raw = getattr(gov_obj, 'raw_json', '') or ''
            ver = ''
            try:
                ver = str((getattr(gov_obj, 'data', {}) or {}).get('version', '') or '')
            except Exception:
                ver = ''
            return hashlib.sha256((ver + "\n" + raw).encode('utf-8', 'ignore')).hexdigest()
        except Exception:
            return ''


    def _build_pinned_governance_message(self) -> str:
        # One-time (per ruleset) canonical ruleset injection.
        gov_obj = getattr(self, 'gov', None) or globals().get('gov')
        raw = getattr(gov_obj, 'raw_json', '') or ''
        ver = ''
        try:
            ver = str((getattr(gov_obj, 'data', {}) or {}).get('version', '') or '')
        except Exception:
            ver = ''

        raw = (raw or '').strip()
        if not raw:
            return ''

        return (
            'COMM-SCI GOVERNANCE (CANONICAL JSON)\n'
            + f'Version: {ver}\n'
            + 'INSTRUCTIONS: The following JSON is the authoritative governance ruleset. '
              'Follow it exactly for all subsequent answers. Do NOT quote it back. '
              "Reply exactly with 'ACK'.\n\n"
            + 'BEGIN JSON\n'
            + raw
            + '\nEND JSON'
        )

    def _ensure_governance_pinned(self, reason: str = ""):


        # Ensure canonical rules were injected once for the current ruleset.
        try:
            if not bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False)):
                return
            fp = self._ruleset_fingerprint()
            if not fp:
                return
            if bool(getattr(self, '_gov_pinned_sent', False)) and str(getattr(self, '_gov_pinned_fp', '') or '') == fp:
                return
            msg = self._build_pinned_governance_message()
            if not msg:
                return
            if getattr(self, 'chat_session', None):
                # Rate limiting: count this pinned injection as an LLM call (best-effort).
                try:
                    if bool(getattr(self, 'rate_limit_enabled', True)) and getattr(self, 'rate_limiter', None) is not None:
                        _prov = (self._active_provider() or 'gemini').strip().lower()
                        _model = str(getattr(self, 'model_name', '') or '') if _prov == 'gemini' else ''
                        ok, _m = self.rate_limiter.allow_call(provider=_prov, model=_model, reason='pinned', consume=True)
                        if not ok:
                            return
                except Exception:
                    pass
                _ = self.chat_session.send_message(msg)
            self._gov_pinned_sent = True
            self._gov_pinned_fp = fp
        except Exception:
            pass

    def _send_state_update_to_model(self, reason: str = ""):
        # Avoid session resets: inject a small state update into the conversation.
        # NOTE: This costs an extra LLM call (Gemini). Disabled by default for performance.
        try:
            if not bool((getattr(cfg, 'config', {}) or {}).get('state_update_llm', False)):
                return
        except Exception:
            return
        try:
            if not getattr(self, 'chat_session', None):
                return
            if not bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False)):
                return
            line = self._state_reminder_line()
            msg = (
                'STATE UPDATE\n'
                + f'Reason: {reason or "state_changed"}\n'
                + line
                + "\nInstruction: Use this state for all subsequent answers. Reply exactly with 'ACK'."
            )
            # Rate limiting: count state update as an LLM call (best-effort).
            try:
                if bool(getattr(self, 'rate_limit_enabled', True)) and getattr(self, 'rate_limiter', None) is not None:
                    _prov = (self._active_provider() or 'gemini').strip().lower()
                    _model = str(getattr(self, 'model_name', '') or '') if _prov == 'gemini' else ''
                    ok, _m = self.rate_limiter.allow_call(provider=_prov, model=_model, reason='state_update', consume=True)
                    if not ok:
                        return
            except Exception:
                pass
            _ = self.chat_session.send_message(msg)
        except Exception:
            pass

    def _get_governed_system_instruction(self):
        # Minimal system instruction; canonical JSON is injected once as a pinned message.
        try:
            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            ver = str((getattr(gov_obj, 'data', {}) or {}).get('version', '') or '').strip()
        except Exception:
            ver = ''

        base = 'You are governed by Comm-SCI-Control'
        if ver:
            base += f' v{ver}'
        base += '. The canonical ruleset will be provided in this conversation as JSON. Follow it exactly.'

        note = (
            'IMPORTANT: Keep all scaffolding (headers/labels/SCI step names/QC labels/command outputs) in English. '
            'The wrapper may request a different final answer language via [OUTPUT LANGUAGE].'
        )

        qc_note = (
            'QC-Matrix footer format MUST be: QC-Matrix: Clarity <v> (Δ<d>) · Brevity <v> (Δ<d>) · Evidence <v> (Δ<d>) · Empathy <v> (Δ<d>) · Consistency <v> (Δ<d>) · Neutrality <v> (Δ<d>). '
            'Delta calculation (MANDATORY): for each target corridor [min,max]: if value<min → Δ=value-min; if value>max → Δ=value-max; else Δ0.'
        )

        state_note = ("Runtime state is provided in each user message via a single line starting with [CURRENT STATE]. "
                      "Treat that line as authoritative for profile/SCI/overlay/color and do NOT repeat it in your answer.")

        return base + "\n" + note + "\n" + state_note + "\n" + qc_note


    def _active_provider(self) -> str:
        try:
            psvc = getattr(self, 'provider_service', None)
            if psvc is not None and hasattr(psvc, 'router') and getattr(self, 'provider_router', None) is not None:
                # Keep service/router references in sync for tests and hot-swaps.
                psvc.router = getattr(self, 'provider_router', None)
            if psvc is not None:
                pr = getattr(psvc, 'router', None)
                if pr is not None and hasattr(pr, 'get_active_provider'):
                    return str(pr.get_active_provider() or 'gemini').strip().lower()
        except Exception:
            pass
        try:
            pr = getattr(self, 'provider_router', None)
            if pr is not None and hasattr(pr, 'get_active_provider'):
                return pr.get_active_provider()
        except Exception:
            pass
        try:
            # Fallback: config key
            return (getattr(cfg, 'config', {}) or {}).get('active_provider', 'gemini')
        except Exception:
            return 'gemini'

    def _provider_model(self, provider: str = '', fallback_model: str = '') -> str:
        try:
            psvc = getattr(self, 'provider_service', None)
            if psvc is not None and hasattr(psvc, 'router') and getattr(self, 'provider_router', None) is not None:
                psvc.router = getattr(self, 'provider_router', None)
            if psvc is not None:
                return psvc.get_provider_model(provider, fallback_model=fallback_model)
        except Exception:
            pass
        try:
            pr = getattr(self, 'provider_router', None)
            if pr is not None and hasattr(pr, 'get_provider_model'):
                return pr.get_provider_model(provider, fallback_model=fallback_model)
        except Exception:
            pass
        return (fallback_model or '').strip()

    def _build_openai_messages(self, user_text: str):
        """Build OpenAI-compatible messages payload (system + sliding history + user).

        Stage A: minimal governed system instruction + wrapper-managed history.
        NOTE: For stateless providers, we do NOT inject the full canonical JSON each call.
        The wrapper enforces contracts deterministically.
        """
        msgs = []
        try:
            sys = self._get_governed_system_instruction()
            msgs.append({'role': 'system', 'content': sys})
        except Exception:
            pass

        # Sliding window history (best-effort)
        try:
            hist = getattr(self, 'history', None) or []
            # Keep it modest; provider calls can get expensive quickly.
            tail = hist[-10:] if isinstance(hist, list) else []
            for h in tail:
                if not isinstance(h, dict):
                    continue
                role = (h.get('role') or '').strip().lower()
                content = h.get('content')
                if not isinstance(content, str) or not content.strip():
                    continue
                if role in ('user', 'assistant', 'system'):
                    msgs.append({'role': role, 'content': content})
                elif role in ('bot', 'assistant'):
                    msgs.append({'role': 'assistant', 'content': content})
                elif role == 'user':
                    msgs.append({'role': 'user', 'content': content})
        except Exception:
            pass

        msgs.append({'role': 'user', 'content': user_text or ''})
        return msgs

    def _wrap_user_text_for_model(self, user_text: str) -> str:
        """Prefix user message with authoritative runtime state and compact meta-instructions."""
        try:
            state_line = self._state_reminder_line()
        except Exception:
            state_line = "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"
        try:
            lang = (getattr(getattr(self, "gov_state", None), "answer_language", "") or "").strip().lower()
            if not lang:
                lang = getattr(cfg, "get_answer_language", lambda: "de")() or "de"
            if lang not in ("de", "en"):
                lang = "de"
        except Exception:
            lang = "de"

        evidence = ""
        try:
            if bool(getattr(getattr(self, "gov_state", None), "comm_active", False)) and (getattr(getattr(self, "gov_state", None), "color", "off") == "on"):
                evidence = ("EVIDENCE-LINKER: For each atomic factual claim in the FINAL ANSWER, prefix exactly one tag: "
                            "[GREEN] for well-established knowledge, [YELLOW] for plausible/uncertain, [RED] for speculative. "
                            "Do not tag headers, SCI Trace, or QC-Matrix. Keep tags in the final answer.")
        except Exception:
            evidence = ""

        meta = f"[OUTPUT LANGUAGE] {lang}"
        parts = [state_line, meta]
        if evidence:
            parts.append(evidence)
        parts.append(user_text or "")
        return "\n\n".join([p for p in parts if p])

    def _wrap_user_text_for_model(self, user_text: str) -> str:
        """Prefix user message with authoritative runtime state and compact meta-instructions."""
        try:
            state_line = self._state_reminder_line()
        except Exception:
            state_line = "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"

        try:
            lang = (getattr(getattr(self, 'gov_state', None), 'answer_language', '') or '').strip().lower()
            if not lang:
                lang = (getattr(cfg, 'get_answer_language', lambda: 'de')() or 'de').strip().lower()
            if lang not in ('de', 'en'):
                lang = 'de'
        except Exception:
            lang = 'de'

        evidence = ''
        try:
            comm = bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False))
            color = (getattr(getattr(self, 'gov_state', None), 'color', 'off') or 'off').strip().lower()
            if comm and color == 'on':
                evidence = (
                    "EVIDENCE-LINKER: For each atomic factual claim in the FINAL ANSWER, prefix exactly one tag: "
                    "[GREEN] for well-established knowledge, [YELLOW] for plausible/uncertain, [RED] for speculative. "
                    "Do not tag headers, SCI Trace, or QC-Matrix. Keep tags in the final answer."
                )
        except Exception:
            evidence = ''

        meta = f"[OUTPUT LANGUAGE] {lang}"
        parts = [state_line, meta]
        if evidence:
            parts.append(evidence)
        parts.append(user_text or '')
        return "\n\n".join([p for p in parts if p])

    def _wrap_user_text_for_model(self, user_text: str) -> str:
        """Prefix user message with authoritative runtime state and compact meta-instructions."""
        try:
            state_line = self._state_reminder_line()
        except Exception:
            state_line = "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"

        try:
            lang = (getattr(getattr(self, 'gov_state', None), 'answer_language', '') or '').strip().lower()
            if not lang:
                lang = (getattr(cfg, 'get_answer_language', lambda: 'de')() or 'de').strip().lower()
            if lang not in ('de', 'en'):
                lang = 'de'
        except Exception:
            lang = 'de'

        evidence = ''
        try:
            if bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False)) and (getattr(getattr(self, 'gov_state', None), 'color', 'off') == 'on'):
                evidence = (
                    "EVIDENCE-LINKER: For each atomic factual claim in the FINAL ANSWER, prefix exactly one tag: "
                    "[GREEN] for well-established knowledge, [YELLOW] for plausible/uncertain, [RED] for speculative. "
                    "Do not tag headers, SCI Trace, or QC-Matrix. Keep tags in the final answer."
                )
        except Exception:
            evidence = ''

        meta = f"[OUTPUT LANGUAGE] {lang}"
        parts = [state_line, meta]
        if evidence:
            parts.append(evidence)
        parts.append(user_text or '')
        return "\n\n".join([p for p in parts if p])

    def _wrap_user_text_for_model(self, user_text: str) -> str:
        """Prefix user message with authoritative runtime state and compact meta-instructions."""
        try:
            state_line = self._state_reminder_line()
        except Exception:
            state_line = "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"

        try:
            lang = (getattr(getattr(self, 'gov_state', None), 'answer_language', '') or '').strip().lower()
            if not lang:
                lang = (getattr(cfg, 'get_answer_language', lambda: 'de')() or 'de').strip().lower()
            if lang not in ('de', 'en'):
                lang = 'de'
        except Exception:
            lang = 'de'

        evidence = ''
        try:
            if bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False)) and (getattr(getattr(self, 'gov_state', None), 'color', 'off') == 'on'):
                evidence = (
                    "EVIDENCE-LINKER: For each atomic factual claim in the FINAL ANSWER, prefix exactly one tag: "
                    "[GREEN] for well-established knowledge, [YELLOW] for plausible/uncertain, [RED] for speculative. "
                    "Do not tag headers, SCI Trace, or QC-Matrix. Keep tags in the final answer."
                )
        except Exception:
            evidence = ''

        meta = f"[OUTPUT LANGUAGE] {lang}"
        parts = [state_line, meta]
        if evidence:
            parts.append(evidence)
        parts.append(user_text or '')
        return "\n\n".join([p for p in parts if p])

    def _wrap_user_text_for_model(self, user_text: str) -> str:
        """Prefix user message with authoritative runtime state and compact meta-instructions.

        Key goals:
        - Make the model respect the *current* runtime state (Profile/SCI/Overlay/Color).
        - If Color=on, strongly request Evidence-Linker tags so the UI can colorize.
        - Prevent the model from emitting internal scaffolding like "Profile: Standard" lines.
        """
        try:
            state_line = self._state_reminder_line()
        except Exception:
            state_line = "[CURRENT STATE] Profile=Standard | Overlay=off | SCI=off | Color=off | Comm=off"

        # Desired answer language (content only)
        try:
            lang = (getattr(getattr(self, 'gov_state', None), 'answer_language', '') or '').strip().lower()
            if not lang:
                lang = (getattr(cfg, 'get_answer_language', lambda: 'de')() or 'de').strip().lower()
            if lang not in ('de', 'en'):
                lang = 'de'
        except Exception:
            lang = 'de'

        # Evidence tags (only useful when Color=on)
        evidence = ''
        try:
            comm = bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False))
            color = (getattr(getattr(self, 'gov_state', None), 'color', 'off') or 'off').strip().lower()
            if comm and color == 'on':
                evidence = (
                    "EVIDENCE-LINKER (WHEN COLOR=ON): In the FINAL ANSWER, prefix substantive factual statements "
                    "with exactly ONE tag: [GREEN] well-established, [YELLOW] plausible/uncertain, [RED] speculative. "
                    "Prefer one tag per content sentence or bullet statement. "
                    "Do NOT tag pure section headings/list headers, SCI Trace, Self-Debunking, or QC-Matrix."
                )
        except Exception:
            evidence = ''

        retrieval_line = ""
        try:
            provider = (self._active_provider() or 'gemini').strip().lower()
            mode = self._native_retrieval_mode_for_provider(provider)
            supports = bool(self._provider_supports_native_retrieval(provider))
            active = bool(getattr(self, '_native_retrieval_active', False))
            if mode == 'off':
                retrieval_line = "[RETRIEVAL TOOL] disabled by wrapper configuration."
            elif supports and active:
                retrieval_line = "[RETRIEVAL TOOL] native_web_search=available."
            elif supports and (not active):
                retrieval_line = "[RETRIEVAL TOOL] capability available; session fallback currently without native tool."
            else:
                retrieval_line = (
                    "[RETRIEVAL TOOL] unavailable in current provider path. "
                    "Do NOT claim live retrieval; use uncertainty U5 when live verification is requested."
                )
        except Exception:
            retrieval_line = (
                "[RETRIEVAL TOOL] status unknown; do NOT claim live retrieval unless explicit tool output is present."
            )

        dont_echo = (
            "DO NOT OUTPUT INTERNAL SCAFFOLDING: Do not write lines like 'Profile: ...' or 'SCI: ...'. "
            "Follow the [CURRENT STATE] above silently."
        )

        parts = [state_line, f"[OUTPUT LANGUAGE] {lang}", retrieval_line, dont_echo]
        if evidence:
            parts.append(evidence)
        parts.append(user_text or '')
        return "\n\n".join([p for p in parts if isinstance(p, str) and p.strip()])


    def _llm_call(self, user_text: str, *, reason: str = 'chat', model_override: str = ''):
        """Single choke point for provider calls.

        Returns assistant text (string). Usage is provider-specific; stats remain best-effort.
        """
        provider = (self._active_provider() or 'gemini').strip().lower()

        try:
            self.log_event('provider_call_start', {'provider': provider, 'reason': reason})
        except Exception:
            pass

        # Gemini path: keep fix19 behavior (chat_session.send_message) to avoid breaking stability.
        if provider == 'gemini':
            t0 = time.time()
            if not getattr(self, 'chat_session', None):
                raise RuntimeError('No chat_session for Gemini provider')
            self._ensure_governance_pinned(reason=reason)
            ut = user_text
            try:
                if bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False)) and (getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip():
                    ut = self._wrap_user_with_sci(ut, variant=(getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip())
            except Exception:
                ut = user_text
            wrapped = self._wrap_user_text_for_model(ut)
            resp = self.chat_session.send_message(wrapped)
            try:
                ms = int((time.time() - t0) * 1000)
                self.last_call_info = {'provider': 'gemini', 'model': str(getattr(self, 'model_name', '') or ''), 'ms': ms, 'usage': {}}
                try:
                    self.log_event('provider_call_end', {'provider': 'gemini', 'ms': ms, 'model': str(getattr(self, 'model_name', '') or '')})
                except Exception:
                    pass
            except Exception:
                pass
            return getattr(resp, 'text', '') or ''

        # OpenAI-compatible providers path (OpenRouter / Hugging Face router)
        if provider in ('openrouter', 'openai', 'openai_compat', 'huggingface', 'hf'):
            pr = getattr(self, 'provider_router', None)
            psvc = getattr(self, 'provider_service', None)
            try:
                canon_pid = psvc.canonical_provider_id(provider) if psvc is not None else ('huggingface' if provider in ('huggingface', 'hf') else 'openrouter')
            except Exception:
                canon_pid = 'huggingface' if provider in ('huggingface', 'hf') else 'openrouter'
            client = None
            try:
                if psvc is not None:
                    client = psvc.get_openai_client(canon_pid)
                elif canon_pid == 'huggingface':
                    if pr is not None and hasattr(pr, 'build_huggingface_client'):
                        client = pr.build_huggingface_client()
                else:
                    if pr is not None and hasattr(pr, 'build_openrouter_client'):
                        client = pr.build_openrouter_client()
            except Exception:
                client = None
            if client is None or not getattr(client, 'api_key', ''):
                # Provider configured but no key found
                pname = 'Hugging Face' if canon_pid == 'huggingface' else 'OpenRouter'
                raise RuntimeError(f"{pname} client not configured (missing API key?)")

            # Choose model
            try:
                fallback = str(getattr(cfg, 'get_model', lambda: '')() or '')
            except Exception:
                fallback = ''
            prov_id = canon_pid
            models = []
            # Load provider-specific model candidates up-front so fallback can work
            # even when a configured default model is invalid for the active provider.
            try:
                if psvc is not None:
                    models, _meta = psvc.get_cached_models(canon_pid, force_refresh=False)
                elif canon_pid == 'huggingface':
                    if pr is not None and hasattr(pr, 'get_huggingface_models_cached'):
                        models, _meta = pr.get_huggingface_models_cached(force_refresh=False)
                    if (not models) and pr is not None and hasattr(pr, 'get_huggingface_models_from_config'):
                        models = pr.get_huggingface_models_from_config() or []
                else:
                    if pr is not None and hasattr(pr, 'get_openrouter_models_cached'):
                        models, _meta = pr.get_openrouter_models_cached(force_refresh=False)
            except Exception:
                models = []
            try:
                model = (
                    model_override
                    or (
                        psvc.get_provider_model(prov_id, fallback_model=fallback)
                        if psvc is not None
                        else self._provider_model(prov_id, fallback_model=fallback)
                    )
                    or ''
                ).strip()
            except Exception:
                model = (model_override or self._provider_model(prov_id, fallback_model=fallback) or '').strip()
            if not model:
                # Optional: auto-pick first model from cached /models list (best-effort)
                try:
                    if models:
                        model = str(models[0]).strip()
                except Exception:
                    model = ''
            if not model:
                model = 'zai-org/GLM-4.7:cerebras' if canon_pid == 'huggingface' else 'openai/gpt-4.1-mini'

            # IMPORTANT: OpenAI-compatible providers are stateless and do not automatically
            # retain our runtime governance state. Therefore we MUST prefix each user turn
            # with the authoritative runtime state line ([CURRENT STATE]) and output prefs,
            # just like the Gemini send_message() path.
            ut = user_text
            try:
                if bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False)) and (getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip():
                    ut = self._wrap_user_with_sci(ut, variant=(getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip())
            except Exception:
                ut = user_text

            # Resolve desired answer language for friendly provider errors (and UI).
            lang = None
            try:
                lang = getattr(getattr(self, 'gov_state', None), 'answer_language', None)
            except Exception:
                lang = None
            if not lang:
                try:
                    lang = getattr(self.cfg_mgr, 'get_answer_language', lambda: 'de')()
                except Exception:
                    lang = 'de'
            lang = (lang or 'de').strip().lower()
            if lang not in ('de', 'en'):
                lang = 'de'

            wrapped = self._wrap_user_text_for_model(ut)
            msgs = self._build_openai_messages(wrapped)
            # Robust call with fallback models and transient retry handling.
            # Some free "reasoning" models may return empty message.content; we treat that as an error
            # and fall back to other models without ever surfacing hidden reasoning fields.
            cand = []
            try:
                cand.append(model)
                # Optional explicit fallback list from config
                fb = psvc.get_config_fallback_models(prov_id) if psvc is not None else None
                if not isinstance(fb, list):
                    provs = (self.cfg_mgr.config or {}).get('providers') or {}
                    pconf = provs.get(prov_id) if isinstance(provs, dict) else {}
                    fb = (pconf or {}).get('fallback_models') if isinstance(pconf, dict) else None
                if isinstance(fb, list):
                    for x in fb:
                        sx = str(x or '').strip()
                        if sx and sx not in cand:
                            cand.append(sx)
            except Exception:
                pass
            try:
                # If current model is :free, prefer other :free models as fallbacks.
                if (model or '').endswith(':free'):
                    for m in (models or []):
                        sm = str(m or '').strip()
                        if sm and sm.endswith(':free') and sm not in cand:
                            cand.append(sm)
                else:
                    for m in (models or []):
                        sm = str(m or '').strip()
                        if sm and sm not in cand:
                            cand.append(sm)
            except Exception:
                pass

            # Keep attempts bounded.
            # Hugging Face catalogs can contain many provider-specific entries where early
            # candidates may fail with 4xx; allow a slightly wider search window there.
            _max_cand = 12 if provider in ('huggingface', 'hf') else 5
            cand = cand[:_max_cand] if isinstance(cand, list) else [model]

            last_err = None
            for mi, mname in enumerate(cand):
                # 429/backoff and one "bigger max_tokens" retry for empty completion
                for attempt in range(3):
                    t0 = time.time()
                    try:
                        # On second attempt for empty completion, allow a larger max_tokens budget.
                        mx = 1024
                        if attempt >= 1:
                            mx = 2048
                        txt, _usage = client.chat(messages=msgs, model=mname, max_tokens=mx, lang=lang)
                        try:
                            ms = int((time.time() - t0) * 1000)
                            self.last_call_info = {'provider': 'openrouter', 'model': mname, 'ms': ms, 'usage': _usage or {}}
                            try:
                                self.log_event('provider_call_end', {'provider': provider, 'ms': ms, 'model': mname})
                            except Exception:
                                pass
                        except Exception:
                            pass
                        return txt or ''
                    except Exception as e:
                        err_s = str(e)
                        last_err = err_s
                        # Upstream rate limit: backoff then retry same model.
                        if (' 429 ' in err_s) or ('rate-limited' in err_s.lower()) or ('rate limited' in err_s.lower()) or ('temporarily rate-limited' in err_s.lower()):
                            try:
                                delay = [0.5, 1.5, 3.5][min(attempt, 2)]
                                time.sleep(delay)
                            except Exception:
                                pass
                            continue
                        # Empty completion: try once more with larger max_tokens, then fall back to next model.
                        if 'empty completion' in err_s.lower() or 'no content' in err_s.lower():
                            if attempt < 1:
                                continue
                            break
                        # Any other error: stop retrying this model and fall back.
                        break
                # Try next model

            raise RuntimeError(last_err or 'OpenRouter request failed (no usable completion)')

        # Unknown provider
        raise RuntimeError(f'Unknown provider: {provider}')


    def _render_profile_switch_control_html(self, timestamp: str, audit_line: str = "") -> str:


        """Minimal deterministic control output for Profile switches.

        Requirements:
        - First: header line
        - Then: QC-Matrix line
        - Last (for this control block): timestamp footer
        - No redundant parameter table.
        """
        if not getattr(gov, 'loaded', False):
            return '<div class="comm-help comm-state">No ruleset loaded.</div>'

        try:
            ver = (gov.data or {}).get('version', '')
            sysname = (gov.data or {}).get('system_name', 'Comm-SCI-Control')
        except Exception:
            ver, sysname = '', 'Comm-SCI-Control'

        prof = getattr(getattr(self, 'gov_state', None), 'active_profile', 'Standard') or 'Standard'
        overlay = getattr(getattr(self, 'gov_state', None), 'overlay', '') or 'off'
        sci = getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or ''
        color = getattr(getattr(self, 'gov_state', None), 'color', 'off') or 'off'
        disp_color = 'off' if prof in {'Sandbox', 'Briefing'} else color

        header = (
            f"Active profile: {prof} · SCI: {sci or 'off'} · Overlay: {overlay or 'off'} · "
            f"Control Layer: on · QC: on · CGI: on · Color: {disp_color}"
        )

        qc_line = ''
        try:
            qc_line = self._qc_footer_for_profile(prof)
        except Exception:
            qc_line = 'QC-Matrix: Clarity 3 (Δ0) · Brevity 2 (Δ0) · Evidence 2 (Δ0) · Empathy 2 (Δ0) · Consistency 3 (Δ0) · Neutrality 3 (Δ0)'

        out = []
        out.append('<div class="comm-help comm-state">')
        out.append(f'<div class="help-status">{html.escape(header)}</div>')
        if audit_line:
            out.append(f"<div style='margin-top:8px'>{html.escape(str(audit_line))}</div>")
        out.append(f"<div style='margin-top:10px'>{html.escape(qc_line)}</div>")
        out.append('</div>')
        out.append(f'<div class="ts-footer">Response at {html.escape(str(timestamp))}</div>')
        return "\n".join(out)

    def ask(self, txt):
        # NOTE: Profile switch outputs should be minimal (header -> QC-Matrix -> timestamp).
        # SECURITY: Sicherer Import
        import html as h_lib

        def _session_stamp():
            try:
                return (
                    getattr(self.gov_state, "active_profile", "Standard") or "Standard",
                    getattr(self.gov_state, "overlay", "") or "",
                    getattr(self.gov_state, "color", "off") or "off",
                    bool(getattr(self.gov_state, "sci_active", False)),
                    getattr(self.gov_state, "sci_variant", "") or "",
                    getattr(self.gov_state, "conversation_language", "") or "",
                    bool(getattr(self.gov_state, "comm_active", False)),
                    bool(getattr(self.gov_state, "dynamic_one_shot_active", False)),
                )
            except Exception:
                return ("Standard", "", "off", False, "", "", False, False)

        try:
            timestamp = _format_response_timestamp()
            raw_txt = txt or ""
            self._last_response_is_content_answer = False

            # STUFE 0: lightweight observability (never raises)
            try:
                self.log_event(
                    'input',
                    {
                        'len': len(raw_txt),
                        'sha': _safe_sha256(raw_txt),
                        'preview': _safe_preview_text(raw_txt, 160),
                    },
                )
            except Exception:
                pass

            # QC-Override modal gate for Main input:
            # while the dialog is open, Main input must remain locked.
            try:
                if self._is_qc_override_modal_active():
                    _raw = str(raw_txt or '').strip()
                    if _raw == "QC Override":
                        self._bring_qc_override_to_front()
                        msg = (
                            "<div class='sys'>QC Override ist bereits geöffnet.</div>"
                            if str(self._lang() or "de").lower() != "en"
                            else "<div class='sys'>QC Override is already open.</div>"
                        )
                        try:
                            self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                            self.history.append({"role": "bot", "content": msg, "ts": datetime.now().isoformat(), "csc": None})
                        except Exception:
                            pass
                        return {"html": msg, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
                    self._bring_qc_override_to_front()
                    try:
                        self.log_event('qc_override_modal_block', {'source': 'ask', 'preview': _safe_preview_text(raw_txt, 80)})
                    except Exception:
                        pass
                    _blocked_html = self._qc_override_modal_block_html()
                    try:
                        self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                        self.history.append({"role": "bot", "content": _blocked_html, "ts": datetime.now().isoformat(), "csc": None})
                    except Exception:
                        pass
                    return {"html": _blocked_html, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
            except Exception:
                pass

            # Exit-confirm modal gate for Main/Panel-triggered asks:
            # while the exclusive exit dialog is open, all new actions are blocked.
            try:
                if self._is_exit_confirm_modal_active():
                    try:
                        self.log_event('exit_confirm_block', {'source': 'ask', 'preview': _safe_preview_text(raw_txt, 80)})
                    except Exception:
                        pass
                    _blocked_html = self._exit_confirm_block_html()
                    try:
                        self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                        self.history.append({"role": "bot", "content": _blocked_html, "ts": datetime.now().isoformat(), "csc": None})
                    except Exception:
                        pass
                    return {"html": _blocked_html, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
            except Exception:
                pass

            # 1. Routing
            route = route_input(raw_txt, self.gov_state, self)
            # STUFE 1 contract: route shape (fail-soft; no behavior change)
            try:
                if not contract_route_shape(route):
                    try:
                        self.log_event('contract_violation', {'where': 'route_input', 'kind': str(route.get('kind'))})
                    except Exception:
                        pass
            except Exception:
                pass


            try:
                self.log_event(
                    'route',
                    {
                        'kind': route.get('kind'),
                        'is_command': bool(route.get('kind') == 'command'),
                        'is_sci_selection': bool(route.get('is_sci_selection')),
                        'standalone_only_violation': bool(route.get('standalone_only_violation')),
                    },
                )
            except Exception:
                pass

            if route["kind"] == "error":
                self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                self.history.append({"role": "bot", "content": "Blocked.", "ts": datetime.now().isoformat()})
                try:
                    self.log_event('blocked', {'reason': _safe_preview_text(route.get('html') or '', 120)})
                except Exception:
                    pass
                return {"html": route["html"], "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

            # SCI Selection (A-H)
            if route.get("is_sci_selection"):
                self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                try:
                    self.log_event('sci_selection', {'value': _safe_preview_text(route.get('query_text') or '', 16)})
                except Exception:
                    pass
                sci_res = self._handle_sci_selection(route["query_text"])
                if isinstance(sci_res, dict):
                    sci_res.setdefault("cgi_bar", False)
                    sci_res.setdefault("answer_lang", self._answer_lang())
                    return sci_res
                return {"html": str(sci_res or ""), "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

            if route["kind"] == "noop":
                return {"html": "", "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

            # 2. Commands
            if route["kind"] == "command":
                self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                cmd = route["canonical_cmd"]
                prev_profile_for_audit = (getattr(self.gov_state, "active_profile", "Standard") or "Standard")

                try:
                    self.log_event('command', {'cmd': cmd, 'phase': 'begin'})
                except Exception:
                    pass

                # Special Renderers (help/state/config/audit/anchor etc.)
                handled_res = self._handle_command_deterministic(cmd, timestamp)
                if handled_res:
                    try:
                        self._ui_refresh_panel()
                    except Exception:
                        pass
                    try:
                        self.log_event('command', {'cmd': cmd, 'phase': 'deterministic'})
                    except Exception:
                        pass
                    if isinstance(handled_res, dict):
                        handled_res.setdefault("cgi_bar", False)
                        handled_res.setdefault("answer_lang", self._answer_lang())
                        return handled_res
                    return {"html": str(handled_res or ""), "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

                # State Change (Phase 3): explicit intent dispatch at the router boundary.
                _applied_via_intent = False
                try:
                    if _controller_dispatch_intent is not None and _intent_from_command is not None:
                        _intent = _intent_from_command(cmd)
                        if _intent is not None:
                            _gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                            _data = getattr(_gov_obj, 'data', {}) if _gov_obj is not None else {}

                            def _mirror():
                                gov_obj2 = getattr(self, 'gov', None) or globals().get('gov')
                                if gov_obj2 is not None:
                                    if hasattr(self.gov_state, 'qc_overrides'):
                                        setattr(gov_obj2, 'qc_overrides', dict(getattr(self.gov_state, 'qc_overrides', {}) or {}))
                                    gov.runtime_state = self.gov_state
                                    setattr(gov_obj2, 'runtime_state', self.gov_state)

                            _outcome = _controller_dispatch_intent(
                                intent=_intent,
                                cmd=cmd,
                                runtime_state=self.gov_state,
                                ruleset_data=_data if isinstance(_data, dict) else {},
                                mirror_callback=_mirror,
                            )
                            _applied_via_intent = bool(_outcome.applied)
                except Exception:
                    _applied_via_intent = False

                if not _applied_via_intent:
                    self._execute_legacy_command(cmd)

                try:
                    self.log_event('command', {'cmd': cmd, 'phase': 'state_changed'})
                except Exception:
                    pass

                # After state change: update the model state WITHOUT recreating the session (huge token savings)
                try:
                    if cmd == 'Comm Stop':
                        self._recreate_chat_session(with_governance=False, reason='Comm Stop')
                        self._gov_pinned_sent = False

                        self._gov_pinned_fp = ''
                    elif cmd == 'Comm Start':
                        self._recreate_chat_session(with_governance=True, reason='Comm Start')
                        self._gov_pinned_sent = False

                        self._gov_pinned_fp = ''
                        self._ensure_governance_pinned(reason='Comm Start')
                    else:
                        self._ensure_governance_pinned(reason=f'cmd:{cmd}')
                        self._send_state_update_to_model(reason=f'cmd:{cmd}')
                    self._last_session_stamp = _session_stamp()
                except Exception:
                    pass

                # Response after command execution
                # - For Profile switches: minimal control output (header -> QC-Matrix -> timestamp)
                # - Additionally, ONLY for Profile Sparring/Expert: show SCI menu to choose variant
                current_profile = getattr(self.gov_state, "active_profile", "")
                comm_start_audit_line = ""
                if cmd == "Comm Start" and str(current_profile) != str(prev_profile_for_audit):
                    comm_start_audit_line = _build_profile_switch_audit_line(
                        "Comm Start",
                        str(prev_profile_for_audit),
                        str(current_profile),
                    )
                profile_switch_audit_line = ""
                if cmd.startswith("Profile ") and str(current_profile) != str(prev_profile_for_audit):
                    profile_switch_audit_line = _build_profile_switch_audit_line(
                        cmd,
                        str(prev_profile_for_audit),
                        str(current_profile),
                    )

                if cmd.startswith("Profile "):
                    html_content = self._render_profile_switch_control_html(timestamp, audit_line=profile_switch_audit_line)
                    if current_profile in ["Expert", "Sparring"]:
                        try:
                            self.gov_state.sci_pending = True
                        except Exception:
                            pass
                        # Show SCI menu as an additional block (after the control output)
                        menu_html = self._render_sci_menu_html(lang=self._lang())
                        html_content = html_content + "\n<div style='margin-top:12px'></div>\n" + menu_html
                else:
                    triggered_sci = (cmd in ["SCI on", "SCI menu"])
                    if triggered_sci:
                        try:
                            self.gov_state.sci_pending = True
                        except Exception:
                            pass
                        html_content = self._render_sci_menu_html(lang=self._lang())
                    else:
                        html_content = self._render_comm_state_html(audit_line=comm_start_audit_line)

                self.history.append({"role": "bot", "content": f"Command executed: {cmd}", "ts": datetime.now().isoformat()})
                try:
                    self._ui_refresh_panel()
                except Exception:
                    pass
                try:
                    self.log_event('command', {'cmd': cmd, 'phase': 'end', 'profile': getattr(self.gov_state, 'active_profile', '')})
                except Exception:
                    pass
                return {"html": html_content, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

            # 3. Chat (Normal Question)
            self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})

            # CGI user feedback triplets are optional and must not be forwarded as a standalone LLM request.
            try:
                if bool(route.get('is_user_feedback_triplet')) or bool(route.get('is_process_cgi_feedback')):
                    fb = (raw_txt or '').strip()
                    try:
                        if bool(route.get('is_user_feedback_triplet')):
                            self.gov_state.last_user_feedback_triplet = fb
                        if bool(route.get('is_process_cgi_feedback')):
                            self.gov_state.last_process_cgi_feedback = fb
                        self.gov_state.cgi_feedback_pending_for_model = True
                    except Exception:
                        pass
                    # Log event (best-effort)
                    try:
                        self.log_event('cgi_feedback', {'value': _safe_preview_text(fb, 32), 'kind': 'user_feedback_triplet' if bool(route.get('is_user_feedback_triplet')) else 'process_cgi_feedback'})
                    except Exception:
                        pass
                    ts = _format_response_timestamp()
                    note = (
                        "<div style='border:1px solid #bbf7d0; background:#f0fdf4; padding:10px; "
                        "border-radius:10px; margin:8px 0; color:#166534;'>"
                        "<b>CGI feedback recorded.</b><br>"
                        + html.escape(fb) +
                        "</div>"
                    )
                    return {"html": note + f"<div class='ts-footer'>Response at {html.escape(ts)}</div>", "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
            except Exception:
                pass

            # Turn counter (used for anchor auto snapshots, and as a general monotonic user-turn index)
            try:
                self.gov_state.user_turns = int(getattr(self.gov_state, 'user_turns', 0) or 0) + 1
            except Exception:
                pass

            # --- SCI pending: timeout + extension-condition (canonical JSON) ---
            # If the user does not select a variant (A–H) while SCI menu is pending, we either:
            # - keep pending for one extra turn IF the input is a contextual query about SCI methodology,
            # - otherwise assume variant A and continue (with a deterministic note).
            sci_note_html = ""

            # Standalone-only violation notice (deterministic; does not block the chat).
            # Example: "Profile Expert what is time?" must NOT execute "Profile Expert".
            try:
                if bool(route.get('standalone_only_violation')):
                    bad_cmd = str(route.get('standalone_violation_cmd') or '').strip()
                    msg = "Standalone-only rule: command tokens must be sent as standalone commands."
                    if bad_cmd:
                        msg += f" Detected mixed command token: {bad_cmd!r}. Interpreting as chat."
                    sci_note_html = (
                        "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                        "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                        "<b>CONTROL LAYER ALERT (Parser)</b><br>" + html.escape(msg) +
                        "</div>"
                    )
            except Exception:
                pass
            try:
                if bool(getattr(self.gov_state, 'sci_pending', False)) and not route.get('is_sci_selection'):
                    txt_clean = (raw_txt or '').strip()

                    # Canonical JSON lookup
                    gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                    data = getattr(gov_obj, 'data', {}) if gov_obj is not None else {}
                    svs = (((data.get('syntax_rules') or {}).get('special_parsing') or {}).get('sci_variant_selection') or {})
                    timeout_turns = int(svs.get('timeout_turns', 2) or 2)
                    timeout_turns_ext = int(svs.get('timeout_turns_extended', 3) or 3)

                    ext_cond = (svs.get('extension_condition') or {}) if isinstance(svs, dict) else {}
                    ext_keywords = []
                    if isinstance(ext_cond, dict):
                        ext_keywords = ext_cond.get('keywords_any', []) or []
                    if not ext_keywords:
                        ext_keywords = ["sci", "variant", "mode", "trace", "steps", "deep", "dive"]

                    is_contextual = any(str(k).lower() in txt_clean.lower() for k in ext_keywords)

                    # increment pending turns
                    try:
                        self.gov_state.sci_pending_turns = int(getattr(self.gov_state, 'sci_pending_turns', 0) or 0) + 1
                    except Exception:
                        self.gov_state.sci_pending_turns = 1

                    max_turns = timeout_turns_ext if is_contextual else timeout_turns

                    if self.gov_state.sci_pending_turns < max_turns and is_contextual:
                        # Keep pending and show clarification + menu deterministically (no LLM call)
                        lang = self._lang()
                        if str(lang).lower().startswith('de'):
                            note_title = 'SCI-Auswahl ausstehend.'
                            note_body = (
                                'Deine Eingabe wirkt wie eine Frage zur SCI-Methodik. '
                                'Bitte wähle eine SCI-Variante (A–H), um fortzufahren.'
                            )
                        else:
                            note_title = 'SCI selection pending.'
                            note_body = (
                                'Your input looks like a question about the SCI methodology. '
                                'Please select an SCI variant (A–H) to continue.'
                            )
                        note = (
                            "<div style='border:1px solid #c7d2fe; background:#eef2ff; padding:10px; "
                            "border-radius:10px; margin:8px 0; color:#1e3a8a;'>"
                            f"<b>{note_title}</b><br>"
                            f"{note_body}"
                            "</div>"
                        )
                        menu_html = self._render_sci_menu_html(lang=self._lang())
                        return {"html": note + "\n" + menu_html, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}

                    if self.gov_state.sci_pending_turns >= max_turns:
                        # Timeout fallback → assume Variant A
                        try:
                            self.gov_state.sci_pending = False
                            self.gov_state.sci_active = True
                            self.gov_state.sci_variant = 'A'
                            self.gov_state.sci_pending_turns = 0
                        except Exception:
                            pass
                        # Ensure model sees the new state (no session recreation)
                        try:
                            self._ensure_governance_pinned(reason='SCI pending timeout -> A')
                            self._send_state_update_to_model(reason='SCI pending timeout -> A')
                            self._last_session_stamp = _session_stamp()
                        except Exception:
                            pass

                        sci_note_html = (
                            "<div style='border:1px solid #fed7aa; background:#fff7ed; padding:10px; "
                            "border-radius:10px; margin:8px 0; color:#9a3412;'>"
                            "<b>Note:</b> SCI variant selection was not provided in time → assumed variant A."
                            "</div>"
                        )
            except Exception:
                pass

            # Ensure session exists AND matches current runtime state
            provider_now = ''
            try:
                provider_now = (self._active_provider() or 'gemini').strip().lower()
            except Exception:
                provider_now = 'gemini'

            try:
                if provider_now == "gemini" and (not getattr(self, "chat_session", None)):
                    self._recreate_chat_session(with_governance=True, reason="no_session")
                    self._last_session_stamp = _session_stamp()
                else:
                    cur = _session_stamp()
                    last = getattr(self, "_last_session_stamp", None)
                    if last is None:
                        # First turn in this session: remember stamp, but don't send a STATE UPDATE (avoids extra LLM call).
                        self._last_session_stamp = cur
                    elif last != cur:
                        self._ensure_governance_pinned(reason='state_changed')
                        self._send_state_update_to_model(reason='state_changed')
                        self._last_session_stamp = cur
            except Exception:
                pass

            # Ensure canonical governance is pinned once (saves huge system-instruction tokens)
            try:
                self._ensure_governance_pinned(reason='pre_send')
            except Exception:
                pass

            # Send (CSC enforcement may wrap the user message deterministically)
            # Snapshot one-shot flags that must be auto-reset ONLY if they were active for THIS request.
            dynamic_was_active = bool(getattr(self.gov_state, 'dynamic_one_shot_active', False))

            # Cross-version leak guard (ignore foreign Comm-SCI versions in user input; keep active version only)
            raw_txt_for_model = raw_txt
            try:
                import re
                active_version = str(((self.gov.data or {}).get('version') or '')).strip()
                active_token = ''
                if active_version:
                    m = re.search(r"\b\d+\.\d+\.\d+\b", active_version)
                    active_token = m.group(0) if m else active_version
                found = sorted(set(re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", raw_txt or "")))
                foreign = [v for v in found if active_token and v != active_token]
                if foreign:
                    setattr(self.gov_state, 'cross_version_guard_hits', list(foreign))
                    for v in foreign:
                        raw_txt_for_model = raw_txt_for_model.replace(v, active_token or v)
            except Exception:
                raw_txt_for_model = raw_txt

            user_for_model = self._apply_output_prefs_to_user_message(raw_txt_for_model)

            # If CGI feedback was provided earlier, forward it ONCE as a neutral note (no rule/code changes).
            try:
                if bool(getattr(self.gov_state, 'cgi_feedback_pending_for_model', False)):
                    fb_parts = []
                    lu = (getattr(self.gov_state, 'last_user_feedback_triplet', '') or '').strip()
                    lp = (getattr(self.gov_state, 'last_process_cgi_feedback', '') or '').strip()
                    if lu:
                        fb_parts.append(f"User feedback triplet (CGI): {lu}")
                    if lp:
                        fb_parts.append(f"Process CGI feedback: {lp}")
                    extra_blocks = []
                    if fb_parts:
                        extra_blocks.append("[CGI Feedback]\n" + "\n".join(fb_parts))
                    if lu:
                        cgi_constraints = self._build_cgi_one_shot_constraints(lu, lang=self._answer_lang())
                        if cgi_constraints:
                            extra_blocks.append("[CGI One-Shot Rewrite Constraints]\n" + "\n".join(cgi_constraints))
                    if extra_blocks:
                        user_for_model = (user_for_model.rstrip() + "\n\n" + "\n\n".join(extra_blocks)).strip()
                    # Mark as sent (single-use)
                    self.gov_state.cgi_feedback_pending_for_model = False
            except Exception:
                pass

            # SCI Recursion: capture scope for next answer (canonical JSON: sci.recursive_sci)
            try:
                if bool(getattr(self.gov_state, 'sci_recursion_one_shot', False)):
                    scope = (raw_txt or '').strip()
                    if len(scope) > 180:
                        scope = scope[:177] + '...'
                    self.gov_state.sci_recursion_scope = scope
            except Exception:
                pass

            send_txt, pre_meta = self._csc_wrap_user_message(user_for_model)
            # Rate limiting (LLM calls only)
            try:
                if bool(getattr(self, 'rate_limit_enabled', True)) and getattr(self, 'rate_limiter', None) is not None:
                    _provider_rl = (provider_now or 'gemini').strip().lower()
                    _model_rl = ''
                    try:
                        if _provider_rl == 'gemini':
                            _model_rl = str(getattr(self, 'model_name', '') or '')
                            if not _model_rl:
                                _model_rl = str(getattr(cfg, 'get_model', lambda: '')() or '')
                        elif _provider_rl in ('openrouter', 'openai', 'openai_compat'):
                            fb = str(getattr(cfg, 'get_model', lambda: '')() or '')
                            _model_rl = (self._provider_model('openrouter', fallback_model=fb) or '').strip()
                        elif _provider_rl in ('huggingface', 'hf'):
                            fb = str(getattr(cfg, 'get_model', lambda: '')() or '')
                            _model_rl = (self._provider_model('huggingface', fallback_model=fb) or '').strip()
                            self.session_requests = int(getattr(self, 'session_requests', 0) or 0) + 1
                    except Exception:
                        _model_rl = ''

                    ok, msg, retry_s = self.rate_limiter.allow_call(provider=_provider_rl, model=_model_rl, reason='chat', consume=True, return_retry=True)
                    if not ok:
                        try:
                            self.session_rate_limit_hits = int(getattr(self, 'session_rate_limit_hits', 0) or 0) + 1
                            self.session_events.append({'ts': datetime.now().isoformat(), 'type': 'rate_limit_hit', 'data': {'message': msg}})
                        except Exception:
                            pass
                        ts = _format_response_timestamp()
                        warn = (
                            "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                            "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                            "<b>CONTROL LAYER BLOCK:</b><br>" + html.escape(str(msg)) +
                                        "<br><span style='font-size:12px; color:#7f1d1d;'>Retry after " + html.escape(str(retry_s)) + "s</span>" +
                            "<br><span style='font-size:12px; color:#7f1d1d;'>"
                            "Tip: adjust limits in Config/Comm-SCI-Config.json (rate_limit_per_minute / rate_limit_per_hour; optional rate_limit_scopes)"
                            "</span></div>"
                        )
                        return {"html": warn + f"<div class='ts-footer'>Response at {html.escape(ts)}</div>", "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
            except Exception:
                pass


            raw_resp = self._llm_call(send_txt, reason="chat")


            # Session token stats (best-effort; whitespace-token approximation)
            try:
                self.session_req_count = int(getattr(self, 'session_req_count', 0) or 0) + 1
                self.session_tokens_in = int(getattr(self, 'session_tokens_in', 0) or 0) + int(self.count_ws_tokens(send_txt))
                self.session_tokens_out = int(getattr(self, 'session_tokens_out', 0) or 0) + int(self.count_ws_tokens(raw_resp))
                try:
                    self.update_stats_ui()
                except Exception:
                    pass
            except Exception:
                pass

                        # --- Normalize RAW model output for validation (plain text only) ---
            repaired_raw = raw_resp
            governance_enabled_now = (
                bool(getattr(self, 'session_with_governance', True))
                and bool(getattr(self.gov_state, 'comm_active', False))
            )
            repaired_raw = self._normalize_raw_output_contracts(
                repaired_raw,
                governance_enabled=governance_enabled_now,
                is_command=False,
            )

            # --- Validate + ONE repair pass for HARD violations (on RAW text, not HTML) ---
            repair_banner_html = ""
            meta = None
            try:
                validator = getattr(self, 'validator', None)
                if not governance_enabled_now:
                    validator = None
                if validator is not None:
                    # Expect SCI trace iff the selected variant actually has required steps (A typically has none).
                    vk = getattr(self.gov_state, 'sci_variant', '') or ''
                    steps = []
                    try:
                        if bool(getattr(self.gov_state, 'sci_active', False)) and hasattr(validator, '_required_trace_steps_for_variant'):
                            steps = validator._required_trace_steps_for_variant(vk) or []
                    except Exception:
                        steps = []
                    # Expect SCI trace whenever SCI is active and a variant is selected.
                    try:
                        expect_trace = bool(getattr(self.gov_state, "sci_active", False)) and bool(vk)
                    except Exception:
                        expect_trace = bool(steps)

                    hard_vios, soft_vios = validator.validate(
                        text=repaired_raw,
                        state=self.gov_state,
                        expect_menu=False,
                        expect_trace=expect_trace,
                        is_command=False,
                        user_prompt=raw_txt,
                    )

                    if hard_vios:
                        try:
                            self.session_repair_passes = int(getattr(self, 'session_repair_passes', 0) or 0) + 1
                            self.session_events.append({'ts': datetime.now().isoformat(), 'type': 'repair_pass', 'data': {'violations': list(hard_vios)}})
                        except Exception:
                            pass
                        # Exactly ONE repair pass via the model.
                        repair_prompt = validator.build_repair_prompt(
                            user_prompt=raw_txt,
                            raw_response=repaired_raw,
                            state=self.gov_state,
                            hard_violations=hard_vios,
                            soft_violations=soft_vios,
                        )
                        # Respect answer-language preference.
                        repair_for_model = self._apply_output_prefs_to_user_message(repair_prompt)
                        # Rate limiting (repair pass counts as an extra LLM call)
                        try:
                            if bool(getattr(self, 'rate_limit_enabled', True)) and getattr(self, 'rate_limiter', None) is not None:
                                _provider_rl = (provider_now or 'gemini').strip().lower()
                                _model_rl = ''
                                try:
                                    if _provider_rl == 'gemini':
                                        _model_rl = str(getattr(self, 'model_name', '') or '')
                                        if not _model_rl:
                                            _model_rl = str(getattr(cfg, 'get_model', lambda: '')() or '')
                                    elif _provider_rl in ('openrouter', 'openai', 'openai_compat'):
                                        fb = str(getattr(cfg, 'get_model', lambda: '')() or '')
                                        _model_rl = (self._provider_model('openrouter', fallback_model=fb) or '').strip()
                                    elif _provider_rl in ('huggingface', 'hf'):
                                        fb = str(getattr(cfg, 'get_model', lambda: '')() or '')
                                        _model_rl = (self._provider_model('huggingface', fallback_model=fb) or '').strip()
                                except Exception:
                                    _model_rl = ''

                                ok, msg, retry_s = self.rate_limiter.allow_call(provider=_provider_rl, model=_model_rl, reason='repair', consume=True, return_retry=True)
                                if not ok:
                                    ts = _format_response_timestamp()
                                    warn = (
                                        "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                                        "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                                        "<b>CONTROL LAYER BLOCK:</b><br>" + html.escape(str(msg)) +
                                        "<br><span style='font-size:12px; color:#7f1d1d;'>"
                                        "Tip: adjust limits in Config/Comm-SCI-Config.json (rate_limit_per_minute / rate_limit_per_hour; optional rate_limit_scopes)"
                                        "</span></div>"
                                    )
                                    return {"html": warn + f"<div class='ts-footer'>Response at {html.escape(ts)}</div>", "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
                        except Exception:
                            pass
                            self.session_requests = int(getattr(self, 'session_requests', 0) or 0) + 1

                        raw2 = self._llm_call(repair_for_model, reason="repair")

                        # Session token stats (repair pass)
                        try:
                            self.session_req_count = int(getattr(self, 'session_req_count', 0) or 0) + 1
                            self.session_tokens_in = int(getattr(self, 'session_tokens_in', 0) or 0) + int(self.count_ws_tokens(repair_for_model))
                            self.session_tokens_out = int(getattr(self, 'session_tokens_out', 0) or 0) + int(self.count_ws_tokens(raw2))
                            try:
                                self.update_stats_ui()
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Normalize again (raw text)
                        repaired_raw = raw2
                        repaired_raw = self._normalize_raw_output_contracts(
                            repaired_raw,
                            governance_enabled=governance_enabled_now,
                            is_command=False,
                        )

                        # Banner (visible; does not claim perfection beyond one pass)
                        note_tip_attr = ""
                        try:
                            note_tip = html.escape(
                                _control_layer_tooltip_text(lang=self._answer_lang(), severity="warn"),
                                quote=True,
                            )
                            if note_tip:
                                note_tip_attr = f" data-u-title='{note_tip}' style='cursor:help;'"
                        except Exception:
                            note_tip_attr = ""
                        try:
                            if _should_show_repair_pass_banner(hard_vios):
                                items = "".join([f"<li class='control-layer-violation'>{html.escape(str(v))}</li>" for v in hard_vios])
                                repair_banner_html = (
                                    f"<div class='control-layer-note csc-warning' style='border:1px solid #f59e0b; background:#fffbeb; padding:10px; "
                                    "border-radius:10px; margin:8px 0; color:#92400e;'>"
                                    f"<b{note_tip_attr}>CONTROL LAYER NOTE</b><br>One repair pass was applied for hard contract violations."
                                    f"<ul class='control-layer-violations' style='margin:6px 0 0 18px; padding:0;'>{items}</ul></div>"
                                )
                            else:
                                repair_banner_html = ""
                        except Exception:
                            repair_banner_html = (
                                f"<div class='control-layer-note csc-warning' style='border:1px solid #f59e0b; background:#fffbeb; padding:10px; "
                                "border-radius:10px; margin:8px 0; color:#92400e;'>"
                                f"<b{note_tip_attr}>CONTROL LAYER NOTE</b><br>One repair pass was applied for hard contract violations."
                                "</div>"
                            )
            except Exception:
                pass

            # --- Render ONCE (CSC renderer produces final HTML) ---
            final_work, meta = self._apply_csc_strict(repaired_raw, user_raw=raw_txt, is_command=False)
            try:
                if isinstance(meta, dict) and meta.get('applied'):
                    self.session_csc_applied_count = int(getattr(self, 'session_csc_applied_count', 0) or 0) + 1
            except Exception:
                pass

            # Track renderer outcome per session (for diagnostics; does not change UI output)
            try:
                if isinstance(meta, dict):
                    ns = meta.get("normalization")
                    if isinstance(ns, dict):
                        if bool(ns.get("render_ok")):
                            self.session_render_ok_count = int(getattr(self, "session_render_ok_count", 0) or 0) + 1
                        elif bool(ns.get("render_fallback")):
                            self.session_render_fallback_count = int(getattr(self, "session_render_fallback_count", 0) or 0) + 1
            except Exception:
                pass

            # If CSC was applied on prompt-side, prefer that metadata when renderer didn't produce any.
            if (meta is None) and pre_meta is not None:
                meta = pre_meta
# 4) Persist history + render
            # Prepend the repair banner if present.
            if repair_banner_html:
                try:
                    final_work = repair_banner_html + final_work
                except Exception:
                    pass


            # Cross-Version Guard: if user text contained foreign Comm-SCI version tokens, show a deterministic alert.
            try:
                _hits = list(getattr(self.gov_state, 'cross_version_guard_hits', []) or [])
                try:
                    self.session_guard_hits = int(getattr(self, 'session_guard_hits', 0) or 0) + len(_hits)
                except Exception:
                    pass
                if _hits and bool(getattr(self.gov_state, 'comm_active', False)):
                    _hits_s = ', '.join(str(x) for x in _hits)
                    _active_v = str((self.gov.data or {}).get('version', '') or '').strip()
                    crossv_html = (
                        "<div class='csc-warning' style='background:#fff7ed; border:1px solid #fb923c; padding:10px; "
                        "border-radius:10px; margin:8px 0; color:#9a3412;'>"
                        "<b>CONTROL LAYER ALERT</b><br><b>Cross-Version Guard</b>: "
                        f"Ignored foreign version token(s) in user input. Active: {_active_v}."
                        "</div>"
                    )
                    final_work = crossv_html + final_work
            except Exception:
                pass
            try:
                final_work = self._append_uncertainty_explanation_if_needed(final_work, user_text=raw_txt)
            except Exception:
                pass
            self.history.append({"role": "bot", "content": final_work, "ts": datetime.now().isoformat(), "csc": meta})

            # Use the repaired/enforced text for rendering.
            final = final_work
# Auto-reset: Dynamic one-shot (canonical JSON)
            try:
                if bool(dynamic_was_active):
                    self.gov_state.dynamic_one_shot_active = False
                    # legacy compatibility
                    if getattr(self.gov_state, 'dynamic_nudge', '') == 'one-shot':
                        self.gov_state.dynamic_nudge = ""
                    # update model state without recreating the session
                    try:
                        self._send_state_update_to_model(reason='Dynamic one-shot auto-reset')
                        self._last_session_stamp = _session_stamp()
                    except Exception:
                        pass
            except Exception:
                pass

            # Dynamic prompting auto-activation (best-effort): if QC deltas repeatedly exceed the JSON threshold,
            # enable dynamic_one_shot_active for the NEXT answer.
            try:
                _prof_now = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
                cur_qc, _rep = gov.parse_qc_footer(final_work)
                if isinstance(cur_qc, dict) and cur_qc:
                    try:
                        exp_delta = gov.expected_qc_deltas(_prof_now, cur_qc, overrides=getattr(self.gov_state, "qc_overrides", {})) or {}
                    except Exception:
                        exp_delta = {}
                    try:
                        self.gov_state.last_qc = dict(cur_qc)
                        self.gov_state.last_qc_deltas = dict(exp_delta)
                    except Exception:
                        pass

                    dp = ((gov.data.get('global_defaults') or {}).get('dynamic_prompting') or {})
                    trigger = (dp.get('trigger') or {}) if isinstance(dp, dict) else {}
                    try:
                        thr = int(trigger.get('delta_abs_threshold', 2) or 2)
                    except Exception:
                        thr = 2
                    # Optional: JSON may define consecutive_turns; default 2.
                    try:
                        consec_need = int(trigger.get('consecutive_turns', 2) or 2)
                    except Exception:
                        consec_need = 2

                    # Only count if any delta exceeds threshold
                    if any(abs(int(d or 0)) >= thr for d in exp_delta.values()):
                        try:
                            self.gov_state.dynamic_consecutive_turns = int(getattr(self.gov_state, 'dynamic_consecutive_turns', 0) or 0) + 1
                        except Exception:
                            self.gov_state.dynamic_consecutive_turns = 1
                        if int(getattr(self.gov_state, 'dynamic_consecutive_turns', 0) or 0) >= max(1, consec_need):
                            # Enable for next request only
                            self.gov_state.dynamic_one_shot_active = True
                            self.gov_state.dynamic_consecutive_turns = 0
                    else:
                        self.gov_state.dynamic_consecutive_turns = 0
            except Exception:
                pass

            # Anchor auto snapshots (best-effort): build a deterministic snapshot every N user turns.
            try:
                data = getattr(gov, 'data', {}) or {}
                anchor_cfg = ((data.get('global_defaults') or {}).get('anchor') or {})
                try:
                    auto_interval = int(anchor_cfg.get('auto_interval_turns', 10) or 10)
                except Exception:
                    auto_interval = 10

                if bool(getattr(self.gov_state, 'anchor_auto', True)) and auto_interval > 0:
                    turns = int(getattr(self.gov_state, 'user_turns', 0) or 0)
                    if turns > 0 and (turns % auto_interval) == 0:
                        try:
                            snapshot_html = self._render_anchor_snapshot_html()
                        except Exception:
                            snapshot_html = ''
                        if snapshot_html:
                            self.gov_state.last_anchor = snapshot_html
                            # Optional persistence to audit log if configured.
                            if bool(anchor_cfg.get('persist_to_audit', False)):
                                try:
                                    self.export(audit_event={
                                        "event": "anchor_auto",
                                        "ts": datetime.now().isoformat(),
                                        "turn": turns,
                                        "snapshot": snapshot_html,
                                    }, audit_only=True)
                                except Exception:
                                    pass
            except Exception:
                pass

            # Auto-return: SCI recursion (one-shot) — decrement depth and restore parent variant
            try:
                if bool(getattr(self.gov_state, 'sci_recursion_one_shot', False)):
                    cur = int(getattr(self.gov_state, 'sci_recursion_depth', 0) or 0)
                    parent = getattr(self.gov_state, 'sci_recursion_parent_variant', '') or ''
                    self.gov_state.sci_recursion_one_shot = False
                    self.gov_state.sci_recursion_scope = ""
                    if cur > 0:
                        self.gov_state.sci_recursion_depth = cur - 1
                    # restore
                    if parent:
                        self.gov_state.sci_variant = parent
                    if int(getattr(self.gov_state, 'sci_recursion_depth', 0) or 0) <= 0:
                        self.gov_state.sci_recursion_parent_variant = ""
                    # notify model of restored parent state (avoid session recreation)
                    try:
                        self._ensure_governance_pinned(reason='SCI recurse return')
                        self._send_state_update_to_model(reason='SCI recurse return')
                        self._last_session_stamp = _session_stamp()
                    except Exception:
                        pass
            except Exception:
                pass

            # Prepend SCI-timeout note if any
            if sci_note_html:
                final = sci_note_html + "\n" + final

            # Ensure pywebview return value is JSON-serializable (avoid JS receiving null)
            safe_meta = None
            try:
                import json as _json
                if meta is not None:
                    safe_meta = _json.loads(_json.dumps(meta, default=str))
            except Exception:
                safe_meta = None

            self._last_response_is_content_answer = True
            try:
                self._last_content_user_prompt = str(raw_txt or "").strip()
            except Exception:
                pass
            return {"html": final, "csc": safe_meta, "cgi_bar": True, "answer_lang": self._answer_lang()}

        except Exception as e:
            # Always persist a bot entry so exported logs are complete.
            try:
                err_html = _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error', lang=self._answer_lang())
                self.history.append({"role": "bot", "content": err_html, "ts": datetime.now().isoformat(), "csc": None})
            except Exception:
                err_html = _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error', lang=self._answer_lang())
            self._last_response_is_content_answer = False
            return {"html": err_html, "csc": None, "cgi_bar": False, "answer_lang": self._answer_lang()}
    
    def update_stats_ui(self):
        if self.main_win:
            reqs = int(getattr(self, 'session_req_count', 0) or 0)
            tin = int(getattr(self, 'session_tokens_in', 0) or 0)
            tout = int(getattr(self, 'session_tokens_out', 0) or 0)
            stats_txt = f"Reqs: {reqs} | In: {tin} | Out: {tout}"
            try:
                ui = getattr(self, 'ui_controller', None)
                if ui is not None and hasattr(ui, 'update_stats'):
                    if ui.update_stats(self.main_win, stats_txt):
                        return
            except Exception:
                pass
            self.main_win.evaluate_js(f"updateStats('{stats_txt}')")


    def remote_cmd(self, cmd):
        """Inject a command into the main UI input and trigger send() via JS."""
        if not getattr(self, 'main_win', None):
            return {'ok': False, 'error': 'no_main_win'}
        try:
            cmd_s = str(cmd or '')
            if self._is_exit_confirm_modal_active():
                try:
                    self.log_event('exit_confirm_block', {'source': 'remote_cmd', 'cmd': cmd_s})
                except Exception:
                    pass
                return {'ok': False, 'error': 'exit_confirm_open_blocked'}
            if self._is_qc_override_modal_active():
                if cmd_s.strip() == "QC Override":
                    self._bring_qc_override_to_front()
                    return {'ok': True, 'already_open': True}
                self._bring_qc_override_to_front()
                try:
                    self.log_event('qc_override_modal_block', {'source': 'remote_cmd', 'cmd': cmd_s})
                except Exception:
                    pass
                return {'ok': False, 'error': 'qc_override_modal_blocked'}
            return {'ok': bool(self._ui_remote_input(cmd_s))}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def save_stats(self):
        if self.session_req_count > 0:
            line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {cfg_get_model()} | In: {self.session_tokens_in} | Out: {self.session_tokens_out} | Reqs: {self.session_req_count}\n"
            try:
                with open(STATS_FILENAME, "a", encoding="utf-8") as f:
                    f.write(line)
                print(f"[System] Stats gespeichert in {STATS_FILENAME}")
            except Exception as e:
                print(f"[System] Error beim Speichern der Stats: {e}")

    def close_app(self):
        """Close all windows and terminate the process reliably (macOS-friendly).

        - Closes panel + main window (and any other pywebview windows if present).
        - Hard-exits after a short delay to avoid orphaned UI threads.
        """
        if getattr(self, 'is_closing', False):
            return
        self.is_closing = True

        try:
            print('[System] Exiting...')
        except Exception:
            pass

        try:
            self.save_stats()
        except Exception:
            pass
        try:
            self._save_input_history_entries(reason="app_close")
        except Exception:
            pass

        # Best-effort: destroy all known pywebview windows
        try:
            wins = list(getattr(webview, 'windows', []) or [])
            for w in wins:
                try:
                    w.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        # Direct handles (in case they are not in webview.windows)
        try:
            if getattr(self, 'panel_win', None):
                self.panel_win.destroy()
        except Exception:
            pass

        try:
            if getattr(self, 'main_win', None):
                self.main_win.destroy()
        except Exception:
            pass

        time.sleep(0.2)
        os._exit(0)

    def on_main_window_close(self):
        # Wird gerufen, wenn man das X drückt
        self.close_app()

    def _ui_add_system_message(self, message: str) -> bool:
        try:
            msg = str(message or "")
            ui = getattr(self, "ui_controller", None)
            if ui is not None and hasattr(ui, "add_system_message"):
                if ui.add_system_message(getattr(self, "main_win", None), msg):
                    return True
            win = getattr(self, "main_win", None)
            if win is None:
                return False
            win.evaluate_js(f"addMsg('sys', {json.dumps(msg, ensure_ascii=False)});")
            return True
        except Exception:
            return False

    def _ui_update_rule_file(self, filename: str) -> bool:
        try:
            name = str(filename or "")
            ui = getattr(self, "ui_controller", None)
            if ui is not None and hasattr(ui, "update_rule_file"):
                if ui.update_rule_file(getattr(self, "main_win", None), name):
                    return True
            win = getattr(self, "main_win", None)
            if win is None:
                return False
            win.evaluate_js(f"updateRuleFile({json.dumps(name, ensure_ascii=False)});")
            return True
        except Exception:
            return False

    def _ui_refresh_panel(self) -> bool:
        try:
            win = getattr(self, "panel_win", None)
            if win is None:
                return False
            ui = getattr(self, "ui_controller", None)
            if ui is not None and hasattr(ui, "eval_js"):
                if ui.eval_js(win, "window.refresh_panel && window.refresh_panel();"):
                    return True
            win.evaluate_js("window.refresh_panel && window.refresh_panel()")
            return True
        except Exception:
            return False

    def _ui_remote_input(self, cmd: str) -> bool:
        try:
            text = str(cmd or "")
            ui = getattr(self, "ui_controller", None)
            if ui is not None and hasattr(ui, "remote_input"):
                if ui.remote_input(getattr(self, "main_win", None), text):
                    return True
            win = getattr(self, "main_win", None)
            if win is None:
                return False
            win.evaluate_js(f"remoteInput({json.dumps(text, ensure_ascii=False)});")
            return True
        except Exception:
            return False

    def _is_qc_override_modal_active(self) -> bool:
        try:
            return bool(getattr(self, "_qc_override_open", False))
        except Exception:
            return False

    def _bring_qc_override_to_front(self) -> bool:
        try:
            win = getattr(self, "qc_win", None)
            if win is None:
                return False
            used = False
            for _meth in ("show", "restore", "bring_to_front", "focus"):
                try:
                    if hasattr(win, _meth):
                        getattr(win, _meth)()
                        used = True
                except Exception:
                    pass
            return bool(used)
        except Exception:
            return False

    def _qc_override_modal_block_html(self) -> str:
        try:
            lang = str(self._lang() or "de").strip().lower()
        except Exception:
            lang = "de"
        if lang == "en":
            msg = (
                "QC Override is open. Please finish with Apply, Clear Overrides, or Cancel "
                "before using Main or Panel actions."
            )
        else:
            msg = (
                "QC Override ist geöffnet. Bitte zuerst mit Apply, Clear Overrides oder Cancel beenden, "
                "bevor Main- oder Panel-Aktionen genutzt werden."
            )
        return _control_layer_alert_html(msg, title="QC Override aktiv", severity="warn", lang=self._answer_lang())

    def _is_exit_confirm_modal_active(self) -> bool:
        try:
            return bool(getattr(self, "_exit_confirm_open", False))
        except Exception:
            return False

    def _exit_confirm_block_html(self) -> str:
        lang = self._answer_lang()
        if lang == "en":
            msg = (
                "Exit confirmation is open. Choose Cancel or Exit in the modal "
                "before sending new actions."
            )
            title = "Exit confirmation active"
        else:
            msg = (
                "Die Exit-Bestaetigung ist geoeffnet. Bitte zuerst im Dialog Cancel "
                "oder Exit waehlen, bevor neue Aktionen gesendet werden."
            )
            title = "Exit-Bestaetigung aktiv"
        return _control_layer_alert_html(msg, title=title, severity="warn", lang=lang)

    def set_exit_confirm_open(self, is_open):
        """Set exclusive Exit-confirm modal state (global main/panel action gate)."""
        try:
            flag = bool(is_open)
        except Exception:
            flag = False
        self._exit_confirm_open = flag
        try:
            self.log_event("exit_confirm_state", {"open": bool(flag)})
        except Exception:
            pass
        return {"ok": True, "open": bool(flag)}

    def get_help_content(self):
        """Return localized stage-1 help payload in answer language."""
        lang = self._answer_lang()
        payload = None
        try:
            if _help_i18n_mod is not None and hasattr(_help_i18n_mod, "load_help_payload"):
                payload = _help_i18n_mod.load_help_payload(lang=lang)
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            payload = {
                "button_label": "❓ Help" if lang == "en" else "❓ Hilfe",
                "title": "Comm-SCI Help" if lang == "en" else "Comm-SCI Hilfe",
                "subtitle": "Quickstart, commands, SCI and troubleshooting"
                if lang == "en"
                else "Quickstart, Kommandos, SCI und Troubleshooting",
                "close_label": "Close" if lang == "en" else "Schliessen",
                "sections": [],
                "footer": "Tip: Press F1 to open help."
                if lang == "en"
                else "Tipp: Mit F1 Hilfe oeffnen.",
            }
        return {"ok": True, "lang": lang, "payload": payload}

    def _answer_lang(self) -> str:
        """Return answer language (de/en), independent of UI language."""
        try:
            lang = str(getattr(getattr(self, "gov_state", None), "answer_language", "") or "").strip().lower()
        except Exception:
            lang = ""
        if not lang:
            try:
                cfg = getattr(self, "cfg_mgr", None) or globals().get("cfg")
                lang = str(getattr(cfg, "get_answer_language", lambda: "de")() or "").strip().lower()
            except Exception:
                lang = ""
        if lang.startswith("de"):
            return "de"
        if lang.startswith("en"):
            return "en"
        try:
            ui_lang = str(self._lang() or "de").strip().lower()
        except Exception:
            ui_lang = "de"
        return "de" if ui_lang.startswith("de") else "en"

    def _language_policy_mode(self) -> str:
        """Return active language policy mode (production|benchmark)."""
        try:
            mode = str(getattr(getattr(self, "gov_state", None), "language_policy_mode", "") or "").strip().lower()
        except Exception:
            mode = ""
        if mode not in ("production", "benchmark"):
            try:
                cfg_obj = getattr(self, "cfg_mgr", None) or globals().get("cfg")
                mode = str(getattr(cfg_obj, "get_language_policy_mode", lambda: "production")() or "").strip().lower()
            except Exception:
                mode = ""
        return "benchmark" if mode == "benchmark" else "production"

    def _append_uncertainty_explanation_if_needed(self, html_text: str, user_text: str = "") -> str:
        txt = str(html_text or "")
        if not txt:
            return txt
        lang = self._answer_lang()
        out = txt
        try:
            if _uncertainty_codes_mod is not None and hasattr(_uncertainty_codes_mod, "ensure_uncertainty_annotations_html"):
                out = _uncertainty_codes_mod.ensure_uncertainty_annotations_html(
                    txt,
                    lang=lang,
                    user_text=str(user_text or ""),
                )
            elif _uncertainty_codes_mod is not None and hasattr(_uncertainty_codes_mod, "append_uncertainty_legend_html"):
                out = _uncertainty_codes_mod.append_uncertainty_legend_html(txt, lang=lang)
        except Exception:
            out = txt
        try:
            out = annotate_signal_dot_tooltips_html(out, lang=lang)
        except Exception:
            pass
        try:
            out = re.sub(
                r"(?is)<details\b[^>]*class=(?:\"|')[^\"']*\buncertainty-legend\b[^\"']*(?:\"|')[^>]*>.*?</details>\s*",
                "",
                str(out or ""),
            )
        except Exception:
            pass
        return out

    def ping(self, _payload=None):
        """Panel health check."""
        try:
            return {'ok': True, 'ts': datetime.now().isoformat()}
        except Exception:
            return {'ok': True}



    def panel_action(self, action, payload=None):
        """Single entrypoint for Panel UI actions (robust against missing per-method bridges).

        This keeps the Panel functional even if certain individual JS API methods are not
        exposed reliably by the backend for secondary windows.
        """
        try:
            action_s = str(action or '').strip()
        except Exception:
            action_s = ''
        payload = payload or {}
        try:
            if not isinstance(payload, dict):
                payload = {'value': payload}
        except Exception:
            payload = {}

        def _ok(result=None, **extra):
            out = {'ok': True, 'action': action_s, 'result': result, 'error': None}
            if extra:
                out.update(extra)
            return out

        def _err(message: str):
            return {'ok': False, 'action': action_s, 'result': None, 'error': str(message or 'error')}

        # Exit-confirm modal gate: exclusive program-wide modal state.
        if self._is_exit_confirm_modal_active():
            try:
                self.log_event(
                    'exit_confirm_block',
                    {'source': 'panel_action', 'action': action_s},
                )
            except Exception:
                pass
            return _err("exit_confirm_open_blocked")

        # QC-Override modal gate: while dialog is open, block Main/Panel actions except QC actions.
        if self._is_qc_override_modal_active():
            _allowed_modal_actions = {
                'qc_override_apply',
                'qc_override_clear',
                'qc_override_cancel',
                'panel_bootstrap_selftest',
            }
            if action_s not in _allowed_modal_actions:
                try:
                    self._bring_qc_override_to_front()
                except Exception:
                    pass
                try:
                    self.log_event(
                        'qc_override_modal_block',
                        {'source': 'panel_action', 'action': action_s},
                    )
                except Exception:
                    pass
                return _err("qc_override_modal_blocked")

        # Strict Comm-off panel gate: block rule-workflow actions even if stale/hidden UI calls still fire.
        try:
            _comm_on = bool(getattr(getattr(self, 'gov_state', None), 'comm_active', False))
        except Exception:
            _comm_on = False
        if not _comm_on:
            _blocked_actions = {
                'qc_override_apply', 'qc_override_clear',
                'manual_test_monitor_show', 'manual_test_monitor_hide', 'manual_test_monitor_reset',
                'manual_test_monitor_append', 'manual_test_monitor_header', 'save_manual_test_report',
                'manual_test_main_chat_append',
            }
            if action_s in _blocked_actions:
                return _err("comm_off_blocked")
            if action_s in {'cmd', 'ask'}:
                try:
                    _txt = str((payload or {}).get('text', '') or '').strip()
                except Exception:
                    _txt = ''
                if _txt != 'Comm Start':
                    return _err("comm_off_blocked")

        # S6: delegate stable panel-action subsets to UIController (primary route).
        try:
            _ui = getattr(self, 'ui_controller', None)
            if _ui is not None and hasattr(_ui, 'try_handle_panel_aux_action'):
                _delegated = _ui.try_handle_panel_aux_action(self, action_s, payload)
                if isinstance(_delegated, dict):
                    return _delegated
        except Exception:
            pass

        if action_s == 'panel_bootstrap_selftest':
            try:
                info = self._panel_accept_bootstrap_report(payload)
            except Exception as e:
                info = {'accepted': False, 'runtime_ok': False, 'reason': f'{type(e).__name__}: {e}'}
            return _ok(info, runtime_ok=bool((info or {}).get('runtime_ok')))

        try:
            if action_s == 'cmd':
                # Execute via main window pipeline so results appear in the chat UI.
                text = payload.get('text', '')
                try:
                    if hasattr(self, 'remote_cmd'):
                        self.remote_cmd(str(text or ''))
                        return _ok({'queued': True}, queued=True)
                except Exception as e:
                    return _err(str(e))
                return _err('remote_cmd_unavailable')
            if action_s == 'ask':
                # Blocking ask-path for panel-side automation/test runners.
                text = payload.get('text', '')
                try:
                    if hasattr(self, 'ask'):
                        res = self.ask(str(text or ''))
                        if isinstance(res, dict):
                            return _ok(res, **res)
                        return _ok({'html': str(res or '')})
                except Exception as e:
                    return _err(str(e))
                return _err('ask_unavailable')
            if action_s == 'export':
                # Panel bridge does not expose api.export directly; provide a deterministic
                # route through panel_action for manual-test checkpoints.
                try:
                    chat_path, audit_path = self.export()
                    return _ok({'chat_path': chat_path, 'audit_path': audit_path}, chat_path=chat_path, audit_path=audit_path)
                except Exception as e:
                    return _err(str(e))
            if action_s == 'manual_test_stop':
                fn = getattr(self, 'manual_test_request_stop', None)
                res = fn(payload or {}) if callable(fn) else {'ok': False, 'error': 'manual_test_request_stop unavailable'}
                if isinstance(res, dict):
                    if bool(res.get('ok', True)):
                        return _ok(res, **res)
                    return _err(str(res.get('error', 'manual_test_stop_failed')))
                return _ok({'ok': True})
        except Exception as e:
            return _err(str(e))
        return _err('unknown action')

    def clear_chat(self):
        """Clear in-memory chat history and reset the main chat UI (no model call).

        Intended for the Panel 'Clear' button.
        Defensive: must not crash if UI is not yet available.
        """
        try:
            # 1) Clear history
            try:
                self.history = []
            except Exception:
                pass

            # Clear QC overrides (session-local)
            try:
                self.gov_state.qc_overrides = {}
            except Exception:
                pass

            # 2) Best-effort reset of session identifiers/counters (no secrets)
            try:
                import uuid as _uuid
                self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + _uuid.uuid4().hex[:6]
                self.trace_id = self.session_id
                self.session_start_dt = datetime.now()
            except Exception:
                pass

            # 3) Reset main UI (if present)
            try:
                win = getattr(self, 'main_win', None)
                if win is not None:
                    msg = 'Chat cleared.'
                    try:
                        import json as _json
                        sm = _json.dumps(msg, ensure_ascii=False)
                    except Exception:
                        sm = '"Chat cleared."'
                    win.evaluate_js(f"resetChatToStatus({sm});")
            except Exception:
                pass

            return {'ok': True, 'history_len': 0}
        except Exception as e:
            try:
                return {'ok': False, 'error': f"{type(e).__name__}: {e}"}
            except Exception:
                return {'ok': False, 'error': 'error'}

    def save_manual_test_report(self, report):
        """Persist panel manual-test run results as JSON under Logs/ManualTests (best-effort)."""
        try:
            payload = report if isinstance(report, dict) else {'raw': report}
        except Exception:
            payload = {'raw': str(report)}
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        except Exception:
            ts = str(int(time.time()))
        try:
            scenario = str((payload or {}).get('scenario') or 'manual_test').strip().lower()
            scenario = re.sub(r'[^a-z0-9._-]+', '_', scenario).strip('_') or 'manual_test'
        except Exception:
            scenario = 'manual_test'
        target_dir = os.path.join(LOGS_DIR, 'ManualTests')
        target = os.path.join(target_dir, f'ManualTest_{ts}_{scenario}.json')
        try:
            svc = getattr(self, 'storage_service', None)
        except Exception:
            svc = None
        if svc is None and _StorageService is not None:
            try:
                svc = _StorageService()
            except Exception:
                svc = None
        ok = False
        try:
            if svc is not None and hasattr(svc, 'write_json'):
                ok = bool(svc.write_json(target, payload, ensure_ascii=False))
            else:
                os.makedirs(target_dir, exist_ok=True)
                with open(target, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                ok = True
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}', 'path': target}
        return {'ok': bool(ok), 'path': target}

    def manual_test_main_chat_append(self, payload=None):
        """Best-effort append helper for mirroring manual-test steps into the main chat UI."""
        p = payload if isinstance(payload, dict) else {}
        role = str((p or {}).get("role") or "sys").strip().lower()
        if role not in {"user", "bot", "sys"}:
            role = "sys"
        text = str((p or {}).get("text") or "")
        html_text = str((p or {}).get("html") or "")
        cgi_bar = bool((p or {}).get("cgi_bar", False))
        answer_lang = str((p or {}).get("answer_lang") or "").strip().lower()
        if answer_lang not in {"de", "en"}:
            answer_lang = ""

        csc = (p or {}).get("csc")
        if not isinstance(csc, (dict, list, str, int, float, bool)) and csc is not None:
            try:
                csc = str(csc)
            except Exception:
                csc = None

        win = getattr(self, "main_win", None)
        if win is None:
            return {"ok": False, "error": "main_win unavailable"}

        try:
            if role == "bot":
                content = html_text if html_text else text
                opts = {"answerLang": answer_lang} if answer_lang else {}
                js = (
                    f"addMsg('bot', {json.dumps(str(content or ''), ensure_ascii=False)}, "
                    f"{'true' if cgi_bar else 'false'}, "
                    f"{json.dumps(csc, ensure_ascii=False)}, "
                    f"{json.dumps(opts, ensure_ascii=False)});"
                )
            elif role == "user":
                content = text if text else html_text
                js = f"addMsg('user', {json.dumps(str(content or ''), ensure_ascii=False)});"
            else:
                content = text if text else html_text
                js = f"addMsg('sys', {json.dumps(str(content or ''), ensure_ascii=False)});"
            win.evaluate_js(js)
            return {"ok": True, "role": role}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def manual_test_request_stop(self, payload=None):
        """Request stop for the running panel-side manual-test runner."""
        p = payload if isinstance(payload, dict) else {}
        lang = str((p or {}).get("lang") or "").strip().lower()
        msg = "Stop requested (monitor)." if lang == "en" else "Stop angefordert (Monitor)."
        try:
            win = getattr(self, "panel_win", None)
            if win is None:
                return {"ok": False, "error": "panel_win unavailable"}
            js = (
                "(function(){"
                "try{"
                "if(!window.__manualTestRunner){return {ok:false,error:'manual_test_runner unavailable'};}"
                "window.__manualTestRunner.stop = true;"
                f"if(typeof _mtLog==='function'){{_mtLog({json.dumps(msg, ensure_ascii=False)});}}"
                "return {ok:true,running:!!window.__manualTestRunner.running};"
                "}catch(e){return {ok:false,error:String(e&&e.message?e.message:e)};}"
                "})();"
            )
            out = win.evaluate_js(js)
            if isinstance(out, dict):
                if bool(out.get("ok", True)):
                    return {"ok": True, "running": bool(out.get("running", False))}
                return {"ok": False, "error": str(out.get("error") or "manual_test_stop_failed")}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def on_manual_test_monitor_closed(self):
        try:
            self.manual_test_monitor_win = None
        except Exception:
            pass

    def _bind_manual_test_monitor_window_events(self, win):
        if not win:
            return
        evs = getattr(win, 'events', None)
        closed_ev = getattr(evs, 'closed', None)
        if closed_ev is not None:
            try:
                closed_ev += self.on_manual_test_monitor_closed
            except Exception:
                pass

    def _create_manual_test_monitor(self):
        """Pre-create Manual Test Monitor dialog window (hidden)."""
        try:
            if getattr(self, 'manual_test_monitor_win', None) is not None:
                return
        except Exception:
            pass
        try:
            self.manual_test_monitor_state = {
                'scenario': '',
                'status': 'idle',
                'summary': '-',
                'events': [],
            }
        except Exception:
            pass
        try:
            self.manual_test_monitor_win = webview.create_window(
                "Comm-SCI Manual Test Monitor",
                html=HTML_MANUAL_TEST_MONITOR,
                width=760,
                height=700,
                resizable=True,
                hidden=True,
                on_top=False,
                js_api=(getattr(self, 'panel_bridge', None) or self),
            )
            try:
                self._bind_manual_test_monitor_window_events(self.manual_test_monitor_win)
            except Exception:
                pass
        except Exception:
            try:
                self.manual_test_monitor_win = None
            except Exception:
                pass

    def _manual_test_monitor_eval(self, js_code: str) -> bool:
        try:
            win = getattr(self, 'manual_test_monitor_win', None)
            if win is None:
                return False
            win.evaluate_js(str(js_code or ''))
            return True
        except Exception:
            return False

    def _manual_test_monitor_apply_seam_state_plan(self, plan) -> dict:
        self.manual_test_monitor_state = (plan or {}).get('state')
        _js_code = (plan or {}).get('js_code')
        if _js_code:
            try:
                self._manual_test_monitor_eval(str(_js_code))
            except Exception:
                pass
        return {'ok': True}

    def manual_test_monitor_show(self, payload=None):
        try:
            def _ensure_window():
                _win = getattr(self, 'manual_test_monitor_win', None)
                if _win is None:
                    self._create_manual_test_monitor()
                    _win = getattr(self, 'manual_test_monitor_win', None)
                return _win

            win = _ensure_window()
            if win is None:
                return {'ok': False, 'error': 'manual_test_monitor_win unavailable'}

            show_ok = False
            try:
                if hasattr(win, 'show'):
                    win.show()
                show_ok = True
            except Exception as e:
                show_ok = False

            if not show_ok:
                try:
                    self.manual_test_monitor_win = None
                except Exception:
                    pass
                win = _ensure_window()
                if win is None:
                    return {'ok': False, 'error': 'manual_test_monitor_win unavailable'}
                try:
                    if hasattr(win, 'show'):
                        win.show()
                    show_ok = True
                except Exception as e2:
                    return {'ok': False, 'error': f"{type(e2).__name__}: {e2}"}

            try:
                if hasattr(win, 'bring_to_front'):
                    win.bring_to_front()
            except Exception:
                pass
            # Push current state if available
            try:
                st = getattr(self, 'manual_test_monitor_state', None) or {}
                import json as _json
                self._manual_test_monitor_eval(f"mtmReplace({_json.dumps(st, ensure_ascii=False)});")
            except Exception:
                pass
            return {'ok': True}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def manual_test_monitor_hide(self):
        try:
            win = getattr(self, 'manual_test_monitor_win', None)
            if win is None:
                return {'ok': True, 'hidden': True, 'skipped': True}
            try:
                if hasattr(win, 'hide'):
                    win.hide()
                elif hasattr(win, 'minimize'):
                    win.minimize()
            except Exception:
                pass
            return {'ok': True, 'hidden': True}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def manual_test_monitor_reset(self, payload=None):
        payload = payload or {}
        try:
            if _manual_test_monitor_seam_mod is None:
                return {'ok': False, 'error': 'manual_test_monitor_seam unavailable'}
            return self._manual_test_monitor_apply_seam_state_plan(
                _manual_test_monitor_seam_mod.manual_test_monitor_reset_state_plan(
                    state=getattr(self, 'manual_test_monitor_state', None),
                    payload=payload,
                )
            )
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def manual_test_monitor_append(self, entry):
        try:
            if _manual_test_monitor_seam_mod is None:
                return {'ok': False, 'error': 'manual_test_monitor_seam unavailable'}
            return self._manual_test_monitor_apply_seam_state_plan(
                _manual_test_monitor_seam_mod.manual_test_monitor_append_state_plan(
                    state=getattr(self, 'manual_test_monitor_state', None),
                    entry=entry,
                    max_events=1000,
                )
            )
        except Exception as ex:
            return {'ok': False, 'error': f"{type(ex).__name__}: {ex}"}

    def manual_test_monitor_set_header(self, payload=None):
        payload = payload or {}
        try:
            if _manual_test_monitor_seam_mod is None:
                return {'ok': False, 'error': 'manual_test_monitor_seam unavailable'}
            return self._manual_test_monitor_apply_seam_state_plan(
                _manual_test_monitor_seam_mod.manual_test_monitor_set_header_state_plan(
                    state=getattr(self, 'manual_test_monitor_state', None),
                    payload=payload,
                )
            )
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def get_ui(self):
        """Return a UI snapshot for the Panel.

        Requirements:
        - Must be fast and JSON-serializable.
        - Must be safe during early init (no hard dependency on gov/cfg).
        - No network calls.

        Strategy:
        - Always return a minimal, fully-usable default snapshot (fail-open).
        - If the ruleset/governance runtime is available, merge the richer button schema
          (comm/profiles/sci/overlays/tools/logs) from gov.get_ui_data().
        - Provide a cheap local listing of chat logs for the loader UI.
        """
        def _attach_passphrase_status(payload):
            out = payload if isinstance(payload, dict) else {}
            try:
                pst = self.get_passphrase_status()
            except Exception:
                pst = {}
            if isinstance(pst, dict):
                try:
                    out['passphrase_required'] = bool(pst.get('required', False))
                    out['passphrase_provider'] = str(pst.get('provider') or '')
                    out['passphrase_reason'] = str(pst.get('reason') or '')
                except Exception:
                    pass
            return out

        if _panel_ui_snapshot_seam_mod is not None:
            try:
                gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                snap = _panel_ui_snapshot_seam_mod.panel_ui_build_snapshot(
                    provider_router=getattr(self, 'provider_router', None),
                    cfg_obj=globals().get('cfg'),
                    get_available_models_fn=getattr(self, 'get_available_models', None),
                    gov_state=getattr(self, 'gov_state', None),
                    panel_state_snapshot_ctor=_PanelStateSnapshot,
                    panel_normalize_ui_fn=_panel_normalize_ui,
                    gov_obj=gov_obj,
                    list_chat_logs_fn=getattr(self, 'list_chat_logs', None),
                    chat_log_limit=200,
                )
                return _attach_passphrase_status(snap)
            except Exception:
                pass

        # Fail-open fallback if the seam module is unavailable or failed.
        if _panel_ui_snapshot_seam_mod is not None:
            try:
                snap = _panel_ui_snapshot_seam_mod.panel_ui_failopen_snapshot(
                    gov_state=getattr(self, 'gov_state', None),
                )
                return _attach_passphrase_status(snap)
            except Exception:
                pass
        return _attach_passphrase_status({
            'providers': ['gemini', 'openrouter', 'huggingface'],
            'current_provider': 'gemini',
            'current_model': 'gemini-2.0-flash',
            'available_models': ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-pro'],
            'answer_language': 'de',
            'language_policy_mode': 'production',
            'comm': [{'name': 'Comm Start', 'cmd': 'Comm Start', 'desc': 'Start Comm Control Layer'}],
            'profiles': [],
            'sci': [],
            'overlays': [],
            'tools': [],
            'logs': [],
            'chat_logs': [],
            'model_hint': '',
            'comm_active': False,
            'manual_test_visible': False,
            'qc_override_visible': False,
            'provider': 'gemini',
            'model': 'gemini-2.0-flash',
        })

    def _warm_model_caches_from_disk(self):
        """Load cached provider model lists from disk into memory. No network."""
        try:
            pr = getattr(self, 'provider_router', None)
            if pr is None:
                return
            # OpenRouter cache
            try:
                p = pr._openrouter_cache_path() if hasattr(pr, '_openrouter_cache_path') else ''
                if p and os.path.exists(p):
                    raw = Path(p).read_text(encoding='utf-8')
                    obj = json.loads(raw) if raw else {}
                    models = obj.get('models') or []
                    if isinstance(models, list):
                        self._openrouter_models_cache = [str(m).strip() for m in models if str(m).strip()]
            except Exception:
                pass
            # Hugging Face cache
            try:
                p = pr._huggingface_cache_path() if hasattr(pr, '_huggingface_cache_path') else ''
                if p and os.path.exists(p):
                    raw = Path(p).read_text(encoding='utf-8')
                    obj = json.loads(raw) if raw else {}
                    models = obj.get('models') or []
                    if isinstance(models, list):
                        self._hf_models_cache = [str(m).strip() for m in models if str(m).strip()]
            except Exception:
                pass
            # Gemini cache
            try:
                p = pr._gemini_cache_path() if hasattr(pr, '_gemini_cache_path') else ''
                if p and os.path.exists(p):
                    raw = Path(p).read_text(encoding='utf-8')
                    obj = json.loads(raw) if raw else {}
                    models = obj.get('models') or []
                    if isinstance(models, list):
                        self._gemini_models_cache = [str(m).strip() for m in models if str(m).strip()]
            except Exception:
                pass
        except Exception:
            return

    def get_available_models(self, provider: str):
        """Return cached/known models for provider. Must be fast and never do network I/O."""
        try:
            p = (provider or '').strip().lower()
            if p == 'gemini':
                cache = getattr(self, '_gemini_models_cache', None)
                if isinstance(cache, list) and cache:
                    return cache
                return [
                    'gemini-2.0-flash',
                    'gemini-2.5-flash',
                    'gemini-3-flash',
                    'gemini-1.5-pro',
                ]
            if p == 'openrouter':
                # Use cached list if available
                cache = getattr(self, '_openrouter_models_cache', None)
                if isinstance(cache, list) and cache:
                    return cache
                return []
            if p == 'huggingface':
                cache = getattr(self, '_hf_models_cache', None)
                if isinstance(cache, list) and cache:
                    return cache
                return []
            return []
        except Exception:
            return []

    def list_chat_logs(self, limit: int = 200):
        """List available chat logs from Logs/Chats (filenames only).

        Safe, local-only, and must not throw.
        """
        try:
            lim = int(limit) if limit is not None else 200
        except Exception:
            lim = 200
        if lim <= 0:
            lim = 200

        try:
            base = globals().get('CHAT_LOG_DIR')
            if not base:
                return {'ok': True, 'logs': []}
            svc = getattr(self, 'storage_service', None)
            if svc is not None and hasattr(svc, 'list_json_filenames'):
                return {'ok': True, 'logs': svc.list_json_filenames(str(base), limit=lim)}

            p = pathlib.Path(base)
            if not p.exists() or not p.is_dir():
                return {'ok': True, 'logs': []}
            files = []
            for f in p.iterdir():
                if f.is_file() and f.name.lower().endswith('.json'):
                    files.append(f.name)
            files.sort(reverse=True)
            if len(files) > lim:
                files = files[:lim]
            return {'ok': True, 'logs': files}
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}', 'logs': []}

    def load_chat_log(self, filename: str, fork: bool = True):
        """Load a chat log from Logs/Chats by filename.

        Prevents path traversal by resolving under CHAT_LOG_DIR.
        Delegates to load_log_from_path(..., fork=...).
        """
        try:
            name = str(filename or '').strip()
            if not name:
                return {'ok': False, 'error': 'missing_filename'}

            base = globals().get('CHAT_LOG_DIR')
            if not base:
                return {'ok': False, 'error': 'chat_log_dir_missing'}

            svc = getattr(self, 'storage_service', None)
            candidate = None
            if svc is not None and hasattr(svc, 'safe_resolve_in_dir'):
                candidate = svc.safe_resolve_in_dir(str(base), name)
            if not candidate:
                import os
                base_abs = os.path.abspath(base)
                candidate = os.path.abspath(os.path.join(base_abs, os.path.basename(name)))
                if not candidate.startswith(base_abs):
                    return {'ok': False, 'error': 'path_traversal_blocked'}

            exists = False
            if svc is not None and hasattr(svc, 'exists'):
                exists = bool(svc.exists(candidate))
            else:
                import os
                exists = os.path.exists(candidate)
            if not exists:
                return {'ok': False, 'error': 'file_not_found'}

            # Prefer bound method on self (unit tests), else call module helper.
            if hasattr(self, 'load_log_from_path'):
                return self.load_log_from_path(candidate, fork=bool(fork))
            return {'ok': False, 'error': 'load_log_from_path_unavailable'}
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


    def _normalize_provider_id(self, provider: str) -> str:
        try:
            p = str(provider or '').strip().lower()
        except Exception:
            p = ''
        if p in ('hf',):
            p = 'huggingface'
        if p == 'google':
            p = 'gemini'
        if p not in ('gemini', 'openrouter', 'huggingface'):
            p = 'gemini'
        return p

    def _encrypted_key_material_for_provider(self, provider: str):
        """Return (enc_b64, salt_b64) for provider if encrypted key material exists."""
        p = self._normalize_provider_id(provider)
        aliases = [p]
        if p == 'gemini':
            aliases.append('google')
        if p == 'huggingface':
            aliases.append('hf')

        def _extract(entry):
            try:
                if not isinstance(entry, dict):
                    return '', ''
                enc = str(entry.get('api_key_enc') or '').strip()
                salt = str(entry.get('api_key_salt') or '').strip()
                if enc and salt:
                    return enc, salt
            except Exception:
                pass
            return '', ''

        # 1) Config providers.* (runtime overrides)
        try:
            provs_cfg = ((getattr(cfg, 'config', {}) or {}).get('providers') or {})
            if isinstance(provs_cfg, dict):
                for a in aliases:
                    enc, salt = _extract(provs_cfg.get(a))
                    if enc and salt:
                        return enc, salt
        except Exception:
            pass

        # 2) Keys file providers.*
        data = {}
        try:
            data, _used_path, _err = _load_keys_json()
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        try:
            provs_keys = data.get('providers') or {}
            if isinstance(provs_keys, dict):
                for a in aliases:
                    enc, salt = _extract(provs_keys.get(a))
                    if enc and salt:
                        return enc, salt
        except Exception:
            pass

        # 3) Legacy top-level keys (Gemini)
        if p == 'gemini':
            try:
                enc = str(data.get('GOOGLE_API_KEY_ENC') or '').strip()
                salt = str(data.get('GOOGLE_API_KEY_SALT') or '').strip()
                if enc and salt:
                    return enc, salt
            except Exception:
                pass

        return '', ''

    def _passphrase_requirement_for_provider(self, provider: str, *, passphrase_override: str = "") -> dict:
        """Determine whether provider access is blocked by missing/invalid passphrase."""
        p = self._normalize_provider_id(provider)
        enc, salt = self._encrypted_key_material_for_provider(p)
        if not enc or not salt:
            return {'required': False, 'provider': p, 'encrypted': False, 'reason': ''}

        try:
            passphrase = str(passphrase_override or os.environ.get('COMM_SCI_KEY_PASSPHRASE') or '').strip()
        except Exception:
            passphrase = ''
        if not passphrase:
            return {'required': True, 'provider': p, 'encrypted': True, 'reason': 'missing_passphrase'}

        plain = _try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt)
        if plain:
            return {'required': False, 'provider': p, 'encrypted': True, 'reason': ''}
        return {'required': True, 'provider': p, 'encrypted': True, 'reason': 'invalid_passphrase'}

    def _queue_passphrase_request(self, provider: str, reason: str = "connect"):
        try:
            self._passphrase_pending_provider = self._normalize_provider_id(provider)
            self._passphrase_pending_reason = str(reason or 'connect').strip().lower() or 'connect'
            self._passphrase_pending_since = datetime.now().isoformat()
        except Exception:
            pass

    def _clear_passphrase_request(self, provider: str = ""):
        try:
            if provider:
                p = self._normalize_provider_id(provider)
                cur = self._normalize_provider_id(getattr(self, '_passphrase_pending_provider', '') or '')
                if p and cur and p != cur:
                    return
            self._passphrase_pending_provider = ""
            self._passphrase_pending_reason = ""
            self._passphrase_pending_since = ""
        except Exception:
            pass

    def get_passphrase_status(self):
        """Public panel-state helper for passphrase-required flows."""
        try:
            pending_provider = self._normalize_provider_id(getattr(self, '_passphrase_pending_provider', '') or '')
        except Exception:
            pending_provider = ''
        provider = pending_provider or self._normalize_provider_id(self._active_provider())
        check = self._passphrase_requirement_for_provider(provider)
        if bool(check.get('required')):
            reason = str(getattr(self, '_passphrase_pending_reason', '') or check.get('reason') or 'missing_passphrase')
            if not pending_provider:
                self._queue_passphrase_request(provider, reason=reason)
            return {
                'required': True,
                'provider': provider,
                'reason': reason,
                'encrypted': True,
                'pending_since': str(getattr(self, '_passphrase_pending_since', '') or ''),
            }
        if pending_provider:
            self._clear_passphrase_request()
        return {
            'required': False,
            'provider': provider,
            'reason': '',
            'encrypted': bool(check.get('encrypted', False)),
            'pending_since': '',
        }

    def set_key_passphrase(self, passphrase: str, provider: str = "", *, reason: str = "", reconnect: bool = True):
        """Accept passphrase for encrypted provider keys and optionally reconnect."""
        try:
            pw = str(passphrase or '').strip()
        except Exception:
            pw = ''
        if not pw:
            return {'ok': False, 'error': 'passphrase_missing'}

        p = self._normalize_provider_id(
            provider
            or getattr(self, '_passphrase_pending_provider', '')
            or self._active_provider()
        )

        # Validate immediately against encrypted key material if present.
        check = self._passphrase_requirement_for_provider(p, passphrase_override=pw)
        if bool(check.get('required')):
            return {
                'ok': False,
                'error': str(check.get('reason') or 'invalid_passphrase'),
                'provider': p,
            }

        try:
            os.environ['COMM_SCI_KEY_PASSPHRASE'] = pw
        except Exception:
            pass

        self._clear_passphrase_request(p)

        switched = False
        try:
            active_now = self._normalize_provider_id(self._active_provider())
        except Exception:
            active_now = 'gemini'
        if p and p != active_now:
            sw = self.set_provider(p)
            if isinstance(sw, dict) and not bool(sw.get('ok', True)):
                return sw
            switched = True

        if reconnect:
            try:
                if p == 'gemini':
                    self._trigger_reconnect("Passphrase akzeptiert (Gemini).")
                else:
                    self._connect_api(reason='passphrase')
            except Exception:
                pass
        try:
            self._ui_add_system_message(f"Passphrase accepted for encrypted key ({p}).")
        except Exception:
            pass
        try:
            self._ui_refresh_panel()
        except Exception:
            pass
        try:
            self.log_event('passphrase_set', {'provider': p, 'reason': str(reason or '').strip()})
        except Exception:
            pass
        return {'ok': True, 'provider': p, 'switched': bool(switched)}


    def set_provider(self, provider: str):
        """Set active provider (gemini/openrouter) from the panel.

        Gemini provider changes trigger a reconnect (session-based).
        OpenRouter is stateless; no reconnect is required.
        """
        try:
            # Snapshot old provider/model for auditability (best-effort; no behavior change)
            try:
                _old_p = (getattr(cfg, 'get_active_provider', lambda: 'gemini')() or 'gemini').strip().lower()
            except Exception:
                _old_p = 'gemini'
            try:
                _old_m = ''
                if hasattr(cfg, 'get_provider_model'):
                    _old_m = str(cfg.get_provider_model(_old_p) or '').strip()
                if not _old_m:
                    _old_m = str(cfg_get_model() or '').strip()
            except Exception:
                _old_m = str(cfg_get_model() or '').strip()
            provider = self._normalize_provider_id(provider)

            # Gate provider switch if encrypted key exists but passphrase is missing/invalid.
            gate = self._passphrase_requirement_for_provider(provider)
            if bool(gate.get('required')):
                reason = str(gate.get('reason') or 'missing_passphrase')
                self._queue_passphrase_request(provider, reason='provider_switch')
                try:
                    self.log_event('passphrase_required', {'provider': provider, 'reason': reason, 'scope': 'provider_switch'})
                except Exception:
                    pass
                self._ui_add_system_message(
                    f"Encrypted API key detected for {provider}. Passphrase required before provider switch."
                )
                self._ui_refresh_panel()
                return {'ok': False, 'error': 'passphrase_required', 'provider': provider, 'reason': reason}
            if hasattr(cfg, 'set_active_provider'):
                cfg.set_active_provider(provider)
            else:
                try:
                    cfg.config['active_provider'] = provider
                    cfg.save()
                except Exception:
                    pass

            # Ensure model is present
            try:
                cur_m = (cfg.get_provider_model(provider) if hasattr(cfg, 'get_provider_model') else '') or ''
                if not cur_m:
                    # fall back to legacy
                    cur_m = (cfg_get_model() or '').strip()
                if cur_m:
                    if hasattr(cfg, 'set_provider_model'):
                        cfg.set_provider_model(provider, cur_m)
            except Exception:
                pass

            # Record provider switch event (history + session_events) before any reconnect
            try:
                _new_p = provider
            except Exception:
                _new_p = 'gemini'
            try:
                _new_m = ''
                if hasattr(cfg, 'get_provider_model'):
                    _new_m = str(cfg.get_provider_model(_new_p) or '').strip()
                if not _new_m:
                    _new_m = str(cfg_get_model() or '').strip()
            except Exception:
                _new_m = str(cfg_get_model() or '').strip()
            try:
                if not isinstance(getattr(self, 'provider_model_history', None), list):
                    self.provider_model_history = []
                self.provider_model_history.append({
                    'ts': datetime.now().isoformat(),
                    'event': 'provider_switch',
                    'old_provider': _old_p,
                    'old_model': _old_m,
                    'new_provider': _new_p,
                    'new_model': _new_m,
                })
            except Exception:
                pass
            try:
                self.log_event('provider', {
                    'event': 'provider_switch',
                    'old_provider': _old_p,
                    'old_model': _old_m,
                    'new_provider': _new_p,
                    'new_model': _new_m,
                })
            except Exception:
                pass
            try:
                msg = f"Provider switched: {_old_p} → {_new_p} (model: {_new_m})"
                self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
            except Exception:
                pass

            # UI notice
            self._ui_add_system_message(f"Active provider set to: {provider}.")

            # Reconnect only for Gemini (session-based)
            if provider == 'gemini':
                self._trigger_reconnect(f"Providerwechsel (Gemini)...")
            else:
                # For stateless providers, just refresh panel
                self._ui_refresh_panel()
            return {'ok': True, 'provider': provider}
        except Exception:
            return {'ok': False, 'error': 'set_provider_failed'}

    def refresh_models(self):
        """Refresh provider model list cache (Gemini/OpenRouter/Hugging Face best-effort).

        - Gemini: refresh cached model list from Google GenAI models API (best-effort).
        - OpenRouter: refresh cached /models list.
        - Hugging Face: tries /models; if unavailable, keeps config-defined list.
        """
        try:
            pr = getattr(self, 'provider_router', None)
            psvc = getattr(self, 'provider_service', None)
            if psvc is not None and hasattr(psvc, 'router') and pr is not None:
                psvc.router = pr
            curp = (self._active_provider() or 'gemini')
            curp = (curp or 'gemini').strip().lower()

            if curp == 'gemini':
                if psvc is not None:
                    models, meta = psvc.get_cached_models('gemini', force_refresh=True)
                else:
                    models, meta = pr.get_gemini_models_cached(force_refresh=True) if pr is not None and hasattr(pr, 'get_gemini_models_cached') else ([], {})
                try:
                    self._gemini_models_cache = list(models) if isinstance(models, list) else []
                except Exception:
                    pass
                self._ui_refresh_panel()
                self._ui_add_system_message(
                    f"Gemini models refreshed: {len(models)} (source: {meta.get('source','?')})."
                )
                return {'status': True, 'provider': 'gemini', 'count': len(models), 'meta': meta}

            if curp == 'openrouter':
                if psvc is not None:
                    models, meta = psvc.get_cached_models('openrouter', force_refresh=True)
                else:
                    models, meta = pr.get_openrouter_models_cached(force_refresh=True) if pr is not None and hasattr(pr, 'get_openrouter_models_cached') else ([], {})
                try:
                    self._openrouter_models_cache = list(models) if isinstance(models, list) else []
                except Exception:
                    pass
                self._ui_add_system_message(
                    f"OpenRouter models refreshed: {len(models)} (source: {meta.get('source','?')})."
                )
                return {'status': True, 'provider': 'openrouter', 'count': len(models), 'meta': meta}

            if curp in ('huggingface', 'hf'):
                models = []
                meta = {'source': 'none'}
                try:
                    if psvc is not None:
                        models, meta = psvc.get_cached_models('huggingface', force_refresh=True)
                    elif pr is not None and hasattr(pr, 'get_huggingface_models_cached'):
                        models, meta = pr.get_huggingface_models_cached(force_refresh=True)
                except Exception:
                    models = []
                    meta = {'source': 'none'}
                # Cache models for get_ui() / panel dropdown
                try:
                    self._hf_models_cache = list(models) if isinstance(models, list) else []
                except Exception:
                    pass
                self._ui_refresh_panel()
                # UI notice
                self._ui_add_system_message(
                    f"Hugging Face models refreshed: {len(models)} (source: {meta.get('source','?')})."
                )
                return {'status': True, 'provider': 'huggingface', 'count': len(models), 'meta': meta}

            return {'status': True, 'provider': curp, 'message': 'No refresh needed.'}
        except Exception as e:
            return {'status': False, 'error': str(e)}


    def hf_catalog(self, top_n: int = 200, provider_filter: str = "all"):
        """Fetch & cache Hugging Face Hub catalog models (Top N) and return summary.

        This does NOT switch provider/model automatically. It only refreshes the dropdown source.
        """
        try:
            pr = getattr(self, 'provider_router', None)
            if pr is None:
                try:
                    pr = globals().get('provider_router') or ProviderRouter(globals().get('cfg'))
                    self.provider_router = pr
                except Exception:
                    pr = None
            if pr is None or (not hasattr(pr, 'get_huggingface_catalog_cached')):
                return {"ok": False, "msg": "Hugging Face catalog backend is not initialized (provider_router missing)."}
            top_n_i = int(top_n or 200)
            pf = (provider_filter or "all").strip()
            # Remember last used catalog parameters for backend-side UI refresh
            try:
                setattr(self, 'hf_catalog_top_n', int(top_n_i))
                setattr(self, 'hf_catalog_provider_filter', pf)
            except Exception:
                pass
            models, meta = pr.get_huggingface_catalog_cached(top_n=top_n_i, provider_filter=pf, force_refresh=True)
            try:
                self._hf_models_cache = list(models) if isinstance(models, list) else []
            except Exception:
                pass
            return {"ok": True, "count": len(models), "meta": meta}
        except Exception as e:
            return {"ok": False, "msg": f"HF catalog refresh failed: {e}"}


    def set_model(self, model):
        """Set model for the active provider.

        For Gemini: triggers reconnect. For OpenRouter: stateless, no reconnect required.
        """
        try:
            pr = getattr(self, 'provider_router', None)
            provider = (pr.get_active_provider() if pr is not None and hasattr(pr, 'get_active_provider') else None)
            provider = (provider or (getattr(cfg, 'get_active_provider', lambda: 'gemini')() or 'gemini')).strip().lower()
        except Exception:
            provider = 'gemini'

        # Snapshot old model for auditability (best-effort; no behavior change)
        try:
            _old_model = ''
            if hasattr(cfg, 'get_provider_model'):
                _old_model = str(cfg.get_provider_model(provider) or '').strip()
            if not _old_model:
                _old_model = str(cfg_get_model() or '').strip()
        except Exception:
            _old_model = str(cfg_get_model() or '').strip()

        # No-op guard: selecting the same model again should not trigger a reconnect storm.
        try:
            _new_model = str(model or '').strip()
        except Exception:
            _new_model = ''
        try:
            _cur_model = ''
            if hasattr(cfg, 'get_provider_model'):
                _cur_model = str(cfg.get_provider_model(provider) or '').strip()
            if not _cur_model and hasattr(cfg, 'get_model'):
                try:
                    _cur_model = str(cfg.get_model() or '').strip()
                except Exception:
                    _cur_model = ''
            if _cur_model and _new_model and _cur_model == _new_model:
                try:
                    self.log_event("provider", {"event": "set_model_noop", "provider": provider, "model": _new_model})
                except Exception:
                    pass
                return {"ok": True, "provider": provider, "model": _new_model, "noop": True}
        except Exception:
            pass

        print(f"Switching model for {provider} to: {model}")
        try:
            if hasattr(cfg, 'set_provider_model'):
                cfg.set_provider_model(provider, model)
            else:
                cfg.set_model(model)
        except Exception:
            try:
                cfg.set_model(model)
            except Exception:
                pass

        # Record model switch event (history + session_events)
        try:
            _new_model_eff = str(model or '').strip()
        except Exception:
            _new_model_eff = ''
        try:
            if not isinstance(getattr(self, 'provider_model_history', None), list):
                self.provider_model_history = []
            self.provider_model_history.append({
                'ts': datetime.now().isoformat(),
                'event': 'model_switch',
                'provider': provider,
                'old_model': _old_model,
                'new_model': _new_model_eff,
            })
        except Exception:
            pass
        try:
            self.log_event('provider', {
                'event': 'model_switch',
                'provider': provider,
                'old_model': _old_model,
                'new_model': _new_model_eff,
            })
        except Exception:
            pass
        try:
            msg = f"Model switched ({provider}): {_old_model} → {_new_model_eff}"
            self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
        except Exception:
            pass


        if provider == 'gemini':
            self._trigger_reconnect(f"Modellwechsel ({model})...")
        else:
            self._ui_add_system_message(f"Model set to: {model} (provider: {provider}).")
            self._ui_refresh_panel()

    def set_answer_language(self, lang: str):
        """Set desired language for the LLM answer content only (en/de).

        All deterministic UI renderers (help/state/config/SCI/header/footer/QC) remain English.
        The preference is enforced via a small wrapper directive added to the next user message.
        """
        try:
            lang = (lang or 'de').strip().lower()
            if lang not in ('en', 'de'):
                lang = 'de'
            try:
                self.gov_state.answer_language = lang
            except Exception:
                pass
            try:
                if hasattr(cfg, 'set_answer_language'):
                    cfg.set_answer_language(lang)
            except Exception:
                pass
            self._ui_add_system_message(f"Answer language (LLM) set to: {lang}.")
            self._ui_refresh_panel()
        except Exception:
            pass

    def set_language_policy_mode(self, mode: str):
        """Set language policy mode: production (enforce) or benchmark (log-only)."""
        try:
            m = (mode or "production").strip().lower()
            if m not in ("production", "benchmark"):
                m = "production"
            try:
                self.gov_state.language_policy_mode = m
            except Exception:
                pass
            try:
                if hasattr(cfg, "set_language_policy_mode"):
                    cfg.set_language_policy_mode(m)
            except Exception:
                pass
            self._ui_add_system_message(f"Language policy mode set to: {m}.")
            self._ui_refresh_panel()
        except Exception:
            pass

    def _trigger_reconnect(self, msg):
        self.ready_status = {"status": False, "msg": msg}
        self._ui_add_system_message(f"{msg} Restarting session...")
        threading.Thread(target=self._reconnect_bg).start()

    def _reconnect_bg(self):
        self._connect_api(reason="reconnect")

    def _remember_window_geom(self, win, kind: str):
        """Best-effort: remember window geometry (x/y/width/height) into Comm-SCI-Config.json.
        Works only if the backend exposes these attributes (depends on pywebview backend).
        """
        if not win:
            return {}
        geom = {}
        for k in ("x", "y", "width", "height"):
            try:
                v = getattr(win, k, None)
                if isinstance(v, (int, float)):
                    geom[k] = int(v)
            except Exception:
                pass
        if geom:
            try:
                if kind == "panel":
                    cfg.set_panel_geom(geom)
                    self.panel_geom = geom
                elif kind == "main":
                    cfg.set_main_geom(geom)
            except Exception:
                pass
        return geom

    def _panel_get_embedded_html(self) -> str:
        if _panel_html_source_mod is not None:
            try:
                return _panel_html_source_mod.panel_embedded_html(
                    globals().get("HTML_PANEL_EMBEDDED"),
                    globals().get("HTML_PANEL"),
                )
            except Exception:
                pass
        try:
            txt = globals().get("HTML_PANEL_EMBEDDED")
            if isinstance(txt, str) and txt:
                return txt
        except Exception:
            pass
        try:
            txt = globals().get("HTML_PANEL")
            if isinstance(txt, str):
                return txt
        except Exception:
            pass
        return ""

    def _panel_select_html_for_window(self):
        if _panel_lifecycle_seam_mod is not None:
            try:
                return _panel_lifecycle_seam_mod.panel_window_html_plan(
                    force_embedded_html=bool(getattr(self, "_panel_force_embedded_html", False)),
                    html_panel=globals().get("HTML_PANEL"),
                    html_panel_embedded=globals().get("HTML_PANEL_EMBEDDED"),
                    panel_html_asset_meta=globals().get("PANEL_HTML_ASSET_META"),
                )
            except Exception:
                pass

        if bool(getattr(self, "_panel_force_embedded_html", False)):
            return self._panel_get_embedded_html(), "embedded"
        try:
            txt = globals().get("HTML_PANEL")
            if not isinstance(txt, str) or not txt:
                txt = self._panel_get_embedded_html()
        except Exception:
            txt = self._panel_get_embedded_html()
        try:
            meta = globals().get("PANEL_HTML_ASSET_META")
            src = "external" if isinstance(meta, dict) and str(meta.get("source") or "") == "external" and bool(txt) else "embedded"
        except Exception:
            src = "embedded"
        return txt, src

    def _panel_begin_bootstrap_probe(self, source: str) -> None:
        src = str(source or "embedded")
        try:
            now_iso = datetime.now().isoformat()
        except Exception:
            now_iso = None

        plan = None
        if _panel_lifecycle_seam_mod is not None:
            try:
                plan = _panel_lifecycle_seam_mod.panel_bootstrap_probe_plan(src, now_iso=now_iso)
            except Exception:
                plan = None

        if not isinstance(plan, dict):
            _status = ("pending" if src == "external" else "skipped")
            plan = {
                "source": src,
                "event_action": ("clear" if src == "external" else "set"),
                "bootstrap_state": {
                    "status": _status,
                    "source": src,
                    "reason": "",
                    "created_at": now_iso,
                    "reported_at": None,
                },
            }

        try:
            ev = getattr(self, "_panel_bootstrap_ready_event", None)
            if ev is None:
                ev = threading.Event()
                self._panel_bootstrap_ready_event = ev
            if str(plan.get("event_action") or "") == "clear":
                ev.clear()
            else:
                ev.set()
        except Exception:
            pass

        _st = plan.get("bootstrap_state") if isinstance(plan, dict) else None
        if isinstance(_st, dict):
            self.panel_bootstrap_state = _st
        else:
            self.panel_bootstrap_state = {
                "status": ("pending" if src == "external" else "skipped"),
                "source": src,
                "reason": "",
                "created_at": now_iso,
                "reported_at": None,
            }
        self.panel_html_source = str(plan.get("source") or src)

    def _panel_accept_bootstrap_report(self, payload=None) -> dict:
        payload = payload or {}
        state = getattr(self, "panel_bootstrap_state", None)
        result = None

        if _panel_bootstrap_state_mod is not None:
            try:
                _default_source = str(getattr(self, "panel_html_source", "embedded") or "embedded")
                state = _panel_bootstrap_state_mod.panel_bootstrap_ensure_state(state, default_source=_default_source)
                self.panel_bootstrap_state = state
                try:
                    now_iso = datetime.now().isoformat()
                except Exception:
                    now_iso = None
                state, result = _panel_bootstrap_state_mod.panel_bootstrap_accept_report(
                    state,
                    payload,
                    validate_report=_panel_runtime_selftest_payload_ok,
                    now_iso=now_iso,
                )
                self.panel_bootstrap_state = state
            except Exception:
                result = None

        if result is None:
            if not isinstance(state, dict):
                state = {}
                self.panel_bootstrap_state = state
            if str(state.get("source") or "embedded") != "external":
                try:
                    ev = getattr(self, "_panel_bootstrap_ready_event", None)
                    if ev is not None:
                        ev.set()
                except Exception:
                    pass
                return {"accepted": False, "ignored": True, "reason": "panel_source_not_external"}

            ok, why = _panel_runtime_selftest_payload_ok(payload)
            state["status"] = "passed" if ok else "failed"
            state["reason"] = ("" if ok else str(why or "invalid_runtime_selftest"))
            try:
                state["reported_at"] = datetime.now().isoformat()
            except Exception:
                state["reported_at"] = None
            result = {"accepted": True, "runtime_ok": bool(ok), "reason": ("" if ok else str(why or ""))}

        if result.get("ignored"):
            try:
                ev = getattr(self, "_panel_bootstrap_ready_event", None)
                if ev is not None:
                    ev.set()
            except Exception:
                pass
            return result

        try:
            self.panel_bootstrap_last_report = dict(payload) if isinstance(payload, dict) else {"raw": str(payload)}
        except Exception:
            self.panel_bootstrap_last_report = {"raw": "<unserializable>"}
        try:
            ev = getattr(self, "_panel_bootstrap_ready_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass
        try:
            self.log_event("panel_bootstrap", {
                "event": "runtime_selftest_report",
                "ok": bool(result.get("runtime_ok")),
                "reason": str(result.get("reason") or ""),
                "source": "external",
            })
        except Exception:
            pass
        return result

    def _panel_swap_to_embedded_fallback(self, reason: str = "runtime_selftest_failed") -> bool:
        """Replace a pending/failed external panel with the embedded fallback before showing it."""
        _plan = None
        old_win = getattr(self, "panel_win", None)
        _current_ignore = getattr(self, "_panel_closed_ignore_count", 0)
        try:
            now_iso = datetime.now().isoformat()
        except Exception:
            now_iso = None

        if _panel_lifecycle_seam_mod is not None:
            try:
                _plan = _panel_lifecycle_seam_mod.panel_embedded_fallback_swap_plan(
                    state=getattr(self, "panel_bootstrap_state", None),
                    reason=str(reason or "runtime_selftest_failed"),
                    now_iso=now_iso,
                    old_window_exists=(old_win is not None),
                    ignore_count=_current_ignore,
                )
            except Exception:
                _plan = None

        if not isinstance(_plan, dict):
            try:
                state = getattr(self, "panel_bootstrap_state", None)
                if _panel_bootstrap_state_mod is not None:
                    state = _panel_bootstrap_state_mod.panel_bootstrap_mark_failed_for_fallback(
                        state,
                        reason=str(reason or "runtime_selftest_failed"),
                        now_iso=now_iso,
                    )
                if not isinstance(state, dict):
                    state = {}
                state["status"] = "failed"
                state["reason"] = str(reason or "runtime_selftest_failed")
                state["source"] = "external"
                state["reported_at"] = now_iso
            except Exception:
                state = {"status": "failed", "reason": str(reason or "runtime_selftest_failed"), "source": "external", "reported_at": now_iso}
            _plan = {
                "bootstrap_state": (state if isinstance(state, dict) else {}),
                "ready_event_action": "set",
                "log_event": {"event": "fallback_to_embedded", "reason": str(reason or "runtime_selftest_failed")},
                "recreate_plan": None,
            }

        try:
            self.panel_bootstrap_state = dict(_plan.get("bootstrap_state") or {})
        except Exception:
            pass
        try:
            ev = getattr(self, "_panel_bootstrap_ready_event", None)
            if ev is not None and str(_plan.get("ready_event_action") or "") == "set":
                ev.set()
        except Exception:
            pass
        try:
            _evt = _plan.get("log_event") if isinstance(_plan, dict) else None
            if isinstance(_evt, dict):
                self.log_event("panel_bootstrap", dict(_evt))
        except Exception:
            pass

        try:
            self._remember_window_geom(old_win, "panel")
        except Exception:
            pass
        _recreate_plan = (_plan.get("recreate_plan") if isinstance(_plan, dict) else None)
        self.panel_win = None if not isinstance(_recreate_plan, dict) or _recreate_plan.get("clear_panel_window", True) else self.panel_win
        self.panel_hidden = (False if not isinstance(_recreate_plan, dict) else bool(_recreate_plan.get("panel_hidden", False)))
        self._panel_force_embedded_html = (True if not isinstance(_recreate_plan, dict) else bool(_recreate_plan.get("force_embedded_html", True)))
        try:
            # Create replacement panel first (hidden), then retire the old one.
            # The old window closed callback is ignored once below.
            self._create_panel()
            if old_win is not None:
                if isinstance(_recreate_plan, dict):
                    try:
                        self._panel_closed_ignore_count = int(_recreate_plan.get("next_ignore_count", 1) or 1)
                    except Exception:
                        self._panel_closed_ignore_count = 1
                else:
                    try:
                        self._panel_closed_ignore_count = int(getattr(self, "_panel_closed_ignore_count", 0) or 0) + 1
                    except Exception:
                        self._panel_closed_ignore_count = 1
                try:
                    old_win.destroy()
                except Exception:
                    pass
            return True
        except Exception as e:
            try:
                self.log_event("panel_bootstrap", {
                    "event": "fallback_recreate_failed",
                    "reason": str(e),
                }, level="error")
            except Exception:
                pass
            return False

    def _panel_wait_bootstrap_or_fallback(self, timeout_s=None) -> bool:
        def _local_ready_and_reason(st):
            if not isinstance(st, dict):
                return True, None
            status = str(st.get("status") or "")
            if str(st.get("source") or "embedded") != "external" or status == "passed":
                return True, None
            reason = str(st.get("reason") or "").strip()
            if not reason:
                reason = ("runtime_selftest_timeout" if status == "pending" else "runtime_selftest_failed")
            return False, reason

        state = getattr(self, "panel_bootstrap_state", None)
        if _panel_lifecycle_seam_mod is not None:
            try:
                _initial = _panel_lifecycle_seam_mod.panel_bootstrap_ready_and_reason(state)
            except Exception:
                _initial = None
        else:
            _initial = None

        if isinstance(_initial, tuple):
            ready = bool(_initial[0])
        else:
            if _panel_bootstrap_state_mod is not None:
                try:
                    ready = bool(_panel_bootstrap_state_mod.panel_bootstrap_is_runtime_ready(state))
                except Exception:
                    ready, _ = _local_ready_and_reason(state)
            else:
                ready, _ = _local_ready_and_reason(state)

        if ready:
            return True

        wait_s = timeout_s
        if wait_s is None:
            try:
                wait_s = float(getattr(self, "panel_bootstrap_timeout_s", 2.5) or 2.5)
            except Exception:
                wait_s = 2.5
        _wait_plan = None
        if _panel_lifecycle_seam_mod is not None:
            try:
                _wait_plan = _panel_lifecycle_seam_mod.panel_bootstrap_wait_plan(
                    state,
                    timeout_s=wait_s,
                    default_timeout_s=2.5,
                )
            except Exception:
                _wait_plan = None
        if isinstance(_wait_plan, dict):
            try:
                wait_s = float(_wait_plan.get("wait_seconds", 2.5))
            except Exception:
                wait_s = 2.5
        else:
            try:
                if _panel_bootstrap_state_mod is not None:
                    wait_s = _panel_bootstrap_state_mod.panel_bootstrap_timeout_seconds(wait_s, default=2.5)
                else:
                    wait_s = max(0.0, float(wait_s))
            except Exception:
                wait_s = 2.5

        try:
            ev = getattr(self, "_panel_bootstrap_ready_event", None)
            _should_wait = (
                bool(_wait_plan.get("should_wait"))
                if isinstance(_wait_plan, dict)
                else bool(isinstance(state, dict) and str(state.get("status") or "") == "pending")
            )
            if ev is not None and _should_wait:
                ev.wait(wait_s)
        except Exception:
            pass

        state = getattr(self, "panel_bootstrap_state", None)
        if _panel_lifecycle_seam_mod is not None:
            try:
                ready, reason = _panel_lifecycle_seam_mod.panel_bootstrap_ready_and_reason(state)
            except Exception:
                ready, reason = True, None
        else:
            if _panel_bootstrap_state_mod is not None:
                try:
                    ready = bool(_panel_bootstrap_state_mod.panel_bootstrap_is_runtime_ready(state))
                    reason = (None if ready else _panel_bootstrap_state_mod.panel_bootstrap_fallback_reason(state))
                except Exception:
                    ready, reason = _local_ready_and_reason(state)
            else:
                ready, reason = _local_ready_and_reason(state)
        if ready or not reason:
            return True
        self._panel_swap_to_embedded_fallback(str(reason))
        return False



    def _create_panel(self):
        panel_html, panel_html_source = self._panel_select_html_for_window()
        _create_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _create_plan = _panel_lifecycle_seam_mod.panel_create_window_kwargs_plan(
                    panel_geom=(self.panel_geom or {}),
                    panel_window_title=PANEL_WINDOW_TITLE,
                    panel_html=panel_html,
                    js_api_obj=(self.panel_bridge or self),
                )
        except Exception:
            _create_plan = None
        if isinstance(_create_plan, dict) and isinstance(_create_plan.get("kwargs"), dict):
            kwargs = dict(_create_plan.get("kwargs") or {})
        else:
            # Fallback: safe defaults if seam planning is unavailable.
            kwargs = dict(
                title=PANEL_WINDOW_TITLE,
                html=panel_html,
                js_api=(self.panel_bridge or self),
                width=340,
                height=1000,
                on_top=False,
                x=50,
                y=50,
            )

        # Pre-create hidden (best effort): avoids Cocoa bridge issues and prevents a 'flash' at startup.
        win = None
        try:
            win = webview.create_window(**kwargs, hidden=True)
            self.panel_hidden = True
        except TypeError:
            win = webview.create_window(**kwargs)
            self.panel_hidden = False

        self.panel_win = win
        try:
            self._panel_begin_bootstrap_probe(panel_html_source)
        except Exception:
            pass
        try:
            self._bind_panel_window_events(self.panel_win)
        except Exception:
            # fallback: at least bind closed
            try:
                self.panel_win.events.closed += self.on_panel_closed
            except Exception:
                pass

    def _create_qc_override(self):
        """Pre-create the QC Override dialog window (hidden) to avoid macOS/Cocoa bridge init issues."""
        try:
            if getattr(self, 'qc_win', None) is not None:
                return
        except Exception:
            pass
        try:
            self.qc_bridge = QCBridge(self)
        except Exception:
            self.qc_bridge = None
        try:
            _qc_create_plan = None
            try:
                if _qc_override_window_seam_mod is not None:
                    _qc_create_plan = _qc_override_window_seam_mod.qc_override_window_create_kwargs_plan(
                        html_qc_override=HTML_QC_OVERRIDE,
                        js_api_obj=(getattr(self, 'qc_bridge', None) or self),
                    )
            except Exception:
                _qc_create_plan = None
            if isinstance(_qc_create_plan, dict) and isinstance(_qc_create_plan.get("kwargs"), dict):
                _kw = dict(_qc_create_plan.get("kwargs") or {})
                _title = _kw.pop("title", "Temporary QC override – Profile: ?")
                self.qc_win = webview.create_window(_title, **_kw)
            else:
                self.qc_win = webview.create_window(
                    "Temporary QC override – Profile: ?",
                    html=HTML_QC_OVERRIDE,
                    width=450,
                    height=550,
                    resizable=False,
                    hidden=True,
                    on_top=True,
                    js_api=getattr(self, 'qc_bridge', None) or self
                )
        except Exception:
            try:
                self.qc_win = None
            except Exception:
                pass

    def show_qc_override(self):
        """Show QC Override dialog window."""
        try:
            _show_plan = None
            try:
                if _qc_override_window_seam_mod is not None:
                    _show_plan = _qc_override_window_seam_mod.qc_override_show_plan(
                        window_exists=(getattr(self, 'qc_win', None) is not None)
                    )
            except Exception:
                _show_plan = None
            win = getattr(self, 'qc_win', None)
            _create_if_missing = (
                bool(_show_plan.get("create_if_missing"))
                if isinstance(_show_plan, dict)
                else (win is None)
            )
            if win is None and _create_if_missing:
                self._create_qc_override()
                win = getattr(self, 'qc_win', None)
            if win is None:
                _err = (
                    str(_show_plan.get("error_if_unavailable") or "qc_win unavailable")
                    if isinstance(_show_plan, dict)
                    else "qc_win unavailable"
                )
                try:
                    self._qc_override_open = False
                except Exception:
                    pass
                return {'ok': False, 'error': _err}
            _methods = (
                tuple(_show_plan.get("window_methods") or ())
                if isinstance(_show_plan, dict)
                else ("show", "bring_to_front")
            )
            for _meth in _methods:
                try:
                    if hasattr(win, _meth):
                        getattr(win, _meth)()
                except Exception:
                    pass
            # Re-sync dialog sliders/title on every show (profile switch resets overrides, but hidden dialog UI may be stale).
            try:
                if hasattr(win, 'evaluate_js'):
                    win.evaluate_js(
                        "(function(){try{"
                        "if(window.QCUI && typeof window.QCUI.refreshState==='function'){window.QCUI.refreshState();return 'ok';}"
                        "if(window.QCUI && typeof window.QCUI.boot==='function'){window.QCUI.boot();return 'boot';}"
                        "return 'noop';"
                        "}catch(e){return 'err';}})();"
                    )
            except Exception:
                pass
            _ok = (
                _show_plan.get("success_result")
                if isinstance(_show_plan, dict) and isinstance(_show_plan.get("success_result"), dict)
                else {'ok': True}
            )
            try:
                self._qc_override_open = bool(dict(_ok).get("ok", True))
            except Exception:
                self._qc_override_open = True
            return dict(_ok)
        except Exception as e:
            try:
                self._qc_override_open = False
            except Exception:
                pass
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_get_state(self, _payload=None):
        """Return current QC defaults (corridors) and current overrides for UI."""
        try:
            prof = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
            defaults = {}
            try:
                gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                prof_data = ((getattr(gov_obj, 'data', {}) or {}).get('profiles', {}) or {}).get(prof, {}) or {}
                defaults = prof_data.get('qc_target') or {}
                if not isinstance(defaults, dict):
                    defaults = {}
            except Exception:
                defaults = {}
            ovs = {}
            try:
                ovs = getattr(self.gov_state, 'qc_overrides', {}) or {}
                if not isinstance(ovs, dict):
                    ovs = {}
            except Exception:
                ovs = {}
            return {'ok': True, 'profile': prof, 'defaults': defaults, 'overrides': ovs, 'note': 'Online.'}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_override_apply(self, values):
        """Apply QC overrides from UI; session-local."""
        try:
            if not isinstance(values, dict):
                return {'ok': False, 'error': 'values must be dict'}
            clean = {}
            mapping = {
                'clarity':'clarity','brevity':'brevity','evidence':'evidence','empathy':'empathy','consistency':'consistency','neutrality':'neutrality',
                'klarheit':'clarity','kürze':'brevity','kuerze':'brevity','evidenz':'evidence','empathie':'empathy','konsistenz':'consistency','neutralität':'neutrality','neutralitaet':'neutrality',
            }
            for k, v in values.items():
                try:
                    vi = int(v)
                except Exception:
                    continue
                if vi < 0: vi = 0
                if vi > 3: vi = 3
                kk = (k or '').strip()
                if not kk:
                    continue
                low = kk.lower()
                key = mapping.get(low)
                if not key:
                    continue
                clean[key] = vi

            try:
                self.gov_state.qc_overrides = dict(clean)
            except Exception:
                try:
                    setattr(self.gov_state, 'qc_overrides', dict(clean))
                except Exception:
                    pass
            # Mirror overrides to gov-manager for deterministic QC enforcement (session-local).
            try:
                gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                if gov_obj is not None:
                    setattr(gov_obj, 'qc_overrides', dict(clean))
                    gov.runtime_state = self.gov_state
                    setattr(gov_obj, 'runtime_state', self.gov_state)
            except Exception:
                pass
            try:
                self._qc_override_prompt_reset_pending = False
            except Exception:
                pass

            _ui_plan = None
            try:
                if _qc_override_window_seam_mod is not None:
                    _ui_plan = _qc_override_window_seam_mod.qc_override_apply_ui_plan(
                        clean_overrides=dict(clean),
                        qc_window_exists=(getattr(self, 'qc_win', None) is not None),
                    )
            except Exception:
                _ui_plan = None
            msg = (
                str(_ui_plan.get("history_message"))
                if isinstance(_ui_plan, dict) and _ui_plan.get("history_message") is not None
                else ("QC-Overrides gesetzt: " + (", ".join(
                    [f"{ {'clarity':'Clarity','brevity':'Brevity','evidence':'Evidence','empathy':'Empathy','consistency':'Consistency','neutrality':'Neutrality'}.get(k, k)}={clean[k]}"
                     for k in ['clarity','brevity','evidence','empathy','consistency','neutrality'] if k in clean]
                ) if clean else "(leer)"))
            )

            try:
                self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
            except Exception:
                pass

            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    _ui_msg = (
                        str(_ui_plan.get("main_ui_message"))
                        if isinstance(_ui_plan, dict) and _ui_plan.get("main_ui_message") is not None
                        else msg
                    )
                    js_msg = _json.dumps(_ui_msg, ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass

            try:
                _win = getattr(self, 'qc_win', None)
                _methods = (
                    tuple(_ui_plan.get("qc_window_methods") or ())
                    if isinstance(_ui_plan, dict)
                    else (("hide",) if _win is not None else ())
                )
                for _meth in _methods:
                    if _win is not None and hasattr(_win, _meth):
                        try:
                            getattr(_win, _meth)()
                        except Exception:
                            pass
                if "hide" in _methods:
                    try:
                        self._qc_override_open = False
                    except Exception:
                        pass
            except Exception:
                pass

            _ok = (
                _ui_plan.get("success_result")
                if isinstance(_ui_plan, dict) and isinstance(_ui_plan.get("success_result"), dict)
                else {'ok': True, 'overrides': clean}
            )
            return dict(_ok)
        except Exception as e:
            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    _warn_prefix = (
                        str(_ui_plan.get("warn_prefix") or "[WARN] QC Override Apply failed: ")
                        if isinstance(locals().get("_ui_plan"), dict)
                        else "[WARN] QC Override Apply failed: "
                    )
                    js_msg = _json.dumps(f"{_warn_prefix}{type(e).__name__}: {e}", ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_override_clear(self, _payload=None):
        """Clear QC overrides."""
        try:
            _ui_plan = None
            try:
                self.gov_state.qc_overrides = {}
            except Exception:
                try:
                    setattr(self.gov_state, 'qc_overrides', {})
                except Exception:
                    pass
            # Mirror clear to gov-manager as well.
            try:
                gov_obj = getattr(self, 'gov', None) or globals().get('gov')
                if gov_obj is not None:
                    setattr(gov_obj, 'qc_overrides', {})
                    setattr(gov_obj, 'runtime_state', self.gov_state)
            except Exception:
                pass
            try:
                self._qc_override_prompt_reset_pending = True
            except Exception:
                pass
            try:
                if _qc_override_window_seam_mod is not None:
                    _ui_plan = _qc_override_window_seam_mod.qc_override_clear_ui_plan(
                        qc_window_exists=(getattr(self, 'qc_win', None) is not None)
                    )
            except Exception:
                _ui_plan = None
            msg = (
                str(_ui_plan.get("history_message"))
                if isinstance(_ui_plan, dict) and _ui_plan.get("history_message") is not None
                else "QC-Overrides zurückgesetzt"
            )
            try:
                self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
            except Exception:
                pass
            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    _ui_msg = (
                        str(_ui_plan.get("main_ui_message"))
                        if isinstance(_ui_plan, dict) and _ui_plan.get("main_ui_message") is not None
                        else msg
                    )
                    js_msg = _json.dumps(_ui_msg, ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass
            try:
                _win = getattr(self, 'qc_win', None)
                _methods = (
                    tuple(_ui_plan.get("qc_window_methods") or ())
                    if isinstance(_ui_plan, dict)
                    else (("hide",) if _win is not None else ())
                )
                for _meth in _methods:
                    if _win is not None and hasattr(_win, _meth):
                        try:
                            getattr(_win, _meth)()
                        except Exception:
                            pass
                if "hide" in _methods:
                    try:
                        self._qc_override_open = False
                    except Exception:
                        pass
            except Exception:
                pass
            _ok = (
                _ui_plan.get("success_result")
                if isinstance(_ui_plan, dict) and isinstance(_ui_plan.get("success_result"), dict)
                else {'ok': True}
            )
            return dict(_ok)
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_override_cancel(self, _payload=None):
        """Close QC dialog without changes."""
        try:
            _cancel_plan = None
            try:
                if _qc_override_window_seam_mod is not None:
                    _cancel_plan = _qc_override_window_seam_mod.qc_override_cancel_plan(
                        window_exists=(getattr(self, 'qc_win', None) is not None)
                    )
            except Exception:
                _cancel_plan = None
            try:
                _win = getattr(self, 'qc_win', None)
                _methods = (
                    tuple(_cancel_plan.get("window_methods") or ())
                    if isinstance(_cancel_plan, dict)
                    else (("hide",) if _win is not None else ())
                )
                for _meth in _methods:
                    if _win is not None and hasattr(_win, _meth):
                        getattr(_win, _meth)()
                if "hide" in _methods:
                    try:
                        self._qc_override_open = False
                    except Exception:
                        pass
            except Exception:
                pass
            _ok = (
                _cancel_plan.get("success_result")
                if isinstance(_cancel_plan, dict) and isinstance(_cancel_plan.get("success_result"), dict)
                else {'ok': True}
            )
            return dict(_ok)
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def _rebuild_panel(self, reason: str = "reload"):
        """Robust panel rebuild.

        Some pywebview backends (macOS in particular) can get into a weird state after multiple
        evaluate_js refreshes. Recreating the panel window is the most reliable fix.
        Preserves the user's last panel position/size when possible.
        After a successful rebuild, the panel is shown again deterministically.
        """
        _rebuild_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _rebuild_plan = _panel_lifecycle_seam_mod.panel_rebuild_plan(
                    reason=reason,
                    panel_window_exists=(getattr(self, "panel_win", None) is not None),
                )
        except Exception:
            _rebuild_plan = None

        try:
            # remember geometry before destroying
            _should_remember = (
                bool(_rebuild_plan.get("remember_geometry_before_destroy"))
                if isinstance(_rebuild_plan, dict)
                else bool(self.panel_win)
            )
            if _should_remember and self.panel_win:
                self._remember_window_geom(self.panel_win, "panel")
        except Exception:
            pass

        # Try to destroy existing panel window
        try:
            _should_destroy = (
                bool(_rebuild_plan.get("destroy_old_window"))
                if isinstance(_rebuild_plan, dict)
                else bool(self.panel_win)
            )
            if _should_destroy and self.panel_win:
                self.panel_win.destroy()
        except Exception:
            pass

        self.panel_win = None
        try:
            self.panel_bridge = PanelBridge(self)
        except Exception:
            self.panel_bridge = None
        self.panel_hidden = (
            bool(_rebuild_plan.get("reset_panel_hidden", False))
            if isinstance(_rebuild_plan, dict)
            else False
        )

        # Recreate and show panel again (ruleset reload must not leave panel hidden)
        try:
            self._create_panel()
            # best-effort extra window methods from seam plan
            try:
                _methods = (
                    tuple(_rebuild_plan.get("post_create_window_methods") or ())
                    if isinstance(_rebuild_plan, dict)
                    else ("focus", "restore")
                )
                for _meth in _methods:
                    if hasattr(self.panel_win, _meth):
                        getattr(self.panel_win, _meth)()
            except Exception:
                pass
            try:
                self._show_panel(activate_panel=False, return_focus_to_main=True)
            except Exception:
                pass
            if self.main_win:
                _msg = (
                    str(_rebuild_plan.get("success_main_message") or f"Panel rebuilt ({reason}).")
                    if isinstance(_rebuild_plan, dict)
                    else f"Panel rebuilt ({reason})."
                )
                self.main_win.evaluate_js(f"addMsg('sys', '{_msg}')")
        except Exception as e:
            if self.main_win:
                safe = str(e).replace("'", "'").replace('"', '\"')
                _prefix = (
                    str(_rebuild_plan.get("failure_main_message_prefix") or "Panel rebuild failed: ")
                    if isinstance(_rebuild_plan, dict)
                    else "Panel rebuild failed: "
                )
                self.main_win.evaluate_js(f"addMsg('sys', '{_prefix}{safe}')")

    def _hide_panel(self):
        _win = getattr(self, "panel_win", None)
        _hide_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _hide_plan = _panel_lifecycle_seam_mod.panel_hide_plan(
                    panel_window_exists=(_win is not None),
                    has_hide=bool(_win is not None and hasattr(_win, "hide")),
                    has_minimize=bool(_win is not None and hasattr(_win, "minimize")),
                )
        except Exception:
            _hide_plan = None

        if isinstance(_hide_plan, dict):
            _action = str(_hide_plan.get("action") or "noop")
            if _action == "noop":
                return
            if bool(_hide_plan.get("remember_geometry")):
                try:
                    self._remember_window_geom(_win, "panel")
                except Exception:
                    pass
            if _action == "hide":
                try:
                    _win.hide()
                    self.panel_hidden = bool(_hide_plan.get("panel_hidden", True))
                    return
                except Exception:
                    pass
            elif _action == "minimize":
                try:
                    _win.minimize()
                    self.panel_hidden = bool(_hide_plan.get("panel_hidden", True))
                    return
                except Exception:
                    pass
            elif _action == "destroy":
                try:
                    _win.destroy()
                except Exception:
                    pass
                if bool(_hide_plan.get("clear_panel_window", True)):
                    self.panel_win = None
                self.panel_hidden = bool(_hide_plan.get("panel_hidden", False))
                return

        if not self.panel_win:
            return
        # remember geometry before hiding
        self._remember_window_geom(self.panel_win, "panel")
        # Prefer real hide if supported
        try:
            if hasattr(self.panel_win, "hide"):
                self.panel_win.hide()
                self.panel_hidden = True
                return
        except Exception:
            pass
        # Fallback: minimize
        try:
            if hasattr(self.panel_win, "minimize"):
                self.panel_win.minimize()
                self.panel_hidden = True
                return
        except Exception:
            pass
        # Last resort: destroy and recreate later
        try:
            self.panel_win.destroy()
        except Exception:
            pass
        self.panel_win = None
        self.panel_hidden = False

    def _focus_main_window(self, *, skip_if_qc_override: bool = True):
        """Best-effort main-window activation for panel toggle UX."""
        try:
            if skip_if_qc_override and bool(getattr(self, "_qc_override_open", False)):
                return False
            win = getattr(self, "main_win", None)
            if win is None:
                return False
            used = False
            for _meth in ("restore", "bring_to_front", "focus"):
                try:
                    if hasattr(win, _meth):
                        getattr(win, _meth)()
                        used = True
                except Exception:
                    pass
            return bool(used)
        except Exception:
            return False

    def _show_panel(self, *, activate_panel: bool = True, return_focus_to_main: bool = False):
        _show_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _show_plan = _panel_lifecycle_seam_mod.panel_show_plan(
                    panel_window_exists=(getattr(self, "panel_win", None) is not None),
                    activate_panel=bool(activate_panel),
                )
        except Exception:
            _show_plan = None
        if isinstance(_show_plan, dict):
            _action = str(_show_plan.get("action") or "")
            if _action == "create_panel":
                self._create_panel()
                return
            if bool(_show_plan.get("wait_bootstrap_before_show")):
                try:
                    # S7: external panel.html is only shown after a runtime self-test callback.
                    # If it never arrives, rebuild hidden with the embedded fallback first.
                    self._panel_wait_bootstrap_or_fallback()
                except Exception:
                    pass
            try:
                _win = getattr(self, "panel_win", None)
                for _meth in tuple(_show_plan.get("window_methods") or ()):
                    if _win is not None and hasattr(_win, _meth):
                        getattr(_win, _meth)()
            except Exception:
                pass
            if _show_plan.get("panel_hidden") is not None:
                self.panel_hidden = bool(_show_plan.get("panel_hidden"))
            if return_focus_to_main:
                try:
                    self._focus_main_window(skip_if_qc_override=True)
                except Exception:
                    pass
            return

        if not self.panel_win:
            self._create_panel()
            return
        try:
            # S7: external panel.html is only shown after a runtime self-test callback.
            # If it never arrives, rebuild hidden with the embedded fallback first.
            self._panel_wait_bootstrap_or_fallback()
        except Exception:
            pass
        try:
            if hasattr(self.panel_win, "show"):
                self.panel_win.show()
            if hasattr(self.panel_win, "restore"):
                self.panel_win.restore()
            if bool(activate_panel) and hasattr(self.panel_win, "focus"):
                self.panel_win.focus()
        except Exception:
            pass
        self.panel_hidden = False
        if return_focus_to_main:
            try:
                self._focus_main_window(skip_if_qc_override=True)
            except Exception:
                pass

    def ensure_panel_visible(self):
        """Called from JS once the main UI is ready: show the panel automatically."""
        try:
            _plan = None
            try:
                if _panel_lifecycle_seam_mod is not None:
                    _plan = _panel_lifecycle_seam_mod.panel_ensure_visible_plan(
                        panel_window_exists=(getattr(self, "panel_win", None) is not None),
                        panel_hidden=bool(getattr(self, "panel_hidden", False)),
                    )
            except Exception:
                _plan = None
            if isinstance(_plan, dict):
                _action = str(_plan.get("action") or "")
                if _action == "create_panel":
                    self._create_panel()
                    return
                if _action == "show_panel":
                    self._show_panel()
                    return
                if _action == "focus_existing":
                    try:
                        _win = getattr(self, "panel_win", None)
                        for _meth in tuple(_plan.get("window_methods") or ()):
                            if _win is not None and hasattr(_win, _meth):
                                getattr(_win, _meth)()
                    except Exception:
                        pass
                    return
            if not self.panel_win:
                self._create_panel()
            else:
                # If minimized/hidden: bring back
                if self.panel_hidden:
                    self._show_panel()
                else:
                    try:
                        if hasattr(self.panel_win, "restore"):
                            self.panel_win.restore()
                        if hasattr(self.panel_win, "focus"):
                            self.panel_win.focus()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Panel] ensure_panel_visible error: {e}")

    def settings(self):
        """Toggle panel visibility."""
        _plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _plan = _panel_lifecycle_seam_mod.panel_settings_toggle_plan(
                    panel_window_exists=(getattr(self, "panel_win", None) is not None),
                    panel_hidden=bool(getattr(self, "panel_hidden", False)),
                )
        except Exception:
            _plan = None
            if isinstance(_plan, dict):
                _action = str(_plan.get("action") or "")
                if _action == "create_panel":
                    self._create_panel()
                    return
                if _action == "show_panel":
                    self._show_panel(activate_panel=False, return_focus_to_main=True)
                    return
                if _action == "hide_panel":
                    self._hide_panel()
                    return
        if not self.panel_win:
            self._create_panel()
            return
        # If currently hidden/minimized -> show; else hide
        if self.panel_hidden:
            self._show_panel(activate_panel=False, return_focus_to_main=True)
        else:
            self._hide_panel()

    def on_panel_closing(self):
        """Intercept the panel close action ("X") and hide instead of destroying when possible."""
        _closing_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _w = getattr(self, "panel_win", None)
                _closing_plan = _panel_lifecycle_seam_mod.panel_on_closing_plan(
                    panel_window_exists=(_w is not None),
                    has_hide=bool(_w is not None and hasattr(_w, "hide")),
                )
        except Exception:
            _closing_plan = None
        try:
            self._hide_panel()
        except Exception:
            # best-effort hide
            try:
                _fallback_action = (
                    str(_closing_plan.get("fallback_action") or "")
                    if isinstance(_closing_plan, dict)
                    else "direct_hide"
                )
                if _fallback_action == "direct_hide" and self.panel_win and hasattr(self.panel_win, "hide"):
                    self.panel_win.hide()
                    self.panel_hidden = bool(
                        _closing_plan.get("fallback_sets_panel_hidden", True)
                        if isinstance(_closing_plan, dict)
                        else True
                    )
            except Exception:
                pass
        # Returning False cancels the close on backends that support it (best-effort).
        if isinstance(_closing_plan, dict):
            return bool(_closing_plan.get("return_value", False))
        return False

    def _bind_panel_window_events(self, win):
        """Bind panel lifecycle events defensively.

        - If the backend supports a cancelable 'closing' event, we hide the panel (keeps state, avoids destroy).
        - Always bind 'closed' as a fallback cleanup if the window is destroyed anyway.
        """
        if not win:
            return
        evs = getattr(win, "events", None)
        _bind_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _bind_plan = _panel_lifecycle_seam_mod.panel_bind_window_events_plan(
                    window_exists=bool(win),
                    has_events=(evs is not None),
                    has_closing_event=(getattr(evs, "closing", None) is not None),
                    has_closed_event=(getattr(evs, "closed", None) is not None),
                )
        except Exception:
            _bind_plan = None

        closing_ev = getattr(evs, "closing", None)
        _bind_closing = (
            bool(_bind_plan.get("bind_closing"))
            if isinstance(_bind_plan, dict)
            else (closing_ev is not None)
        )
        if _bind_closing and closing_ev is not None:
            try:
                closing_ev += self.on_panel_closing
            except Exception:
                pass

        closed_ev = getattr(evs, "closed", None)
        _bind_closed = (
            bool(_bind_plan.get("bind_closed"))
            if isinstance(_bind_plan, dict)
            else (closed_ev is not None)
        )
        if _bind_closed and closed_ev is not None:
            try:
                closed_ev += self.on_panel_closed
            except Exception:
                pass

    def on_panel_closed(self):
        # S7 fallback can destroy a retired panel after a replacement panel already exists.
        # Ignore that one late closed event so it does not clear the replacement handle.
        _closed_plan = None
        try:
            if _panel_lifecycle_seam_mod is not None:
                _closed_plan = _panel_lifecycle_seam_mod.panel_closed_event_plan(
                    panel_window_exists=(getattr(self, "panel_win", None) is not None),
                    ignore_count=getattr(self, "_panel_closed_ignore_count", 0),
                    panel_html_source=getattr(self, "panel_html_source", "embedded"),
                )
        except Exception:
            _closed_plan = None

        try:
            if isinstance(_closed_plan, dict):
                self._panel_closed_ignore_count = int(_closed_plan.get("next_ignore_count", 0) or 0)
                if bool(_closed_plan.get("ignore_event")):
                    return
            else:
                _panel_exists = (getattr(self, "panel_win", None) is not None)
                _ignore_count = getattr(self, "_panel_closed_ignore_count", 0)
                _ignore = False
                _next_n = None
                if _panel_window_fallback_mod is not None:
                    try:
                        _ignore, _next_n = _panel_window_fallback_mod.panel_closed_retired_event_decision(
                            panel_window_exists=bool(_panel_exists),
                            ignore_count=_ignore_count,
                        )
                    except Exception:
                        _next_n = None
                if _next_n is not None:
                    self._panel_closed_ignore_count = _next_n
                    if _ignore:
                        return
                elif _panel_exists:
                    _n = int(getattr(self, "_panel_closed_ignore_count", 0) or 0)
                    if _n > 0:
                        self._panel_closed_ignore_count = _n - 1
                        return
        except Exception:
            pass
        # remember last geometry if possible
        try:
            self._remember_window_geom(self.panel_win, "panel")
        except Exception:
            pass
        try:
            if isinstance(_closed_plan, dict):
                _closed_state = dict(_closed_plan.get("bootstrap_state") or {})
            else:
                _src = str(getattr(self, "panel_html_source", "embedded") or "embedded")
                _closed_state = {
                    "status": "idle",
                    "source": _src,
                    "reason": "window_closed",
                    "created_at": None,
                    "reported_at": None,
                }
                if _panel_bootstrap_state_mod is not None:
                    try:
                        _closed_state = _panel_bootstrap_state_mod.panel_bootstrap_closed_state(_src)
                    except Exception:
                        pass
            self.panel_bootstrap_state = _closed_state
            ev = getattr(self, "_panel_bootstrap_ready_event", None)
            _ev_action = (
                str(_closed_plan.get("ready_event_action") or "")
                if isinstance(_closed_plan, dict)
                else "set"
            )
            if ev is not None and _ev_action == "set":
                ev.set()
        except Exception:
            pass
        if isinstance(_closed_plan, dict):
            self.panel_win = (None if bool(_closed_plan.get("clear_panel_window", True)) else self.panel_win)
            self.panel_hidden = bool(_closed_plan.get("panel_hidden", False))
        else:
            self.panel_win = None
            self.panel_hidden = False


    def export(self, audit_event=None, audit_only: bool = False, extra_audit=None):
        """Export chat + audit logs deterministically.

        - Filenames include microseconds for uniqueness.
        - If audit_only is True, only the audit file is written.
        - audit_event (dict) is included in audit payload to make Comm Audit visibly different.
        Returns (chat_path, audit_path).
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        chat_path = None
        if not audit_only:
            # Chat-Log (voller Verlauf)
            chat_name = f"Log_{ts}.json"
            chat_path = os.path.join(CHAT_LOG_DIR, chat_name)
            try:
                _chat_payload = {"meta": WRAPPER_NAME, "model": cfg_get_model(), "history": self.history}
                # --- B8: Persist provider/model + fork metadata (additive; backwards compatible) ---
                try:
                    pr = getattr(self, 'provider_router', None)
                    _p = (pr.get_active_provider() if pr is not None and hasattr(pr, 'get_active_provider') else None)
                    _p = (_p or (getattr(cfg, 'get_active_provider', lambda: 'gemini')() or 'gemini')).strip().lower()
                except Exception:
                    _p = 'gemini'
                try:
                    _m = ''
                    if hasattr(cfg, 'get_provider_model'):
                        _m = str(cfg.get_provider_model(_p) or '').strip()
                    if not _m:
                        _m = str(cfg_get_model() or '').strip()
                except Exception:
                    _m = str(cfg_get_model() or '').strip()
                try:
                    _chat_payload["active_provider"] = _p
                    _chat_payload["active_model"] = _m
                    _chat_payload["provider_model_history"] = list(getattr(self, "provider_model_history", []) or [])
                    _chat_payload["forked_from_log_path"] = getattr(self, "forked_from_log_path", None)
                    _chat_payload["fork_parent_trace_id"] = getattr(self, "fork_parent_trace_id", None)
                except Exception:
                    pass
                # --- /B8 ---
                # --- B9: Ensure trace_id/session_id persisted in chat logs (for fork provenance) ---
                try:
                    import uuid as _uuid
                    _sid = str(getattr(self, "session_id", "") or "").strip()
                    if not _sid:
                        _sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + _uuid.uuid4().hex[:6]
                        self.session_id = _sid
                    _tid = str(getattr(self, "trace_id", "") or "").strip()
                    if not _tid:
                        _tid = _sid
                        self.trace_id = _tid
                    _chat_payload["session_id"] = _sid
                    _chat_payload["trace_id"] = _tid
                except Exception:
                    pass
                # --- /B9 ---
                svc = getattr(self, 'storage_service', None)
                wrote = False
                if svc is not None and hasattr(svc, 'write_json'):
                    wrote = bool(svc.write_json(chat_path, _chat_payload, indent=2, ensure_ascii=True))
                if not wrote:
                    os.makedirs(CHAT_LOG_DIR, exist_ok=True)
                    with open(chat_path, "w", encoding="utf-8") as f:
                        json.dump(_chat_payload, f, indent=2)
                print(f"Exportiert (Chat): {chat_path}")
            except Exception as e:
                print(f"[System] Export-Error (Chat): {e}")

        # Audit-Log (v2 Schema)
        audit_name = f"Audit_{ts}.json"
        audit_path = os.path.join(AUDIT_LOG_DIR, audit_name)
        try:
            # Write audit v2 (provider/model/trace_id aware). Do not re-export chat here.
            self.export_audit_v2(audit_event=audit_event, audit_only=True, ts=ts, audit_path=audit_path)
        except Exception as e:
            # Fail-open: at least keep chat export.
            print(f"[System] Export-Error (Audit v2): {e}")

        return chat_path, audit_path

    def _provider_snapshot(self) -> dict:
        """Sanitized provider config snapshot (no secrets). Best-effort."""
        try:
            provider = 'unknown'
            try:
                if hasattr(self, '_active_provider'):
                    provider = self._active_provider() or 'unknown'
            except Exception:
                provider = 'unknown'

            model = None
            try:
                model = cfg_get_model()
            except Exception:
                model = None

            snap = {
                'active_provider': provider or 'unknown',
                'model': model or 'unknown',
            }

            # Provider-specific (best-effort)
            if provider == 'gemini':
                snap['temperature'] = 0.0
                snap['top_p'] = 0.1
                snap['max_tokens'] = 65536
                snap['api_key_source'] = (
                    'env:GEMINI_API_KEY' if os.getenv('GEMINI_API_KEY')
                    else 'env:GOOGLE_API_KEY' if os.getenv('GOOGLE_API_KEY')
                    else 'file:Config/Comm-SCI-API-Keys.json'
                )
            elif provider == 'openrouter':
                snap['api_key_source'] = (
                    'env:OPENROUTER_API_KEY' if os.getenv('OPENROUTER_API_KEY')
                    else 'file:Config/Comm-SCI-API-Keys.json'
                )
                try:
                    snap['base_url'] = getattr(getattr(self, 'provider_router', None), 'openrouter_base_url', None) or 'unknown'
                except Exception:
                    snap['base_url'] = 'unknown'
            elif provider == 'huggingface':
                snap['api_key_source'] = (
                    'env:HF_TOKEN' if os.getenv('HF_TOKEN')
                    else 'file:Config/Comm-SCI-API-Keys.json'
                )
            return snap
        except Exception:
            return {'active_provider': 'unknown', 'model': 'unknown'}


    def export_audit_v2(self, *, audit_event=None, audit_only: bool = False, ts: str | None = None, audit_path: str | None = None):
        """Enhanced audit export (v2). Keeps legacy export() untouched."""
        import platform
        import sys
        import hashlib

        if ts is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        def file_hash(path: str) -> str:
            try:
                with open(path, 'rb') as f:
                    return 'sha256:' + hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                return 'unknown'

        def ruleset_hash() -> str:
            try:
                raw = getattr(gov, 'raw_json', '') or ''
                if raw:
                    return 'sha256:' + hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
                fn = getattr(gov, 'current_filename', '') or ''
                return file_hash(fn) if fn else 'unknown'
            except Exception:
                return 'unknown'

        def duration_seconds():
            try:
                start = getattr(self, 'session_start_dt', None)
                if start:
                    return int((datetime.now() - start).total_seconds())
            except Exception:
                pass
            return None

        payload = {
            'export_version': '2.0',
            'export_timestamp': datetime.now().isoformat(),
            'session_metadata': {
                'session_id': getattr(self, 'session_id', 'unknown'),
                'trace_id': getattr(self, 'trace_id', getattr(self, 'session_id', 'unknown')),

                'session_start': getattr(self, 'session_start_dt', datetime.now()).isoformat(),
                'session_end': datetime.now().isoformat(),
                'duration_seconds': duration_seconds(),
                'total_requests': getattr(self, 'session_requests', getattr(self, 'session_req_count', 0)),
                'rate_limit_hits': getattr(self, 'session_rate_limit_hits', 0),
                'repair_passes': getattr(self, 'session_repair_passes', 0),
                'csc_applied_count': getattr(self, 'session_csc_applied_count', 0),
                'cross_version_guard_hits': getattr(self, 'session_guard_hits', 0),
            },
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'os': platform.system(),
                'platform': platform.platform(),
                'pywebview_version': getattr(webview, '__version__', 'unknown'),
                'comm_sci_version': (getattr(gov, 'data', {}) or {}).get('version', 'unknown'),
                'wrapper_file_hash': file_hash(__file__),
            },
            'provider_config': self._provider_snapshot(),
            'governance_config': {
                'ruleset_file': os.path.basename(getattr(gov, 'current_filename', 'unknown') or 'unknown'),
                'ruleset_version': (getattr(gov, 'data', {}) or {}).get('version', 'unknown'),
                'ruleset_hash': ruleset_hash(),
                'default_profile': (getattr(gov, 'data', {}) or {}).get('default_profile', 'Standard'),
                'cross_version_guard_enabled': True,
                'language_policy_mode': getattr(getattr(self, 'gov_state', None), 'language_policy_mode', 'production'),
            },
            'conversation': getattr(self, 'history', []) or [],
            'governance_logs_tail': (getattr(gov, 'logs', []) or [])[-50:],
            'session_events': getattr(self, 'session_events', []) or [],
        }

        if audit_event:
            payload['audit_event'] = audit_event

        # Write audit file
        if audit_path is None:
            audit_path = os.path.join(AUDIT_LOG_DIR, f"Audit_{ts}.json")
        try:
            svc = getattr(self, 'storage_service', None)
            wrote = False
            if svc is not None and hasattr(svc, 'write_json'):
                wrote = bool(svc.write_json(audit_path, payload, indent=2, ensure_ascii=False))
            if not wrote:
                os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
                with open(audit_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Exportiert (Audit v2): {audit_path}")
        except Exception as e:
            print(f"[Export] Audit v2 write failed: {e}")

        chat_path = None
        if not audit_only:
            chat_path = os.path.join(CHAT_LOG_DIR, f"Log_{ts}.json")
            try:
                chat_payload = {'meta': WRAPPER_NAME, 'model': cfg_get_model(), 'history': getattr(self, 'history', []) or []}
                svc = getattr(self, 'storage_service', None)
                wrote = False
                if svc is not None and hasattr(svc, 'write_json'):
                    wrote = bool(svc.write_json(chat_path, chat_payload, indent=2, ensure_ascii=False))
                if not wrote:
                    os.makedirs(CHAT_LOG_DIR, exist_ok=True)
                    with open(chat_path, 'w', encoding='utf-8') as f:
                        json.dump(chat_payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Export] Chat write failed: {e}")

        return chat_path, audit_path



def _normalize_provider_key_id(provider: str) -> str:
    try:
        p = str(provider or '').strip().lower()
    except Exception:
        p = ''
    if p in {'google'}:
        return 'gemini'
    if p in {'hf'}:
        return 'huggingface'
    return p


def _provider_key_env_names(provider: str) -> list[str]:
    p = _normalize_provider_key_id(provider)
    if p == 'gemini':
        return ['GEMINI_API_KEY', 'GOOGLE_API_KEY']
    if p == 'openrouter':
        return ['OPENROUTER_API_KEY']
    if p == 'huggingface':
        return ['HF_TOKEN', 'HUGGINGFACE_TOKEN']
    return []


def set_api_key_for_provider(
    self,
    provider: str,
    api_key: str,
    *,
    persist: bool = True,
    write_path: str = "",
    encrypt: bool = False,
    passphrase: str = "",
):
    """Persist an API key for a provider with optional Fernet encryption."""
    try:
        p = _normalize_provider_key_id(provider)
        if p not in {'gemini', 'openrouter', 'huggingface'}:
            return {'ok': False, 'error': f'unsupported_provider:{p or "missing"}'}
        if api_key is None:
            api_key = ''
        api_key = str(api_key).strip()
        encrypt = bool(encrypt)
        passphrase = str(passphrase or '').strip()

        # Determine path
        target = write_path or os.path.join(CONFIG_DIR, 'Comm-SCI-API-Keys.json')
        _storage = None
        try:
            _storage = getattr(self, 'storage_service', None)
        except Exception:
            _storage = None
        if _storage is None and _StorageService is not None:
            try:
                _storage = _StorageService()
            except Exception:
                _storage = None
        os.makedirs(os.path.dirname(target), exist_ok=True)

        data = {}
        if os.path.exists(target):
            try:
                if _storage is not None and hasattr(_storage, 'read_json'):
                    data = _storage.read_json(target) or {}
                else:
                    with open(target, 'r', encoding='utf-8') as f:
                        data = json.load(f) or {}
            except Exception:
                data = {}

        providers = data.get('providers')
        if not isinstance(providers, dict):
            providers = {}
            data['providers'] = providers

        entry = providers.get(p)
        if not isinstance(entry, dict):
            entry = {}
            providers[p] = entry

        if encrypt:
            if not api_key:
                return {'ok': False, 'error': 'api_key_missing'}
            if not passphrase:
                return {'ok': False, 'error': 'passphrase_missing'}
            enc_b64, salt_b64 = _try_encrypt_api_key(api_key, passphrase=passphrase)
            if not enc_b64 or not salt_b64:
                return {'ok': False, 'error': 'encrypt_failed_or_cryptography_missing'}
            entry['api_key_plain'] = ''
            entry['api_key_enc'] = enc_b64
            entry['api_key_salt'] = salt_b64
            entry['enc_scheme'] = 'fernet'
            try:
                os.environ['COMM_SCI_KEY_PASSPHRASE'] = passphrase
            except Exception:
                pass
        else:
            entry['api_key_plain'] = api_key
            entry.pop('api_key_enc', None)
            entry.pop('api_key_salt', None)
            entry.pop('enc_scheme', None)

        # Keep current process usable without restart.
        for env_name in _provider_key_env_names(p):
            try:
                if api_key:
                    os.environ[env_name] = api_key
                else:
                    os.environ.pop(env_name, None)
            except Exception:
                pass

        if persist:
            if _storage is not None and hasattr(_storage, 'write_json'):
                ok = bool(_storage.write_json(target, data, indent=2, ensure_ascii=False))
                if not ok:
                    with open(target, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(target, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # Recreate provider session best-effort so key changes are effective immediately.
        try:
            rf = getattr(self, '_recreate_chat_session', None)
            if callable(rf):
                try:
                    with_gov = bool(getattr(self, 'session_with_governance', True))
                except Exception:
                    with_gov = True
                try:
                    rf(with_governance=with_gov, reason='api_key_update')
                except TypeError:
                    rf(with_gov)
        except Exception:
            pass

        return {'ok': True, 'path': target, 'provider': p, 'encrypted': bool(encrypt)}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


def delete_api_key_for_provider(self, provider: str, *, persist: bool = True, write_path: str = ""):
    """Delete key material for one provider from key file and process env."""
    try:
        p = _normalize_provider_key_id(provider)
        if p not in {'gemini', 'openrouter', 'huggingface'}:
            return {'ok': False, 'error': f'unsupported_provider:{p or "missing"}'}

        target = write_path or os.path.join(CONFIG_DIR, 'Comm-SCI-API-Keys.json')
        _storage = None
        try:
            _storage = getattr(self, 'storage_service', None)
        except Exception:
            _storage = None
        if _storage is None and _StorageService is not None:
            try:
                _storage = _StorageService()
            except Exception:
                _storage = None

        data = {}
        if os.path.exists(target):
            try:
                if _storage is not None and hasattr(_storage, 'read_json'):
                    data = _storage.read_json(target) or {}
                else:
                    with open(target, 'r', encoding='utf-8') as f:
                        data = json.load(f) or {}
            except Exception:
                data = {}

        providers = data.get('providers')
        if not isinstance(providers, dict):
            providers = {}
            data['providers'] = providers

        entry = providers.get(p)
        if isinstance(entry, dict):
            for k in ('api_key_plain', 'api_key', 'api_key_enc', 'api_key_salt', 'enc_scheme'):
                entry.pop(k, None)
            if not entry:
                providers.pop(p, None)

        for env_name in _provider_key_env_names(p):
            try:
                os.environ.pop(env_name, None)
            except Exception:
                pass

        if persist:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if _storage is not None and hasattr(_storage, 'write_json'):
                ok = bool(_storage.write_json(target, data, indent=2, ensure_ascii=False))
                if not ok:
                    with open(target, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(target, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        try:
            rf = getattr(self, '_recreate_chat_session', None)
            if callable(rf):
                try:
                    with_gov = bool(getattr(self, 'session_with_governance', True))
                except Exception:
                    with_gov = True
                try:
                    rf(with_governance=with_gov, reason='api_key_delete')
                except TypeError:
                    rf(with_gov)
        except Exception:
            pass

        return {'ok': True, 'path': target, 'provider': p}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}

def load_log_from_path(self, path: str, *, fork: bool = False):
    """B7: Load a legacy chat log JSON from disk into history.
    If fork=True, create a fresh session_id/session_start and keep loaded history.
    """
    try:
        p = str(path)
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f) or {}

        hist = data.get('history') or data.get('conversation') or []
        if not isinstance(hist, list):
            return {'ok': False, 'error': 'history_not_list'}

        # Normalize roles (legacy might use 'assistant' instead of 'bot')
        norm = []
        for msg in hist:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            if role == 'assistant':
                role = 'bot'
            if role == 'system':
                role = 'system'
            norm.append({**msg, 'role': role})

        self.history = norm

        if fork:
            try:
                import uuid as _uuid
                self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + _uuid.uuid4().hex[:6]
                self.trace_id = self.session_id
                self.session_start_dt = datetime.now()
                # Reset counters/events (best-effort)
                self.session_requests = 0
                self.session_rate_limit_hits = 0
                self.session_repair_passes = 0
                self.session_csc_applied_count = 0
                self.session_guard_hits = 0
                self.session_events = []
                # Fork metadata (exported in chat logs; no secrets)
                try:
                    self.forked_from_log_path = p
                except Exception:
                    pass
                try:
                    sm = data.get('session_metadata') or {}
                    parent_trace_id = (
                        data.get('trace_id')
                        or sm.get('trace_id')
                        or data.get('session_id')
                        or sm.get('session_id')
                        or data.get('meta_trace_id')
                        or None
                    )
                    if not parent_trace_id:
                        # Legacy logs might not carry trace_id/session_id.
                        # Use a deterministic hash so forks from the same legacy log remain stable/auditable.
                        try:
                            blob = json.dumps(
                                {
                                    'session_metadata': sm,
                                    'provider_model_history': data.get('provider_model_history', []),
                                    'history': data.get('history', []),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        except Exception:
                            blob = repr(data)
                        h = hashlib.sha256(blob.encode('utf-8', errors='ignore')).hexdigest()[:12]
                        parent_trace_id = f"legacy_{h}"
                    self.fork_parent_trace_id = parent_trace_id
                except Exception:
                    self.fork_parent_trace_id = None
                try:
                    import os as _os
                    msg = f"Forked from chat log: {_os.path.basename(p)}"
                    self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
                except Exception:
                    pass
            except Exception:
                pass

        return {'ok': True, 'history_len': len(self.history), 'forked': bool(fork)}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    def get_ui(self):
        data = gov.get_ui_data()
        # Enrich with provider/model lists for panel dropdowns
        try:
            pr = getattr(self, 'provider_router', None)
            curp = 'gemini'
            if pr is not None and hasattr(pr, 'get_active_provider'):
                curp = (pr.get_active_provider() or 'gemini').strip().lower()
            else:
                curp = (getattr(cfg, 'get_active_provider', lambda: 'gemini')() or 'gemini').strip().lower()
        except Exception:
            curp = 'gemini'
        try:
            data['current_provider'] = curp
            data['providers'] = ['gemini', 'openrouter', 'huggingface']
            data['model_hint'] = ''
        except Exception:
            pass

        # Determine model for current provider
        try:
            cm = ''
            if hasattr(cfg, 'get_provider_model'):
                cm = (cfg.get_provider_model(curp) or '').strip()
            if not cm:
                cm = (cfg_get_model() or '').strip()
            data['current_model'] = cm
        except Exception:
            pass
        # Available models list (must stay fast and local: no network here)
        try:
            models = []
            if curp == 'gemini':
                # Use warmed in-memory cache only; avoid live fetch in get_ui().
                models = self.get_available_models(curp)
            elif curp == 'openrouter':
                # Use warmed in-memory cache only; avoid live fetch in get_ui().
                models = self.get_available_models(curp)
            elif curp == 'huggingface':
                pr = getattr(self, 'provider_router', None)
                models = []
                # UI controls for HF catalog
                data['hf_provider_filter_options'] = ['all', 'zai-org', 'novita', 'cerebras', 'together', 'groq', 'fireworks', 'sambanova', 'hyperbolic', 'hf-inference']
                data['hf_catalog_default_top_n'] = int(getattr(self, 'hf_catalog_top_n', 200) or 200)
                data['hf_catalog_default_provider_filter'] = (getattr(self, 'hf_catalog_provider_filter', 'all') or 'all')

                # 1) Use warmed in-memory cache only; avoid live fetch in get_ui().
                models = self.get_available_models(curp)

                # 2) Fallback: configured HF models list (local config/key file)
                if not models:
                    try:
                        if pr is not None and hasattr(pr, 'get_huggingface_models_from_config'):
                            models = pr.get_huggingface_models_from_config() or []
                            data['huggingface_models_meta'] = {'source': 'config', 'count': len(models)}
                    except Exception:
                        models = []

                if not models:
                    models = ['zai-org/GLM-4.7:cerebras']
                    data['model_hint'] = ("Hugging Face: keine Modellliste konfiguriert oder abrufbar. "
                                          "Nutze 'HF Catalog (Top N)' oder trage unter providers.huggingface.models "
                                          "in Comm-SCI-API-Keys.json deine Wunschmodelle ein.")
            data['available_models'] = models
        except Exception:
            data['available_models'] = []

        return data
    
    def remote_cmd(self, cmd):
        """Inject a command into the main UI input and trigger send() via JS."""
        if not getattr(self, 'main_win', None):
            return {'ok': False, 'error': 'no_main_win'}
        try:
            cmd_s = str(cmd or '')
            if self._is_qc_override_modal_active():
                if cmd_s.strip() == "QC Override":
                    self._bring_qc_override_to_front()
                    return {'ok': True, 'already_open': True}
                self._bring_qc_override_to_front()
                try:
                    self.log_event('qc_override_modal_block', {'source': 'remote_cmd', 'cmd': cmd_s})
                except Exception:
                    pass
                return {'ok': False, 'error': 'qc_override_modal_blocked'}
            return {'ok': bool(self._ui_remote_input(cmd_s))}
        except Exception as e:
            return {'ok': False, 'error': str(e)}



# ----------------------------

# ----------------------------
# Deterministic command helpers (EN-only)
# These are kept at module scope and then bound into Api via the fixup loop.
# ----------------------------


def _control_layer_alert_html(
    message: str,
    *,
    title: str = "CONTROL LAYER ALERT",
    severity: str = "error",
    lang: str = "de",
) -> str:
    """Render a human-friendly Control-Layer box for UI (HTML).
    - No raw JSON blobs in the chat UI.
    - Keep logs complete by returning deterministic HTML.
    """
    try:
        msg = (message or "").strip()
    except Exception:
        msg = str(message)

    # Optional safe action-hints (rendered as non-clickable "button" labels)
    _action_switch_free = False
    try:
        if "[[ACTION:SWITCH_FREE_MODEL]]" in msg:
            _action_switch_free = True
            msg = msg.replace("[[ACTION:SWITCH_FREE_MODEL]]", "").strip()
    except Exception:
        _action_switch_free = False
    safe = html.escape(msg)
    safe = safe.replace("\n", "<br>")
    # Use existing .csc-warning styling, but tint for errors.
    style = ""
    if str(severity).lower() == "error":
        style = "border: 1px solid #c00; background: #fee; color: #600;"
    elif str(severity).lower() == "warn":
        style = "border: 1px solid #f9ab00; background: #fff7e0; color: #3c2b00;"
    else:
        style = "border: 1px solid #999; background: #f5f5f5; color: #222;"
    action_html = ""
    if _action_switch_free:
        # Clickable UI action (handled by JS event delegation)
        action_html = (
            "<br><br>"
            "<a href=\"#\" class=\"ctl-action action-next-free\" "
            "style=\"display:inline-block;padding:2px 8px;border:1px solid #888;"
            "border-radius:10px;background:#eee;font-family:monospace;text-decoration:none;color:inherit;\">"
            "Tip: choose another :free model</a>"
        )

    try:
        t = html.escape(str(title or "CONTROL LAYER ALERT"))
    except Exception:
        t = "CONTROL LAYER ALERT"
    try:
        tip = html.escape(_control_layer_tooltip_text(lang=lang, severity=severity), quote=True)
    except Exception:
        tip = ""
    tip_attr = f" data-u-title='{tip}' style='cursor:help;'" if tip else ""
    return (
        f"<details class='csc-warning' open style='{style}'{tip_attr}>"
        f"<summary{tip_attr}>⚠️ {t}</summary>"
        f"<div class='csc-details'>{safe}{action_html}</div>"
        f"</details>"
    )


def _detect_probable_truncation(raw_text: str, rendered_html: str = "") -> tuple[bool, str]:
    """Best-effort detector for provider/model outputs that likely ended prematurely.

    Goal: warn the user without changing the answer text or inventing content.
    """
    try:
        raw = str(raw_text or "").strip()
        if not raw:
            return False, ""

        plain = str(raw)
        # Strong signals: abrupt cut inside a word or known SCI step likely cut.
        tail = plain[-220:]
        # Ends with an alnum fragment and no sentence/list terminator.
        if re.search(r"[A-Za-zÄÖÜäöüß]{2,}$", tail) and not re.search(r"[.!?…:;\]\)\"']\s*$", plain):
            return True, "Antwort wirkt unvollständig (endet vermutlich mitten im Satz/Wort)."

        # SCI-specific abrupt endings like \"Dialectic_*: ... subjektives Er\"
        if ("SCI Trace" in str(rendered_html or "") or "Dialectic_" in plain) and re.search(r"(?:Dialectic_|Plan:|Solution:|Critic:|Check:).{0,200}[A-Za-zÄÖÜäöüß]{2,}$", tail):
            return True, "SCI-Trace wirkt unvollständig (mindestens ein Schritt endet abrupt)."

        return False, ""
    except Exception:
        return False, ""


def _render_error_html(self, context: str, err: Exception) -> str:
    """Never crash the UI on renderer errors; show a small deterministic error box."""
    ctx = html.escape(str(context or "renderer"))
    msg = html.escape(f"{type(err).__name__}: {err}")
    return (
        f'<div class="comm-help comm-error">'
        f'<b>Error</b> <span style="opacity:.8">[{ctx}]</span><br>'
        f'<code>{msg}</code>'
        f'</div>'
    )


def _safe_html(self, context: str, fn):
    """Run a renderer safely; return error HTML on any exception."""
    try:
        return fn()
    except Exception as e:
        try:
            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            if gov_obj and hasattr(gov_obj, 'log'):
                gov_obj.log(f"[UI] Renderer failed ({context}): {type(e).__name__}: {e}")
        except Exception:
            pass
        try:
            return _render_error_html(self, context, e)
        except Exception:
            # last resort: plain div
            lang = "de"
            try:
                lang = str(getattr(self, "_answer_lang", lambda: "de")() or "de")
            except Exception:
                lang = "de"
            return _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error', lang=lang)


def _build_profile_switch_audit_line(command: str, from_profile: str, to_profile: str) -> str:
    return (
        f"Profile-Switch-Audit: command={command} · from={from_profile} · "
        f"to={to_profile} · rule=explicit-standalone-only"
    )


def _handle_command_deterministic(self, canonical_cmd: str, timestamp: str):
    """Central deterministic command router (no LLM calls).

    Returns a dict (e.g. {html, csc, t_in, t_out, total_in, total_out}) if handled, else None.
    """
    cmd = (canonical_cmd or "").strip()
    # QC Override (wrapper-local UI dialog)
    if cmd == "QC Override":
        try:
            self.show_qc_override()
        except Exception:
            pass
        return {"html": "<div class='sys'>QC Override dialog opened.</div>", "csc": None}
    if not cmd:
        return None

    # Comm Audit: deterministic audit export + lightweight compliance scan (no LLM)
    if cmd in ("Comm Audit", "Comm Audi"):
        # tolerate common typo: "Comm Audi"
        ts_iso = datetime.now().isoformat()
        audit_event = {
            "event": "comm_audit_called",
            "ts": ts_iso,
            "n_last": 25,
            "profile": getattr(getattr(self, 'gov_state', None), 'active_profile', '') or '',
            "overlay": getattr(getattr(self, 'gov_state', None), 'overlay', '') or '',
            "sci": {
                "pending": bool(getattr(getattr(self, 'gov_state', None), 'sci_pending', False)),
                "variant": getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '',
                "active": bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False)),
            },
        }

        chat_path, audit_path = (None, None)

        # Include last provider call info (best-effort; does not trigger LLM)
        try:
            audit_event['last_call'] = getattr(self, 'last_call_info', {}) or {}
        except Exception:
            pass

        try:
            # ENONLY requirement: Comm Audit should export audit only.
            chat_path, audit_path = self.export(audit_event=audit_event, audit_only=True)
        except Exception:
            try:
                try:
                    if hasattr(self, 'export_audit_v2'):
                        self.export_audit_v2(audit_event=audit_event, audit_only=True)
                    else:
                        self.export(audit_event=audit_event, audit_only=True)
                except Exception:
                    try:
                        self.export(audit_event=audit_event)
                    except Exception:
                        pass
            except Exception:
                pass

        # --- Deterministic compliance scan of last N bot answers (best-effort) ---
        n = 5
        try:
            n = int((getattr(gov, 'data', {}) or {}).get('global_defaults', {}).get('comm_audit', {}).get('window', 5) or 5)
        except Exception:
            n = 5

        bot_msgs = []
        try:
            bot_msgs = [h for h in (getattr(self, 'history', []) or []) if (h or {}).get('role') == 'bot']
        except Exception:
            bot_msgs = []

        sample = bot_msgs[-n:] if n > 0 else []
        # Build a best-effort mapping from each bot message to the immediately preceding user input.
        # This allows the scan to reliably classify command responses without depending on model output text.
        rows = []
        if _compliance_scan is not None:
            try:
                _hist = (getattr(self, 'history', []) or [])
                rows = _compliance_scan.scan_message_compliance_best_effort(
                    sample=sample,
                    history=_hist,
                    build_route_ctx=_build_route_ctx,
                    api=self,
                    gov=gov,
                )
            except Exception:
                rows = []
        if not rows:
            prev_user_by_bot_id = {}
            try:
                _hist = (getattr(self, 'history', []) or [])
                for _ix, _m in enumerate(_hist):
                    if (_m or {}).get('role') != 'bot':
                        continue
                    _prev_user = ''
                    for _jx in range(_ix - 1, -1, -1):
                        _pm = _hist[_jx] or {}
                        if (_pm.get('role') or '') in ('user', 'human'):
                            _prev_user = str(_pm.get('content', '') or '')
                            break
                    prev_user_by_bot_id[id(_m)] = _prev_user
            except Exception:
                prev_user_by_bot_id = {}


            rows = []
            for i, msg_obj in enumerate(sample, 1):
                txt = (msg_obj or {}).get('content', '') or ''
                vios = []

                # Command responses are deterministic UI/control outputs; they must not be
                # evaluated against Self-Debunking / QC footer / SCI Trace contracts.
                # Classification is based primarily on the preceding user message (Comm ...),
                # and only secondarily on the assistant's first line.
                try:
                    prev_user = (prev_user_by_bot_id.get(id(msg_obj), '') or '').lstrip()
                    prev_user_first = ''
                    for _ln in str(prev_user).splitlines():
                        if _ln.strip():
                            prev_user_first = _ln.strip()
                            break
                    first_line = ''
                    for _ln in str(txt).splitlines():
                        if _ln.strip():
                            first_line = _ln.strip()
                            break
                    ctx_turn = _build_route_ctx(self, user_raw=prev_user_first, is_command=prev_user_first.startswith('Comm '))
                    is_cmd_msg = bool(ctx_turn.get('is_command'))
                    if not is_cmd_msg:
                        is_cmd_msg = bool(re.match(r"^(?:Command executed:\s+)?(?:Comm\s+\w+|Profile\s+\w+|SCI\s+(?:on|off|menu)|QC\s+Override)\b", first_line))
                except Exception:
                    is_cmd_msg = False

                # QC footer present? (skip for command responses)
                if (not is_cmd_msg) and ('QC-Matrix:' not in txt and 'QC:' not in txt):
                    vios.append('Missing QC footer')

                # Self-Debunking required? (skip for command responses)
                try:
                    prof_now = getattr(getattr(self, 'gov_state', None), 'active_profile', '') or 'Standard'
                    if not is_cmd_msg:
                        sd_msg = gov.check_self_debunking(txt, prof_now)
                        if sd_msg:
                            vios.append(sd_msg)
                except Exception:
                    pass

                # Verification Route Gate
                try:
                    vr_msg = gov.check_verification_route_gate(txt)
                    if vr_msg:
                        vios.append(vr_msg)
                except Exception:
                    pass

                # SCI Trace contract (if a variant is active)
                try:
                    vk = getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or ''
                    if vk and not is_cmd_msg:
                        if 'SCI Trace' not in txt:
                            vios.append('Missing SCI Trace block')
                except Exception:
                    pass

                status = '✓ Compliant' if not vios else '⚠ ' + '; '.join(vios)
                rows.append((i, status))

        # Render
        msg = "Audit exported."

        # Module dependency warnings (Stage 3d)
        dep_warn_html = ""
        try:
            ds = getattr(self, 'deps_status', {}) or {}
            missing = [k for k in ('auditstream','rendering_utils','compliance_scan') if not (ds.get(k, (False,''))[0])]
            if missing:
                dep_warn_html = (
                    "<div style='margin-top:8px; padding:8px; border:1px solid #d77; background:#fff3f3; border-radius:8px;'>"
                    "<b>Warning:</b> Optional modules missing; fallback paths active: "
                    + html.escape(", ".join(missing))
                    + "</div>"
                )
        except Exception:
            dep_warn_html = ""

        detail = ""
        rel_audit = ""
        if audit_path:
            # Prefer a stable relative path for UI/logs; never require it.
            try:
                rel_audit = os.path.relpath(audit_path, start=os.getcwd())
                if str(rel_audit).startswith(".."):
                    rel_audit = os.path.join("Logs", "Audit", os.path.basename(audit_path))
            except Exception:
                rel_audit = os.path.basename(audit_path)
            detail = f"<br><code>{html.escape(str(rel_audit))}</code>"

        # Last call debug line (best-effort)
        last_line = ""
        try:
            lc = getattr(self, 'last_call_info', {}) or {}
            prov = (lc.get('provider') or '').strip()
            modl = (lc.get('model') or '').strip()
            ms = int(lc.get('ms') or 0)
            usage = lc.get('usage') or {}
            # normalize common usage keys
            u_in = usage.get('prompt_tokens', usage.get('input_tokens', usage.get('input', usage.get('in', 0))))
            u_out = usage.get('completion_tokens', usage.get('output_tokens', usage.get('output', usage.get('out', 0))))
            if prov or modl or ms or usage:
                last_line = (
                    "<div style='margin-top:6px; font-size:12px; color:#444;'>"
                    + f"Last call: <code>{html.escape(prov or 'n/a')}</code> · <code>{html.escape(modl or 'n/a')}</code> · {ms} ms"
                    + (f" · usage in/out: {html.escape(str(u_in))}/{html.escape(str(u_out))}" if (u_in or u_out) else "")
                    + "</div>"
                )
        except Exception:
            last_line = ""

        tbl = ""
        if rows:
            tr = "".join([
                "<tr>"
                f"<td style='padding:6px 8px; border-bottom:1px solid #ddd; width:80px;'>#{idx}</td>"
                f"<td style='padding:6px 8px; border-bottom:1px solid #ddd;'>{html.escape(st)}</td>"
                "</tr>" for idx, st in rows
            ])
            tbl = (
                "<div style='margin-top:8px;'>"
                "<table style='width:100%; border-collapse:collapse; font-size:13px;'>"
                "<thead><tr>"
                "<th style='text-align:left; padding:6px 8px; border-bottom:2px solid #bbb;'>Message</th>"
                "<th style='text-align:left; padding:6px 8px; border-bottom:2px solid #bbb;'>Compliance scan (best-effort)</th>"
                "</tr></thead>"
                f"<tbody>{tr}</tbody></table></div>"
            )

        html_content = (
            "<div style='border:1px solid #bbb; background:#f7f7f7; padding:10px; border-radius:10px; margin:8px 0;'>"
            f"<b>Comm Audit</b><br>{msg}{detail}{dep_warn_html}{last_line}{tbl}</div>"
        )
        html_content += f'<div class="ts-footer">Response at {html.escape(str(timestamp))}</div>'

        try:
            bot_txt = "Comm Audit"
            try:
                if rel_audit:
                    bot_txt += f"\nExportiert (Audit): {rel_audit}"
            except Exception:
                pass
            self.history.append({"role": "bot", "content": bot_txt, "ts": datetime.now().isoformat(), "csc": None})
        except Exception:
            pass

        return {
            "html": html_content,
            "t_in": 0,
            "t_out": 0,
            "total_in": getattr(self, "session_tokens_in", 0),
            "total_out": getattr(self, "session_tokens_out", 0),
            "csc": None,
        }

    # Renderer lookup (deterministic)
    renderer_map = {
        "Comm Help": ("Comm Help", getattr(self, "_render_comm_help", lambda: "Comm Help"), getattr(self, "_render_comm_help_html", lambda: "")),
        "Comm State": ("Comm State", getattr(self, "_render_comm_state", lambda: "Comm State"), getattr(self, "_render_comm_state_html", lambda: "")),
        "Comm Config": ("Comm Config", getattr(self, "_render_comm_config", lambda: "Comm Config"), getattr(self, "_render_comm_config_html", lambda: "")),
        "Comm Anchor": ("Comm Anchor", (lambda: "Comm Anchor"), getattr(self, "_render_anchor_snapshot_html", lambda: "")),
    }

    if cmd in renderer_map:
        label, raw_fn, html_fn = renderer_map[cmd]

        try:
            self.gov_state.user_turns += 1
        except Exception:
            pass

        try:
            raw_text = raw_fn() or label
        except Exception:
            raw_text = label

        html_content = _safe_html(self, label, html_fn)
        html_content += f'<div class="ts-footer">Response at {html.escape(str(timestamp))}</div>'

        try:
            self.history.append({"role": "bot", "content": raw_text, "ts": datetime.now().isoformat(), "csc": None})
        except Exception:
            pass

        return {"html": html_content, "csc": None}


    # SCI menu trigger (explicit only)
    if cmd in ("SCI on", "SCI menu"):
        try:
            self.gov_state.sci_variant = ""
            self.gov_state.sci_pending = True
            self.gov_state.sci_active = False
        except Exception:
            pass

        try:
            self.gov_state.user_turns += 1
        except Exception:
            pass

        html_content = _safe_html(self, "SCI menu", getattr(self, "_render_sci_menu_html", lambda: ""))
        html_content += f'<div class="ts-footer">Response at {html.escape(str(timestamp))}</div>'

        try:
            self.history.append({"role": "bot", "content": _safe_preview_text(re.sub(r"<[^>]+>", "", html_content), 2000) or "SCI menu", "ts": datetime.now().isoformat(), "csc": None})
        except Exception:
            pass

        return {"html": html_content, "csc": None}

    return None

# Api bridge (pywebview js_api)
# NOTE: In this ENONLY build, Api is the concrete js_api object.


# ----------------------------
# PROVIDER ADAPTERS (single-file)
# ----------------------------


def _openrouter_friendly_http_error(
    status_code: int,
    raw_body: str,
    *,
    lang: str = "de",
    tz: str = "Europe/Berlin",
    provider_label: str = "OpenRouter",
) -> str:
    """Translate common OpenRouter HTTP errors into human-friendly messages.

    Notes:
    - Does NOT expose user_id or other sensitive fields.
    - Keeps a short technical tail for debugging.
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    try:
        from zoneinfo import ZoneInfo as _ZoneInfo
    except Exception:  # pragma: no cover
        _ZoneInfo = None

    lang = (lang or "de").strip().lower()
    if lang not in ("de", "en"):
        lang = "de"

    body = (raw_body or "").strip()
    obj = None
    try:
        obj = _json.loads(body) if body else None
    except Exception:
        obj = None

    err = {}
    msg = ""
    ecode = status_code
    meta_hdr = {}

    if isinstance(obj, dict):
        e = obj.get("error")
        if isinstance(e, dict):
            err = e
            msg = (e.get("message") or "").strip()
            try:
                ecode = int(e.get("code") or status_code)
            except Exception:
                ecode = status_code
            md = e.get("metadata")
            if isinstance(md, dict):
                hdr = md.get("headers")
                if isinstance(hdr, dict):
                    meta_hdr = hdr

    # Rate-limit helpers
    lim = meta_hdr.get("X-RateLimit-Limit")
    rem = meta_hdr.get("X-RateLimit-Remaining")
    reset_ms = meta_hdr.get("X-RateLimit-Reset")

    reset_str = None
    try:
        if reset_ms is not None:
            ts = int(reset_ms) / 1000.0
            dt = _dt.fromtimestamp(ts, tz=_tz.utc)
            if _ZoneInfo is not None:
                dt = dt.astimezone(_ZoneInfo(tz))
            reset_str = dt.strftime("%d.%m.%Y, %H:%M Uhr")
            try:
                if _ZoneInfo is not None:
                    now_dt = _dt.now(_ZoneInfo(tz))
                else:
                    now_dt = _dt.now(_tz.utc)
                delta_s = int((dt - now_dt).total_seconds())
                if delta_s > 0:
                    mins = (delta_s + 59) // 60
                    h = mins // 60
                    m2 = mins % 60
                    if h > 0:
                        reset_in_str = (f"{h}h {m2}m" if lang != "en" else f"{h}h {m2}m")
                    else:
                        reset_in_str = (f"{m2}m" if lang != "en" else f"{m2}m")
            except Exception:
                reset_in_str = None
    except Exception:
        reset_str = None
        reset_in_str = None
    def _fmt_quota():
        parts = []
        if lim is not None and rem is not None:
            try:
                lim_i = int(lim); rem_i = int(rem)
                used_i = max(0, lim_i - rem_i)
                if lang == "en":
                    parts.append(f"Today: {used_i}/{lim_i} used ({rem_i} remaining).")
                else:
                    parts.append(f"Heute: {used_i}/{lim_i} verbraucht (noch {rem_i}).")
            except Exception:
                pass
        if reset_str:
            if lang == "en":
                parts.append(f"Resets: {reset_str}." + (f" (in {reset_in_str})" if reset_in_str else ""))
            else:
                parts.append(f"Nächster Reset: {reset_str}." + (f" (in {reset_in_str})" if reset_in_str else ""))
        return " ".join(parts).strip()

    # Human-friendly mapping
    lower = (msg or "").lower()

    pl = (provider_label or "OpenRouter").strip()

    if int(ecode) == 429:
        quota = _fmt_quota()
        if "free-models-per-day" in lower:
            if lang == "en":
                head = f"{pl} limit reached (free models per day)."
                tail = "Options: wait for reset, use a paid model/provider, or add credits."
            else:
                head = f"{pl}-Limit erreicht (Free-Modelle pro Tag)."
                tail = "Optionen: bis zum Reset warten, anderes Modell/Provider nutzen oder Credits hinzufügen."
            parts = [head]
            if quota:
                parts.append(quota)
            return " ".join(parts + [tail]).strip() + " [[ACTION:SWITCH_FREE_MODEL]]" + f" [HTTP 429]"
        else:
            if lang == "en":
                head = f"{pl} rate limit reached."
                tail = "Options: wait briefly and retry, or switch model/provider."
            else:
                head = f"{pl}-Rate-Limit erreicht."
                tail = "Optionen: kurz warten und erneut versuchen oder Modell/Provider wechseln."
            parts = [head]
            if quota:
                parts.append(quota)
            if msg:
                parts.append(msg)
            return " ".join(parts + [tail]).strip() + " [[ACTION:SWITCH_FREE_MODEL]]" + f" [HTTP 429]"

    if int(ecode) == 404 and ("privacy" in lower or "data policy" in lower or "no endpoints found" in lower):
        if lang == "en":
            return (f"{pl} cannot route your request because your Privacy/Data-Policy settings exclude all endpoints "
                    "for this model. Check OpenRouter → Settings → Privacy (and any provider restrictions). "
                    f"[HTTP {status_code}]")
        return (f"{pl} kann nicht routen, weil deine Privacy/Data-Policy-Einstellungen alle passenden Endpoints "
                "für dieses Modell ausschließen. Prüfe OpenRouter → Settings → Privacy (und ggf. Provider-Restrictions). "
                f"[HTTP {status_code}]")

    if int(ecode) == 402 or "insufficient credits" in lower or "add credits" in lower:
        if lang == "en":
            return (f"{pl}: insufficient credits for this request. Add credits or choose a free/eligible model. "
                    f"[HTTP {status_code}]")
        return (f"{pl}: Nicht genügend Guthaben für diese Anfrage. Guthaben hinzufügen oder ein passendes "
                "choose a (possibly free) model. "
                f"[HTTP {status_code}]")

    if int(ecode) in (401, 403):
        if lang == "en":
            return (f"{pl} authentication/permission error. Check your API key and account settings. "
                    f"[HTTP {status_code}]")
        return (f"{pl}: auth/permission error. Check API key and account/privacy settings. "
                f"[HTTP {status_code}]")

    # Fallback
    if msg:
        if lang == "en":
            return f"{pl} error: {msg} [HTTP {status_code}]"
        return f"{pl} error: {msg} [HTTP {status_code}]"
    if lang == "en":
        return f"{pl} request failed. [HTTP {status_code}]"
    return f"{pl}-Anfrage fehlgeschlagen. [HTTP {status_code}]"

class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client (used for OpenRouter).

    - No external deps (urllib).
    - Returns (text, usage_dict).
    """

    def __init__(self, *, base_url: str, api_key: str, app_referrer: str = '', app_title: str = '', timeout_s: int = 60):
        self.base_url = (base_url or '').rstrip('/')
        self.api_key = (api_key or '').strip()
        self.app_referrer = (app_referrer or '').strip()
        self.app_title = (app_title or '').strip()
        self.timeout_s = int(timeout_s or 60)
        self.max_retries = 2

    def chat(self, *, messages, model: str, temperature: float = 0.2, max_tokens: int = 1024, lang: str = 'de'):
        import json as _json
        import urllib.request as _urlreq
        import urllib.error as _urlerr
        import time as _time

        if not self.base_url:
            raise RuntimeError('OpenAICompatibleClient: base_url is empty')
        if not self.api_key:
            raise RuntimeError('OpenAICompatibleClient: api_key is missing')
        if not model:
            raise RuntimeError('OpenAICompatibleClient: model is empty')

        url = self.base_url + '/chat/completions'
        payload = {
            'model': model,
            'messages': messages,
            'temperature': float(temperature or 0.0),
            'max_tokens': int(max_tokens or 0) if max_tokens is not None else 1024,
        }

        data = _json.dumps(payload).encode('utf-8')
        req = _urlreq.Request(url, data=data, method='POST')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        req.add_header('Content-Type', 'application/json')
        if self.app_referrer:
            req.add_header('HTTP-Referer', self.app_referrer)
        if self.app_title:
            req.add_header('X-Title', self.app_title)

        raw = ''
        maxr = int(getattr(self, 'max_retries', 2) or 2)
        for attempt in range(maxr + 1):
            try:
                with _urlreq.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode('utf-8', errors='replace')
                break
            except _urlerr.HTTPError as e:
                code = getattr(e, 'code', None)
                try:
                    raw_err = e.read().decode('utf-8', errors='replace')
                except Exception:
                    raw_err = str(e)
                # Retry transient upstream failures / rate limits.
                if code in (429, 500, 502, 503, 504) and attempt < maxr:
                    try:
                        delay = [0.25, 1.0, 3.0][min(attempt, 2)]
                        _time.sleep(delay)
                    except Exception:
                        pass
                    continue
                _pl = "Hugging Face" if ("huggingface" in str(self.base_url or "").lower()) else "OpenRouter"
                raise RuntimeError(_openrouter_friendly_http_error(int(code or 0), raw_err, lang=lang, provider_label=_pl))
            except Exception as e:
                if attempt < maxr:
                    try:
                        delay = [0.25, 1.0, 3.0][min(attempt, 2)]
                        _time.sleep(delay)
                    except Exception:
                        pass
                    continue
                raise RuntimeError(f'OpenAICompatibleClient error: {e}')

        obj = {}
        try:
            obj = _json.loads(raw)
        except Exception:
            obj = {}

        # OpenRouter can return HTTP 200 while embedding an error in the body.
        # Detect and raise so the UI doesn't silently show an empty answer.
        try:
            err = obj.get('error') if isinstance(obj, dict) else None
            if err:
                # Expected shape: { error: { code:number, message:str, metadata?:... } }
                code = ''
                msg = ''
                meta = ''
                if isinstance(err, dict):
                    code = str(err.get('code') or '')
                    msg = str(err.get('message') or '')
                    try:
                        meta_obj = err.get('metadata')
                        if meta_obj is not None:
                            meta = _json.dumps(meta_obj, ensure_ascii=False)
                    except Exception:
                        meta = ''
                else:
                    msg = str(err)
                details = f"{code} {msg}".strip()
                if meta:
                    details = details + f" :: {meta}"
                raise RuntimeError(f"OpenRouter API error: {details}")
        except RuntimeError:
            raise
        except Exception:
            pass

        # Text
        txt = ''
        try:
            choices = obj.get('choices') or []
            if choices and isinstance(choices, list):
                msg = (choices[0] or {}).get('message') or {}
                txt = (msg.get('content') or '')
        except Exception:
            txt = ''

        # If we still have no content, try to surface a useful error instead of returning empty.
        if not (txt or '').strip():
            try:
                # Some upstream errors are encoded as finish_reason="error" with a top-level error.
                # If that happened but we missed it, include the raw body in the exception.
                raise RuntimeError(f"OpenRouter empty completion (no content). Raw: {raw}")
            except RuntimeError:
                raise

        # Usage (best-effort)
        usage = {}
        try:
            usage = obj.get('usage') or {}
            if not isinstance(usage, dict):
                usage = {}
        except Exception:
            usage = {}

        return txt or '', usage


    def list_models(self, *, lang: str = 'de'):
        '''Fetch models list from /models (best-effort).

        Returns: (models, meta) where meta includes ts and raw counts.
        '''
        import json as _json
        import urllib.request as _urlreq
        import urllib.error as _urlerr
        if not self.base_url:
            raise RuntimeError('OpenAICompatibleClient: base_url is empty')
        if not self.api_key:
            raise RuntimeError('OpenAICompatibleClient: api_key is missing')
        url = self.base_url + '/models'
        req = _urlreq.Request(url, method='GET')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        req.add_header('Content-Type', 'application/json')
        if self.app_referrer:
            req.add_header('HTTP-Referer', self.app_referrer)
        if self.app_title:
            req.add_header('X-Title', self.app_title)
        try:
            with _urlreq.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
        except _urlerr.HTTPError as e:
            try:
                raw_err = e.read().decode('utf-8', errors='replace')
            except Exception:
                raw_err = str(e)
            _pl = "Hugging Face" if ("huggingface" in str(self.base_url or "").lower()) else "OpenRouter"
            raise RuntimeError(_openrouter_friendly_http_error(int(getattr(e,'code',0) or 0), raw_err, lang=lang, provider_label=_pl))
        except Exception as e:
            raise RuntimeError(f'OpenAICompatibleClient error: {e}')

        obj = {}
        try:
            obj = _json.loads(raw)
        except Exception:
            obj = {}

        models = []
        try:
            data = obj.get('data') or []
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict):
                        mid = (it.get('id') or '').strip()
                        if mid:
                            models.append(mid)
        except Exception:
            models = []

        # De-duplicate and sort for UI usability (case-insensitive).
        try:
            seen = set()
            uniq = []
            for m in models:
                k = (m or '').strip()
                if not k:
                    continue
                lk = k.lower()
                if lk in seen:
                    continue
                seen.add(lk)
                uniq.append(k)
            models = sorted(uniq, key=lambda s: s.lower())
        except Exception:
            pass

        meta = {'count': len(models)}
        return models, meta


class ProviderRouter:
    """Routes provider calls based on ConfigManager settings.

    Stage A (fix20): provider selection via config only (no UI).
    """

    def __init__(self, cfg_mgr):
        self.cfg = cfg_mgr

    def get_active_provider(self) -> str:
        try:
            p = (self.cfg.config or {}).get('active_provider', 'gemini')
            return (p or 'gemini').strip().lower()
        except Exception:
            return 'gemini'

    def get_provider_model(self, provider: str, fallback_model: str = '') -> str:
        try:
            provider = (provider or '').strip().lower() or 'gemini'
            provs = (self.cfg.config or {}).get('providers') or {}
            if isinstance(provs, dict):
                pconf = provs.get(provider) or {}
                if isinstance(pconf, dict):
                    m = (pconf.get('default_model') or '').strip()
                    if m:
                        return m
            # Back-compat: old single model key
            m2 = (self.cfg.config or {}).get('model', '')
            if provider == 'gemini' and isinstance(m2, str) and m2.strip():
                return m2.strip()
        except Exception:
            pass
        return (fallback_model or '').strip() or ''

    def build_openrouter_client(self):
        """Build an OpenRouter client.

        Key lookup order:
          1) ENV var from providers.openrouter.api_key_env (default OPENROUTER_API_KEY)
          2) Config/Comm-SCI-Config.json: providers.openrouter.api_key_plain / api_key_enc
          3) Key file (KEYS_PATH):
             - provider-structured: providers.openrouter.api_key_plain / api_key_enc
             - legacy: OPENROUTER_API_KEY field
        """
        try:
            provider = 'openrouter'
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get(provider) or {}) if isinstance(provs, dict) else {}
            base_url = (pconf.get('base_url') or 'https://openrouter.ai/api/v1').strip()
            key_env = (pconf.get('api_key_env') or 'OPENROUTER_API_KEY').strip()

            # 1) env
            key = ''
            try:
                key = (os.environ.get(key_env) or '').strip()
            except Exception:
                key = ''

            # 2) config plaintext
            if not key:
                key = (pconf.get('api_key_plain') or '').strip()

            # 2b) config encrypted
            if not key:
                enc = (pconf.get('api_key_enc') or '').strip()
                salt = (pconf.get('api_key_salt') or '').strip()
                scheme = (pconf.get('enc_scheme') or '').strip().lower()
                if enc and salt and (scheme in {'fernet', ''}):
                    passphrase = (os.environ.get('COMM_SCI_KEY_PASSPHRASE') or '').strip()
                    key = (_try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt) or '').strip()

            # 3) key file fallback (provider-structured or legacy)
            if not key and os.path.exists(KEYS_PATH):
                try:
                    data = _storage_read_json(KEYS_PATH) or {}
                    if isinstance(data, dict):
                        provs2 = data.get('providers')
                        if isinstance(provs2, dict):
                            o = provs2.get(provider) or {}
                            if isinstance(o, dict):
                                key = (o.get('api_key_plain') or o.get('api_key') or '').strip()
                                if not key:
                                    enc = (o.get('api_key_enc') or '').strip()
                                    salt = (o.get('api_key_salt') or '').strip()
                                    scheme = (o.get('enc_scheme') or '').strip().lower()
                                    if enc and salt and (scheme in {'fernet', ''}):
                                        passphrase = (os.environ.get('COMM_SCI_KEY_PASSPHRASE') or '').strip()
                                        key = (_try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt) or '').strip()
                        if not key:
                            key = (data.get('OPENROUTER_API_KEY') or '').strip()
                except Exception:
                    pass

            app_ref = (pconf.get('app_referrer') or '').strip()
            app_title = (pconf.get('app_title') or 'Comm-SCI Desktop').strip()
            return OpenAICompatibleClient(base_url=base_url, api_key=key, app_referrer=app_ref, app_title=app_title)
        except Exception:
            return None


    def build_huggingface_client(self):
        """Build an OpenAI-compatible client for Hugging Face (router.huggingface.co).

        Key lookup order:
          1) ENV var from providers.huggingface.api_key_env (default HF_TOKEN)
          2) Config providers.huggingface.api_key_plain / api_key_enc
          3) Key file (KEYS_PATH): providers.huggingface.api_key_plain / api_key_enc (or legacy HF_TOKEN fields)
        """
        try:
            provider = 'huggingface'
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get(provider) or {}) if isinstance(provs, dict) else {}
            base_url = (pconf.get('base_url') or 'https://router.huggingface.co/v1').strip()
            key_env = (pconf.get('api_key_env') or 'HF_TOKEN').strip()

            # 1) env
            key = ''
            try:
                key = (os.environ.get(key_env) or '').strip()
            except Exception:
                key = ''

            # 2) config plaintext
            if not key:
                key = (pconf.get('api_key_plain') or '').strip()

            # 2b) config encrypted
            if not key:
                enc = (pconf.get('api_key_enc') or '').strip()
                salt = (pconf.get('api_key_salt') or '').strip()
                scheme = (pconf.get('enc_scheme') or '').strip().lower()
                if enc and salt and (scheme in {'fernet', ''}):
                    passphrase = (os.environ.get('COMM_SCI_KEY_PASSPHRASE') or '').strip()
                    key = (_try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt) or '').strip()

            # 3) key file fallback
            if not key and os.path.exists(KEYS_PATH):
                try:
                    data = _storage_read_json(KEYS_PATH) or {}
                    if isinstance(data, dict):
                        provs2 = data.get('providers')
                        if isinstance(provs2, dict):
                            h = provs2.get('huggingface') or provs2.get('hf') or {}
                            if isinstance(h, dict):
                                key = (h.get('api_key_plain') or h.get('api_key') or '').strip()
                                if not key:
                                    enc = (h.get('api_key_enc') or '').strip()
                                    salt = (h.get('api_key_salt') or '').strip()
                                    scheme = (h.get('enc_scheme') or '').strip().lower()
                                    if enc and salt and (scheme in {'fernet', ''}):
                                        passphrase = (os.environ.get('COMM_SCI_KEY_PASSPHRASE') or '').strip()
                                        key = (_try_decrypt_api_key(enc, passphrase=passphrase, salt_b64=salt) or '').strip()
                        if not key:
                            key = (data.get('HF_TOKEN') or data.get('HUGGINGFACE_TOKEN') or '').strip()
                except Exception:
                    pass

            return OpenAICompatibleClient(base_url=base_url, api_key=key, app_referrer='', app_title='Comm-SCI Desktop')
        except Exception:
            return None

    def _models_cache_path(self, filename: str) -> str:
        """Return cache path in Logs/Cache with best-effort migration from legacy Config path."""
        name = str(filename or '').strip() or 'models_cache.json'
        try:
            target = os.path.join(CACHE_LOG_DIR, name)
        except Exception:
            target = name
        try:
            legacy = os.path.join(CONFIG_DIR, name)
        except Exception:
            legacy = ''

        # Legacy migration: keep existing cache content when path moved from Config -> Logs/Cache.
        try:
            if target and legacy and (not os.path.exists(target)) and os.path.exists(legacy):
                raw = _storage_read_text(legacy, encoding='utf-8')
                if raw:
                    _storage_write_text(target, raw, encoding='utf-8')
        except Exception:
            pass
        return target

    def _gemini_cache_path(self) -> str:
        try:
            return self._models_cache_path('gemini_models_cache.json')
        except Exception:
            return 'gemini_models_cache.json'

    def _gemini_default_models(self) -> list:
        models = []
        try:
            m = self.get_provider_model('gemini', fallback_model='').strip()
            if m:
                models.append(m)
        except Exception:
            pass
        for m in ('gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-3-flash', 'gemini-1.5-pro'):
            models.append(m)
        seen = set()
        uniq = []
        for m in models:
            s = str(m or '').strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(s)
        return uniq

    def _normalize_gemini_model_name(self, raw_name: str) -> str:
        s = str(raw_name or '').strip()
        if s.startswith('models/'):
            s = s.split('/', 1)[1].strip()
        return s

    def get_gemini_models_cached(self, *, force_refresh: bool = False):
        """Return (models, meta) using a local cache plus best-effort live Gemini model listing.

        meta: {'source': 'cache'|'cache-stale'|'live'|'fallback'|'none', 'age_s': int, 'count': int}
        """
        cache_path = self._gemini_cache_path()

        cache_minutes = 30
        try:
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get('gemini') or {}) if isinstance(provs, dict) else {}
            cache_minutes = int((pconf.get('model_cache_minutes') or 30) or 30)
        except Exception:
            cache_minutes = 30

        now = time.time()
        cached = None
        try:
            if os.path.exists(cache_path):
                raw = _storage_read_text(cache_path, encoding='utf-8')
                cached = json.loads(raw) if raw else None
        except Exception:
            cached = None

        def _extract_models(obj):
            out = []
            try:
                src = (obj or {}).get('models') or []
                if isinstance(src, list):
                    for m in src:
                        mm = self._normalize_gemini_model_name(m)
                        if mm:
                            out.append(mm)
            except Exception:
                out = []
            seen = set()
            uniq = []
            for m in out:
                k = m.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(m)
            return sorted(uniq, key=lambda s: s.lower())

        age_s = 10**9
        ts = 0.0
        try:
            ts = float((cached or {}).get('ts') or 0.0)
            if ts > 0:
                age_s = int(max(0.0, now - ts))
        except Exception:
            ts = 0.0
            age_s = 10**9

        models_cached = _extract_models(cached)
        fresh = bool(ts) and (cache_minutes > 0) and (age_s <= int(cache_minutes * 60))
        if fresh and (not force_refresh) and models_cached:
            return models_cached, {'source': 'cache', 'age_s': age_s, 'count': len(models_cached)}

        models_live = []
        if genai is not None:
            try:
                key = (get_api_key() or '').strip()
            except Exception:
                key = ''
            if key:
                try:
                    client = genai.Client(api_key=key)
                    stream = client.models.list()
                    for md in stream:
                        try:
                            name_raw = getattr(md, 'name', '')
                            name = self._normalize_gemini_model_name(name_raw)
                            if not name or not name.lower().startswith('gemini'):
                                continue
                            actions = getattr(md, 'supported_actions', None)
                            if isinstance(actions, list) and actions:
                                allow = False
                                for a in actions:
                                    aa = str(a or '').strip().lower()
                                    if aa in ('generatecontent', 'generate_content'):
                                        allow = True
                                        break
                                if not allow:
                                    continue
                            models_live.append(name)
                        except Exception:
                            continue
                except Exception:
                    models_live = []

        if models_live:
            seen = set()
            uniq = []
            for m in models_live:
                k = str(m or '').strip().lower()
                if not k or k in seen:
                    continue
                seen.add(k)
                uniq.append(str(m).strip())
            models_live = sorted(uniq, key=lambda s: s.lower())
            try:
                _storage_write_json(cache_path, {'ts': now, 'models': models_live}, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return models_live, {'source': 'live', 'age_s': 0, 'count': len(models_live)}

        if models_cached:
            return models_cached, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models_cached)}

        fallback = self._gemini_default_models()
        if fallback:
            return fallback, {'source': 'fallback', 'age_s': age_s, 'count': len(fallback)}
        return [], {'source': 'none', 'age_s': age_s, 'count': 0}

    def _openrouter_cache_path(self) -> str:
        try:
            return self._models_cache_path('openrouter_models_cache.json')
        except Exception:
            return 'openrouter_models_cache.json'

    def get_openrouter_models_cached(self, *, force_refresh: bool = False):
        """Return (models, meta) from OpenRouter /models using a small on-disk cache.

        meta: {'source': 'cache'|'cache-stale'|'live'|'none', 'age_s': int, 'count': int}
        """
        provider = 'openrouter'
        cache_path = self._openrouter_cache_path()

        # cache settings
        cache_minutes = 30
        try:
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get(provider) or {}) if isinstance(provs, dict) else {}
            cache_minutes = int((pconf.get('model_cache_minutes') or 30) or 30)
        except Exception:
            cache_minutes = 30

        now = time.time()

        # load cache
        cached = None
        try:
            if os.path.exists(cache_path):
                raw = _storage_read_text(cache_path, encoding='utf-8')
                cached = json.loads(raw)
        except Exception:
            cached = None

        def _cache_ok(obj):
            if not obj or not isinstance(obj, dict):
                return False
            ts = obj.get('ts')
            if not isinstance(ts, (int, float)):
                return False
            age = now - float(ts)
            if cache_minutes <= 0:
                return False
            return age <= (cache_minutes * 60)

        def _extract_models(obj):
            models = []
            try:
                models = obj.get('models') or []
                if not isinstance(models, list):
                    models = []
                models = [str(m).strip() for m in models if str(m).strip()]
            except Exception:
                models = []
            # dedup + sort
            try:
                seen = set()
                uniq = []
                for m in models:
                    lm = m.lower()
                    if lm in seen:
                        continue
                    seen.add(lm)
                    uniq.append(m)
                models = sorted(uniq, key=lambda s: s.lower())
            except Exception:
                pass
            return models

        if (not force_refresh) and _cache_ok(cached):
            models = _extract_models(cached)
            age_s = int(max(0, now - float(cached.get('ts'))))
            return models, {'source': 'cache', 'age_s': age_s, 'count': len(models)}

        # refresh live
        client = self.build_openrouter_client()
        if client is None or not getattr(client, 'api_key', ''):
            # fall back to stale cache if present
            if cached and isinstance(cached, dict):
                models = _extract_models(cached)
                age_s = 0
                try:
                    age_s = int(max(0, now - float(cached.get('ts') or now)))
                except Exception:
                    age_s = 0
                return models, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models)}
            return [], {'source': 'none', 'age_s': 0, 'count': 0}

        try:
            models, _meta = client.list_models(lang='de')
        except Exception:
            if cached and isinstance(cached, dict):
                models = _extract_models(cached)
                age_s = 0
                try:
                    age_s = int(max(0, now - float(cached.get('ts') or now)))
                except Exception:
                    age_s = 0
                return models, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models)}
            return [], {'source': 'none', 'age_s': 0, 'count': 0}

        # write cache
        try:
            _storage_write_json(cache_path, {'ts': now, 'models': models}, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return models, {'source': 'live', 'age_s': 0, 'count': len(models)}


    def get_huggingface_models_from_config(self):
        """Return models list for Hugging Face from config OR key file (best-effort)."""
        try:
            pconf = self._merged_provider_conf('huggingface') or {}
            models = pconf.get('models') or []
            if not isinstance(models, list):
                models = []
            models = [str(m).strip() for m in models if str(m).strip()]
            # dedup + sort
            seen = set()
            uniq = []
            for mm in models:
                lm = mm.lower()
                if lm in seen:
                    continue
                seen.add(lm)
                uniq.append(mm)
            return sorted(uniq, key=lambda s: s.lower())
        except Exception:
            return []

    def _huggingface_cache_path(self) -> str:
        try:
            return self._models_cache_path('huggingface_models_cache.json')
        except Exception:
            return 'huggingface_models_cache.json'

    def get_huggingface_models_cached(self, *, force_refresh: bool = False):
        """Return (models, meta) for Hugging Face router /models using a small on-disk cache.

        The HF router may not always expose a public model catalog; in that case we fall back
        to the configured list in providers.huggingface.models.

        meta: {'source': 'cache'|'cache-stale'|'live'|'config'|'none', 'age_s': int, 'count': int}
        """
        provider = 'huggingface'
        cache_path = self._huggingface_cache_path()

        # read cache TTL (minutes)
        cache_minutes = 30
        try:
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get(provider) or provs.get('hf') or {}) if isinstance(provs, dict) else {}
            cache_minutes = int((pconf.get('model_cache_minutes') or 30) or 30)
        except Exception:
            cache_minutes = 30

        now = time.time()

        # load cache
        cached = None
        try:
            if os.path.exists(cache_path):
                raw = _storage_read_text(cache_path, encoding='utf-8')
                cached = json.loads(raw)
        except Exception:
            cached = None

        try:
            ts = float((cached or {}).get('ts') or 0.0)
            models_cached = (cached or {}).get('models') or []
        except Exception:
            ts = 0.0
            models_cached = []

        age_s = int(max(0.0, now - ts)) if ts else 10**9
        fresh = bool(ts) and age_s <= int(cache_minutes * 60)

        if fresh and (not force_refresh) and isinstance(models_cached, list) and models_cached:
            return models_cached, {'source': 'cache', 'age_s': age_s, 'count': len(models_cached)}

        # live fetch best-effort
        models_live = []
        try:
            client = self.build_huggingface_client() if hasattr(self, 'build_huggingface_client') else None
            if client is not None and getattr(client, 'api_key', ''):
                models_live, _meta = client.list_models(lang='de')
                if not isinstance(models_live, list):
                    models_live = []
        except Exception:
            models_live = []

        if models_live:
            # write cache
            try:
                _storage_write_json(cache_path, {'ts': now, 'models': models_live}, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return models_live, {'source': 'live', 'age_s': 0, 'count': len(models_live)}

        # fallback to config list
        try:
            models_cfg = self.get_huggingface_models_from_config() if hasattr(self, 'get_huggingface_models_from_config') else []
            if isinstance(models_cfg, list) and models_cfg:
                # write cache as config snapshot (so UI remains fast/offline)
                try:
                    _storage_write_json(cache_path, {'ts': now, 'models': models_cfg}, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                src = 'cache-stale' if (models_cached and not fresh) else 'config'
                return models_cfg, {'source': src, 'age_s': age_s, 'count': len(models_cfg)}
        except Exception:
            pass

        # last resort: stale cache if any
        if isinstance(models_cached, list) and models_cached:
            return models_cached, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models_cached)}

        return [], {'source': 'none', 'age_s': age_s, 'count': 0}

    def _openrouter_cache_path(self) -> str:
        try:
            return self._models_cache_path('openrouter_models_cache.json')
        except Exception:
            return 'openrouter_models_cache.json'

    def get_openrouter_models_cached(self, *, force_refresh: bool = False):
        '''Return (models, meta) using a small on-disk cache.

        meta: {'source': 'cache'|'live'|'none', 'age_s': int, 'count': int}
        '''
        cache_path = self._openrouter_cache_path()

        # read settings
        cache_minutes = 30
        try:
            provs = (self.cfg.config or {}).get('providers') or {}
            pconf = (provs.get(provider) or {}) if isinstance(provs, dict) else {}
            cache_minutes = int((pconf.get('model_cache_minutes') or 30) or 30)
        except Exception:
            cache_minutes = 30

        now = time.time()
        # try load cache
        cached = None
        try:
            if os.path.exists(cache_path):
                raw = _storage_read_text(cache_path, encoding='utf-8')
                cached = json.loads(raw)
        except Exception:
            cached = None

        def _cache_ok(obj):
            if not obj or not isinstance(obj, dict):
                return False
            ts = obj.get('ts')
            if not isinstance(ts, (int, float)):
                return False
            age = now - float(ts)
            if cache_minutes <= 0:
                return False
            return age <= (cache_minutes * 60)

        if (not force_refresh) and _cache_ok(cached):
            models = cached.get('models') or []
            if isinstance(models, list):
                models = [str(m) for m in models if str(m).strip()]
            else:
                models = []
            age_s = int(max(0, now - float(cached.get('ts'))))
            return models, {'source': 'cache', 'age_s': age_s, 'count': len(models)}

        # refresh live
        client = self.build_openrouter_client()
        if client is None:
            # fall back to stale cache if present
            if cached and isinstance(cached, dict):
                models = cached.get('models') or []
                if isinstance(models, list):
                    models = [str(m) for m in models if str(m).strip()]
                else:
                    models = []
                age_s = 0
                try:
                    age_s = int(max(0, now - float(cached.get('ts') or now)))
                except Exception:
                    age_s = 0
                return models, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models)}
            return [], {'source': 'none', 'age_s': 0, 'count': 0}

        try:
            models, meta = client.list_models(lang='de')
        except Exception:
            # fallback to stale cache
            if cached and isinstance(cached, dict):
                models = cached.get('models') or []
                if isinstance(models, list):
                    models = [str(m) for m in models if str(m).strip()]
                else:
                    models = []
                age_s = 0
                try:
                    age_s = int(max(0, now - float(cached.get('ts') or now)))
                except Exception:
                    age_s = 0
                return models, {'source': 'cache-stale', 'age_s': age_s, 'count': len(models)}
            return [], {'source': 'none', 'age_s': 0, 'count': 0}

        # write cache
        try:
            Path(cache_path).write_text(json.dumps({'ts': now, 'models': models}, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass
        return models, {'source': 'live', 'age_s': 0, 'count': len(models)}


# We subclass CSCRefiner because the large UI/command handler block is currently implemented as
# methods on CSCRefiner in this codebase. This is intentional to avoid invasive re-indentation
# and keeps behavior stable.
# ----------------------------


    # ----------------------------
    # Hugging Face Hub Catalog (Top N) - cached
    # ----------------------------
    def _huggingface_catalog_cache_path(self) -> str:
        try:
            return self._models_cache_path("huggingface_catalog_cache.json")
        except Exception:
            return os.path.join(".", "huggingface_catalog_cache.json")


    def _huggingface_catalog_ttl_minutes(self) -> int:
        try:
            prov = self._merged_provider_conf("huggingface") or {}
            v = prov.get("catalog_cache_minutes", None)
            if v is None:
                v = prov.get("model_cache_minutes", 30)
            v = int(v or 30)
            return max(1, min(24*60, v))
        except Exception:
            return 30


    def _huggingface_token(self) -> str:
        try:
            prov = self._merged_provider_conf("huggingface") or {}
            envk = (prov.get("api_key_env") or "HF_TOKEN").strip()
            if envk:
                v = os.environ.get(envk, "") or ""
                if v.strip():
                    return v.strip()
            v = (prov.get("api_key_plain") or "").strip()
            return v
        except Exception:
            return ""

    def _fetch_hf_hub_catalog(self, *, top_n: int, provider_filter: str) -> list:
        """Fetch Hugging Face Hub models list (best-effort) using the public Hub API.

        We intentionally keep this lightweight: pipeline_tag=text-generation, sort by downloads.
        provider_filter: 'all' or inference provider id (e.g. 'novita', 'zai-org', 'cerebras').
        """
        import urllib.request as _urlreq
        top_n = int(top_n or 200)
        top_n = max(1, min(10000, top_n))
        pf = (provider_filter or "all").strip()
        try:
            from urllib.parse import urlencode
        except Exception:
            urlencode = None

        base = "https://huggingface.co/api/models"
        params = {
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(top_n),
        }
        if pf and pf.lower() != "all":
            # Official filter for Hub API: inference_provider
            params["inference_provider"] = pf

        url = base
        if urlencode is not None:
            url = base + "?" + urlencode(params)

        headers = {
            "Accept": "application/json",
            "User-Agent": "Comm-SCI Desktop (HF catalog)",
        }
        tok = self._huggingface_token()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"

        req = _urlreq.Request(url, headers=headers, method="GET")
        with _urlreq.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else []
        out = []
        if isinstance(data, list):
            for it in data:
                mid = ""
                try:
                    obj = (it or {}) if isinstance(it, dict) else {}
                    # HF Hub API commonly uses 'modelId' (e.g. 'Qwen/Qwen2.5-3B-Instruct')
                    mid = obj.get("modelId") or obj.get("id") or obj.get("name") or ""
                except Exception:
                    mid = ""
                mid = str(mid).strip()
                if mid:
                    out.append(mid)
        # Dedup + sort alpha for dropdown usability
        out = sorted(set(out), key=lambda s: s.lower())
        return out

    def get_huggingface_catalog_cached(self, *, top_n: int = 200, provider_filter: str = "all", force_refresh: bool = False):
        """Return (models, meta) from HF Hub catalog with on-disk cache.

        meta: {'source': 'cache'|'cache-stale'|'live'|'none', 'age_s': int, 'count': int, 'top_n': int, 'provider_filter': str, 'error': str?}
        """
        cache_path = self._huggingface_catalog_cache_path()
        ttl_min = self._huggingface_catalog_ttl_minutes()
        now = int(time.time())
        want_pf = (provider_filter or "all").strip()
        want_top = int(top_n or 200)
        want_top = max(1, min(10000, want_top))

        def _meta(source: str, age_s: int, count: int, err: str = "") -> dict:
            out = {"source": source, "age_s": int(age_s or 0), "count": int(count or 0),
                   "top_n": int(want_top), "provider_filter": want_pf}
            if err:
                out["error"] = err
            return out

        def _cache_matches(c: dict) -> bool:
            try:
                return (str((c or {}).get("provider_filter", "all")).strip().lower() == want_pf.lower()
                        and int((c or {}).get("top_n", 0) or 0) == want_top)
            except Exception:
                return False

        # read cache
        cached = None
        try:
            if os.path.exists(cache_path):
                cached = _storage_read_json(cache_path)
        except Exception:
            cached = None

        if (not force_refresh) and cached and _cache_matches(cached):
            try:
                ts = int(cached.get("ts", 0) or 0)
                age = max(0, now - ts)
                models = cached.get("models", []) or []
                if isinstance(models, list) and models and age <= (ttl_min * 60):
                    return models, _meta("cache", age, len(models))
                if isinstance(models, list) and models:
                    # stale cache still useful
                    return models, _meta("cache-stale", age, len(models))
            except Exception:
                pass

        # live fetch
        live_err = ""
        live_models = []
        try:
            live_models = self._fetch_hf_hub_catalog(top_n=want_top, provider_filter=want_pf) or []
            if not isinstance(live_models, list):
                live_models = []
        except Exception as e:
            live_err = str(e)
            live_models = []

        if live_models:
            # persist cache
            try:
                payload = {"ts": now, "top_n": want_top, "provider_filter": want_pf, "models": live_models}
                _storage_write_json(cache_path, payload, indent=2, ensure_ascii=False)
            except Exception:
                # cache write failure should not break the UI
                pass
            return live_models, _meta("live", 0, len(live_models))

        # live empty or failed: fall back to any cache (even mismatch) as last resort
        if cached:
            try:
                models = cached.get("models", []) or []
                if isinstance(models, list) and models:
                    ts = int(cached.get("ts", 0) or 0)
                    age = max(0, now - ts)
                    return models, _meta("cache-stale", age, len(models), live_err)
            except Exception:
                pass

        return [], _meta("none", 0, 0, live_err)

        def _cache_matches(c):
            try:
                return (str((c or {}).get("provider_filter", "all")).strip().lower() == want_pf.lower()
                        and int((c or {}).get("top_n", 0) or 0) == want_top)
            except Exception:
                return False

        if not force_refresh and cached and _cache_matches(cached):
            try:
                ts = int(cached.get("ts", 0) or 0)
                age = max(0, now - ts)
                models = cached.get("models", []) or []
                if isinstance(models, list) and models and age <= (ttl_min * 60):
                    return models, {"source": "cache", "age_s": age, "count": len(models), "top_n": want_top, "provider_filter": want_pf}
                if isinstance(models, list) and models:
                    return models, {"source": "cache-stale", "age_s": age, "count": len(models), "top_n": want_top, "provider_filter": want_pf}
            except Exception:
                pass

        # live fetch
        try:
            models = self._fetch_hf_hub_catalog(top_n=want_top, provider_filter=want_pf)
            meta = {"source": "live", "age_s": 0, "count": len(models), "top_n": want_top, "provider_filter": want_pf}
            try:
                _storage_write_json(cache_path, {"ts": now, "top_n": want_top, "provider_filter": want_pf, "models": models}, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return models, meta
        except Exception:
            # fallback: no catalog available
            return [], {"source": "none", "age_s": 0, "count": 0, "top_n": want_top, "provider_filter": want_pf}



class ProviderService:
    """Thin provider facade used by Api._llm_call and model refresh paths."""

    def __init__(self, cfg_mgr, provider_router):
        self.cfg = cfg_mgr
        self.router = provider_router

    def canonical_provider_id(self, provider: str) -> str:
        p = str(provider or "").strip().lower()
        if p in ("hf", "huggingface"):
            return "huggingface"
        if p in ("openai", "openai_compat", "openrouter"):
            return "openrouter"
        if p == "gemini":
            return "gemini"
        return "openrouter"

    def supports_native_retrieval(self, provider: str) -> bool:
        """Return whether this provider path can wire native retrieval tools in wrapper runtime."""
        pid = self.canonical_provider_id(provider)
        if pid == "gemini":
            return True
        try:
            r = self.router
            if r is not None and hasattr(r, "supports_native_retrieval"):
                return bool(r.supports_native_retrieval(pid))
        except Exception:
            pass
        return False

    def get_openai_client(self, provider: str):
        pid = self.canonical_provider_id(provider)
        r = self.router
        if r is None:
            return None
        try:
            if pid == "huggingface" and hasattr(r, "build_huggingface_client"):
                return r.build_huggingface_client()
            if pid == "openrouter" and hasattr(r, "build_openrouter_client"):
                return r.build_openrouter_client()
        except Exception:
            return None
        return None

    def get_cached_models(self, provider: str, *, force_refresh: bool = False):
        pid = self.canonical_provider_id(provider)
        r = self.router
        models, meta = [], {}
        if r is None:
            return models, meta
        try:
            if pid == "huggingface":
                if hasattr(r, "get_huggingface_models_cached"):
                    models, meta = r.get_huggingface_models_cached(force_refresh=force_refresh)
                if (not models) and hasattr(r, "get_huggingface_models_from_config"):
                    models = r.get_huggingface_models_from_config() or []
            elif pid == "openrouter":
                if hasattr(r, "get_openrouter_models_cached"):
                    models, meta = r.get_openrouter_models_cached(force_refresh=force_refresh)
            elif pid == "gemini":
                if hasattr(r, "get_gemini_models_cached"):
                    models, meta = r.get_gemini_models_cached(force_refresh=force_refresh)
        except Exception:
            models, meta = [], {}
        return models or [], meta or {}

    def get_provider_model(self, provider: str, *, fallback_model: str = "") -> str:
        pid = self.canonical_provider_id(provider)
        r = self.router
        if r is None:
            return str(fallback_model or "").strip()
        try:
            if hasattr(r, "get_provider_model"):
                return str(r.get_provider_model(pid, fallback_model=fallback_model) or "").strip()
        except Exception:
            pass
        return str(fallback_model or "").strip()

    def get_config_fallback_models(self, provider: str):
        pid = self.canonical_provider_id(provider)
        out = []
        try:
            provs = (getattr(self.cfg, "config", {}) or {}).get("providers") or {}
            pconf = provs.get(pid) if isinstance(provs, dict) else {}
            fb = (pconf or {}).get("fallback_models") if isinstance(pconf, dict) else None
            if isinstance(fb, list):
                for x in fb:
                    sx = str(x or "").strip()
                    if sx and sx not in out:
                        out.append(sx)
        except Exception:
            out = []
        return out


# ----------------------------
# STUFE 2+: JSONL instrumentation (audit stream, append-only)
# ----------------------------
_jsonl_audit_stream_date_cache = None
_jsonl_audit_stream_path_cache = None

def _get_audit_stream_jsonl_path(now: datetime) -> str:
    """Return JSONL audit stream path for current day (best-effort)."""
    global _jsonl_audit_stream_date_cache, _jsonl_audit_stream_path_cache
    try:
        day = now.strftime('%Y%m%d')
    except Exception:
        day = 'unknown'
    if _jsonl_audit_stream_date_cache != day or not _jsonl_audit_stream_path_cache:
        _jsonl_audit_stream_date_cache = day
        try:
            _jsonl_audit_stream_path_cache = os.path.join(AUDIT_LOG_DIR, f'AuditStream_{day}.jsonl')
        except Exception:
            _jsonl_audit_stream_path_cache = None
    return _jsonl_audit_stream_path_cache

def _append_jsonl_line(path: str, obj: dict) -> None:
    if _auditstream is not None:
        try:
            _auditstream.append_jsonl_line(path, obj)
            return
        except Exception:
            pass
    """Append one JSON object as a single line. Must never raise."""
    try:
        if not path:
            return
        line = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        if _StorageService is not None:
            try:
                _svc = _StorageService()
                _svc.append_text(path, line + '\n')
                return
            except Exception:
                pass
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        return


def _storage_read_json(path: str):
    try:
        if _StorageService is not None:
            svc = _StorageService()
            if hasattr(svc, 'read_json'):
                return svc.read_json(path)
    except Exception:
        pass
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _storage_read_text(path: str, *, encoding: str = 'utf-8'):
    try:
        if _StorageService is not None:
            svc = _StorageService()
            if hasattr(svc, 'read_text'):
                return svc.read_text(path, encoding=encoding)
    except Exception:
        pass
    try:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception:
        return None


def _storage_write_json(path: str, payload, *, indent: int = 2, ensure_ascii: bool = True) -> bool:
    try:
        if _StorageService is not None:
            svc = _StorageService()
            if hasattr(svc, 'write_json'):
                return bool(svc.write_json(path, payload, indent=indent, ensure_ascii=ensure_ascii))
    except Exception:
        pass
    try:
        parent = os.path.dirname(str(path or ''))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
        return True
    except Exception:
        return False


def _storage_write_text(path: str, text: str, *, encoding: str = 'utf-8') -> bool:
    try:
        if _StorageService is not None:
            svc = _StorageService()
            if hasattr(svc, 'write_text'):
                return bool(svc.write_text(path, text, encoding=encoding))
    except Exception:
        pass
    try:
        parent = os.path.dirname(str(path or ''))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            f.write(str(text))
        return True
    except Exception:
        return False

def _build_jsonl_meta(api) -> dict:
    """Best-effort metadata for JSONL events (no secrets)."""
    wrapper_name = globals().get('WRAPPER_NAME')
    wrapper_version = globals().get('WRAPPER_VERSION')
    ruleset_version = None
    provider = None
    model = None
    language_policy_mode = None

    try:
        gov = getattr(api, 'gov_state', None)
        try:
            provider = getattr(gov, 'provider', None) or getattr(gov, 'provider_name', None)
            model = getattr(gov, 'model', None) or getattr(gov, 'model_name', None)
            language_policy_mode = getattr(gov, 'language_policy_mode', None)
        except Exception:
            pass
        try:
            rules = getattr(gov, 'rules', None)
            if rules is not None:
                ruleset_version = getattr(rules, 'version', None) or getattr(rules, 'ruleset_version', None)
        except Exception:
            pass
    except Exception:
        pass

    if _auditstream is not None:
        try:
            return _auditstream.build_jsonl_meta(
                wrapper_name=str(wrapper_name) if wrapper_name is not None else None,
                wrapper_version=str(wrapper_version) if wrapper_version is not None else None,
                ruleset_version=str(ruleset_version) if ruleset_version is not None else None,
                provider=str(provider) if provider is not None else None,
                model=str(model) if model is not None else None,
                language_policy_mode=str(language_policy_mode) if language_policy_mode is not None else None,
            )
        except Exception:
            pass

    meta = {'wrapper_name': wrapper_name, 'wrapper_version': wrapper_version}
    if ruleset_version:
        meta['ruleset_version'] = ruleset_version
    if provider:
        meta['provider'] = provider
    if model:
        meta['model'] = model
    if language_policy_mode:
        meta['language_policy_mode'] = language_policy_mode
    return meta



class Api(CSCRefiner):
    def __init__(self):
        # Bind governance + config safely
        super().__init__(globals().get('gov'), globals().get('cfg'))
        # True iff the current model chat-session was created with the ruleset injected
        self.session_with_governance: bool = True
        try:
            _g = globals().get('gov')
            if _g is not None:
                gov.runtime_state = self.gov_state
                setattr(_g, 'runtime_state', self.gov_state)
        except Exception:
            pass


        # Dependency health (Stage 3d/3e)
        _strict = _strict_modules_enabled()
        if _strict:
            _check_optional_module('Module.auditstream', strict=True)
            _check_optional_module('Module.rendering_utils', strict=True)
            _check_optional_module('Module.compliance_scan', strict=True)

        self.deps_status = {
            'auditstream': _check_optional_module('Module.auditstream'),
            'rendering_utils': _check_optional_module('Module.rendering_utils'),
            'compliance_scan': _check_optional_module('Module.compliance_scan'),
        }

        # Provider routing (single-file adapters)
        try:
            self.provider_router = ProviderRouter(globals().get('cfg'))
            self.provider_service = ProviderService(globals().get('cfg'), self.provider_router)
            # Warm provider model caches from disk (no network).
            try:
                self._warm_model_caches_from_disk()
            except Exception:
                pass
            # --- B5 MVP: Session tracking (best-effort, no secrets) ---
            import uuid
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            self.trace_id = self.session_id
            self.session_start_dt = datetime.now()
            self.session_requests = 0
            self.session_rate_limit_hits = 0
            self.session_repair_passes = 0
            self.session_csc_applied_count = 0
            self.session_guard_hits = 0
            self.session_events = []  # list of {ts,type,data}
            self.SESSION_EVENTS_MAX = SESSION_EVENTS_MAX
            # --- /B5 ---
            
        except Exception:
            self.provider_router = None
            self.provider_service = None

        try:
            if _GovernanceService is not None:
                self.governance_service = _GovernanceService(
                    normalize_headings_fn=normalize_known_markdown_control_headings,
                    enforce_self_debunking_fn=enforce_self_debunking_contract,
                    normalize_sci_trace_fn=normalize_sci_trace_numbering,
                    normalize_self_debunking_numbering_fn=normalize_self_debunking_numbering,
                    enforce_qc_footer_fn=enforce_qc_footer_deltas,
                    ensure_qc_footer_present_fn=ensure_qc_footer_present,
                    normalize_evidence_tags_fn=normalize_evidence_tags,
                )
            else:
                self.governance_service = None
        except Exception:
            self.governance_service = None

        try:
            self.ui_controller = _UIController() if _UIController is not None else None
        except Exception:
            self.ui_controller = None

        try:
            self.storage_service = _StorageService() if _StorageService is not None else None
        except Exception:
            self.storage_service = None

        # Window handles
        self.main_win = None
        self.panel_win = None
        self.panel_hidden = False
        self._qc_override_open = False
        self._exit_confirm_open = False
        self._qc_override_prompt_reset_pending = False

        # Panel API bridge (small surface to avoid pywebview method enumeration issues)
        try:
            self.panel_bridge = PanelBridge(self)
        except Exception:
            self.panel_bridge = None
        try:
            self.main_bridge = MainBridge(self)
        except Exception:
            self.main_bridge = None

        # S7 panel asset bootstrap/runtime self-test state
        self._panel_force_embedded_html = False
        try:
            _meta = globals().get("PANEL_HTML_ASSET_META") or {}
            if not isinstance(_meta, dict):
                _meta = {}
        except Exception:
            _meta = {}
        self.panel_html_source = str(_meta.get("source") or "embedded")
        self.panel_html_asset_meta = dict(_meta)
        if _panel_bootstrap_state_mod is not None:
            try:
                self.panel_bootstrap_state = _panel_bootstrap_state_mod.panel_bootstrap_initial_state(self.panel_html_source)
            except Exception:
                self.panel_bootstrap_state = {
                    "status": "idle",          # idle | pending | passed | failed | skipped
                    "source": self.panel_html_source,
                    "reason": "",
                    "created_at": None,
                    "reported_at": None,
                }
        else:
            self.panel_bootstrap_state = {
                "status": "idle",          # idle | pending | passed | failed | skipped
                "source": self.panel_html_source,
                "reason": "",
                "created_at": None,
                "reported_at": None,
            }
        self.panel_bootstrap_last_report = None
        self.panel_bootstrap_timeout_s = 2.5
        self._panel_bootstrap_ready_event = threading.Event()
        self._panel_closed_ignore_count = 0
        try:
            self._panel_bootstrap_ready_event.set()
        except Exception:
            pass


        # Closing guard (prevents double-exit)
        self.is_closing = False
        self._connect_inflight = False  # connect dedupe guard
        self._passphrase_pending_provider = ""
        self._passphrase_pending_reason = ""
        self._passphrase_pending_since = ""

        # Persisted geometry (best-effort)
        self.panel_geom = {}
        try:
            _cfg = globals().get('cfg')
            if _cfg and hasattr(_cfg, 'get_panel_geom'):
                self.panel_geom = _cfg.get_panel_geom() or {}
        except Exception:
            self.panel_geom = {}

        # Session stats
        self.session_req_count = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0

        # Last provider call debug info (for Comm Audit)
        self.last_call_info = {
            'provider': '',
            'model': '',
            'ms': 0,
            'usage': {},
        }

        # Rate limiting for LLM calls (best-effort; configurable via Comm-SCI-Config.json)
        self.rate_limiter = None
        self.rate_limit_enabled = True
        try:
            _cfg = globals().get('cfg')
            if _cfg is not None and hasattr(_cfg, 'config'):
                conf = getattr(_cfg, 'config', {}) or {}
                self.rate_limit_enabled = bool(conf.get('rate_limit_enabled', True))
                per_m = int(conf.get('rate_limit_per_minute', 30) or 30)
                per_h = int(conf.get('rate_limit_per_hour', 120) or 120)
                self.rate_limiter = RateLimiter(per_minute=per_m, per_hour=per_h, scopes=conf.get('rate_limit_scopes'))
        except Exception:
            self.rate_limiter = RateLimiter(per_minute=30, per_hour=120, scopes=None)

        # Model client/chat (initialized later)
        self.client = None
        self.chat = None

        # Chat history for export
        self.history = []
        self._last_content_user_prompt = ""
        self._last_response_is_content_answer = False
        self.input_history_max_entries = 200
        self.input_cmd_history = []
        try:
            self.input_cmd_history = self._load_input_history_entries()
        except Exception:
            self.input_cmd_history = []

        # Runtime governance state (fail-safe default)
        try:
            self.gov_state = _init_state_from_rules()
        except Exception:
            self.gov_state = GovernanceRuntimeState()

        # Attach validator and CSC refiner hook (best-effort)
        try:
            self.validator = OutputComplianceValidator(globals().get('gov'), globals().get('cfg'))
        except Exception:
            self.validator = None

        # Make refiner reachable for the CSC bridge path
        try:
            _gov = globals().get('gov')
            if _gov is not None:
                setattr(_gov, 'csc_refiner', self)
        except Exception:
            pass


    # ----------------------------
    # STUFE 0: Minimal observability
    # ----------------------------
    def log_event(self, kind: str, payload=None, *, level: str = "info") -> None:
        """Append a lightweight, JSON-safe runtime event entry.

        Design goals:
        - Never raise (fail-safe).
        - No secrets: store only short previews + hashes by default.
        - Keeps existing session_events behavior intact (additive).
        """
        try:
            k = str(kind or "").strip() or "event"
            lvl = str(level or "info").strip().lower() or "info"

            # Ensure list exists (defensive for older states/tests)
            if not isinstance(getattr(self, 'session_events', None), list):
                self.session_events = []

            data = payload
            # Make payload JSON-ish without deep recursion
            if isinstance(payload, (str, int, float, bool)) or payload is None:
                data = payload
            elif isinstance(payload, dict):
                safe = {}
                for kk, vv in list(payload.items())[:50]:
                    try:
                        sk = str(kk)
                    except Exception:
                        continue

                    # Redact obvious secret-bearing keys (defensive)
                    try:
                        _skl = sk.lower()
                    except Exception:
                        _skl = ''
                    if any(t in _skl for t in ('api_key','apikey','token','secret','password','bearer')):
                        safe[sk] = '<redacted>'
                        continue
                    # short-circuit big strings
                    if isinstance(vv, str) and len(vv) > 300:
                        safe[sk] = _safe_preview_text(vv, 300)
                    elif isinstance(vv, (str, int, float, bool)) or vv is None:
                        safe[sk] = vv
                    else:
                        safe[sk] = _safe_preview_text(vv, 120)
                data = safe
            else:
                data = _safe_preview_text(payload, 200)

            # Attach minimal correlation/diagnostics context (no behavior changes).
            try:
                _prov = self.cfg.get_provider() if hasattr(self, 'cfg') else None
            except Exception:
                _prov = None
            try:
                _gs = getattr(self, 'gov_state', None)
            except Exception:
                _gs = None

            self.session_events.append({
                'ts': datetime.now().isoformat(),
                'type': k,
                'level': lvl,
                'trace_id': getattr(self, 'trace_id', getattr(self, 'session_id', None)),
                'provider': _prov,
                'profile': getattr(_gs, 'active_profile', None),
                'sci_active': getattr(_gs, 'sci_active', None),
                'sci_variant': getattr(_gs, 'sci_variant', None),
                'comm_active': getattr(_gs, 'comm_active', None),
                'language_policy_mode': getattr(_gs, 'language_policy_mode', None),
                'data': data,
            })

            # Cap in-memory session event list (avoid unbounded RAM growth).
            try:
                _cap = int(getattr(self, 'SESSION_EVENTS_MAX', globals().get('SESSION_EVENTS_MAX', 2000)))
                if _cap > 0 and isinstance(self.session_events, list) and len(self.session_events) > _cap:
                    del self.session_events[:-_cap]
            except Exception:
                pass


            # JSONL audit stream (best-effort, no behavior change)
            try:
                now = datetime.now()
                stream_path = _get_audit_stream_jsonl_path(now)
                gs = getattr(self, 'gov_state', None)
                rec = {
                    'ts': now.isoformat(),
                    'event': k,
                    'level': lvl,
                    'trace_id': getattr(self, 'trace_id', getattr(self, 'session_id', None)),
                    'route': {
                        'is_command': None,
                        'comm_active': getattr(gs, 'comm_active', None),
                        'sci_variant': getattr(gs, 'sci_variant', None),
                        'ui_lang': getattr(gs, 'ui_lang', None),
                        'color': getattr(gs, 'color', None),
                        'language_policy_mode': getattr(gs, 'language_policy_mode', None),
                    },
                    'data': data,
                    'meta': _build_jsonl_meta(self),
                }
                _append_jsonl_line(stream_path, rec)
            except Exception:
                pass
        except Exception:
            return




    def _get_enforcement_settings(self) -> dict:
        """Return normalized enforcement settings from config (fail-safe defaults)."""
        conf = {}
        try:
            conf = (getattr(cfg, "config", {}) or {})
            if not isinstance(conf, dict):
                conf = {}
        except Exception:
            conf = {}

        nested = conf.get("enforcement")
        if not isinstance(nested, dict):
            nested = {}

        try:
            pol_raw = nested.get("policy", conf.get("enforcement_policy", "audit_only"))
            pol = str(pol_raw or "audit_only").strip().lower()
        except Exception:
            pol = "audit_only"
        if pol not in ("audit_only", "strict_warn", "strict_block"):
            pol = "audit_only"

        try:
            enabled = bool(nested.get("enabled", conf.get("enforcement_enabled", True)))
        except Exception:
            enabled = True

        raw_bs = nested.get("blocked_severities", conf.get("enforcement_blocked_severities", ["critical", "major"]))
        bs = []
        if isinstance(raw_bs, list):
            for item in raw_bs:
                s = str(item or "").strip().lower()
                if s in ("critical", "major", "minor") and s not in bs:
                    bs.append(s)
        if not bs:
            bs = ["critical", "major"]

        return {"enabled": enabled, "policy": pol, "blocked_severities": bs}

    def _get_enforcement_policy(self) -> str:
        """Return normalized enforcement policy from config."""
        try:
            return str((self._get_enforcement_settings() or {}).get("policy") or "audit_only")
        except Exception:
            return "audit_only"


    def clear_chat(self):
        """Clear in-memory history and reset the main chat UI (no model call, no provider switch)."""
        try:
            try:
                self.history = []
            except Exception:
                pass

            # Best-effort: treat as a new local session
            try:
                import uuid
                self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
                self.session_start_dt = datetime.now()
                self.session_requests = 0
                self.session_rate_limit_hits = 0
                self.session_repair_passes = 0
                self.session_csc_applied_count = 0
                self.session_guard_hits = 0
                self.session_events = []
            except Exception:
                pass

            # Clear UI (main window)
            try:
                win = getattr(self, 'main_win', None)
                if win is not None:
                    sm = json.dumps('Chat cleared.', ensure_ascii=False)
                    win.evaluate_js(f"resetChatToStatus({sm});")
            except Exception:
                pass

            return {'ok': True, 'history_len': 0}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


if _QCBridge is not None:
    QCBridge = _QCBridge
else:
    class QCBridge:
        """Minimal JS bridge for the QC Override dialog (fallback local implementation)."""

        def __init__(self, api):
            self._api = api

        def ping(self, _payload=None):
            try:
                import time as _time
                return {"ok": True, "ts": _time.time()}
            except Exception:
                return {"ok": True}

        def _call(self, fn_name: str, *args):
            try:
                return getattr(self._api, fn_name)(*args)
            except Exception as e:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        def qc_get_state(self, _payload=None):
            return self._call("qc_get_state")

        def qc_override_apply(self, values):
            return self._call("qc_override_apply", values)

        def qc_override_clear(self, _payload=None):
            return self._call("qc_override_clear")

        def qc_override_cancel(self, _payload=None):
            return self._call("qc_override_cancel")


if _PanelBridge is not None:
    PanelBridge = _PanelBridge
else:
    class PanelBridge:
        """Separate JS-API bridge for the Panel window (fallback local implementation)."""

        def __init__(self, api):
            self._api = api

        def ping(self, _payload=None):
            return self._api.ping()

        def get_ui(self):
            return self._api.get_ui()

        def panel_action(self, action, payload=None):
            return self._api.panel_action(action, payload)


if _MainBridge is not None:
    MainBridge = _MainBridge
else:
    class MainBridge:
        """Slim JS-API bridge for the main chat window (fallback local implementation)."""

        def __init__(self, api):
            self._api = api

    def _main_bridge_forwarder(_name):
        def _call(self, *args, **kwargs):
            return getattr(self._api, _name)(*args, **kwargs)
        return _call

    for _mb_name in (
        "ask",
        "remote_cmd",
        "submit_cgi_feedback",
        "get_input_history",
        "append_input_history",
        "ui_qc_bar_enabled",
        "is_ready",
        "ping",
        "update_stats_ui",
        "ensure_panel_visible",
        "load_rule_file",
        "export",
        "settings",
        "close_app",
        "set_exit_confirm_open",
        "get_help_content",
    ):
        setattr(MainBridge, _mb_name, _main_bridge_forwarder(_mb_name))

# ----------------------------

def _ui_replay_loaded_history(self, status_msg: str = "Loaded chat log."):
    """Rebuild main chat UI from self.history without calling the model (robust).

    Uses window.resetChatFromHistory(history, statusMsg) when available; otherwise falls back
    to incremental replay to avoid a stuck/blank UI for large logs.
    """
    try:
        win = getattr(self, 'main_win', None)
        if not win:
            return
        hist = getattr(self, 'history', None)
        if not isinstance(hist, list):
            hist = []

        ui_hist = []
        for msg in hist:
            if not isinstance(msg, dict):
                continue
            role = (msg.get('role', '') or '').strip().lower()
            content = msg.get('content', '') if 'content' in msg else msg.get('text', '')
            if content is None:
                content = ''
            if role == 'assistant':
                role = 'bot'
            elif role == 'system':
                role = 'sys'
            elif role not in ('user', 'bot', 'sys'):
                role = 'user'

            if role == 'bot':

                # If the loaded history contains a legacy plaintext 'Comm Config' dump (very large JSON),
                # re-render it deterministically as collapsible HTML to keep the UI readable and within margins.
                try:
                    _c = str(content)
                    if 'Loaded rules file:' in _c and ('QC-Matrix:' in _c or 'QC Matrix:' in _c):
                        _lines = _c.splitlines()
                        if _lines and 'Loaded rules file:' in _lines[0]:
                            _json_i = None
                            for _i in range(1, len(_lines)):
                                _ls = _lines[_i].lstrip()
                                if _ls.startswith('{') or _ls.startswith('['):
                                    _json_i = _i
                                    break
                            _qc_i = None
                            for _i in range(len(_lines)-1, -1, -1):
                                _s = _lines[_i].strip()
                                if _s.startswith('QC-Matrix:') or _s.startswith('QC Matrix:'):
                                    _qc_i = _i
                                    break
                            if _json_i is not None:
                                _status = _lines[0].strip()
                                _qc = _lines[_qc_i].strip() if _qc_i is not None else ''
                                _json_end = _qc_i if (_qc_i is not None and _qc_i > _json_i) else len(_lines)
                                _raw_json = "\n".join(_lines[_json_i:_json_end]).strip()
                                _ui_lang = self._lang()
                                _summary = 'Raw JSON anzeigen' if _ui_lang == 'de' else 'Show raw JSON'
                                _minor = (
                                    'Read-only view of the full governance configuration (deterministic from JSON, no LLM).'
                                    if _ui_lang != 'de'
                                    else 'Nur-Lese-Ansicht der vollständigen Governance-Konfiguration (deterministisch aus JSON, ohne LLM).'
                                )
                                content = (
                                    '<div class="comm-help comm-config">'
                                    f'<div class="help-status">{html.escape(_status)}</div>'
                                    f'<div class="minor">{html.escape(_minor)}</div>'
                                    '<details class="config-details">'
                                    f'<summary>{html.escape(_summary)}</summary>'
                                    f'<pre class="raw-json">{html.escape(_raw_json)}</pre>'
                                    '</details>'
                                    + (f"<div style='margin-top:10px'>{html.escape(_qc)}</div>" if _qc else '')
                                    + '</div>'
                                )
                except Exception:
                    pass
                try:
                    _c = str(content)
                    if _c.lstrip().startswith('<'):
                        _h = sanitize_html(_c)
                    else:
                        import markdown as _markdown
                        _h = _markdown.markdown(_c, extensions=['extra', 'codehilite'])
                        _h = sanitize_html(_h)
                except Exception:
                    _h = sanitize_html(html.escape(str(content)))
                ui_hist.append({'role': 'bot', 'html': _h})
            else:
                ui_hist.append({'role': role, 'content': str(content)})

        payload = json.dumps(ui_hist, ensure_ascii=False)
        sm = json.dumps(str(status_msg or "Loaded chat log."), ensure_ascii=False)

        # Bulk path
        try:
            js = (
                "(function(){try{"
                "if(window.resetChatFromHistory){window.resetChatFromHistory(%s,%s); return 'OK';}"
                "return 'NOFUNC';"
                "}catch(e){return 'ERR:'+String(e);}})()"
            ) % (payload, sm)
            res = win.evaluate_js(js)
            if isinstance(res, str) and res == 'OK':
                return
        except Exception:
            pass

        # Fallback: incremental replay
        try:
            win.evaluate_js(f"resetChatToStatus({sm});")
        except Exception:
            return

        for m in ui_hist:
            try:
                r = (m.get('role') or 'user')
                if r == 'bot':
                    h_js = json.dumps(str(m.get('html', '')), ensure_ascii=False)
                    win.evaluate_js(f"addMsg('bot', {h_js}, false, null);")
                else:
                    c_js = json.dumps(html.escape(str(m.get('content', ''))), ensure_ascii=False)
                    rr = 'sys' if r == 'sys' else 'user'
                    win.evaluate_js(f"addMsg('{rr}', {c_js});")
            except Exception:
                continue
    except Exception:
        return

# Bind into CSCRefiner (the big mixin class that owns UI handlers) if missing.
try:
    if 'CSCRefiner' in globals() and not hasattr(CSCRefiner, '_ui_replay_loaded_history'):
        setattr(CSCRefiner, '_ui_replay_loaded_history', _ui_replay_loaded_history)
except Exception:
    pass


# FIXUP: bind top-level helpers into Api
# (Some patch steps can accidentally place helper defs at module scope.)
# ----------------------------
for _name in ("_render_error_html", "_safe_html", "_handle_command_deterministic", "_as_dict", "_as_list", "_safe_get", "_render_error_fallback", "set_api_key_for_provider", "delete_api_key_for_provider", "load_log_from_path"):
    if _name in globals() and not hasattr(Api, _name):
        setattr(Api, _name, globals()[_name])

def _run_module_selftest():
    # Minimal offline self-tests (no webview / no network)
    gov_local = GovernanceManager()
    gov_local.load_file(DEFAULT_JSON)
    st = GovernanceRuntimeState()
    # max_depth from rules if available
    try:
        md = int(((gov_local.data.get('sci') or {}).get('recursive_sci') or {}).get('max_depth', 2))
    except Exception:
        md = 2
    # Enter recursion md times should succeed; one more should fail
    for _ in range(md):
        assert try_enter_sci_recursion(st, max_depth=md) is True
    assert try_enter_sci_recursion(st, max_depth=md) is False
    # Simulate one-shot auto-return
    st.sci_recursion_one_shot = True
    cur = int(getattr(st, 'sci_recursion_depth', 0) or 0)
    st.sci_recursion_depth = max(cur - 1, 0)
    assert st.sci_recursion_depth >= 0
    print('[SelfTest] OK')


def _bootstrap_desktop_windows(api, webview_module):
    if _app_bootstrap is not None:
        return _app_bootstrap.bootstrap_desktop_windows(
            api,
            webview_module,
            title=MAIN_WINDOW_TITLE,
            html_chat=HTML_CHAT,
        )
    _main_w, _main_h, _main_x, _main_y = 1100, 1000, 0, 0
    _panel_w, _panel_h, _panel_x, _panel_y = 340, 1000, 1100, 0
    try:
        _screens = getattr(webview_module, "screens", None)
        if _screens:
            _s0 = _screens[0]
            _sx = int(getattr(_s0, "x", 0))
            _sy = int(getattr(_s0, "y", 0))
            _sw = int(getattr(_s0, "width", 0))
            _sh = int(getattr(_s0, "height", 0))
            if _sw >= 800 and _sh >= 500:
                _panel_w = max(320, min(420, int(round(_sw * 0.26))))
                if _panel_w > _sw - 480:
                    _panel_w = max(280, _sw - 480)
                _panel_w = max(220, min(_panel_w, max(_sw - 220, 220)))
                _main_w = max(_sw - _panel_w, 220)
                _panel_w = max(_sw - _main_w, 220)
                if _main_w + _panel_w > _sw:
                    _panel_w = max(_sw - _main_w, 1)
                _main_h = _sh
                _panel_h = _sh
                _main_x = _sx
                _main_y = _sy
                _panel_x = _sx + _main_w
                _panel_y = _sy
    except Exception:
        pass
    try:
        api.panel_geom = {"x": _panel_x, "y": _panel_y, "width": _panel_w, "height": _panel_h}
    except Exception:
        pass
    api.main_win = webview_module.create_window(
        MAIN_WINDOW_TITLE, html=HTML_CHAT, js_api=(getattr(api, 'main_bridge', None) or api),
        width=_main_w, height=_main_h,
        x=_main_x, y=_main_y
    )
    # Pre-create the Panel window *before* webview.start().
    # On macOS/Cocoa, creating secondary windows from a JS->Python callback can leave the JS API bridge uninitialized,
    # resulting in a Panel stuck at 'Loading panel...'.
    api._create_panel()
    api._create_qc_override()
    # HIER: Binden des Schließen-Events ("X") an unsere Logik
    api.main_win.events.closed += api.on_main_window_close


def _run_desktop_app():
    if _app_bootstrap is not None:
        _app_bootstrap.run_desktop_app(
            api_factory=Api,
            webview_module=webview,
            genai_module=genai,
            genai_types=types,
            title=MAIN_WINDOW_TITLE,
            html_chat=HTML_CHAT,
        )
        return
    if webview is None:
        raise SystemExit('pywebview is required. Install with: pip install pywebview')
    if genai is None or types is None:
        raise SystemExit('google-genai is required. Install with: pip install google-genai')
    api = Api()
    _bootstrap_desktop_windows(api, webview)
    webview.start(api.start_background_thread)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _run_module_selftest()
        raise SystemExit(0)
    _run_desktop_app()
