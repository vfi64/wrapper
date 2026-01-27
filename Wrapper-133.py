import os
import json
import re
import html
import sys
import traceback
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
    import bleach  # type: ignore
except Exception:
    bleach = None  # type: ignore

def sanitize_html(html_text: str) -> str:
    # Defensive HTML sanitization for model outputs before injecting into pywebview.
    # Keeps a conservative allow-list while preserving our own injected spans/images.
    if not html_text:
        return ''
    if bleach is None:
        return html_text

    allowed_tags = [
        'p','br','b','strong','i','em','u','code','pre','blockquote',
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
        'th': ['colspan','rowspan','class','style'],
        'td': ['colspan','rowspan','class','style'],
    }
    try:
        cleaned = bleach.clean(
            html_text,
            tags=allowed_tags,
            attributes=allowed_attrs,
            protocols=['http','https','mailto'],
            strip=True,
        )
        return cleaned
    except Exception:
        return html_text

# ----------------------------
# WRAPPER IDENTITY (dynamic, derived from filename)
# ----------------------------

def _detect_wrapper_identity() -> tuple[str, str]:
    """Return (WRAPPER_NAME, WRAPPER_VERSION) based on this file's name.

    Expected filename: Wrapper-<NNN>.py. Falls back safely if pattern is missing.
    """
    try:
        stem = Path(__file__).stem  # e.g. 'Wrapper-115'
        m = re.match(r'^(Wrapper)-(\d+)$', stem)
        if m:
            return f"Wrapper-{m.group(2)}", m.group(2)
    except Exception:
        pass
    return "Wrapper", "000"

WRAPPER_NAME, WRAPPER_VERSION = _detect_wrapper_identity()
MAIN_WINDOW_TITLE = f"{WRAPPER_NAME} Comm-SCI-Control"
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

PROJECT_DIR = SCRIPT_DIR  # backwards-compat alias

JSON_DIR = os.path.join(SCRIPT_DIR, 'JSON')
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'Config')

LOGS_DIR = os.path.join(SCRIPT_DIR, 'Logs')
AUDIT_LOG_DIR = os.path.join(LOGS_DIR, 'Audit')
CHAT_LOG_DIR = os.path.join(LOGS_DIR, 'Chats')
USAGE_LOG_DIR = os.path.join(LOGS_DIR, 'Usage_statistics')

# Create directories (idempotent)
for _d in (JSON_DIR, CONFIG_DIR, LOGS_DIR, AUDIT_LOG_DIR, CHAT_LOG_DIR, USAGE_LOG_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass

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

# Default ruleset location: ./JSON/Comm-SCI-v19.6.8.json
DEFAULT_JSON = os.path.join(JSON_DIR, 'Comm-SCI-v19.6.8.json')

# Fallback: if the ruleset is placed next to the script (legacy layout), use it.
_alt_ruleset = os.path.join(SCRIPT_DIR, 'Comm-SCI-v19.6.8.json')
if (not os.path.exists(DEFAULT_JSON)) and os.path.exists(_alt_ruleset):
    DEFAULT_JSON = _alt_ruleset

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

# Backwards-compatible alias (some forks used STATS_PATH)
STATS_PATH = STATS_FILENAME

# ----------------------------
# UI (English-only)
# ----------------------------
UI_LANG = "de"  # hard-fixed; language switching removed

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

    # SCI selection is treated as a chat input that triggers deterministic logic.
    if sci_pending and re.match(r'^[A-Ha-h]$', txt):
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
        
        lang_code = UI_LANG
        lang_instruction = "IMPORTANT: You must reply in English. All explanations and outputs must be in English unless the active profile explicitly requires otherwise."

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
            "1. Schwäche: Die Antwort kann Vereinfachungen enthalten oder stillschweigende Annahmen machen.\n"
            "   Warum relevant: Vereinfachungen können Randfälle oder alternative Deutungen verdecken.\n"
            "   Prüfen/Widerlegen (nächster Schritt): Die zentralen Annahmen explizit machen und gegen Primärquellen/Definitionen prüfen.\n\n"
            "2. Schwäche: Die Antwort kann wichtige Gegenpositionen oder Unsicherheitsgrenzen auslassen.\n"
            "   Warum relevant: Fehlende Einschränkungen können die Gültigkeit überdehnen oder Sicherheit vortäuschen.\n"
            "   Prüfen/Widerlegen (nächster Schritt): Mindestens ein starkes Gegenbeispiel ergänzen und prüfen, ob die Kernaussagen bestehen bleiben.\n"
        )
    else:
        block = (
            f"\n\n{title}:\n\n"
            "1. Weakness: The answer may rely on simplified framing or implicit assumptions.\n"
            "   Why it matters: Simplifications can hide edge-cases or alternative interpretations.\n"
            "   What would verify/falsify (next check): Identify key assumptions and test them against primary sources or formal definitions.\n\n"
            "2. Weakness: The answer may omit important counter-perspectives or uncertainty boundaries.\n"
            "   Why it matters: Missing caveats can overstate confidence or applicability.\n"
            "   What would verify/falsify (next check): Add at least one strong counter-example and check whether conclusions still hold.\n"
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
        m = re.search(r"(?is)(\bSelf-Debunking\b.*?)(\n\s*QC\-Matrix:|\Z)", text)
        if not m:
            return text

        block = m.group(1)
        tail_marker = m.group(2)  # either QC footer marker or end

        # Translate label phrases (keep punctuation/colon style flexible).
        repl = [
            (r"(?i)\bWeakness\b\s*:", "Schwäche:"),
            (r"(?i)\bWhy\s+it\s+matters\b\s*:", "Warum das wichtig ist:"),
            (r"(?i)\bWhat\s+would\s+verify\s*/\s*falsify\s*\(next\s+check\)\b\s*:",
             "Was würde verifizieren/falsifizieren (nächster Check):"),
            (r"(?i)\bWhat\s+would\s+verify\s+or\s+falsify\s*\(next\s+check\)\b\s*:",
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

def enforce_self_debunking_contract(text: str, gov_mgr, profile_name: str, *, is_command: bool = False, lang: str = "en") -> str:
    """Deterministically enforce the Self-Debunking contract (2–3 numbered points) when required.

    Note: is_command is accepted for API compatibility and does not change behavior here.

    - If missing: inject a minimal compliant block (no new factual claims).
    - If too many points: keep the first 3.
    - If too few points: add generic points to reach the minimum.
    - Ensure placement BEFORE the QC footer.
    """
    try:
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

        # Locate QC footer (insertion anchor)
        qc_m = re.search(r"(?im)^\s*QC(?:-Matrix)?\s*:\s*.*$", text)
        qc_pos = qc_m.start() if qc_m else None

        # Locate Self-Debunking title line
        title_re = re.compile(rf"(?im)^\s*(?:#+\s*)?\*{{0,2}}{re.escape(title)}\*{{0,2}}\s*:?\s*$")
        m = title_re.search(text)
        if not m:
            # Missing entirely -> inject minimal
            out = inject_minimal_self_debunking(text, title=title, lang=lang)
            return normalize_self_debunking_language(out, lang)
        # If title occurs after QC, remove that trailing block and inject at the correct place.
        if qc_pos is not None and m.start() > qc_pos:
            trimmed = text[:m.start()].rstrip()
            out = inject_minimal_self_debunking(trimmed, title=title, lang=lang)
            return out

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

        # If there are no numbered points at all, treat as missing.
        if not points:
            injected = inject_minimal_self_debunking(before.rstrip() + "\n\n" + after.lstrip(), title=title)
            return normalize_self_debunking_language(injected, lang)
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
                    f"   Why it matters: Missing caveats can overstate confidence or applicability.\n"
                    f"   What would verify/falsify (next check): Identify a concrete counterexample and test whether the conclusion still holds."
                )
        # Re-number points sequentially (1..k)
        normalized = []
        for i, p in enumerate(points, 1):
            p = re.sub(r"(?m)^\s*\d+\s*[\.)]\s+", f"{i}. ", p, count=1)
            normalized.append(p.strip())

        new_block = "\n\n" + "\n\n".join(normalized).rstrip() + "\n\n"

        # Reassemble
        out = before.rstrip() + new_block + after.lstrip()
        return out

    except Exception:
        return text

def apply_color_spans(text: str, enabled: bool = True) -> str:
    """Render Evidence-Linker tags with actual HTML colors (does not invent tags)."""
    if not enabled or not text:
        return text

    def repl(m: re.Match) -> str:
        tag = m.group("tag")
        suffix = m.group("suffix") or ""
        emoji = m.group("emoji") or ""
        color = _EVIDENCE_COLOR.get(tag, "#616161")
        token = f"[{tag}{suffix}]"
        if emoji:
            token = f"{token} {emoji}"
        return f"<span style=\"color:{color}; font-weight:600;\">{token}</span>"

    # Patterns like: [GREEN] 🟢  or [GREEN-WEB] 🟢
    pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z]+)?)\]\s*(?P<emoji>[🟢🟡🔴⚪️])?")
    return pat.sub(repl, text)


@dataclass
class GovernanceRuntimeState:
    comm_active: bool = False
    active_profile: str = "Standard"
    overlay: str = ""
    color: str = "on"
    conversation_language: str = ""
    answer_language: str = "de"
    sci_pending: bool = False
    sci_variant: str = ""
    sci_active: bool = False

    qc_overrides: dict = field(default_factory=dict)
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

        # Helper: detect step header lines (allow optional leading ordered-list prefix)
        step_set = {str(s) for s in required_steps}
        hdr_re = re.compile(r"^\s*(?:\d+\.)?\s*([A-Za-z][A-Za-z0-9_]*)(\s*:)\s*$")

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
            m = hdr_re.match(ln)
            if m:
                name = m.group(1)
                if name in step_set:
                    flush()
                    cur = name
                    recognized += 1
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

        step_set = {str(s) for s in required_steps}
        # Accept optional leading numbering + colon
        hdr_re = re.compile(r"^\s*(?:\d+\.)?\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)$" )

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
            m = hdr_re.match(ln)
            if m:
                name = m.group(1)
                if name in step_set:
                    flush()
                    cur = name
                    recognized += 1
                    continue
            if cur is not None:
                buf.append(ln)
        flush()
        if recognized < 2:
            return text

        # Build deterministic HTML
        # Keep styling minimal and consistent with existing CSS; rely on browser defaults.
        html_parts = [
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
        out_lines.append("SCI Trace:")
        out_lines.append("\n".join(html_parts))
        out_lines.extend(post)
        return "\n".join(out_lines)
    except Exception:
        return text


def _init_state_from_rules():
    if not gov.loaded:
        return GovernanceRuntimeState()
    ui = gov.get_ui_data()
    prof = ui.get("defaults", {}).get("profile", "Standard") or "Standard"
    ov = ui.get("defaults", {}).get("overlay", "") or ""
    col = ui.get("defaults", {}).get("color_default", "on") or "on"
    return GovernanceRuntimeState(comm_active=False, active_profile=prof, overlay=ov, color=col, conversation_language=(UI_LANG or '').lower(), answer_language=(getattr(cfg, "get_answer_language", lambda: "de")() or "de"), sci_pending=False, sci_variant="", sci_active=False)


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

  .qc-bar { margin-top: 8px; border-top: 1px solid #eee; padding-top: 5px; font-size: 11px; color:#555; }
  .qc-btn { cursor: pointer; margin-right: 8px; color: #1a73e8; background:#f1f3f4; padding:2px 6px; border-radius:4px; }
  .qc-btn:hover { background:#1a73e8; color:white; }

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
  .comm-help pre.raw-json { white-space: pre; overflow-x: auto; background: #f6f8fa; padding: 10px; border-radius: 10px; font-size: 11px; border: 1px solid #e0e0e0; }
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
        <span style="font-weight:bold; margin-right:10px;">__WRAPPER_LABEL__</span>
        
        <button class="load-btn" onclick="window.pywebview.api.load_rule_file()" title="Load ruleset">📂</button>
        <span id="rulefile" style="font-size:11px; color:#8ab4f8; margin-right:10px;"></span>
        
        <span id="stats" class="top-stats">Session: loading...</span>
    </div>
    <div>
       <button class="menu-btn" onclick="window.pywebview.api.export()">💾 EXPORT</button>
       <button class="menu-btn" onclick="window.pywebview.api.settings()">⚙️ PANEL</button>
       <button class="exit-btn" onclick="window.pywebview.api.close_app()">❌ EXIT</button>
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
      const thrTok = escHtml(csc.min_token_count ?? '');
      const thrGov = escHtml(csc.min_token_count_governance ?? '');

      let details = '';
      details += `<details><summary>Details</summary>`;
      details += `<div class="csc-details">`;
      details += `Trigger: ${trig || '—'}<br>`;
      details += `Mode: ${mode || '—'}${csc.governance_triggered ? ' (governance)' : ''}<br>`;
      details += `Overlay: ${ov || 'off'} · Profile: ${prof || '—'}<br>`;
      details += `Score: f=${fs} · tokens=${tok}<br>`;
      details += `Thresholds (x${mult || 1}): f≥${thrFs}, tok≥${thrTok}, gov_tok≥${thrGov}`;
      details += `</div></details>`;

      return `<span class="csc-badge" title="CSC applied">CSC applied</span>` +
             `<div class="csc-warning"><b>CSC</b>: ${msg}${details}</div>`;
  }

  function addMsg(role, text, qc=false, csc=null) {
      const d = document.createElement('div');
      d.className = 'msg ' + role;
      let html = '<button class="copy-btn" onclick="copyToClipboard(this)" title="Copy">📋</button>';
      if(role === 'bot') html += renderCscBlock(csc);
      html += text;
      if(qc) html += '<div class="qc-bar">QC (CGI): <span class="qc-btn" onclick="rate(0)">0</span><span class="qc-btn" onclick="rate(1)">1</span><span class="qc-btn" onclick="rate(2)">2</span><span class="qc-btn" onclick="rate(3)">3</span></div>';
      d.innerHTML = html;
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

  async function send() {
      const txt = document.getElementById('inp').value.trim();
      if(!txt) return;
      document.getElementById('inp').value = '';
      addMsg('user', txt);
      const btn = document.getElementById('btn');
      btn.disabled = true;
      try {
          const res = await window.pywebview.api.ask(txt);
          const qc = await window.pywebview.api.ui_qc_bar_enabled();
          if(typeof res === 'string') {
              addMsg('bot', res, qc);
          } else {
              addMsg('bot', res.html || '', qc, res.csc || null);
          }
      } catch(e) {
          addMsg('bot', '<span style="color:red">Error: '+e+'</span>');
      }
      btn.disabled = false;
      document.getElementById('inp').focus();
  }

  function rate(v) { window.pywebview.api.remote_cmd('CGI Rating: '+v); }
  function remoteInput(txt) { document.getElementById('inp').value=txt; send(); }
  
  document.getElementById('inp').addEventListener('keydown', (e)=>{
      if(e.key==='Enter' && !e.shiftKey){e.preventDefault(); send();}
  });

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
  .status-box { font-family: monospace; font-size: 10px; color: #111; margin-bottom: 8px; background: #fff; padding: 6px; border-radius: 6px; border: 1px solid #ddd; }
  .hint { font-size: 9px; color: #666; margin-bottom: 8px; text-align: center; }
  .row { display:flex; gap:6px; }
  .row > * { flex: 1; }
  .smallbtn { padding: 7px 6px; font-size: 10px; }
  .err { color: #b00020; }
  .ok { color: #0b6b0b; }
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
    <button id="refreshModelsBtn" class="smallbtn" onclick="refreshModels()" title="Fetch /models and refresh cache (OpenRouter/HF)">Refresh Models</button>
    <button class="smallbtn" id="qcOverrideBtn" onclick="run('QC Override')" title="QC Override">⚙ QC</button>
</div>

  <div id="hfCatalogRow" class="row" style="display:none;">
    <select id="hfProviderFilter" class="setting-select" onchange="onHFProviderFilterChange()">
      <option value="all">HF Provider: all</option>
    </select>
    <input id="hfTopN" class="setting-select" type="number" min="1" max="1000" value="200" />
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
  if(btn) btn.style.display = (p === 'openrouter' || p === 'huggingface') ? 'block' : 'none';

  // HF catalog controls
  const hfRow = document.getElementById('hfCatalogRow');
  if(hfRow) hfRow.style.display = (p === 'huggingface') ? 'flex' : 'none';
  if(p === 'huggingface'){
    const opts = (data.hf_provider_filter_options || ['all']);
    let savedPF = (data.hf_catalog_default_provider_filter || 'all');
    let savedTopN = String(data.hf_catalog_default_top_n || 200);
    try {
      savedPF = (localStorage.getItem('hf_provider_filter') || savedPF);
      savedTopN = (localStorage.getItem('hf_catalog_topn') || savedTopN);
    } catch(e) {}
    _setSelectOptions('hfProviderFilter', opts.map(x => ({value:x, label:`HF Provider: ${x}`})), savedPF);
    const topInp = document.getElementById('hfTopN');
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
    freeOnly = isOpenRouter && (localStorage.getItem('openrouter_free_only') === '1');
    if(freeCb) freeCb.checked = freeOnly;
  } catch(e) {}

  // Restore model search query per provider
  try {
    const key = _storageKeyForModelQuery(p);
    const savedQ = localStorage.getItem(key) || '';
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
      _setStatus('cmd failed: ' + (r.error || 'unknown error'), 'err');
    }
  } catch(e) {
    _setStatus('cmd failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
}

async function changeProvider() {
  const provider = document.getElementById('provider').value;
  try {
    await _apiCall('panel_action', ['set_provider', {provider: provider}], 8000);
    await _apiCall('panel_action', ['refresh_models', {provider: provider}], 15000);
  } catch(e) {
    _setStatus('set_provider failed: ' + (e && e.message ? e.message : String(e)), 'err');
    return;
  }
  await buildUI();
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
  try { localStorage.setItem(_storageKeyForModelQuery(p), q); } catch(e) {}
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
  try { localStorage.setItem('openrouter_free_only', v ? '1' : '0'); } catch(e) {}
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
  try {
    const v = (document.getElementById('hfProviderFilter') || {}).value || 'all';
    localStorage.setItem('hf_provider_filter', v);
  } catch(e) {}
}

async function fetchHFCatalog(){
  const api = _api();
  if(!api){ _setStatus('offline: cannot fetch HF catalog', 'err'); return; }
  let topN = 200;
  let pf = 'all';
  try {
    topN = parseInt((document.getElementById('hfTopN') || {}).value || '200', 10);
    if(!isFinite(topN) || topN < 1) topN = 200;
    pf = (document.getElementById('hfProviderFilter') || {}).value || 'all';
    localStorage.setItem('hf_catalog_topn', String(topN));
  } catch(e) {}
  try {
    await _apiCall('panel_action', ['hf_catalog', {top_n: topN, provider_filter: pf, force_refresh: true}], 20000);
  } catch(e) {
    _setStatus('hf_catalog failed: ' + (e && e.message ? e.message : String(e)), 'err');
  }
  await buildUI();
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

  async function callApi(fn, payload){
    if(!window.pywebview || !pywebview.api || !pywebview.api[fn]){
      throw new Error('Bridge not ready: ' + fn);
    }
    return await pywebview.api[fn](payload || {});
  }

  async function boot(){
    setStatus('Offline (bridge not ready). Trying…', false);
    attach();

    for(let i=0;i<40;i++){
      try{
        if(window.pywebview && pywebview.api && pywebview.api.ping){
          const pong = await pywebview.api.ping();
          if(pong && pong.ok){
            const st = await callApi('qc_get_state', {});
            if(st && st.ok){
              const prof = st.profile || 'Standard';
              document.title = 'QC-Vorgaben temporär anpassen – Profil: ' + prof;
              const h = qs('#title');
              if(h) h.textContent = 'QC-Vorgaben temporär anpassen – Profil: ' + prof;

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
            }else{
              setStatus('Online, aber qc_get_state fehlgeschlagen.', true);
            }
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
          const st = await callApi('qc_get_state', {});
          if(st && st.ok){
            const defaults = st.defaults || {};
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
          }
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

  window.QCUI = { boot, onApply, onClear, onCancel, preset };
})();
</script>
</head>
<body onload="QCUI.boot()">
  <h2 id="title">QC-Vorgaben temporär anpassen – Profil: ?</h2>
  <p class="sub">Temporäre QC-Anpassung (gilt bis Profilwechsel / Clear)</p>

  <div class="row"><div class="lbl">Clarity</div><input id="sl-clarity" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-clarity">2</div></div>
  <div class="row"><div class="lbl">Brevity</div><input id="sl-brevity" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-brevity">2</div></div>
  <div class="row"><div class="lbl">Evidence</div><input id="sl-evidence" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-evidence">2</div></div>
  <div class="row"><div class="lbl">Empathy</div><input id="sl-empathy" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-empathy">2</div></div>
  <div class="row"><div class="lbl">Consistency</div><input id="sl-consistency" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-consistency">2</div></div>
  <div class="row"><div class="lbl">Neutrality</div><input id="sl-neutrality" type="range" min="0" max="3" step="1" value="2"><div class="val" id="v-neutrality">2</div></div>

  <div class="presets">
    <button onclick="QCUI.preset('verbose')">Ausführlicher</button>
    <button onclick="QCUI.preset('short')">Kürzer</button>
    <button onclick="QCUI.preset('evidence')">Evidenzlastig</button>
    <button onclick="QCUI.preset('neutral')">Neutral</button>
  </div>

  <div class="actions">
    <button onclick="QCUI.onApply()">Apply</button>
    <button onclick="QCUI.onClear()">Clear Overrides</button>
    <button onclick="QCUI.onCancel()">Abbrechen</button>
  </div>

  <div id="status" class="status"></div>
</body>
</html>
"""


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
        return UI_LANG

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
        maps = (vdef.get("maps_to") or {})
        mtype = maps.get("type")
        mval = maps.get("value")

        if mtype == "sci_mode" and isinstance(mval, str):
            steps = self._get_path(self.gov.data, f"sci.modes.{mval}.steps", []) or []
            if isinstance(steps, list):
                return [str(s) for s in steps]

        steps = self._get_path(self.gov.data, "sci.modes.SCI.steps", []) or []
        if isinstance(steps, list):
            return [str(s) for s in steps]
        return []

    def _label_regex(self, label: str):
        parts = re.split(r"[_\s]+", label.strip())
        core = r"[_\s\-]*".join([re.escape(p) for p in parts if p])
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

        if all(p is not None for p in positions):
            if any(positions[i] >= positions[i+1] for i in range(len(positions)-1)):
                vios.append("SCI Trace steps not in required order.")
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


    # -------- Language drift (heuristic; warning by default) --------
    
    # -------- Verification Route Gate (hard) --------
    def validate_verification_route_gate(self, text: str, *, is_command: bool):
        """Hard check: if strong-claim heuristic triggers, require at least one verification route marker.
        Compliance shortcuts (to reduce false alarms):
          - Any explicit uncertainty label U1..U6 counts as 'downgraded' and passes.
          - Any Evidence-Linker provenance tag like [GREEN-WEB]/[YELLOW-DOC]/... counts as route presence.
        """
        vios = []
        if is_command or (not self.gov.loaded) or (not text):
            return vios
    
        gate = (self.gov.data.get("global_defaults", {}) or {}).get("verification_route_gate", {}) or {}
        if not gate.get("enabled", False):
            return vios
    
        if re.search(r"\bU[1-6]\b", text):
            return vios
    
        heur = gate.get("strong_claim_heuristics", {}) or {}
        if not heur.get("enabled", False):
            return vios
    
        kw = []
        kw += list(heur.get("keywords_de", []) or [])
        kw += list(heur.get("keywords_en", []) or [])
        text_l = text.lower()
        has_strong = any(str(k).lower() in text_l for k in kw) if kw else False
        if not has_strong:
            return vios
    
        markers = gate.get("route_presence_markers", {}) or {}
        found = False
        # route_presence_markers may contain non-dict metadata entries (e.g. "enabled", "notes").
        # Guard strictly to avoid "str has no attribute get" crashes.
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
    
        if not found:
            vios.append("Verification Route Gate: strong-claim heuristic triggered, but no verification route markers found (Source/Measurement/Contrast/Web Check).")
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

    def validate_language(self, text: str):
        """Heuristic language drift detection (soft warning).

        Strategy: score common DE vs EN stopwords on the first explanatory slice,
        excluding code blocks, to reduce false positives in technical answers.
        """
        vios = []
        tgt = self._conversation_lang()
        if not tgt:
            return vios

        # Strip code blocks and inline code to avoid bias from snippets
        sample = re.sub(r"```[\s\S]*?```", " ", text)
        sample = re.sub(r"`[^`]*`", " ", sample)
        sample = sample[:700]

        tokens = re.findall(r"[A-Za-zÄÖÜäöüß]+", sample.lower())
        if not tokens:
            return vios

        de_markers = {
            "der","die","das","und","ist","sind","wird","wurde","mit","für","nicht",
            "dass","ich","du","wir","sie","es","ein","eine","als","auch","bei","auf",
            "im","in","zu","von","oder","wenn","dann","weil"
        }
        en_markers = {
            "the","and","is","are","will","was","were","with","for","not","that","this",
            "you","your","we","they","a","an","as","also","in","to","of","or","if",
            "then","because"
        }

        de_score = sum(1 for t in tokens if t in de_markers)
        en_score = sum(1 for t in tokens if t in en_markers)

        # Require a minimum signal to avoid noisy warnings
        if tgt == "de":
            if en_score >= 8 and en_score > max(2, de_score) * 2:
                vios.append("Language drift: expected DE, output appears predominantly EN.")
        elif tgt == "en":
            if de_score >= 8 and de_score > max(2, en_score) * 2:
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

        # Language drift: follow the dialog language contract.
        # - For command outputs (help/state/config/audit/menu), enforce HARD.
        # - For content answers, keep SOFT (the wrapper may intentionally request a different answer language).
        try:
            lang_vios = self.validate_language(text)
            if is_command and lang_vios:
                hard += lang_vios
            else:
                soft += lang_vios
        except Exception:
            pass

        return hard, soft


    def build_repair_prompt(self, *, user_prompt: str, raw_response: str, state, hard_violations: list, soft_violations: list):
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
        if any("Verification Route Gate" in v for v in hard_violations):
            parts.append("Verification Route Gate repair guidance:")
            parts.append("- Add at least ONE verification route marker line.")
            parts.append("- Prefer safe/transparent markers (do NOT fabricate web checks):")
            parts.append("  - Source: TRAIN (general background knowledge)")
            parts.append("  - Measurement: not performed")
            parts.append("  - Contrast: plausible alternative noted but not evaluated")
            parts.append("  - Web-Check: not performed")
            parts.append("- If you cannot support the strong claim, downgrade it and include an uncertainty label U1–U6.")
            parts.append("")

            parts.append("SCI Trace requirements:")
            parts.append("- Include a visible block titled 'SCI Trace'.")
            parts.append("- Include ALL step labels exactly (underscores/spaces/hyphens allowed), in this order:")
            for s in steps:
                parts.append(f"  - {s}")

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

    Key principle (v19.6.8 intent): refinement_only.
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

        # Profile constraints (strict per v19.6.8 intent)
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
        """English-only build."""
        return UI_LANG


    # ----------------------------
    # SCI prompt helpers (deterministic, zero-LLM)
    # ----------------------------
    def _sci_variant_def(self, letter: str):
        """Return (variant_def, steps, mapped_mode). Safe on malformed JSON."""
        L = (letter or "").strip().upper()
        sci = self.gov.data.get("sci", {}) if getattr(self, "gov", None) else {}
        variant_menu = sci.get("variant_menu", {}) if isinstance(sci, dict) else {}
        variants = variant_menu.get("variants", {}) if isinstance(variant_menu, dict) else {}
        vdef = variants.get(L, {}) if isinstance(variants, dict) else {}
        maps_to = "SCI"
        if isinstance(vdef, dict):
            maps_to = vdef.get("maps_to", "SCI") or "SCI"
        modes = sci.get("modes", {}) if isinstance(sci, dict) else {}
        mode_obj = modes.get(maps_to, {}) if isinstance(modes, dict) else {}
        steps = mode_obj.get("steps", []) if isinstance(mode_obj, dict) else []
        if not isinstance(steps, list):
            steps = []
        steps = [str(s) for s in steps if s is not None]
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
            "2) In that section, list the required steps below in the same order, each as a short line/bullet.\n"
            "3) After the SCI Trace section, provide the final answer.\n"
            "4) Do not rename steps, do not invent extra steps.\n\n"
            "Required SCI Trace steps:\n" + step_lines +
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
            f"{sysname} v{ver} · Active profile: {profile} · SCI: {sci_out} · Overlay: {overlay} · "
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

        out.append("\n1) Comm commands (commands.primary)")
        _render_cmd_group("", "primary")

        out.append("\n2) Profiles (commands.profile_control)")
        _render_cmd_group("", "profile_control")

        out.append("\n3) Modes & Overlays (commands.mode_control)")
        _render_cmd_group("", "mode_control")

        out.append("\n4) SCI control (commands.sci_control)")
        _render_cmd_group("", "sci_control")

        # SCI variants
        out.append("\n5) SCI variants (A–H)")
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
        out.append("\n6) Numeric codes (numeric_codes)")
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
        out.append("\n7) Quality/control modules")
        out.append("- QC: Rating footer (Clarity/Brevity/Evidence/Empathy/Consistency/Neutrality). Active while QC=on (profile-dependent).") 
        out.append("- CGI: Optional user feedback (cognitive gain), if CGI=on.")
        out.append("- Control Layer: deterministic token/output contracts (no silent adjustments).")

        # Parsing rule (SCI pending)
        out.append("\n8) Parsing rules (SCI pending)")
        out.append("- If SCI selection is pending: a single letter A–H selects the variant.")
        out.append("- Otherwise, a standalone letter is treated as normal input text (not a command).")

        # Deterministic QC footer (keeps toolchain stable)
        out.append("\n" + self._qc_footer_for_profile(prof))

        return "\n".join(out).strip()
 
    def _render_comm_help_html(self, lang=None):
        """HTML help: renders explanatory text in the current conversation language (de/en),
        keeps command tokens canonical (English-only), preserves styling/tables/colors,
        and uses JSON as Source of Truth. Optional I18N keys are used only if present.
        """
        ui_lang = UI_LANG
        ui_lang = UI_LANG

        def tr(key: str, fallback: str = "") -> str:
            try:
                s = key
                return s if s and s != key else fallback
            except Exception:
                return fallback

        def norm_key(token: str) -> str:
            try:
                s = (token or "").strip().lower()
                s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
                return s
            except Exception:
                return ""

        def tr_cmd_desc(token: str, fallback: str) -> str:
            """If you have per-command translations in I18N, they can be provided as:
               cmd_desc_<normalized_token> or cmd_<normalized_token>.
               Example: token 'Profile Expert' -> key 'cmd_desc_profile_expert'
            """
            nk = norm_key(token)
            if nk:
                for k in (f"cmd_desc_{nk}", f"cmd_{nk}", f"help_{nk}", f"desc_{nk}"):
                    s = tr(k, "")
                    if s:
                        return s
            return fallback

        gov_obj = getattr(self, "gov", None) or globals().get("gov")
        data = getattr(gov_obj, "data", None) if gov_obj else None
        if not isinstance(data, dict):
            return "<div class='comm-help' style='color:red'>Error: Governance JSON not available.</div>"

        commands = data.get("commands") or {}
        if not isinstance(commands, dict):
            commands = {}

        # Titles / chrome (I18N first; fallback to JSON l10n where available; else EN defaults)
        col_cmd = tr("help_col_cmd", "Command")
        col_desc = tr("help_col_desc", "Description / Function")

        sec_primary = tr("help_sec_primary", "Primary Commands")
        sec_profiles = tr("help_sec_profiles", "Profiles")
        sec_modes = tr("help_sec_modes", "Modes & Overlays")
        sec_sci = tr("help_sec_sci", "Scientific Control (SCI)")
        sec_sci_variants = tr("help_sec_sci_variants", "SCI Variants")
        sec_codes = tr("help_sec_codes", "Numeric Codes")
        sec_modules = tr("help_sec_modules", "Modules / Tools")
        sec_parsing = "Parsing rules"

        # Didactic hint: prefer JSON meta.rendering.l10n.comm_help.didactic_hint, fallback to I18N
        didactic_hint = ""
        try:
            did = (((((data.get("meta") or {}).get("rendering") or {}).get("l10n") or {})
                    .get("comm_help") or {}).get("didactic_hint") or {})
            if isinstance(did, dict):
                didactic_hint = (did.get("en") or "")
        except Exception:
            didactic_hint = ""
        if not didactic_hint:
            didactic_hint = tr("help_didactic", "")

        def render_command_group(title_text: str, group_dict):
            if not isinstance(group_dict, dict) or not group_dict:
                return ""
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(str(title_text))}</h3>")
            out.append(
                "<table><thead><tr>"
                f"<th>{html.escape(str(col_cmd))}</th>"
                f"<th>{html.escape(str(col_desc))}</th>"
                "</tr></thead><tbody>"
            )
            # keep deterministic ordering
            for token in sorted(group_dict.keys(), key=lambda x: str(x)):
                spec = group_dict.get(token)
                desc = ""
                if isinstance(spec, dict):
                    desc = spec.get("function") or ""
                else:
                    desc = str(spec) if spec is not None else ""
                # Optional: localized per-command desc via I18N (only if present)
                desc = tr_cmd_desc(str(token), desc)
                out.append(
                    "<tr><td class='cmd'>%s</td><td>%s</td></tr>"
                    % (html.escape(str(token)), html.escape(str(desc)))
                )
            out.append("</tbody></table></div>")
            return "\n".join(out)

        parts = []
        parts.append("<div class='comm-help'>")
        if didactic_hint:
            parts.append(f"<p><i>{html.escape(str(didactic_hint))}</i></p>")

        # 1–4: command groups (JSON = source of truth)
        parts.append(render_command_group(sec_primary, commands.get("primary")))
        parts.append(render_command_group(sec_profiles, commands.get("profile_control")))
        parts.append(render_command_group(sec_modes, commands.get("mode_control")))
        parts.append(render_command_group(sec_sci, commands.get("sci_control")))

        # 5) SCI variants (A–H): JSON source of truth; optional I18N if present
        sci = data.get("sci") or {}
        variant_menu = (sci.get("variant_menu") or {}) if isinstance(sci, dict) else {}
        variants = (variant_menu.get("variants") or {}) if isinstance(variant_menu, dict) else {}
        if isinstance(variants, dict) and variants:
            col_var = tr("sci_menu_col_var", "Variant")
            col_name = "Name"
            col_focus = tr("sci_menu_col_focus", "Focus / Method")

            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(str(sec_sci_variants))}</h3>")
            out.append(
                "<table><thead><tr>"
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

                # Optional I18N overrides (only if your I18N defines them)
                name_i18n = tr(f"sci_name_{letter}", "") or tr(f"sci_var_{letter}", "")
                focus_i18n = tr(f"sci_focus_{letter}", "")

                name = name_i18n or name_json
                focus = focus_i18n or focus_json

                out.append(
                    "<tr><td class='cmd'><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                    % (html.escape(letter), html.escape(str(name)), html.escape(str(focus)))
                )
            out.append("</tbody></table></div>")
            parts.append("".join(out))

        # 6) Numeric codes (kept stable; light localized headers only)
        nc = data.get("numeric_codes") or {}
        cats = nc.get("categories") or []
        dash_meaning = (nc.get("special_values") or {}).get("dash") or nc.get("dash_meaning") or ""
        default_code = nc.get("default") or ""
        if isinstance(cats, list) and cats:
            h_cat = "Category"
            h_opt = "Option"
            h_mean = "Meaning"

            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(str(sec_codes))}</h3>")
            if default_code:
                dc_lbl = "Default code"
                out.append(f"<div class='minor'>{html.escape(dc_lbl)}: <b>{html.escape(str(default_code))}</b></div>")

            out.append("<table class='numcodes-table'>")
            out.append(f"<thead><tr><th>{h_cat}</th><th>{h_opt}</th><th>{h_mean}</th></tr></thead><tbody>")

            for cat in cats:
                if not isinstance(cat, dict):
                    continue
                nm = cat.get("name") or ""
                idx = cat.get("index") or ""
                opts = cat.get("options") or {}
                if not isinstance(opts, dict):
                    opts = {}
                keys = list(opts.keys())
                keys.sort(key=lambda x: int(x) if str(x).isdigit() else str(x))
                first = True
                for k in keys:
                    meaning = opts.get(k, "")
                    if first:
                        out.append(
                            "<tr><td><b>%s</b> (Index %s)</td><td><b>%s</b></td><td>%s</td></tr>"
                            % (html.escape(str(nm)), html.escape(str(idx)), html.escape(str(k)), html.escape(str(meaning)))
                        )
                        first = False
                    else:
                        out.append(
                            "<tr><td></td><td><b>%s</b></td><td>%s</td></tr>"
                            % (html.escape(str(k)), html.escape(str(meaning)))
                        )

            if dash_meaning:
                out.append("<tr><td><b>Dash</b></td><td><b>-</b></td><td>%s</td></tr>" % html.escape(str(dash_meaning)))
            out.append("</tbody></table></div>")
            parts.append("".join(out))

        # 7) Modules (stable; keep styling)
        gd = data.get("global_defaults") or {}
        qc_notes = ((gd.get("qc") or {}).get("notes")) or ""
        cgi_notes = ((gd.get("cgi") or {}).get("notes")) or ""
        control_layer_desc = ((data.get("control_layer") or {}).get("description")) or ""
        if qc_notes or cgi_notes or control_layer_desc:
            h_mod = "Module"
            h_notes = "Notes"
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(str(sec_modules))}</h3>")
            out.append(f"<table><thead><tr><th>{h_mod}</th><th>{h_notes}</th></tr></thead><tbody>")
            if qc_notes:
                out.append("<tr><td class='cmd'>QC</td><td>%s</td></tr>" % html.escape(str(qc_notes)))
            if cgi_notes:
                out.append("<tr><td class='cmd'>CGI</td><td>%s</td></tr>" % html.escape(str(cgi_notes)))
            if control_layer_desc:
                out.append("<tr><td class='cmd'>Control Layer</td><td>%s</td></tr>" % html.escape(str(control_layer_desc)))
            out.append("</tbody></table></div>")
            parts.append("".join(out))

        # 8) Parsing rules (JSON)
        sr = data.get("syntax_rules") or {}
        sr_desc = sr.get("description") or ""
        sci_sel = ((sr.get("special_parsing") or {}).get("sci_variant_selection")) or {}
        if sr_desc or sci_sel:
            out = []
            out.append('<div class="help-cat">')
            out.append(f"<h3>{html.escape(str(sec_parsing))}</h3>")
            if sr_desc:
                out.append(f"<div class='minor'>{html.escape(str(sr_desc))}</div>")
            if sci_sel:
                rule = sci_sel.get("rule") or ""
                note = sci_sel.get("note") or ""
                if rule:
                    out.append(f"<div style='margin-top:8px'><b>SCI pending:</b> {html.escape(str(rule))}</div>")
                if note:
                    out.append(f"<div class='minor'>{html.escape(str(note))}</div>")
            out.append("</div>")
            parts.append("".join(out))

        # Deterministic QC footer (keeps toolchain stable)
        try:
            prof = getattr(getattr(self, "gov_state", object()), "active_profile", "Standard") or "Standard"
            parts.append(f"<div class='minor' style='margin-top:10px'>{html.escape(self._qc_footer_for_profile(prof))}</div>")
            try:
                ovs = gov.normalize_qc_overrides(getattr(self.gov_state, 'qc_overrides', {}) or {})
                if ovs:
                    disp = {'clarity':'Clarity','brevity':'Brevity','evidence':'Evidence','empathy':'Empathy','consistency':'Consistency','neutrality':'Neutrality'}
                    parts2 = [f"{disp.get(k,k)}={v}" for k, v in ovs.items()]
                    parts.append(f"<div class='minor'>QC-Overrides: {html.escape(' · '.join(parts2))}</div>")
            except Exception:
                pass
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
        out.append(f"{sysname} v{ver} · Comm: {comm} · Active profile: {prof} · SCI: {sci} · Overlay: {overlay} · Control Layer: {ctl} · QC: {qc} · CGI: {cgi} · Color: {color}")
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


    def _render_comm_state_html(self) -> str:
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
        ctl = getattr(self.gov_state, "control_layer", "on") or "on"
        qc = getattr(self.gov_state, "qc", "on") or "on"
        cgi = getattr(self.gov_state, "cgi", "on") or "on"

        sci_pending = bool(getattr(self.gov_state, "sci_pending", False))
        sci_variant = getattr(self.gov_state, "sci_variant", "") or ""
        sci = "PENDING" if sci_pending else (sci_variant.upper() if sci_variant else "OFF")

        anchor_auto = "on" if bool(getattr(self.gov_state, "anchor_auto", True)) else "off"
        user_turns = int(getattr(self.gov_state, "user_turns", 0) or 0)
        dyn = getattr(self.gov_state, "dynamic_nudge", "") or ""

        status = f"{sysname} v{ver} · Comm: {comm} · Active profile: {prof} · SCI: {sci} · Overlay: {overlay} · Control Layer: {ctl} · QC: {qc} · CGI: {cgi} · Color: {color}"

        rows = [
            ("Comm active", comm),
            ("Active profile", prof),
            ("Overlay", overlay),
            ("SCI", sci),
            ("Control Layer", ctl),
            ("QC", qc),
            ("CGI", cgi),
            ("Color", color),
            ("Anchor auto", anchor_auto),
            ("User turns", str(user_turns)),
        ]
        if dyn:
            rows.append(("Dynamic nudge", dyn))

        out = []
        out.append('<div class="comm-help comm-state">')
        if note_html:
            out.append(note_html)
        out.append(f'<div class="help-status">{html.escape(status)}</div>')
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
        out.append(f"{sysname} v{ver} · Loaded rules file: {fname}")
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

        status = f"{sysname} v{ver} · Loaded rules file: {fname}"

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
        ui_lang = UI_LANG
        ui_lang = UI_LANG

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

        
        
        
        
        try:
            _g = globals().get('gov')
            if _g is not None:
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

        self._connect_api()

        # Visible system notice (does not send a message to the model).
        try:
            if getattr(self, "main_win", None):
                self.main_win.evaluate_js("addMsg('sys', 'Auto: Comm Start · Strict on · Color on.')")
        except Exception:
            pass


    def _get_plain_system_instruction(self) -> str:
        """Minimal system instruction used when Comm-SCI is stopped."""
        return "You are a helpful assistant. Answer in English."

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

        self.chat_session = self.client.chats.create(
            model=current_model,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                temperature=0.0,
                top_p=0.1,
                candidate_count=1,
                max_output_tokens=65536
            )
        )

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
    
    def _connect_api(self):
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


            # --- Stateless provider (OpenRouter) ---
            if provider in ('openrouter', 'huggingface', 'hf', 'openai', 'openai_compat'):
                pr = getattr(self, 'provider_router', None)
                client = None
                try:
                    if pr is not None:
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
                    model = (getattr(cfg, 'get_provider_model', lambda _p: '')(provider) or '').strip()
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
            lang = UI_LANG

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
            try:
                self.main_win.evaluate_js(f"addMsg('sys', 'Selected: {html.escape(base)}')")
            except Exception:
                pass

            # Harte Sperre: Config-Datei ist KEIN Ruleset.
            if base.lower() == "comm-sci-config.json" or base.lower().endswith("-config.json") or base.lower().endswith("config.json"):
                try:
                    self.main_win.evaluate_js(
                        "addMsg('sys', 'JSON ERROR: You selected the configuration file. Please choose a ruleset file (e.g., Comm-SCI-v19.6.8.json).')"
                    )
                except Exception:
                    pass
                start_dir = os.path.dirname(new_file)
                continue  # retry once

            # Versuch laden (Schema-Guard im GovernanceManager)
            success = gov.load_file(new_file)
            if not success:
                try:
                    self.main_win.evaluate_js("addMsg('sys', 'Ruleset NOT loaded (invalid Comm-SCI ruleset).')")
                except Exception:
                    pass
                start_dir = os.path.dirname(new_file)
                if attempt == 0:
                    continue
                return

            # Ab hier: erfolgreich geladen → Session deterministisch neu setzen
            self.gov_state = _init_state_from_rules()

            # 1) API reconnecten (damit System Instructions neu gesetzt werden)
            try:
                self.main_win.evaluate_js(f"addMsg('sys', 'Loading new ruleset: {html.escape(os.path.basename(gov.current_filename))}...')")
            except Exception:
                pass
            self._connect_api()
            self._auto_comm_start('rules-reload')

            # 2) UI im Chatfenster updaten (Dateiname oben)
            try:
                self.main_win.evaluate_js(f"updateRuleFile('{html.escape(os.path.basename(gov.current_filename))}')")
            except Exception:
                pass

            # 3) Panel robust neu aufbauen: altes Panel zerstören & neu erzeugen
            self._rebuild_panel(reason='rules-reload')

            try:
                self.main_win.evaluate_js("addMsg('sys', 'Ruleset loaded and panel updated.')")
            except Exception:
                pass
            return


    def _render_sci_trace_as_html_runtime(self, text_in: str) -> str:
        """Hard-render the SCI Trace section as HTML using the *active* SCI variant step list.

        This is a renderer-only fix to avoid Markdown list runaway numbering and to ensure
        SCI A/B produces a visibly structured block even when the ruleset doesn't expose
        required_steps under global_defaults.
        """
        try:
            if not text_in or 'SCI Trace' not in text_in:
                return text_in

            # Determine required steps from selected SCI variant (A/B/...) -> mapped mode -> steps
            variant = (getattr(getattr(self, 'gov_state', None), 'sci_variant', '') or '').strip().upper()
            sci_active = bool(getattr(getattr(self, 'gov_state', None), 'sci_active', False))
            if not sci_active or not variant:
                return text_in

            try:
                _vdef, steps, _maps_to = self._sci_variant_def(variant)
            except Exception:
                steps = []

            required_steps = [str(s) for s in (steps or []) if str(s).strip()]
            if not required_steps:
                return text_in

            lines = text_in.splitlines()
            sci_idx = None
            for i, ln in enumerate(lines):
                if re.match(r"^\s*SCI\s+Trace\s*:?\s*$", ln) or re.match(r"^\s*SCI\s+Trace\s*:.*$", ln):
                    sci_idx = i
                    break
            if sci_idx is None:
                return text_in

            end_idx = len(lines)
            end_pat = re.compile(r"^\s*(Final\s+Answer\s*:|Self-?Debunking\s*:|QC-?Matrix\s*:)")
            for j in range(sci_idx + 1, len(lines)):
                if end_pat.match(lines[j]):
                    end_idx = j
                    break

            pre = lines[:sci_idx]
            body = lines[sci_idx + 1:end_idx]
            post = lines[end_idx:]

            step_set = {s for s in required_steps}
            hdr_re = re.compile(r"^\s*(?:\d+\.)?\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)$")

            blocks = {}
            cur = None
            buf = []

            def flush():
                nonlocal cur, buf
                if cur is None:
                    return
                cleaned = []
                for x in buf:
                    cleaned.append(re.sub(r"^\s*\d+\.\s+", "", x))
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                blocks[cur] = cleaned
                cur = None
                buf = []

            recognized = 0
            for ln in body:
                m = hdr_re.match(ln)
                if m:
                    name = m.group(1)
                    if name in step_set:
                        flush()
                        cur = name
                        recognized += 1
                        inline = (m.group(2) or '').strip()
                        if inline:
                            buf.append(inline)
                        continue
                if cur is not None:
                    buf.append(ln)
            flush()

            if recognized < 2:
                return text_in

            html_parts = [
                "<div class='sci-trace' style='margin:10px 0; padding:10px; border:1px solid #ddd; border-radius:12px;'>",
                "<div style='font-weight:700; margin-bottom:6px;'>SCI Trace</div>",
                "<ol style='margin:0 0 0 22px; padding:0;'>",
            ]

            for step in required_steps:
                if step not in blocks:
                    continue
                html_parts.append("<li style='margin:4px 0 10px 0;'>")
                html_parts.append(f"<div style='font-weight:700; margin:0 0 4px 0;'>{html.escape(step)}:</div>")
                for ln in (blocks.get(step) or []):
                    t = (ln or '').rstrip('\n')
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
            out_lines.append('SCI Trace:')
            out_lines.append("\n".join(html_parts))
            out_lines.extend(post)
            return "\n".join(out_lines)
        except Exception:
            return text_in

    def _apply_csc_strict(self, raw_response: str, *, user_raw: str, is_command: bool):
        """Wrapper-enforced CSC (strict) with Full Rendering (Ported from Fix7c5-Plus)."""
        
        # --- Nested Helper: Color Spans (Logic from Fix7c5-Plus) ---
        def apply_color_spans(text):
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
            pat = re.compile(r"\[(?P<tag>GREEN|YELLOW|RED|GRAY)(?P<suffix>(?:-[A-Z]+)?)\]\s*(?P<emoji>[🟢🟡🔴⚪️])?")
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
            # 1. Command? -> Nur Markdown Rendering
            if is_command:
                _h = markdown.markdown(raw_response, extensions=['extra', 'codehilite'])
                return sanitize_html(_h), None

            # 2. Comm Inactive? -> Nur Markdown
            if not getattr(self.gov_state, 'comm_active', False):
                _h = markdown.markdown(raw_response, extensions=['extra', 'codehilite'])
                return sanitize_html(_h), None
            
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
                    lang = self._lang()
                    msg = CSC_WARNING_TEXT
                    csc_meta = {
                        "applied": True, "message": msg,
                        "trigger": str(getattr(dec, 'trigger_source', '')),
                        "mode": str(getattr(dec, 'mode', '')),
                        "governance_triggered": bool(getattr(dec, 'governance_triggered', False)),
                        "f_score": int(getattr(dec, 'f_score', 0)),
                        "overlay": overlay
                    }

            # 4. Alerts generieren (Wiederhergestellt aus alter Version)
            alerts = []

            # CSC marker presence check (deterministic): if CSC was triggered/applied, the marker should be visible.
            try:
                if csc_meta and csc_meta.get('applied'):
                    marker = getattr(refiner, 'marker', '') or ''
                    if marker and (marker not in (raw_response or '')):
                        alerts.append(("CSC", f"CSC applied but marker missing in output: {marker}"))
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
                    f"{sysname} v{ver} · Active profile: {prof} · SCI: {sci or 'off'} · Overlay: {overlay or 'off'} · "
                    f"Control Layer: on · QC: on · CGI: on · Color: {disp_color}"
                )

                # Dynamic one-shot marker (canonical JSON requires a visible marker)
                try:
                    if bool(getattr(self.gov_state, 'dynamic_one_shot_active', False)):
                        header += " · Dynamic: one-shot (active)"
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

            # Enforce Self-Debunking contract deterministically (when required by JSON).
            try:
                raw_for_render = enforce_self_debunking_contract(raw_for_render, gov, prof, lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'))
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
            
            # A: Header voranstellen
            if header:
                raw_for_render = header + "\n\n" + raw_for_render
            # Strict enforcement gate (optional): validate final text (pre-render) and optionally warn/block.
            strict_banner_html = ""
            try:
                pol = self._get_enforcement_policy()
            except Exception:
                pol = "audit_only"
            if pol in ("strict_warn", "strict_block"):
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
                except Exception as e:
                    hv2, sv2 = [], []
                    # Fail-soft: show a warning in chat, but never crash.
                    try:
                        self._append_system_message(f"⚠️ QC/Validator error in strict enforcement: {e}")
                    except Exception:
                        pass
                if hv2:
                    if pol == "strict_block":
                        blocked_html = (
                            "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
                            "<summary>⛔ STRICT BLOCK (hard violations)</summary>"
                            "<div class='csc-details'>"
                            "<p>Die Modellantwort wurde vom Wrapper blockiert, weil nach Repair/Enforcement weiterhin harte Regelverstöße vorliegen.</p>"
                            "<ul>"
                            + "".join(f"<li>{html.escape(str(x))}</li>" for x in hv2)
                            + "</ul>"
                            "<p><i>(Content withheld by wrapper)</i></p>"
                            "</div></details>"
                        )
                        try:
                            return ({"html": sanitize_html(blocked_html), "text": "", "csc": None}, {"strict_enforcement": "blocked", "hard_violations": hv2})
                        except Exception:
                            return ({"html": blocked_html, "text": "", "csc": None}, {"strict_enforcement": "blocked", "hard_violations": hv2})
                    else:
                        # strict_warn
                        strict_banner_html = (
                            "<details class='csc-warning' open style='border: 2px solid #c00; background: #fee; color: #600;'>"
                            "<summary>⚠️ RULE VIOLATION DETECTED (strict_warn)</summary>"
                            "<div class='csc-details'>"
                            "<p>Die folgende Antwort hat nach Repair/Enforcement weiterhin harte Regelverstöße:</p>"
                            "<ul>"
                            + "".join(f"<li>{html.escape(str(x))}</li>" for x in hv2)
                            + "</ul>"
                            "</div></details><hr>"
                        )
            # Strict warn: show banner above the normal alerts/content
            if strict_banner_html:
                alert_html = strict_banner_html + alert_html

            
            # B: Bilder einbetten
            raw_for_render = _auto_embed_image_urls(raw_for_render)
            
            # C: Farben anwenden
            if getattr(self.gov_state, 'color', 'off') == 'on':
                raw_for_render = apply_color_spans(raw_for_render)
            
            # D: Markdown Cleanup (Abstände)
            raw_for_render = re.sub(r'(?<!\n)\n([*-]|\d+\.) ', r'\n\n\1 ', raw_for_render)
            raw_for_render = re.sub(r'(?<!\n)\nQC-Matrix:', r'\n\nQC-Matrix:', raw_for_render)
            
            # E: Markdown Render
            try:
                # Versuch 'extra' (Tabellen etc.), Fallback auf Standard
                final_html_body = markdown.markdown(raw_for_render, extensions=['extra', 'codehilite'])
                final_html_body = sanitize_html(final_html_body)
            except:
                final_html_body = markdown.markdown(raw_for_render, extensions=['fenced_code', 'tables'])
                final_html_body = sanitize_html(final_html_body)
            
            # F: Alerts + Body + Timestamp zusammenbauen
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            final_html = alert_html + final_html_body + f'<div class="ts-footer">Response at {timestamp}</div>'

            return final_html, csc_meta

        except Exception as e:
            # Fallback bei schwerem Error
            return f"<span style='color:red'>Runtime Error in Renderer: {e}</span>", None

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
                    lang = getattr(cfg, 'get_answer_language', lambda: 'en')()
                except Exception:
                    lang = 'en'
            lang = (lang or 'en').strip().lower()
            if lang not in ('en', 'de'):
                lang = 'en'

            lang_name = 'English' if lang == 'en' else 'German'

            # Small, explicit wrapper directives.
            lines = []
            lines.append(f"[OUTPUT LANGUAGE] Final answer content in {lang_name} ({lang}). Keep ALL headings/labels/scaffolding in English.")
            lines.append("[ANSWER LENGTH] Be slightly more detailed than minimal (+10-20%). Avoid one-liners.")

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
                            lines[1] = "[ANSWER LENGTH] Be detailed and thorough. Do not compress; include steps/examples when helpful."
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
            lines.append("")
            return "\n".join(lines) + raw
        except Exception:
            return user_raw

    def _csc_wrap_user_message(self, user_raw: str):
        """Deterministic CSC enforcement on the prompt side.

        Returns (text_to_send, pre_csc_meta or None).
        We only use strings/configs from Comm-SCI-v19.6.8.json.
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
            uncertainty_U4 = bool(re.search(r"\bU[4-6]\b", user_raw or ""))
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

            pre_meta = {
                'applied': True,
                'message': CSC_WARNING_TEXT,
                'trigger': str(getattr(dec, 'trigger_source', '')),
                'mode': str(getattr(dec, 'mode', '')),
                'governance_triggered': bool(getattr(dec, 'governance_triggered', False)),
                'f_score': int(getattr(dec, 'f_score', 0)),
                'overlay': overlay,
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
                    "EVIDENCE-LINKER (MANDATORY WHEN COLOR=ON): In the FINAL ANSWER, prefix EACH paragraph or bullet item "
                    "with exactly ONE tag: [GREEN] well-established, [YELLOW] plausible/uncertain, [RED] speculative. "
                    "Do NOT tag headers, SCI Trace, Self-Debunking, or QC-Matrix."
                )
        except Exception:
            evidence = ''

        dont_echo = (
            "DO NOT OUTPUT INTERNAL SCAFFOLDING: Do not write lines like 'Profile: ...' or 'SCI: ...'. "
            "Follow the [CURRENT STATE] above silently."
        )

        parts = [state_line, f"[OUTPUT LANGUAGE] {lang}", dont_echo]
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
            client = None
            try:
                if provider in ('huggingface', 'hf'):
                    if pr is not None and hasattr(pr, 'build_huggingface_client'):
                        client = pr.build_huggingface_client()
                else:
                    if pr is not None and hasattr(pr, 'build_openrouter_client'):
                        client = pr.build_openrouter_client()
            except Exception:
                client = None
            if client is None or not getattr(client, 'api_key', ''):
                # Provider configured but no key found
                pname = 'Hugging Face' if provider in ('huggingface', 'hf') else 'OpenRouter'
                raise RuntimeError(f"{pname} client not configured (missing API key?)")

            # Choose model
            try:
                fallback = str(getattr(cfg, 'get_model', lambda: '')() or '')
            except Exception:
                fallback = ''
            prov_id = 'huggingface' if provider in ('huggingface','hf') else 'openrouter'
            model = (model_override or self._provider_model(prov_id, fallback_model=fallback) or '').strip()
            if not model:
                # Optional: auto-pick first model from cached /models list (best-effort)
                try:
                    models, _meta = (pr.get_openrouter_models_cached(force_refresh=False) if pr is not None and hasattr(pr,'get_openrouter_models_cached') else ([], {}))
                    if provider in ('huggingface','hf') and (not models):
                        try:
                            models = pr.get_huggingface_models_from_config() if pr is not None and hasattr(pr,'get_huggingface_models_from_config') else []
                        except Exception:
                            models = []
                    if models:
                        model = str(models[0]).strip()
                except Exception:
                    model = ''
            if not model:
                model = 'zai-org/GLM-4.7:cerebras' if provider in ('huggingface','hf') else 'openai/gpt-4.1-mini'

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
                provs = (self.cfg_mgr.config or {}).get('providers') or {}
                pconf = provs.get(provider) if isinstance(provs, dict) else {}
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
            cand = cand[:5] if isinstance(cand, list) else [model]

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


    def _render_profile_switch_control_html(self, timestamp: str) -> str:


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
            f"{sysname} v{ver} · Active profile: {prof} · SCI: {sci or 'off'} · Overlay: {overlay or 'off'} · "
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
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            raw_txt = txt or ""

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

            # 1. Routing
            route = route_input(raw_txt, self.gov_state, self)

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
                return {"html": route["html"], "csc": None}

            # SCI Selection (A-H)
            if route.get("is_sci_selection"):
                self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                try:
                    self.log_event('sci_selection', {'value': _safe_preview_text(route.get('query_text') or '', 16)})
                except Exception:
                    pass
                return self._handle_sci_selection(route["query_text"])

            if route["kind"] == "noop":
                return {"html": "", "csc": None}

            # 2. Commands
            if route["kind"] == "command":
                self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})
                cmd = route["canonical_cmd"]

                try:
                    self.log_event('command', {'cmd': cmd, 'phase': 'begin'})
                except Exception:
                    pass

                # Special Renderers (help/state/config/audit/anchor etc.)
                handled_res = self._handle_command_deterministic(cmd, timestamp)
                if handled_res:
                    try:
                        self.log_event('command', {'cmd': cmd, 'phase': 'deterministic'})
                    except Exception:
                        pass
                    return handled_res

                # State Change
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

                if cmd.startswith("Profile "):
                    html_content = self._render_profile_switch_control_html(timestamp)
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
                        html_content = self._render_comm_state_html()

                self.history.append({"role": "bot", "content": f"Command executed: {cmd}", "ts": datetime.now().isoformat()})
                try:
                    self.log_event('command', {'cmd': cmd, 'phase': 'end', 'profile': getattr(self.gov_state, 'active_profile', '')})
                except Exception:
                    pass
                return {"html": html_content, "csc": None}

            # 3. Chat (Normal Question)
            self.history.append({"role": "user", "content": raw_txt, "ts": datetime.now().isoformat()})

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
                        note = (
                            "<div style='border:1px solid #c7d2fe; background:#eef2ff; padding:10px; "
                            "border-radius:10px; margin:8px 0; color:#1e3a8a;'>"
                            "<b>SCI selection pending.</b><br>"
                            "Your input looks like a question about the SCI methodology. "
                            "Please select an SCI variant (A–H) to continue."
                            "</div>"
                        )
                        menu_html = self._render_sci_menu_html(lang=self._lang())
                        return {"html": note + "\n" + menu_html, "csc": None}

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
                        ts = datetime.now().isoformat()
                        warn = (
                            "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                            "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                            "<b>CONTROL LAYER BLOCK:</b><br>" + html.escape(str(msg)) +
                                        "<br><span style='font-size:12px; color:#7f1d1d;'>Retry after " + html.escape(str(retry_s)) + "s</span>" +
                            "<br><span style='font-size:12px; color:#7f1d1d;'>"
                            "Tip: adjust limits in Config/Comm-SCI-Config.json (rate_limit_per_minute / rate_limit_per_hour; optional rate_limit_scopes)"
                            "</span></div>"
                        )
                        return {"html": warn + f"<div class='ts-footer'>Response at {html.escape(ts)}</div>", "csc": None}
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
            governance_enabled_now = bool(getattr(self, 'session_with_governance', True))
            try:
                if not governance_enabled_now:
                    raise RuntimeError('governance disabled')
                _prof_now = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
                repaired_raw = enforce_qc_footer_deltas(repaired_raw, gov, _prof_now)
            except Exception:
                pass
            try:
                repaired_raw = normalize_evidence_tags(repaired_raw)
            except Exception:
                pass
            try:
                if not governance_enabled_now:
                    raise RuntimeError('governance disabled')
                _prof_now = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
                repaired_raw = enforce_self_debunking_contract(repaired_raw, gov, _prof_now, is_command=False, lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'))
            except Exception:
                pass
            try:
                repaired_raw = normalize_sci_trace_numbering(repaired_raw, gov)
            except Exception:
                pass

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
                                    ts = datetime.now().isoformat()
                                    warn = (
                                        "<div style='border:1px solid #fca5a5; background:#fef2f2; padding:10px; "
                                        "border-radius:10px; margin:8px 0; color:#991b1b;'>"
                                        "<b>CONTROL LAYER BLOCK:</b><br>" + html.escape(str(msg)) +
                                        "<br><span style='font-size:12px; color:#7f1d1d;'>"
                                        "Tip: adjust limits in Config/Comm-SCI-Config.json (rate_limit_per_minute / rate_limit_per_hour; optional rate_limit_scopes)"
                                        "</span></div>"
                                    )
                                    return {"html": warn + f"<div class='ts-footer'>Response at {html.escape(ts)}</div>", "csc": None}
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
                        try:
                            if not governance_enabled_now:
                                raise RuntimeError('governance disabled')
                            _prof_now = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
                            repaired_raw = enforce_qc_footer_deltas(repaired_raw, gov, _prof_now)
                        except Exception:
                            pass
                        try:
                            repaired_raw = normalize_evidence_tags(repaired_raw)
                        except Exception:
                            pass
                        try:
                            _prof_now = getattr(self.gov_state, 'active_profile', 'Standard') or 'Standard'
                            repaired_raw = enforce_self_debunking_contract(repaired_raw, gov, _prof_now, is_command=False, lang=getattr(getattr(self, 'gov_state', None), 'answer_language', 'de'))
                        except Exception:
                            pass
                        try:
                            repaired_raw = normalize_sci_trace_numbering(repaired_raw, gov)
                        except Exception:
                            pass

                        # Banner (visible; does not claim perfection beyond one pass)
                        try:
                            items = "".join([f"<li>{html.escape(str(v))}</li>" for v in hard_vios])
                            repair_banner_html = (
                                "<div style='border:1px solid #f59e0b; background:#fffbeb; padding:10px; "
                                "border-radius:10px; margin:8px 0; color:#92400e;'>"
                                "<b>CONTROL LAYER NOTE</b><br>One repair pass was applied for hard contract violations."
                                f"<ul style='margin:6px 0 0 18px; padding:0;'>{items}</ul></div>"
                            )
                        except Exception:
                            repair_banner_html = (
                                "<div style='border:1px solid #f59e0b; background:#fffbeb; padding:10px; "
                                "border-radius:10px; margin:8px 0; color:#92400e;'>"
                                "<b>CONTROL LAYER NOTE</b><br>One repair pass was applied for hard contract violations."
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

            return {"html": final, "csc": safe_meta}

        except Exception as e:
            # Always persist a bot entry so exported logs are complete.
            try:
                err_html = _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error')
                self.history.append({"role": "bot", "content": err_html, "ts": datetime.now().isoformat(), "csc": None})
            except Exception:
                err_html = _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error')
            return {"html": err_html, "csc": None}
    
    def update_stats_ui(self):
        if self.main_win:
            reqs = int(getattr(self, 'session_req_count', 0) or 0)
            tin = int(getattr(self, 'session_tokens_in', 0) or 0)
            tout = int(getattr(self, 'session_tokens_out', 0) or 0)
            stats_txt = f"Reqs: {reqs} | In: {tin} | Out: {tout}"
            self.main_win.evaluate_js(f"updateStats('{stats_txt}')")


    def remote_cmd(self, cmd):
        """Inject a command into the main UI input and trigger send() via JS."""
        if not getattr(self, 'main_win', None):
            return {'ok': False, 'error': 'no_main_win'}
        try:
            safe = json.dumps(str(cmd or ''), ensure_ascii=False)
            self.main_win.evaluate_js(f"remoteInput({safe});")
            return {'ok': True}
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

        try:
            if action_s == 'cmd':
                # Execute via main window pipeline so results appear in the chat UI.
                text = payload.get('text', '')
                try:
                    if hasattr(self, 'remote_cmd'):
                        self.remote_cmd(str(text or ''))
                        return {'ok': True, 'action': 'cmd', 'queued': True}
                except Exception as e:
                    return {'ok': False, 'action': 'cmd', 'error': str(e)}
                return {'ok': False, 'action': 'cmd', 'error': 'remote_cmd_unavailable'}
            if action_s == 'set_provider':
                provider = payload.get('provider', '')
                return {'ok': True, 'action': action_s, 'result': self.set_provider(str(provider or ''))}
            if action_s == 'set_model':
                model = payload.get('model', '')
                return {'ok': True, 'action': action_s, 'result': self.set_model(str(model or ''))}
            if action_s == 'set_answer_language':
                lang = payload.get('lang', '')
                return {'ok': True, 'action': action_s, 'result': self.set_answer_language(str(lang or ''))}
            if action_s == 'refresh_models':
                provider = payload.get('provider', '')
                try:
                    p = (str(provider or '')).strip().lower()
                except Exception:
                    p = ''
                if p:
                    try:
                        curp = (self.cfg.get_active_provider() or 'gemini').strip().lower()
                    except Exception:
                        curp = 'gemini'
                    if p != curp:
                        try:
                            self.set_provider(p)
                        except Exception:
                            pass
                return {'ok': True, 'action': action_s, 'result': self.refresh_models()}
            if action_s == 'hf_catalog':
                top_n = payload.get('top_n', 200)
                provider_filter = payload.get('provider_filter', 'all')
                force_refresh = bool(payload.get('force_refresh', False))
                return {'ok': True, 'action': action_s,
                        'result': self.hf_catalog(top_n=int(top_n or 200),
                                                  provider_filter=str(provider_filter or 'all'),
                                                  force_refresh=force_refresh)}
            if action_s == 'list_chat_logs':
                limit = payload.get('limit', 200)
                lr = self.list_chat_logs(limit=int(limit or 200))
                logs = []
                try:
                    if isinstance(lr, dict):
                        logs = lr.get('logs') or []
                    elif isinstance(lr, list):
                        logs = lr
                except Exception:
                    logs = []
                if not isinstance(logs, list):
                    logs = []
                return {'ok': True, 'action': action_s, 'logs': logs}
            if action_s == 'load_chat_log':
                name = payload.get('name', '')
                fork = bool(payload.get('fork', True))
                res = self.load_chat_log(str(name or ''), fork=fork)
                try:
                    if isinstance(res, dict) and res.get('ok') is True:
                        self._ui_replay_loaded_history(status_msg=f"Loaded: {os.path.basename(str(name or ''))} ({'fork' if fork else 'no fork'})")
                except Exception:
                    pass
                return {'ok': True, 'action': action_s, **(res or {})}
            if action_s == 'clear_chat':
                res = self.clear_chat()
                try:
                    if isinstance(res, dict):
                        ok = bool(res.get('ok', True))
                    else:
                        ok = True
                except Exception:
                    ok = True
                return {'ok': ok, 'action': action_s, **(res or {})}

        except Exception as e:
            try:
                return {'ok': False, 'action': action_s, 'error': str(e)}
            except Exception:
                return {'ok': False, 'action': action_s, 'error': 'error'}
        return {'ok': False, 'action': action_s, 'error': 'unknown action'}

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
        data = {
            'providers': ['gemini', 'openrouter', 'huggingface'],
            'current_provider': 'gemini',
            'current_model': 'gemini-2.0-flash',
            'available_models': ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-pro'],
            'answer_language': 'de',
            'comm': [],
            'profiles': [],
            'sci': [],
            'overlays': [],
            'tools': [],
            'logs': [],
            'chat_logs': [],
            'model_hint': '',
        }

        # Provider
        try:
            pr = getattr(self, 'provider_router', None)
            if pr is not None and hasattr(pr, 'get_active_provider'):
                cp = (pr.get_active_provider() or 'gemini').strip().lower()
            else:
                cp = 'gemini'
            if cp:
                data['current_provider'] = cp
        except Exception:
            pass

        # Model
        try:
            cfg_obj = globals().get('cfg')
            if cfg_obj is not None and hasattr(cfg_obj, 'get_provider_model'):
                cm = (cfg_obj.get_provider_model(data['current_provider']) or '').strip()
                if cm:
                    data['current_model'] = cm
        except Exception:
            pass

        # Model lists (offline-fast): use in-memory caches warmed from disk; no network here.
        try:
            curp = data.get('current_provider', 'gemini')
            if curp == 'gemini':
                data['available_models'] = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-pro']
            elif curp in ('openrouter', 'huggingface'):
                data['available_models'] = self.get_available_models(curp)
        except Exception:
            pass

        # Answer language
        try:
            cfg_obj = globals().get('cfg')
            if cfg_obj is not None and hasattr(cfg_obj, 'get_answer_language'):
                al = (cfg_obj.get_answer_language() or 'de').strip().lower()
                if al in ('de', 'en'):
                    data['answer_language'] = al
        except Exception:
            pass

        # Merge richer command/button schema when governance runtime is available
        try:
            gov_obj = getattr(self, 'gov', None) or globals().get('gov')
            if gov_obj is not None and hasattr(gov_obj, 'get_ui_data'):
                ui = gov_obj.get_ui_data() or {}
                if isinstance(ui, dict):
                    for k in ('comm', 'profiles', 'sci', 'overlays', 'tools', 'logs'):
                        v = ui.get(k)
                        if isinstance(v, list):
                            data[k] = v
                    # Optional keys (best-effort)
                    for k in ('current_rule_file', 'version', 'loaded'):
                        if k in ui:
                            data[k] = ui.get(k)
                    if isinstance(ui.get('answer_language'), str):
                        data['answer_language'] = ui.get('answer_language')
        except Exception:
            pass

        # Local chat log listing (for loader UI)
        try:
            res = self.list_chat_logs(limit=200)
            if isinstance(res, dict) and res.get('ok') is True:
                logs = res.get('logs')
                if isinstance(logs, list):
                    data['chat_logs'] = logs
                    if logs:
                        data['chat_log_selected'] = logs[0]
        except Exception:
            pass

        # Backward-compat aliases for older Panel JS builds
        data['provider'] = data.get('current_provider', 'gemini')
        data['model'] = data.get('current_model', 'gemini-2.0-flash')
        return data

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
        except Exception:
            return

    def get_available_models(self, provider: str):
        """Return cached/known models for provider. Must be fast and never do network I/O."""
        try:
            p = (provider or '').strip().lower()
            if p == 'gemini':
                # Keep stable, small default list (non-authoritative). User can type a model manually.
                return [
                    'gemini-2.0-flash',
                    'gemini-2.5-flash',
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
            p = pathlib.Path(base)
            if not p.exists() or not p.is_dir():
                return {'ok': True, 'logs': []}
            files = []
            for f in p.iterdir():
                if f.is_file() and f.name.lower().endswith('.json'):
                    files.append(f.name)
            # sort by name descending (timestamps in filename)
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

            import os
            base_abs = os.path.abspath(base)
            candidate = os.path.abspath(os.path.join(base_abs, os.path.basename(name)))
            if not candidate.startswith(base_abs):
                return {'ok': False, 'error': 'path_traversal_blocked'}
            if not os.path.exists(candidate):
                return {'ok': False, 'error': 'file_not_found'}

            # Prefer bound method on self (unit tests), else call module helper.
            if hasattr(self, 'load_log_from_path'):
                return self.load_log_from_path(candidate, fork=bool(fork))
            return {'ok': False, 'error': 'load_log_from_path_unavailable'}
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}



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
            provider = (provider or 'gemini').strip().lower()
            if provider in ('hf',):
                provider = 'huggingface'
            if provider not in ('gemini', 'openrouter', 'huggingface'):
                provider = 'gemini'
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
            try:
                if self.main_win:
                    self.main_win.evaluate_js(f"addMsg('sys', 'Active provider set to: {provider}.')")
            except Exception:
                pass

            # Reconnect only for Gemini (session-based)
            if provider == 'gemini':
                self._trigger_reconnect(f"Providerwechsel (Gemini)...")
            else:
                # For stateless providers, just refresh panel
                try:
                    if self.panel_win:
                        self.panel_win.evaluate_js('window.refresh_panel && window.refresh_panel()')
                except Exception:
                    pass
        except Exception:
            pass

    def refresh_models(self):
        """Refresh provider model list cache (OpenRouter/Hugging Face best-effort).

        - OpenRouter: refresh cached /models list.
        - Hugging Face: tries /models; if unavailable, keeps config-defined list.
        """
        try:
            pr = getattr(self, 'provider_router', None)
            curp = (pr.get_active_provider() if pr is not None and hasattr(pr, 'get_active_provider') else 'gemini')
            curp = (curp or 'gemini').strip().lower()

            if curp == 'openrouter':
                models, meta = pr.get_openrouter_models_cached(force_refresh=True) if pr is not None and hasattr(pr, 'get_openrouter_models_cached') else ([], {})
                try:
                    self._openrouter_models_cache = list(models) if isinstance(models, list) else []
                except Exception:
                    pass
                try:
                    if self.main_win:
                        self.main_win.evaluate_js(f"addMsg('sys', 'OpenRouter models refreshed: {len(models)} (source: {meta.get('source','?')}).')")
                except Exception:
                    pass
                return {'status': True, 'provider': 'openrouter', 'count': len(models), 'meta': meta}

            if curp in ('huggingface', 'hf'):
                models = []
                meta = {'source': 'none'}
                try:
                    if pr is not None and hasattr(pr, 'get_huggingface_models_cached'):
                        models, meta = pr.get_huggingface_models_cached(force_refresh=True)
                except Exception:
                    models = []
                    meta = {'source': 'none'}
                # Cache models for get_ui() / panel dropdown
                try:
                    self._hf_models_cache = list(models) if isinstance(models, list) else []
                except Exception:
                    pass
                try:
                    if getattr(self, 'panel_win', None):
                        self.panel_win.evaluate_js('window.refresh_panel && window.refresh_panel()')
                except Exception:
                    pass
                # UI notice
                try:
                    if self.main_win:
                        self.main_win.evaluate_js(
                            f"addMsg('sys', 'Hugging Face models refreshed: {len(models)} (source: {meta.get('source','?')}).')"
                        )
                except Exception:
                    pass
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
            try:
                if self.main_win:
                    self.main_win.evaluate_js(f"addMsg('sys', 'Model set to: {model} (provider: {provider}).')")
            except Exception:
                pass
            try:
                if self.panel_win:
                    self.panel_win.evaluate_js('window.refresh_panel && window.refresh_panel()')
            except Exception:
                pass

    def set_answer_language(self, lang: str):
        """Set desired language for the LLM answer content only (en/de).

        All deterministic UI renderers (help/state/config/SCI/header/footer/QC) remain English.
        The preference is enforced via a small wrapper directive added to the next user message.
        """
        try:
            lang = (lang or 'en').strip().lower()
            if lang not in ('en', 'de'):
                lang = 'en'
            try:
                self.gov_state.answer_language = lang
            except Exception:
                pass
            try:
                if hasattr(cfg, 'set_answer_language'):
                    cfg.set_answer_language(lang)
            except Exception:
                pass
            try:
                if self.main_win:
                    self.main_win.evaluate_js(f"addMsg('sys', 'Answer language (LLM) set to: {lang}.')")
            except Exception:
                pass
            try:
                if self.panel_win:
                    self.panel_win.evaluate_js('window.refresh_panel && window.refresh_panel()')
            except Exception:
                pass
        except Exception:
            pass

    def _trigger_reconnect(self, msg):
        self.ready_status = {"status": False, "msg": msg}
        if self.main_win:
            self.main_win.evaluate_js(f"addMsg('sys', '{msg} Restarting session...')")
        threading.Thread(target=self._reconnect_bg).start()

    def _reconnect_bg(self):
        self._connect_api()

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


    def _create_panel(self):
        # Geometry: prefer persisted config; fallback to current defaults
        geom = self.panel_geom or {}
        def _safe_int(v, default):
            try:
                return int(v)
            except Exception:
                return int(default)

        panel_x = _safe_int(geom.get('x', 1100), 1100)
        panel_y = _safe_int(geom.get('y', 0), 0)
        panel_w = _safe_int(geom.get('width', 340), 340)
        panel_h = _safe_int(geom.get('height', 1000), 1000)

        # macOS/pywebview: a persisted off-screen position makes the panel look 'missing'.
        # Keep values in a sane corridor; otherwise reset to defaults near top-left.
        if panel_w < 250:
            panel_w = 250
        if panel_h < 300:
            panel_h = 300
        if panel_x < 0 or panel_x > 5000:
            panel_x = 50
        if panel_y < 0 or panel_y > 3000:
            panel_y = 50

        # Panel window must receive the same js_api object as the main window.
        # (Secondary windows can otherwise miss methods like get_ui/ping on some backends.)
        kwargs = dict(
            title=PANEL_WINDOW_TITLE,
            html=HTML_PANEL,
            js_api=(self.panel_bridge or self),
            width=panel_w,
            height=panel_h,
            on_top=False
        )

        # Only set x/y if we have something sensible
        if panel_x is not None and panel_y is not None:
            kwargs.update(dict(x=panel_x, y=panel_y))

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
            self.qc_win = webview.create_window(
                "QC-Vorgaben temporär anpassen – Profil: ?",
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
            win = getattr(self, 'qc_win', None)
            if win is None:
                self._create_qc_override()
                win = getattr(self, 'qc_win', None)
            if win is None:
                return {'ok': False, 'error': 'qc_win unavailable'}
            try:
                win.show()
            except Exception:
                pass
            try:
                win.bring_to_front()
            except Exception:
                pass
            return {'ok': True}
        except Exception as e:
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
                    setattr(gov_obj, 'runtime_state', self.gov_state)
            except Exception:
                pass

            msg_parts = []
            disp = {'clarity':'Clarity','brevity':'Brevity','evidence':'Evidence','empathy':'Empathy','consistency':'Consistency','neutrality':'Neutrality'}
            for key in ['clarity','brevity','evidence','empathy','consistency','neutrality']:
                if key in clean:
                    msg_parts.append(f"{disp.get(key, key)}={clean[key]}")
            msg = "QC-Overrides gesetzt: " + (", ".join(msg_parts) if msg_parts else "(leer)")

            try:
                self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
            except Exception:
                pass

            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    js_msg = _json.dumps(msg, ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass

            try:
                if getattr(self, 'qc_win', None) is not None:
                    try:
                        self.qc_win.hide()
                    except Exception:
                        pass
            except Exception:
                pass

            return {'ok': True, 'overrides': clean}
        except Exception as e:
            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    js_msg = _json.dumps(f"[WARN] QC Override Apply failed: {type(e).__name__}: {e}", ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_override_clear(self, _payload=None):
        """Clear QC overrides."""
        try:
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
            msg = "QC-Overrides zurückgesetzt"
            try:
                self.history.append({'role': 'sys', 'content': msg, 'ts': datetime.now().isoformat()})
            except Exception:
                pass
            try:
                if getattr(self, 'main_win', None) is not None:
                    import json as _json
                    js_msg = _json.dumps(msg, ensure_ascii=False)
                    self.main_win.evaluate_js(f"addMsg('sys', {js_msg});")
            except Exception:
                pass
            return {'ok': True}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def qc_override_cancel(self, _payload=None):
        """Close QC dialog without changes."""
        try:
            try:
                if getattr(self, 'qc_win', None) is not None:
                    self.qc_win.hide()
            except Exception:
                pass
            return {'ok': True}
        except Exception as e:
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

        def _safe_int(v, default):
            try:
                return int(v)
            except Exception:
                return int(default)
        panel_x = _safe_int(geom.get('x', 1100), 1100)
        panel_y = _safe_int(geom.get('y', 0), 0)
        panel_w = _safe_int(geom.get('width', 340), 340)
        panel_h = _safe_int(geom.get('height', 1000), 1000)

        # macOS/pywebview: a persisted off-screen position makes the panel look 'missing'.
        # Keep values in a sane corridor; otherwise reset to defaults near top-left.
        if panel_w < 250: panel_w = 250
        if panel_h < 300: panel_h = 300
        if panel_x < 0 or panel_x > 5000: panel_x = 50
        if panel_y < 0 or panel_y > 3000: panel_y = 50

        # Panel window must receive the same js_api object as the main window.
        # (Secondary windows can otherwise miss methods like get_ui/ping on some backends.)
        kwargs = dict(
            title=PANEL_WINDOW_TITLE,
            html=HTML_PANEL,
            js_api=(self.panel_bridge or self),
            width=panel_w,
            height=panel_h,
            on_top=False
        )

        # Only set x/y if we have something sensible
        if panel_x is not None and panel_y is not None:
            kwargs.update(dict(x=panel_x, y=panel_y))

        self.panel_win = webview.create_window(**kwargs)
        self.panel_hidden = False
        self._bind_panel_window_events(self.panel_win)

    def _rebuild_panel(self, reason: str = "reload"):
        """Robust panel rebuild.

        Some pywebview backends (macOS in particular) can get into a weird state after multiple
        evaluate_js refreshes. Recreating the panel window is the most reliable fix.
        Preserves the user's last panel position/size when possible.
        """
        try:
            # remember geometry before destroying
            if self.panel_win:
                self._remember_window_geom(self.panel_win, "panel")
        except Exception:
            pass

        # Try to destroy existing panel window
        try:
            if self.panel_win:
                self.panel_win.destroy()
        except Exception:
            pass

        self.panel_win = None
        try:
            self.panel_bridge = PanelBridge(self)
        except Exception:
            self.panel_bridge = None
        self.panel_hidden = False

        # Recreate and bring to front
        try:
            self._create_panel()
            # best-effort focus
            try:
                if hasattr(self.panel_win, "focus"):
                    self.panel_win.focus()
                if hasattr(self.panel_win, "restore"):
                    self.panel_win.restore()
            except Exception:
                pass
            if self.main_win:
                self.main_win.evaluate_js(f"addMsg('sys', 'Panel rebuilt ({reason}).')")
        except Exception as e:
            if self.main_win:
                safe = str(e).replace("'", "'").replace('"', '\"')
                self.main_win.evaluate_js(f"addMsg('sys', 'Panel rebuild failed: {safe}')")

    def _hide_panel(self):
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

    def _show_panel(self):
        if not self.panel_win:
            self._create_panel()
            return
        try:
            if hasattr(self.panel_win, "show"):
                self.panel_win.show()
            if hasattr(self.panel_win, "restore"):
                self.panel_win.restore()
            if hasattr(self.panel_win, "focus"):
                self.panel_win.focus()
        except Exception:
            pass
        self.panel_hidden = False

    def ensure_panel_visible(self):
        """Called from JS once the main UI is ready: show the panel automatically."""
        try:
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
        if not self.panel_win:
            self._create_panel()
            return
        # If currently hidden/minimized -> show; else hide
        if self.panel_hidden:
            self._show_panel()
        else:
            self._hide_panel()

    def on_panel_closing(self):
        """Intercept the panel close action ("X") and hide instead of destroying when possible."""
        try:
            self._hide_panel()
        except Exception:
            # best-effort hide
            try:
                if self.panel_win and hasattr(self.panel_win, "hide"):
                    self.panel_win.hide()
                    self.panel_hidden = True
            except Exception:
                pass
        # Returning False cancels the close on backends that support it (best-effort).
        return False

    def _bind_panel_window_events(self, win):
        """Bind panel lifecycle events defensively.

        - If the backend supports a cancelable 'closing' event, we hide the panel (keeps state, avoids destroy).
        - Always bind 'closed' as a fallback cleanup if the window is destroyed anyway.
        """
        if not win:
            return
        evs = getattr(win, "events", None)

        closing_ev = getattr(evs, "closing", None)
        if closing_ev is not None:
            try:
                closing_ev += self.on_panel_closing
            except Exception:
                pass

        closed_ev = getattr(evs, "closed", None)
        if closed_ev is not None:
            try:
                closed_ev += self.on_panel_closed
            except Exception:
                pass

    def on_panel_closed(self):
        # remember last geometry if possible
        try:
            self._remember_window_geom(self.panel_win, "panel")
        except Exception:
            pass
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
                with open(chat_path, "w", encoding="utf-8") as f:
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
                    json.dump(_chat_payload, f, indent=2)
                print(f"Exportiert (Chat): {chat_path}")
            except Exception as e:
                print(f"[System] Export-Error (Chat): {e}")

        # Audit-Log (System-/Loader-Logs)
        audit_name = f"Audit_{ts}.json"
        audit_path = os.path.join(AUDIT_LOG_DIR, audit_name)
        try:
            payload = {
                "meta": WRAPPER_NAME,
                "ts": datetime.now().isoformat(),
                "model": cfg_get_model(),
                "ruleset": os.path.basename(getattr(gov, "current_filename", "") or ""),
                "governance_logs": list(getattr(gov, "logs", []) or []),
            }
            if isinstance(audit_event, dict) and audit_event:
                payload["audit_event"] = audit_event
            if isinstance(extra_audit, dict) and extra_audit:
                payload["extra_audit"] = extra_audit
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Exportiert (Audit): {audit_path}")
        except Exception as e:
            print(f"[System] Export-Error (Audit): {e}")

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


    def export_audit_v2(self, *, audit_event=None, audit_only: bool = False):
        """Enhanced audit export (v2). Keeps legacy export() untouched."""
        import platform
        import sys
        import hashlib

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
            },
            'conversation': getattr(self, 'history', []) or [],
            'governance_logs_tail': (getattr(gov, 'logs', []) or [])[-50:],
            'session_events': getattr(self, 'session_events', []) or [],
        }

        if audit_event:
            payload['audit_event'] = audit_event

        # Write audit file
        audit_path = os.path.join(AUDIT_LOG_DIR, f"Audit_{ts}.json")
        try:
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
                os.makedirs(CHAT_LOG_DIR, exist_ok=True)
                with open(chat_path, 'w', encoding='utf-8') as f:
                    json.dump({'meta': WRAPPER_NAME, 'model': cfg_get_model(), 'history': getattr(self, 'history', []) or []}, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Export] Chat write failed: {e}")

        return chat_path, audit_path



def set_api_key_for_provider(self, provider: str, api_key: str, *, persist: bool = True, write_path: str = ""):
    """B6: Persist an API key for a provider (NO secrets in audit exports).
    Stores to the standard keys file schema: {"providers": {"gemini": {"api_key_plain": "..."} } }.
    """
    try:
        p = (provider or '').strip().lower()
        if not p:
            return {'ok': False, 'error': 'provider_missing'}
        if api_key is None:
            api_key = ''
        api_key = str(api_key)

        # Determine path
        target = write_path or os.path.join(CONFIG_DIR, 'Comm-SCI-API-Keys.json')
        os.makedirs(os.path.dirname(target), exist_ok=True)

        data = {}
        if os.path.exists(target):
            try:
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

        # Store as plain by default (MVP), but keep encryption-ready fields intact if present.
        entry['api_key_plain'] = api_key
        # Do NOT write secrets anywhere else (no audit, no logs).

        if persist:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

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
                    self.fork_parent_trace_id = data.get('trace_id') or data.get('session_id') or data.get('meta_trace_id')
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
        # Available models list
        try:
            models = []
            if curp == 'gemini':
                models = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-3-flash', 'gemini-1.5-pro']
            elif curp == 'openrouter':
                pr = getattr(self, 'provider_router', None)
                if pr is not None and hasattr(pr, 'get_openrouter_models_cached'):
                    models, meta = pr.get_openrouter_models_cached(force_refresh=False)
                    data['openrouter_models_meta'] = meta
                else:
                    models = []
            elif curp == 'huggingface':
                pr = getattr(self, 'provider_router', None)
                models = []
                # UI controls for HF catalog
                data['hf_provider_filter_options'] = ['all', 'zai-org', 'novita', 'cerebras', 'together', 'groq', 'fireworks', 'sambanova', 'hyperbolic', 'hf-inference']
                data['hf_catalog_default_top_n'] = int(getattr(self, 'hf_catalog_top_n', 200) or 200)
                data['hf_catalog_default_provider_filter'] = (getattr(self, 'hf_catalog_provider_filter', 'all') or 'all')

                # 1) Prefer HF Hub catalog (cached) when available (default: Top 200, all providers)
                try:
                    if pr is not None and hasattr(pr, 'get_huggingface_catalog_cached'):
                        cat_models, cat_meta = pr.get_huggingface_catalog_cached(top_n=int(getattr(self, 'hf_catalog_top_n', int(data.get('hf_catalog_default_top_n', 200) or 200)) or 200),
                                                                               provider_filter=(getattr(self, 'hf_catalog_provider_filter', 'all') or 'all'),
                                                                               force_refresh=False)
                        if cat_models:
                            models = cat_models
                            data['huggingface_catalog_meta'] = cat_meta
                except Exception:
                    pass

                # 2) Otherwise: HF router /models cache (may be unavailable)
                if not models:
                    meta = {'source': 'none'}
                    try:
                        if pr is not None and hasattr(pr, 'get_huggingface_models_cached'):
                            models, meta = pr.get_huggingface_models_cached(force_refresh=False)
                            data['huggingface_models_meta'] = meta
                    except Exception:
                        models = []

                # 3) Fallback: configured HF models list
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
        if self.main_win:
            safe = cmd.replace("'", "\'").replace('"', '\\"')
            self.main_win.evaluate_js(f"remoteInput('{safe}')")



# ----------------------------

# ----------------------------
# Deterministic command helpers (EN-only)
# These are kept at module scope and then bound into Api via the fixup loop.
# ----------------------------


def _control_layer_alert_html(message: str, *, title: str = "CONTROL LAYER ALERT", severity: str = "error") -> str:
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
            "Tipp: Anderes :free‑Modell wählen</a>"
        )

    try:
        t = html.escape(str(title or "CONTROL LAYER ALERT"))
    except Exception:
        t = "CONTROL LAYER ALERT"
    return (
        f"<details class='csc-warning' open style='{style}'>"
        f"<summary>⚠️ {t}</summary>"
        f"<div class='csc-details'>{safe}{action_html}</div>"
        f"</details>"
    )


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
            return _control_layer_alert_html(str(e), title='CONTROL LAYER ERROR', severity='error')


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

        rows = []
        for i, msg_obj in enumerate(sample, 1):
            txt = (msg_obj or {}).get('content', '') or ''
            vios = []

            # QC footer present?
            if 'QC-Matrix:' not in txt and 'QC:' not in txt:
                vios.append('Missing QC footer')

            # Self-Debunking required?
            try:
                prof_now = getattr(getattr(self, 'gov_state', None), 'active_profile', '') or 'Standard'
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
                if vk:
                    if 'SCI Trace' not in txt:
                        vios.append('Missing SCI Trace block')
            except Exception:
                pass

            status = '✓ Compliant' if not vios else '⚠ ' + '; '.join(vios)
            rows.append((i, status))

        # Render
        msg = "Audit exported."
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
            f"<b>Comm Audit</b><br>{msg}{detail}{last_line}{tbl}</div>"
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
            self.history.append({"role": "bot", "content": "SCI menu", "ts": datetime.now().isoformat(), "csc": None})
        except Exception:
            pass

        return {"html": html_content, "csc": None}

    return None

# Api bridge (pywebview js_api)
# NOTE: In this ENONLY build, Api is the concrete js_api object.


# ----------------------------
# PROVIDER ADAPTERS (single-file)
# ----------------------------


def _openrouter_friendly_http_error(status_code: int, raw_body: str, *, lang: str = "de", tz: str = "Europe/Berlin") -> str:
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



    def _ui_replay_loaded_history(self, status_msg: str = "Loaded chat log."):
        """Rebuild main chat UI from self.history without calling the model.

        Primary path uses the built-in JS helper window.resetChatFromHistory(history, statusMsg).
        If that is unavailable or fails (e.g. very large payloads), falls back to a safe
        incremental replay (reset status + addMsg per message) to avoid a hung UI.
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
                # normalize roles for UI
                if role == 'assistant':
                    role = 'bot'
                elif role == 'system':
                    role = 'sys'
                elif role not in ('user', 'bot', 'sys'):
                    role = 'user'

                if role == 'bot':
                    try:
                        import markdown as _markdown
                        _h = _markdown.markdown(str(content), extensions=['extra', 'codehilite'])
                        _h = sanitize_html(_h)
                    except Exception:
                        _h = sanitize_html(html.escape(str(content)))
                    ui_hist.append({'role': 'bot', 'html': _h})
                else:
                    ui_hist.append({'role': role, 'content': str(content)})

            payload = json.dumps(ui_hist, ensure_ascii=False)
            sm = json.dumps(str(status_msg or "Loaded chat log."), ensure_ascii=False)

            # Attempt bulk helper call and get a success marker back.
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

            # Fallback: incremental replay (small JS snippets, robust for huge logs)
            try:
                win.evaluate_js(f"resetChatToStatus({sm});")
            except Exception:
                # If even reset fails, don't crash the app.
                return

            for m in ui_hist:
                try:
                    r = (m.get('role') or 'user')
                    if r == 'bot':
                        h = m.get('html', '')
                        h_js = json.dumps(str(h), ensure_ascii=False)
                        win.evaluate_js(f"addMsg('bot', {h_js}, false, null);")
                    else:
                        c = m.get('content', '')
                        c_js = json.dumps(html.escape(str(c)), ensure_ascii=False)
                        rr = 'sys' if r == 'sys' else 'user'
                        win.evaluate_js(f"addMsg('{rr}', {c_js});")
                except Exception:
                    # Keep going; best-effort replay.
                    continue
        except Exception:
            pass


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

    if int(ecode) == 429:
        quota = _fmt_quota()
        if "free-models-per-day" in lower:
            if lang == "en":
                head = "OpenRouter limit reached (free models per day)."
                tail = "Options: wait for reset, use a paid model/provider, or add credits."
            else:
                head = "OpenRouter-Limit erreicht (Free-Modelle pro Tag)."
                tail = "Optionen: bis zum Reset warten, anderes Modell/Provider nutzen oder Credits hinzufügen."
            parts = [head]
            if quota:
                parts.append(quota)
            return " ".join(parts + [tail]).strip() + " [[ACTION:SWITCH_FREE_MODEL]]" + f" [HTTP 429]"
        else:
            if lang == "en":
                head = "OpenRouter rate limit reached."
                tail = "Options: wait briefly and retry, or switch model/provider."
            else:
                head = "OpenRouter-Rate-Limit erreicht."
                tail = "Optionen: kurz warten und erneut versuchen oder Modell/Provider wechseln."
            parts = [head]
            if quota:
                parts.append(quota)
            if msg:
                parts.append(msg)
            return " ".join(parts + [tail]).strip() + " [[ACTION:SWITCH_FREE_MODEL]]" + f" [HTTP 429]"

    if int(ecode) == 404 and ("privacy" in lower or "data policy" in lower or "no endpoints found" in lower):
        if lang == "en":
            return ("OpenRouter cannot route your request because your Privacy/Data-Policy settings exclude all endpoints "
                    "for this model. Check OpenRouter → Settings → Privacy (and any provider restrictions). "
                    f"[HTTP {status_code}]")
        return ("OpenRouter kann nicht routen, weil deine Privacy/Data-Policy-Einstellungen alle passenden Endpoints "
                "für dieses Modell ausschließen. Prüfe OpenRouter → Settings → Privacy (und ggf. Provider-Restrictions). "
                f"[HTTP {status_code}]")

    if int(ecode) == 402 or "insufficient credits" in lower or "add credits" in lower:
        if lang == "en":
            return ("OpenRouter: insufficient credits for this request. Add credits or choose a free/eligible model. "
                    f"[HTTP {status_code}]")
        return ("OpenRouter: Nicht genügend Guthaben für diese Anfrage. Guthaben hinzufügen oder ein passendes "
                "(ggf. freies) Modell wählen. "
                f"[HTTP {status_code}]")

    if int(ecode) in (401, 403):
        if lang == "en":
            return ("OpenRouter authentication/permission error. Check your API key and account settings. "
                    f"[HTTP {status_code}]")
        return ("OpenRouter: Auth/Permission-Fehler. Prüfe API-Key und Account-/Privacy-Einstellungen. "
                f"[HTTP {status_code}]")

    # Fallback
    if msg:
        if lang == "en":
            return f"OpenRouter error: {msg} [HTTP {status_code}]"
        return f"OpenRouter-Fehler: {msg} [HTTP {status_code}]"
    if lang == "en":
        return f"OpenRouter request failed. [HTTP {status_code}]"
    return f"OpenRouter-Anfrage fehlgeschlagen. [HTTP {status_code}]"

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
                raise RuntimeError(_openrouter_friendly_http_error(int(code or 0), raw_err, lang=lang))
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
            raise RuntimeError(_openrouter_friendly_http_error(int(getattr(e,'code',0) or 0), raw_err, lang=lang))
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
          2) Config/Comm-SCI-Config.json: providers.openrouter.api_key_plain
          3) Key file (KEYS_PATH):
             - provider-structured: providers.openrouter.api_key_plain
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

            # 3) key file fallback (provider-structured or legacy)
            if not key and os.path.exists(KEYS_PATH):
                try:
                    data = json.loads(Path(KEYS_PATH).read_text(encoding='utf-8')) or {}
                    if isinstance(data, dict):
                        provs2 = data.get('providers')
                        if isinstance(provs2, dict):
                            o = provs2.get(provider) or {}
                            if isinstance(o, dict):
                                key = (o.get('api_key_plain') or o.get('api_key') or '').strip()
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
          2) Config plaintext providers.huggingface.api_key_plain
          3) Key file (KEYS_PATH): providers.huggingface.api_key_plain (or legacy HF_TOKEN fields)
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

            # 3) key file fallback
            if not key and os.path.exists(KEYS_PATH):
                try:
                    data = json.loads(Path(KEYS_PATH).read_text(encoding='utf-8')) or {}
                    if isinstance(data, dict):
                        provs2 = data.get('providers')
                        if isinstance(provs2, dict):
                            h = provs2.get('huggingface') or provs2.get('hf') or {}
                            if isinstance(h, dict):
                                key = (h.get('api_key_plain') or h.get('api_key') or '').strip()
                        if not key:
                            key = (data.get('HF_TOKEN') or data.get('HUGGINGFACE_TOKEN') or '').strip()
                except Exception:
                    pass

            return OpenAICompatibleClient(base_url=base_url, api_key=key, app_referrer='', app_title='Comm-SCI Desktop')
        except Exception:
            return None

    def _openrouter_cache_path(self) -> str:
        try:
            return os.path.join(CONFIG_DIR, 'openrouter_models_cache.json')
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
                raw = Path(cache_path).read_text(encoding='utf-8')
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
            Path(cache_path).write_text(json.dumps({'ts': now, 'models': models}, ensure_ascii=False), encoding='utf-8')
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
            return os.path.join(CONFIG_DIR, 'huggingface_models_cache.json')
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
                raw = Path(cache_path).read_text(encoding='utf-8')
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
                Path(cache_path).write_text(json.dumps({'ts': now, 'models': models_live}, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
            except Exception:
                pass
            return models_live, {'source': 'live', 'age_s': 0, 'count': len(models_live)}

        # fallback to config list
        try:
            models_cfg = self.get_huggingface_models_from_config() if hasattr(self, 'get_huggingface_models_from_config') else []
            if isinstance(models_cfg, list) and models_cfg:
                # write cache as config snapshot (so UI remains fast/offline)
                try:
                    Path(cache_path).write_text(json.dumps({'ts': now, 'models': models_cfg}, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
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
            return os.path.join(CONFIG_DIR, 'openrouter_models_cache.json')
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
                raw = Path(cache_path).read_text(encoding='utf-8')
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
            return os.path.join(CONFIG_DIR, "huggingface_catalog_cache.json")
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
        top_n = max(1, min(1000, top_n))
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
        want_top = max(1, min(1000, want_top))

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
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
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
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
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
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"ts": now, "top_n": want_top, "provider_filter": want_pf, "models": models}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return models, meta
        except Exception:
            # fallback: no catalog available
            return [], {"source": "none", "age_s": 0, "count": 0, "top_n": want_top, "provider_filter": want_pf}

class Api(CSCRefiner):
    def __init__(self):
        # Bind governance + config safely
        super().__init__(globals().get('gov'), globals().get('cfg'))
        # True iff the current model chat-session was created with the ruleset injected
        self.session_with_governance: bool = True
        try:
            _g = globals().get('gov')
            if _g is not None:
                setattr(_g, 'runtime_state', self.gov_state)
        except Exception:
            pass


        # Provider routing (single-file adapters)
        try:
            self.provider_router = ProviderRouter(globals().get('cfg'))
            # Warm provider model caches from disk (no network).
            try:
                self._warm_model_caches_from_disk()
            except Exception:
                pass
            # --- B5 MVP: Session tracking (best-effort, no secrets) ---
            import uuid
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            self.session_start_dt = datetime.now()
            self.session_requests = 0
            self.session_rate_limit_hits = 0
            self.session_repair_passes = 0
            self.session_csc_applied_count = 0
            self.session_guard_hits = 0
            self.session_events = []  # list of {ts,type,data}
            # --- /B5 ---
            
        except Exception:
            self.provider_router = None

        # Window handles
        self.main_win = None
        self.panel_win = None
        self.panel_hidden = False

        # Panel API bridge (small surface to avoid pywebview method enumeration issues)
        try:
            self.panel_bridge = PanelBridge(self)
        except Exception:
            self.panel_bridge = None


        # Closing guard (prevents double-exit)
        self.is_closing = False
        self._connect_inflight = False  # connect dedupe guard

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
                'data': data,
            })
        except Exception:
            return




    def _get_enforcement_policy(self) -> str:
        """Return enforcement policy from config. Defaults to 'audit_only'."""
        try:
            pol = (getattr(cfg, "config", {}) or {}).get("enforcement_policy", "audit_only")
            pol = str(pol).strip().lower()
        except Exception:
            pol = "audit_only"
        if pol not in ("audit_only", "strict_warn", "strict_block"):
            pol = "audit_only"
        return pol


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


class QCBridge:
    """Minimal JS bridge for the QC Override dialog."""
    def __init__(self, api):
        self._api = api

    def ping(self, _payload=None):
        try:
            import time as _time
            return {"ok": True, "ts": _time.time()}
        except Exception:
            return {"ok": True}

    def qc_get_state(self, _payload=None):
        try:
            return self._api.qc_get_state()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_apply(self, values):
        try:
            return self._api.qc_override_apply(values)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_clear(self, _payload=None):
        try:
            return self._api.qc_override_clear()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def qc_override_cancel(self, _payload=None):
        try:
            return self._api.qc_override_cancel()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


class PanelBridge:
    """Separate JS-API bridge for the Panel window.

    Some pywebview backends expose only a subset of methods for secondary windows
    when reusing a large js_api surface. This bridge keeps the Panel stable by
    exposing only a tiny, deterministic API and forwarding to the main Api.

    Exposed methods:
      - ping()
      - get_ui()
      - panel_action(action, payload)
    """

    def __init__(self, api):
        self._api = api

    def ping(self, _payload=None):
        return self._api.ping()

    def get_ui(self):
        return self._api.get_ui()

    def panel_action(self, action, payload=None):
        return self._api.panel_action(action, payload)

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
                try:
                    import markdown as _markdown
                    _h = _markdown.markdown(str(content), extensions=['extra', 'codehilite'])
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
for _name in ("_render_error_html", "_safe_html", "_handle_command_deterministic", "_as_dict", "_as_list", "_safe_get", "_render_error_fallback", "set_api_key_for_provider", "load_log_from_path"):
    if _name in globals() and not hasattr(Api, _name):
        setattr(Api, _name, globals()[_name])

if __name__ == '__main__':
    if '--selftest' in sys.argv:
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
        for i in range(md):
            assert try_enter_sci_recursion(st, max_depth=md) is True
        assert try_enter_sci_recursion(st, max_depth=md) is False
        # Simulate one-shot auto-return
        st.sci_recursion_one_shot = True
        cur = int(getattr(st, 'sci_recursion_depth', 0) or 0)
        st.sci_recursion_depth = max(cur - 1, 0)
        assert st.sci_recursion_depth >= 0
        print('[SelfTest] OK')
        raise SystemExit(0)

    if webview is None:
        raise SystemExit('pywebview is required. Install with: pip install pywebview')
    if genai is None or types is None:
        raise SystemExit('google-genai is required. Install with: pip install google-genai')
    api = Api()
    api.main_win = webview.create_window(
        MAIN_WINDOW_TITLE, html=HTML_CHAT, js_api=api, 
        width=1100, height=1000,
        x=0, y=0
    )
    # Pre-create the Panel window *before* webview.start().
    # On macOS/Cocoa, creating secondary windows from a JS->Python callback can leave the JS API bridge uninitialized,
    # resulting in a Panel stuck at 'Loading panel...'.
    api._create_panel()
    api._create_qc_override()
    # HIER: Binden des Schließen-Events ("X") an unsere Logik
    api.main_win.events.closed += api.on_main_window_close
    
    webview.start(api.start_background_thread)			