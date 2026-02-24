from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


NormalizeHeadingsFn = Callable[[str], str]
EnforceSelfDebunkingFn = Callable[..., str]
NormalizeSciTraceFn = Callable[[str, Any], str]
NormalizeSelfDebunkingNumberingFn = Callable[..., str]
EnforceQcFooterFn = Callable[[str, Any, str], str]
EnsureQcFooterPresentFn = Callable[[str, Any, str], str]
NormalizeEvidenceTagsFn = Callable[[str], str]
StateResetFn = Callable[[Any], None]


@dataclass
class GovernanceService:
    """Thin delegation facade for governance text transforms.

    Methods are fail-soft and return original text on errors.
    Stage-S3 step: centralize raw output contract normalization order.
    """

    normalize_headings_fn: Optional[NormalizeHeadingsFn] = None
    enforce_self_debunking_fn: Optional[EnforceSelfDebunkingFn] = None
    normalize_sci_trace_fn: Optional[NormalizeSciTraceFn] = None
    normalize_self_debunking_numbering_fn: Optional[NormalizeSelfDebunkingNumberingFn] = None
    enforce_qc_footer_fn: Optional[EnforceQcFooterFn] = None
    ensure_qc_footer_present_fn: Optional[EnsureQcFooterPresentFn] = None
    normalize_evidence_tags_fn: Optional[NormalizeEvidenceTagsFn] = None
    apply_comm_stop_fn: Optional[StateResetFn] = None

    def normalize_headings(self, text: str) -> str:
        if not self.normalize_headings_fn:
            return text
        try:
            return self.normalize_headings_fn(text)
        except Exception:
            return text

    def enforce_self_debunking(
        self,
        text: str,
        gov_mgr: Any,
        profile_name: str,
        *,
        is_command: bool = False,
        lang: str = "en",
    ) -> str:
        if not self.enforce_self_debunking_fn:
            return text
        try:
            return self.enforce_self_debunking_fn(
                text,
                gov_mgr,
                profile_name,
                is_command=is_command,
                lang=lang,
            )
        except Exception:
            return text

    def normalize_sci_trace(self, text: str, gov_mgr: Any) -> str:
        if not self.normalize_sci_trace_fn:
            return text
        try:
            return self.normalize_sci_trace_fn(text, gov_mgr)
        except Exception:
            return text

    def normalize_self_debunking_numbering(self, text: str, *, lang: str = "en") -> str:
        if not self.normalize_self_debunking_numbering_fn:
            return text
        try:
            return self.normalize_self_debunking_numbering_fn(text, lang=lang)
        except Exception:
            return text

    def normalize_output_contracts(
        self,
        text: str,
        *,
        gov_mgr: Any,
        profile_name: str,
        governance_enabled: bool,
        is_command: bool = False,
        lang: str = "en",
    ) -> str:
        out = text
        if governance_enabled and self.enforce_qc_footer_fn:
            try:
                out = self.enforce_qc_footer_fn(out, gov_mgr, profile_name)
            except Exception:
                pass

        if self.normalize_evidence_tags_fn:
            try:
                out = self.normalize_evidence_tags_fn(out)
            except Exception:
                pass

        if governance_enabled:
            out = self.enforce_self_debunking(
                out,
                gov_mgr,
                profile_name,
                is_command=is_command,
                lang=lang,
            )

        out = self.normalize_sci_trace(out, gov_mgr)
        if governance_enabled and self.ensure_qc_footer_present_fn:
            try:
                out = self.ensure_qc_footer_present_fn(out, gov_mgr, profile_name)
            except Exception:
                pass
        return out

    def apply_profile_switch_resets(
        self,
        state: Any,
        profile_name: str,
        *,
        keep_sci_profiles: tuple[str, ...] = ("Expert", "Sparring"),
    ) -> None:
        """Apply deterministic reset rules for explicit profile switches."""
        try:
            state.active_profile = profile_name
        except Exception:
            pass
        try:
            state.qc_overrides = {}
        except Exception:
            pass
        try:
            state.sci_pending_turns = 0
        except Exception:
            pass
        if profile_name not in keep_sci_profiles:
            try:
                state.sci_active = False
            except Exception:
                pass
            try:
                state.sci_pending = False
            except Exception:
                pass
            try:
                state.sci_variant = ""
            except Exception:
                pass

    def apply_clear_chat_resets(self, state: Any) -> None:
        """Apply deterministic state resets on Clear Chat (session-local)."""
        try:
            state.qc_overrides = {}
        except Exception:
            pass
        try:
            state.sci_pending_turns = 0
        except Exception:
            pass

    def apply_comm_stop_resets(self, state: Any) -> None:
        """Apply deterministic reset rules for Comm Stop."""
        if self.apply_comm_stop_fn:
            try:
                self.apply_comm_stop_fn(state)
                return
            except Exception:
                pass
        try:
            state.comm_active = False
        except Exception:
            pass
        for _k, _v in (
            ('sci_pending', False),
            ('sci_active', False),
            ('sci_variant', ''),
            ('sci_pending_turns', 0),
            ('sci_recursion_one_shot', False),
            ('sci_recursion_parent_variant', ''),
            ('dynamic_one_shot_active', False),
            ('dynamic_nudge', ''),
        ):
            try:
                setattr(state, _k, _v)
            except Exception:
                pass
        try:
            state.qc_overrides = {}
        except Exception:
            pass

    def apply_legacy_command(self, *, cmd: str, state: Any, ruleset_data: Any) -> bool:
        """Apply deterministic legacy command state transitions.

        Returns True when the command was recognized and state was updated.
        """
        token = str(cmd or "").strip()
        data = ruleset_data if isinstance(ruleset_data, dict) else {}

        if token == "Strict on":
            state.overlay = "Strict"
            return True
        if token == "Strict off":
            state.overlay = ""
            return True
        if token == "Explore on":
            state.overlay = "Explore"
            return True
        if token == "Explore off":
            state.overlay = ""
            return True
        if token == "Color on":
            state.color = "on"
            return True
        if token == "Color off":
            state.color = "off"
            return True
        if token == "SCI on":
            state.sci_pending = True
            try:
                state.sci_pending_turns = 0
            except Exception:
                pass
            return True
        if token == "SCI off":
            state.sci_pending = False
            state.sci_active = False
            state.sci_variant = ""
            try:
                state.sci_pending_turns = 0
            except Exception:
                pass
            return True
        if token == "Comm Stop":
            self.apply_comm_stop_resets(state)
            return True
        if token == "Comm Start":
            state.comm_active = True
            try:
                default_prof = (data.get("default_profile") or "Standard")
                profiles = (data.get("profiles") or {}) if isinstance(data, dict) else {}
                if isinstance(profiles, dict) and default_prof in profiles:
                    state.active_profile = default_prof
                    state.sci_pending_turns = 0
                    if default_prof not in ("Expert", "Sparring"):
                        state.sci_active = False
                        state.sci_pending = False
                        state.sci_variant = ""
            except Exception:
                pass
            return True
        if token in ("Comm Anchor off", "Anchor auto off"):
            try:
                state.anchor_auto = False
                state.anchor_force_next = False
                state.anchor_auto_user_override = True
            except Exception:
                pass
            return True
        if token in ("Comm Anchor on", "Anchor auto on"):
            try:
                state.anchor_auto = True
                state.anchor_auto_user_override = True
            except Exception:
                pass
            try:
                state.anchor_force_next = False
            except Exception:
                pass
            return True
        if token == "Dynamic one-shot on":
            try:
                state.dynamic_one_shot_active = True
            except Exception:
                pass
            try:
                state.dynamic_nudge = "one-shot"
            except Exception:
                pass
            return True

        return False
