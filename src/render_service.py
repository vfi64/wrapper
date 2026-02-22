from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


StripVerificationRouteFn = Callable[[str], str]
SanitizeSelfDebunkingHtmlFn = Callable[[str], str]
NormalizeHashSubheadingsHtmlFn = Callable[[str], str]
NumberSelfDebunkingHtmlFn = Callable[..., str]


@dataclass
class RenderService:
    """Thin post-render facade (Stage 2, additive, fail-soft)."""

    strip_verification_route_fn: Optional[StripVerificationRouteFn] = None
    sanitize_self_debunking_html_fn: Optional[SanitizeSelfDebunkingHtmlFn] = None
    normalize_hash_subheadings_html_fn: Optional[NormalizeHashSubheadingsHtmlFn] = None
    number_self_debunking_html_fn: Optional[NumberSelfDebunkingHtmlFn] = None

    def strip_verification_route_display(self, text: str) -> str:
        if not self.strip_verification_route_fn:
            return text
        try:
            return self.strip_verification_route_fn(text)
        except Exception:
            return text

    def sanitize_self_debunking_html(self, html_text: str) -> str:
        if not self.sanitize_self_debunking_html_fn:
            return html_text
        try:
            return self.sanitize_self_debunking_html_fn(html_text)
        except Exception:
            return html_text

    def normalize_hash_subheadings_html(self, html_text: str) -> str:
        if not self.normalize_hash_subheadings_html_fn:
            return html_text
        try:
            return self.normalize_hash_subheadings_html_fn(html_text)
        except Exception:
            return html_text

    def number_self_debunking_html(self, html_text: str, *, lang: str = "en") -> str:
        if not self.number_self_debunking_html_fn:
            return html_text
        try:
            return self.number_self_debunking_html_fn(html_text, lang=lang)
        except Exception:
            return html_text
