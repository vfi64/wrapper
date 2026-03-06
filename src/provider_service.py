from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderService:
    """Thin provider facade over ProviderRouter (Stage 2, fail-soft)."""

    router: Any = None

    def normalize_provider(self, provider: str) -> str:
        p = (provider or "").strip().lower()
        if p in ("hf",):
            return "huggingface"
        if p in ("gemini", "openrouter", "huggingface", "openai", "openai_compat"):
            return p
        return "gemini"

    def canonical_provider_id(self, provider: str) -> str:
        """Map aliases to canonical internal provider ids.

        Canonical ids used by config/provider maps:
        - gemini
        - openrouter
        - huggingface
        """
        p = self.normalize_provider(provider)
        if p in ("openrouter", "openai", "openai_compat"):
            return "openrouter"
        if p in ("huggingface", "hf"):
            return "huggingface"
        return "gemini"

    def supports_native_retrieval(self, provider: str) -> bool:
        """Return whether native retrieval tool wiring is available for provider path."""
        p = self.canonical_provider_id(provider)
        if p == "gemini":
            return True
        try:
            if self.router is not None and hasattr(self.router, "supports_native_retrieval"):
                return bool(self.router.supports_native_retrieval(p))
        except Exception:
            pass
        return False

    def get_active_provider(self) -> str:
        try:
            p = (self.router.get_active_provider() if self.router is not None else "gemini") or "gemini"
            return self.normalize_provider(str(p))
        except Exception:
            return "gemini"

    def get_active_provider_from_cfg(self, cfg: Any) -> str:
        try:
            if cfg is not None and hasattr(cfg, "get_active_provider"):
                return self.normalize_provider(str(cfg.get_active_provider() or "gemini"))
        except Exception:
            pass
        return "gemini"

    def get_provider_model(self, cfg: Any, provider: str, fallback_get_model_fn: Any = None) -> str:
        p = self.normalize_provider(provider)
        try:
            if cfg is not None and hasattr(cfg, "get_provider_model"):
                m = str(cfg.get_provider_model(p) or "").strip()
                if m:
                    return m
        except Exception:
            pass
        try:
            if fallback_get_model_fn is not None:
                m = str(fallback_get_model_fn() or "").strip()
                if m:
                    return m
        except Exception:
            pass
        return ""

    def set_active_provider(self, cfg: Any, provider: str) -> bool:
        p = self.normalize_provider(provider)
        try:
            if cfg is not None and hasattr(cfg, "set_active_provider"):
                cfg.set_active_provider(p)
                return True
        except Exception:
            pass
        try:
            if cfg is not None and hasattr(cfg, "config"):
                cfg.config["active_provider"] = p
                if hasattr(cfg, "save"):
                    cfg.save()
                return True
        except Exception:
            pass
        return False

    def set_provider_model(self, cfg: Any, provider: str, model: str) -> bool:
        p = self.normalize_provider(provider)
        m = str(model or "").strip()
        if not m:
            return False
        try:
            if cfg is not None and hasattr(cfg, "set_provider_model"):
                cfg.set_provider_model(p, m)
                return True
        except Exception:
            pass
        try:
            if cfg is not None and hasattr(cfg, "set_model"):
                cfg.set_model(m)
                return True
        except Exception:
            pass
        return False

    def build_client(self, provider: str) -> Any:
        try:
            p = self.canonical_provider_id(provider)
            if p == "huggingface":
                if self.router is not None and hasattr(self.router, "build_huggingface_client"):
                    return self.router.build_huggingface_client()
                return None
            if p == "openrouter":
                if self.router is not None and hasattr(self.router, "build_openrouter_client"):
                    return self.router.build_openrouter_client()
                return None
            return None
        except Exception:
            return None

    def get_models_cached(self, provider: str, *, force_refresh: bool = False) -> tuple[list, dict]:
        try:
            p = self.canonical_provider_id(provider)
            if p == "gemini":
                if self.router is not None and hasattr(self.router, "get_gemini_models_cached"):
                    m, meta = self.router.get_gemini_models_cached(force_refresh=bool(force_refresh))
                    return list(m or []), dict(meta or {})
                return [], {}
            if p == "openrouter":
                if self.router is not None and hasattr(self.router, "get_openrouter_models_cached"):
                    m, meta = self.router.get_openrouter_models_cached(force_refresh=bool(force_refresh))
                    return list(m or []), dict(meta or {})
                return [], {}
            if p == "huggingface":
                if self.router is not None and hasattr(self.router, "get_huggingface_models_cached"):
                    m, meta = self.router.get_huggingface_models_cached(force_refresh=bool(force_refresh))
                    return list(m or []), dict(meta or {})
                return [], {"source": "none"}
            return [], {}
        except Exception:
            return [], {}

    def get_models_from_config(self, provider: str) -> list:
        try:
            p = self.canonical_provider_id(provider)
            if p == "huggingface" and self.router is not None and hasattr(self.router, "get_huggingface_models_from_config"):
                return list(self.router.get_huggingface_models_from_config() or [])
            return []
        except Exception:
            return []

    def get_huggingface_catalog_cached(self, *, top_n: int = 200, provider_filter: str = "all", force_refresh: bool = False) -> tuple[list, dict]:
        try:
            if self.router is not None and hasattr(self.router, "get_huggingface_catalog_cached"):
                m, meta = self.router.get_huggingface_catalog_cached(
                    top_n=int(top_n or 200),
                    provider_filter=str(provider_filter or "all").strip(),
                    force_refresh=bool(force_refresh),
                )
                return list(m or []), dict(meta or {})
            return [], {}
        except Exception:
            return [], {}

    def build_model_candidates(
        self,
        *,
        provider: str,
        primary_model: str,
        available_models: list | None = None,
        cfg: Any = None,
    ) -> list[str]:
        """Build deterministic model fallback candidate list for provider calls."""
        pid = self.canonical_provider_id(provider)
        primary = str(primary_model or "").strip()
        models = list(available_models or [])

        cand: list[str] = []
        if primary:
            cand.append(primary)

        # Optional explicit fallback list from config.
        try:
            provs = (getattr(cfg, "config", {}) or {}).get("providers") or {}
            pconf = provs.get(pid) if isinstance(provs, dict) else {}
            fb = (pconf or {}).get("fallback_models") if isinstance(pconf, dict) else None
            if isinstance(fb, list):
                for x in fb:
                    sx = str(x or "").strip()
                    if sx and sx not in cand:
                        cand.append(sx)
        except Exception:
            pass

        # Prefer :free fallbacks first if primary is :free, otherwise keep model list order.
        try:
            if primary.endswith(":free"):
                for m in models:
                    sm = str(m or "").strip()
                    if sm and sm.endswith(":free") and sm not in cand:
                        cand.append(sm)
                for m in models:
                    sm = str(m or "").strip()
                    if sm and sm not in cand:
                        cand.append(sm)
            else:
                for m in models:
                    sm = str(m or "").strip()
                    if sm and sm not in cand:
                        cand.append(sm)
        except Exception:
            pass

        max_cand = 12 if pid == "huggingface" else 5
        return cand[:max_cand]
