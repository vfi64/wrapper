from provider_service import ProviderService


class _RouterOk:
    def get_active_provider(self):
        return "hf"

    def build_huggingface_client(self):
        return {"client": "hf"}

    def build_openrouter_client(self):
        return {"client": "or"}

    def get_gemini_models_cached(self, *, force_refresh=False):
        return ["gemini-2.0-flash"], {"source": "cache"}

    def get_openrouter_models_cached(self, *, force_refresh=False):
        return ["openai/gpt-4.1-mini"], {"source": "live"}

    def get_huggingface_models_cached(self, *, force_refresh=False):
        return ["meta-llama/Llama-3.3-70B-Instruct"], {"source": "live"}

    def get_huggingface_models_from_config(self):
        return ["cfg/model"]

    def get_huggingface_catalog_cached(self, *, top_n=200, provider_filter="all", force_refresh=False):
        return (["catalog/model-a", "catalog/model-b"], {"source": "live", "top_n": top_n, "provider_filter": provider_filter, "force_refresh": bool(force_refresh)})

    def supports_native_retrieval(self, provider):
        return str(provider or "").strip().lower() == "openrouter"


class _RouterBoom:
    def get_active_provider(self):
        raise RuntimeError("boom")


class _Cfg:
    def __init__(self):
        self.config = {"active_provider": "gemini", "providers": {"gemini": {"default_model": "gemini-2.0-flash"}}}
        self.saved = False

    def get_active_provider(self):
        return self.config.get("active_provider", "gemini")

    def set_active_provider(self, provider):
        self.config["active_provider"] = provider

    def get_provider_model(self, provider):
        return ((self.config.get("providers") or {}).get(provider) or {}).get("default_model", "")

    def set_provider_model(self, provider, model):
        provs = self.config.setdefault("providers", {})
        p = provs.setdefault(provider, {})
        p["default_model"] = model

    def set_model(self, model):
        self.config["model"] = model

    def save(self):
        self.saved = True


def test_provider_service_delegates_to_router():
    svc = ProviderService(router=_RouterOk())

    assert svc.get_active_provider() == "huggingface"
    assert svc.build_client("huggingface") == {"client": "hf"}
    assert svc.build_client("hf") == {"client": "hf"}
    assert svc.build_client("openrouter") == {"client": "or"}
    assert svc.build_client("openai") == {"client": "or"}
    assert svc.build_client("openai_compat") == {"client": "or"}
    assert svc.get_models_cached("gemini", force_refresh=True)[0] == ["gemini-2.0-flash"]
    assert svc.get_models_cached("openrouter", force_refresh=True)[0] == ["openai/gpt-4.1-mini"]
    assert svc.get_models_cached("openai", force_refresh=True)[0] == ["openai/gpt-4.1-mini"]
    assert svc.get_models_cached("huggingface", force_refresh=True)[0] == ["meta-llama/Llama-3.3-70B-Instruct"]
    assert svc.get_models_cached("hf", force_refresh=True)[0] == ["meta-llama/Llama-3.3-70B-Instruct"]
    assert svc.get_models_from_config("huggingface") == ["cfg/model"]
    assert svc.get_models_from_config("hf") == ["cfg/model"]
    m, meta = svc.get_huggingface_catalog_cached(top_n=120, provider_filter="inference", force_refresh=True)
    assert m == ["catalog/model-a", "catalog/model-b"]
    assert meta.get("top_n") == 120
    assert meta.get("provider_filter") == "inference"
    assert meta.get("force_refresh") is True
    assert svc.supports_native_retrieval("gemini") is True
    assert svc.supports_native_retrieval("openrouter") is True
    assert svc.supports_native_retrieval("huggingface") is False


def test_provider_service_cfg_helpers():
    svc = ProviderService()
    cfg = _Cfg()

    assert svc.normalize_provider("hf") == "huggingface"
    assert svc.canonical_provider_id("hf") == "huggingface"
    assert svc.canonical_provider_id("openai") == "openrouter"
    assert svc.canonical_provider_id("openai_compat") == "openrouter"
    assert svc.canonical_provider_id("gemini") == "gemini"
    assert svc.get_active_provider_from_cfg(cfg) == "gemini"
    assert svc.get_provider_model(cfg, "gemini", fallback_get_model_fn=lambda: "") == "gemini-2.0-flash"
    assert svc.set_active_provider(cfg, "openrouter") is True
    assert cfg.get_active_provider() == "openrouter"
    assert svc.set_provider_model(cfg, "openrouter", "openai/gpt-4.1-mini") is True
    assert cfg.get_provider_model("openrouter") == "openai/gpt-4.1-mini"


def test_provider_service_is_fail_soft():
    svc = ProviderService(router=_RouterBoom())

    assert svc.get_active_provider() == "gemini"
    assert svc.build_client("huggingface") is None
    assert svc.build_client("openrouter") is None
    assert svc.get_models_cached("gemini", force_refresh=True) == ([], {})
    assert svc.get_models_cached("openrouter", force_refresh=True) == ([], {})
    assert svc.get_models_cached("huggingface", force_refresh=True) == ([], {"source": "none"})
    assert svc.get_models_from_config("huggingface") == []
    assert svc.get_huggingface_catalog_cached(top_n=50, provider_filter="all", force_refresh=True) == ([], {})
    assert svc.supports_native_retrieval("gemini") is True
    assert svc.supports_native_retrieval("openrouter") is False


def test_build_model_candidates_prefers_free_when_primary_is_free():
    svc = ProviderService()
    cfg = _Cfg()
    cfg.config["providers"]["openrouter"] = {
        "default_model": "x",
        "fallback_models": ["fallback/a", "fallback/b"],
    }
    out = svc.build_model_candidates(
        provider="openrouter",
        primary_model="meta/model:free",
        available_models=["foo/bar", "x/y:free", "a/b", "c/d:free"],
        cfg=cfg,
    )
    # openrouter capped to 5
    assert len(out) == 5
    assert out[0] == "meta/model:free"
    # config fallbacks before model-cache candidates
    assert out[1:3] == ["fallback/a", "fallback/b"]
    # then :free models are preferred
    assert out[3:] == ["x/y:free", "c/d:free"]


def test_build_model_candidates_huggingface_cap_and_dedupe():
    svc = ProviderService()
    cfg = _Cfg()
    cfg.config["providers"]["huggingface"] = {
        "default_model": "x",
        "fallback_models": ["m1", "m2", "m3"],
    }
    avail = [f"hf/model-{i}" for i in range(20)]
    out = svc.build_model_candidates(
        provider="huggingface",
        primary_model="m1",  # duplicate with fallback, must remain only once
        available_models=avail,
        cfg=cfg,
    )
    assert out[0] == "m1"
    assert "m2" in out and "m3" in out
    assert out.count("m1") == 1
    # huggingface capped to 12
    assert len(out) == 12
