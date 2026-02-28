from __future__ import annotations


def panel_ui_default_snapshot() -> dict:
    return {
        'providers': ['gemini', 'openrouter', 'huggingface'],
        'current_provider': 'gemini',
        'current_model': 'gemini-2.0-flash',
        'available_models': ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-pro'],
        'answer_language': 'de',
        'language_policy_mode': 'production',
        'comm': [],
        'profiles': [],
        'sci': [],
        'overlays': [],
        'tools': [],
        'logs': [],
        'chat_logs': [],
        'model_hint': '',
    }


def panel_ui_failopen_snapshot(*, gov_state) -> dict:
    """Return the minimal panel snapshot used when seam composition fails."""
    data = panel_ui_default_snapshot()
    data['comm'] = [{'name': 'Comm Start', 'cmd': 'Comm Start', 'desc': 'Start Comm Control Layer'}]
    try:
        comm_active_ui = bool(getattr(gov_state, 'comm_active', False))
    except Exception:
        comm_active_ui = False
    data['comm_active'] = comm_active_ui
    data['manual_test_visible'] = comm_active_ui
    data['qc_override_visible'] = comm_active_ui
    return panel_ui_apply_legacy_aliases(data)


def panel_ui_apply_basic_runtime(
    data,
    *,
    current_provider=None,
    current_model=None,
    available_models=None,
    answer_language=None,
    language_policy_mode=None,
) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()

    cp = str(current_provider or '').strip().lower()
    if cp:
        out['current_provider'] = cp

    cm = str(current_model or '').strip()
    if cm:
        out['current_model'] = cm

    if isinstance(available_models, list):
        out['available_models'] = available_models

    al = str(answer_language or '').strip().lower()
    if al in ('de', 'en'):
        out['answer_language'] = al

    mode = str(language_policy_mode or '').strip().lower()
    if mode in ('production', 'benchmark'):
        out['language_policy_mode'] = mode

    return out


def panel_ui_probe_and_apply_basic_runtime(
    data,
    *,
    provider_router,
    cfg_obj,
    get_available_models_fn,
) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()

    cp = None
    try:
        pr = provider_router
        if pr is not None and hasattr(pr, 'get_active_provider'):
            cp = (pr.get_active_provider() or 'gemini').strip().lower()
        else:
            cp = 'gemini'
    except Exception:
        cp = None

    provider_for_model = str((cp or out.get('current_provider') or 'gemini')).strip().lower() or 'gemini'

    cm = None
    try:
        if cfg_obj is not None and hasattr(cfg_obj, 'get_provider_model'):
            cm = (cfg_obj.get_provider_model(provider_for_model) or '').strip()
    except Exception:
        cm = None

    available_models = None
    try:
        if provider_for_model in ('gemini', 'openrouter', 'huggingface') and callable(get_available_models_fn):
            available_models = get_available_models_fn(provider_for_model)
    except Exception:
        available_models = None

    al = None
    try:
        if cfg_obj is not None and hasattr(cfg_obj, 'get_answer_language'):
            al = (cfg_obj.get_answer_language() or 'de').strip().lower()
    except Exception:
        al = None
    mode = None
    try:
        if cfg_obj is not None and hasattr(cfg_obj, 'get_language_policy_mode'):
            mode = (cfg_obj.get_language_policy_mode() or 'production').strip().lower()
    except Exception:
        mode = None

    return panel_ui_apply_basic_runtime(
        out,
        current_provider=cp,
        current_model=cm,
        available_models=available_models,
        answer_language=al,
        language_policy_mode=mode,
    )


def panel_ui_apply_chat_log_listing(data, *, logs) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    if isinstance(logs, list):
        out['chat_logs'] = logs
        if logs:
            out['chat_log_selected'] = logs[0]
    return out


def panel_ui_merge_governance_ui(data, *, gov_ui) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    ui = gov_ui if isinstance(gov_ui, dict) else {}
    for key in ('comm', 'profiles', 'sci', 'overlays', 'tools', 'logs'):
        value = ui.get(key)
        if isinstance(value, list):
            out[key] = value
    for key in ('current_rule_file', 'version', 'loaded'):
        if key in ui:
            out[key] = ui.get(key)
    if isinstance(ui.get('answer_language'), str):
        out['answer_language'] = ui.get('answer_language')
    if isinstance(ui.get('language_policy_mode'), str):
        out['language_policy_mode'] = ui.get('language_policy_mode')
    return out


def panel_ui_apply_anchor_toggle(data, *, anchor_auto: bool) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    comm = out.get('comm')
    if not isinstance(comm, list) or not comm:
        return out

    tok_off = "Comm Anchor off"
    tok_on = "Comm Anchor on"
    if (tok_off not in comm) or (tok_on not in comm):
        return out

    try:
        i_off = comm.index(tok_off)
    except Exception:
        i_off = 10**9
    try:
        i_on = comm.index(tok_on)
    except Exception:
        i_on = 10**9
    ins = min(i_off, i_on) if min(i_off, i_on) != 10**9 else len(comm)

    comm2 = [c for c in comm if c not in (tok_off, tok_on)]
    btn = {
        "name": (tok_off if anchor_auto else tok_on),
        "cmd": (tok_off if anchor_auto else tok_on),
        "desc": ("Disable Anchor auto snapshots" if anchor_auto else "Enable Anchor auto snapshots"),
    }
    if ins < 0:
        ins = 0
    if ins > len(comm2):
        ins = len(comm2)
    comm2.insert(ins, btn)
    out['comm'] = comm2
    return out


def panel_ui_apply_failsoft_comm_toggle(data, *, comm_active: bool) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    comm = out.get('comm')
    if not isinstance(comm, list):
        return out
    if ("Comm Start" not in comm) or ("Comm Stop" not in comm):
        return out
    comm2 = [c for c in comm if c not in ("Comm Start", "Comm Stop")]
    if bool(comm_active):
        btn = {
            "name": "Comm ⏻: OFF",
            "cmd": "Comm Stop",
            "desc": "Stop Comm Control Layer",
        }
    else:
        btn = {
            "name": "Comm ⏻: ON",
            "cmd": "Comm Start",
            "desc": "Start Comm Control Layer",
        }
    comm2.insert(0, btn)
    out['comm'] = comm2
    return out


def panel_ui_apply_comm_visibility_gate(data, *, comm_active: bool) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    comm_active_ui = bool(comm_active)
    out['comm_active'] = comm_active_ui
    out['manual_test_visible'] = comm_active_ui
    out['qc_override_visible'] = comm_active_ui

    if comm_active_ui:
        return out

    def _cmd_name(_item):
        try:
            if isinstance(_item, dict):
                return str(_item.get('cmd') or _item.get('name') or '').strip()
            return str(_item or '').strip()
        except Exception:
            return ''

    comm_items = out.get('comm') if isinstance(out.get('comm'), list) else []
    kept = [it for it in comm_items if _cmd_name(it) == 'Comm Start']
    if not kept:
        kept = [{
            'name': 'Comm Start',
            'cmd': 'Comm Start',
            'desc': 'Start Comm Control Layer',
        }]
    else:
        fixed = []
        for it in kept:
            if isinstance(it, dict):
                cp = dict(it)
                cp['name'] = 'Comm Start'
                cp['cmd'] = 'Comm Start'
                fixed.append(cp)
            else:
                fixed.append({'name': 'Comm Start', 'cmd': 'Comm Start'})
        kept = fixed
    out['comm'] = kept
    for key in ('profiles', 'sci', 'overlays', 'tools', 'logs'):
        out[key] = []
    return out


def panel_ui_apply_legacy_aliases(data) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()
    out['provider'] = out.get('current_provider', 'gemini')
    out['model'] = out.get('current_model', 'gemini-2.0-flash')
    return out


def panel_ui_apply_state_postprocess(
    data,
    *,
    gov_state,
    panel_state_snapshot_ctor,
    panel_normalize_ui_fn,
) -> dict:
    out = data if isinstance(data, dict) else panel_ui_default_snapshot()

    # Anchor toggle collapse (best-effort)
    try:
        anchor_auto = bool(getattr(gov_state, "anchor_auto", True))
    except Exception:
        anchor_auto = True
    try:
        out = panel_ui_apply_anchor_toggle(out, anchor_auto=anchor_auto)
    except Exception:
        pass

    # Phase-1 pure UI normalization; if unavailable, keep fail-soft comm toggle collapse.
    try:
        gs = gov_state
        state_snapshot = None
        try:
            if callable(panel_state_snapshot_ctor):
                state_snapshot = panel_state_snapshot_ctor(
                    comm_active=bool(getattr(gs, 'comm_active', False)),
                    sci_on=bool(getattr(gs, 'sci_pending', False) or getattr(gs, 'sci_active', False)),
                    overlay=str(getattr(gs, 'overlay', '') or '').strip().lower(),
                    color_on=((getattr(gs, 'color', 'on') or 'on') == 'on'),
                )
        except Exception:
            state_snapshot = None

        if callable(panel_normalize_ui_fn) and state_snapshot is not None:
            out = panel_normalize_ui_fn(out, state_snapshot)
        else:
            out = panel_ui_apply_failsoft_comm_toggle(
                out,
                comm_active=bool(getattr(gs, 'comm_active', False)),
            )
    except Exception:
        pass

    # Comm-off gating and backward-compat aliases
    try:
        out = panel_ui_apply_comm_visibility_gate(
            out,
            comm_active=bool(getattr(gov_state, 'comm_active', False)),
        )
    except Exception:
        pass
    try:
        out = panel_ui_apply_legacy_aliases(out)
    except Exception:
        pass
    return out


def panel_ui_build_snapshot(
    *,
    provider_router,
    cfg_obj,
    get_available_models_fn,
    gov_obj,
    gov_state,
    panel_state_snapshot_ctor,
    panel_normalize_ui_fn,
    list_chat_logs_fn,
    chat_log_limit: int = 200,
) -> dict:
    data = panel_ui_default_snapshot()
    data = panel_ui_probe_and_apply_basic_runtime(
        data,
        provider_router=provider_router,
        cfg_obj=cfg_obj,
        get_available_models_fn=get_available_models_fn,
    )

    try:
        if gov_obj is not None and hasattr(gov_obj, 'get_ui_data'):
            data = panel_ui_merge_governance_ui(data, gov_ui=(gov_obj.get_ui_data() or {}))
    except Exception:
        pass

    data = panel_ui_apply_state_postprocess(
        data,
        gov_state=gov_state,
        panel_state_snapshot_ctor=panel_state_snapshot_ctor,
        panel_normalize_ui_fn=panel_normalize_ui_fn,
    )

    try:
        if callable(list_chat_logs_fn):
            res = list_chat_logs_fn(limit=int(chat_log_limit))
            if isinstance(res, dict) and res.get('ok') is True:
                data = panel_ui_apply_chat_log_listing(data, logs=res.get('logs'))
    except Exception:
        pass

    # Defensive final alias pass (idempotent) for older panel JS builds.
    return panel_ui_apply_legacy_aliases(data)
