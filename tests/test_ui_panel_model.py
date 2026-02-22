from ui_panel_model import StateSnapshot, normalize_panel_ui

def _mk_button(name, cmd):
    return {"name": name, "cmd": cmd, "desc": ""}

def test_toggle_action_labels_and_cmds():
    data = {
        "comm": [_mk_button("Comm Start", "Comm Start"), _mk_button("Comm Stop", "Comm Stop")],
        "sci": [_mk_button("SCI on", "SCI on"), _mk_button("SCI off", "SCI off"), _mk_button("SCI menu", "SCI menu")],
        "overlays": [_mk_button("Strict on", "Strict on"), _mk_button("Strict off", "Strict off"),
                     _mk_button("Explore on", "Explore on"), _mk_button("Explore off", "Explore off")],
        "tools": [_mk_button("Color on", "Color on"), _mk_button("Color off", "Color off")],
    }
    state = StateSnapshot(comm_active=True, sci_on=False, overlay="strict", color_on=True)
    out = normalize_panel_ui(data, state)

    # Comm on => offer OFF
    assert out["comm"][0]["name"].startswith("Comm") and out["comm"][0]["name"].endswith(": OFF")
    assert out["comm"][0]["cmd"] == "Comm Stop"

    # SCI off => offer ON, and hide menu/recurse
    assert out["sci"][0]["name"].startswith("SCI") and out["sci"][0]["name"].endswith(": ON")
    assert out["sci"][0]["cmd"] == "SCI on"
    assert all(b.get("cmd") not in ("SCI menu", "SCI recurse") for b in out["sci"])

    # Strict on => offer OFF
    strict_btn = next(b for b in out["overlays"] if b["name"].startswith("Strict"))
    assert strict_btn["name"].endswith(": OFF")
    assert strict_btn["cmd"] == "Strict off"

    # Explore off => offer ON
    explore_btn = next(b for b in out["overlays"] if b["name"].startswith("Explore"))
    assert explore_btn["name"].endswith(": ON")
    assert explore_btn["cmd"] == "Explore on"

    # Color on => offer OFF
    assert out["tools"][0]["name"].startswith("Color") and out["tools"][0]["name"].endswith(": OFF")
    assert out["tools"][0]["cmd"] == "Color off"
