import re
from pathlib import Path

WRAPPER_DEFAULT = Path(__file__).resolve().parents[1] / "src" / "Comm-SCI-Control-App.py"
SEAM_DEFAULT = Path(__file__).resolve().parents[1] / "src" / "panel_ui_snapshot_seam.py"

def _read_wrapper_source() -> str:
    if not WRAPPER_DEFAULT.exists():
        raise AssertionError(f"Missing wrapper source: {WRAPPER_DEFAULT}")
    return WRAPPER_DEFAULT.read_text(encoding="utf-8")

def test_contract_toggle_buttons_show_action_not_state():
    """
    Contract V2: toggle buttons must show the *action* (inverse of state).
    This is a source-level guard to prevent regressions when other fixes are merged.
    """
    src = _read_wrapper_source()

    # Legacy path: monolith helper.
    m = re.search(r"def\s+_toggle_btn\s*\([^)]*\)\s*:\s*\n(?P<body>(?:\s+.*\n){5,80})", src)
    if m:
        body = m.group("body")

        # Required semantics:
        # is_on True  => label contains ': OFF'  and cmd_off is used
        # is_on False => label contains ': ON'   and cmd_on  is used
        #
        # We accept small formatting differences, but we require evidence of the inversion.
        assert (": OFF" in body) and (": ON" in body), "Toggle labels must include both ': OFF' and ': ON'."

        # Ensure the ON-branch produces OFF label (action), not ON label (state).
        inversion_ok = (
            re.search(r"OFF'\s*if\s*is_on\s*else\s*'ON", body)
            or re.search(r'"OFF"\s*if\s*is_on\s*else\s*"ON"', body)
            or re.search(r"if\s+is_on\s*:\s*\n(?:\s+.*OFF.*\n)+", body)
        )
        assert inversion_ok, "Toggle must show OFF when is_on=True (action, not state)."
        return

    # S13 path: Comm toggle inversion moved into panel_ui_snapshot_seam.
    assert SEAM_DEFAULT.exists(), "Missing _toggle_btn() helper and panel_ui_snapshot_seam.py."
    seam_src = SEAM_DEFAULT.read_text(encoding="utf-8")
    m2 = re.search(
        r"def\s+panel_ui_apply_failsoft_comm_toggle\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n(?P<body>(?:\s+.*\n){5,120})",
        seam_src,
    )
    assert m2, "Missing panel_ui_apply_failsoft_comm_toggle() seam helper (required by contract V2)."
    body = m2.group("body")
    assert "Comm ⏻: OFF" in body and "Comm Stop" in body, "Active Comm state must expose OFF action / Comm Stop."
    assert "Comm ⏻: ON" in body and "Comm Start" in body, "Inactive Comm state must expose ON action / Comm Start."

def test_contract_no_second_state_truth_runtime_state_is_alias_if_present():
    """
    Contract V1: if runtime_state is used, it must be an alias of api.gov_state.
    This is enforced as a source-level guard: sync step must exist after ruleset reload.
    """
    src = _read_wrapper_source()

    # Look for the canonical alias assignment near reload paths.
    # This is intentionally permissive but requires at least one explicit sync.
    assert "runtime_state" in src, "Wrapper does not reference runtime_state; remove this test if architecture changes."

    sync_present = (
        "gov.runtime_state = self.gov_state" in src
        or "self.gov.runtime_state = self.gov_state" in src
        or "runtime_state = self.gov_state" in src
    )
    assert sync_present, "Missing explicit runtime_state alias sync after state replacement (contract V1)."
