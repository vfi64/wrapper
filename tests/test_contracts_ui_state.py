import re
from pathlib import Path

WRAPPER_DEFAULT = Path(__file__).resolve().parents[1] / "src" / "Comm-SCI-Control-App.py"

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

    # Find the toggle helper (the project uses _toggle_btn).
    m = re.search(r"def\s+_toggle_btn\s*\([^)]*\)\s*:\s*\n(?P<body>(?:\s+.*\n){5,80})", src)
    assert m, "Missing _toggle_btn() helper (required by contract V2)."
    body = m.group("body")

    # Required semantics:
    # is_on True  => label contains ': OFF'  and cmd_off is used
    # is_on False => label contains ': ON'   and cmd_on  is used
    #
    # We accept small formatting differences, but we require evidence of the inversion.
    assert (": OFF" in body) and (": ON" in body), "Toggle labels must include both ': OFF' and ': ON'."

    # Ensure the ON-branch produces OFF label (action), not ON label (state).
    # We look for a common pattern: 'OFF' if is_on else 'ON' or the equivalent inversion.
    inversion_ok = (
        re.search(r"OFF'\s*if\s*is_on\s*else\s*'ON", body)
        or re.search(r'"OFF"\s*if\s*is_on\s*else\s*"ON"', body)
        or re.search(r"if\s+is_on\s*:\s*\n(?:\s+.*OFF.*\n)+", body)
    )
    assert inversion_ok, "Toggle must show OFF when is_on=True (action, not state)."

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
