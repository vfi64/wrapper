from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_CTRL_COUNTS: Dict[str, int] = {}
_TERMINAL_REPORTER = None


def pytest_configure(config):
    # Per-session counters for explicit control output.
    global _CTRL_COUNTS, _TERMINAL_REPORTER
    _CTRL_COUNTS = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "errors": 0,
    }
    _TERMINAL_REPORTER = config.pluginmanager.get_plugin("terminalreporter")


def _status_from_report(report) -> str:
    if report.passed:
        return "XPASS" if getattr(report, "wasxfail", False) else "PASS"
    if report.skipped:
        return "XFAIL" if getattr(report, "wasxfail", False) else "SKIP"
    return "FAIL"


def pytest_runtest_logreport(report):
    # Print exactly one control line per test after call stage.
    # For setup/teardown failures, print those as control lines too.
    if report.when not in ("call", "setup", "teardown"):
        return

    is_main_line = report.when == "call"
    is_setup_teardown_error = report.when in ("setup", "teardown") and report.failed
    if not is_main_line and not is_setup_teardown_error:
        return

    status = _status_from_report(report)
    nodeid = report.nodeid
    duration_s = getattr(report, "duration", 0.0) or 0.0
    phase = "" if report.when == "call" else f" [{report.when}]"

    tr = _TERMINAL_REPORTER
    if tr is not None:
        tr.write_line(f"[TEST {status}] {nodeid}{phase} ({duration_s:.3f}s)")

    counts: Dict[str, int] = _CTRL_COUNTS
    if status == "PASS":
        counts["passed"] += 1
    elif status == "FAIL":
        # Setup/teardown failures count as errors for clearer summary.
        if report.when in ("setup", "teardown"):
            counts["errors"] += 1
        else:
            counts["failed"] += 1
    elif status == "SKIP":
        counts["skipped"] += 1
    elif status == "XFAIL":
        counts["xfailed"] += 1
    elif status == "XPASS":
        counts["xpassed"] += 1


def pytest_terminal_summary(terminalreporter):
    counts: Dict[str, int] = _CTRL_COUNTS
    total = sum(counts.values())
    terminalreporter.write_sep("=", "Kontrollzusammenfassung")
    terminalreporter.write_line(f"Gesamt:   {total}")
    terminalreporter.write_line(f"PASS:     {counts['passed']}")
    terminalreporter.write_line(f"FAIL:     {counts['failed']}")
    terminalreporter.write_line(f"ERROR:    {counts['errors']}")
    terminalreporter.write_line(f"SKIP:     {counts['skipped']}")
    terminalreporter.write_line(f"XFAIL:    {counts['xfailed']}")
    terminalreporter.write_line(f"XPASS:    {counts['xpassed']}")
