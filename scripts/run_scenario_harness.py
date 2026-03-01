#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import manual_scenario_harness as harness


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run deterministic S15.1 scenario harness and always persist a final log to Logs/ManualTests.",
    )
    p.add_argument(
        "--scenario",
        default="s15_1_harness",
        help="Scenario label for the report filename.",
    )
    p.add_argument(
        "--driver",
        choices=("synthetic",),
        default="synthetic",
        help="Execution driver. S15.1 provides deterministic synthetic mode.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.driver != "synthetic":
        raise SystemExit("Unsupported driver")

    driver = harness.SyntheticHarnessDriver()
    report, path = harness.run_harness_with_final_log(
        driver=driver,
        root_dir=ROOT,
        scenario=str(args.scenario),
    )

    summary = report.get("summary") or {}
    print(f"Status: {report.get('status')}")
    print(f"Log file: {path}")
    print(
        "Summary: "
        f"cases={summary.get('case_count', 0)} "
        f"prompt_checks={summary.get('prompt_check_count', 0)} "
        f"fails={summary.get('fail_count', 0)} "
        f"influence_fails={summary.get('influence_fail_count', 0)}",
    )
    print("Kurzbericht:")
    for line in report.get("human_report") or []:
        print(f"- {line}")

    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

