#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(f"[quality-gate] $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        raise SystemExit(rc)


def _run_core_tests() -> None:
    _run([sys.executable, "-m", "pytest", "-q", "tests/test_uncertainty_codes.py"])
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_app.py",
            "-k",
            (
                "self_debunking or strip_internal_scaffolding_status or "
                "append_uncertainty_explanation_skips"
            ),
        ]
    )


def _run_replay_contract_tests() -> None:
    _run([sys.executable, "-m", "pytest", "-q", "tests/test_render_contract_replay.py"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contract quality gate for Self-Debunking and uncertainty rendering."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "core", "replay"),
        default="all",
        help="all=core+replay, core=targeted regression suites, replay=render-contract replay only.",
    )
    args = parser.parse_args()

    if args.mode in ("all", "core"):
        _run_core_tests()
    if args.mode in ("all", "replay"):
        _run_replay_contract_tests()

    print("[quality-gate] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

