"""
benchmark/gate.py
──────────────────
CI gate: reads benchmark.json and exits 0 (pass) or 1 (fail).
Called as the final step of the GitHub Actions pipeline.

Usage:
  python -m benchmark.gate
  python -m benchmark.gate --report ./outputs/benchmark.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from configs.settings import cfg


def check_gate(report_path: Path) -> tuple[bool, list[str]]:
    if not report_path.exists():
        return False, [f"benchmark.json not found at {report_path}"]

    with open(report_path) as f:
        report = json.load(f)

    failures = report.get("failures", [])
    passed   = report.get("passed", False)

    return passed, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=cfg.logging.benchmark_json,
        help="Path to benchmark.json produced by runner.py",
    )
    args = parser.parse_args()

    passed, failures = check_gate(Path(args.report))

    if passed:
        print("✅  Hardware gate PASSED")
        return 0
    else:
        print("❌  Hardware gate FAILED:")
        for f in failures:
            print(f"    • {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())