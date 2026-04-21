#!/usr/bin/env python3
"""
HelloData ingestion pipeline — runs ordered steps using the project venv Python.

Steps:
  migrate   — scripts/bootstrap_db.py (idempotent migrations)
  reports   — ingest_hellodata.py --from-property-reports
  portfolio — ingest_hellodata.py --from-portfolio (full GET /property/{id} for UUIDs in hellodata_portfolio)
  refresh   — ingest_hellodata.py --refresh-stored (re-fetch all hellodata_properties)

Environment (.env at repo root):
  HELLO_DATA_API_KEY, DATABASE_URL

Usage:
  venv/bin/python scripts/pipelines/run_hellodata_ingestion.py
  venv/bin/python scripts/pipelines/run_hellodata_ingestion.py --steps migrate,reports
  venv/bin/python scripts/pipelines/run_hellodata_ingestion.py --steps migrate,reports,portfolio --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def python_bin() -> str:
    venv_py = ROOT / "venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def run_cmd(argv: list[str], *, dry_run: bool) -> int:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}$ {' '.join(argv)}", flush=True)
    if dry_run:
        return 0
    r = subprocess.run(argv, cwd=str(ROOT))
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="HelloData ingestion pipeline")
    parser.add_argument(
        "--steps",
        default="migrate,reports",
        help="Comma-separated: migrate, reports, portfolio, refresh (default: migrate,reports)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Passed to ingest_hellodata.py (default: script default 0.2).",
    )
    args = parser.parse_args()

    py = python_bin()
    raw = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    allowed = {"migrate", "reports", "portfolio", "refresh"}
    steps = []
    for s in raw:
        if s not in allowed:
            print(f"Unknown step: {s!r}. Allowed: {sorted(allowed)}", file=sys.stderr)
            return 1
        steps.append(s)

    if not steps:
        print("No steps.", file=sys.stderr)
        return 1

    sleep_args: list[str] = []
    if args.sleep_seconds is not None:
        sleep_args = ["--sleep-seconds", str(args.sleep_seconds)]

    for step in steps:
        if step == "migrate":
            code = run_cmd(
                [py, str(ROOT / "scripts" / "bootstrap_db.py")],
                dry_run=args.dry_run,
            )
        elif step == "reports":
            cmd = [
                py,
                str(ROOT / "scripts" / "ingest_hellodata.py"),
                "--from-property-reports",
            ] + sleep_args
            code = run_cmd(cmd, dry_run=args.dry_run)
        elif step == "portfolio":
            cmd = [
                py,
                str(ROOT / "scripts" / "ingest_hellodata.py"),
                "--from-portfolio",
            ] + sleep_args
            code = run_cmd(cmd, dry_run=args.dry_run)
        elif step == "refresh":
            cmd = [
                py,
                str(ROOT / "scripts" / "ingest_hellodata.py"),
                "--refresh-stored",
            ] + sleep_args
            code = run_cmd(cmd, dry_run=args.dry_run)
        else:
            code = 1

        if code != 0:
            print(f"Pipeline failed at step={step!r} exit={code}", file=sys.stderr)
            return code

    print("Pipeline completed OK:", ",".join(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
