#!/usr/bin/env python
# scripts/ci_pytest_lane.py
# CENSOR Task 13: the default CI Python test lane. Runs the WHOLE tests/ tree
# except the files enumerated in tests/UNRUN.toml (each with a documented,
# owner-attributed, non-expired reason). This is the single honest source of
# "what Python CI runs by default" -- tests/test_no_silent_degradation.py
# imports unrun_files() from here so the UNRUN accounting gate stays truthful.
# RELEVANT FILES: tests/UNRUN.toml, tests/_toml_compat.py, .github/workflows/ci.yml
"""Run pytest over tests/ minus the UNRUN allowlist, forwarding extra argv."""
from __future__ import annotations

import subprocess
import sys
from collections import deque
from os import environ
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
UNRUN_TOML = TESTS / "UNRUN.toml"

# tests/_toml_compat.py is the shared loader (stdlib tomllib on >=3.11, tiny
# hand-rolled fallback on 3.10 where CI still runs).
sys.path.insert(0, str(TESTS))
from _toml_compat import load_toml  # noqa: E402


def unrun_files() -> list[str]:
    """Return the repo-relative test files excluded from the default lane."""
    if not UNRUN_TOML.exists():
        return []
    data = load_toml(UNRUN_TOML)
    return [str(entry["file"]) for entry in data.get("entries", [])]


def _all_test_files() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in TESTS.glob("test_*.py"))


def default_lane_files() -> list[str]:
    """Every tests/test_*.py the default lane runs = all files minus the UNRUN set.

    The honesty gate requires every matching file to be tracked, so this is the
    same suite locally and on a clean CI checkout.
    """
    unrun = set(unrun_files())
    return [f for f in _all_test_files() if f not in unrun]


def build_pytest_args(passthrough: list[str]) -> list[str]:
    """Compose the pytest argv: the explicit run-list + passthrough.

    We pass the file list explicitly rather than `tests/ --ignore=<file>`
    to make the lane's accounting directly inspectable and to prevent UNRUN
    files that fail at collection time from ever being imported.
    """
    return [*default_lane_files(), *passthrough]


def _github_escape(message: str) -> str:
    """Escape a string for GitHub workflow command annotations."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", *build_pytest_args(argv)]
    tail: deque[str] = deque(maxlen=180)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    code = proc.wait()
    if code and environ.get("GITHUB_ACTIONS") == "true":
        message = "\n".join(tail)
        if len(message) > 3500:
            message = message[-3500:]
        print(
            f"::error title=Default Python lane failed::{_github_escape(message)}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
