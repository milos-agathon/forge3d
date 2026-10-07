"""Run the optional Metal probe without crediting ABSENT as GPU evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_probe(path: Path) -> int:
    result = subprocess.run([
        sys.executable, "scripts/terrain_ci_probe.py", "--mode", "terrain",
        "--json", str(path),
    ])
    if result.returncode not in (0, 2):
        return result.returncode
    evidence = json.loads(path.read_text(encoding="utf-8"))
    expected = "passed" if result.returncode == 0 else "absent"
    if evidence.get("status") != expected:
        raise ValueError("optional probe exit code disagrees with its evidence")
    state = "positive" if result.returncode == 0 else "absent"
    if target := os.environ.get("GITHUB_OUTPUT"):
        with Path(target).open("a", encoding="utf-8") as stream:
            stream.write(f"probe={state}\n")
    if result.returncode == 2:
        message = "Optional Metal diagnostic ABSENT: no CI-safe hardware adapter; controls were not run."
        print(f"::notice::{message}")
        if target := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(target).open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_probe(Path(sys.argv[1])))
