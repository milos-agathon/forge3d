"""Record completed pytest timings and every explicit skip in the job summary."""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> str:
    if not path.is_file():
        return "Python completion report ABSENT: pytest did not write its JUnit report.\n"
    cases = ET.parse(path).getroot().findall(".//testcase")
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skips = Counter(
        node.get("message", "unspecified skip")
        for case in cases
        if (node := case.find("skipped")) is not None
    )
    lines = [
        f"Python pytest report: {len(cases)} cases, {failures} failures, "
        f"{errors} errors, {sum(skips.values())} explicit skips.",
        "", "Explicit skip reasons:",
    ]
    lines.extend(f"- {count}: {reason}" for reason, count in sorted(skips.items()))
    lines.extend(["", "Slowest 30 completed test cases (JUnit seconds):"])
    slowest = sorted(cases, key=lambda case: float(case.get("time", "0")), reverse=True)[:30]
    lines.extend(
        f"- {float(case.get('time', '0')):.3f}s: "
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for case in slowest
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = summarize(Path(sys.argv[1]))
    print(report, end="")
    if target := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(target).open("a", encoding="utf-8") as stream:
            stream.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
