"""Fail a required pytest lane unless it ran cleanly with zero skips."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _total(root: ET.Element, field: str) -> int:
    nodes = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return sum(int(node.attrib.get(field, "0")) for node in nodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args()
    root = ET.parse(args.junit_xml).getroot()
    counts = {
        field: _total(root, field)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] <= 0:
        raise SystemExit(f"required lane ran no tests: {counts}")
    if any(counts[field] for field in ("failures", "errors", "skipped")):
        raise SystemExit(f"required lane was not clean and zero-skip: {counts}")
    print(f"required lane clean: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
