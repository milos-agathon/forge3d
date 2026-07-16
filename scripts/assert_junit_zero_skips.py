"""Fail a required pytest lane unless it ran cleanly with zero skips.

The verifier treats testcase elements as the source of truth and validates
every declared suite aggregate against those testcase outcomes.  Parent suite
aggregates are therefore checked, never summed with their children.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


FIELDS = ("tests", "failures", "errors", "skipped")


class JUnitValidationError(ValueError):
    """The JUnit document is missing, malformed, contradictory, or not clean."""


@dataclass(frozen=True)
class Counts:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in FIELDS}


def _declared_counter(element: ET.Element, field: str) -> int | None:
    raw = element.attrib.get(field)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise JUnitValidationError(
            f"non-integer {field} counter on <{element.tag}>: {raw!r}"
        ) from exc
    if value < 0:
        raise JUnitValidationError(
            f"negative {field} counter on <{element.tag}>: {value}"
        )
    return value


def _testcase_counts(element: ET.Element) -> Counts:
    tests = failures = errors = skipped = 0
    for testcase in element.iter("testcase"):
        tests += 1
        outcomes = {child.tag for child in testcase if child.tag in {"failure", "error", "skipped"}}
        if len(outcomes) > 1:
            raise JUnitValidationError(
                "testcase has contradictory terminal outcomes: "
                f"{sorted(outcomes)}"
            )
        failures += "failure" in outcomes
        errors += "error" in outcomes
        # Pytest encodes xfail as <skipped>, so it is deliberately rejected.
        skipped += "skipped" in outcomes
    return Counts(tests, failures, errors, skipped)


def _validate_declared_counts(element: ET.Element) -> None:
    actual = _testcase_counts(element).as_dict()
    for field in FIELDS:
        declared = _declared_counter(element, field)
        if declared is not None and declared != actual[field]:
            raise JUnitValidationError(
                f"contradictory {field} total on <{element.tag}>: "
                f"declared={declared}, actual={actual[field]}"
            )


def verify_junit(path: Path) -> Counts:
    if not path.is_file():
        raise JUnitValidationError(f"JUnit XML does not exist: {path}")
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise JUnitValidationError(f"malformed or unreadable JUnit XML: {path}: {exc}") from exc

    if root.tag not in {"testsuite", "testsuites"}:
        raise JUnitValidationError(
            f"unsupported JUnit root <{root.tag}>; expected <testsuite> or <testsuites>"
        )

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise JUnitValidationError("JUnit XML contains no <testsuite>")

    # Validate each suite against its own descendant testcases.  This handles
    # nested aggregates without adding parent and child counters together.
    for suite in suites:
        _validate_declared_counts(suite)
    if root.tag == "testsuites":
        _validate_declared_counts(root)

    counts = _testcase_counts(root)
    if counts.tests <= 0:
        raise JUnitValidationError(f"required lane ran no tests: {counts.as_dict()}")
    if counts.failures or counts.errors or counts.skipped:
        raise JUnitValidationError(
            f"required lane was not clean and zero-skip: {counts.as_dict()}"
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args()
    try:
        counts = verify_junit(args.junit_xml)
    except JUnitValidationError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"required lane clean: {counts.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
