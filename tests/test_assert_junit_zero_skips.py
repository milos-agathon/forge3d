"""Adversarial fixtures for the required-lane JUnit verifier."""

from pathlib import Path

import pytest

from scripts.assert_junit_zero_skips import JUnitValidationError, verify_junit


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(body, encoding="utf-8")
    return path


def test_zero_tests_rejected(tmp_path):
    path = _write(tmp_path, '<testsuite tests="0" failures="0" errors="0" skipped="0"/>')
    with pytest.raises(JUnitValidationError, match="no tests"):
        verify_junit(path)


def test_one_clean_test_accepted(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="clean"/></testsuite>',
    )
    assert verify_junit(path).as_dict() == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


@pytest.mark.parametrize(
    ("outcome", "counter"),
    [("failure", "failures"), ("error", "errors"), ("skipped", "skipped")],
)
def test_nonclean_outcome_rejected(tmp_path, outcome, counter):
    path = _write(
        tmp_path,
        f'<testsuite tests="1" failures="{int(counter == "failures")}" '
        f'errors="{int(counter == "errors")}" skipped="{int(counter == "skipped")}">'
        f'<testcase name="bad"><{outcome}/></testcase></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="not clean"):
        verify_junit(path)


def test_xfail_encoded_as_skip_rejected(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase name="xfail"><skipped type="pytest.xfail">expected</skipped>'
        '</testcase></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="zero-skip"):
        verify_junit(path)


@pytest.mark.parametrize("outcome", ["failure", "skipped"])
def test_clean_parent_cannot_hide_nested_nonclean_child(tmp_path, outcome):
    child_counter = "failures" if outcome == "failure" else "skipped"
    path = _write(
        tmp_path,
        '<testsuites tests="1" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="1" failures="0" errors="0" skipped="0">'
        f'<testsuite name="child" tests="1" failures="{int(child_counter == "failures")}" '
        f'errors="0" skipped="{int(child_counter == "skipped")}">'
        f'<testcase name="nested"><{outcome}/></testcase>'
        '</testsuite></testsuite></testsuites>',
    )
    with pytest.raises(JUnitValidationError, match="contradictory"):
        verify_junit(path)


def test_nested_aggregate_totals_are_not_double_counted(tmp_path):
    path = _write(
        tmp_path,
        '<testsuites tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="a" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="a"/></testsuite>'
        '<testsuite name="b" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="b"/></testsuite>'
        '</testsuite></testsuites>',
    )
    assert verify_junit(path).tests == 2


def test_missing_file_rejected(tmp_path):
    with pytest.raises(JUnitValidationError, match="does not exist"):
        verify_junit(tmp_path / "missing.xml")


def test_malformed_xml_rejected(tmp_path):
    path = _write(tmp_path, '<testsuite tests="1"><testcase>')
    with pytest.raises(JUnitValidationError, match="malformed"):
        verify_junit(path)


@pytest.mark.parametrize("value", ["one", "-1"])
def test_invalid_counters_rejected(tmp_path, value):
    path = _write(
        tmp_path,
        f'<testsuite tests="{value}" failures="0" errors="0" skipped="0">'
        '<testcase name="x"/></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="counter"):
        verify_junit(path)


def test_declared_and_actual_totals_must_agree(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase name="only"/></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="contradictory tests"):
        verify_junit(path)
