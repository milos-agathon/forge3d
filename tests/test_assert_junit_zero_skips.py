"""Adversarial fixtures for the required-lane JUnit verifier."""

import json
from pathlib import Path
import sys

import pytest

from scripts.assert_junit_zero_skips import JUnitValidationError, verify_junit
from scripts.summarize_m06_evidence import (
    build_summary,
    github_notice,
    main as summarize_m06_main,
    markdown_summary,
    write_summary,
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(body, encoding="utf-8")
    return path


def test_zero_tests_rejected(tmp_path):
    path = _write(
        tmp_path, '<testsuite tests="0" failures="0" errors="0" skipped="0"/>'
    )
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
        "</testcase></testsuite>",
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
        "</testsuite></testsuite></testsuites>",
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
        "</testsuite></testsuites>",
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


def test_m06_evidence_summary_extracts_adapter_and_junit_counts(tmp_path):
    (tmp_path / "run-context.json").write_text(
        json.dumps(
            {
                "repository": "milos-agathon/forge3d",
                "head_sha": "abc123",
                "run_id": "42",
                "run_attempt": "1",
                "runner_name": "forge3d-rtx3070",
                "runner_os": "Windows",
                "runner_arch": "X64",
                "required_labels": [
                    "self-hosted",
                    "Windows",
                    "forge3d-gpu",
                    "gpu-nvidia",
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checked-out-head.txt").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "adapter-probe.json").write_text(
        json.dumps(
            {
                "requested_backend": "vulkan",
                "probe": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "device": 1234,
                    "backend": "Vulkan",
                    "device_type": "DiscreteGpu",
                    "driver": "test-driver",
                    "driver_info": "test-info",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "viewer-adapter.json").write_text(
        json.dumps(
            {
                "validated_identity": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "device": 1234,
                    "backend": "vulkan",
                    "device_type": "discretegpu",
                    "driver": "test-driver",
                    "driver_info": "test-info",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="one"/>'
        '<testcase classname="a" name="two"/>'
        "</testsuite>",
        encoding="utf-8",
    )

    summary = write_summary(tmp_path)

    assert summary["status"] == "pass"
    assert summary["exact_head"] is True
    assert summary["adapter"]["vendor_hex"] == "0x10de"
    assert summary["adapter"]["backend"] == "vulkan"
    assert summary["junit"] == {
        "exists": True,
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "zero_skip_clean": True,
    }
    assert (tmp_path / "m06-public-evidence-summary.json").is_file()
    rendered = markdown_summary(build_summary(tmp_path))
    assert "tests=2 failures=0 errors=0 skipped=0" in rendered
    assert "vendor=0x10de backend=vulkan" in rendered
    annotation = github_notice(summary)
    assert annotation.startswith("::notice title=M-06 exact-head evidence::")
    assert "head_sha=abc123" in annotation
    assert "checked_out_head=abc123 exact_head=true" in annotation
    assert "adapter=NVIDIA GeForce RTX 3070" in annotation
    assert "tests=2 failures=0 errors=0 skipped=0" in annotation


def test_m06_evidence_summary_fails_closed_on_synthetic_merge_checkout(
    tmp_path, monkeypatch
):
    (tmp_path / "run-context.json").write_text(
        json.dumps({"head_sha": "pr-head"}), encoding="utf-8"
    )
    (tmp_path / "checked-out-head.txt").write_text(
        "synthetic-merge\n", encoding="utf-8"
    )
    (tmp_path / "adapter-probe.json").write_text(
        json.dumps(
            {
                "probe": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "backend": "vulkan",
                    "device_type": "discretegpu",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="m06" name="acceptance"/>'
        "</testsuite>",
        encoding="utf-8",
    )

    summary = build_summary(tmp_path)

    assert summary["exact_head"] is False
    assert summary["status"] == "incomplete"
    assert "exact_head=false" in github_notice(summary)
    monkeypatch.setattr(sys, "argv", ["summarize_m06_evidence.py", str(tmp_path)])
    assert summarize_m06_main() == 1


def test_ci_checkout_steps_pin_pull_requests_to_the_exact_head():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
    checkout_steps = workflow.split("- uses: actions/checkout@v4")[1:]
    exact_ref = "ref: ${{ github.event.pull_request.head.sha || github.sha }}"

    assert len(checkout_steps) == 12
    for index, tail in enumerate(checkout_steps, start=1):
        step = tail.split("\n\n", 1)[0]
        assert exact_ref in step, f"checkout step {index} is not exact-head pinned"


def test_ci_materializes_lfs_fixtures_once_and_verifies_every_consumer():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("git lfs pull") == 1
    assert "lfs: true" not in workflow
    assert workflow.count("name: Restore materialized LFS fixtures") == 4
    assert workflow.count("name: Verify materialized LFS fixtures") == 4
    assert "needs: [build-wheels, stage-lfs-fixtures]" in workflow
    assert "needs: [build-wheels, terrain-golden-paths, stage-lfs-fixtures]" in workflow
