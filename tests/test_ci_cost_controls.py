from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_STEP = re.compile(
    r"(?ms)^      - (?:name|uses):.*?"
    r"(?=^      - (?:name|uses):|^  [A-Za-z0-9_-]+:|\Z)"
)


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:.*?(?=^  [A-Za-z0-9_-]+:|\Z)",
        workflow,
    )
    assert match, f"missing workflow job: {name}"
    return match.group()


def _artifact_step(workflow: str, artifact: str) -> str:
    for step in _upload_steps(workflow):
        if re.search(rf"(?m)^\s+name: {re.escape(artifact)}$", step):
            return step
    raise AssertionError(f"missing artifact upload: {artifact}")


def _upload_steps(workflow: str) -> list[str]:
    return [
        step
        for step in UPLOAD_STEP.findall(workflow)
        if "uses: actions/upload-artifact@v4" in step
    ]


def test_ci_cost_controls_are_scoped_and_retained() -> None:
    workflow = _workflow("ci.yml")

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^env:)", workflow)
    assert concurrency
    body = concurrency.group()
    assert "format('certificate-refresh-{0}', github.run_id)" in body
    assert "format('pr-{0}', github.event.pull_request.number)" in body
    assert "format('manual-{0}', github.ref_name)" in body
    assert "format('run-{0}', github.run_id)" in body
    assert (
        "cancel-in-progress: ${{ github.event_name == 'pull_request' || "
        "(github.event_name == 'workflow_dispatch' && !inputs.update_recipe_certificates) }}"
    ) in body

    matrix_line = re.search(
        r"matrix:\s*\$\{\{\s*fromJSON\(github\.event_name == 'pull_request' "
        r"&& '(?P<pr>\{[^']+\})' \|\| '(?P<non_pr>\{[^']+\})'\)\s*\}\}",
        workflow,
    )
    assert matrix_line, "test-python matrix contract is missing"
    assert json.loads(matrix_line.group("pr")) == {
        "include": [
            {"os": "ubuntu-latest", "python-version": "3.10"},
            {"os": "ubuntu-latest", "python-version": "3.11"},
            {"os": "ubuntu-latest", "python-version": "3.12"},
            {"os": "ubuntu-latest", "python-version": "3.13"},
            {"os": "windows-latest", "python-version": "3.11"},
            {"os": "macos-latest", "python-version": "3.11"},
        ]
    }
    assert json.loads(matrix_line.group("non_pr")) == {
        "os": ["windows-latest", "ubuntu-latest", "macos-latest"],
        "python-version": ["3.10", "3.11", "3.12", "3.13"],
    }

    assert "retention-days: 1" in _artifact_step(
        _job(workflow, "prepare-lfs-fixtures"), "lfs-fixture-bundles"
    )
    assert "retention-days: 1" in _artifact_step(
        _job(workflow, "build-wheels"), "wheels-${{ matrix.platform.os }}"
    )
    for job, artifact in (
        ("test-terminus-fuzz", "terminus-fuzzer-evidence"),
        ("test-golden-images", "golden-lane-marker"),
        ("test-golden-images", "visual-golden-diffs"),
    ):
        assert "retention-days: 7" in _artifact_step(_job(workflow, job), artifact)

    for job, artifact in (
        ("refresh-recipe-certificates", "refreshed-recipe-certificates"),
        ("test-m06-full-geospatial-viewer", "m06-full-geospatial-viewer-evidence"),
        ("test-f3dz-gpu", "f3dz-physical-gpu-evidence"),
        ("test-anamnesis-portability-seed", "anamnesis-physical-portable-store"),
        ("test-anamnesis-portability", "anamnesis-portability-evidence"),
        ("test-anamnesis-production", "anamnesis-production-evidence"),
    ):
        assert "retention-days: 90" in _artifact_step(_job(workflow, job), artifact)

    assert "sphinx" not in workflow.lower()
    assert "name: documentation" not in workflow
    assert "  build-docs:" in workflow
    assert "cargo doc --workspace" in _job(workflow, "build-docs")


def test_publish_cost_controls_are_tag_only_and_retention_aware() -> None:
    workflow = _workflow("publish.yml")
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "  pull_request:" not in trigger
    assert re.search(r"(?m)^  push:\n    tags:\n      - 'v\*'$", trigger)
    assert "  workflow_dispatch:" in trigger

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^jobs:)", workflow)
    assert concurrency
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in concurrency.group()
    assert "cancel-in-progress: false" in concurrency.group()
    assert "queue: max" in concurrency.group()

    publish = _job(workflow, "publish")
    condition = re.search(r"(?m)^    if: (.+)$", publish)
    assert condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition.group(1)
    assert "github.event_name == 'push'" in condition.group(1)
    assert "github.event_name == 'workflow_dispatch'" in condition.group(1)
    assert "github.event.inputs.dry_run == 'false'" in condition.group(1)

    expected = "retention-days: ${{ startsWith(github.ref, 'refs/tags/v') && 90 || 1 }}"
    uploads = _upload_steps(workflow)
    assert len(uploads) == 2
    assert all(expected in upload for upload in uploads)


def test_determinism_cost_controls_keep_pr_cancellation_and_evidence() -> None:
    workflow = _workflow("determinism-matrix.yml")
    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^env:)", workflow)
    assert concurrency
    body = concurrency.group()
    assert "format('pr-{0}', github.event.pull_request.number)" in body
    assert "format('run-{0}', github.run_id)" in body
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in body

    uploads = _upload_steps(workflow)
    assert len(uploads) == 5
    assert all("retention-days: 7" in upload for upload in uploads)
