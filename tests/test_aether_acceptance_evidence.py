"""Adversarial checks for durable AETHER physical-acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aether_acceptance_evidence import (
    EXPECTED_DELTA_E_KEYS,
    REQUIRED_JUNIT_CASES,
    build_summary,
)


HEAD = "a" * 40
ROOT = Path(__file__).resolve().parents[1]


def _fixture(directory: Path) -> None:
    (directory / "checked-out-head.txt").write_text(HEAD + "\n", encoding="utf-8")
    (directory / "adapter-probe.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "mode": "aether-metal",
                "requested_backend": "metal",
                "probe": {
                    "status": "ok",
                    "name": "Apple M4",
                    "backend": "Metal",
                    "device_type": "IntegratedGpu",
                    "software_fallback": False,
                },
            }
        ),
        encoding="utf-8",
    )
    cases = "".join(
        f'<testcase classname="{classname}" name="{name}"/>'
        for classname, name in sorted(REQUIRED_JUNIT_CASES)
    )
    (directory / "junit.xml").write_text(
        f'<testsuite tests="2" failures="0" errors="0" skipped="0">{cases}</testsuite>',
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "delta_e_sweep": {key: 1.0 for key in EXPECTED_DELTA_E_KEYS},
                "saturation_falloff": {
                    "distances_m": [20_000.0, 40_000.0],
                    "measured_saturation": [0.8, 0.72],
                    "predicted_saturation": [0.8, 0.72],
                    "measured_far_near_ratio": 0.9,
                    "predicted_far_near_ratio": 0.9,
                    "relative_error": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _summary(directory: Path, *, head_sha: str = HEAD):
    return build_summary(
        directory,
        head_sha=head_sha,
        repository="milos-agathon/forge3d",
        run_id="42",
        run_attempt="1",
        job="test-golden-images",
    )


def test_summary_binds_clean_thresholds_adapter_and_exact_head(tmp_path: Path) -> None:
    _fixture(tmp_path)
    summary = _summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["head_sha"] == HEAD
    assert summary["junit"] == {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    assert summary["thresholds"]["max_delta_e_2000"] == 1.0
    assert summary["thresholds"]["saturation_relative_error"] == 0.0
    assert all(len(digest) == 64 for digest in summary["sha256"].values())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data["probe"].update(software_fallback=True), "software fallback"),
        (lambda data: data.update(status="absent"), "did not pass"),
    ],
)
def test_summary_rejects_unproven_adapter(tmp_path: Path, mutation, match: str) -> None:
    _fixture(tmp_path)
    path = tmp_path / "adapter-probe.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _summary(tmp_path)


NVIDIA_VULKAN_RECORD = {
    "status": "passed",
    "mode": "aether-vulkan",
    "requested_backend": "vulkan",
    "probe": {
        "status": "ok",
        "name": "NVIDIA GeForce RTX 3070",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "software_fallback": False,
    },
}


def _write_adapter(directory: Path, record: dict) -> None:
    (directory / "adapter-probe.json").write_text(json.dumps(record), encoding="utf-8")


def test_summary_accepts_physical_nvidia_vulkan_closure(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write_adapter(tmp_path, NVIDIA_VULKAN_RECORD)
    summary = _summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["backend"] == "vulkan"
    assert summary["adapter"]["name"] == "NVIDIA GeForce RTX 3070"
    assert summary["thresholds"]["delta_e_2000_strict_upper_bound"] == 2.0
    assert summary["thresholds"]["saturation_relative_error_upper_bound"] == 0.10


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data["probe"].update(vendor=0x1002, name="AMD Radeon RX 6800"), "not an NVIDIA"),
        (lambda data: data["probe"].update(device_type="IntegratedGpu"), "not a discrete"),
        (lambda data: data["probe"].update(software_fallback=True), "software fallback"),
        (lambda data: data["probe"].update(backend="Dx12"), "not a successful Vulkan"),
        (lambda data: data.update(requested_backend="metal"), "requested backend must be Vulkan"),
        (lambda data: data.update(mode="aether-dx12"), "did not pass"),
    ],
)
def test_summary_rejects_unqualified_vulkan_adapter(tmp_path: Path, mutation, match: str) -> None:
    _fixture(tmp_path)
    record = json.loads(json.dumps(NVIDIA_VULKAN_RECORD))
    mutation(record)
    _write_adapter(tmp_path, record)
    with pytest.raises(ValueError, match=match):
        _summary(tmp_path)


def test_workflow_runs_required_aether_closure_on_nvidia_vulkan() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  test-golden-images-nvidia:\n", 1)[1].split("\n  test-substratia-gpu:\n", 1)[0]
    assert "aether_lane: ${{ steps.aether-nvidia-ran.outputs.aether_lane || 'unknown' }}" in job
    blocks: dict[str, str] = {}
    for step_name in (
        "Probe AETHER NVIDIA Vulkan closure backend",
        "Verify exact AETHER wheel custom-bake provenance on NVIDIA Vulkan",
        "Run AETHER NVIDIA Vulkan closure and golden gates",
    ):
        marker = f"      - name: {step_name}"
        assert marker in job
        block = job.split(marker, 1)[1].split("\n      - name:", 1)[0]
        assert "!cancelled()" in block, f"{step_name} can be suppressed by an unrelated failure"
        assert "continue-on-error" not in block
        blocks[step_name] = block
    probe = blocks["Probe AETHER NVIDIA Vulkan closure backend"]
    closure = blocks["Run AETHER NVIDIA Vulkan closure and golden gates"]
    assert "--mode aether-vulkan" in probe
    assert "steps.aether_nvidia_custom_bake.outcome == 'success'" in closure
    assert "FORGE3D_AETHER_PHYSICAL_BACKEND: vulkan" in closure
    assert "FORGE3D_AETHER_PHYSICAL_REQUIRED: '1'" in closure
    for test_file in (
        "tests/test_atmosphere_spectral.py",
        "tests/test_atmosphere_reference.py",
        "tests/test_atmosphere_pt_reference.py",
        "tests/test_atmosphere_golden.py",
        "tests/test_atmosphere_lut_handoff.py",
    ):
        assert test_file in closure
    assert "python scripts/assert_junit_zero_skips.py" in closure
    assert "python scripts/aether_acceptance_evidence.py" in closure
    assert '"aether_lane=ran"' in closure
    assert "ABSENT" not in closure

    summary = workflow.split("\n  full-acceptance-summary:\n", 1)[1]
    assert "needs.test-golden-images-nvidia.outputs.aether_lane" in summary
    assert "AETHER physical NVIDIA Vulkan closure lane was selected but reported" in summary

    bake = workflow.split("      - name: Run AETHER offline-bake acceptance tests", 1)[1].split("\n      - name:", 1)[0]
    assert "!cancelled()" in bake


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("delta_e_sweep", {key: 2.0 for key in EXPECTED_DELTA_E_KEYS}, "below 2"),
        (
            "saturation_falloff",
            {
                "distances_m": [20_000.0, 40_000.0],
                "measured_saturation": [0.8, 0.64],
                "predicted_saturation": [0.8, 0.72],
                "measured_far_near_ratio": 0.8,
                "predicted_far_near_ratio": 0.9,
                "relative_error": 1.0 / 9.0,
            },
            "exceeds 10",
        ),
    ],
)
def test_summary_rejects_threshold_failure(
    tmp_path: Path, field: str, value, match: str
) -> None:
    _fixture(tmp_path)
    path = tmp_path / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _summary(tmp_path)


def test_summary_rejects_exact_head_mismatch_or_skip(tmp_path: Path) -> None:
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        _summary(tmp_path, head_sha="b" * 40)
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase name="missing"><skipped/></testcase></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero-skip"):
        _summary(tmp_path)


def test_summary_rejects_incomplete_sweep_or_missing_gate_case(tmp_path: Path) -> None:
    _fixture(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["delta_e_sweep"].pop(next(iter(EXPECTED_DELTA_E_KEYS)))
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="topology mismatch"):
        _summary(tmp_path)

    _fixture(tmp_path)
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_atmosphere_reference" name="renamed-delta"/>'
        '<testcase classname="tests.test_atmosphere_reference" name="renamed-saturation"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gate cases are missing"):
        _summary(tmp_path)


def test_summary_recomputes_saturation_and_requires_schema(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["saturation_falloff"]["relative_error"] = 0.01
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="relative error is inconsistent"):
        _summary(tmp_path)

    _fixture(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        _summary(tmp_path)


def test_summary_rejects_vacuous_saturation_falloff(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["saturation_falloff"] = {
        "distances_m": [20_000.0, 40_000.0],
        "measured_saturation": [0.8, 0.799999],
        "predicted_saturation": [0.8, 0.799999],
        "measured_far_near_ratio": 0.799999 / 0.8,
        "predicted_far_near_ratio": 0.799999 / 0.8,
        "relative_error": 0.0,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the gate domain"):
        _summary(tmp_path)


def test_workflow_retains_exact_head_aether_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "FORGE3D_AETHER_METRICS_PATH: tests/artifacts/aether/metrics.json" in workflow
    assert "git rev-parse HEAD > tests/artifacts/aether/checked-out-head.txt" in workflow
    assert "python scripts/aether_acceptance_evidence.py tests/artifacts/aether" in workflow
    assert "--head-sha '${{ github.sha }}'" in workflow
    assert "name: aether-physical-metal-evidence" in workflow
    assert "path: tests/artifacts/aether/" in workflow
    assert "if-no-files-found: error" in workflow

    # An unrelated signed-golden failure must not suppress AETHER's own
    # physical proof or its honest ABSENT marker. `!cancelled()` bypasses an
    # earlier failure without defeating operator cancellation, while uploads
    # deliberately retain diagnostics under `always()`.
    executable_steps = (
        "Probe AETHER physical Metal closure backend",
        "Verify exact AETHER wheel custom-bake provenance",
        "Run AETHER Metal closure and golden gates",
        "Record AETHER Metal lane ABSENT",
        "Record AETHER Metal lane ABSENT under software override",
    )
    blocks: dict[str, str] = {}
    for step_name in executable_steps:
        marker = f"      - name: {step_name}"
        assert marker in workflow
        block = workflow.split(marker, 1)[1].split("\n      - name:", 1)[0]
        blocks[step_name] = block
        assert "!cancelled()" in block, f"{step_name} can be suppressed"
        assert "if: always()" not in block
        assert "continue-on-error: true" not in block

    provenance = blocks["Verify exact AETHER wheel custom-bake provenance"]
    closure = blocks["Run AETHER Metal closure and golden gates"]
    assert "id: aether_custom_bake" in provenance
    assert "steps.aether_custom_bake.outcome == 'success'" in closure

    for step_name in (
        "Upload AETHER physical Metal evidence",
        "Upload AETHER Metal lane marker",
    ):
        marker = f"      - name: {step_name}"
        assert marker in workflow
        block = workflow.split(marker, 1)[1].split("\n      - name:", 1)[0]
        assert "if: always()" in block, f"{step_name} drops diagnostics"
