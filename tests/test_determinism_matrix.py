import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_determinism_hashes.py"
DUPLA_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dupla_proof.py"
WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "determinism-matrix.yml"
SCENE = "terra_determinata_v1"
SHA = "d" * 64


def _artifact(root, leg, *, sha=None, adapter=True, marker=None):
    path = root / f"determinism-hash-{leg}"
    path.mkdir(parents=True)
    if sha:
        (path / f"{SCENE}.sha256").write_text(sha + "\n")
        if adapter:
            (path / f"{SCENE}.json").write_text(
                json.dumps(
                    {
                        "adapter": {
                            "name": "hardware",
                            "backend": "Vulkan",
                            "device_type": "DiscreteGpu",
                            "software_fallback": False,
                        }
                    }
                )
            )
    if marker:
        (path / f"{SCENE}.{marker}").write_text(f"{leg} {marker.lower()}\n")


def _run(tmp_path, golden=SHA):
    golden_file = tmp_path / "golden.sha256"
    if golden is not None:
        golden_file.write_text(golden + "\n")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hashes",
            str(tmp_path / "hashes"),
            "--golden",
            str(golden_file),
            "--scene",
            SCENE,
        ],
        capture_output=True,
        text=True,
    )


def _dupla_proof() -> dict:
    operation = {
        "generated_count": 100_000_000,
        "adversarial_count": 1_000_000,
        "mismatch_count": 0,
        "max_err_u2": 1.0,
        "cited_bound_u2": 3.0,
    }
    return {
        "schema": "forge3d.dupla-proof.v1",
        "backend": "vulkan",
        "adapter": "hardware",
        "selftest": {"passed": True, "mismatch_count": 0},
        "harness": {name: dict(operation) for name in ("add", "mul", "div", "sqrt")},
        "jitter": {
            "dd_max_error_px": 0.001,
            "raw_over_one_px": 100,
            "dd_hash_a": SHA,
            "dd_hash_b": SHA,
        },
    }


def _validate_dupla(tmp_path, *legs):
    return subprocess.run(
        [
            sys.executable,
            str(DUPLA_SCRIPT),
            "--validate-artifacts",
            str(tmp_path),
            "--expected-legs",
            *legs,
        ],
        capture_output=True,
        text=True,
    )


def test_matrix_rejects_zero_hardware_hashes(tmp_path):
    (tmp_path / "hashes").mkdir()
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "no hardware-backed leg produced a hash" in result.stderr


def test_hosted_apple_virtual_adapter_is_a_nonblocking_gated_failure():
    workflow = WORKFLOW.read_text()
    apple = re.search(r"- leg: apple.*?gated: (true|false)", workflow, re.DOTALL)
    assert apple, "apple render leg missing"
    assert apple.group(1) == "true"
    assert 'if [ "${{ matrix.leg }}" = "apple" ] && grep -q "hypervisor-virtualized GPU" render.err; then' in workflow


def test_matrix_accepts_documented_gated_infrastructure_failure(tmp_path):
    _artifact(tmp_path / "hashes", "apple", marker="FAILED")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "GATED-FAILURE" in result.stdout


def test_matrix_rejects_unattributed_hash(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA, adapter=False)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "missing attributable adapter metadata" in result.stderr


def test_matrix_accepts_matching_hardware_hash_with_documented_gated_failure(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "apple", marker="FAILED")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "GATED-FAILURE" in result.stdout


@pytest.mark.parametrize("actual", ["e" * 64, "f" * 64])
def test_matrix_rejects_golden_or_pairwise_mismatch(tmp_path, actual):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "amd", sha=actual)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "mismatch" in result.stderr


def test_dupla_aggregation_accepts_verified_and_explicit_absence(tmp_path):
    (tmp_path / "dupla-proof-nvidia.json").write_text(json.dumps(_dupla_proof()))
    (tmp_path / "dupla-proof-intel.ABSENT").write_text("no physical adapter\n")
    result = _validate_dupla(tmp_path, "nvidia", "intel")
    assert result.returncode == 0, result.stderr
    assert "VERIFIED" in result.stdout and "ABSENT" in result.stdout


def test_dupla_aggregation_rejects_failed_or_missing_proof(tmp_path):
    (tmp_path / "dupla-proof-amd.FAILED").write_text("bound exceeded\n")
    failed = _validate_dupla(tmp_path, "amd")
    assert failed.returncode == 1
    assert "DUPLA proof failed" in failed.stderr
    missing = _validate_dupla(tmp_path, "nvidia")
    assert missing.returncode == 1
    assert "expected exactly one DUPLA result" in missing.stderr


def test_dupla_aggregation_rejects_invalid_evidence(tmp_path):
    proof = _dupla_proof()
    proof["harness"]["mul"]["max_err_u2"] = 8.0
    proof["harness"]["mul"]["cited_bound_u2"] = 7.0
    (tmp_path / "dupla-proof-nvidia.json").write_text(json.dumps(proof))
    result = _validate_dupla(tmp_path, "nvidia")
    assert result.returncode == 1
    assert "cited bound exceeded" in result.stderr
