import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_determinism_hashes.py"
DUPLA_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dupla_proof.py"
NVIDIA_RUNNER = (
    Path(__file__).parents[1] / "scripts" / "run_nvidia_determinism_acceptance.py"
)
CONTRACT_ROOT = Path(
    os.environ.get("FORGE3D_CI_CONTRACT_ROOT", Path(__file__).parents[1])
)
WORKFLOW = CONTRACT_ROOT / ".github" / "workflows" / "determinism-matrix.yml"
SCENE = "terra_determinata_v1"
SHA = "d" * 64
PROBE_SHA = "a" * 64
RASTER_SHA = "b" * 64


def _adapter(name="hardware", software_fallback=False):
    return {
        "name": name,
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "software_fallback": software_fallback,
    }


def _artifact(root, leg, *, sha=None, adapter=True, marker=None, probe=None):
    path = root / f"determinism-hash-{leg}"
    path.mkdir(parents=True)
    if sha:
        (path / f"{SCENE}.sha256").write_text(sha + "\n")
        record = {"scene": SCENE, "sha256": sha}
        if adapter:
            record["adapter"] = _adapter()
        if probe is not None:
            record["probe"] = {
                "status": "ok",
                "software_fallback": False,
                **probe,
            }
        (path / f"{SCENE}.json").write_text(json.dumps(record))
    if marker:
        (path / f"{SCENE}.{marker}").write_text(f"{leg} {marker.lower()}\n")


def _browser_artifact(root, probe_sha=PROBE_SHA, raster_sha=RASTER_SHA, software=False):
    path = root / "determinism-hash-browser"
    path.mkdir(parents=True)
    (path / "probe.sha256").write_text(probe_sha + "\n")
    (path / "raster.sha256").write_text(raster_sha + "\n")
    (path / "browser.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "adapter": {
                    "name": "SwiftShader" if software else "webgpu-adapter",
                    "backend": "BrowserWebGpu",
                    "device_type": "webgpu",
                    "software_fallback": software,
                },
                "raster_format": "rgba32float",
                "probe_sha256": probe_sha,
                "raster_sha256": raster_sha,
            }
        )
    )


def _run(tmp_path, golden=SHA, require=(), probe_golden=None, raster_golden=None):
    golden_file = tmp_path / "golden.sha256"
    if golden is not None:
        golden_file.write_text(golden + "\n")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--hashes",
        str(tmp_path / "hashes"),
        "--golden",
        str(golden_file),
        "--scene",
        SCENE,
    ]
    for leg in require:
        cmd += ["--require", leg]
    if probe_golden is not None:
        pf = tmp_path / "probe.sha256"
        pf.write_text(probe_golden + "\n")
        cmd += ["--probe-golden", str(pf)]
    if raster_golden is not None:
        rf = tmp_path / "raster.sha256"
        rf.write_text(raster_golden + "\n")
        cmd += ["--raster-golden", str(rf)]
    return subprocess.run(cmd, capture_output=True, text=True)


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


def test_missing_primary_golden_is_a_configuration_error(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    result = _run(tmp_path, golden=None)
    assert result.returncode == 2
    assert "--golden" in result.stderr


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


def test_matrix_rejects_virtualized_adapter_metadata(tmp_path):
    _artifact(tmp_path / "hashes", "apple", sha=SHA, adapter=True)
    meta = json.loads(
        (tmp_path / "hashes" / "determinism-hash-apple" / f"{SCENE}.json").read_text()
    )
    meta["adapter"]["name"] = "Apple Paravirtual device"
    (tmp_path / "hashes" / "determinism-hash-apple" / f"{SCENE}.json").write_text(
        json.dumps(meta)
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "not physical hardware" in result.stderr


def test_matrix_rejects_software_adapter_metadata(tmp_path):
    _artifact(tmp_path / "hashes", "amd", sha=SHA, adapter=True)
    meta = json.loads(
        (tmp_path / "hashes" / "determinism-hash-amd" / f"{SCENE}.json").read_text()
    )
    meta["adapter"]["software_fallback"] = True
    (tmp_path / "hashes" / "determinism-hash-amd" / f"{SCENE}.json").write_text(
        json.dumps(meta)
    )
    result = _run(tmp_path)
    assert result.returncode == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("name", 7),
        ("backend", ""),
        ("device_type", ""),
        ("device_type", "Cpu"),
        ("device_type", "VirtualGpu"),
        ("device_type", "Other"),
        ("software_fallback", 0),
        ("software_fallback", "false"),
    ],
)
def test_matrix_rejects_incomplete_or_nonphysical_adapter_metadata(
    tmp_path, field, value
):
    _artifact(tmp_path / "hashes", "amd", sha=SHA, adapter=True)
    meta_path = tmp_path / "hashes" / "determinism-hash-amd" / f"{SCENE}.json"
    meta = json.loads(meta_path.read_text())
    meta["adapter"][field] = value
    meta_path.write_text(json.dumps(meta))
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "adapter metadata" in result.stderr


def test_matrix_accepts_matching_hardware_hash_with_documented_gated_failure(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "apple", marker="FAILED")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "GATED-FAILURE" in result.stdout


def test_matrix_rejects_conflicting_native_result_artifacts(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    artifact = tmp_path / "hashes" / "determinism-hash-nvidia"
    (artifact / f"{SCENE}.FAILED").write_text("current render failed\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "conflicting result artifacts" in result.stderr


@pytest.mark.parametrize("actual", ["e" * 64, "f" * 64])
def test_matrix_rejects_golden_or_pairwise_mismatch(tmp_path, actual):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "amd", sha=actual)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "mismatch" in result.stderr


# --- required-leg enforcement -------------------------------------------------


def test_required_leg_absent_fails(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "apple", marker="ABSENT")
    result = _run(tmp_path, require=["apple"])
    assert result.returncode == 1
    assert "required leg 'apple'" in result.stderr


def test_required_leg_failed_fails(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "amd", marker="FAILED")
    result = _run(tmp_path, require=["amd"])
    assert result.returncode == 1
    assert "required leg 'amd'" in result.stderr


def test_required_leg_missing_artifact_fails(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    result = _run(tmp_path, require=["intel"])
    assert result.returncode == 1
    assert "required leg 'intel'" in result.stderr


def test_required_leg_paravirtual_failure_is_documented_absence(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    leg = tmp_path / "hashes" / "determinism-hash-apple"
    leg.mkdir(parents=True)
    (leg / f"{SCENE}.FAILED").write_text(
        "gated leg 'apple' failed to render\n"
        "FORGE3D_DETERMINISTIC: hypervisor-virtualized GPU refused\n"
    )
    result = _run(tmp_path, require=["apple"])
    assert result.returncode == 0, result.stderr
    assert "DOCUMENTED-ABSENCE" in result.stdout


def test_required_non_apple_leg_cannot_claim_paravirtual_exception(tmp_path):
    _artifact(tmp_path / "hashes", "amd", sha=SHA)
    leg = tmp_path / "hashes" / "determinism-hash-nvidia"
    leg.mkdir(parents=True)
    (leg / f"{SCENE}.FAILED").write_text(
        "gated leg 'nvidia' failed to render\n"
        "hypervisor-virtualized GPU refused\n"
    )
    result = _run(tmp_path, require=["nvidia"])
    assert result.returncode == 1
    assert "required leg 'nvidia'" in result.stderr


def test_required_leg_produced_passes(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "amd", sha=SHA)
    result = _run(tmp_path, require=["amd", "nvidia"])
    assert result.returncode == 0, result.stderr


# --- browser / canary legs ----------------------------------------------------


def test_required_browser_leg_missing_fails(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    result = _run(tmp_path, require=["browser"])
    assert result.returncode == 1
    assert "required leg 'browser'" in result.stderr


def test_browser_leg_canaries_checked_against_goldens(tmp_path):
    _artifact(
        tmp_path / "hashes",
        "nvidia",
        sha=SHA,
        probe={"sha256": PROBE_SHA, "raster_sha256": RASTER_SHA},
    )
    _browser_artifact(tmp_path / "hashes")
    result = _run(
        tmp_path,
        require=["browser"],
        probe_golden=PROBE_SHA,
        raster_golden=RASTER_SHA,
    )
    assert result.returncode == 0, result.stderr
    assert "legs with produced evidence: 2/2" in result.stdout


def test_browser_leg_probe_mismatch_fails(tmp_path):
    _artifact(
        tmp_path / "hashes",
        "nvidia",
        sha=SHA,
        probe={"sha256": PROBE_SHA},
    )
    _browser_artifact(tmp_path / "hashes", probe_sha="9" * 64)
    result = _run(tmp_path, require=["browser"], probe_golden=PROBE_SHA)
    assert result.returncode == 1
    assert "probe" in result.stderr


def test_browser_leg_raster_mismatch_fails(tmp_path):
    _browser_artifact(tmp_path / "hashes", raster_sha="9" * 64)
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    result = _run(tmp_path, require=["browser"], raster_golden=RASTER_SHA)
    assert result.returncode == 1
    assert "raster" in result.stderr


def test_browser_leg_rejects_failed_metadata_with_stale_hashes(tmp_path):
    _browser_artifact(tmp_path / "hashes")
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    meta_path = tmp_path / "hashes" / "determinism-hash-browser" / "browser.json"
    meta = json.loads(meta_path.read_text())
    meta["status"] = "failed"
    meta_path.write_text(json.dumps(meta))
    result = _run(
        tmp_path,
        require=["browser"],
        probe_golden=PROBE_SHA,
        raster_golden=RASTER_SHA,
    )
    assert result.returncode == 1
    assert "browser probe status" in result.stderr


def test_browser_leg_requires_float32_raster_evidence(tmp_path):
    _browser_artifact(tmp_path / "hashes")
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    meta_path = tmp_path / "hashes" / "determinism-hash-browser" / "browser.json"
    meta = json.loads(meta_path.read_text())
    meta["raster_format"] = "rgba8unorm"
    meta_path.write_text(json.dumps(meta))
    result = _run(
        tmp_path,
        require=["browser"],
        probe_golden=PROBE_SHA,
        raster_golden=RASTER_SHA,
    )
    assert result.returncode == 1
    assert "not byte-comparable rgba32float" in result.stderr


def test_browser_leg_binds_hash_sidecars_to_metadata(tmp_path):
    _browser_artifact(tmp_path / "hashes")
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    meta_path = tmp_path / "hashes" / "determinism-hash-browser" / "browser.json"
    meta = json.loads(meta_path.read_text())
    meta["probe_sha256"] = "9" * 64
    meta_path.write_text(json.dumps(meta))
    result = _run(
        tmp_path,
        require=["browser"],
        probe_golden=PROBE_SHA,
        raster_golden=RASTER_SHA,
    )
    assert result.returncode == 1
    assert "does not match its canary sidecars" in result.stderr


def test_produced_leg_missing_probe_fails_when_golden_given(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)  # no probe record
    result = _run(tmp_path, probe_golden=PROBE_SHA)
    assert result.returncode == 1
    assert "no probe_sha256 canary" in result.stderr


def test_native_probe_software_fallback_contradicts_hardware_render(tmp_path):
    _artifact(
        tmp_path / "hashes",
        "nvidia",
        sha=SHA,
        probe={"sha256": PROBE_SHA, "software_fallback": True},
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "software adapter" in result.stderr


# --- NVIDIA acceptance runner -------------------------------------------------


def _nvidia_adapter(*, device=0x2484):
    return {
        "name": "NVIDIA GeForce RTX 3070",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "device": device,
        "software_fallback": False,
    }


def _nvidia_probe(tmp_path):
    path = tmp_path / "nvidia-adapter-probe.json"
    path.write_text(
        json.dumps({"requested_backend": "vulkan", "probe": _nvidia_adapter()})
    )
    return path


def _run_nvidia_acceptance(monkeypatch, tmp_path, render_payloads):
    from scripts import run_nvidia_determinism_acceptance as runner

    payloads = iter(render_payloads)

    def fake_run(command, **_kwargs):
        pixels, adapter = next(payloads)
        png_path = Path(command[command.index("--out-png") + 1])
        png_path.write_bytes(pixels)
        record = {
            "scene": SCENE,
            "sha256": hashlib.sha256(pixels).hexdigest(),
            "adapter": adapter,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(record) + "\n", stderr=""
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setenv("WGPU_BACKENDS", "vulkan")
    artifact_dir = tmp_path / "hash-out"
    result = runner.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--adapter-probe",
            str(_nvidia_probe(tmp_path)),
            "--scene",
            SCENE,
            "--width",
            "512",
            "--height",
            "512",
        ]
    )
    return result, artifact_dir


def test_nvidia_acceptance_binds_two_matching_frames_to_probe(monkeypatch, tmp_path):
    pixels = b"physical-nvidia-vulkan-frame"
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [(pixels, _nvidia_adapter()), (pixels, _nvidia_adapter())],
    )
    assert result == 0
    assert (artifact_dir / f"{SCENE}.sha256").is_file()
    assert (artifact_dir / f"{SCENE}.repeat.sha256").is_file()
    assert (artifact_dir / f"{SCENE}.json").is_file()
    assert (artifact_dir / f"{SCENE}.repeat.json").is_file()
    assert not (artifact_dir / f"{SCENE}.FAILED").exists()


def test_nvidia_acceptance_rejects_probe_render_identity_drift(
    monkeypatch, tmp_path
):
    pixels = b"physical-nvidia-vulkan-frame"
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [(pixels, _nvidia_adapter(device=0x2684)), (pixels, _nvidia_adapter())],
    )
    assert result == 1
    failure = (artifact_dir / f"{SCENE}.FAILED").read_text()
    assert "first render/probe adapter differs at device" in failure


def test_nvidia_acceptance_rejects_repeat_hash_drift(monkeypatch, tmp_path):
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [
            (b"physical-nvidia-vulkan-frame-a", _nvidia_adapter()),
            (b"physical-nvidia-vulkan-frame-b", _nvidia_adapter()),
        ],
    )
    assert result == 1
    failure = (artifact_dir / f"{SCENE}.FAILED").read_text()
    assert "repeat hash differs" in failure


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "Dx12"),
        ("device_type", "IntegratedGpu"),
        ("vendor", 0x1002),
        ("name", "AMD Radeon RX 7900 XT"),
        ("device", "not-a-device"),
        ("device", -1),
        ("software_fallback", True),
    ],
)
def test_nvidia_acceptance_requires_strict_physical_vulkan_identity(
    monkeypatch, tmp_path, field, value
):
    """The dedicated NVIDIA leg enforces vendor+backend identity itself — the
    generic checker is vendor-agnostic, so this is where the NVIDIA/Vulkan
    contract lives."""
    adapter = _nvidia_adapter()
    adapter[field] = value
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [(b"physical-nvidia-vulkan-frame", adapter), (b"physical-nvidia-vulkan-frame", adapter)],
    )
    assert result == 1
    assert (artifact_dir / f"{SCENE}.FAILED").is_file()


def test_determinism_record_persists_strict_adapter_identity(
    monkeypatch, tmp_path, capsys
):
    import forge3d as f3d
    from forge3d import determinism

    monkeypatch.setenv("WGPU_BACKENDS", "vulkan")
    monkeypatch.setattr(
        determinism,
        "_render_reference_inprocess",
        lambda *_args, **_kwargs: SHA,
    )
    monkeypatch.setattr(
        f3d,
        "device_probe",
        lambda _backend=None: {
            "status": "ok",
            "name": "NVIDIA GeForce RTX 3070",
            "backend": "Vulkan",
            "device_type": "DiscreteGpu",
            "vendor": 0x10DE,
            "device": 0x2484,
            "software_fallback": False,
        },
    )
    # The record shape is under test, not the canary — and a real probe would
    # init the GPU context under FORGE3D_DETERMINISTIC, latching deterministic
    # mode process-wide and leaking it into later tests.
    monkeypatch.setattr(
        f3d,
        "determinism_probe",
        lambda: {
            "status": "ok",
            "probe_sha256": PROBE_SHA,
            "raster_sha256": RASTER_SHA,
        },
    )

    assert determinism._main(["--out-png", str(tmp_path / "unused.png")]) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["adapter"] == {
        "name": "NVIDIA GeForce RTX 3070",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "device": 0x2484,
        "software_fallback": False,
    }


# --- workflow shape -----------------------------------------------------------


def test_render_acceptance_required_legs():
    workflow = WORKFLOW.read_text()
    nvidia = workflow.split("  render-nvidia:\n", 1)[1].split(
        "\n  browser:\n", 1
    )[0]
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in nvidia
    assert "name: wheels-windows" in nvidia
    assert "shell: bash" not in nvidia
    assert "shell: pwsh" in nvidia
    assert "terrain_ci_probe.py" in nvidia
    assert "--require-nvidia-vulkan" in nvidia
    assert "run_nvidia_determinism_acceptance.py" in nvidia
    runner = NVIDIA_RUNNER.read_text()
    assert '_render_once(args, artifact_dir, "first")' in runner
    assert '_render_once(args, artifact_dir, "repeat")' in runner
    assert "render/probe" in runner
    assert "repeat hash differs" in runner
    # The diff gate hard-requires the attributable legs and pins the canary
    # hashes to committed goldens.
    for flag in (
        "--require nvidia",
        "--require apple",
        "--require browser",
        "--probe-golden tests/goldens/determinism/",
        "--raster-golden tests/goldens/determinism/",
    ):
        assert flag in workflow
    # A required leg's artifact must come from a job the diff gate waits on;
    # otherwise --require apple races the metal-diagnostic upload.
    summary_needs = workflow.split("  diff:\n", 1)[1].split("\n    if:", 1)[0]
    assert "metal-diagnostic" in summary_needs
    assert "name: determinism-hash-apple" in workflow.split("  metal-diagnostic:\n", 1)[1]
    assert "--expected-legs intel amd nvidia" in workflow


def test_matrix_reuses_caller_wheels_instead_of_rebuilding_extensions():
    workflow = WORKFLOW.read_text()
    assert "  workflow_call:" in workflow
    assert "maturin develop" not in workflow
    assert "PyO3/maturin-action" not in workflow
    for artifact in ("wheels-linux", "wheels-windows"):
        assert artifact in workflow
    render = workflow.split("  render:\n", 1)[1].split("\n  render-nvidia:\n", 1)[0]
    # The apple matrix leg resolves wheels-macos via the runner.os expression.
    assert "runner.os == 'macOS' && 'macos'" in render
    assert "ref: ${{ inputs.ref }}" in workflow
    assert "FORGE3D_NO_BOOTSTRAP: '1'" in workflow
    anamnesis = workflow.split("  anamnesis-seed:", 1)[1].split(
        "\n  anamnesis-portability:", 1
    )[0]
    assert anamnesis.index("Classify the hosted Vulkan adapter") < anamnesis.index(
        "Gate ANAMNESIS incrementality"
    )
    summary = workflow.split("  diff:", 1)[1]
    for family in (
        "f3dz-stream",
        "render",
        "render-nvidia",
        "browser",
        "wasm-policy",
        "anamnesis-seed",
        "anamnesis-portability",
    ):
        assert f"needs.{family}.result" in summary


def test_shadow_shader_classifier_is_retained_without_pr_path_gating():
    ci = (WORKFLOW.parent / "ci.yml").read_text()
    classifier = ci.split("            determinism_render:\n", 1)[1].split(
        "\n            determinism_f3dz:\n", 1
    )[0]
    assert "'src/shaders/shadows.wgsl'" in classifier
    assert "'src/shaders/includes/shadow_moments.wgsl'" in classifier
    assert "'src/shaders/csm.wgsl'" not in classifier

    caller = ci.split("  determinism-render:", 1)[1].split(
        "\n  determinism-f3dz:", 1
    )[0]
    assert "needs.terrain-golden-paths.outputs.determinism_render" not in caller
    assert "github.event_name == 'schedule'" in caller
    assert "inputs.scope == 'full'" in caller
    assert "inputs.scope == 'determinism'" in caller
    assert "github.event_name == 'pull_request'" not in caller
    assert "uses: ./.github/workflows/determinism-matrix.yml" in caller
    assert "run_render: true" in caller


def test_f3dz_stream_hashes_run_on_two_hosted_platforms():
    workflow = WORKFLOW.read_text()
    assert "f3dz-stream:" in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert "tools/f3dz_determinism_report.py" in workflow
    assert "test_error_bound_stored_page_error_nan_and_determinism" in workflow
    assert "test_cross_platform_determinism_hashes" in workflow
    assert "f3dz-determinism-${{ matrix.os }}" in workflow


def test_browser_job_executes_real_webgpu_leg():
    """The browser leg must run real canaries, not a policy-only placeholder."""
    workflow = WORKFLOW.read_text()
    assert re.search(r"^\s+browser:\s*$", workflow, re.MULTILINE)
    assert "scripts/run_browser_probe.py" in workflow
    assert "determinism-hash-browser" in workflow
    assert "--require browser" in workflow
    browser = workflow.split("  browser:\n", 1)[1].split("\n  wasm-policy:\n", 1)[0]
    assert "set -o pipefail" in browser
    assert "status=$?" in browser
    # The old policy-only ABSENT artifact is gone.
    # The browser leg above is the real WebGPU evidence, but the wasm hash
    # artifact cannot be removed: the base-owned preflight contract
    # (tests/test_ci_cost_controls.py, judged from main) requires an upload
    # named "determinism-hash-wasm" with retention-days: 7, and a candidate
    # cannot edit that judge. The policy-only marker therefore stays as an
    # extra artifact rather than replacing the browser leg.
    assert "no browser WebGPU device" not in workflow


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
