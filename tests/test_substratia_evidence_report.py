"""Negative controls for the SUBSTRATIA physical-evidence verifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "substratia_evidence_report", ROOT / "scripts" / "substratia_evidence_report.py"
)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _png(pixels: np.ndarray) -> bytes:
    height, width, channels = pixels.shape
    assert channels == 4 and pixels.dtype == np.uint8
    scanlines = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _chunk(b"IEND", b"")
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _junit(status: str = "pass", *, omitted: str | None = None) -> str:
    cases = []
    for name in reporter.CORE_TESTS:
        if name == omitted:
            continue
        child = ""
        if name == reporter.CORE_TESTS[0] and status != "pass":
            child = f'<{status} message="negative control" />'
        cases.append(
            f'<testcase classname="tests.test_terrain_vt_pbr_families.TestTerrainVTPbrFamilies" '
            f'name="{name}" time="0.1">{child}</testcase>'
        )
    return f'<?xml version="1.0"?><testsuites><testsuite>{"".join(cases)}</testsuite></testsuites>'


READBACK_BYTES = 256 * 256 * 4


def _render_memory(*samples: tuple[int, int]) -> dict:
    renders = [
        {
            "peak_host_visible_bytes": host,
            "peak_device_local_bytes": device,
            "total_tracked_bytes": host + device,
            "readback_staging_bytes": min(READBACK_BYTES, host),
        }
        for host, device in samples
    ]
    return {
        "source": reporter.RENDER_MEMORY_SOURCE,
        "renders": renders,
        "peak_host_visible_bytes": max(render["peak_host_visible_bytes"] for render in renders),
        "total_tracked_bytes": max(render["total_tracked_bytes"] for render in renders),
        "memory_limit_bytes": reporter.MEMORY_LIMIT_BYTES,
    }


def _make_fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict]:
    repo = tmp_path / "repo"
    artifacts = repo / "artifacts"
    golden_dir = repo / "tests" / "golden" / "terrain"
    golden_dir.mkdir(parents=True)
    artifacts.mkdir()

    yy, xx = np.indices((32, 32))
    checker = ((xx // 2 + yy // 2) % 2).astype(np.uint8) * 220 + 20
    baseline = np.stack([checker, checker, checker, np.full_like(checker, 255)], axis=-1)
    normal_value = 240 - checker
    normal = np.stack(
        [normal_value, np.roll(normal_value, 1, axis=0), normal_value, np.full_like(checker, 255)],
        axis=-1,
    ).astype(np.uint8)
    baseline_bytes = _png(baseline)
    normal_bytes = _png(normal)
    baseline_path = golden_dir / "substratia_grazing_baseline.nvidia-vulkan.png"
    normal_path = golden_dir / "substratia_grazing_normal.nvidia-vulkan.png"
    baseline_path.write_bytes(baseline_bytes)
    normal_path.write_bytes(normal_bytes)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "evidence@example.invalid")
    _git(repo, "config", "user.name", "Evidence Test")
    _git(repo, "add", "README.md", "tests/golden/terrain")
    _git(repo, "commit", "-qm", "fixture")
    candidate_sha = _git(repo, "rev-parse", "HEAD")

    for name, payload in (
        ("actual_baseline.png", baseline_bytes),
        ("actual_normal.png", normal_bytes),
        ("golden_baseline.png", baseline_bytes),
        ("golden_normal.png", normal_bytes),
    ):
        (artifacts / name).write_bytes(payload)

    baseline_lum = reporter._luminance(baseline)
    normal_lum = reporter._luminance(normal)
    top, bottom, left, right = reporter.GRAZING_REGION
    rows = slice(int(32 * top), int(32 * bottom))
    cols = slice(int(32 * left), int(32 * right))
    ssim = reporter._ssim(baseline_lum[rows, cols], normal_lum[rows, cols])
    results = {
        "schema": reporter.SCHEMA,
        "candidate_sha": candidate_sha,
        "gates": {
            "normal_lighting_ssim": {
                "status": "PASS",
                "ssim": ssim,
                "ssim_delta": 1.0 - ssim,
                "threshold": 0.05,
                "region": list(reporter.GRAZING_REGION),
                "actual_baseline": "actual_baseline.png",
                "actual_normal": "actual_normal.png",
                "golden_baseline": "golden_baseline.png",
                "golden_normal": "golden_normal.png",
                "golden_ssim_baseline": 1.0,
                "golden_ssim_normal": 1.0,
                "golden_mean_error_baseline": 0.0,
                "golden_mean_error_normal": 0.0,
                "actual_baseline_rgba_sha256": hashlib.sha256(
                    baseline.tobytes()
                ).hexdigest(),
                "actual_normal_rgba_sha256": hashlib.sha256(
                    normal.tobytes()
                ).hexdigest(),
                "render_memory": _render_memory((4 << 20, 96 << 20), (6 << 20, 128 << 20)),
            },
            "family_residency_budget": {
                "status": "PASS",
                "resident_bytes": {"albedo": 262144, "normal": 262144, "mask": 262144},
                "family_budget_bytes": {
                    "albedo": 33554432,
                    "normal": 33554432,
                    "mask": 33554432,
                },
                "total_resident_bytes": 786432,
                "configured_budget_bytes": 100663296,
                "memory_limit_bytes": reporter.MEMORY_LIMIT_BYTES,
                "render_memory": _render_memory(
                    (8 << 20, 160 << 20), (8 << 20, 160 << 20), (40 << 20, 200 << 20)
                ),
            },
            "missing_family_fatal": {
                "status": "PASS",
                "message": "terrain VT: family 'normal' requested but no source registered; refusing to render with corrupted PBR",
            },
            "partial_normal_residency": {
                "status": "PASS",
                "fallback_coverage": 0.25,
                "mean_luminance_error": 0.005,
                "error_threshold": 0.02,
                "render_memory": _render_memory(
                    (4 << 20, 64 << 20), (4 << 20, 72 << 20), (4 << 20, 80 << 20)
                ),
            },
        },
    }
    _write_json(artifacts / "results.json", results)
    probe = {
        "requested_backend": "vulkan",
        "probe": {
            "status": "ok",
            "backend": "vulkan",
            "device_type": "discretegpu",
            "name": "NVIDIA GeForce RTX 3070",
            "vendor": 0x10DE,
            "device": 9352,
            "software_fallback": False,
        },
    }
    _write_json(artifacts / "adapter-probe.json", probe)
    _write_json(artifacts / "render-process-adapter.json", probe["probe"])
    (artifacts / "pytest-junit.xml").write_text(_junit(), encoding="utf-8")
    args = argparse.Namespace(
        artifact_dir=artifacts,
        repository=repo,
        candidate_sha=candidate_sha,
        adapter_probe=artifacts / "adapter-probe.json",
        render_adapter=artifacts / "render-process-adapter.json",
        junit=artifacts / "pytest-junit.xml",
        expected_backend="vulkan",
    )
    return args, results


def _rewrite_results(args: argparse.Namespace, results: dict) -> None:
    _write_json(args.artifact_dir / "results.json", results)


def test_valid_physical_evidence_writes_bound_pass_and_lane_marker(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    report = reporter.verify(args)
    assert report["status"] == "PASS"
    assert report["candidate_sha"] == args.candidate_sha
    assert report["adapter"]["device_type"] == "discretegpu"
    marker = json.loads((args.artifact_dir / "lane-ran.json").read_text(encoding="utf-8"))
    assert marker["status"] == "RAN"
    assert marker["verifier_status"] == "PASS"
    assert marker["candidate_sha"] == args.candidate_sha
    metrics = report["metrics"]
    assert metrics["render_peak_host_visible_bytes"] == 40 << 20
    assert metrics["render_total_tracked_bytes"] == 240 << 20
    assert metrics["render_peak_host_visible_bytes_by_gate"] == {
        "normal_lighting_ssim": 6 << 20,
        "family_residency_budget": 40 << 20,
        "partial_normal_residency": 4 << 20,
    }
    assert metrics["render_memory_sample_count"] == 8


def test_render_memory_at_exact_limit_passes(tmp_path: Path) -> None:
    args, results = _make_fixture(tmp_path)
    results["gates"]["partial_normal_residency"]["render_memory"] = _render_memory(
        (reporter.MEMORY_LIMIT_BYTES, 0), (1 << 20, 0), (1 << 20, 0)
    )
    _rewrite_results(args, results)
    assert reporter.verify(args)["metrics"]["render_peak_host_visible_bytes"] == (
        reporter.MEMORY_LIMIT_BYTES
    )


def _mutate_render_memory(memory: dict, mutation: str) -> None:
    render = memory["renders"][0]
    if mutation == "missing_peak":
        memory.pop("peak_host_visible_bytes")
    elif mutation == "missing_total":
        memory.pop("total_tracked_bytes")
    elif mutation == "missing_render_peak":
        render.pop("peak_host_visible_bytes")
    elif mutation == "missing_render_total":
        render.pop("total_tracked_bytes")
    elif mutation == "float_peak":
        memory["peak_host_visible_bytes"] = float(memory["peak_host_visible_bytes"])
    elif mutation == "string_total":
        memory["total_tracked_bytes"] = str(memory["total_tracked_bytes"])
    elif mutation == "bool_render_device":
        render["peak_device_local_bytes"] = True
    elif mutation == "negative_render_host":
        render["peak_host_visible_bytes"] = -1
    elif mutation == "over_limit":
        host = reporter.MEMORY_LIMIT_BYTES + 1
        render["peak_host_visible_bytes"] = host
        render["total_tracked_bytes"] = host + render["peak_device_local_bytes"]
        memory["peak_host_visible_bytes"] = host
        memory["total_tracked_bytes"] = max(
            sample["total_tracked_bytes"] for sample in memory["renders"]
        )
    elif mutation == "render_total_mismatch":
        render["total_tracked_bytes"] += 1
    elif mutation == "aggregate_peak_understated":
        memory["peak_host_visible_bytes"] -= 1
    elif mutation == "aggregate_total_mismatch":
        memory["total_tracked_bytes"] += 1
    elif mutation == "empty_renders":
        memory["renders"] = []
    elif mutation == "zero_tracked":
        render.update(
            peak_host_visible_bytes=0, peak_device_local_bytes=0, total_tracked_bytes=0
        )
    elif mutation == "limit_drift":
        memory["memory_limit_bytes"] = reporter.MEMORY_LIMIT_BYTES * 2
    elif mutation == "source_drift":
        memory["source"] = "forge3d.memory_metrics"
    elif mutation == "too_few_renders":
        memory["renders"] = memory["renders"][:1]
        memory["peak_host_visible_bytes"] = memory["renders"][0]["peak_host_visible_bytes"]
        memory["total_tracked_bytes"] = memory["renders"][0]["total_tracked_bytes"]
    elif mutation == "missing_readback":
        render.pop("readback_staging_bytes")
    elif mutation == "zero_readback":
        render["readback_staging_bytes"] = 0
    elif mutation == "readback_over_host_peak":
        render["readback_staging_bytes"] = render["peak_host_visible_bytes"] + 1
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize("gate", tuple(reporter.RENDER_MEMORY_MIN_RENDERS))
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_block", "no render_memory evidence"),
        ("missing_peak", "peak_host_visible_bytes is not a non-negative integer"),
        ("missing_total", "total_tracked_bytes is not a non-negative integer"),
        ("missing_render_peak", "render 0 peak_host_visible_bytes is not"),
        ("missing_render_total", "render 0 total_tracked_bytes is not"),
        ("float_peak", "peak_host_visible_bytes is not a non-negative integer"),
        ("string_total", "total_tracked_bytes is not a non-negative integer"),
        ("bool_render_device", "render 0 peak_device_local_bytes is not"),
        ("negative_render_host", "render 0 peak_host_visible_bytes is not"),
        ("over_limit", "exceeds the 512 MiB contract"),
        ("render_total_mismatch", "render 0 total_tracked_bytes is inconsistent"),
        ("aggregate_peak_understated", "host-visible peak is inconsistent"),
        ("aggregate_total_mismatch", "total tracked bytes are inconsistent"),
        ("empty_renders", "no per-render memory samples"),
        ("zero_tracked", "tracked no allocations"),
        ("limit_drift", "render memory ceiling drift"),
        ("source_drift", "render_memory source drift"),
        ("too_few_renders", "expected at least"),
        ("missing_readback", "render 0 readback_staging_bytes is not"),
        ("zero_readback", "readback staging is outside"),
        ("readback_over_host_peak", "readback staging is outside"),
    ],
)
def test_render_memory_evidence_has_hard_negative_controls(
    tmp_path: Path, gate: str, mutation: str, message: str
) -> None:
    args, results = _make_fixture(tmp_path)
    if mutation == "missing_block":
        results["gates"][gate].pop("render_memory")
    else:
        _mutate_render_memory(results["gates"][gate]["render_memory"], mutation)
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError, match=message):
        reporter.verify(args)
    assert not (args.artifact_dir / "lane-ran.json").exists()


@pytest.mark.parametrize("candidate", ["HEAD", "abc123", "A" * 40, "0" * 40])
def test_candidate_sha_must_be_explicit_exact_clean_head(tmp_path: Path, candidate: str) -> None:
    args, _ = _make_fixture(tmp_path)
    args.candidate_sha = candidate
    with pytest.raises(reporter.EvidenceError):
        reporter.verify(args)


def test_tracked_dirty_tree_is_rejected(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    (args.repository / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(reporter.EvidenceError, match="working-tree changes"):
        reporter.verify(args)


@pytest.mark.parametrize(
    ("device_type", "name"),
    [
        ("virtualgpu", "Virtual GPU"),
        ("integratedgpu", "llvmpipe"),
        ("discretegpu", "Paravirtual Vulkan Device"),
        ("cpu", "Apple CPU"),
    ],
)
def test_software_virtual_and_paravirtual_adapters_are_rejected(
    tmp_path: Path, device_type: str, name: str
) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    probe["probe"]["device_type"] = device_type
    probe["probe"]["name"] = name
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="adapter|physical|forbidden"):
        reporter.verify(args)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_wrong_backend_is_rejected(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    probe["requested_backend"] = "metal"
    probe["probe"]["backend"] = "metal"
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="wrong backend"):
        reporter.verify(args)


@pytest.mark.parametrize("fallback", [True, None])
def test_nvidia_probe_must_explicitly_reject_software_fallback(
    tmp_path: Path, fallback: bool | None
) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    if fallback is None:
        probe["probe"].pop("software_fallback")
    else:
        probe["probe"]["software_fallback"] = fallback
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="software_fallback=false"):
        reporter.verify(args)


def test_nvidia_probe_requires_discrete_gpu(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    probe["probe"]["device_type"] = "integratedgpu"
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="discrete GPU"):
        reporter.verify(args)


def test_nvidia_name_cannot_spoof_non_nvidia_vendor(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    probe["probe"]["vendor"] = 0x1002
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="vendor ID"):
        reporter.verify(args)


def test_nvidia_vendor_cannot_spoof_non_nvidia_name(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    probe = _read(args.adapter_probe)
    probe["probe"]["name"] = "AMD Radeon RX 7900 XT"
    _write_json(args.adapter_probe, probe)
    with pytest.raises(reporter.EvidenceError, match="adapter name"):
        reporter.verify(args)


def test_render_process_adapter_must_match_workflow_probe(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    render_probe = _read(args.render_adapter)
    render_probe["device"] += 1
    _write_json(args.render_adapter, render_probe)
    with pytest.raises(reporter.EvidenceError, match="render-process adapter device"):
        reporter.verify(args)


def test_corrupt_png_crc_is_rejected(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    path = args.artifact_dir / "actual_normal.png"
    data = bytearray(path.read_bytes())
    data[-8] ^= 0x01
    path.write_bytes(data)
    with pytest.raises(reporter.EvidenceError, match="CRC mismatch"):
        reporter.verify(args)


def test_artifact_golden_must_be_candidate_tracked_bytes(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    (args.artifact_dir / "golden_normal.png").write_bytes(
        (args.artifact_dir / "golden_baseline.png").read_bytes()
    )
    with pytest.raises(reporter.EvidenceError, match="candidate-tracked"):
        reporter.verify(args)


def test_noncanonical_golden_filename_is_rejected(tmp_path: Path) -> None:
    args, results = _make_fixture(tmp_path)
    results["gates"]["normal_lighting_ssim"]["golden_normal"] = "golden_baseline.png"
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError, match="non-canonical"):
        reporter.verify(args)


def test_declared_image_metrics_are_recomputed(tmp_path: Path) -> None:
    args, results = _make_fixture(tmp_path)
    results["gates"]["normal_lighting_ssim"]["ssim_delta"] = 0.99
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError, match="does not match recomputation"):
        reporter.verify(args)


def test_declared_rgba_hash_is_recomputed_from_decoded_pixels(tmp_path: Path) -> None:
    args, results = _make_fixture(tmp_path)
    results["gates"]["normal_lighting_ssim"][
        "actual_normal_rgba_sha256"
    ] = "0" * 64
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError, match="decoded RGBA"):
        reporter.verify(args)


def test_actual_image_cannot_be_replaced_while_keeping_claimed_hash(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    (args.artifact_dir / "actual_normal.png").write_bytes(
        (args.artifact_dir / "actual_baseline.png").read_bytes()
    )
    with pytest.raises(reporter.EvidenceError, match="decoded RGBA"):
        reporter.verify(args)


@pytest.mark.parametrize("junit_status", ["failure", "error", "skipped"])
def test_junit_requires_each_core_test_to_pass(tmp_path: Path, junit_status: str) -> None:
    args, _ = _make_fixture(tmp_path)
    args.junit.write_text(_junit(junit_status), encoding="utf-8")
    with pytest.raises(reporter.EvidenceError, match="did not pass"):
        reporter.verify(args)


def test_junit_requires_all_four_exact_core_tests(tmp_path: Path) -> None:
    args, _ = _make_fixture(tmp_path)
    args.junit.write_text(_junit(omitted=reporter.CORE_TESTS[-1]), encoding="utf-8")
    with pytest.raises(reporter.EvidenceError, match="exactly one"):
        reporter.verify(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("zero_family", "no resident bytes"),
        ("family_over_budget", "exceeds its family budget"),
        ("total_over_budget", "512 MiB"),
        ("missing_not_fatal", "fatal diagnostic"),
        ("partial_corrupt", "fallback is corrupted"),
    ],
)
def test_non_image_claims_have_hard_negative_controls(
    tmp_path: Path, mutation: str, message: str
) -> None:
    args, original = _make_fixture(tmp_path)
    results = copy.deepcopy(original)
    gates = results["gates"]
    if mutation == "zero_family":
        gates["family_residency_budget"]["resident_bytes"]["mask"] = 0
        gates["family_residency_budget"]["total_resident_bytes"] -= 262144
    elif mutation == "family_over_budget":
        gates["family_residency_budget"]["resident_bytes"]["normal"] = 40000000
        gates["family_residency_budget"]["total_resident_bytes"] = 40524288
    elif mutation == "total_over_budget":
        gate = gates["family_residency_budget"]
        gate["configured_budget_bytes"] = reporter.MEMORY_LIMIT_BYTES + 1
    elif mutation == "missing_not_fatal":
        gates["missing_family_fatal"]["message"] = "normal was missing; used fallback"
    elif mutation == "partial_corrupt":
        gates["partial_normal_residency"]["mean_luminance_error"] = 0.2
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError, match=message):
        reporter.verify(args)


def test_failed_verification_never_writes_ran_or_pass_markers(tmp_path: Path) -> None:
    args, results = _make_fixture(tmp_path)
    results["candidate_sha"] = "0" * 40
    _rewrite_results(args, results)
    with pytest.raises(reporter.EvidenceError):
        reporter.verify(args)
    assert not (args.artifact_dir / "verification.json").exists()
    assert not (args.artifact_dir / "lane-ran.json").exists()
