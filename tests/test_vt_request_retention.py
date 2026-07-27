from __future__ import annotations

from pathlib import Path

import forge3d as f3d


def test_thirty_not_ready_frames_preserve_requests_and_converge():
    report = f3d.vt_request_retention_probe(30)
    assert report == {
        "not_ready_frames": 30,
        "preserved": True,
        "converged": True,
        "remaining_requests": 0,
    }


def test_no_shipped_synthetic_reader_callers():
    root = Path(__file__).resolve().parents[1] / "src"
    callers = []
    for path in root.rglob("*.rs"):
        if path.name == "readers.rs":
            continue
        text = path.read_text(encoding="utf-8")
        for symbol in ("SyntheticHeightReader", "SyntheticOverlayReader"):
            for line_number, line in enumerate(text.splitlines(), 1):
                if symbol in line and "pub use" not in line:
                    callers.append(f"{path.relative_to(root)}:{line_number}:{line.strip()}")
    assert callers == []


def test_gpu_lod_and_visibility_shaders_have_live_callsites():
    root = Path(__file__).resolve().parents[1]
    geometry = (root / "src/terrain/renderer/geometry.rs").read_text()
    execute = (root / "src/terrain/renderer/draw/execute.rs").read_text()
    assert "GpuLodSelector" in geometry
    assert "encode_indirect" in geometry
    assert "terrain_visbuffer_write.shader" in execute
    assert "stage_visibility_stats" in execute


def test_tessella_acceptance_is_a_required_zero_skip_hardware_lane():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  test-tessella-gpu:", 1)[1].split(
        "\n  # ============================================================================",
        1,
    )[0]
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
    assert "terrain_ci_probe.py --mode terrain --require-nvidia-vulkan" in job
    assert "scripts/assert_junit_zero_skips.py" in job
    for test_file in (
        "test_vt_out_of_core.py",
        "test_hzb_culling.py",
        "test_visibility_buffer.py",
        "test_bc_encoders.py",
        "test_flythrough_popping.py",
        "test_vt_request_retention.py",
    ):
        assert test_file in job
    aggregator = workflow.split("\n  ci-success:", 1)[1]
    assert "test-tessella-gpu" in aggregator
