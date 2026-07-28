from __future__ import annotations

import tempfile
from pathlib import Path

import forge3d as f3d
import numpy as np
import pytest

from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily
from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_thirty_not_ready_frames_preserve_requests_and_converge():
    side = 512
    vt = TerrainVTSettings(
        enabled=True,
        layers=[
            VTLayerFamily(
                family=family,
                virtual_size_px=(side, side),
                tile_size=120,
                tile_border=4,
            )
            for family in ("albedo", "normal", "mask")
        ],
        atlas_size=1024,
        residency_budget_mb=32.0,
        max_mip_levels=4,
        use_feedback=True,
    )
    sources = {
        "albedo": np.full((side, side, 4), [112, 132, 96, 255], dtype=np.uint8),
        "normal": np.full((side, side, 4), [128, 128, 255, 255], dtype=np.uint8),
        "mask": np.full((side, side, 4), [255, 128, 255, 255], dtype=np.uint8),
    }
    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        for material_index in range(4):
            for family, source in sources.items():
                renderer.register_material_vt_source(
                    material_index,
                    family,
                    source,
                    (side, side),
                    [0.5, 0.5, 1.0, 1.0],
                )
        params = _make_params(size_px=(64, 64), vt=vt)
        dem = _steep_dem(64)
        _render_rgba(renderer, params, dem, ibl)
        renderer.force_vt_feedback_not_ready_for_test(30)
        for _ in range(30):
            _render_rgba(renderer, params, dem, ibl)
            assert renderer.get_material_vt_stats()["retained_requests"] >= 1
        convergence_frame = 0
        for convergence_frame in range(1, 9):
            _render_rgba(renderer, params, dem, ibl)
            if renderer.get_material_vt_stats()["retained_requests"] == 0:
                break
        final_stats = renderer.get_material_vt_stats()
        assert final_stats["retained_requests"] == 0
        record_tessella_result(
            "vt_request_retention",
            {
                "feedback_not_ready_frames": 30,
                "convergence_budget_frames": 8,
                "convergence_frames": convergence_frame,
                "retained_requests_after_convergence": int(
                    final_stats["retained_requests"]
                ),
                "tiles_streamed": int(final_stats["tiles_streamed"]),
            },
        )


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


def test_retention_fault_injection_delays_the_real_feedback_map():
    root = Path(__file__).resolve().parents[1] / "src"
    feedback = (root / "core/feedback_buffer.rs").read_text(encoding="utf-8")
    runtime = (root / "terrain/renderer/virtual_texture.rs").read_text(
        encoding="utf-8"
    )
    assert "forced_not_ready_polls" in feedback
    assert "try_read_feedback_entries" in feedback
    assert "force_not_ready_polls_for_test" in runtime
    assert "feedback_not_ready_frames" not in runtime


def test_gpu_lod_and_visibility_shaders_have_live_callsites():
    root = Path(__file__).resolve().parents[1]
    geometry = (root / "src/terrain/renderer/geometry.rs").read_text()
    execute = (root / "src/terrain/renderer/draw/execute.rs").read_text()
    assert "GpuLodSelector" in geometry
    assert "encode_indirect" in geometry
    assert "terrain_visbuffer_write.shader" in execute
    assert "stage_visibility_stats" in execute
    assert "encode_visibility_resolve_pass" in execute
    assert "pass.draw(0..3, 0..1)" in execute
    pipeline = (root / "src/terrain/renderer/pipeline_cache.rs").read_text()
    assert 'entry_point: "vs_visibility_fullscreen"' in pipeline
    shader = (
        root / "src/shaders/terrain_visibility_fullscreen.wgsl"
    ).read_text()
    assert "visibility_barycentrics" in shader
    assert "terrain_visibility_indices" in shader


def test_height_is_the_fourth_feedback_driven_vt_family():
    root = Path(__file__).resolve().parents[1]
    residency = (root / "src/terrain/vt_family_residency.rs").read_text()
    streaming = (root / "src/terrain/renderer/streaming.rs").read_text()
    py_api = (root / "src/terrain/renderer/py_api.rs").read_text()
    assert "VT_FAMILY_COUNT: usize = 4" in residency
    assert "HeightVtFamilyRuntime" in streaming
    assert "RetainedRequestSet" in streaming
    assert "FamilyResidencyTracker" in streaming
    assert "ReaderPageStore" in streaming
    assert "VirtualTextureStore for ReaderPageStore" in streaming
    assert "latest_feedback_uvs" in py_api


def test_public_vt_stats_diagnostics_surface_is_registered():
    from forge3d import diagnostics

    assert callable(diagnostics.vt_stats)


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
