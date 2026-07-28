from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.diagnostics import render_certificate, visibility_stats
from forge3d.terrain_params import (
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)
from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem


requires_terrain = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)


def test_visibility_parameter_contract():
    required = {
        "size_px": (64, 64),
        "render_scale": 1.0,
        "terrain_span": 10.0,
        "msaa_samples": 1,
        "z_scale": 1.0,
        "exposure": 1.0,
        "domain": (0.0, 1.0),
    }
    forward = f3d.TerrainRenderParams(make_terrain_params_config(**required))
    visibility = f3d.TerrainRenderParams(
        make_terrain_params_config(**required, shading="visibility")
    )
    assert visibility.shading == "visibility"
    assert forward.culling in {"none", "frustum", "hzb_two_phase"}


def test_feedback_counter_tracks_the_physical_surface_write():
    root = Path(__file__).resolve().parents[1]
    shader = (root / "src/shaders/terrain_pbr_pom.wgsl").read_text(
        encoding="utf-8"
    )
    fullscreen = (
        root / "src/shaders/terrain_visibility_fullscreen.wgsl"
    ).read_text(encoding="utf-8")
    assert shader.count(
        "atomicAdd(&terrain_frame_counters.feedback_records, 1u)"
    ) == 1
    assert "terrain_vt_write_surface_feedback(input.tex_coord, 0u)" in shader
    assert "terrain_vt_write_surface_feedback(surface.tex_coord, 0u)" in fullscreen


@pytest.mark.gpu_lane
@requires_terrain
def test_visibility_resolve_pays_once_and_picking_is_stable_for_10000_pixels():
    size = (640, 360)
    virtual_size = 128
    vt = TerrainVTSettings(
        enabled=True,
        layers=[
            VTLayerFamily(
                family=family,
                virtual_size_px=(virtual_size, virtual_size),
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
        "albedo": np.full((virtual_size, virtual_size, 4), [112, 132, 96, 255], dtype=np.uint8),
        "normal": np.full((virtual_size, virtual_size, 4), [128, 128, 255, 255], dtype=np.uint8),
        "mask": np.full((virtual_size, virtual_size, 4), [255, 128, 255, 255], dtype=np.uint8),
    }

    def register_sources(renderer):
        for material_index in range(4):
            for family, source in sources.items():
                renderer.register_material_vt_source(
                    material_index,
                    family,
                    source,
                    (virtual_size, virtual_size),
                    [0.5, 0.5, 1.0, 1.0],
                )

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        dem = _steep_dem(96)
        forward_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        register_sources(forward_renderer)
        forward_params = _make_params(size_px=size, vt=vt)
        forward = _render_rgba(forward_renderer, forward_params, dem, ibl)

        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        register_sources(renderer)
        visibility = _render_rgba(
            renderer,
            _make_params(size_px=size, shading="visibility", vt=vt),
            dem,
            ibl,
        )

    np.testing.assert_array_equal(visibility, forward)
    stats = visibility_stats()
    assert stats["visible_pixels"] + stats["background_pixels"] == size[0] * size[1]
    assert stats["visibility_feedback_records"] == stats["visible_pixels"]
    assert stats["material_invocations"] == stats["visible_pixels"]
    assert stats["forward_material_invocations"] >= stats["visible_pixels"]
    assert (
        stats["forward_feedback_records"] == stats["forward_material_invocations"]
    )
    assert (
        stats["forward_feedback_records"] >= stats["visibility_feedback_records"]
    )
    overdraw_factor = (
        stats["forward_feedback_records"] / stats["visibility_feedback_records"]
    )
    assert stats["fallback_texels"] == 0
    shader_hashes = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert "terrain_visbuffer_write.shader" in shader_hashes
    assert "terrain_visbuffer_resolve.shader" in shader_hashes
    assert "terrain_visbuffer_resolve" in shader_hashes

    rng = np.random.default_rng(19)
    pixels = list(
        zip(
            rng.integers(0, size[0], size=10_000).tolist(),
            rng.integers(0, size[1], size=10_000).tolist(),
        )
    )
    first = renderer.pick_visibility_pixels(pixels)
    cpu = renderer.pick_visibility_pixels_cpu(pixels)
    assert first == cpu
    assert len(first) == 10_000
    assert sum(value is not None for value in first) > 0
    record_tessella_result(
        "visibility_buffer",
        {
            "visible_pixels": int(stats["visible_pixels"]),
            "background_pixels": int(stats["background_pixels"]),
            "visibility_feedback_records": int(
                stats["visibility_feedback_records"]
            ),
            "forward_feedback_records": int(stats["forward_feedback_records"]),
            "material_invocations": int(stats["material_invocations"]),
            "forward_material_invocations": int(
                stats["forward_material_invocations"]
            ),
            "measured_overdraw_factor": float(overdraw_factor),
            "fallback_texels": int(stats["fallback_texels"]),
            "picking_samples": len(first),
            "picking_hits": sum(value is not None for value in first),
            "gpu_cpu_picking_matches": sum(
                gpu_value == cpu_value
                for gpu_value, cpu_value in zip(first, cpu, strict=True)
            ),
            "bitwise_identical_to_forward": True,
        },
    )
