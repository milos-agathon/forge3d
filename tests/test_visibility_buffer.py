from __future__ import annotations

import tempfile
import json
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.diagnostics import (
    capabilities,
    render_certificate,
    visibility_stats,
    vt_stats,
)
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


def _exact_cpu_raster_ties(frame, pixels, gpu, cpu):
    """D6 exclusions: exact edge zeros on the oracle's existing 1/256 grid.

    GPU answers identify the triangle involved in a comparison; they never
    supply geometry, coverage, depth or a tolerance to the CPU oracle.
    """
    snapped = np.rint(
        np.asarray(frame["vertices"], dtype=np.float32)[:, :2] * np.float32(256)
    ).astype(np.int64)
    points = np.asarray(pixels, dtype=np.int64) * 256 + 128
    ties = [[] for _ in pixels]
    for index, source in enumerate(frame["indices"]):
        triangle = snapped[source].copy()
        a, b, c = triangle
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0:
            continue
        if area < 0:
            triangle[[1, 2]] = triangle[[2, 1]]
        candidates = np.flatnonzero(
            np.all((points >= triangle.min(axis=0)) & (points <= triangle.max(axis=0)), axis=1)
        )
        delta = np.roll(triangle, -1, axis=0) - triangle
        relative = points[candidates, None, :] - triangle
        edges = delta[:, 0] * relative[:, :, 1] - delta[:, 1] * relative[:, :, 0]
        on_edge = np.any(edges == 0, axis=1) & np.all(edges >= 0, axis=1)
        identity = tuple(frame["identities"][index])
        for sample, values in zip(candidates[on_edge], edges[on_edge], strict=True):
            involved = {value for value in (gpu[sample], cpu[sample]) if value is not None}
            if not involved or identity in involved:
                ties[sample].append({
                    "identity": identity,
                    "raster_triangle_index": index,
                    "edge_functions": [int(value) for value in values],
                })
    return ties


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
    shader_compact = "".join(shader.split())
    fullscreen_compact = "".join(fullscreen.split())
    assert (
        "terrain_vt_write_surface_feedback("
        "input.tex_coord,input.world_position,feedback_ddx_uv,feedback_ddy_uv,"
        "feedback_ddx_world,feedback_ddy_world,0u,input.clip_position.xy,)"
        in shader_compact
    )
    assert (
        "terrain_vt_write_surface_feedback("
        "input.tex_coord,input.world_position,resolve_ddx_uv,resolve_ddy_uv,"
        "resolve_ddx_world,resolve_ddy_world,0u,input.clip_position.xy,)"
        in fullscreen_compact
    )
    assert (
        "terrain_vt_write_surface_feedback("
        "surface.tex_coord,surface.world_position,feedback_ddx_uv,feedback_ddy_uv,"
        "feedback_ddx_world,feedback_ddy_world,0u,input.clip_position.xy,)"
        in fullscreen_compact
    )


def test_visibility_write_and_resolve_are_separate_shader_sources():
    """Pass 1 is its own file, not an alias of the forward module.

    Before the split both visibility pipelines compiled one string produced by
    two `String::replace` calls over terrain_pbr_pom.wgsl, so the literal
    `fs_visibility` in that file was dead code carrying a packing the runtime
    never used.
    """
    root = Path(__file__).resolve().parents[1]
    write = (root / "src/shaders/terrain_visbuffer_write.wgsl").read_text(
        encoding="utf-8"
    )
    assert "fn fs_visibility(" in write
    assert "TERRAIN_VISBUFFER_TILE_SHIFT" in write

    pbr = (root / "src/shaders/terrain_pbr_pom.wgsl").read_text(encoding="utf-8")
    assert "fn fs_visibility(" not in pbr  # pass 1 moved to its own file
    assert "fn fs_visibility_resolve(" not in pbr  # dead depth-equal variant
    assert "0x00ffffffu" not in pbr  # stale 24|8 packing comment and code
    # The packed tile/LOD identity is written unconditionally now, so no
    # assembly-time rewrite decides what pass 1 reads.
    assert "out.tile_id = ((_tile_id_lod.y & 0xfu) << 12u)" in pbr

    sources = (root / "src/shader_sources.rs").read_text(encoding="utf-8")
    assert "fn terrain_visbuffer_write(" in sources
    assert "fn terrain_visbuffer_resolve(" in sources
    assert "fn terrain_visibility(" not in sources
    assert "0x00ffffffu" not in sources
    assert "out.tile_id = _tile_id_lod.x" not in sources

    pipeline = (root / "src/terrain/renderer/pipeline_cache.rs").read_text(
        encoding="utf-8"
    )
    assert "preprocess_visibility_write_shader" in pipeline
    assert "preprocess_visibility_resolve_shader" in pipeline
    # The atlas variant is selected, and the compatibility path is recorded.
    assert "shader_sources::terrain_bindless()" in pipeline
    assert "terrain_vt_bindless_atlas" in pipeline


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
        # A 1 m camera radius against a 100 km terrain puts the eye inside the
        # near-origin relief and makes the visibility differential vacuous on
        # some adapters. Span several rings at a real view distance and keep
        # the indirect frustum path active so the 10,000-pixel differential
        # covers tile/LOD packing as well as primitive identity.
        forward_params = _make_params(
            size_px=size,
            theta_deg=10.0,
            cam_radius=50_000.0,
            culling="frustum",
            vt=vt,
        )
        forward = _render_rgba(forward_renderer, forward_params, dem, ibl)

        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        register_sources(renderer)
        visibility = _render_rgba(
            renderer,
            _make_params(
                size_px=size,
                theta_deg=10.0,
                cam_radius=50_000.0,
                culling="frustum",
                shading="visibility",
                vt=vt,
            ),
            dem,
            ibl,
        )

    np.testing.assert_array_equal(visibility, forward)
    stats = visibility_stats()
    assert stats["visible_pixels"] + stats["background_pixels"] == size[0] * size[1]
    assert stats["visible_pixels"] > 0
    assert stats["visibility_feedback_records"] == stats["visible_pixels"]
    assert stats["material_invocations"] == stats["visible_pixels"]
    assert stats["material_invocations"] > 0
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
    # Two labels must not alias one source: pass 1 carries no material stage
    # and pass 2 carries no fs_visibility entry, so their hashes differ.
    assert (
        shader_hashes["terrain_visbuffer_write.shader"]
        != shader_hashes["terrain_visbuffer_resolve.shader"]
    )

    rng = np.random.default_rng(19)
    pixels = list(
        zip(
            rng.integers(0, size[0], size=10_000).tolist(),
            rng.integers(0, size[1], size=10_000).tolist(),
        )
    )
    first = renderer.pick_visibility_pixels(pixels)
    second = renderer.pick_visibility_pixels(pixels)
    assert second == first

    # Preserve the existing primitive-interior criterion, but actually compare
    # 10,000 distinct CPU/GPU picks instead of discarding most random samples.
    # Eligibility uses only raster identity; CPU answers do not select samples.
    all_pixels = [(x, y) for y in range(size[1]) for x in range(size[0])]
    all_gpu = renderer.pick_visibility_pixels(all_pixels)
    identities = np.asarray(
        [0 if value is None else 1 + (value[0] << 32) + value[1] for value in all_gpu],
        dtype=np.uint64,
    ).reshape(size[1], size[0])
    centers = identities[1:-1, 1:-1]
    stable = centers != 0
    for dy in range(3):
        for dx in range(3):
            stable &= centers == identities[dy : dy + size[1] - 2, dx : dx + size[0] - 2]
    # Keep the gated keys on their original seed-19 random-pool accounting.
    compared_indices = [
        index for index, (x, y) in enumerate(pixels)
        if 0 < x < size[0] - 1 and 0 < y < size[1] - 1 and stable[y - 1, x - 1]
    ]
    assert compared_indices, "no unambiguous 3x3 picking neighborhoods"
    random_cpu = renderer.pick_visibility_pixels_cpu(pixels)
    frame = json.loads(renderer._visibility_cpu_frame())
    lod_comparison = frame["lod_comparison"]
    assert lod_comparison["submitted_equals_cpu"], lod_comparison
    assert lod_comparison["submitted_tiles"] == lod_comparison["cpu_tiles"]
    ties = _exact_cpu_raster_ties(frame, pixels, first, random_cpu)
    non_tie_disagreements = [
        {"sample_index": index, "pixel": pixel, "gpu": gpu, "cpu": cpu, "tie": False}
        for index, (pixel, gpu, cpu, tie) in enumerate(zip(pixels, first, random_cpu, ties, strict=True))
        if not tie and gpu != cpu
    ]
    record_tessella_result("visibility_unfiltered_picking", {
        "seed": 19,
        "samples": len(pixels),
        "distinct_pixels": len(set(pixels)),
        "gpu_cpu_matches": sum(a == b for a, b in zip(first, random_cpu, strict=True)),
        "tie_exclusions": sum(bool(value) for value in ties),
        "non_tie_compared": sum(not value for value in ties),
        "non_tie_matches": sum(not tie and a == b for tie, a, b in zip(ties, first, random_cpu, strict=True)),
        "non_tie_disagreements": len(non_tie_disagreements),
        "disagreements": non_tie_disagreements,
        "tie_pixels": [
            {"sample_index": index, "pixel": pixels[index], "triangles": value}
            for index, value in enumerate(ties) if value
        ],
        "tie_classification": "exact zero edge function on the existing 1/256 snap grid",
        "lod_comparison": lod_comparison,
    })
    assert not non_tie_disagreements, non_tie_disagreements
    compared_gpu = [first[index] for index in compared_indices]
    compared_cpu = [random_cpu[index] for index in compared_indices]
    assert compared_gpu == compared_cpu

    eligible_y, eligible_x = np.nonzero(stable)
    assert len(eligible_x) >= 10_000, "fewer than 10,000 unambiguous visible pixels"
    selected = rng.choice(len(eligible_x), size=10_000, replace=False)
    eligible_pixels = [
        (int(eligible_x[index]) + 1, int(eligible_y[index]) + 1) for index in selected
    ]
    assert len(set(eligible_pixels)) == 10_000
    eligible_gpu = [all_gpu[y * size[0] + x] for x, y in eligible_pixels]
    eligible_repeat = renderer.pick_visibility_pixels(eligible_pixels)
    assert eligible_repeat == eligible_gpu
    eligible_cpu = renderer.pick_visibility_pixels_cpu(eligible_pixels)
    eligible_compared = sum(bool(stable[y - 1, x - 1]) for x, y in eligible_pixels)
    if eligible_gpu != eligible_cpu:
        mismatch = next(
            index
            for index, (gpu_value, cpu_value) in enumerate(
                zip(eligible_gpu, eligible_cpu, strict=True)
            )
            if gpu_value != cpu_value
        )
        mismatch_count = sum(
            gpu_value != cpu_value
            for gpu_value, cpu_value in zip(
                eligible_gpu, eligible_cpu, strict=True
            )
        )
        pytest.fail(
            repr({
                "mismatch_count": mismatch_count,
                "compared_count": eligible_compared,
                "excluded_count": len(eligible_pixels) - eligible_compared,
                "first_index": mismatch,
                "pixel": eligible_pixels[mismatch],
                "gpu": eligible_gpu[mismatch],
                "cpu": eligible_cpu[mismatch],
            })
        )
    assert len(first) == 10_000
    visible_identities = [value for value in first if value is not None]
    assert len(visible_identities) >= 1_000, len(visible_identities)
    tile_lod_ids = {value[0] for value in visible_identities}
    assert len(tile_lod_ids) > 1, tile_lod_ids
    assert any(tile_lod_id != 0 for tile_lod_id in tile_lod_ids), tile_lod_ids
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
            "gpu_picking_repeat_matches": sum(
                first_value == second_value
                for first_value, second_value in zip(first, second, strict=True)
            ),
            "gpu_cpu_picking_compared": len(compared_indices),
            "gpu_cpu_picking_excluded": len(first) - len(compared_indices),
            "picking_sample_selection": "original random pool; visible centers with one 3x3 raster identity",
            "eligible_picking_samples": len(eligible_pixels),
            "eligible_gpu_cpu_picking_compared": eligible_compared,
            "eligible_gpu_cpu_picking_excluded": len(eligible_pixels) - eligible_compared,
            "eligible_gpu_cpu_picking_matches": sum(
                a == b for a, b in zip(eligible_gpu, eligible_cpu, strict=True)
            ),
            "eligible_gpu_picking_repeat_matches": sum(
                a == b for a, b in zip(eligible_gpu, eligible_repeat, strict=True)
            ),
            "lod_comparison": lod_comparison,
            "gpu_cpu_picking_matches": sum(
                gpu_value == cpu_value
                for gpu_value, cpu_value in zip(
                    compared_gpu, compared_cpu, strict=True
                )
            ),
            "bitwise_identical_to_forward": True,
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_bindless_atlas_path_is_selected_or_recorded_as_a_degradation():
    """The descriptor-indexing atlas path is asserted, never assumed.

    Both arms are positive assertions, so neither adapter class passes
    vacuously: an adapter that grants the three bindless features must compile
    and keep the `binding_array` assembly with no fallback degradation, and an
    adapter without them must name the fallback in the certificate. The
    capability set is read before the render because it describes the device
    the pipeline cache compiles against.
    """
    granted = set(capabilities()["granted"])
    bindless_capable = {
        "texture_binding_array",
        "sampled_texture_and_storage_buffer_array_non_uniform_indexing",
        "texture_compression_bc",
    } <= granted

    size = (320, 180)
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

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        dem = _steep_dem(96)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        for material_index in range(4):
            for family, source in sources.items():
                renderer.register_material_vt_source(
                    material_index,
                    family,
                    source,
                    (virtual_size, virtual_size),
                    [0.5, 0.5, 1.0, 1.0],
                )
        _render_rgba(
            renderer,
            _make_params(size_px=size, shading="visibility", vt=vt),
            dem,
            ibl,
        )

    certificate = render_certificate(sign=False)
    degraded = {entry["name"] for entry in certificate["degradations"]}
    shader_hashes = certificate["engine"]["wgsl_module_hashes"]
    # The atlas is only sampled by the pass that shades, so the resolve module
    # must be in the render's provenance for either arm below to mean anything.
    assert "terrain_visbuffer_resolve.shader" in shader_hashes, sorted(shader_hashes)
    stats = vt_stats()
    assert "bindless_bc" in stats, stats
    if bindless_capable:
        assert stats["bindless_bc"] == 1.0, stats
        assert "terrain_vt_bindless_atlas" not in degraded, sorted(degraded)
    else:
        assert stats["bindless_bc"] == 0.0, stats
        assert "terrain_vt_bindless_atlas" in degraded, sorted(degraded)
    record_tessella_result(
        "bindless_atlas",
        {
            "bindless_capable": bindless_capable,
            "bindless_bc": float(stats["bindless_bc"]),
            "bindless_fallback_recorded": (
                "terrain_vt_bindless_atlas" in degraded
            ),
            "bc_atlas_fallback_recorded": "terrain_vt_bc_atlas" in degraded,
            "granted_bindless_features": sorted(
                granted
                & {
                    "texture_binding_array",
                    "sampled_texture_and_storage_buffer_array_non_uniform_indexing",
                    "texture_compression_bc",
                }
            ),
            "visbuffer_resolve_shader_sha256": shader_hashes[
                "terrain_visbuffer_resolve.shader"
            ],
        },
    )
