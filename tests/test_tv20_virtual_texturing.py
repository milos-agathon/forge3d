"""TV20: Terrain material virtual texturing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import _build_heightmap, terrain_rendering_available
from forge3d.terrain_params import (
    AovSettings,
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)


TERRAIN_SHADER = Path(__file__).resolve().parents[1] / "src" / "shaders" / "terrain_pbr_pom.wgsl"
VT_RUNTIME = Path(__file__).resolve().parents[1] / "src" / "terrain" / "renderer" / "virtual_texture.rs"
GPU_AVAILABLE = terrain_rendering_available()
VT_MATERIAL_COUNT = 4


def test_terrain_shader_declares_vt_sampling_and_feedback_bindings() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    required_tokens = (
        "struct TerrainVTUniforms",
        "@group(6) @binding(6)",
        "@group(6) @binding(7)",
        "@group(6) @binding(8)",
        "@group(6) @binding(9)",
        "@group(6) @binding(10)",
        "@group(6) @binding(11)",
        "fn terrain_vt_enabled()",
        "fn terrain_vt_write_feedback(",
        "fn sample_material_layer_uv(",
    )
    for token in required_tokens:
        assert token in source, f"Missing terrain VT shader token: {token}"


def test_vt_layer_family_defaults() -> None:
    family = VTLayerFamily(family="albedo")
    assert family.tile_size == 248
    assert family.tile_border == 4
    assert family.slot_size == 256
    assert family.pages_x0 == 17
    assert family.pages_y0 == 17


def test_vt_layer_family_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="family must be one of"):
        VTLayerFamily(family="diffuse")


def test_vt_layer_family_accepts_forward_compatible_families() -> None:
    VTLayerFamily(family="normal")
    VTLayerFamily(family="mask")


def test_native_terrain_vt_runtime_is_currently_albedo_only() -> None:
    source = VT_RUNTIME.read_text(encoding="utf-8")
    assert 'const TERRAIN_VT_SUPPORTED_FAMILY: &str = "albedo";' in source


def test_vt_settings_reject_duplicate_families() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        TerrainVTSettings(
            layers=[
                VTLayerFamily(family="albedo"),
                VTLayerFamily(family="albedo"),
            ]
        )


def test_vt_settings_reject_indivisible_atlas() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TerrainVTSettings(
            atlas_size=1000,
            layers=[VTLayerFamily(family="albedo", tile_size=248, tile_border=4)],
        )


def test_vt_settings_actual_mip_count() -> None:
    settings = TerrainVTSettings(max_mip_levels=8)
    assert settings.actual_mip_count("albedo") >= 1


def test_vt_layer_pages_at_mip_non_power_of_two() -> None:
    family = VTLayerFamily(family="albedo", virtual_size_px=(3000, 2000), tile_size=248)
    assert family.pages_at_mip(0) == (13, 9)
    assert family.pages_at_mip(1) == (7, 5)


def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 164, 128]))


def _build_test_ibl():
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_test_hdr(hdr_path)
        return f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)


def _build_vt_source(size: int, material_index: int) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    stripe = 0.5 + 0.5 * np.sin(
        (xx * (material_index + 1.5) * 14.0 + yy * (material_index + 2.0) * 11.0) * np.pi
    )
    checker = (
        (
            np.floor(xx * (10 + material_index * 3))
            + np.floor(yy * (12 + material_index * 2))
        )
        % 2.0
    ).astype(np.float32)
    modulation = 0.30 + 0.70 * (0.55 * checker + 0.45 * stripe)
    palette = np.array(
        [
            [0.88, 0.18, 0.12],
            [0.14, 0.72, 0.22],
            [0.16, 0.34, 0.90],
            [0.92, 0.84, 0.18],
        ],
        dtype=np.float32,
    )
    base = palette[material_index % len(palette)]
    rgb = np.clip(base * modulation[..., None] + (1.0 - modulation[..., None]) * 0.08, 0.0, 1.0)
    alpha = np.ones((size, size, 1), dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return np.ascontiguousarray((rgba * 255.0).round().astype(np.uint8))


def _register_vt_sources(renderer: f3d.TerrainRenderer, virtual_size: int) -> None:
    renderer.clear_material_vt_sources()
    for material_index in range(VT_MATERIAL_COUNT):
        source = _build_vt_source(virtual_size, material_index)
        fallback_rgb = source[..., :3].astype(np.float32).mean(axis=(0, 1)) / 255.0
        renderer.register_material_vt_source(
            material_index,
            "albedo",
            source,
            (virtual_size, virtual_size),
            [float(fallback_rgb[0]), float(fallback_rgb[1]), float(fallback_rgb[2]), 1.0],
        )


def _build_render_params(
    *,
    vt_settings: TerrainVTSettings | None,
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size_px: tuple[int, int] = (224, 160),
    terrain_span: float = 8.0,
    cam_radius: float = 4.0,
) -> f3d.TerrainRenderParams:
    config = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=1.6,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="material",
        colormap_strength=0.0,
        ibl_enabled=True,
        ibl_intensity=1.8,
        light_azimuth_deg=136.0,
        light_elevation_deg=24.0,
        sun_intensity=2.2,
        cam_radius=cam_radius,
        cam_phi_deg=142.0,
        cam_theta_deg=58.0,
        fov_y_deg=50.0,
        camera_mode="mesh",
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aov=AovSettings(enabled=True, albedo=True, normal=False, depth=False),
    )
    config.cam_target = [float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
    config.vt = vt_settings
    return f3d.TerrainRenderParams(config)


def _render_albedo(
    renderer: f3d.TerrainRenderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    *,
    vt_settings: TerrainVTSettings | None,
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_radius: float = 4.0,
) -> np.ndarray:
    params = _build_render_params(
        vt_settings=vt_settings,
        cam_target=cam_target,
        cam_radius=cam_radius,
    )
    _, aov_frame = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
    )
    return np.asarray(aov_frame.albedo(), dtype=np.float32)


def _render_beauty(
    renderer: f3d.TerrainRenderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    *,
    vt_settings: TerrainVTSettings | None,
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_radius: float = 4.0,
) -> np.ndarray:
    params = _build_render_params(
        vt_settings=vt_settings,
        cam_target=cam_target,
        cam_radius=cam_radius,
        size_px=(192, 144),
    )
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    return frame.to_numpy()


def _mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left[..., :3] - right[..., :3])))


@pytest.fixture()
def tv20_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("TV20 runtime tests require a terrain-capable hardware-backed forge3d runtime")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap(160)
    ibl = _build_test_ibl()
    renderer.clear_material_vt_sources()
    try:
        yield renderer, material_set, ibl, heightmap
    finally:
        try:
            renderer.clear_material_vt_sources()
        except RuntimeError:
            pass


@pytest.mark.skipif(not GPU_AVAILABLE, reason="TV20 runtime tests require GPU-backed forge3d module")
class TestTerrainMaterialVirtualTexturing:
    def test_runtime_api_surface_exists(self, tv20_render_env) -> None:
        renderer, _, _, _ = tv20_render_env
        assert hasattr(renderer, "register_material_vt_source")
        assert hasattr(renderer, "get_material_vt_stats")
        assert hasattr(renderer, "clear_material_vt_sources")

    def test_vt_disabled_preserves_baseline_output(self, tv20_render_env) -> None:
        renderer, material_set, ibl, heightmap = tv20_render_env
        renderer.clear_material_vt_sources()

        baseline = _render_albedo(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=None,
        )

        _register_vt_sources(renderer, virtual_size=2048)
        disabled = _render_albedo(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=TerrainVTSettings(enabled=False),
        )
        stats = renderer.get_material_vt_stats()

        max_diff = float(np.max(np.abs(baseline - disabled)))
        assert max_diff <= 1e-5, f"Disabled VT regressed baseline albedo by {max_diff:.6f}"
        assert stats["resident_pages"] == pytest.approx(0.0)
        assert stats["total_pages"] == pytest.approx(0.0)

    def test_vt_enabled_changes_albedo_and_reports_residency(self, tv20_render_env) -> None:
        renderer, material_set, ibl, heightmap = tv20_render_env
        renderer.clear_material_vt_sources()

        baseline = _render_albedo(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=None,
        )

        _register_vt_sources(renderer, virtual_size=2048)
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=1024,
            residency_budget_mb=24.0,
            max_mip_levels=6,
            layers=[VTLayerFamily(family="albedo", virtual_size_px=(2048, 2048))],
        )
        vt_albedo = _render_albedo(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=vt_settings,
        )
        stats = renderer.get_material_vt_stats()

        assert _mean_abs_diff(baseline, vt_albedo) > 0.05
        assert stats["resident_pages"] > 0.0
        assert stats["total_pages"] > 0.0
        assert stats["cache_budget_pages"] > 0.0
        assert stats["resident_pages"] <= stats["cache_budget_pages"]
        assert stats["cache_misses"] > 0.0
        assert stats["tiles_streamed"] > 0.0
        assert stats["avg_upload_ms"] > 0.0
        assert stats["source_count"] == pytest.approx(float(VT_MATERIAL_COUNT))

    def test_vt_budget_is_enforced_under_camera_motion(self, tv20_render_env) -> None:
        renderer, material_set, ibl, heightmap = tv20_render_env
        renderer.clear_material_vt_sources()
        _register_vt_sources(renderer, virtual_size=4096)

        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=512,
            residency_budget_mb=0.75,
            max_mip_levels=6,
            layers=[VTLayerFamily(family="albedo", virtual_size_px=(4096, 4096))],
        )

        _render_beauty(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=vt_settings,
            cam_target=(-2.0, -2.0, 0.0),
            cam_radius=1.25,
        )
        stats_left = renderer.get_material_vt_stats()

        _render_beauty(
            renderer,
            material_set,
            ibl,
            heightmap,
            vt_settings=vt_settings,
            cam_target=(2.0, 2.0, 0.0),
            cam_radius=1.25,
        )
        stats_right = renderer.get_material_vt_stats()

        assert stats_left["total_pages"] > stats_left["cache_budget_pages"] > 0.0
        assert stats_right["total_pages"] > stats_right["cache_budget_pages"] > 0.0
        assert stats_left["resident_pages"] <= stats_left["cache_budget_pages"]
        assert stats_right["resident_pages"] <= stats_right["cache_budget_pages"]
        assert stats_left["tiles_streamed"] > 0.0
        assert stats_right["tiles_streamed"] > 0.0
        assert max(stats_left["evictions"], stats_right["evictions"]) > 0.0
