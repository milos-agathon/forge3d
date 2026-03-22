from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    PomSettings,
    ProbeSettings,
    ReflectionProbeSettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SHADER = ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl"
PROBE_SHADER = ROOT / "src" / "shaders" / "terrain_probes.wgsl"
GPU_AVAILABLE = terrain_rendering_available()


def _write_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def _build_bowl_heightmap(size: int = 192) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    rim = 0.85 * np.exp(-((radius - 0.42) ** 2) * 90.0)
    basin = -0.55 * np.exp(-(radius ** 2) * 28.0)
    spur = 0.20 * np.exp(-((xx - 0.55) ** 2 * 22.0 + (yy + 0.10) ** 2 * 18.0))
    heightmap = rim + basin + spur
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#16341a"),
            (0.35, "#40692f"),
            (0.65, "#8b7b53"),
            (1.0, "#f3f6fb"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _mean_luminance(image: np.ndarray) -> float:
    rgb = image[..., :3].astype(np.float32)
    return float(np.mean(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722))


def _mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left[..., :3].astype(np.float32) - right[..., :3].astype(np.float32))))


def _render_probe_scene(
    renderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    overlay,
    *,
    probes: ProbeSettings | None,
    reflection_probes: ReflectionProbeSettings | None = None,
    debug_mode: int = 0,
    cam_phi_deg: float = 138.0,
    cam_theta_deg: float = 58.0,
) -> np.ndarray:
    params = make_terrain_params_config(
        size_px=(224, 224),
        render_scale=1.0,
        terrain_span=4.0,
        msaa_samples=1,
        z_scale=2.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        ibl_intensity=3.0,
        light_azimuth_deg=138.0,
        light_elevation_deg=16.0,
        sun_intensity=0.8,
        cam_radius=6.2,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=48.0,
        camera_mode="screen",
        debug_mode=debug_mode,
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        probes=probes,
        reflection_probes=reflection_probes,
    )
    native_params = f3d.TerrainRenderParams(params)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
    )
    return frame.to_numpy()


def test_probe_shader_module_exists() -> None:
    assert PROBE_SHADER.exists(), f"Missing probe shader include: {PROBE_SHADER}"
    source = PROBE_SHADER.read_text(encoding="utf-8")
    assert "struct ProbeGridUniforms" in source
    assert "fn sample_probe_irradiance" in source


def test_terrain_shader_includes_probe_module() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    assert '#include "terrain_probes.wgsl"' in source
    assert "DBG_PROBE_IRRADIANCE" in source
    assert "DBG_PROBE_WEIGHT" in source


def test_probe_settings_defaults_disabled() -> None:
    settings = ProbeSettings()
    assert settings.enabled is False
    assert settings.grid_dims == (8, 8)


def test_probe_settings_validation_zero_dims() -> None:
    with pytest.raises(ValueError, match="grid_dims must be >= \\(1, 1\\)"):
        ProbeSettings(enabled=True, grid_dims=(0, 0))


def test_probe_settings_validation_probe_limit() -> None:
    with pytest.raises(ValueError, match="probe count limit"):
        ProbeSettings(enabled=True, grid_dims=(65, 65))


def test_probe_settings_single_probe_grid() -> None:
    settings = ProbeSettings(enabled=True, grid_dims=(1, 1))
    assert settings.grid_dims == (1, 1)


def test_reflection_probe_settings_defaults_disabled() -> None:
    settings = ReflectionProbeSettings()
    assert settings.enabled is False
    assert settings.grid_dims == (4, 4)


def test_reflection_probe_settings_validation_probe_limit() -> None:
    with pytest.raises(ValueError, match="reflection probe count limit"):
        ReflectionProbeSettings(enabled=True, grid_dims=(17, 17))


@pytest.fixture(scope="module")
def probe_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("Probe lighting tests require a terrain-capable hardware-backed forge3d runtime")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()
    heightmap = _build_bowl_heightmap()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = tmp.name
    _write_test_hdr(hdr_path)
    ibl = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
    Path(hdr_path).unlink(missing_ok=True)

    return renderer, material_set, ibl, heightmap, overlay


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Probe lighting tests require GPU-backed forge3d module")
class TestTerrainProbeLighting:
    def test_probe_fallback_pixel_identical(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=None,
        )
        disabled = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
        )
        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - disabled.astype(np.int16))))
        assert max_diff <= 1, f"Disabled probes regressed baseline output by {max_diff} LSB"

    def test_probe_memory_tracked(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(4, 4), ray_count=32),
        )
        report = renderer.get_probe_memory_report()
        assert report["probe_count"] == 16
        assert report["grid_uniform_bytes"] == 48
        assert report["probe_ssbo_bytes"] == 16 * 144
        assert report["total_bytes"] == 48 + 16 * 144

    def test_probe_valley_darker(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        probe_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=48),
            debug_mode=50,
        )
        valley = _mean_luminance(probe_debug[96:128, 96:128])
        ridge = _mean_luminance(probe_debug[150:182, 96:128])
        assert ridge > valley * 1.02, f"Expected ridge probe irradiance > valley, got ridge={ridge:.3f}, valley={valley:.3f}"

    def test_probe_ridge_brighter(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        probe_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=48),
            debug_mode=50,
        )
        ridge = _mean_luminance(probe_debug[96:128, 40:72])
        shoulder = _mean_luminance(probe_debug[96:128, 96:128])
        assert ridge > shoulder, f"Expected exposed ridge > shoulder luminance, got ridge={ridge:.3f}, shoulder={shoulder:.3f}"

    def test_probe_out_of_bounds_weight_zero(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=0.18,
                ray_count=24,
            ),
            debug_mode=51,
        )
        center = float(np.mean(weight[92:132, 92:132, 0]))
        corner = float(np.mean(weight[0:24, 0:24, 0]))
        assert center > 20.0, f"Expected non-zero probe coverage near center, got {center:.2f}"
        assert corner < 5.0, f"Expected out-of-bounds weight to fall back to zero, got {corner:.2f}"

    def test_probe_edge_blend_smooth(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=0.24,
                ray_count=24,
            ),
            debug_mode=51,
        )
        weight_map = weight[..., 0].astype(np.float32)
        midtones = weight_map[(weight_map > 10.0) & (weight_map < 245.0)]
        assert midtones.size > 0, "Expected smooth edge blend to produce intermediate probe weights"

    def test_probe_single_probe_grid(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(1, 1), ray_count=24),
            debug_mode=51,
        )
        report = renderer.get_probe_memory_report()
        weight_map = weight[..., 0].astype(np.float32)
        assert report["probe_count"] == 1
        assert float(np.mean(weight_map)) > 240.0
        assert float(np.std(weight_map)) < 2.0

    def test_probe_invalidation_triggers(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        base = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(5, 5), ray_count=32),
            debug_mode=50,
        )
        sky_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(5, 5),
                ray_count=32,
                sky_color=(1.0, 0.65, 0.45),
            ),
            debug_mode=50,
        )
        dims_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(3, 3), ray_count=32),
            debug_mode=50,
        )

        assert _mean_abs_diff(base, sky_changed) > 0.20
        assert _mean_abs_diff(base, dims_changed) > 0.20

    def test_reflection_probe_fallback_pixel_identical(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=None,
        )
        disabled = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=False),
        )
        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - disabled.astype(np.int16))))
        assert max_diff <= 1, f"Disabled reflection probes regressed baseline output by {max_diff} LSB"

    def test_reflection_probe_memory_tracked(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=True, grid_dims=(4, 4), ray_count=9),
        )
        report = renderer.get_probe_memory_report()
        assert report["reflection_probe_count"] == 16
        assert report["reflection_grid_uniform_bytes"] == 48
        assert report["reflection_probe_ssbo_bytes"] == 16 * 112
        assert report["reflection_total_bytes"] == 48 + 16 * 112

    def test_reflection_probe_spatially_varies(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        reflection_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=16),
            debug_mode=52,
        )
        center = _mean_luminance(reflection_debug[96:128, 96:128])
        lower = _mean_luminance(reflection_debug[150:182, 96:128])
        upper = _mean_luminance(reflection_debug[40:72, 96:128])
        spread = max(center, lower, upper) - min(center, lower, upper)
        assert spread > 5.0, (
            "Expected reflection probe debug view to vary across the terrain, "
            f"got center={center:.3f}, lower={lower:.3f}, upper={upper:.3f}"
        )

    def test_reflection_probe_out_of_bounds_weight_zero(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=0.18,
                ray_count=9,
            ),
            debug_mode=53,
        )
        center = float(np.mean(weight[92:132, 92:132, 0]))
        corner = float(np.mean(weight[0:24, 0:24, 0]))
        assert center > 20.0, f"Expected non-zero reflection probe coverage near center, got {center:.2f}"
        assert corner < 5.0, f"Expected out-of-bounds reflection weight to fall back to zero, got {corner:.2f}"

    def test_reflection_probe_invalidation_triggers(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = probe_render_env
        base = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=True, grid_dims=(5, 5), ray_count=9),
            debug_mode=52,
        )
        ground_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(5, 5),
                ray_count=9,
                ground_color=(0.45, 0.30, 0.18),
            ),
            debug_mode=52,
        )
        dims_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=True, grid_dims=(3, 3), ray_count=9),
            debug_mode=52,
        )

        assert _mean_abs_diff(base, ground_changed) > 0.20
        assert _mean_abs_diff(base, dims_changed) > 0.20
