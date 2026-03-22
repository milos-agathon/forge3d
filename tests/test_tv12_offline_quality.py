"""TV12: Offline render quality pipeline tests.

Tests the offline accumulation pipeline: settings validation, GPU-based
accumulation, adaptive sampling metrics, determinism, and image output.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    DenoiseSettings,
    OfflineQualitySettings,
    PomSettings,
    make_terrain_params_config,
)
from forge3d.denoise_oidn import oidn_available, oidn_denoise


ROOT = Path(__file__).resolve().parents[1]
GPU_AVAILABLE = terrain_rendering_available()


# ---------------------------------------------------------------------------
# Helpers (shared with test_terrain_probes.py pattern)
# ---------------------------------------------------------------------------

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


def _mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left[..., :3].astype(np.float32) - right[..., :3].astype(np.float32))))


def _mean_luminance(image: np.ndarray) -> float:
    rgb = image[..., :3].astype(np.float32)
    return float(np.mean(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722))


def _make_offline_params(overlay, *, aa_samples=4, aa_seed=42):
    config = make_terrain_params_config(
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
        light_azimuth_deg=138.0,
        light_elevation_deg=16.0,
        sun_intensity=0.8,
        cam_radius=6.2,
        cam_phi_deg=138.0,
        cam_theta_deg=58.0,
        fov_y_deg=48.0,
        camera_mode="screen",
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aa_samples=aa_samples,
        aa_seed=aa_seed,
    )
    return f3d.TerrainRenderParams(config)


# ---------------------------------------------------------------------------
# Module-scoped fixture: shared renderer, material set, IBL, heightmap
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def offline_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("Offline quality tests require a terrain-capable hardware-backed forge3d runtime")

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


# ===================================================================
# Pure Python tests (no GPU needed)
# ===================================================================

class TestOfflineQualitySettingsDefaults:
    def test_offline_quality_settings_defaults(self) -> None:
        settings = OfflineQualitySettings()
        assert settings.enabled is False
        assert settings.adaptive is False
        assert settings.target_variance == 0.001
        assert settings.max_samples == 64
        assert settings.min_samples == 4
        assert settings.batch_size == 4
        assert settings.tile_size == 16
        assert settings.convergence_ratio == 0.95


class TestOfflineQualitySettingsValidation:
    def test_offline_quality_settings_validation(self) -> None:
        with pytest.raises(ValueError, match="max_samples must be >= 1"):
            OfflineQualitySettings(max_samples=0)
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            OfflineQualitySettings(min_samples=0)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            OfflineQualitySettings(batch_size=0)
        with pytest.raises(ValueError, match="tile_size must be >= 1"):
            OfflineQualitySettings(tile_size=0)
        with pytest.raises(ValueError, match="convergence_ratio must be in"):
            OfflineQualitySettings(convergence_ratio=-0.1)
        with pytest.raises(ValueError, match="convergence_ratio must be in"):
            OfflineQualitySettings(convergence_ratio=1.1)
        with pytest.raises(ValueError, match="target_variance must be >= 0"):
            OfflineQualitySettings(target_variance=-1.0)


def test_denoise_settings_accepts_oidn() -> None:
    settings = DenoiseSettings(enabled=True, method="oidn")
    assert settings.method == "oidn"


def test_denoise_settings_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        DenoiseSettings(method="magic")


def test_oidn_available_returns_bool() -> None:
    result = oidn_available()
    assert isinstance(result, bool)


def test_oidn_denoise_validates_input_shape() -> None:
    bad_2d = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="beauty must be"):
        oidn_denoise(bad_2d)

    bad_4ch = np.zeros((64, 64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="beauty must be"):
        oidn_denoise(bad_4ch)


# ===================================================================
# GPU integration tests (skip if no hardware adapter)
# ===================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="Offline quality tests require GPU-backed forge3d module")
class TestOfflineAccumulation:
    def test_begin_accumulate_resolve_tonemap(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=4, aa_seed=42)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            batch_result = renderer.accumulate_batch(4)
            assert batch_result.total_samples >= 4
            assert batch_result.batch_time_ms >= 0.0

            hdr_frame, aov_frame = renderer.resolve_offline_hdr()
            assert hdr_frame.size == (224, 224)

            frame = renderer.tonemap_offline_hdr(hdr_frame)
            arr = frame.to_numpy()
            assert arr.shape == (224, 224, 4)
            assert arr.dtype == np.uint8
        finally:
            renderer.end_offline_accumulation()

    def test_determinism(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env

        frames = []
        for _ in range(2):
            params = _make_offline_params(overlay, aa_samples=4, aa_seed=42)
            renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
            try:
                renderer.accumulate_batch(4)
                hdr_frame, _aov = renderer.resolve_offline_hdr()
                frames.append(hdr_frame.to_numpy_f32())
            finally:
                renderer.end_offline_accumulation()

        assert np.array_equal(frames[0], frames[1]), "Same seed should produce identical HDR output"

    def test_quality_improvement(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env

        results = []
        for n_samples in (1, 16):
            params = _make_offline_params(overlay, aa_samples=n_samples, aa_seed=7)
            renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
            try:
                renderer.accumulate_batch(n_samples)
                hdr_frame, _aov = renderer.resolve_offline_hdr()
                results.append(hdr_frame.to_numpy_f32())
            finally:
                renderer.end_offline_accumulation()

        diff = float(np.mean(np.abs(results[0] - results[1])))
        assert diff > 0.0, "16-sample render should differ from 1-sample render"

    def test_session_guard_double_begin(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=4, aa_seed=42)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            with pytest.raises(Exception):
                renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        finally:
            # Clean up: accumulate, resolve, tonemap, then end
            renderer.accumulate_batch(1)
            hdr_frame, _aov = renderer.resolve_offline_hdr()
            renderer.tonemap_offline_hdr(hdr_frame)
            renderer.end_offline_accumulation()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Adaptive sampling tests require GPU-backed forge3d module")
class TestAdaptiveSampling:
    def test_metrics_queryable(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=8, aa_seed=99)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            renderer.accumulate_batch(4)
            renderer.accumulate_batch(4)

            metrics = renderer.read_accumulation_metrics(0.001)
            assert metrics.total_samples >= 8
            assert isinstance(metrics.mean_delta, float)
            assert isinstance(metrics.p95_delta, float)
            assert isinstance(metrics.max_tile_delta, float)
            assert 0.0 <= metrics.converged_tile_ratio <= 1.0
        finally:
            renderer.end_offline_accumulation()

    def test_max_samples_respected(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=12, aa_seed=55)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            r1 = renderer.accumulate_batch(4)
            r2 = renderer.accumulate_batch(4)
            r3 = renderer.accumulate_batch(4)

            assert r1.total_samples == 4
            assert r2.total_samples == 8
            assert r3.total_samples == 12
        finally:
            renderer.end_offline_accumulation()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Image output tests require GPU-backed forge3d module")
class TestImageOutput:
    def test_png_save(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=4, aa_seed=42)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            renderer.accumulate_batch(4)
            hdr_frame, _aov = renderer.resolve_offline_hdr()
            frame = renderer.tonemap_offline_hdr(hdr_frame)
        finally:
            renderer.end_offline_accumulation()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = Path(tmp.name)
        try:
            frame.save(str(png_path))
            assert png_path.exists()
            assert png_path.stat().st_size > 100
        finally:
            png_path.unlink(missing_ok=True)

    def test_hdr_numpy_values_plausible(self, offline_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = offline_render_env
        params = _make_offline_params(overlay, aa_samples=4, aa_seed=42)

        renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
        try:
            renderer.accumulate_batch(4)
            hdr_frame, _aov = renderer.resolve_offline_hdr()
            hdr_np = hdr_frame.to_numpy_f32()
        finally:
            renderer.end_offline_accumulation()

        assert hdr_np.shape == (224, 224, 4)
        assert hdr_np.dtype == np.float32
        assert np.all(np.isfinite(hdr_np)), "HDR data must be finite"
        assert float(np.max(np.abs(hdr_np[:, :, :3]))) > 0.0, "HDR data must contain non-zero values"
