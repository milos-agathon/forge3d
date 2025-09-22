# tests/test_b7_cloudshade.py
# Tests for B7 cloud shade overlay modulation on terrain lighting.
# Ensures cloud shadow compute pass exists to justify Workstream B coverage.
# RELEVANT FILES:src/scene/mod.rs,src/core/cloud_shadows.rs,shaders/cloud_shadows.wgsl,src/shaders/terrain_descriptor_indexing.wgsl

"""B7: Validate cloud shadow overlay darkens terrain without introducing banding."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import forge3d as f3d
except Exception:  # pragma: no cover - module import guard
    f3d = None


def _gpu_available() -> bool:
    """Return True when the forge3d scene can be constructed (GPU present)."""
    if f3d is None:
        return False
    try:
        scene = f3d.Scene(16, 16, 16)
        scene.render_rgba()
        return True
    except Exception:
        return False


def _build_heightmap(size: int) -> np.ndarray:
    """Create a gently varying terrain to visualize lighting modulation."""
    coords = np.linspace(-2.0, 2.0, size, dtype=np.float32)
    z, x = np.meshgrid(coords, coords, indexing="ij")
    hills = 0.2 * np.sin(x * 1.3) * np.cos(z * 1.1)
    ridges = 0.05 * np.sin(x * 4.0 + z * 2.5)
    basin = 0.1 * np.exp(-((x - 0.5) ** 2 + (z + 0.25) ** 2))
    return (hills + ridges - basin).astype(np.float32)


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
class TestCloudShadeOverlay:
    """Cloud shade overlay acceptance checks."""

    def setup_method(self) -> None:
        self.scene = f3d.Scene(128, 128, grid=64, colormap="terrain")
        self.scene.set_height_from_r32f(_build_heightmap(64))
        self.scene.set_camera_look_at(
            eye=(2.0, 2.2, 2.0),
            target=(0.0, 0.2, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=50.0,
            znear=0.1,
            zfar=15.0,
        )

    def teardown_method(self) -> None:
        self.scene = None  # type: ignore[assignment]

    def test_cloud_shadows_modulate_and_move(self) -> None:
        """Cloud overlay darkens terrain and updates when animated."""
        baseline_rgba = self.scene.render_rgba()
        baseline_luma = baseline_rgba[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))

        self.scene.enable_cloud_shadows(quality="low")
        self.scene.set_cloud_density(0.85)
        self.scene.set_cloud_coverage(0.6)
        self.scene.set_cloud_shadow_intensity(0.9)
        self.scene.set_cloud_shadow_softness(0.2)
        self.scene.set_cloud_scale(1.5)
        self.scene.set_cloud_speed(0.02, 0.01)
        self.scene.update_cloud_animation(0.0)

        shaded_rgba = self.scene.render_rgba()
        shaded_luma = shaded_rgba[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))

        assert not np.array_equal(baseline_rgba, shaded_rgba)

        baseline_mean = float(baseline_luma.mean())
        shaded_mean = float(shaded_luma.mean())
        assert shaded_mean < baseline_mean - 2.0

        diff = baseline_luma.astype(np.float32) - shaded_luma.astype(np.float32)
        assert float(diff.std()) > 1.0
        assert np.unique(shaded_luma.astype(np.uint8)).size > 32

        self.scene.update_cloud_animation(1.5)
        shifted_rgba = self.scene.render_rgba()
        assert not np.array_equal(shaded_rgba, shifted_rgba)

        params = self.scene.get_cloud_params()
        assert pytest.approx(params[0], rel=1e-3) == 0.85
        assert pytest.approx(params[1], rel=1e-3) == 0.6

