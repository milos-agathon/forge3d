#!/usr/bin/env python3
# tests/test_workstream_c_hydrology.py
# Targeted tests for Workstream C (C1, C2, C3) Python fallback APIs

import numpy as np
import pytest

try:
    import forge3d as f3d
except Exception:
    f3d = None

pytestmark = pytest.mark.skipif(f3d is None, reason="forge3d not available")


def make_heightmap(h=32, w=48):
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    hm = 0.6 * (y + x) / 2.0
    hm[h//3: h//3*2, w//4: w//4*3] *= 0.25  # basin
    return hm


class TestC1WaterDetection:
    def test_detect_water_auto_and_flat(self):
        s = f3d.Scene(256, 192, grid=32)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)

        s.enable_water_surface()
        s.set_water_surface_height(0.4)

        m0 = s.detect_water_from_dem(method='auto', smooth_iters=1)
        assert m0.shape == hm.shape
        assert m0.dtype == np.bool_
        frac0 = float(m0.mean())
        assert 0.01 <= frac0 <= 0.9  # some reasonable coverage

        m1 = s.detect_water_from_dem(method='flat', smooth_iters=1)
        frac1 = float(m1.mean())
        # flat should be at least as permissive as auto (often more)
        assert frac1 >= frac0 * 0.75

    def test_set_external_water_mask_shape_validation(self):
        s = f3d.Scene(128, 96, grid=16)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)
        s.enable_water_surface()
        s.set_water_surface_height(0.3)

        good = (hm <= 0.2)
        s.set_water_mask(good)  # should succeed

        bad = np.zeros((25, 32), dtype=bool)
        with pytest.raises(ValueError):
            s.set_water_mask(bad)


class TestC2WaterMaterial:
    def test_depth_aware_coloration(self):
        s = f3d.Scene(200, 150, grid=32)
        hm = make_heightmap(20, 30)
        s.set_height_from_r32f(hm)
        s.enable_water_surface()
        s.set_water_surface_height(0.5)
        s.set_water_depth_colors((0.1, 0.6, 1.0), (0.0, 0.12, 0.25))
        s.set_water_alpha(0.6)

        # With mask covering half
        mask = np.zeros_like(hm, dtype=bool)
        mask[:, : hm.shape[1] // 2] = True
        s.set_water_mask(mask)

        img = s.render_rgba()
        assert img.shape == (150, 200, 4)
        # Left half should be bluer vs right half (no mask)
        left_mean = img[:, :100, :3].mean()
        right_mean = img[:, 100:, :3].mean()
        assert left_mean != right_mean


class TestC3FoamOverlay:
    def test_shoreline_foam_enabling(self):
        s = f3d.Scene(160, 120, grid=16)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)
        s.enable_water_surface()
        s.set_water_surface_height(0.35)
        s.detect_water_from_dem(method='flat', smooth_iters=1)

        s.enable_shoreline_foam()
        s.set_shoreline_foam_params(width_px=2, intensity=0.8, noise_scale=16.0)
        s.set_water_alpha(0.5)

        img1 = s.render_rgba()
        s.disable_shoreline_foam()
        img2 = s.render_rgba()

        # Foam should visually modify pixels near the shoreline; ensure not identical
        assert not np.array_equal(img1, img2)
