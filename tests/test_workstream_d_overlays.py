#!/usr/bin/env python3
# tests/test_workstream_d_overlays.py
# Targeted tests for Workstream D overlays/annotations in Python fallback

import numpy as np
import pytest

try:
    import forge3d as f3d
except Exception:
    f3d = None

pytestmark = pytest.mark.skipif(f3d is None, reason="forge3d not available")


def make_heightmap(h=48, w=64):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    hm = 0.5 * (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * 0.5 + 0.5) + 0.1 * y
    return hm


class TestD4DrapeRaster:
    def test_drape_overlay_applies(self):
        s = f3d.Scene(256, 192, grid=32)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)

        base = s.render_rgba()
        ov = np.zeros((40, 60, 3), dtype=np.uint8)
        ov[::5, :, 0] = 255
        ov[:, ::7, 1] = 255
        s.set_raster_overlay(ov, alpha=0.5, offset_xy=(20, 10), scale=1.0)
        img = s.render_rgba()
        assert img.shape == base.shape
        assert not np.array_equal(base, img)


class TestD5Altitude:
    def test_altitude_overlay_requires_heightmap_and_changes_image(self):
        s = f3d.Scene(200, 150, grid=32)
        hm = make_heightmap(20, 30)
        s.set_height_from_r32f(hm)
        base = s.render_rgba()
        s.enable_altitude_overlay(alpha=0.3)
        img = s.render_rgba()
        assert not np.array_equal(base, img)


class TestD6D7Contours:
    def test_generate_and_render_contours(self):
        s = f3d.Scene(256, 192, grid=32)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)
        lines = s.generate_contours(interval=0.05, smooth=1)
        assert isinstance(lines, list)
        assert len(lines) > 0
        s.enable_contours_overlay(color=(0,0,0), width_px=1)
        base = s.render_rgba()
        s.disable_contours_overlay()
        img = s.render_rgba()
        # With and without contours differ
        assert not np.array_equal(base, img)


class TestD8Hillshade:
    def test_hillshade_overlay_changes_image(self):
        s = f3d.Scene(256, 192, grid=32)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)
        base = s.render_rgba()
        s.enable_shadow_overlay(azimuth_deg=300.0, altitude_deg=35.0, strength=0.6, blend='multiply')
        img = s.render_rgba()
        assert not np.array_equal(base, img)


class TestD2D3D10TextCompassScaleTitle:
    def test_text_compass_scale_title(self):
        s = f3d.Scene(300, 200, grid=32)
        hm = make_heightmap(24, 32)
        s.set_height_from_r32f(hm)
        s.add_text_overlay("Hello", x=20, y=40, size_px=18, color=(255,255,0))
        s.enable_compass_rose(position='top_right', size_px=48, color=(255,255,255), bg_alpha=0.2)
        s.enable_scale_bar(position='bottom_left', max_width_px=160, color=(255,255,255))
        s.set_title_bar("Title", height_px=24, bg_rgba=(0,0,0,128), color=(255,255,255))
        img = s.render_rgba()
        assert img.shape == (200, 300, 4)
        # Remove overlays and compare
        s.clear_text_overlays()
        s.disable_compass_rose()
        s.disable_scale_bar()
        s.clear_title_bar()
        img2 = s.render_rgba()
        assert not np.array_equal(img, img2)
