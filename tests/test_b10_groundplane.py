# tests/test_b10_groundplane.py
# B10: Ground Plane (Raster) – Python fallback tests
# RELEVANT FILES: python/forge3d/__init__.py

import numpy as np
import pytest


def test_ground_plane_toggle_and_params_roundtrip():
    import forge3d as f3d

    scene = f3d.Scene(128, 96, grid=64, colormap="terrain")
    assert not scene.is_ground_plane_enabled()

    # Enable and set parameters
    ok = scene.enable_ground_plane(True)
    assert ok is True
    assert scene.is_ground_plane_enabled()

    scene.set_ground_plane_params(color=(10, 20, 30), grid_color=(200, 210, 220), grid_px=8, alpha=200)
    color, grid_color, grid_px, alpha = scene.get_ground_plane_params()
    assert color == (10, 20, 30)
    assert grid_color == (200, 210, 220)
    assert grid_px == 8
    assert alpha == 200

    # Disable
    ok2 = scene.enable_ground_plane(False)
    assert ok2 is False
    assert not scene.is_ground_plane_enabled()


def test_ground_plane_grid_renders_when_enabled():
    import forge3d as f3d

    w, h = 96, 64
    scene = f3d.Scene(w, h, grid=64, colormap="terrain")

    # No heightmap – background
    # Baseline render without ground: should not exhibit strict grid color lines
    img0 = scene.render_rgba()

    scene.enable_ground_plane(True)
    scene.set_ground_plane_params(color=(32, 32, 32), grid_color=(180, 180, 180), grid_px=8, alpha=255)
    img1 = scene.render_rgba()

    # Check a known grid line row and a non-grid pixel
    grid_row = 0
    non_grid_px = (1, 1)  # not on grid row or column when step=8

    # On grid line, expect grid_color
    assert tuple(img1[grid_row, 0, :3]) == (180, 180, 180)

    # Off grid (1,1), expect base color
    assert tuple(img1[non_grid_px[1], non_grid_px[0], :3]) == (32, 32, 32)

    # Ensure the two renders differ (ground plane actually applied)
    assert not np.array_equal(img0, img1)


def test_renderer_triangle_above_ground_plane():
    # Renderer fallback draws ground first, then triangle on top
    from forge3d import Renderer

    r = Renderer(64, 64)
    # Manually enable ground plane in fallback renderer
    r._ground_plane_enabled = True
    r._gp_color = (0, 0, 0)
    r._gp_grid_color = (255, 255, 255)
    r._gp_grid_px = 4

    img = r.render_triangle_rgba()

    # The triangle occupies around the image center; sample a center pixel
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    center_rgb = tuple(img[cy, cx, :3])

    # With ground plane alone, a center pixel would likely be grid or base color.
    # Triangle should write a non-(0,0,0) and non-(255,255,255) value due to gradients.
    assert center_rgb not in [(0, 0, 0), (255, 255, 255)]
