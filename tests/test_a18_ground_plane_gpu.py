# tests/test_a18_ground_plane_gpu.py
# GPU ground plane smoke test for A18 (Path Tracing ground/shadow catcher).
# Exists to validate presence and basic behavior of the PT ground plane kernel vs sky.
# RELEVANT FILES:src/shaders/pt_kernel.wgsl,src/path_tracing/compute.rs,python/forge3d/__init__.py

import numpy as np
import pytest


def _gpu_available() -> bool:
    try:
        from forge3d import enumerate_adapters  # type: ignore

        return bool(enumerate_adapters())
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="No compatible GPU adapter")
def test_ground_plane_sky_contrast():
    # Use native binding if available
    from forge3d import _forge3d as _f  # type: ignore

    cam = {
        "origin": (0.0, 1.5, 3.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 45.0,
        "aspect": 1.0,
        "exposure": 1.0,
    }

    # Empty scene is fine; ground plane is implicit in shader
    scene = []
    w = h = 64
    img = _f._pt_render_gpu(w, h, scene, cam, 7, 1)
    assert img.shape == (h, w, 4) and img.dtype == np.uint8

    # Top row should be mostly sky (brighter blue), bottom rows ground (darker grey)
    top_mean = img[1, :, :3].mean()
    bot_mean = img[h - 2, :, :3].mean()
    assert top_mean > bot_mean, "sky should be brighter than ground"

