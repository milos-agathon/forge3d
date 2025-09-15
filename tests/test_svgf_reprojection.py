# tests/test_svgf_reprojection.py
# Temporal reprojection sanity: small camera shift should yield similar denoised images
# Uses Python fallback path to keep determinism without a GPU requirement
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_svgf_gpu_variance.py,python/forge3d/denoise.py

import os
import numpy as np
import pytest

from forge3d.path_tracing import PathTracer, make_sphere, make_camera


_RUN = os.environ.get("FORGE3D_RUN_SVGF_TESTS", "0") == "1"


@pytest.mark.skipif(not _RUN, reason="SVGF tests disabled by default; set FORGE3D_RUN_SVGF_TESTS=1 to enable")
def test_reprojection_consistency():
    tr = PathTracer()
    scene = [make_sphere(center=(0, 0, -3), radius=1.0, albedo=(0.9, 0.9, 0.9))]
    cam1 = make_camera(origin=(0, 0, 0), look_at=(0, 0, -1), up=(0, 1, 0), fov_y=45.0, aspect=1.0, exposure=1.0)
    cam2 = make_camera(origin=(0.002, 0, 0), look_at=(0.002, 0, -1), up=(0, 1, 0), fov_y=45.0, aspect=1.0, exposure=1.0)
    img1 = tr.render_rgba(96, 96, scene, cam1, seed=1, frames=1, use_gpu=True, denoiser="svgf")
    img2 = tr.render_rgba(96, 96, scene, cam2, seed=1, frames=1, use_gpu=True, denoiser="svgf")
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)).mean()
    assert diff < 8.0

