# tests/test_svgf_gpu_variance.py
# SVGF variance reduction check using Python fallback denoiser path
# Ensures denoised variance is lower on a synthetic scene; GPU tests skip by default
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/denoise.py,tests/test_svgf_reprojection.py

import os
import numpy as np
import pytest

from forge3d.path_tracing import PathTracer, make_sphere, make_camera


_RUN = os.environ.get("FORGE3D_RUN_SVGF_TESTS", "0") == "1"


def _scene():
    scene = [make_sphere(center=(0, 0, -3), radius=1.0, albedo=(0.7, 0.7, 0.7))]
    cam = make_camera(origin=(0, 0, 0), look_at=(0, 0, -1), up=(0, 1, 0), fov_y=45.0, aspect=1.0, exposure=1.0)
    return scene, cam


@pytest.mark.skipif(not _RUN, reason="SVGF tests disabled by default; set FORGE3D_RUN_SVGF_TESTS=1 to enable")
def test_variance_reduction_128x128():
    tr = PathTracer()
    scene, cam = _scene()
    noisy = tr.render_rgba(128, 128, scene, cam, seed=3, frames=1, use_gpu=True, denoiser="off")
    den = tr.render_rgba(128, 128, scene, cam, seed=3, frames=1, use_gpu=True, denoiser="svgf", svgf_iters=5)
    def var(img: np.ndarray) -> float:
        return np.var(img.astype(np.float32), axis=(0, 1)).mean()
    assert var(den) < 0.5 * var(noisy)

