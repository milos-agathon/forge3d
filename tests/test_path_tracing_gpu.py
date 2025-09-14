# tests/test_path_tracing_gpu.py
# GPU MVP tests for A1: adapter skip, deterministic single-sphere render, and CPU fallback.
# This exists to validate the WGSL kernel, Rust backend, and Python bridge wiring with minimal scope.
# RELEVANT FILES:python/forge3d/path_tracing.py,src/shaders/pt_kernel.wgsl,src/path_tracing/compute.rs,src/lib.rs

import os, pytest, numpy as np
from forge3d.path_tracing import PathTracer


def _gpu_available():
    try:
        from forge3d import enumerate_adapters  # type: ignore
        return bool(enumerate_adapters())
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="No compatible GPU adapter")
def test_gpu_single_sphere_64x64_deterministic(tmp_path):
    from forge3d import _forge3d as _f  # type: ignore
    scene = [{"center": (0,0,-3), "radius": 1.0, "albedo": (0.8,0.2,0.2)}]
    cam   = {"origin": (0,0,0), "look_at": (0,0,-1), "up": (0,1,0), "fov_y": 45.0, "aspect": 1.0, "exposure": 1.0}
    img1 = _f._pt_render_gpu(64,64,scene,cam,123,1)
    img2 = _f._pt_render_gpu(64,64,scene,cam,123,1)
    assert img1.shape == (64,64,4)
    assert np.all(img1 == img2), "deterministic for 1 frame with fixed seed"


def test_fallback_cpu(tmp_path):
    tracer = PathTracer(32, 32)
    tracer.add_sphere((0,0,-3), 1.0, (1,1,1))
    img = tracer.render_rgba(spp=1)
    assert img.shape == (32,32,4)
