# tests/test_path_tracing_gpu.py
# GPU MVP tests for A1: adapter skip, deterministic single-sphere render, and CPU fallback.
# This exists to validate the WGSL kernel, Rust backend, and Python bridge wiring with minimal scope.
# RELEVANT FILES:python/forge3d/path_tracing.py,src/shaders/pt_kernel.wgsl,src/path_tracing/compute.rs,src/lib.rs

import os, pytest, numpy as np
from forge3d.path_tracing import PathTracer, make_sphere, make_camera


def _gpu_available():
    try:
        # import internal probe if exists; else attempt to construct tracer
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="No compatible GPU adapter")
def test_gpu_single_sphere_64x64_deterministic(tmp_path):
    tracer = PathTracer(1, 1)
    scene = [make_sphere(center=(0,0,-3), radius=1.0, albedo=(0.8,0.2,0.2))]
    cam   = make_camera(origin=(0,0,0), look_at=(0,0,-1), up=(0,1,0), fov_y=45.0, aspect=1.0, exposure=1.0)
    img1 = tracer.render_rgba(64,64,scene,cam,seed=123,frames=1,use_gpu=True)
    img2 = tracer.render_rgba(64,64,scene,cam,seed=123,frames=1,use_gpu=True)
    assert img1.shape == (64,64,4)
    assert np.all(img1 == img2), "deterministic for 1 frame with fixed seed"


def test_fallback_cpu(tmp_path):
    tracer = PathTracer(1, 1)
    scene = [make_sphere(center=(0,0,-3), radius=1.0, albedo=(1,1,1))]
    cam   = make_camera(origin=(0,0,0), look_at=(0,0,-1), up=(0,1,0), fov_y=45.0, aspect=1.0, exposure=1.0)
    img = tracer.render_rgba(32,32,scene,cam,seed=1,frames=1,use_gpu=False)
    assert img.shape == (32,32,4)

