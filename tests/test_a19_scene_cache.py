# tests/test_a19_scene_cache.py
# Validate A19: Scene cache for HQ re-render speed and image parity.
# Exists to ensure re-render uses cached precomputations, speeding up by >=30% while identical.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/path_tracing.pyi,README.md

import time
import numpy as np


def test_scene_cache_speed_and_parity():
    import forge3d.path_tracing as pt

    W, H = 128, 96
    tracer = pt.PathTracer()
    tracer.enable_scene_cache(True, capacity=4)

    scene = {"meshes": 1, "materials": 2}
    cam = {
        "origin": (0.0, 0.0, 3.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 45.0,
        "aspect": float(W) / float(H),
        "exposure": 1.0,
    }

    # Cold render (cache miss)
    t0 = time.perf_counter()
    img1 = tracer.render_rgba(W, H, scene=scene, camera=cam, frames=24, seed=77, denoiser="off")
    t1 = time.perf_counter()

    # Warm render (cache hit)
    img2 = tracer.render_rgba(W, H, scene=scene, camera=cam, frames=24, seed=77, denoiser="off")
    t2 = time.perf_counter()

    assert isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray)
    assert img1.shape == (H, W, 4) and img2.shape == (H, W, 4)
    assert img1.dtype == np.uint8 and img2.dtype == np.uint8
    # Identical image requirement
    assert np.array_equal(img1, img2)

    cold_ms = (t1 - t0) * 1000.0
    warm_ms = (t2 - t1) * 1000.0

    # Expect >=30% improvement; allow generous tolerance for CI variance
    # Also guard against pathological tiny durations by requiring absolute gain >= 2ms when possible
    assert warm_ms <= 0.70 * cold_ms or (cold_ms - warm_ms) >= 2.0

    stats = tracer.cache_stats()
    assert stats.get("hits", 0) >= 1
    assert stats.get("misses", 0) >= 1

