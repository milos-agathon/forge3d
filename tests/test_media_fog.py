# tests/test_media_fog.py
# Basic tests for homogeneous height fog factor and single scatter proxy
# RELEVANT FILES:python/forge3d/lighting.py,src/shaders/lighting_media.wgsl

import numpy as np


def test_height_fog_monotonic_and_bounds():
    import forge3d.lighting as lighting

    d = np.linspace(0.0, 200.0, 1024, dtype=np.float32)
    fog = lighting.height_fog_factor(d, density=0.03)
    assert fog.dtype == np.float32
    # Bounds
    assert np.all(fog >= 0.0) and np.all(fog <= 1.0)
    # Monotonic non-decreasing
    assert np.all(np.diff(fog) >= -1e-5)


def test_single_scatter_positive_and_sensible():
    import forge3d.lighting as lighting

    d = np.array([0.0, 1.0, 5.0, 10.0, 50.0], dtype=np.float32)
    L = lighting.single_scatter_estimate(d, sun_intensity=2.0, density=0.05, g=0.5)
    assert L.shape == d.shape
    assert np.all(L >= 0.0)
    # Should increase with distance and approach a finite asymptote
    assert np.all(np.diff(L) >= -1e-6)

