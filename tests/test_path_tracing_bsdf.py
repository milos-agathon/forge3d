# tests/test_path_tracing_bsdf.py
# BSDF checks for A2 (Lambert vs Metal vs Dielectric) and env.
# This exists to validate material behavior differences are visible and deterministic.
# RELEVANT FILES:python/forge3d/path_tracing.py

import numpy as np


def test_env_gradient_nonzero_on_empty_scene():
    import forge3d.path_tracing as pt
    t = pt.PathTracer(32, 16, seed=1)
    img = t.render_rgba(spp=1)
    rgb = img[..., :3]
    assert rgb.sum() > 0


def test_metal_center_brighter_than_lambert():
    import forge3d.path_tracing as pt
    w = h = 64
    lam = pt.PathTracer(w, h, seed=2)
    met = pt.PathTracer(w, h, seed=2)
    mat_lam = {"type": "lambert", "base_color": (0.8, 0.8, 0.8)}
    mat_met = {"type": "metal", "base_color": (0.95, 0.85, 0.6), "roughness": 0.2}
    lam.add_sphere((0.0, 0.0, 0.0), 0.6, mat_lam)
    met.add_sphere((0.0, 0.0, 0.0), 0.6, mat_met)
    a = lam.render_rgba(spp=1)[h//2, w//2, :3].astype(np.int32).sum()
    b = met.render_rgba(spp=1)[h//2, w//2, :3].astype(np.int32).sum()
    assert b >= a


def test_fresnel_schlick_extremes():
    from forge3d.path_tracing import _fresnel_schlick
    x0 = np.array([[0.0]], dtype=np.float32)
    x1 = np.array([[1.0]], dtype=np.float32)
    F0 = np.array([[0.04, 0.04, 0.04]], dtype=np.float32)
    a = _fresnel_schlick(x1, F0)
    b = _fresnel_schlick(x0, F0)
    # cos=1 -> F0; cos=0 -> 1
    assert np.allclose(a, F0, atol=1e-6)
    assert np.allclose(b, np.ones_like(b), atol=1e-6)

