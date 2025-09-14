# tests/test_path_tracing_tri_materials.py
# Triangle material BSDF checks with BVH traversal.
# This exists to verify A2 materials also apply to triangles, using the BVH.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_path_tracing_triangles.py

import numpy as np


def scene_triangle(center_color):
    import forge3d.path_tracing as pt
    t = pt.PathTracer(64, 64, seed=4)
    t.add_triangle((-0.8, -0.6, 0.0), (0.8, -0.6, 0.0), (0.0, 0.9, 0.0), center_color)
    return t


def test_triangle_metal_center_brighter_than_lambert():
    lam = scene_triangle({"type": "lambert", "base_color": (0.8, 0.8, 0.8)})
    met = scene_triangle({"type": "metal", "base_color": (0.95, 0.85, 0.6), "roughness": 0.2})
    a = lam.render_rgba(spp=1)[32, 32, :3].astype(np.int32).sum()
    b = met.render_rgba(spp=1)[32, 32, :3].astype(np.int32).sum()
    assert b >= a


def test_triangle_dielectric_differs_from_lambert():
    lam = scene_triangle({"type": "lambert", "base_color": (0.7, 0.8, 1.0)})
    die = scene_triangle({"type": "dielectric", "base_color": (0.7, 0.8, 1.0)})
    a = lam.render_rgba(spp=1)
    b = die.render_rgba(spp=1)
    # Images should not be identical; center pixel likely differs
    assert not np.array_equal(a, b)

