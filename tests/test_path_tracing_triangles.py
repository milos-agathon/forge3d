# tests/test_path_tracing_triangles.py
# Triangle intersection and BVH smoke tests for A1.
# This exists to validate triangles rendering path and BVH traversal.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_path_tracing_a1.py,docs/user/path_tracing.rst

import numpy as np


def test_single_triangle_faces_camera_produces_nonzero_pixels():
    import forge3d.path_tracing as pt
    t = pt.PathTracer(64, 64, seed=3)
    # Large triangle centered under camera, facing +Z (camera at z=1.5 looking -Z)
    t.add_triangle((-0.8, -0.8, 0.0), (0.8, -0.8, 0.0), (0.0, 0.9, 0.0), (1.0, 0.0, 0.0))
    img = t.render_rgba(spp=1)
    rgb = img[..., :3]
    assert rgb.sum() > 0
    # Center pixel should likely be lit
    cx, cy = 32, 32
    assert int(rgb[cy, cx].sum()) >= 1

