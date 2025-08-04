# T11-BEGIN:tests-grid
import numpy as np
import vulkan_forge as vf

def test_grid_shapes_and_counts():
    pos, uv, idx = vf.generate_grid(8, 6, spacing=(2.0, 1.0), origin="center")
    assert pos.shape == (8*6, 3)
    assert uv.shape  == (8*6, 2)
    assert idx.ndim == 1
    assert idx.size == (8-1)*(6-1)*6

def test_winding_ccw_from_plus_y():
    pos, uv, idx = vf.generate_grid(3, 3, spacing=(1.0, 1.0), origin="min")
    # first quad triangles: (0,1,4) and (0,4,3) by construction
    tri0 = idx[:3].astype(np.int64)
    a, b, c = pos[tri0[0]], pos[tri0[1]], pos[tri0[2]]
    # compute normal via cross((b-a),(c-a))
    n = np.cross(b - a, c - a)
    # Y should be positive for CCW when viewed from +Y
    assert n[1] > 0.0
# T11-END:tests-grid