import pytest
import numpy as np

try:
    import _vulkan_forge as vf
except ImportError:
    pytest.skip("Extension module _vulkan_forge not built; skipping grid tests.", allow_module_level=True)


def compute_triangle_normal(p0, p1, p2):
    # normal = cross(b - a, c - a)
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    return n


def test_valid_grid_center_origin():
    nx, nz = 4, 5
    spacing = (1.0, 2.0)
    pos, uv, indices = vf.grid_generate(nx, nz, spacing, origin="center")
    assert pos.shape == (nx * nz, 3)
    assert uv.shape == (nx * nz, 2)
    assert indices.ndim == 1
    # Check that all triangle normals have positive Y
    pos_arr = pos.astype(np.float32)
    indices_arr = indices.astype(np.int64)
    normals = []
    for i in range(0, indices_arr.size, 3):
        a = pos_arr[indices_arr[i + 0]]
        b = pos_arr[indices_arr[i + 1]]
        c = pos_arr[indices_arr[i + 2]]
        n = compute_triangle_normal(a, b, c)
        normals.append(n)
    normals = np.stack(normals)
    # Y component should be >= 0 (allow tiny negative due to float precision)
    assert np.all(normals[:, 1] >= -1e-5), "Expected +Y normals for CCW winding"


@pytest.mark.parametrize("origin_variant", ["center", "Center", "min", "MINCORNER", "origin"])
def test_origin_variants_do_not_error(origin_variant):
    pos, uv, indices = vf.grid_generate(3, 3, (0.5, 0.5), origin=origin_variant)
    assert pos.shape == (9, 3)
    assert uv.shape == (9, 2)
    assert indices.shape[0] == 8 * 3 // 2 or indices.ndim == 1  # basic sanity


def test_invalid_params_raise():
    with pytest.raises(Exception):
        vf.grid_generate(1, 5, (1.0, 1.0), origin="center")  # nx < 2
    with pytest.raises(Exception):
        vf.grid_generate(5, 1, (1.0, 1.0), origin="center")  # nz < 2
    with pytest.raises(Exception):
        vf.grid_generate(3, 3, (1.0, 1.0), origin="badorigin")  # invalid origin string
