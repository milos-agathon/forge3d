import numpy as np
import pytest

from forge3d.mesh import make_mesh


@pytest.mark.parametrize("dtype,value", [(np.uint64, 2**32), (np.int64, -1), (np.int64, 2**32)])
def test_reject_indices_before_narrowing(dtype, value):
    with pytest.raises(RuntimeError, match="range"):
        make_mesh(np.zeros((3, 3), np.float32), np.array([[value, 1, 2]], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.uint32, np.int64])
def test_preserve_in_range_indices(dtype):
    indices = np.array([[0, 1, 2]], dtype=dtype)
    mesh = make_mesh(np.zeros((3, 3), np.float32), indices)
    np.testing.assert_array_equal(mesh["indices"], indices)
    assert mesh["indices"].dtype == np.uint32
