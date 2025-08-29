import math, pytest

def test_c10_parent_z90_child_unitx_world():
    try:
        import forge3d as f3d
    except ImportError as e:
        pytest.skip(f"forge3d unavailable: {e}")
    x, y, z = f3d.c10_parent_z90_child_unitx_world()
    assert math.isclose(x, 0.0, abs_tol=1e-5)
    assert math.isclose(y, 1.0, abs_tol=1e-5)
    assert math.isclose(z, 0.0, abs_tol=1e-5)