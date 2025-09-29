import numpy as np
import pytest

from forge3d.geometry import generate_thick_polyline


@pytest.mark.geometry
def test_thick_polyline_bevel_width_consistent():
    # L-shaped path
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    width = 0.2
    mesh = generate_thick_polyline(path, width_world=width, depth_offset=0.01, join_style="bevel")

    # Two vertices per point; span across side at index i
    def span_at(i: int) -> float:
        a0, a1 = 2 * i, 2 * i + 1
        p0 = mesh.positions[a0]
        p1 = mesh.positions[a1]
        return float(np.linalg.norm(p1 - p0))

    # At endpoints and bevel corner, span should be ~width
    assert pytest.approx(span_at(0), rel=1e-5, abs=1e-7) == width
    assert pytest.approx(span_at(1), rel=1e-5, abs=1e-7) == width
    assert pytest.approx(span_at(2), rel=1e-5, abs=1e-7) == width

    # Depth offset applied to all vertices
    zs = mesh.positions[:, 2]
    assert np.allclose(zs, 0.01, atol=2e-5)


@pytest.mark.geometry
def test_thick_polyline_miter_wider_than_bevel():
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    width = 0.2
    bevel = generate_thick_polyline(path, width_world=width, join_style="bevel")
    miter = generate_thick_polyline(path, width_world=width, join_style="miter", miter_limit=8.0)

    i = 1
    a0, a1 = 2 * i, 2 * i + 1

    span_bev = float(np.linalg.norm(bevel.positions[a1] - bevel.positions[a0]))
    span_mit = float(np.linalg.norm(miter.positions[a1] - miter.positions[a0]))
    assert span_mit > span_bev * 1.05
