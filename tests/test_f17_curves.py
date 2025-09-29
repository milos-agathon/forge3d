# tests/test_f17_curves.py
# Curves & tubes generation tests

import numpy as np
import pytest

from forge3d.geometry import generate_ribbon, generate_tube


@pytest.mark.geometry
def test_generate_ribbon_counts() -> None:
    # Simple polyline path
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ribbon = generate_ribbon(path, width_start=0.2, width_end=0.1)
    # Two vertices per path point
    assert ribbon.vertex_count == path.shape[0] * 2
    # Two triangles per segment
    assert ribbon.triangle_count == 2 * (path.shape[0] - 1)
    # UVs match vertex count
    assert ribbon.uvs.shape == (ribbon.vertex_count, 2)


@pytest.mark.geometry
def test_generate_tube_counts() -> None:
    # Helix path
    t = np.linspace(0.0, 4.0 * np.pi, 64, dtype=np.float32)
    path = np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1)
    tube = generate_tube(path, radius_start=0.1, radius_end=0.05, radial_segments=12, cap_ends=True)
    # Each ring has rs vertices
    rs = 12
    assert tube.vertex_count >= path.shape[0] * rs  # + optional caps
    # Faces: per segment: 2*rs quads -> 2*rs triangles
    assert tube.triangle_count >= (path.shape[0] - 1) * 2 * rs
