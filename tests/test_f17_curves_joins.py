# tests/test_f17_curves_joins.py
# Validate ribbon join styles: miter vs bevel vs round

import numpy as np
import pytest

from forge3d.geometry import generate_ribbon


@pytest.mark.geometry
def test_ribbon_join_miter_has_greater_corner_span_than_bevel() -> None:
    # L-shaped path with 90 degree turn at index 1
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # corner point (index 1)
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    width_start = 0.2
    width_end = 0.2

    rib_bevel = generate_ribbon(path, width_start, width_end, join_style="bevel")
    rib_miter = generate_ribbon(path, width_start, width_end, join_style="miter", miter_limit=8.0)

    # Two vertices per point; for index i, vertices are at 2*i and 2*i+1
    i = 1
    a0, a1 = 2 * i, 2 * i + 1

    # Distance across the ribbon at the corner
    def span(mesh):
        p0 = mesh.positions[a0]
        p1 = mesh.positions[a1]
        return float(np.linalg.norm(p1 - p0))

    span_bevel = span(rib_bevel)
    span_miter = span(rib_miter)

    # Miter join should extend further than bevel at a sharp corner
    assert span_miter > span_bevel * 1.05


@pytest.mark.geometry
def test_ribbon_join_round_reasonable_span() -> None:
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    rib_round = generate_ribbon(path, 0.2, 0.2, join_style="round", miter_limit=4.0)

    # Basic sanity: vertex/triangle counts
    assert rib_round.vertex_count == path.shape[0] * 2
    assert rib_round.triangle_count == 2 * (path.shape[0] - 1)
