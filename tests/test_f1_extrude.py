# tests/test_f1_extrude.py
# Validates polygon extrusion mesh generation via Python geometry API
# Exists to ensure F1 core extrusion pipelines stay deterministic
# RELEVANT FILES:src/geometry/extrude.rs,python/forge3d/geometry.py,examples/f1_extrude_demo.py,tests/test_f9_primitives.py

import numpy as np
import pytest

geometry = pytest.importorskip("forge3d.geometry")


def test_extrude_square_counts() -> None:
    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    mesh = geometry.extrude_polygon(polygon, height=2.0)
    assert mesh.vertex_count == 24
    assert mesh.triangle_count == 12
    assert mesh.positions.shape == (24, 3)
    assert mesh.normals.shape == (24, 3)
    assert mesh.indices.shape == (12, 3)


def test_extrude_rejects_invalid_polygon() -> None:
    polygon = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        geometry.extrude_polygon(polygon, height=1.0)
