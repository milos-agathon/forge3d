# tests/test_f7_multipolygonz_obj.py
# MultipolygonZ -> OBJ mesh conversion tests.

import numpy as np
import pytest

import forge3d.converters as conv

_requires_native = pytest.mark.skipif(
    getattr(conv, "_forge3d", None) is None,
    reason="forge3d native extension is not available",
)


@_requires_native
@pytest.mark.io
def test_single_ring_square() -> None:
    # CCW square in XY plane at Z=0
    ring = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    mesh = conv.multipolygonz_to_mesh([ring])
    # 4 vertices, 2 triangles (fan)
    assert mesh.vertex_count == 4
    assert mesh.triangle_count == 2
    # Normals should be roughly +Z
    assert np.allclose(mesh.normals.mean(axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5)


@_requires_native
@pytest.mark.io
def test_two_rings_merged() -> None:
    r1 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
    r2 = np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]], dtype=np.float32)
    mesh = conv.multipolygonz_to_mesh([r1, r2])
    # 8 vertices total, 4 triangles total
    assert mesh.vertex_count == 8
    assert mesh.triangle_count == 4
