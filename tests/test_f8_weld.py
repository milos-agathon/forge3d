# tests/test_f8_weld.py
# Vertex welding behaviour coverage for geometry API
# Exists to verify F8 weld output, remap table, and UV tolerance handling
# RELEVANT FILES:src/geometry/weld.rs,python/forge3d/geometry.py,tests/test_f15_validate.py,tests/test_f1_extrude.py

import numpy as np
import pytest

geometry = pytest.importorskip("forge3d.geometry")


def test_weld_merges_duplicate_vertices() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1e-6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint32)
    mesh, remap, collapsed = geometry.weld_mesh(positions, indices)
    assert mesh.vertex_count == 3
    assert collapsed == 1
    assert remap.shape[0] == positions.shape[0]


def test_weld_respects_uv_epsilon() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    indices = np.array([[0, 1, 1]], dtype=np.uint32)
    uvs = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    mesh, _, _ = geometry.weld_mesh(
        positions,
        indices,
        uvs=uvs,
        position_epsilon=1e-5,
        uv_epsilon=1e-3,
    )
    assert mesh.vertex_count == 2
