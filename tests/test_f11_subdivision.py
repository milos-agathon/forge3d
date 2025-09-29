# tests/test_f11_subdivision.py
# Subdivision surface tests (Loop-like refinement without smoothing step)

import numpy as np
import pytest

from forge3d.geometry import MeshBuffers, subdivide_mesh


@pytest.mark.geometry
def test_subdivide_quad_once() -> None:
    # Unit square in XY plane, two triangles
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ],
        dtype=np.float32,
    )
    uvs = positions[:, :2].copy()
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0, 3), np.float32), uvs=uvs, indices=indices)

    out = subdivide_mesh(mesh, levels=1)

    # Triangles quadruple
    assert out.triangle_count == 4 * mesh.triangle_count
    # Unique edges in two-triangle quad = 5, so vertices increase by 5 (4 -> 9)
    assert out.vertex_count == 9
    # Normals recomputed to match vertex count
    assert out.normals.shape == (out.vertex_count, 3)
    # UVs carried and interpolated
    assert out.uvs.shape == (out.vertex_count, 2)


@pytest.mark.geometry
def test_subdivide_multiple_levels_counts() -> None:
    # Single triangle
    positions = np.array([[0.0, 0.0, 0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=np.float32)
    indices = np.array([[0,1,2]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=np.empty((0,2), np.float32), indices=indices)

    out = subdivide_mesh(mesh, levels=2)
    # Triangles multiply by 4^levels: 1 -> 16
    assert out.triangle_count == 16
    assert out.indices.shape == (16, 3)
