# tests/test_f11_subdivision_creases.py
# Verify Loop smoothing with crease and boundary preservation affects edge points

import numpy as np
import pytest

from forge3d.geometry import MeshBuffers, subdivide_mesh


def make_two_tris_irregular():
    # Non-parallelogram quad split into two triangles sharing edge v0-v2
    v0 = [0.0, 0.0, 0.0]
    v1 = [2.0, 0.2, 0.0]
    v2 = [2.0, 1.0, 0.0]
    v3 = [0.0, 1.5, 0.0]
    positions = np.array([v0, v1, v2, v3], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = positions[:, :2].copy()
    mesh = MeshBuffers(positions=positions, normals=np.empty((0, 3), np.float32), uvs=uvs, indices=indices)
    return mesh


@pytest.mark.geometry
def test_interior_edge_point_changes_with_crease() -> None:
    mesh = make_two_tris_irregular()
    # Expected interior edge point for edge (0,2)
    a = mesh.positions[0, :2]
    b = mesh.positions[2, :2]
    c = mesh.positions[1, :2]
    d = mesh.positions[3, :2]
    interior = 0.375 * (a + b) + 0.125 * (c + d)
    midpoint = 0.5 * (a + b)

    # No creases
    out_smooth = subdivide_mesh(mesh, levels=1)
    # With crease along the shared interior edge (0,2)
    creases = np.array([[0, 2]], dtype=np.uint32)
    out_crease = subdivide_mesh(mesh, levels=1, creases=creases)

    # Find nearest vertex in refined mesh to theoretical points
    def nearest(p, pts):
        d = np.linalg.norm(pts[:, :2] - p[None, :], axis=1)
        i = int(np.argmin(d))
        return pts[i], float(d[i])

    v_int, d_int = nearest(interior, out_smooth.positions)
    v_mid, d_mid = nearest(midpoint, out_crease.positions)

    assert d_int < 1e-3, f"interior edge point not found near expected: {d_int}"
    assert d_mid < 1e-3, f"crease midpoint not found near expected: {d_mid}"
