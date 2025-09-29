# tests/test_f10_uv_unwrap.py
# UV unwrap helpers tests for planar and spherical projections.

import numpy as np
import pytest

from forge3d.geometry import MeshBuffers, unwrap_planar, unwrap_spherical


@pytest.mark.geometry
def test_planar_unwrap_xy_from_z_axis() -> None:
    # Simple square in XY plane
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=np.empty((0,2), np.float32), indices=indices)
    out = unwrap_planar(mesh, axis=2)
    assert out.uvs.shape == (4, 2)
    assert np.all(out.uvs >= 0.0) and np.all(out.uvs <= 1.0)
    # Geometry untouched
    assert np.allclose(out.positions, positions)
    assert np.allclose(out.indices, indices)


@pytest.mark.geometry
def test_spherical_unwrap_cube() -> None:
    # Vertices of a unit cube
    positions = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    indices = np.array([
        [0,1,2],[0,2,3],  # bottom
        [4,5,6],[4,6,7],  # top
        [0,1,5],[0,5,4],  # front
        [2,3,7],[2,7,6],  # back
        [1,2,6],[1,6,5],  # right
        [3,0,4],[3,4,7],  # left
    ], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=np.empty((0,2), np.float32), indices=indices)
    out = unwrap_spherical(mesh)
    assert out.uvs.shape == (8, 2)
    assert np.all(out.uvs >= 0.0) and np.all(out.uvs <= 1.0)
    assert np.allclose(out.positions, positions)
    assert np.allclose(out.indices, indices)
