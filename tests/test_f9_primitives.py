# tests/test_f9_primitives.py
# Exercise primitive mesh library through Python geometry facade
# Exists to cover F9 acceptance including normals, UVs, and counts
# RELEVANT FILES:src/geometry/primitives.rs,python/forge3d/geometry.py,tests/test_f1_extrude.py,examples/f9_primitives_demo.py

import numpy as np
import pytest

geometry = pytest.importorskip("forge3d.geometry")


def test_plane_resolution() -> None:
    mesh = geometry.primitive_mesh("plane", resolution=(2, 2))
    assert mesh.positions.shape == (9, 3)
    assert mesh.indices.shape == (8, 3)
    assert np.allclose(mesh.positions[:, 2], 0.0)


def test_sphere_normals_unit_length() -> None:
    mesh = geometry.primitive_mesh("sphere", rings=6, radial_segments=8)
    lengths = np.linalg.norm(mesh.normals, axis=1)
    assert lengths.shape[0] == mesh.vertex_count
    assert np.allclose(lengths[lengths > 0], 1.0, atol=1e-3)


def test_torus_not_empty() -> None:
    mesh = geometry.primitive_mesh("torus", radial_segments=6, tube_segments=4)
    assert mesh.vertex_count > 0
    assert mesh.indices.size > 0
