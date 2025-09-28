# tests/test_f14_transform.py
# Unit tests for Workstream F14 mesh transform utilities exposed via the Python API.
# Validates center/scale/flip/swap operations and bounding boxes using the native extension.
# RELEVANT FILES:python/forge3d/geometry.py,src/geometry/transform.rs

import numpy as np
import pytest

from forge3d import geometry

_requires_native = pytest.mark.skipif(
    getattr(geometry, "_forge3d", None) is None,
    reason="forge3d native extension is not available",
)


@pytest.fixture()
def base_mesh() -> geometry.MeshBuffers:
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    uvs = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    return geometry.MeshBuffers(positions=positions, normals=normals, uvs=uvs, indices=indices)


@_requires_native
@pytest.mark.geometry
def test_center_mesh_moves_to_origin_without_altering_topology(base_mesh: geometry.MeshBuffers) -> None:
    centered, previous_center = geometry.center_mesh(base_mesh)

    # Original mesh remains unchanged
    assert np.allclose(base_mesh.positions[:, 0], np.array([1.0, -1.0, 0.0], dtype=np.float32))

    # Bounding box center aligns with origin
    min_bounds, max_bounds = geometry.mesh_bounds(centered)
    assert np.allclose((min_bounds + max_bounds) * 0.5, np.zeros(3, dtype=np.float32))

    # Returned previous center should match the centroid of the old bounds
    assert np.allclose(previous_center, np.array([0.0, 0.5, 0.0], dtype=np.float32))


@_requires_native
@pytest.mark.geometry
def test_scale_mesh_applies_pivot_and_reports_winding_flip(base_mesh: geometry.MeshBuffers) -> None:
    scaled, flipped = geometry.scale_mesh(base_mesh, (-1.0, 1.0, 1.0))

    # X coordinate inverted, winding flipped
    assert flipped is True
    assert np.allclose(scaled.positions[:, 0], -base_mesh.positions[:, 0])
    assert np.allclose(scaled.normals[:, 0], -base_mesh.normals[:, 0])

    # Scaling about a pivot keeps pivot vertex fixed and does not flip winding
    pivot = (1.0, 0.0, 0.0)
    scaled_pivot, flipped_pivot = geometry.scale_mesh(base_mesh, (2.0, 1.0, 1.0), pivot=pivot)
    assert flipped_pivot is False
    assert np.allclose(scaled_pivot.positions[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))


@_requires_native
@pytest.mark.geometry
def test_flip_mesh_axis_updates_normals_and_winding(base_mesh: geometry.MeshBuffers) -> None:
    flipped_mesh, flipped = geometry.flip_mesh_axis(base_mesh, axis=1)
    assert flipped is True
    assert np.allclose(flipped_mesh.positions[:, 1], -base_mesh.positions[:, 1])
    assert np.allclose(flipped_mesh.normals[:, 1], -base_mesh.normals[:, 1])


@_requires_native
@pytest.mark.geometry
def test_swap_mesh_axes(base_mesh: geometry.MeshBuffers) -> None:
    swapped_mesh, flipped = geometry.swap_mesh_axes(base_mesh, 0, 2)
    assert flipped is True
    # X and Z components should be exchanged
    assert np.allclose(swapped_mesh.positions[:, 0], base_mesh.positions[:, 2])
    assert np.allclose(swapped_mesh.positions[:, 2], base_mesh.positions[:, 0])
    assert np.allclose(swapped_mesh.normals[:, 0], base_mesh.normals[:, 2])
    assert np.allclose(swapped_mesh.normals[:, 2], base_mesh.normals[:, 0])


@_requires_native
@pytest.mark.geometry
def test_mesh_bounds_returns_expected_extents(base_mesh: geometry.MeshBuffers) -> None:
    min_bounds, max_bounds = geometry.mesh_bounds(base_mesh)
    assert np.allclose(min_bounds, np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(max_bounds, np.array([1.0, 1.0, 0.0], dtype=np.float32))
